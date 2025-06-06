# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import ray
import time

import torch
from collections import defaultdict
from tqdm import tqdm

import chatlearn
from chatlearn import TorchModule
from chatlearn.runtime.engine import Engine
from chatlearn.runtime.environment import Environment
from chatlearn.runtime.trainer import Trainer
from chatlearn.utils import future


class TestTorchModule(TorchModule):

    def _get_rank(self):
        return int(os.environ["RANK"])

    def get_local_param_ranks(self):
        global_rank = self._get_rank()
        data_modulo_expert_parallel_ranks = [global_rank]
        return data_modulo_expert_parallel_ranks, 0

    def check_param_exists(self, names):
        return True

    def tensor_and_expert_model_parallel_size(self):
        return 4


class CustomEngine(Engine):
    """Custom Engine"""

    def __init__(self,
                 policy: TestTorchModule,
                 policy_trainer: TestTorchModule):
        def env_compute_flow(batch):
            policy_out = policy.forward_step(batch)
            return policy_out

        def trainer_compute_flow(batch):
            policy_trainer.train_step(batch)

        env = Environment(env_compute_flow)
        trainer = Trainer(trainer_compute_flow)
        super().__init__(env, trainer)
        self.set_parameter_sync(policy_trainer, policy)


class PolicyModel(TestTorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data

    @property
    def data_parallel_size(self):
        return 1

    @property
    def data_parallel_rank(self):
        return 0

    def tensor_parallel_rank(self):
        return int(self._get_rank() % 4)

    def expert_parallel_rank(self):
        return 0

    def pipeline_parallel_rank(self):
        if self._get_rank() < 4:
            return 0
        return 1


class PPOPolicy(TestTorchModule):

    @property
    def data_parallel_size(self):
        return 4

    @property
    def data_parallel_rank(self):
        return int(self._get_rank() % 4)

    def tensor_parallel_rank(self):
        return 0

    def expert_parallel_rank(self):
        return int(self._get_rank() % 4)

    def pipeline_parallel_rank(self):
        if self._get_rank() < 4:
            return 0
        return 1


def create_engine(parallel_size : tuple):
    for _, model_config in chatlearn.get_args().models.items():
        model_config.num_gpu = 8
    chatlearn.get_args().models['policy'].expert_model_parallel_size = parallel_size[0]
    chatlearn.get_args().models['policy'].tensor_model_parallel_size = parallel_size[1]
    chatlearn.get_args().models['policy'].pipeline_model_parallel_size = parallel_size[2]
    chatlearn.get_args().models['ppo_policy'].expert_model_parallel_size = parallel_size[3]
    chatlearn.get_args().models['ppo_policy'].tensor_model_parallel_size = parallel_size[4]
    chatlearn.get_args().models['ppo_policy'].pipeline_model_parallel_size = parallel_size[5]

    chatlearn.get_args().runtime_args.colocation = [["policy", "ppo_policy"]]

    policy = PolicyModel("policy")
    ppo_policy = PPOPolicy("ppo_policy")

    os.environ['QWEN_VERSION'] = "qwen_moe_v1" # enable ParameterSyncGroupwithHEP

    engine = CustomEngine(policy, ppo_policy)
    engine.set_dataset(["Mock dataset"])
    engine.setup()
    return engine

def test_balanced_hep_1():
    """
    Test:
    1. trainer_tp < inference_tp
    2. trainer_tp can divide inference_tp
    3. trainer_tp * trainer_ep == inference_tp * inference_ep
    4. HEP is enabled for both trainer and inference.

    Current test case: 
    (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 4, 1, 4, 1, 1).
    """
    # parallel_size: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
    parallel_size = (1, 4, 1, 4, 1, 1)
    engine = create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.ep_num_mapping == parallel_size[0] / parallel_size[3]
    assert param_sync_group.tp_num_mapping == parallel_size[1] // parallel_size[4]
    assert param_sync_group.hep_num_mapping == 1

    actor2rank = param_sync_group.actor2rank

    # test for actors to create_group_experts_regrouping
    send_actors = []
    for actors in param_sync_group.send_actors_to_regroup_routed_experts:
        send_actors.append([actor2rank[actor] for actor in actors])
    assert send_actors == [[4, 5, 6, 7], [0, 1, 2, 3]]
    assert param_sync_group._comm_type_to_regroup_routed_experts == "allgather"

    # Judge routed experts and parameters except routed experts
    comm_pair_routed_experts = []
    comm_pair_stage_1 = []
    comm_pair_stage_2 = []

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_for_routed_experts.items():
        for dst_rank in dst_ranks:
            comm_pair_routed_experts.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_1.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_2.append((actor2rank[src_rank], actor2rank[dst_rank]))

    # The replica iter for routed experts will be reversed because num_src_tensor_parallel == 1 
    # and src_model is colocated with dst_model (Please see parameter_sync.py#L257-260).
    assert comm_pair_routed_experts == [
        (4, 8), (5, 9), (6, 10), (7, 11), (0, 12), (1, 13), (2, 14), (3, 15)
    ]
    assert comm_pair_stage_1 == [(0, 8), (1, 12)]
    assert comm_pair_stage_2 == [(8, 9), (8, 10), (8, 11), (12, 13), (12, 14), (12, 15)]

    print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {parallel_size}")

    engine.model_manager.sync_parameters(requires_grad=False, validate=True)
    print(f"pass parameter sync validation for hyper expert parallel.")
    engine.stop()

def test_balanced_hep_2():
    """
    Test:
    1. trainer_tp < inference_tp
    2. trainer_tp can divide inference_tp
    3. trainer_tp * trainer_ep == inference_tp * inference_ep
    4. HEP is enabled for both trainer and inference.

    Current test case: 
    (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 4, 2, 4, 1, 2).
    """
    # parallel_size: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
    parallel_size = (1, 4, 2, 4, 1, 2)

    engine = create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.ep_num_mapping == parallel_size[0] / parallel_size[3]
    assert param_sync_group.tp_num_mapping == parallel_size[1] // parallel_size[4]
    assert param_sync_group.hep_num_mapping == 1

    actor2rank = param_sync_group.actor2rank

    # Judge routed experts and parameters except routed experts
    comm_pair_routed_experts = []
    comm_pair_stage_1 = []
    comm_pair_stage_2 = []

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_for_routed_experts.items():
        for dst_rank in dst_ranks:
            comm_pair_routed_experts.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_1.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_2.append((actor2rank[src_rank], actor2rank[dst_rank]))

    # The replica iter for routed experts will be reversed because num_src_tensor_parallel == 1 
    # and src_model is colocated with dst_model (Please see parameter_sync.py#L257-260).
    assert comm_pair_routed_experts == [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)
    ]
    # Please note that 4 could send to one of the recv_ranks in [12, 15].
    # We send the weight to 13 instead of 12 to avoid GPU collision in the algorithm.
    assert comm_pair_stage_1 == [(0, 8), (4, 13)]
    assert comm_pair_stage_2 == [(8, 9), (8, 10), (8, 11), (13, 12), (13, 14), (13, 15)]
    engine.stop()

    print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {parallel_size}")

def test_balanced_hep_3():
    """
    Test:
    1. trainer_tp < inference_tp
    2. trainer_tp can divide inference_tp
    3. trainer_tp * trainer_ep == inference_tp * inference_ep
    4. HEP is enabled for both trainer and inference.

    Current test case: 
    (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 4, 1, 4, 1, 2).
    """
    # parallel_size: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
    parallel_size = (1, 4, 1, 4, 1, 2)

    engine = create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.ep_num_mapping == parallel_size[0] / parallel_size[3]
    assert param_sync_group.tp_num_mapping == parallel_size[1] // parallel_size[4]
    assert param_sync_group.hep_num_mapping == 1

    actor2rank = param_sync_group.actor2rank

    # Judge routed experts and parameters except routed experts
    comm_pair_routed_experts = []
    comm_pair_stage_1 = []
    comm_pair_stage_2 = []

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_for_routed_experts.items():
        for dst_rank in dst_ranks:
            comm_pair_routed_experts.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_1.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_2.append((actor2rank[src_rank], actor2rank[dst_rank]))

    # The replica iter for routed experts will be reversed because num_src_tensor_parallel == 1 
    # and src_model is colocated with dst_model (Please see parameter_sync.py#L257-260).
    assert comm_pair_routed_experts == [
        (0, 8), (0, 12), (1, 9), (1, 13),
        (2, 10), (2, 14), (3, 11), (3, 15),
        (4, 8), (4, 12), (5, 9), (5, 13),
        (6, 10), (6, 14), (7, 11), (7, 15)
    ]
    assert comm_pair_stage_1 == [(0, 8), (4, 9), (1, 12), (5, 13)]
    assert comm_pair_stage_2 == [
        (8, 9), (8, 10), (8, 11),
        (9, 8), (9, 10), (9, 11),
        (12, 13), (12, 14), (12, 15),
        (13, 12), (13, 14), (13, 15)
    ]
    engine.stop()
    print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {parallel_size}")


def test_balanced_hep_4():
    """
    Test:
    1. trainer_tp < inference_tp
    2. trainer_tp can divide inference_tp
    3. trainer_tp * trainer_ep == inference_tp * inference_ep
    4. HEP is enabled for both trainer and inference.

    Current test case: 
    (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 2, 1, 2, 1, 4).
    """
    # parallel_size: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
    parallel_size = (1, 2, 1, 2, 1, 4)

    engine = create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.ep_num_mapping == parallel_size[0] / parallel_size[3]
    assert param_sync_group.tp_num_mapping == parallel_size[1] // parallel_size[4]
    assert param_sync_group.hep_num_mapping == 1

    actor2rank = param_sync_group.actor2rank

    # Judge routed experts and parameters except routed experts
    comm_pair_routed_experts = []
    comm_pair_stage_1 = []
    comm_pair_stage_2 = []

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_for_routed_experts.items():
        for dst_rank in dst_ranks:
            comm_pair_routed_experts.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_1.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_2.append((actor2rank[src_rank], actor2rank[dst_rank]))

    # The replica iter for routed experts will be reversed because num_src_tensor_parallel == 1 
    # and src_model is colocated with dst_model (Please see parameter_sync.py#L257-260).
    assert comm_pair_routed_experts == [
        (0, 8), (0, 10), (0, 12), (0, 14),
        (1, 9), (1, 11), (1, 13), (1, 15),
        (2, 8), (2, 10), (2, 12), (2, 14),
        (3, 9), (3, 11), (3, 13), (3, 15),
        (4, 8), (4, 10), (4, 12), (4, 14),
        (5, 9), (5, 11), (5, 13), (5, 15),
        (6, 8), (6, 10), (6, 12), (6, 14),
        (7, 9), (7, 11), (7, 13), (7, 15)
    ], f"{comm_pair_routed_experts}"
    assert sorted(comm_pair_stage_1) == [
        (0, 8), (0, 13), # not (0, 8) because of gpu collision
        (1, 10), (1, 14),
        (2, 9), (2, 12),
        (3, 11), (3, 15), # not (3, 10) because of gpu collision
        (4, 8), (4, 12), # not (4, 13) because of gpu collision
        (5, 11), (5, 14),
        (6, 9), (6, 13),
        (7, 10), (7, 15) # not (7, 14) because of gpu collision
    ], f"{sorted(comm_pair_stage_1)}"
    assert comm_pair_stage_2 == [
        (8, 9), (9, 8),
        (10, 11), (11, 10),
        (12, 13), (13, 12),
        (14, 15), (15, 14)
    ], f"{comm_pair_stage_2}"

    engine.stop()
    print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {parallel_size}")

def test_balanced_hep_5():
    """
    Test:
    1. trainer_tp < inference_tp
    2. trainer_tp can divide inference_tp
    3. trainer_tp * trainer_ep == inference_tp * inference_ep
    4. HEP is enabled for both trainer and inference.

    Current test case: 
    (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp) = (1, 4, 1, 2, 2, 2).
    """
    # parallel_size: (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp)
    parallel_size = (1, 4, 1, 2, 2, 2)
    
    engine = create_engine(parallel_size)
    param_sync_group = engine.model_manager.parameter_sync_groups["ppo_policy2policy"]

    assert param_sync_group.ep_num_mapping == parallel_size[0] / parallel_size[3]
    assert param_sync_group.tp_num_mapping == parallel_size[1] // parallel_size[4]
    assert param_sync_group.hep_num_mapping == 1

    actor2rank = param_sync_group.actor2rank

    # Judge routed experts and parameters except routed experts
    comm_pair_routed_experts = []
    comm_pair_stage_1 = []
    comm_pair_stage_2 = []

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_for_routed_experts.items():
        for dst_rank in dst_ranks:
            comm_pair_routed_experts.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_1.append((actor2rank[src_rank], actor2rank[dst_rank]))

    for src_rank, dst_ranks in param_sync_group.send_recv_actor_mappings_stage2.items():
        for dst_rank in dst_ranks:
            comm_pair_stage_2.append((actor2rank[src_rank], actor2rank[dst_rank]))

    # The replica iter for routed experts will be reversed because num_src_tensor_parallel == 1 
    # and src_model is colocated with dst_model (Please see parameter_sync.py#L257-260).
    assert comm_pair_routed_experts == [
        (0, 8), (0, 12), (1, 9), (1, 13),
        (2, 10), (2, 14), (3, 11), (3, 15),
        (4, 8), (4, 12), (5, 9), (5, 13),
        (6, 10), (6, 14), (7, 11), (7, 15)
    ], f"{comm_pair_routed_experts}"
    assert sorted(comm_pair_stage_1) == [
        (0, 8), (1, 11), # not (1, 10) because of gpu collision
        (2, 13), (3, 14), # not (2, 12) because of load balance
        (4, 9), (5, 10), # not (5, 11) because of load balance
        (6, 12), (7, 15) # not (6, 13) because of gpu collision
    ], f"{sorted(comm_pair_stage_1)}"
    assert comm_pair_stage_2 == [
        (8, 9), (9, 8), (10, 11), (11, 10),
        (12, 13), (13, 12), (14, 15), (15, 14)
    ], f"{comm_pair_stage_2}"

    engine.stop()
    print(f"pass test_case (dst_ep, dst_tp, dst_pp, src_ep, src_tp, src_pp): {parallel_size}")


TEST_CASE = [
    test_balanced_hep_1,
    test_balanced_hep_2,
    test_balanced_hep_3,
    test_balanced_hep_4,
    test_balanced_hep_5
]