# Copyright 2024 Alibaba-inc. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
"""rule reward"""
from typing import Dict, List, Any
from collections import deque, defaultdict
import uuid
import random
import time

import torch
from chatlearn.data.prompt_dataset import VLLMPromptPipeline

from chatlearn import BaseModule

class RolloutManager(BaseModule):
    """rule reward"""

    def setup(self):
        self.stats = {}
        self._metric_prefix = "rollout_manager"
        self.rollout_finished_no_train = defaultdict(list)
        self.rollout_not_finished = []
        self.max_rollout_round = self.module_args.get("max_rollout_round")
        self.max_gen_len = self.module_args.get("max_gen_len")
        self.num_inference_per_prompt = self.module_args.get("num_inference_per_prompt")
        self.recompute_logprobs = False

    def build_dataset(self, prompts: List[Dict], is_eval=False):
        # prompts seems like the total data set by engine.set_dataset(dataset)
        seq_length = self.module_args.get("seq_length")
        tokenizer_path = self.module_args["load"]

        prompts_dataset = VLLMPromptPipeline(
            prompts,
            seq_length,
            tokenizer_path,
            enable_thinking=self.module_args.get("enable_thinking", False),
        )
        return prompts_dataset

    def split_data(self, data:Dict[str, Any], first_stage=True):
        for key in data:
            if isinstance(data[key], list):
                continue
            elif isinstance(data[key], torch.Tensor):
                data[key] = torch.split(data[key], 1, dim=0)
        data_list = [dict(zip(data.keys(), values)) for values in zip(*data.values())]

        if first_stage:
            for i in range(len(data_list)):
                data_list[i]["uuid"] = uuid.uuid4()
                data_list[i]["prompt_uid"] = hash(data_list[i]["prompt"])
                data_list[i]["rollout_round"] = 0
                data_list[i]["str_outputs"] = ""
                data_list[i]["original_prompt"] = data_list[i]["prompt"]
                data_list[i]["prompt_token_ids"] = data_list[i]["input_ids"]
                data_list[i]["prompt_token_length"] = len(data_list[i]["input_ids"])
                data_list[i]["response_token_length"] = 0
                data_list[i]["all_tokens"] = []
        return len(data_list), data_list
    
    def merge_data(self, data_list: List[dict]):
        if len(data_list) == 0:
            return {}
        keys = data_list[0].keys()
        data = {key: [d[key] for d in data_list] for key in keys}
        for key in data:
            if isinstance(data[key][0], torch.Tensor):
                data[key] = torch.cat(data[key], dim=0)
        return data

    def get_sample_for_rollout(self, data: Dict[str, Any]):
        # if len(self.rollout_not_finished) == 0:
        #     return data
        # else:
        start = time.time()
        print(f"debugyy episode id: {self._episode_id}")
        sample_per_episode, data_list = self.split_data(data)
        self.rollout_not_finished.extend(data_list)
        train_batch = self.rollout_not_finished[:sample_per_episode]
        for single_data in train_batch:
            if "start_episode" not in single_data:
                single_data["start_episode"] = self._episode_id
        random.shuffle(train_batch)
        print("data preprocess time: %.3f" % (time.time() - start), flush=True)
        return self.merge_data(train_batch)

    def remove_pad(self, data_b, total_response_len):
        valid_seqlen = data_b["prompt_token_length"] + total_response_len
        input_ids = data_b["all_tokens"][0, :valid_seqlen]
        return input_ids

    def is_finished(self, data_b):
        # determine whether the rollout is finished
        #print(f"response len: {data_b["response_token_length"]}, rollout round: {data_b["rollout_round"]}")
        return (data_b["response_token_length"] < self.max_gen_len) or \
            (data_b["rollout_round"] == self.max_rollout_round)

    def update_data(self, data, rollout_result, is_finished):
        assert data["uuid"] == rollout_result["uuid"]
        data["str_outputs"] += rollout_result["str_outputs"]
        data["rollout_round"] = rollout_result["rollout_round"]
        data["response_token_length"] += rollout_result["response_token_length"]
        data["input_ids"] = self.remove_pad(rollout_result, data["response_token_length"]).tolist()
        if is_finished:
            data["all_tokens"] = rollout_result["all_tokens"]
        return data

    def post_process_rollout_results(self, data):
        start = time.time()
        sample_per_episode, data_list = self.split_data(data,first_stage=False)
        finished_uuid = []
        unfinished_data = []
        for data_b in data_list:
            uuid = data_b["uuid"]
            prompt_uid = data_b["prompt_uid"]
            finished = self.is_finished(data_b)
            for idx, data_ori in enumerate(self.rollout_not_finished[:sample_per_episode]):
                if data_ori["uuid"] == uuid:
                    data_b = self.update_data(data_ori, data_b, finished)
            if finished:
                # Finished, add data to self.rollout_finished_no_train[prompt_uid]
                self.rollout_finished_no_train[prompt_uid].append(data_b)
                finished_uuid.append(uuid)
            else:
                # If not finished, update data in rollout_not_finished
                # for idx, data_ori in enumerate(self.rollout_not_finished[:sample_per_episode]):
                #     if data_ori["uuid"] == uuid:
                unfinished_data.append(data_b)
        # update remaining data
        unfinished_data.extend(self.rollout_not_finished[sample_per_episode:])
        self.rollout_not_finished = unfinished_data
        train_data = []
        pop_keys = []
        for key, data_list in self.rollout_finished_no_train.items():
            if len(data_list) == self.num_inference_per_prompt:
                train_data.extend(data_list)
                pop_keys.append(key)
        for key in pop_keys:
            self.rollout_finished_no_train.pop(key)
        print(f"debugyy final sum train: {len(train_data)}")
        print("data preprocess time: %.3f" % (time.time() - start), flush=True)
        return self.merge_data(train_data)
