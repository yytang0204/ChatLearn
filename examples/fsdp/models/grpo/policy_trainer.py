from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from flash_attn.bert_padding import pad_input, unpad_input

from chatlearn import FSDPModule
from chatlearn.utils import to_device
from chatlearn.utils.communication_op import get_sp_parallel_group, gather

from .loss_gallery import calculate_grpo_loss
from .trainer_utils import logprobs_from_logits, sp_split, generate_loss_mask_position_ids, bin_packing, prepare_packing_attn_mask, regroup_data_packing
import time

REF_TAG = "ref_logprobs"
OLD_TAG = "old_logprobs"

def all_gather_get_rank(local_tensor, world_size):
    tensor_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, local_tensor)
    return tensor_list[7]

class PolicyTrainer(FSDPModule):

    def setup(self):
        super().setup()
        self._metric_prefix = "policy_trainer"

    def preprocess_data_list(self, data_list, training:bool):
        top = time.time()
        if self.packing:
            # When packing is enabled, data_list will only contain one microbatch
            # True microbatch will be regrouped
            if not training:
                keys_forward = ["all_tokens", "prompt_token_length", "response_token_length"]
            else:
                keys_forward = ["all_tokens", "prompt_token_length", "response_token_length", "advantages", REF_TAG, OLD_TAG, OLD_TAG + '_rank', REF_TAG + '_rank']
            data_list = regroup_data_packing(data_list, keys_forward, self.max_token_in_seq)

        data_after_process = []
        for data_b in data_list:
            tokens_ = data_b["all_tokens"].long()
            prompt_token_length = data_b["prompt_token_length"]
            response_token_length = data_b["response_token_length"]
            ori_batch_size, ori_seq_len = tokens_.size()
            if self.packing:
                #TODO: remove pad first
                print("do packing")
                # Packing data into one batch
                attn_mask, loss_mask, position_ids = generate_loss_mask_position_ids(tokens_, prompt_token_length, response_token_length)
                tokens_, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(tokens_.unsqueeze(-1).cuda(), attn_mask.cuda())
                tokens_ = tokens_.permute(1,0).cpu() # For compatible with transformers

                position_ids, _, _, _ = unpad_input(position_ids.unsqueeze(-1).cuda(), attn_mask.cuda())
                position_ids = position_ids.permute(1,0).cpu() # For compatible with transformers
                # Pad tokens_ to ensure max valid token length meets sp requirements
                pad_size = 0
                if self.sp_size > 1:
                    # Pad inputs to ensure seq_len is divisible by sp_size
                    valid_len = tokens_.shape[1]
                    pad_size = math.ceil(valid_len / self.sp_size) * self.sp_size - valid_len

                    tokens = F.pad(tokens_, (0, pad_size), value=self.tokenizer.pad_token_id)
                    position_ids = F.pad(position_ids,(0, pad_size), value=pad_size)

                    labels = torch.roll(tokens_, shifts=-1, dims=1)

                    # Split tensor by sp_size
                    sp_group = get_sp_parallel_group()
                    sp_local_rank = dist.get_rank(sp_group)
                    tokens = sp_split(input_tensor=tokens, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                    labels = sp_split(input_tensor=labels, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                else:
                    tokens = tokens_
                    labels = torch.roll(tokens, shifts=-1, dims=1)
                #micro_batch_seqlen = [prompt_len + response_len for prompt_len, response_len in zip(prompt_token_length, response_token_length)]
                #attention_mask = prepare_packing_attn_mask(micro_batch_seqlen, dtype=torch.get_default_dtype(), pad_size=pad_size)
                if not training:
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "labels": labels,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "indices": indices,
                            "bin_ids": data_b["bin_ids"],
                            "bin_seqlen": data_b["bin_seqlen"],
                            "pad_size": pad_size,
                        }
                    )
                else:
                    loss_mask = torch.roll(attn_mask, shifts=-1, dims=1)
                    loss_mask[:, -1] = 0
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "labels": labels,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "indices": indices,
                            "bin_ids": data_b["bin_ids"],
                            "bin_seqlen": data_b["bin_seqlen"],
                            "pad_size": pad_size,
                            "loss_mask": loss_mask,
                            "old_logprobs": data_b[OLD_TAG],
                            "ref_logprobs": data_b[REF_TAG],
                            "advantages": data_b["advantages"],
                            REF_TAG+'_rank': torch.tensor(data_b[REF_TAG+'_rank']),
                            OLD_TAG+'_rank': torch.tensor(data_b[OLD_TAG+'_rank']),
                        }
                    )
            else:
                # If packing is disabled, data list will contain a 
                attn_mask, loss_mask, position_ids = generate_loss_mask_position_ids(tokens_, prompt_token_length, response_token_length)
                pad_size = 0
                if self.sp_size > 1:
                    # Pad inputs to ensure seq_len is divisible by sp_size
                    valid_len = tokens_.shape[1]
                    pad_size = math.ceil(valid_len / self.sp_size) * self.sp_size - valid_len

                    tokens = F.pad(tokens_, (0, pad_size), value=self.tokenizer.pad_token_id)
                    position_ids = F.pad(position_ids,(0, pad_size), value=pad_size)

                    labels = torch.roll(tokens_, shifts=-1, dims=1)

                    # Split tensor by sp_size
                    sp_group = get_sp_parallel_group()
                    sp_local_rank = dist.get_rank(sp_group)
                    tokens = sp_split(input_tensor=tokens, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                    labels = sp_split(input_tensor=labels, split_dim=1, sp_size=self.sp_size, sp_local_rank=sp_local_rank)
                else:
                    tokens = tokens_
                    labels = torch.roll(tokens, shifts=-1, dims=1)
                if not training:
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "labels": labels,
                            "pad_size": pad_size,
                        }
                    )
                else:
                    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1)
                    data_after_process.append(
                        {
                            "all_tokens": tokens,
                            "position_ids": position_ids,
                            "ori_seq_len": ori_seq_len,
                            "ori_batch_size": ori_batch_size,
                            "labels": labels,
                            "pad_size": pad_size,
                            "loss_mask": loss_mask,
                            "old_logprobs": data_b[OLD_TAG],
                            "ref_logprobs": data_b[REF_TAG],
                            "advantages": data_b["advantages"],
                        }
                    )
        local_rank = dist.get_rank()
        # if local_rank == 7 and training:
        #     torch.save(data_after_process, "/mnt/workspace/chatlearn-exp/test/fsdp_chatlearn/charlearn_open/ChatLearn/data_after_process_rank7.pt")
        if training:
            data_after_process = torch.load("/mnt/workspace/chatlearn-exp/test/fsdp_chatlearn/charlearn_open/ChatLearn/data_after_process_rank7.pt")
            # dist.barrier()
            # exit()
        print(f"debugyy total preprocess time cost: {time.time()-top}")
        return data_after_process

    def train_step(self, data_list):
        '''
        data_list: list of micro batchs [micro_bs0, micro_bs1]
        '''
        top = time.time()
        self.model.train()  # reset model state
        self.optimizer.zero_grad()
        pg_loss_list = []
        entropy_loss_list = []
        kl_loss_list = []
        sp_group = get_sp_parallel_group()
        data_list = self.preprocess_data_list(data_list=data_list, training=True)
        for inputs in data_list:
            #inputs = self.preprocess_data_train(data_b)
            for k, v in inputs.items():
                inputs[k] = to_device(torch.cuda.current_device(), v)
            output = self.model(
                input_ids=inputs["all_tokens"],
                attention_mask=None,
                position_ids=inputs["position_ids"],
                use_cache=False
            )
            logprobs = logprobs_from_logits(output.logits, inputs["labels"])
            if sp_group is not None:
                logprobs = gather(input_tensor=logprobs, sp_group=sp_group, gather_dim=1)
            if self.packing:
                # Recover packing sequence
                logprobs = pad_input(
                    logprobs[0, :logprobs.shape[1] - inputs['pad_size']].unsqueeze(-1), 
                    inputs['indices'], 
                    inputs['ori_batch_size'], 
                    inputs['ori_seq_len']).squeeze(-1)
            else:
                logprobs_len = logprobs.shape[1]
                logprobs = F.pad(logprobs, (0, inputs['ori_seq_len'] - logprobs_len), mode='constant', value=0)
            loss = calculate_grpo_loss(
                log_probs=logprobs,
                old_log_probs=inputs["old_logprobs"],
                advantages=inputs["advantages"],
                diff_clip_ratio=self.module_args.args_dict.get("diff_clip_ratio", 10),
                pos_clip_ratio=self.module_args.args_dict.get("pos_clip_ratio", 0.2),
                negative_clip_ratio=self.module_args.args_dict.get("negative_clip_ratio", 0.2),
                final_clip_ratio=self.module_args.args_dict.get("final_clip_ratio", 3)
                )
            
            pg_loss = torch.masked_select(loss, inputs["loss_mask"].bool())
            # Reference: https://github.com/pytorch/pytorch/blob/c45515c2eda19b1a1ff5762f1571c6fe63773c8a/torch/distributed/fsdp/_runtime_utils.py#L848
            # Since grad will be divided by fsdp world size in backward hook
            # We need to multiple pg_loss_mean by sp_size to avoid mean calculate of grad within dp rank
            # Sample based loss
           # print(f"debugyy dp_rank: {torch.distributed.get_rank()}, pg_shape: {pg_loss.shape}, pg_sum:  {torch.sum(pg_loss)}, num > 0: {torch.nonzero(pg_loss != 0)}")
            pg_loss_mean = torch.mean(pg_loss) / self.train_global_batch_size * self.sp_size
            pg_loss_mean.backward()
            pg_loss_list.append(pg_loss)

            # kl loss
            kl = inputs['ref_logprobs'] - logprobs
            ratio = torch.exp(kl)
            assert not torch.isinf(ratio).any(), "kl loss ratio has inf values"
            assert not torch.isnan(ratio).any(), "kl loss ratio has nan values"
            kld = (ratio - kl - 1).contiguous()
            kl_loss = torch.clamp(kld, min=-10, max=10)
            kl_loss = torch.masked_select(kl_loss, inputs["loss_mask"].bool())
            kl_diff = torch.masked_select(kl, inputs["loss_mask"].bool())
            per_sample_diff = torch.sum((kl * inputs["loss_mask"])!=0, dim=1)
            print(per_sample_diff)
            current_rank = torch.distributed.get_rank()
            print(f"debugyy dp_rank: {torch.distributed.get_rank()}, kl_shape: {kl_diff.shape}, kl_sum:  {torch.sum(kl_diff)}, num > 0: {torch.sum(kl_diff != 0)}, non_zero_sample: {torch.nonzero(per_sample_diff)}")# + \
                    #f"debugyy wrong rank kl: {torch.sum((inputs[REF_TAG + '_rank'] - current_rank)!=0)}, debugyy wrong rank pg: {torch.sum((inputs[OLD_TAG + '_rank'] - current_rank)!=0)}")
            kl_loss_list.append(kl_loss)

            # entropy loss
            entropy_loss = torch.masked_select(-logprobs, inputs["loss_mask"].bool())
            entropy_loss_list.append(entropy_loss)
        grad_norm = self.model.clip_grad_norm_(max_norm=self.module_args.args_dict.get("grad_clip", 1)).detach().item()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # collect metric
        pg_loss = torch.mean(torch.cat(pg_loss_list)).detach().item()
        kl_loss = torch.mean(torch.cat(kl_loss_list)).detach().item()
        entropy_loss = torch.mean(torch.cat(entropy_loss_list)).detach().item()

        train_stats = {
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "grad_norm": grad_norm
        }
        self._metric_list.append(train_stats)

    def forward_step(self, data):
        start = time.time()
        total_size = data['all_tokens'].shape[0]
        ori_seq_len = data['all_tokens'].shape[1]
        #print(f"forward ori_size: {total_size}, ori_seq_len: {ori_seq_len}")
        data_list = self.preprocess_data_list(data_list=[data], training=False)
        output_logprobs = torch.empty(total_size, ori_seq_len, dtype=torch.bfloat16)
        print(f"debugyy data_length forward: {len(data_list)}")
        token_in_seq = []
        for inputs in data_list:
            for k, v in inputs.items():
                inputs[k] = to_device(torch.cuda.current_device(), v)
            with torch.no_grad():
                output = self.model(
                    input_ids=inputs['all_tokens'],
                    attention_mask=None,#inputs['attention_mask'],
                    position_ids=inputs['position_ids'],
                    use_cache=False
                )
                sp_group = get_sp_parallel_group()
                logprobs = logprobs_from_logits(output.logits, inputs["labels"])
                if sp_group is not None:
                    logprobs = gather(input_tensor=logprobs, sp_group=sp_group, gather_dim=1)
                # Repad logprobs to max_seq_len to allow concatenation
                if self.packing:
                    # Recover packing sequence
                    logprobs = pad_input(
                        logprobs[0, :logprobs.shape[1] - inputs['pad_size']].unsqueeze(-1), 
                        inputs['indices'], 
                        inputs['ori_batch_size'], 
                        inputs['ori_seq_len']).squeeze(-1)
                else:
                    logprobs_len = logprobs.shape[1]
                    logprobs = F.pad(logprobs, (0, inputs['ori_seq_len'] - logprobs_len), mode='constant', value=0)
            if self.packing:
                output_logprobs[torch.tensor(inputs["bin_ids"])] = logprobs.cpu()
                token_in_seq.append(sum(inputs["bin_seqlen"]))
            else:
                output_logprobs = logprobs.cpu()
        rank_caculate = torch.distributed.get_rank() 
        tag = OLD_TAG
        if OLD_TAG in data.keys():
            tag = REF_TAG
        data.update({tag: output_logprobs})
        data.update({tag + '_rank': [rank_caculate] * total_size})
        print(f"debugyy rank {torch.distributed.get_rank()}, total cost of forward: {time.time() - start}, forward ori_size: {total_size}, ori_seq_len: {ori_seq_len}, seq in pack {token_in_seq}", flush=True)
        print(f"debugyy data_length: {len(data_list)}", flush=True)
        return data