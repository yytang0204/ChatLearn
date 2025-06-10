import torch
import time
import re
import os
import functools
from typing import Callable, List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (size_based_auto_wrap_policy, transformer_auto_wrap_policy, lambda_auto_wrap_policy,
                                        _or_policy)
import torch
from transformers.trainer_pt_utils import get_module_class_from_name
from torch import optim, nn

import sys
from einops import rearrange

import random
import numpy as np
from examples.fsdp.models.grpo.trainer_utils import logprobs_from_logits, sp_split, generate_loss_mask_position_ids, bin_packing, prepare_packing_attn_mask, regroup_data_packing

from flash_attn.bert_padding import pad_input, unpad_input

# dist.init_process_group(backend="nccl")
# rank = int(os.environ.get('RANK'))
qwen3_config = AutoConfig.from_pretrained("/mnt/workspace/chatlearn-exp/hub/hf/Qwen3_8B")
qwen3_model = AutoModelForCausalLM.from_pretrained("/mnt/workspace/chatlearn-exp/hub/hf/Qwen3_8B", attn_implementation="flash_attention_2", trust_remote_code=True).to(torch.bfloat16).cuda()
qwen3_tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/chatlearn-exp/hub/hf/Qwen3_8B")

data_input = torch.load("data_after_process_rank7.pt")
inputs = data_input[0]
unpad_tokens = pad_input(
                    inputs['all_tokens'][0, :].unsqueeze(-1), 
                    inputs['indices'], 
                    inputs['ori_batch_size'], 
                    inputs['ori_seq_len']).squeeze(-1)
print(unpad_tokens)
output = qwen3_model(unpad_tokens[0:1,:].cuda()).logits
logprobs = logprobs_from_logits(output, torch.roll(unpad_tokens[0:1,:], -1, dims=1).cuda())
logprobs = logprobs * inputs['loss_mask'][0:1, :].cuda()
diff = torch.sum(logprobs - inputs['old_logprobs'][0:1,:].cuda())
print(diff)