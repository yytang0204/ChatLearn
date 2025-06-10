from transformer_engine.pytorch import GroupedLinear
import torch
import numpy as np

groupgemm_layer = GroupedLinear(
    num_gemms = 32,
    in_features=2048,
    out_features=2048,
    sequence_parallel=False,
)
hidden_states = [[] for i in range(3)]
for i in range(3):
    for j in range(32):
        ep_token_size = np.random.randint(32) + 1
        hidden_states[i].append(torch.randn(ep_token_size, 2048))

# exp0: [00000111111]
# outp: [00000111111]

# exp1: [00000222]
# outp: [00000222]

# cal sample 0 and sample 1
# 10 11 -》 21
# 10 12 -〉 22
input_tensors = [torch.cat([hidden_states[0][j], hidden_states[1][j]], dim=0) for j in range(32)]
m_splits = [input_tensor.shape[0] for input_tensor in input_tensors]
input_tensors = torch.cat(input_tensors, dim=0)
output_1 = groupgemm_layer(input_tensors.cuda(), m_splits)
print(m_splits)
print(output_1[0:m_splits[0],:])
output_sample_0_1 = torch.cat([output_1[sum(m_splits[:j]):sum(m_splits[:j+1]),:][:hidden_states[0][j].shape[0], :] for j in range(32)], dim=0)
# cal sample 0 and sample 2
# 10 9 -》 19
# 10 15 -〉 25
input_tensors = [torch.cat([hidden_states[0][j], hidden_states[2][j]], dim=0) for j in range(32)]
m_splits = [input_tensor.shape[0] for input_tensor in input_tensors]
input_tensors = torch.cat(input_tensors, dim=0)
output_1 = groupgemm_layer(input_tensors.cuda(), m_splits)
output_sample_0_2 = torch.cat([output_1[sum(m_splits[:j]):sum(m_splits[:j+1]),:][:hidden_states[0][j].shape[0], :] for j in range(32)], dim=0)
print(output_sample_0_1)
print(output_sample_0_2)
print(torch.sum(output_sample_0_1 - output_sample_0_2))

# forward
[0 0, 1 1, 2 2, 3 3]
[0 0 1 1, 2 2 3 3]

# train
[0 0, 3 3, 2 2, 1 1]
[0 0 3 3, 2 2 1 1]
