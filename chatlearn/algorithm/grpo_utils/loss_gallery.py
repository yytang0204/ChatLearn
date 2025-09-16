"""pg loss"""
from typing import List

import torch

def calculate_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: List[float],
    rollout_logprobs: torch.Tensor=None,
    diff_clip_ratio: float = 10,
    pos_clip_ratio: float = 0.2,
    neg_clip_ratio: float = 0.2,
    final_clip_ratio: float = 3.0,
    tis_ratio: float = 5.0
):
    loss_metric = {}
    logprobs_diff = log_probs - old_log_probs
    # clip logprobs_diff before exp to avoid overflow
    logprobs_diff = torch.clamp(logprobs_diff, max=diff_clip_ratio)
    ratio = torch.exp(logprobs_diff)
    advantages = torch.tensor(advantages).to(logprobs_diff.device)
    pg_loss = -advantages.unsqueeze(-1) * ratio
    # Upper and lower bound clip
    pg_loss_2 = -advantages.unsqueeze(-1) * torch.clamp(
        ratio, 1 - neg_clip_ratio, 1 + pos_clip_ratio
    )
    pg_loss_clip = torch.max(pg_loss, pg_loss_2)
    pg_loss_upperbound = torch.ones_like(pg_loss) * final_clip_ratio
    # final clip on loss
    pg_loss = torch.min(pg_loss_clip, pg_loss_upperbound)
    if rollout_logprobs is not None and tis_ratio > 0:
        negative_approx_kl2 = old_log_probs - rollout_logprobs.to(old_log_probs.dtype)
        ratio_behav = torch.exp(negative_approx_kl2)
        ratio_behav = torch.clamp(ratio_behav, max=tis_ratio)
        valid_ratio_behave = ratio_behav < tis_ratio
        loss_metric["ratio_behav"] = valid_ratio_behave
        pg_loss = pg_loss * ratio_behav

    # check pg_loss nan
    assert not torch.isnan(pg_loss).any(), "pg loss is nan"

    return pg_loss.contiguous(), loss_metric

def calculate_gspo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    diff_clip_ratio: float = 10,
    pos_clip_ratio: float = 1e-3,
    neg_clip_ratio: float = 1e-3,
    final_clip_ratio: float = 0.01,
    loss_mask: torch.Tensor = None
):

    logprobs_diff = log_probs - old_log_probs
    valid_values = torch.where(loss_mask.bool(), logprobs_diff.detach(), 0.0)
    logprobs_diff_mean = (valid_values * loss_mask).sum(axis=1) / (loss_mask.sum(axis=1) + 1e-8)
    seq_logprobs_diff = log_probs - log_probs.detach() + logprobs_diff_mean.unsqueeze(1)
    logprobs_diff = torch.clamp(seq_logprobs_diff, max=diff_clip_ratio)

    ratio = torch.exp(logprobs_diff)
    advantages.unsqueeze_(-1)

    pg_loss = -advantages * ratio
    # Upper and lower bound clip
    is_positive_clipped = (ratio > (1 + pos_clip_ratio)) * (advantages > 0)
    is_negative_clipped = (ratio < (1 - neg_clip_ratio)) * (advantages < 0)

    pg_loss_2 = -advantages * torch.clamp(
        ratio, 1 - neg_clip_ratio, 1 + pos_clip_ratio
    )
    pg_loss_clip = torch.max(pg_loss, pg_loss_2)

    is_clipped = pg_loss_2 > pg_loss
    pg_loss_upperbound = torch.ones_like(pg_loss) * final_clip_ratio
    # final clip on loss
    loss = torch.min(pg_loss_clip, pg_loss_upperbound)

    # check pg_loss nan
    assert not torch.isnan(loss).any(), "pg loss is nan"

    return (loss.contiguous(), is_positive_clipped, is_negative_clipped, is_clipped)
    