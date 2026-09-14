import torch

from slime.utils.types import Sample
from torch.nn.functional import kl_div


# parametric reward model
async def parametric_reward(args, sample: Sample) -> float:
    print(sample.response)
    tlp_tensor = torch.tensor(sample.teacher_log_probs)
    rlp_tensor = torch.tensor(sample.rollout_log_probs)
    mask_tensor = torch.tensor(sample.loss_mask)
    kl_div: float = kl_div(tlp_tensor * mask_tensor, rlp_tensor * mask_tensor, reduction="average")
    parametric_reward: float = -kl_div
    return parametric_reward

async def batched_parametric_rm(args, samples: list[Sample]) -> list[float]:
    return [await parametric_reward(args, sample) for sample in samples]