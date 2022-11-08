from dataclasses import dataclass
from typing import List

from torch.functional import Tensor


@dataclass
class BatchSlice:
    i_from: int
    i_to: int
    i_start: int


def get_slices(batch: Tensor, dim=1, window: int = 16, overlap: int = 8) -> List[BatchSlice]:
    num_imgs = batch.size(dim)
    if num_imgs <= window:
        return [BatchSlice(0, num_imgs, 0)]
    stride = window - overlap
    result = []
    current_idx = 0
    while True:
        next_idx = current_idx + window

        if next_idx >= num_imgs:
            current_idx = num_imgs - window
            offset = overlap // 2 if current_idx > 0 else 0
            next_idx = num_imgs
            result.append(BatchSlice(current_idx, next_idx, offset))
            break
        else:
            offset = overlap // 2 if current_idx > 0 else 0
            result.append(BatchSlice(current_idx, next_idx, offset))
        current_idx += stride
    return result
