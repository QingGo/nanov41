"""Minimal device helpers for the vendored Engram gate kernels (CUDA only)."""
from __future__ import annotations

import torch

__all__ = ["get_num_sms", "get_max_smem_per_sm", "get_pdl", "is_ascend"]


def is_ascend() -> bool:
    return False


def get_num_sms() -> int:
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


def get_max_smem_per_sm() -> int:
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_multiprocessor)


def get_pdl() -> bool:
    return False
