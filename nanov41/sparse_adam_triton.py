# SPDX-License-Identifier: MIT
"""Fused sparse-Adam row update for Engram tables (one Triton kernel per table per step).

Replaces the torch sequence `index_select -> gather m/v -> fused math -> index_copy_` that dominated
the graph step's out-of-graph sparse optimizer.  Each program updates one row block of the table and
its moments; rows are unique/sorted (guaranteed by the deterministic segment-sum collector) and the
optimizer update is row-independent, so this keeps the training path bitwise reproducible.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = None
    tl = None
    _HAS_TRITON = False

__all__ = ["triton_available", "sparse_adam_update", "segment_adam_update"]


def triton_available() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:
    @triton.jit
    def _sparse_adam_kernel(weight, rows, grads, row_to_slot, m_buf, v_buf,
                            lr, b0, b1, bias0, bias1, eps,
                            D: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)
        row = tl.load(rows + i)
        slot = tl.load(row_to_slot + row)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        w_ptr = weight + row * D
        g_ptr = grads + i * D
        m_ptr = m_buf + slot * D
        v_ptr = v_buf + slot * D
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        m = b0 * m + (1.0 - b0) * g
        v = b1 * v + (1.0 - b1) * g * g
        tl.store(m_ptr + offs, m, mask=mask)
        tl.store(v_ptr + offs, v, mask=mask)
        new = w - lr * (m / bias0) / (tl.sqrt(v / bias1) + eps)
        tl.store(w_ptr + offs, new.to(weight.dtype.element_ty), mask=mask)


def sparse_adam_update(weight: torch.Tensor, rows: torch.Tensor, grads: torch.Tensor,
                       row_to_slot: torch.Tensor, m_buf: torch.Tensor, v_buf: torch.Tensor,
                       *, lr: float, b0: float, b1: float, bias0: float, bias1: float,
                       eps: float) -> None:
    """In-place update of ``weight[rows]`` and ``m_buf/v_buf[row_to_slot[rows]]``."""
    n_rows = rows.numel()
    if n_rows == 0:
        return
    dim = weight.shape[1]
    block = max(32, triton.next_power_of_2(int(dim)))
    grid = (n_rows,)
    _sparse_adam_kernel[grid](weight, rows, grads, row_to_slot, m_buf, v_buf,
                              float(lr), float(b0), float(b1), float(bias0), float(bias1),
                              float(eps), D=int(dim), BLOCK=int(block), num_warps=1)


if _HAS_TRITON:
    @triton.jit
    def _segment_adam_kernel(weight, uniq, starts, counts, sorted_grad, row_to_slot,
                             m_buf, v_buf, lr, b0, b1, bias0, bias1, eps,
                             D: tl.constexpr, MAXC: tl.constexpr, BLOCK: tl.constexpr):
        """Segment-sum row grads (sorted ids) and apply the Adam update in one program per row."""
        pid = tl.program_id(0)
        row = tl.load(uniq + pid)
        start = tl.load(starts + pid)
        cnt = tl.load(counts + pid)
        slot = tl.load(row_to_slot + row)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        acc = tl.zeros((BLOCK,), tl.float32)
        for j in tl.static_range(0, MAXC):
            live = j < cnt
            g = tl.load(sorted_grad + (start + j) * D + offs, mask=mask & live, other=0.0)
            acc += g.to(tl.float32)
        w_ptr = weight + row * D
        m_ptr = m_buf + slot * D
        v_ptr = v_buf + slot * D
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        m = b0 * m + (1.0 - b0) * acc
        v = b1 * v + (1.0 - b1) * acc * acc
        tl.store(m_ptr + offs, m, mask=mask)
        tl.store(v_ptr + offs, v, mask=mask)
        new = w - lr * (m / bias0) / (tl.sqrt(v / bias1) + eps)
        tl.store(w_ptr + offs, new.to(weight.dtype.element_ty), mask=mask)


def segment_adam_update(weight: torch.Tensor, uniq: torch.Tensor, starts: torch.Tensor,
                        counts: torch.Tensor, sorted_grad: torch.Tensor,
                        row_to_slot: torch.Tensor, m_buf: torch.Tensor, v_buf: torch.Tensor,
                        *, lr: float, b0: float, b1: float, bias0: float, bias1: float,
                        eps: float, max_count: int = 8) -> None:
    """Fused deterministic segment-sum + sparse Adam for one Engram table."""
    n_rows = uniq.numel()
    if n_rows == 0:
        return
    dim = weight.shape[1]
    block = max(32, triton.next_power_of_2(int(dim)))
    _segment_adam_kernel[(n_rows,)](weight, uniq, starts, counts, sorted_grad, row_to_slot,
                                    m_buf, v_buf, float(lr), float(b0), float(b1),
                                    float(bias0), float(bias1), float(eps),
                                    D=int(dim), MAXC=int(max_count), BLOCK=int(block),
                                    num_warps=1)
