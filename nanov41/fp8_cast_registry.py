"""Producer-fused FP8 cast registry (R83).

Producer kernels (RMSNorm, SwiGLU, wo_a, ...) can optionally emit an FP8 copy of their bf16
output plus per-row scales ("cast fused into the producer": the official V4 Fig.5 pattern,
"SwiGLU + FP8 Cast").  The consumer (``nanov41.fused_fp8_linear``) then skips its standalone
quantisation pass and reads the producer's fp8 buffer directly.

Handoff contract
----------------
* producer: ``register(y, y8, s_row)`` where ``y`` is the **exact tensor object** it returns,
  ``y8`` is fp8 ``[T, K]`` contiguous, ``s_row`` is fp32 ``[T, 1]`` contiguous (per-row scale);
* consumer: ``lookup(x)``; returns ``None`` or ``(y8, s_row)`` only if the registered object is
  *identical* to ``x`` (weakref identity check — protects against ``id()`` reuse after GC);
* safe inside CUDA-graph capture: the producer writes its static (capture-pool) buffers and the
  consumer's kernel was recorded against them; replay order guarantees producer-before-consumer.

Env gate: ``NANOV41_FP8_CAST_FUSE=1`` (read by the producer modules; default OFF).
"""
from __future__ import annotations

import os
import weakref

_ENTRIES: dict[int, tuple] = {}
_PTR_ENTRIES: dict[int, tuple] = {}
_MAX = 4096
_HITS = [0, 0]  # [hits, misses]
_ENABLED: list[bool | None] = [None]


def enabled() -> bool:
    v = _ENABLED[0]
    if v is None:
        raw = os.environ.get("NANOV41_FP8_CAST_FUSE", "0").strip().lower()
        v = raw not in ("", "0", "false", "off", "no")
        _ENABLED[0] = v
    return v


def register(y, y8, s_row) -> None:
    if y is None or y8 is None or s_row is None:
        return
    if len(_ENTRIES) > _MAX:
        _ENTRIES.clear()
        _PTR_ENTRIES.clear()
    ref = weakref.ref(y)
    _ENTRIES[id(y)] = (ref, y8, s_row)
    # secondary index: autograd hands a *view* (same storage, different Python object) of the
    # producer output to the consumer in several bwd chains; the identity check above misses those.
    try:
        _PTR_ENTRIES[y.data_ptr()] = (ref, y8, s_row)
    except Exception:  # noqa: BLE001
        pass


def lookup(x):
    ent = _ENTRIES.get(id(x))
    if ent is not None:
        ref, y8, s_row = ent
        if (ref() is x and y8 is not None and s_row is not None
                and x.numel() == y8.numel()):
            _HITS[0] += 1
            return y8, s_row
    else:
        # view / re-wrapped tensor of a registered producer output (same storage, offset 0):
        # the producer's weakref must still be alive, otherwise the address may have been reused.
        ent = _PTR_ENTRIES.get(x.data_ptr())
        if ent is not None:
            ref, y8, s_row = ent
            obj = ref()
            if (obj is not None and y8 is not None and s_row is not None
                    and x.numel() == y8.numel()):
                _HITS[0] += 1
                return y8, s_row
    _HITS[1] += 1
    return None


def hits() -> tuple[int, int]:
    return _HITS[0], _HITS[1]


def reset_stats() -> None:
    _HITS[0] = _HITS[1] = 0


def clear() -> None:
    _ENTRIES.clear()
    _PTR_ENTRIES.clear()
