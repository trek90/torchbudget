"""Offline, token-budgeted batching for PyTorch sequence inference."""

from collections.abc import Callable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

__all__ = ["plan_batches", "batched_infer"]


def plan_batches(
    lengths: Sequence[int], *, max_tokens: int, max_batch_size: int | None = None
) -> list[list[int]]:
    """Group request indices by length; each batch uses at most max_tokens slots."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if max_batch_size is not None and (
        type(max_batch_size) is not int or max_batch_size < 1
    ):
        raise ValueError("max_batch_size must be a positive integer")
    for i, length in enumerate(lengths):
        if type(length) is not int or not 1 <= length <= max_tokens:
            raise ValueError(f"lengths[{i}] must be an integer in [1, max_tokens]")

    groups: list[list[int]] = []
    current: list[int] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        size = len(current) + 1
        if size * lengths[i] > max_tokens or (
            max_batch_size is not None and size > max_batch_size
        ):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


@torch.inference_mode()
def batched_infer(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    requests: Sequence[torch.Tensor],
    *,
    max_tokens: int,
    max_batch_size: int | None = None,
    pad_id: int = 0,
) -> list[torch.Tensor]:
    """Call forward(padded_ids, valid_mask) and return per-request outputs.

    Inputs are 1D int64 tensors on a common device. The boolean mask is
    True for real tokens. Forward must return a batch-first tensor.
    The caller sets eval mode; outputs are inference tensors.
    """
    if type(pad_id) is not int or not -(2**63) <= pad_id < 2**63:
        raise ValueError("pad_id must be an int64 integer")
    lengths = []
    for i, request in enumerate(requests):
        if (
            not isinstance(request, torch.Tensor)
            or request.ndim != 1
            or request.dtype != torch.int64
            or request.layout != torch.strided
        ):
            raise ValueError(f"request {i} must be a dense 1D int64 tensor")
        if request.device != requests[0].device:
            raise ValueError("all requests must be on the same device")
        lengths.append(request.numel())
    groups = plan_batches(lengths, max_tokens=max_tokens, max_batch_size=max_batch_size)
    outputs: dict[int, torch.Tensor] = {}
    for indices in groups:
        ids = pad_sequence([requests[i] for i in indices], batch_first=True)
        sizes = torch.tensor([lengths[i] for i in indices], device=ids.device)
        mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < sizes[:, None]
        if pad_id != 0:
            ids.masked_fill_(~mask, pad_id)
        result = forward(ids, mask)
        if not isinstance(result, torch.Tensor) or result.ndim == 0 or result.shape[0] != len(indices):
            raise ValueError("forward must return a tensor with one row per request")
        for row, i in enumerate(indices):
            # Own each row's storage, even if forward reuses a scratch buffer.
            outputs[i] = result[row].clone()
    return [outputs[i] for i in range(len(requests))]
