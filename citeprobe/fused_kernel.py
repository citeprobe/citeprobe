"""The fused sentence-attention pass (Appendix "Fused Kernel Proof").

A FlashAttention-style tiled online softmax carries a per-sentence mass
buffer through the same rescaling the output accumulator receives, so the
sentence grid of Equation 1 is produced without ever materializing the T x T
attention matrix. This is a PyTorch tile loop, not a compiled kernel: it
saves the memory, it does not claim the speed.

It is registered with transformers as the attention implementation
"citeprobe_fused". A custom implementation must also register a mask
function, otherwise transformers passes attention_mask=None and causality is
silently lost; the guard in the forward makes that loud.
"""

from dataclasses import dataclass, field

import torch
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

IMPLEMENTATION = "citeprobe_fused"


@dataclass
class FusedState:
    """Per-example inputs to the fused pass and the grids it produces.

    segments[u] is the paper's g_u shifted by one: 0 for a key outside every
    source sentence, j + 1 for a key inside sentence j. statement_average is
    the [m, T] operator that averages a query row over each statement's
    tokens. tile is the key-tile width. reduced receives one [H, m, d] grid
    per attention layer, on CPU.
    """

    segments: torch.Tensor
    statement_average: torch.Tensor
    tile: int
    reduced: list[torch.Tensor] = field(default_factory=list)


# Installed by fused_sentence_attention() for one forward pass, None otherwise.
_STATE: FusedState | None = None


def expand_key_values(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    """[B, KV, T, dh] -> [B, KV*repeats, T, dh] (grouped-query expansion)."""
    if repeats == 1:
        return tensor
    batch, kv_heads, length, head_dim = tensor.shape
    expanded = tensor[:, :, None, :, :].expand(batch, kv_heads, repeats, length, head_dim)
    return expanded.reshape(batch, kv_heads * repeats, length, head_dim)


def _fused_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Tiled online softmax with the sentence-mass reduction fused in.

    The **kwargs is mandated by the transformers AttentionInterface signature.
    Key and value tiles, not the full tensors, are grouped-query expanded, to
    keep the O(T^2) memory this pass exists to avoid.
    """
    if attention_mask is None and query.shape[2] > 1:
        raise ValueError("citeprobe_fused received no causal mask")
    state = _STATE
    if state is None:
        raise ValueError("citeprobe_fused forward without an installed state")

    num_heads = query.shape[1]
    num_queries = query.shape[2]
    head_dim = query.shape[3]
    num_keys = key.shape[2]
    num_segments = int(state.segments.max()) + 1
    # Accumulate in at least fp32, as FlashAttention does for bf16 inputs.
    accumulate_dtype = torch.promote_types(query.dtype, torch.float32)

    def zeros(*trailing: int) -> torch.Tensor:
        return torch.zeros(
            (1, num_heads, num_queries, *trailing), dtype=accumulate_dtype, device=query.device
        )

    # m = -inf, l = 0, O = 0, M = 0.
    running_max = torch.full_like(zeros(), float("-inf"))
    running_sum = zeros()
    output = zeros(head_dim)
    sentence_mass = zeros(num_segments)

    for start in range(0, num_keys, state.tile):
        stop = min(start + state.tile, num_keys)
        key_tile = expand_key_values(key[:, :, start:stop, :], module.num_key_value_groups)
        value_tile = expand_key_values(value[:, :, start:stop, :], module.num_key_value_groups)

        # S = Q K_T^T / sqrt(d), masked entries pushed to finfo.min.
        scores = torch.matmul(query, key_tile.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[..., start:stop]
        scores = scores.to(accumulate_dtype)

        # m' = max(m, rowmax S), c = exp(m - m'), P = exp(S - m').
        new_max = torch.maximum(running_max, scores.max(dim=-1).values)
        correction = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max[..., None])

        # l = l c + rowsum P;  O = O c + P V_T.
        running_sum = running_sum * correction + weights.sum(dim=-1)
        output = output * correction[..., None] + torch.matmul(
            weights.to(value_tile.dtype), value_tile
        ).to(accumulate_dtype)
        # M = M c, then M[:, g_u] += P[:, u]. Segment 0 collects the keys
        # outside every sentence and is dropped below.
        sentence_mass = sentence_mass * correction[..., None]
        sentence_mass.index_add_(-1, state.segments[start:stop], weights)
        running_max = new_max

    # O / l and M / l, rowwise.
    output = (output / running_sum[..., None]).to(query.dtype)
    sentence_mass = sentence_mass[..., 1:] / running_sum[..., None]
    # Equation 1: average the per-query sentence mass over each statement's
    # tokens, giving this layer's [H, m, d] grid.
    grid = torch.einsum(
        "mt,htd->hmd", state.statement_average.to(accumulate_dtype), sentence_mass[0]
    )
    state.reduced.append(grid.cpu())
    return output.transpose(1, 2).contiguous(), None


AttentionInterface.register(IMPLEMENTATION, _fused_attention)
AttentionMaskInterface.register(IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["eager"])


def fused_sentence_attention(
    model,
    token_ids: list[int],
    source_token_sets: list[list[int]],
    statement_token_sets: list[list[int]],
    tile: int = 256,
) -> list[torch.Tensor]:
    """Run one teacher-forced pass and return one [H, m, d] grid per attention layer.

    Builds the segment vector and the statement-average operator, installs
    them as the pass's state, switches the model to the fused implementation
    and runs the forward. Always restores "sdpa" and clears the state, even
    if the forward raises.
    """
    global _STATE
    sequence_length = len(token_ids)
    input_ids = torch.tensor([token_ids], device=model.device)

    statement_average = torch.zeros(
        len(statement_token_sets), sequence_length, device=model.device, dtype=torch.float32
    )
    for statement_index, token_set in enumerate(statement_token_sets):
        statement_average[statement_index, token_set] = 1.0 / len(token_set)

    segments = torch.zeros(sequence_length, dtype=torch.long, device=model.device)
    for source_index, token_set in enumerate(source_token_sets):
        segments[token_set] = source_index + 1

    state = FusedState(segments=segments, statement_average=statement_average, tile=tile)
    _STATE = state
    model.set_attn_implementation(IMPLEMENTATION)
    try:
        with torch.no_grad():
            # Skip the LM head over every position; the logits are never read.
            model(input_ids=input_ids, logits_to_keep=1)
    finally:
        model.set_attn_implementation("sdpa")
        _STATE = None
    return state.reduced
