# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PagedAttention context manager for Janus pipeline.

Provides block-based KV cache (PagedAttention) and attention metadata
setup that mirrors vLLM's model runner, enabling the Janus pipeline
to use vLLM's optimized attention kernels while maintaining the
existing CFG + AR loop structure.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.forward_context import ForwardContext, set_forward_context
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig


@dataclass
class FlashAttentionMetadata:
    """Minimal attention metadata compatible with vLLM FlashAttention backend."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    use_cascade: bool = False
    common_prefix_len: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    causal: bool = True


class JanusPagedAttentionContext:
    """Manages PagedAttention KV cache and metadata for Janus AR loop.

    Replaces StaticCache with vLLM's block-based KV cache, enabling
    PagedAttention kernels and proper CUDA graph capture via the
    vLLM CUDAGraphWrapper pattern.

    The Janus AR loop has fixed-length sequences (prompt_len + 576),
    so block allocation is pre-computed and static across all steps.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq_len: int,
        num_seqs: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.num_seqs = num_seqs
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # Compute block allocation
        self.num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        self.total_blocks = num_seqs * self.num_blocks_per_seq

        # KV cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
        # Shape convention: vLLM uses [num_blocks, 2, block_size, num_kv_heads, head_size]
        self.kv_cache = torch.zeros(
            self.total_blocks,
            2,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device=device,
        )

        # Block table: [num_seqs, num_blocks_per_seq]
        self.block_table = torch.arange(self.total_blocks, dtype=torch.int32, device=device).view(
            num_seqs, self.num_blocks_per_seq
        )

        self.seq_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device)
        self._prefill_metadata: FlashAttentionMetadata | None = None
        self._decode_metadata: FlashAttentionMetadata | None = None

    def _make_slot_mapping(self, num_tokens: int, seq_lens: torch.Tensor) -> torch.Tensor:
        """Create slot_mapping for current token positions.

        Maps each token to its KV cache slot: slot = block_idx * block_size + offset.
        """
        slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        offset = 0
        for s in range(self.num_seqs):
            seq_len = int(seq_lens[s].item()) if s < seq_lens.shape[0] else 0
            for t in range(seq_len if num_tokens > self.num_seqs else (1 if s < num_tokens else 0)):
                pos = offset + t
                block_idx = pos // self.block_size
                block_offset = pos % self.block_size
                if num_tokens > self.num_seqs:
                    pass  # Prefill: handle below
                else:
                    slot_mapping[s] = block_idx * self.block_size + block_offset
            if num_tokens > self.num_seqs:
                for t in range(seq_len):
                    pos = offset + t
                    block_idx = pos // self.block_size
                    block_offset = pos % self.block_size
                    sm_val = block_idx * self.block_size + block_offset
                    slot_mapping[offset + t] = sm_val
            offset += seq_len if num_tokens > self.num_seqs else 1
        return slot_mapping

    def build_prefill_metadata(self, prompt_len: int) -> tuple[FlashAttentionMetadata, dict[str, torch.Tensor]]:
        """Build attention metadata for the prefill step.

        All sequences have the same prompt length (input_ids after chat template).
        """
        num_tokens = self.num_seqs * prompt_len
        self.seq_lens.fill_(prompt_len)

        query_start_loc = torch.zeros(self.num_seqs + 1, dtype=torch.int32, device=self.device)
        for s in range(self.num_seqs):
            query_start_loc[s + 1] = query_start_loc[s] + prompt_len

        slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        for s in range(self.num_seqs):
            seq_start = s * prompt_len
            for t in range(prompt_len):
                pos = t
                block_idx = pos // self.block_size
                block_offset = pos % self.block_size
                physical_block = int(self.block_table[s, block_idx].item())
                slot_mapping[seq_start + t] = physical_block * self.block_size + block_offset

        metadata = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=prompt_len,
            query_start_loc=query_start_loc,
            max_seq_len=prompt_len,
            seq_lens=self.seq_lens.clone(),
            block_table=self.block_table,
            slot_mapping=slot_mapping,
        )

        # Per-layer slot mappings
        layer_slot_mappings: dict[str, torch.Tensor] = {}
        for i in range(self.num_layers):
            layer_slot_mappings[f"model.layers.{i}.self_attn"] = slot_mapping

        self._prefill_metadata = metadata
        return metadata, layer_slot_mappings

    def build_decode_metadata(
        self, step: int, prompt_len: int
    ) -> tuple[FlashAttentionMetadata, dict[str, torch.Tensor]]:
        """Build attention metadata for decode step ``step`` (0-indexed).

        Each decode step adds 1 token per sequence.
        """
        current_len = prompt_len + step
        self.seq_lens.fill_(current_len)

        # query_start_loc: [0, 1, 2, 3, ...] for decode
        query_start_loc = torch.arange(self.num_seqs + 1, dtype=torch.int32, device=self.device)

        num_tokens = self.num_seqs
        slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        for s in range(self.num_seqs):
            pos = current_len - 1
            block_idx = pos // self.block_size
            block_offset = pos % self.block_size
            physical_block = int(self.block_table[s, block_idx].item())
            slot_mapping[s] = physical_block * self.block_size + block_offset

        metadata = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=1,
            query_start_loc=query_start_loc,
            max_seq_len=current_len,
            seq_lens=self.seq_lens.clone(),
            block_table=self.block_table,
            slot_mapping=slot_mapping,
        )

        layer_slot_mappings: dict[str, torch.Tensor] = {}
        for i in range(self.num_layers):
            layer_slot_mappings[f"model.layers.{i}.self_attn"] = slot_mapping

        self._decode_metadata = metadata
        return metadata, layer_slot_mappings

    @contextmanager
    def forward_context(self, attn_metadata: FlashAttentionMetadata, layer_slot_mappings: dict[str, torch.Tensor]):
        """Context manager that sets up vLLM ForwardContext for attention layers."""
        # Build per-layer attention metadata dict
        attn_metadata_dict: dict[str, FlashAttentionMetadata] = {}
        for i in range(self.num_layers):
            attn_metadata_dict[f"model.layers.{i}.self_attn"] = attn_metadata

        # Build no_compile_layers dict (maps layer names to Attention instances)
        # This is populated by the attention layer replacement step
        no_compile_layers: dict[str, Any] = {}

        fc = ForwardContext(
            attn_metadata=attn_metadata_dict,
            no_compile_layers=no_compile_layers,
            slot_mapping=layer_slot_mappings,
            vllm_config=None,
        )

        token = set_forward_context(fc)
        try:
            with token:
                yield
        finally:
            pass

    def bind_kv_cache_to_layer(self, layer_idx: int, attn_module: nn.Module) -> None:
        """Bind the pre-allocated KV cache blocks to a vLLM Attention layer."""
        layer_kv = self.kv_cache[:, :, :, :, :]  # Full cache — the attention layer uses slot_mapping to index
        attn_module.kv_cache = layer_kv


def get_janus_kv_cache_config(block_size: int = 16) -> KVCacheConfig:
    """Create a minimal KV cache config for Janus pipeline.

    This mirrors vLLM's KVCacheConfig but is stripped down for standalone use.
    """
    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,  # Will be updated per layer
        head_size=1,  # Will be updated per layer
        dtype=torch.bfloat16,
    )
    # Return raw components for cache construction
    return kv_cache_spec
