# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus — AR stage (Llama backbone only).

The HF checkpoint nests the causal LM under ``language_model`` inside
``MultiModalityCausalLM``.  This class subclasses vLLM's :class:`LlamaForCausalLM`
and loads only those weights, mirroring how other omni stacks peel an LLM out of
a larger multimodal checkpoint.

Text-only AR works out of the box.  Image-conditioned understanding requires
feeding ``inputs_embeds`` from ``prepare_inputs_embeds`` (future multimodal
processor), analogous to Hunyuan Image3's AR wrapping.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM

logger = init_logger(__name__)


class OmniDeepSeekJanusForConditionalGeneration(LlamaForCausalLM):
    """LLaMA decoder (+ LM head) for the Janus ``language_model`` subtree."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[torch.nn.Module] = LlamaDecoderLayer,
    ) -> None:
        mm_cfg = vllm_config.model_config.hf_config
        lang = getattr(mm_cfg, "language_config", None)
        if lang is None:
            raise ValueError("DeepSeek Janus expects a MultiModality-style HF config with ``language_config`` (Llama).")
        if isinstance(lang, LlamaConfig):
            lang_cfg = lang
        elif hasattr(lang, "to_dict"):
            lang_cfg = LlamaConfig(**lang.to_dict())
        else:
            lang_cfg = LlamaConfig(**dict(lang))

        orig_hf = vllm_config.model_config.hf_config
        object.__setattr__(vllm_config.model_config, "hf_config", lang_cfg)
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        finally:
            object.__setattr__(vllm_config.model_config, "hf_config", orig_hf)

        self.janus_multi_modality_hf_config = mm_cfg

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Checkpoint layout: language_model.model.* / language_model.lm_head.*
        def _language_rows(it: Iterable[tuple[str, torch.Tensor]]):
            for name, tensor in it:
                if name.startswith("language_model."):
                    yield name[len("language_model.") :], tensor

        return super().load_weights(_language_rows(weights))
