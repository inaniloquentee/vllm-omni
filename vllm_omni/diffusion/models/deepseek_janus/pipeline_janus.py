# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus image generation for vLLM-Omni.

Follows the integration shape of ``BagelPipeline`` (single-stage diffusion worker)
and ``OmniDiffusionConfig`` routing used by Hunyuan Image3.

Upstream reference:
  https://github.com/deepseek-ai/Janus — ``generation_inference.py`` autoregressive
  image token loop + ``gen_vision_model.decode_code``.

This pipeline loads ``MultiModalityCausalLM`` via Hugging Face ``trust_remote_code``
and runs text-to-image generation. VL understanding (chat + ``prepare_inputs_embeds``)
can be added later as a separate AR stage, mirroring ``hunyuan_image3_it2i.yaml``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, LlamaTokenizerFast
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsModuleOffload
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def _register_janus_hf_classes() -> None:
    """Register ``multi_modality`` with Transformers.

    The public ``deepseek-ai/Janus-1.3B`` repo ships weights + JSON only (no ``modeling_*.py``),
    so ``trust_remote_code`` cannot load the architecture. We vendor DeepSeek's registration
    from https://github.com/deepseek-ai/Janus (Apache-2.0 / MIT licensed code).
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "multi_modality" in CONFIG_MAPPING:
        return
    from vllm_omni.diffusion.models.deepseek_janus._janus_hf_vendor import modeling_vlm  # noqa: F401


def _build_janus_vl_chat_processor(model: str | Path, revision: str | None = None) -> Any:
    """Construct ``VLChatProcessor`` from checkpoint JSON (no processor Python on the Hub)."""
    from vllm_omni.diffusion.models.deepseek_janus._janus_hf_vendor.image_processing_vlm import (
        VLMImageProcessor,
    )
    from vllm_omni.diffusion.models.deepseek_janus._janus_hf_vendor.processing_vlm import VLChatProcessor

    root = Path(model)
    with open(root / "preprocessor_config.json") as f:
        pre = json.load(f)
    with open(root / "processor_config.json") as f:
        proc = json.load(f)
    pre_keys_drop = {"processor_class", "image_processor_type"}
    pre_args = {k: v for k, v in pre.items() if k not in pre_keys_drop}
    image_processor = VLMImageProcessor(**pre_args)
    tok_kw: dict[str, Any] = {}
    if revision:
        tok_kw["revision"] = revision
    tokenizer = LlamaTokenizerFast.from_pretrained(str(root), **tok_kw)
    return VLChatProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_tag=str(proc.get("image_tag", "<image_placeholder>")),
        num_image_tokens=int(proc.get("num_image_tokens", 576)),
        add_special_token=bool(proc.get("add_special_token", False)),
        sft_format=str(proc.get("sft_format", "deepseek")),
        mask_prompt=bool(proc.get("mask_prompt", True)),
        ignore_id=int(proc.get("ignore_id", -100)),
    )


def get_janus_post_process_func(od_config: OmniDiffusionConfig):
    """Janus returns PIL images directly from ``forward()``."""

    def post_process_func(x: Any) -> Any:
        return x

    return post_process_func


class JanusPipeline(nn.Module, SupportsModuleOffload, DiffusionPipelineProfilerMixin):
    """HF remote-code Janus model packaged for the Omni diffusion engine."""

    _dit_modules: ClassVar[list[str]] = ["mm_model.language_model.model"]
    _encoder_modules: ClassVar[list[str]] = ["mm_model.vision_model", "mm_model.aligner"]
    _vae_modules: ClassVar[list[str]] = ["mm_model.gen_vision_model"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        remote_kw: dict[str, Any] = {"trust_remote_code": True}
        if getattr(od_config, "revision", None):
            remote_kw["revision"] = od_config.revision

        dtype = getattr(od_config, "dtype", None) or torch.bfloat16
        _register_janus_hf_classes()
        cfg_kw: dict[str, Any] = {}
        if getattr(od_config, "revision", None):
            cfg_kw["revision"] = od_config.revision
        cfg = AutoConfig.from_pretrained(od_config.model, **cfg_kw)
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            if getattr(cfg.language_config, "_attn_implementation", None) == "flash_attention_2":
                try:
                    cfg.language_config._attn_implementation = "sdpa"  # type: ignore[attr-defined]
                except (TypeError, AttributeError):
                    object.__setattr__(cfg.language_config, "_attn_implementation", "sdpa")
        self.mm_model = AutoModelForCausalLM.from_pretrained(
            od_config.model,
            config=cfg,
            torch_dtype=dtype,
            **remote_kw,
        )
        rev = getattr(od_config, "revision", None)
        try:
            self.processor = _build_janus_vl_chat_processor(od_config.model, rev)
        except Exception as e:
            logger.warning("Built-in Janus VLChatProcessor failed (%s); trying AutoProcessor.", e)
            try:
                self.processor = AutoProcessor.from_pretrained(od_config.model, **remote_kw)
            except Exception as e2:
                logger.warning(
                    "AutoProcessor.from_pretrained failed for Janus (%s). "
                    "Chat / multimodal prompts may require full HF repo files.",
                    e2,
                )
                self.processor = None

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=getattr(od_config, "revision", None),
                prefix="",
                fall_back_to_pt=True,
            )
        ]

        if getattr(od_config, "quantization_config", None) is None and not getattr(
            od_config, "enable_layerwise_offload", False
        ):
            self.mm_model.to(self.device)

        # Torch-compile hooks target ``transformer`` on several pipelines.
        self.transformer = self.mm_model.language_model.model

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Text-to-image using Janus AR image-token prediction + VQ decode."""
        if self.processor is None:
            return DiffusionOutput(
                error="Janus processor failed to load; cannot build prompts.",
                aborted=True,
            )

        device = next(self.mm_model.parameters()).device
        dtype = next(self.mm_model.parameters()).dtype

        sp = req.sampling_params
        extra = sp.extra_step_kwargs or {}
        parallel_size = max(1, int(sp.num_outputs_per_prompt))
        cfg_weight = (
            float(sp.guidance_scale)
            if getattr(sp, "guidance_scale_provided", False)
            else float(extra.get("cfg_weight", 5.0))
        )
        temperature = float(extra.get("temperature", 1.0))
        image_token_num = int(extra.get("image_token_num_per_image", 576))
        img_size = int(extra.get("img_size", 384))
        patch_size = int(extra.get("patch_size", 16))

        images_out: list[Image.Image] = []

        for prompt in req.prompts:
            text = prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")
            conversation = [
                {"role": "User", "content": text},
                {"role": "Assistant", "content": ""},
            ]
            sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            full_prompt = sft_format + self.processor.image_start_tag

            token_ids = self.processor.tokenizer.encode(full_prompt)
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)

            pair_rows = parallel_size * 2
            tokens = torch.zeros((pair_rows, input_ids.shape[0]), dtype=torch.long, device=device)
            for i in range(pair_rows):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.processor.pad_id

            inputs_embeds = self.mm_model.language_model.get_input_embeddings()(tokens).to(dtype=dtype)

            generated = torch.zeros((parallel_size, image_token_num), dtype=torch.long, device=device)

            past_kv = None
            for step_i in range(image_token_num):
                lm_out = self.mm_model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_kv,
                    return_dict=True,
                )
                past_kv = lm_out.past_key_values
                hidden = lm_out.last_hidden_state[:, -1, :]

                logits = self.mm_model.gen_head(hidden)
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits_merged = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits_merged / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated[:, step_i] = next_token.squeeze(-1)

                stacked = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).reshape(-1)
                img_embeds = self.mm_model.prepare_gen_img_embeds(stacked)
                inputs_embeds = img_embeds.unsqueeze(1).to(dtype=dtype)

            dec = self.mm_model.gen_vision_model.decode_code(
                generated.to(dtype=torch.int),
                shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
            )
            dec_np = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec_np = np.clip((dec_np + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            for bi in range(parallel_size):
                images_out.append(Image.fromarray(dec_np[bi]))

        primary = images_out[0] if images_out else None
        return DiffusionOutput(
            output=primary,
            trajectory_decoded=None,
            custom_output={"images": images_out, "num_images": len(images_out)},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self.mm_model)
        inner_loaded = loader.load_weights(weights)
        # ``named_parameters()`` on ``JanusPipeline`` uses the ``mm_model.`` prefix; the
        # nested ``AutoWeightsLoader`` reports keys relative to ``mm_model``.
        return {f"mm_model.{name}" for name in inner_loaded}
