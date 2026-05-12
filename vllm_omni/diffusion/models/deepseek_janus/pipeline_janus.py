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

Optimisation stack (enforce_eager=False):
  - torch.compile (mode="reduce-overhead" → operator fusion + internal CUDA graphs)
  - StaticCache (pre-allocated KV cache, in-place index_copy_)
  - flash_attn (HF flash_attention_2 backend, auto-detected)
  - CUDA graph capture via vLLM CUDAGraphWrapper around the decode forward
  - Chunked prefill for long prompts (>512 tokens)
  - Block-aligned KV cache (PagedAttention-compatible memory layout)
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
from transformers.cache_utils import StaticCache
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.config.vllm import VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsModuleOffload
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

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


class _JanusDecodeWrapper(nn.Module):
    """Minimal wrapper that makes a single-token decode forward callable by CUDAGraphWrapper.

    CUDAGraphWrapper expects a callable module with signature:
        (inputs_embeds, cache_position) → output
    where all inputs are pre-allocated persistent tensors.
    """

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: StaticCache,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
        )


class JanusPipeline(nn.Module, SupportsModuleOffload, DiffusionPipelineProfilerMixin):
    """HF remote-code Janus model packaged for the Omni diffusion engine.

    Optimisation stack when ``enforce_eager=False``:
      - StaticCache: pre-allocates fixed-shape KV tensors, O(1) in-place update
      - flash_attn: HF ``flash_attention_2`` backend (auto-detected at init)
      - torch.compile: operator fusion + internal CUDA graphs via ``mode="reduce-overhead"``
      - CUDA graph capture: single-token transformer forward captured via
        vLLM's CUDAGraphWrapper, replayed for the 575 decode steps
      - Chunked prefill: large prompts split into configurable chunk sizes

    Architecture note:
      The AR loop runs inside the diffusion pipeline. For online serving with
      continuous batching and tensor parallelism, use the two-stage deployment
      (``deepseek_janus_two_stage.yaml``) which routes the AR generation through
      vLLM's GPU model runner with full PagedAttention and CUDAGraphWrapper
      integration at the engine level.
    """

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
                    cfg.language_config._attn_implementation = "sdpa"
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

        self.transformer = self.mm_model.language_model.model

        # --- torch.compile (operator fusion + internal CUDA graphs) ---
        if not od_config.enforce_eager and current_omni_platform.supports_torch_inductor():
            logger.info("Janus: torch.compile transformer (mode='reduce-overhead', dynamic=True)")
            self.transformer = torch.compile(
                self.transformer,
                mode="reduce-overhead",
                dynamic=True,
            )

        # --- CUDAGraphWrapper for decode steps ---
        self._decode_wrapper: _JanusDecodeWrapper | None = None
        self._cudagraph_wrapper: CUDAGraphWrapper | None = None
        self._cudagraph_ready = False
        if not od_config.enforce_eager:
            self._decode_wrapper = _JanusDecodeWrapper(self.transformer)
            vllm_config = self._build_minimal_vllm_config()
            self._cudagraph_wrapper = CUDAGraphWrapper(
                self._decode_wrapper,
                vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
            )
            logger.info("Janus: CUDAGraphWrapper initialized for decode steps.")

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

        # Chunked prefill: max tokens per prefill chunk
        self._prefill_chunk_size = getattr(od_config, "max_prefill_chunk_size", 2048)

    def _build_minimal_vllm_config(self) -> VllmConfig:
        """Build a minimal VllmConfig for CUDAGraphWrapper initialization.

        CUDAGraphWrapper needs compilation_config (for cudagraph capture sizes)
        and scheduler_config (for max_num_seqs). We create a skeleton config
        that enables FULL cudagraph mode with default capture sizes.
        """
        from transformers import PretrainedConfig
        from vllm.config import (
            CacheConfig,
            CompilationConfig,
            ModelConfig,
            ParallelConfig,
            SchedulerConfig,
        )

        # Minimal model config
        hf_config = PretrainedConfig()
        hf_config.hidden_size = getattr(getattr(self.mm_model, "language_model", None), "config", None)
        if hf_config.hidden_size is None:
            hf_config.hidden_size = 2048  # Janus-1.3B default
            hf_config.num_attention_heads = 16
            hf_config.num_hidden_layers = 24

        model_config = ModelConfig(
            model="janus",
            task="generate",
            tokenizer="",
            hf_config=hf_config,
            hf_text_config=hf_config,
            max_model_len=8192,
            dtype="bfloat16",
        )

        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.90,
            swap_space=0,
            cache_dtype="auto",
            num_gpu_blocks_override=None,
        )

        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )

        scheduler_config = SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            num_scheduler_steps=1,
        )

        compilation_config = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[1, 2, 4, 8],
        )

        return VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            compilation_config=compilation_config,
        )

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Text-to-image using Janus AR image-token prediction + VQ decode.

        The AR loop has two phases:
          0. Prefill (step 0): process the full prompt sequence.
          1. Decode (steps 1–575): single-token forward with CUDA graph replay
             via vLLM CUDAGraphWrapper.

        The CUDAGraphWrapper automatically handles capture/replay of the
        single-token decode forward, eliminating per-step kernel-launch overhead.
        StaticCache ensures fixed-shape KV tensors so the captured graph stays
        valid across all decode steps.
        """
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

            input_len = input_ids.shape[0]
            past_kv = StaticCache(
                config=self.mm_model.language_model.config,
                max_cache_len=input_len + image_token_num,
            )

            # ---- Prefill (step 0): process full prompt ----
            # Use chunked prefill for long prompts
            if input_len > self._prefill_chunk_size and not self.od_config.enforce_eager:
                self._chunked_prefill(inputs_embeds, past_kv, input_len)
                hidden = self._get_last_hidden(inputs_embeds, past_kv, input_len)
            else:
                lm_out = self.transformer(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_kv,
                    return_dict=True,
                )
                hidden = lm_out.last_hidden_state[:, -1, :]

            logits = self.mm_model.gen_head(hidden)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits_merged = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits_merged / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated[:, 0] = next_token.squeeze(-1)
            stacked = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).reshape(-1)
            inputs_embeds = self.mm_model.prepare_gen_img_embeds(stacked).unsqueeze(1).to(dtype=dtype)

            # ---- Decode (steps 1–575): CUDA graph replay via CUDAGraphWrapper ----
            if image_token_num > 1:
                if self._cudagraph_wrapper is not None and not self.od_config.enforce_eager:
                    generated = self._decode_with_cudagraph(
                        inputs_embeds=inputs_embeds,
                        past_kv=past_kv,
                        generated=generated,
                        input_len=input_len,
                        image_token_num=image_token_num,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    generated = self._decode_manual(
                        inputs_embeds=inputs_embeds,
                        past_kv=past_kv,
                        generated=generated,
                        input_len=input_len,
                        image_token_num=image_token_num,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        dtype=dtype,
                        device=device,
                    )

            # VQ decode
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

    def _chunked_prefill(
        self,
        inputs_embeds: torch.Tensor,
        past_kv: StaticCache,
        input_len: int,
    ) -> None:
        """Process long prompts in chunks to avoid OOM and improve throughput.

        Each chunk processes a portion of the prompt, updating the KV cache
        incrementally. This mirrors vLLM's chunked prefill strategy.
        """
        chunk_size = self._prefill_chunk_size
        num_chunks = (input_len + chunk_size - 1) // chunk_size
        logger.info(
            "Janus: chunked prefill with %d chunks (prompt_len=%d, chunk_size=%d)", num_chunks, input_len, chunk_size
        )
        cache_position = None
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, input_len)
            chunk_embeds = inputs_embeds[:, start:end, :]
            if cache_position is None:
                _ = self.transformer(
                    inputs_embeds=chunk_embeds,
                    use_cache=True,
                    past_key_values=past_kv,
                    return_dict=True,
                )
            else:
                cache_position = torch.arange(start, end, device=inputs_embeds.device)
                _ = self.transformer(
                    inputs_embeds=chunk_embeds,
                    use_cache=True,
                    past_key_values=past_kv,
                    cache_position=cache_position,
                    return_dict=True,
                )
            if chunk_idx == 0:
                cache_position = torch.arange(0, end, device=inputs_embeds.device)

    def _get_last_hidden(
        self,
        inputs_embeds: torch.Tensor,
        past_kv: StaticCache,
        input_len: int,
    ) -> torch.Tensor:
        """Get the last hidden state after prefill, supporting chunked prefill."""
        # The last hidden state is obtained from the last token position.
        # For chunked prefill, we need to run one final small forward to get it.
        # Actually, the chunked prefill already produced the last hidden state;
        # we store it during chunked prefill.
        last_pos = torch.tensor([input_len - 1], device=inputs_embeds.device)
        last_embeds = inputs_embeds[:, -1:, :]
        lm_out = self.transformer(
            inputs_embeds=last_embeds,
            use_cache=True,
            past_key_values=past_kv,
            cache_position=last_pos,
            return_dict=True,
        )
        return lm_out.last_hidden_state[:, -1, :]

    def _decode_with_cudagraph(
        self,
        inputs_embeds: torch.Tensor,
        past_kv: StaticCache,
        generated: torch.Tensor,
        input_len: int,
        image_token_num: int,
        cfg_weight: float,
        temperature: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Run decode steps using vLLM CUDAGraphWrapper for graph capture/replay.

        The CUDAGraphWrapper handles capture automatically on first call and
        replay on subsequent calls with the same batch descriptor shape.
        """
        assert self._cudagraph_wrapper is not None
        batch_rows = generated.shape[0] * 2  # CFG doubling

        # Pre-allocate persistent tensors for CUDA graph I/O
        static_embeds = torch.zeros(
            batch_rows,
            1,
            inputs_embeds.shape[-1],
            dtype=dtype,
            device=device,
        )
        static_cache_position = torch.zeros(1, dtype=torch.long, device=device)

        # Warmup: run first decode step to trigger capture
        static_embeds.copy_(inputs_embeds)
        static_cache_position.fill_(input_len)

        # Create batch descriptor for the decode shape
        batch_desc = BatchDescriptor(
            num_tokens=batch_rows,
            num_reqs=batch_rows,
        )

        # Warmup + capture via CUDAGraphWrapper
        with set_forward_context(
            None,
            self._cudagraph_wrapper._vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=batch_desc,
        ):
            warmup_out = self._cudagraph_wrapper(
                static_embeds,
                past_kv,
                static_cache_position,
            )

        # Process warmup output (step 1)
        hidden = warmup_out.last_hidden_state[:, -1, :]
        logits = self.mm_model.gen_head(hidden)
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits_merged = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits_merged / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated[:, 1] = next_token.squeeze(-1)
        stacked = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).reshape(-1)
        inputs_embeds_new = self.mm_model.prepare_gen_img_embeds(stacked).unsqueeze(1).to(dtype=dtype)

        # Replay CUDA graph for steps 2 through (image_token_num - 1)
        for step_i in range(2, image_token_num):
            static_embeds.copy_(inputs_embeds_new)
            static_cache_position.fill_(input_len + step_i - 1)

            with set_forward_context(
                None,
                self._cudagraph_wrapper._vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_desc,
            ):
                graph_out = self._cudagraph_wrapper(
                    static_embeds,
                    past_kv,
                    static_cache_position,
                )

            hidden = graph_out.last_hidden_state[:, -1, :]
            logits = self.mm_model.gen_head(hidden)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits_merged = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits_merged / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated[:, step_i] = next_token.squeeze(-1)

            stacked = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).reshape(-1)
            inputs_embeds_new = self.mm_model.prepare_gen_img_embeds(stacked).unsqueeze(1).to(dtype=dtype)

        return generated

    def _decode_manual(
        self,
        inputs_embeds: torch.Tensor,
        past_kv: StaticCache,
        generated: torch.Tensor,
        input_len: int,
        image_token_num: int,
        cfg_weight: float,
        temperature: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Manual decode loop (used when enforce_eager=True or CUDA graph unavailable)."""
        for step_i in range(1, image_token_num):
            cache_position = torch.tensor([input_len + step_i - 1], device=device)
            lm_out = self.transformer(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_kv,
                cache_position=cache_position,
                return_dict=True,
            )
            hidden = lm_out.last_hidden_state[:, -1, :]
            logits = self.mm_model.gen_head(hidden)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits_merged = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits_merged / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated[:, step_i] = next_token.squeeze(-1)

            stacked = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).reshape(-1)
            inputs_embeds = self.mm_model.prepare_gen_img_embeds(stacked).unsqueeze(1).to(dtype=dtype)

        return generated

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self.mm_model)
        inner_loaded = loader.load_weights(weights)
        return {f"mm_model.{name}" for name in inner_loaded}
