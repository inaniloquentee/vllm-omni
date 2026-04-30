# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus: AR stage → JanusPipeline (diffusion) bridge.

Stage 0 outputs natural-language text (prompt refinement, instructions, or a
caption).  Stage 1 runs :class:`~vllm_omni.diffusion.models.deepseek_janus.pipeline_janus.JanusPipeline`
text-to-image generation using the merged prompt.

Pattern aligns with :mod:`vllm_omni.model_executor.stage_input_processors.hunyuan_image3`.
"""

from __future__ import annotations

from typing import Any

from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def ar2generation(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Build diffusion-stage request dicts from AR outputs."""
    del requires_multimodal_data
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid source stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    out: list[dict[str, Any]] = []

    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_row in enumerate(ar_outputs):
        ao = ar_row.outputs[0]
        ar_text = getattr(ao, "text", "") or ""

        original = prompt[i] if i < len(prompt) else {}
        if isinstance(original, dict):
            pass
        elif hasattr(original, "_asdict"):
            original = original._asdict()
        elif hasattr(original, "__dict__"):
            original = vars(original)
        else:
            original = {}

        base_prompt = original.get("prompt", "") or ""

        if ar_text.strip():
            merged = f"{base_prompt}\n{ar_text}".strip() if base_prompt else ar_text.strip()
        else:
            merged = base_prompt

        height = original.get("height", 384)
        width = original.get("width", 384)

        diffusion_input: dict[str, Any] = {
            "prompt": merged,
            "height": height,
            "width": width,
            "extra": {
                "ar_generated_text": ar_text,
                "base_prompt": base_prompt,
            },
        }

        mm_data = original.get("multi_modal_data")
        if mm_data:
            diffusion_input["multi_modal_data"] = mm_data

        for key in ("seed", "num_inference_steps", "guidance_scale"):
            if key in original:
                diffusion_input[key] = original[key]

        logger.debug(
            "[ar2generation] merged prompt len=%d (base=%d ar=%d)",
            len(merged),
            len(base_prompt),
            len(ar_text),
        )
        out.append(diffusion_input)

    return out
