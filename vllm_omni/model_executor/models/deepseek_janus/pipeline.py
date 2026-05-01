# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus omni pipeline topology (frozen).

Single-stage (image generation only):
  Stage 0: Diffusion worker runs :class:`~vllm_omni.diffusion.models.deepseek_janus.pipeline_janus.JanusPipeline`
  (HF ``MultiModalityCausalLM`` remote code) — analogous to ``bagel_single_stage``.

Two-stage (text AR → image generation):
  Stage 0: Llama backbone via :class:`OmniDeepSeekJanusForConditionalGeneration` (weights under ``language_model.*``).
  Stage 1: Same HF checkpoint, ``JanusPipeline`` consumes merged prompts from ``ar2generation``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.deepseek_janus"

DEEPSEEK_JANUS_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="deepseek_janus_single_stage",
    model_arch="MultiModalityCausalLM",
    hf_architectures=("MultiModalityCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
        ),
    ),
)

DEEPSEEK_JANUS_TWO_STAGE_PIPELINE = PipelineConfig(
    model_type="deepseek_janus_two_stage",
    model_arch="OmniDeepSeekJanusForConditionalGeneration",
    hf_architectures=("MultiModalityCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=False,
            model_arch="OmniDeepSeekJanusForConditionalGeneration",
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            model_arch="JanusPipeline",
            custom_process_input_func=f"{_PROC}.ar2generation",
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)
