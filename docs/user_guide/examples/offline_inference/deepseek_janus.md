# DeepSeek Janus (text-to-image)

Source: [`examples/offline_inference/text_to_image`](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image), deploy YAMLs under [`vllm_omni/deploy/`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy), and pipeline code [`vllm_omni/diffusion/models/deepseek_janus/`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/diffusion/models/deepseek_janus).

DeepSeek Janus uses Hugging Face `trust_remote_code` (`MultiModalityCausalLM`). The Omni diffusion worker loads [`JanusPipeline`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/deepseek_janus/pipeline_janus.py) when `config.json` has `model_type: multi_modality` (see [`OmniDiffusionConfig.enrich_config`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/data.py)).

## Config schema

Janus follows the post-#2383 split config schema:

- Topology lives in [`vllm_omni/model_executor/models/deepseek_janus/pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/deepseek_janus/pipeline.py).
- Deployment defaults live in [`vllm_omni/deploy/deepseek_janus_single_stage.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/deepseek_janus_single_stage.yaml) and [`vllm_omni/deploy/deepseek_janus_two_stage.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/deepseek_janus_two_stage.yaml).

The split matters because Janus exposes two valid Omni topologies:

- `deepseek_janus_single_stage`: one diffusion stage runs the full Janus image-generation stack.
- `deepseek_janus_two_stage`: stage 0 runs the AR language backbone and stage 1 runs `JanusPipeline` for image generation.

Because the Hugging Face checkpoint alone cannot disambiguate which topology you want, pass the matching `--deploy-config` explicitly for Janus examples.

## Offline (generic text-to-image script)

```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
  --model deepseek-ai/Janus-1.3B \
  --deploy-config vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --prompt "A scenic mountain lake at sunset" \
  --output janus_out.png \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --tensor-parallel-size 1 \
  --height 384 \
  --width 384 \
  --enforce-eager
```

`--deploy-config` pins the single-stage topology (`pipeline: deepseek_janus_single_stage`). If omitted, routing still resolves `JanusPipeline` from the checkpoint when possible.

## Online serving

```bash
vllm serve deepseek-ai/Janus-1.3B --omni \
  --deploy-config vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --tensor-parallel-size 1
```

Use the OpenAI-compatible image API as in [Image Generation](../../../serving/image_generation_api.md).

## Two-stage (AR text → image generation)

Use [`vllm_omni/deploy/deepseek_janus_two_stage.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/deepseek_janus_two_stage.yaml). Stage bridging is [`ar2generation`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/deepseek_janus.py).

## Text rendering prompts

Reviewer-requested text-on-image prompts that work well for Janus are usually short, high-contrast, and placed on a single obvious surface. These are the prompts used in the PR validation runs:

```text
A clean white poster on a grass field that clearly reads "vLLM-Omni" in large colorful letters, with a brown bear holding a paintbrush beside it
```

```text
A street cafe chalkboard sign that says "HELLO JANUS" in large white block letters, realistic lighting, centered composition
```

```text
A bakery display card with the words "OPEN SOURCE" written in bold icing-style letters, close-up product photo
```

## Why supported_models has two Janus rows

Janus appears twice in [`supported_models.md`](../../../models/supported_models.md) because the two-stage topology uses two different execution targets:

- `JanusPipeline` is the diffusion/image-generation worker.
- `MultiModalityCausalLM` is the AR text stage that consumes only the Llama-style `language_model.*` subtree before handing the prompt to the diffusion stage.

## Implementation pointers

| Component | Location |
|-----------|----------|
| Diffusion pipeline | `vllm_omni/diffusion/models/deepseek_janus/pipeline_janus.py` |
| Omni pipeline configs | `vllm_omni/model_executor/models/deepseek_janus/pipeline.py` |
| AR stage (Llama weights from `language_model.*`) | `vllm_omni/model_executor/models/deepseek_janus/deepseek_janus_ar.py` |
