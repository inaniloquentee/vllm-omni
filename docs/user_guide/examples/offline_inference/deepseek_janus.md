# DeepSeek Janus (text-to-image)

Source: [`examples/offline_inference/text_to_image`](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image), deploy YAMLs under [`vllm_omni/deploy/`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy), and pipeline code [`vllm_omni/diffusion/models/deepseek_janus/`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/diffusion/models/deepseek_janus).

DeepSeek Janus uses Hugging Face `trust_remote_code` (`MultiModalityCausalLM`). The Omni diffusion worker loads [`JanusPipeline`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/deepseek_janus/pipeline_janus.py) when `config.json` has `model_type: multi_modality` (see [`OmniDiffusionConfig.enrich_config`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/data.py)).

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

Use the OpenAI-compatible image API as in [Image Generation](../serving/image_generation_api.md).

## Two-stage (AR text → image generation)

Use [`vllm_omni/deploy/deepseek_janus_two_stage.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/deepseek_janus_two_stage.yaml) or [`vllm_omni/model_executor/stage_configs/deepseek_janus_two_stage.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_configs/deepseek_janus_two_stage.yaml). Stage bridging is [`ar2generation`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/deepseek_janus.py).

## Implementation pointers

| Component | Location |
|-----------|----------|
| Diffusion pipeline | `vllm_omni/diffusion/models/deepseek_janus/pipeline_janus.py` |
| Omni pipeline configs | `vllm_omni/model_executor/models/deepseek_janus/pipeline.py` |
| AR stage (Llama weights from `language_model.*`) | `vllm_omni/model_executor/models/deepseek_janus/deepseek_janus_ar.py` |
