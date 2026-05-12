# DeepSeek Janus (text-to-image)

Source: [`examples/offline_inference/text_to_image`](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image), deploy YAMLs under [`vllm_omni/deploy/`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy), and pipeline code [`vllm_omni/diffusion/models/deepseek_janus/`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/diffusion/models/deepseek_janus).

DeepSeek Janus uses Hugging Face `trust_remote_code` (`MultiModalityCausalLM`). The Omni diffusion worker loads [`JanusPipeline`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/deepseek_janus/pipeline_janus.py) when `config.json` has `model_type: multi_modality` (see [`OmniDiffusionConfig.enrich_config`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/data.py)).

## Topologies

Janus exposes two valid Omni topologies:

- `deepseek_janus_single_stage`: one diffusion stage runs the full Janus image-generation stack (AR loop + VQ decode).
- `deepseek_janus_two_stage`: stage 0 runs the AR image-token generation via vLLM's GPU model runner (with full PagedAttention, CUDAGraphWrapper, etc.); stage 1 runs VQ decode only.

Because the Hugging Face checkpoint alone cannot disambiguate which topology you want, pass the matching `--deploy-config` explicitly for Janus examples.

## Offline (generic text-to-image script)

```bash
# Set TORCHDYNAMO_DISABLE=1 if using PyTorch 2.10 + vLLM 0.19.0
export TORCHDYNAMO_DISABLE=1

python3 examples/offline_inference/text_to_image/text_to_image.py \
  --model /root/autodl-tmp/models/Janus-1.3B \
  --deploy-config vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --prompt "A scenic mountain lake at sunset" \
  --output janus_out.png \
  --guidance-scale 5.0 \
  --tensor-parallel-size 1 \
  --height 384 \
  --width 384
```

For Janus-Pro-7B with optimisations:
```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
  --model /root/autodl-tmp/Janus-Pro-7B \
  --deploy-config vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --prompt "A scenic mountain lake at sunset, photorealistic" \
  --output janus_7b.png \
  --guidance-scale 5.0
```

## Online serving

```bash
vllm serve /root/autodl-tmp/Janus-Pro-7B --omni \
  --deploy-config vllm_omni/deploy/deepseek_janus_single_stage.yaml \
  --port 8091 \
  --tensor-parallel-size 1
```

Use the OpenAI-compatible image API:

```python
import json, urllib.request

url = "http://127.0.0.1:8091/v1/images/generations"
payload = {"prompt": "a photo of a bench", "n": 1, "size": "384x384"}
req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    obj = json.loads(resp.read())
```

## Two-stage (AR image token generation → VQ decode)

The two-stage topology routes the AR loop through vLLM's GPU model runner, getting:
- PagedAttention (block-based KV cache)
- CUDAGraphWrapper (automatic CUDA graph capture/replay)
- Chunked prefill, prefix caching, FP8 KV cache
- FlashInfer/FA3 attention backends
- Tensor parallelism, continuous batching

Use [`vllm_omni/deploy/deepseek_janus_two_stage.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/deepseek_janus_two_stage.yaml). Stage bridging is [`ar_tokens_to_vq`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/deepseek_janus.py) (image tokens → VQ) or [`ar2generation`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/deepseek_janus.py) (text → full Janus pipeline).

## Performance (Janus-Pro-7B, RTX 5090)

| Config | Latency | Memory | Throughput |
|--------|---------|--------|------------|
| bs=1 | 6.48s | 14.75 GB | 0.154 im/s |
| bs=2 | 6.82s | 15.63 GB | 0.293 im/s |
| bs=4 | 7.77s | 17.28 GB | 0.515 im/s |

vs Official Janus (HF): vLLM-Omni is **25% faster** (6.48s vs 8.66s) with **3.3% less memory**.

## Text rendering prompts

Reviewer-requested text-on-image prompts that work well for Janus:

```text
A clean white poster on a grass field that clearly reads "vLLM-Omni" in large colorful letters, with a brown bear holding a paintbrush beside it
```

```text
A street cafe chalkboard sign that says "HELLO JANUS" in large white block letters, realistic lighting, centered composition
```

```text
A bakery display card with the words "OPEN SOURCE" written in bold icing-style letters, close-up product photo
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--guidance-scale` | CFG weight | 5.0 |
| `--height` / `--width` | Output resolution (Janus fixed 384x384) | 384 |
| `--enforce-eager` | Disable torch.compile + CUDA graph | False |
| `--tensor-parallel-size` | Number of GPUs | 1 |
| `--num-images-per-prompt` | Parallel images per generation | 1 |

> Note: Janus uses a fixed 576-token AR loop (24x24 VQ latent grid). `--num-inference-steps` has no effect.

## Implementation pointers

| Component | Location |
|-----------|----------|
| Diffusion pipeline (single-stage) | `vllm_omni/diffusion/models/deepseek_janus/pipeline_janus.py` |
| VQ-decode pipeline (two-stage) | `vllm_omni/diffusion/models/deepseek_janus/pipeline_janus_vq.py` |
| PagedAttention context | `vllm_omni/diffusion/models/deepseek_janus/_paged_attn_context.py` |
| Omni pipeline configs | `vllm_omni/model_executor/models/deepseek_janus/pipeline.py` |
| AR models (text + image tokens) | `vllm_omni/model_executor/models/deepseek_janus/deepseek_janus_ar.py` |
| Stage bridges | `vllm_omni/model_executor/stage_input_processors/deepseek_janus.py` |
| Deploy YAMLs | `vllm_omni/deploy/deepseek_janus_{single,two}_stage.yaml` |
