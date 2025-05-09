# Testing vLLM

## Tiny-Llama-1.1B w/vLLM

```bash
# build it
podman build \
  --build-arg HF_TOKEN=hf_your_actual_token_here \
  --build-arg MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -t vllm-cuda-ubuntu-tinyllama .

# run it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8001:8000 \
  localhost/vllm-ubuntu-tinyllama \
  python3 -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype=half
    --gpu-memory-utilization 0.4

# test it
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": "Once upon a time,",
    "max_tokens": 50
  }'
```