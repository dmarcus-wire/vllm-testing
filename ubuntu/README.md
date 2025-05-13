# Experimenting vLLM

Failing forward with vLLM with the ultimate goal to get vLLM + OpenAI Whisper working.

Minimal Prereqs [provisioned with this procedure](https://github.com/redhat-na-ssa/whitepaper-stt-evaluation-on-kubernetes/blob/main/crawl/RHEL_GPU.md):

1. RHEL VM w/podman
1. GPUs - know your GPU and what models can/can't fit
1. HugginFace access token + CLI python3 -m pip install huggingface-hub

## Search vllm container in a public repo

```bash
# search for vllm
podman search vllm

# pull image
podman pull docker.io/vllm/vllm-openai

# check the baseos
skopeo inspect docker://docker.io/vllm/vllm-openai | jq '.Labels'/vllm-openai | jq '.Labels'
{
  "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
  "org.opencontainers.image.ref.name": "ubuntu",
  "org.opencontainers.image.version": "22.04"
```

## Review supported vLLM models [here](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models)

1. top-ranked models from the Hugging Face LLM leaderboard that
2. are supported by vLLM
3. fit under ~14.5 GiB (so they can run on a Tesla T4 GPU without crashing)

| **Model** | **Approx. Parameters** | **Main Task(s)**| **Why It Fits on T4** |
| ---------------------------------- | ---------------------- | --------------------------------------- | -------------------------------------------------- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B                   | Chat, generation                        | Tiny, easily fits with full KV caches              |
| Phi-2                              | 2.7B                   | General reasoning, math, code           | Small, \~5–6 GiB in FP16                           |
| Phi-3 Mini (Hugging Face release)  | 3.8B                   | General-purpose generation, chat        | Efficient, \~7–8 GiB in FP16                       |
| Qwen2.5-0.5B                       | 0.5B                   | Chat, generation, general LLM tasks     | Lightweight, fits with margin                      |
| Qwen2.5-1.8B                       | 1.8B                   | Chat, reasoning, generation             | Small size, under \~10 GiB with half precision     |
| LLaMA-2-7B                         | 7B                     | Chat, generation, instruction following | Tight but works with --dtype=half, low batch sizes |
| LLaMA-3-8B                         | 8B                     | Chat, general-purpose generation        | Fits with reduced cache or low concurrency         |
| Mistral-7B-Instruct                | 7B                     | Instruction-following, chat             | Runs on T4 with half precision + tuning            |
| Mixtral-8x7B (2 experts active)    | 12B active (2x7B)      | Mixture-of-experts: chat, reasoning     | Activates only 2 experts; memory footprint \~7B    |

## Generation

Explanation of Tasks:

1. hat → Conversational agents, dialogue systems
1. Instruction → Follows user prompts, task-specific
1. Summarization → Text summarization, document compression
1. Reasoning → Multi-step logic, Q&A, problem-solving
1. Multilingual → Supports multiple languages

```bash
# terminal 1
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype=half \
  --gpu-memory-utilization 0.7
```

```
# success
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

```bash
# terminal 2
# prompt the model Hello, TinyLlama! How are you today?
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
          {"role": "user", "content": "Hello, TinyLlama! How are you today?"}
        ]
      }'

# response sample
{"id":"chatcmpl-4dc4330a45de44f7b1f9fd651754346e","object":"chat.completion","created":1746809788,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"I don't have the capability to have a voice or a personal schedule like humans do; however, I can say that I am doing my best to help you today. If you have any questions or concerns, please don't hesitate to reach out to me through this chat app or any other preferred communication method. I hope you enjoy the chat with me!","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":29,"total_tokens":105,"completion_tokens":76,"prompt_tokens_details":null},"prompt_logprobs":null}
```

## Audio

Explanation of Tasks:

1. Transcription → Convert spoken audio to text in the same language
1. Translation → Convert non-English speech to English text
1. Language Detection → Auto-detect spoken language (used implicitly)
1. Timestamps → Include word or phrase-level time codes
1. (Future) Diarization → Separate and label different speakers

### Connected Environments

```bash
# Create the directory before running the container
mkdir -p ~/.cache/huggingface

# Set your HuggingFace token
export HF_TOKEN="<your_huggingface_token>"

# Build the Dockerfile with the same version
podman build -t vllm-latest -f follow-docs/Dockerfile.latest follow-docs/

# Test it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  --env HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm-latest \
  --model openai/whisper-tiny.en \
  --task transcription \
  --dtype=half

# Transcribe
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F file=@sample/harvard.wav \
  -F model=openai/whisper-tiny.en

# expected output
# {"text":" The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."}
```

### Disconnected Environments

Since we're loading a local model from /models/whisper-tiny.en, you must manually set its served name like so:

1. `--model` /models/whisper-tiny.en \
1. `--served-model-name` openai/whisper-tiny.en \
1. `--task` transcription

```bash
# Terminal 1
# Install the huggingface cli
pip install -y huggingface_hub tree

# Download tiny-whisper.en locally
huggingface-cli download openai/whisper-tiny.en --local-dir ubuntu/audio/whisper-tiny.en --local-dir-use-symlinks False --repo-type model

# Ensure the whisper-tiny.en directory is next to your Dockerfile
tree .

# Build it
podman build -t vllm-whisper-offline -f ubuntu/audio/Dockerfile.offline ubuntu/

# Run it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  vllm-offline \
  --model /models/whisper-tiny.en \
  --served-model-name openai/whisper-tiny.en \
  --task transcription

# Terminal 2
# Test it
curl http://localhost:8000/v1/audio/transcriptions \
  -X POST -H "Content-Type: multipart/form-data" \
  -F file=@sample/harvard.wav \
  -F model=openai/whisper-tiny.en

# Terminal 2 output
# {"text":" The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."}

# Terminal 1 output
# ...
# INFO 05-12 17:13:43 [engine.py:310] Added request trsc-991982da47274b2cad929efd4be6fa46.
# INFO 05-12 17:13:43 [metrics.py:486] Avg prompt throughput: 0.7 tokens/s, Avg generation throughput: 0.2 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
# INFO:     127.0.0.1:49116 - "POST /v1/audio/transcriptions HTTP/1.1" 200 OK
# ...
```
