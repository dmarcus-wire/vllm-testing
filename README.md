# Experimenting vLLM

Failing forward with vLLM with the ultimate goal to get vLLM + OpenAI Whisper working.

Minimal Prereqs [provisioned with this procedure](https://github.com/redhat-na-ssa/whitepaper-stt-evaluation-on-kubernetes/blob/main/crawl/RHEL_GPU.md):

1. RHEL VM w/podman
1. GPUs - g4dn12.xlarge (4x NVIDIA Telsa)

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

Here’s a short list of models you can directly use with vllm. A common approach when getting started is checking the [leaderboards](https://huggingface.co/open-llm-leaderboard).

This table shows models ~=<7b models that could fit on the Telsa and are on both the supported models list and leaderboard

| Model Family   | Example Models  | Approx. Size |  Tasks  |
|-|-|-|-|
| LLaMA          | TinyLlama 1.1B, LLaMA-3 8B*            | 1.1B–8B     | Chat, general-purpose, reasoning    |
| DeepSeek       | DeepSeek-V2 1.3B                       | 1.3B        | Chat, reasoning, instruction        |
| Mistral        | Mistral 7B                             | 7B          | Chat, summarization, instruction    |
| GPT-family     | GPT2 (1.5B), GPT-Neo (1.3B, 2.7B), GPT-J 6B | 1.3B–6B     | Text generation, summarization      |
| Phi            | Phi-2 (2.7B), Phi-3-mini (3.8B)        | 2.7B–3.8B   | Chat, instruction, lightweight use  |
| Gemma          | Gemma 2B                               | 2B          | Chat, summarization, instruction    |
| Qwen           | Qwen 1.5B, Qwen 2.5B                   | 1.5B–2.5B   | Chat, summarization, multilingual   |
| Yi             | Yi-6B                                  | 6B          | Chat, instruction, general-purpose  |
| Baichuan       | Baichuan 2 4B                         | 4B          | Chat, summarization, multilingual   |
| Orca           | Orca 2 3B                              | 3B          | Chat, instruction, reasoning        |
| InternLM       | InternLM 1.8B                          | 1.8B        | Chat, lightweight reasoning         |

Explanation of Tasks:
- Chat → Conversational agents, dialogue systems
- Instruction → Follows user prompts, task-specific
- Summarization → Text summarization, document compression
- Reasoning → Multi-step logic, Q&A, problem-solving
- Multilingual → Supports multiple languages

Important: many of these models you must have access to it and be authenticated to access it via HuggingFace, like meta-llama.

## Ubuntu testing

### Try Mistral-7B

1. Accept the [mistralai/Mistral-7B-Instruct-v0.3 user agreement](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
1. Create and pass a [HuggingFace token](https://huggingface.co/settings/tokens) as an environment variable

```bash
# THIS WILL FAIL
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  docker.io/vllm/vllm-openai \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dtype=half \
  --gpu-memory-utilization 0.7
```

```bash
# error
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 40.81 MiB is free
```
You are hitting a hard memory ceiling; the Tesla T4 is simply too small for full-size 7B models like Mistral-7B, even under best-effort settings. Even with:

1. --dtype=half (fp16)
1. --gpu-memory-utilization 0.7
1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

| **Cause**| **Explanation**|
| - | - |
| Model size too large                | The **Mistral-7B-Instruct** model is a 7B-parameter transformer, needing \~13–14 GB just for weights plus extra for the KV cache, activations, and runtime memory.           |
| KV cache + batch preallocations     | vLLM preallocates space for batching and KV cache; even with reduced `gpu-memory-utilization`, this overhead pushes memory use beyond what the T4 can handle.                |
| System + driver GPU memory overhead | The Tesla T4 reports \~14.5 GiB usable, but in reality, a few hundred MB are eaten by drivers (NVIDIA runtime, X11/Wayland, CUDA context), leaving <14 GiB for the workload. |
| No quantization or sharding used    | You’re running full precision (half) single-GPU; to fit large models on limited GPUs, you’d need int8/4-bit quantization or tensor parallelism over multiple GPUs.           |

What can you do about it?

| Option | Notes |
| - | - |
| Switch to a smaller model           | Use TinyLlama (1.1B), phi-2, or other small models that comfortably fit in <8 GB VRAM.                                                             |
| Switch to larger GPUs               | Migrate to A10 (24 GB), L4 (24 GB), A100 (40–80 GB), or H100 GPUs — these can load 7B+ models easily.                                              |
| Use quantized models (if supported) | If a quantized (int8 or 4-bit) checkpoint exists, it may fit — but note vLLM support is still maturing for some quantization formats.              |
| Multi-GPU tensor parallelism        | Requires multiple GPUs on a node **and** enabling `--tensor-parallel-size`, plus a properly configured NCCL environment — not usable on single T4. |

### Review filtered shortlist of models

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

### Try Mixtral-8x7B

1. Accept the [mistralai/Mixtral-8x7B-Instruct-v0.1 user agreement](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

```bash
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -e HF_TOKEN=YOUR_HUGGINGFACE_TOKEN \
  docker.io/vllm/vllm-openai \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --dtype=half \
  --gpu-memory-utilization 0.7

podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -e HF_TOKEN=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 \
  docker.io/vllm/vllm-openai \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --dtype=half \
  --gpu-memory-utilization 0.7
```

```bash
# error
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.75 GiB. GPU 0 has a total capacity of 14.56 GiB of which 602.81 MiB is free.
```

### Try Tiny-Llama

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

- Sends a basic chat prompt (Hello, TinyLlama! How are you today?).
- Hits the OpenAI-compatible /v1/chat/completions endpoint exposed by vLLM.
- Expects a JSON response containing TinyLlama’s reply.

### Create Dockerfiles

```bash
# Base Ubuntu image (match to your preferred version, e.g., 22.04)
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git gcc g++ make curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-compatible PyTorch (adjust CUDA version if needed)
RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (latest)
RUN pip3 install vllm

# Default command (can be overridden)
CMD ["bash"]

```

### Build the container

```bash
podman build -t vllm-ubuntu-tinyllama ubuntu/.
```


```bash
$ podman images
REPOSITORY                       TAG         IMAGE ID      CREATED        SIZE
localhost/vllm-ubuntu-tinyllama  latest      a4120b79c563  2 minutes ago  21.3 GB
docker.io/vllm/vllm-openai       latest      5068c8d73dbd  7 days ago     17.4 GB
docker.io/library/ubuntu         22.04       c42dedf797ba  11 days ago    80.4 MB
```

### Run the container

On the NVIDIA T4 without --gpu-memory-utilization set, you get torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 500.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 44.81 MiB is free. Including non-PyTorch memory, this process has 4.22 GiB memory in use. Of the allocated memory 4.01 GiB is allocated by PyTorch, and 74.75 MiB is reserved by PyTorch but unallocated.

- Your GPU (Tesla T4) has ~14.5 GiB usable VRAM.
- TinyLlama-1.1B model weights fit perfectly (only ~2 GiB)
- After loading weights (~2 GiB) and accounting for runtime,
- During vLLM’s KV cache allocation, it still ran out of memory.
- vLLM tries to preallocate ~10–11 GiB for KV cache for maximum batching/concurrency.
- It hit CUDA OOM when trying to allocate ~500 MiB more for cache.

```bash
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8001:8000 \
  localhost/vllm-ubuntu-tinyllama \
  python3 -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype=half
    --gpu-memory-utilization 0.4
```

## UBI9 testing

Here’s how you can run TinyLlama with vLLM inside a UBI9-minimal–based container instead of the default docker.io/vllm/vllm-openai image.

Because UBI9-minimal doesn’t ship with Python, PyTorch, or vLLM preinstalled, you need to:

1. Build a custom container based on UBI9-minimal
1. Install Python + dependencies
1. Install vLLM
1. Add CUDA + PyTorch support
