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
skopeo inspect docker://docker.io/vllm/vllm-openai | jq '.Labels'

# output
# {
#   "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
#   "org.opencontainers.image.ref.name": "ubuntu",
#   "org.opencontainers.image.version": "22.04"
# }
```

## Review supported vLLM models [here](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models)

The shortlist meets these criteria:

1. top-ranked models from the Hugging Face LLM leaderboard that
2. are supported by vLLM
3. fit under ~14.5 GiB

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

1. Chat → Conversational agents, dialogue systems
1. Instruction → Follows user prompts, task-specific
1. Summarization → Text summarization, document compression
1. Reasoning → Multi-step logic, Q&A, problem-solving
1. Multilingual → Supports multiple languages

### GPU

```bash
# Terminal 1
# Run it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype=half \
  --gpu-memory-utilization 0.7

# Expected output
# ...
# INFO:     Started server process [1]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
```

```bash
# Terminal 2
# Prompt the model Hello, TinyLlama! How are you today?
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
          {"role": "user", "content": "Tell me about Red Hat?"}
        ]
      }'

# response sample
# {"id":"chatcmpl-fe19985ece584222ab12376a73a71325","object":"chat.completion","created":1747102800,"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"Red Hat (NYSE: RHT) is a global provider of enterprise open source solutions, using a community-powered approach to deliver high-performing technologies that help manage and install Linux, mainframe, cloud, and hybrid cloud environments. Led by a global leader team that provides customers with the best technical support, Red Hat operates under the business principles of unparalleled customer success, community love, and responsible innovation. As an open source-friendly company, Red Hat operates as a community-powered board made up of technical experts, each contributing their specific areas of expertise to the Red Hat Open Source Foundation (OFO). This means they monitor compliance with open source business models, work towards specific goals and policies, and contribute their community expertise with passionate coopereness towards Red Hat's initiatives from strategic and business standpoint. Red Hat's products and services are built on the operating system technology of the Linux Foundation's Linux Foundation Linux Foundation Linux Foundation. With over 30,000 customers and 300 partners across 180 countries, the company is a software industry leader driving necessary technological solutions to power, enable and simplify online business.\n\nKey Benefits of Working with Red Hat:\nRed Hat provides comprehensive cloud-enabled computing solutions, helping customers transform their IT initiatives and create digitally enabled businesses. Some of the key benefits include:\n\n1. Comprehensive Cloud Portfolio: Red Hat delivers the industry's largest portfolio of cloud-native applications and services that power cloud applications.\n\n2. Scalable Platform: Ensures the fastest speed for development and deployment while enabling customers to create scalable cloud-native solutions with innovative intelligence solutions.\n\n3. Enterprise-Grade Service: Providing state-of-the-art security, resiliency, and stability-model solutions to help optimize provider migration in all environments.\n\n4. Enhanced Power to Boost Business Growth: Provides enterprise-class, scalable and secure cloud computing and IoT solutions for maximum business growth.\n\n5. End-to-end Digitalization: Enhances business transformation with the technology and infrastructure alongside key digital services to accelerate journey towards digital.\n\nReasons to Choose Red Hat:\n\n1. Open Source Provides Workaround: Provides a community-backed ongoing support based on open source.\n\n2. Stay Directly on the Open Source Framework: With the advantage of immediacy, Red Hat helps companies remain up-to-date with latest releases based on Open Source technologies.\n\n3. Industry-First EULA: Red Hat Legal guaranteeing its customers' and partners’ EULA allowing them to do as per open standards.\n\n4. Comprehensive Software Development: Red Hat provides the inclusive support for software development as world-renowned open-source community.\n\n5. Product Requests and Reviews Support: Red Hat is the trusted source for reviewing, maintaining, and supporting its software, ensuring nothing falls down to manufacturing.\n\nRed Hat and Its Technological Solutions:\n\nHere are some of Red Hat's technological solutions:\n\n1. OpenShift Container Platform: An open-source container delivery system, available in two editions providing containers and Kubernetes support.\n\n2. Open Storage Solutions: Offers Ironic, Ceph-based storage, fault tolerance and scale for burgeoning IT infrastructure.\n\n3. Red Hat OpenStack Platform: The industry leading OpenStack software solution supports labor, available on-demand to deliver a secure cloud.\n\n4. Red Hat Open Source Center: Offers an interactive and easy-to-use, access optimized marketplace, providing customers the chance to buy and hang with supported and ready-made open-source applications.\n\nIn conclusion, Red Hat, a leading provider of enterprise open source solutions, operates under the sole-setting of a community-powered board, where technical experts contribute their critical expertise in specific community domains with a vision to unite the technologies and deliver unmatched technological solutions and services. With its core and diverse products, enabling open-source innovation, it operates as a distinct innovation force to be reckoned with. Red Hat's comprehensive ecosystem of cloud-native applications, such as OpenShift, OpenStack, Red Hat Enterprise Linux, Red Hat middleware, builds distinct business value for companies in various industry domains.","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":23,"total_tokens":997,"completion_tokens":974,"prompt_tokens_details":null},"prompt_logprobs":null}
```

This container:

1. Serves a model with vllm
1. Uses FastAPI to expose an OpenAI-compatible API
1. Accepts inference requests on port 8000
1. Is connected download the model from Hugging Face at runtime --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

#### CPU

```bash
podman run --rm -it \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cpu \
  --dtype float32 \
  --disable-async-output-proc

# Expected output
# TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
# ...
# RuntimeError: Engine process failed to start. See stack trace for the root cause.
```

`self.max_seq_len_to_capture` is None, and vLLM tries to add it to an int. This is an internal bug where the CPU logic path doesn't initialize all config values properly — especially when using the older fallback engine (V0), which we hit due to:

- `--device cpu`
- `--disable-async-output-proc`
- `--worker-cls vllm.worker.worker.Worker`

**So is CPU support broken?**
Yes — as of v0.8.5.post1, vLLM's CPU support is incomplete and unstable, especially for models like TinyLlama, and:

- V1 Engine doesn’t yet support CPU
- V0 Engine is legacy and mostly tuned for GPU logic
- Many configurations (like memory size, block manager, etc.) are not correctly initialized on CPU fallback

## Audio

Explanation of Tasks:

1. Transcription → Convert spoken audio to text in the same language
1. Translation → Convert non-English speech to English text
1. Language Detection → Auto-detect spoken language (used implicitly)
1. Timestamps → Include word or phrase-level time codes
1. (Future) Diarization → Separate and label different speakers

### Connected Environments

```bash
# Terminal 1
# Create the directory before running the container
mkdir -p ~/.cache/huggingface

# Build the Dockerfile with the same version
podman build -t vllm-whisper-basic -f ubuntu/Dockerfile.basic ubuntu/

# Test it
podman run --rm -it \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  vllm-whisper-basic \
  --model openai/whisper-tiny.en \
  --task transcription \
  --dtype=half

# Terminal 2
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
tree ubuntu

# Build it
podman build -t vllm-whisper-offline -f ubuntu/audio/Dockerfile.offline ubuntu/audio/

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
