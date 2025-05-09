# Use NVIDIA CUDA base image built on UBI9 minimal
FROM nvidia/cuda:12.4.1-devel-ubi9

# Install system dependencies (microdnf for UBI)
RUN microdnf update -y && microdnf install -y \
    python3 python3-pip git gcc gcc-c++ make curl \
    && microdnf clean all

# Upgrade pip + install CUDA-compatible PyTorch
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM (latest)
RUN python3 -m pip install vllm

# Set working directory
WORKDIR /workspace

# Default command (can be overridden at runtime)
CMD ["bash"]
