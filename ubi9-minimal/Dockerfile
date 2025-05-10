# Use Red Hat UBI9-minimal as base
FROM registry.access.redhat.com/ubi9/ubi-minimal

# Install basic tools and Python
RUN microdnf install -y python3 python3-pip gcc git \
    && microdnf clean all

# Install CUDA-compatible PyTorch (make sure to match your CUDA version, e.g., 11.8 or 12.x)
RUN pip install --no-cache --upgrade pip \
    && pip install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (latest version)
RUN pip install --no-cache vllm[audio]==0.8.5

# Set container to use bash shell
CMD ["/bin/bash"]
