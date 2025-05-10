# Kludges to make it work

- https://docs.vllm.ai/en/v0.8.5/deployment/docker.html
- https://github.com/vllm-project/vllm/issues/16724

## Quickstart

```sh
# build
podman build -t vllm .

# run
podman run -it --rm \
  -p 8000 \
  --shm-size=2G \
    vllm:latest
```
