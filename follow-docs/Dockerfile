FROM docker.io/vllm/vllm-openai:v0.8.5.post1

RUN uv pip install --system --no-cache vllm[audio]==0.8.5.post1

ENV VLLM_LOGGING_LEVEL=DEBUG \
    VLLM_TRACE_FUNCTION=1 \
    MODEL_NAME=openai/whisper-tiny

# ENTRYPOINT ["sh", "-c" ]
# CMD [ "vllm serve ${MODEL_NAME} --task transcription" ]
