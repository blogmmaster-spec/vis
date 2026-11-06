# Imotan Vision 1 — image worker
#
# Two deployment shapes from one file:
#
#   BAKE_WEIGHTS=1 (default)  weights embedded in the image. Fast cold starts, ~20GB image
#                             and a slow push.
#   BAKE_WEIGHTS=0            small image (~8GB). Point MODEL_PATH at a network volume and
#                             the first worker in a data center downloads the weights into
#                             it; later workers mount that cache. See README.
#
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG BAKE_WEIGHTS=1
ARG HF_TOKEN=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/models/flux2-klein-4b \
    IMOTAN_DTYPE=int8 \
    IMOTAN_FORMAT=jpeg \
    IMOTAN_BEST_OF=2 \
    IMOTAN_STEPS=4

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# int8 weight-only is a nice-to-have, never a build gate. torchao pins an exact torch
# version, so a mismatch here must not fail the image — the worker falls back to bf16.
RUN pip install torchao \
    || echo "torchao unavailable; int8 disabled, worker will serve bf16"

COPY pipeline.py stages.py download_weights.py handler.py server.py ./

# Bake the checkpoint only when asked. HF_HUB_OFFLINE is deliberately NOT set here — it
# would block this very download. handler.py sets it after the weights resolve.
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        HF_TOKEN="$HF_TOKEN" python download_weights.py /models/flux2-klein-4b; \
    else \
        mkdir -p /models/flux2-klein-4b && echo "weights not baked; expecting a volume or a runtime download"; \
    fi

# Bake the super-resolution weights (BSD-3, ~64MB). Always baked — small, and the upscale
# stage then needs no network at runtime.
RUN mkdir -p /models/realesrgan \
    && curl -fsSL -o /models/realesrgan/RealESRGAN_x4plus.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth \
    && test -s /models/realesrgan/RealESRGAN_x4plus.pth \
    && echo "upscaler weights baked"

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
