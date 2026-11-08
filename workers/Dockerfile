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
# torch 2.8, not 2.5/2.7: diffusers needs `torch.accelerator` (torch >= 2.6) and, for int8,
# torchao >= 0.15 — which targets torch >= 2.8. On torch 2.7 those two requirements cannot
# both hold, so the base image has to be current.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

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

# The `-runtime` image is minimal: no git, no curl. We need git for the diffusers
# `git+https://` requirement and curl for the upscaler weights below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# int8 weight-only needs torchao. Three traps: a wrong torchao installs cleanly and then breaks
# `import diffusers`; diffusers only enforces its version floor when the config is *constructed*;
# and the legacy string form ("int8_weight_only") raises that same misleading version error even
# on a good build. The check below therefore exercises the CURRENT API — a config object wrapped
# in a quant mapping — which is what handler.py actually uses at runtime.
RUN for v in 0.18.0 0.17.0 0.16.0 0.15.0 ; do \
        echo "torchao: trying $v" ; \
        if pip install --no-deps "torchao==$v" >/dev/null 2>&1 \
           && python -c "from torchao.quantization import Int8WeightOnlyConfig; from torchao.quantization.granularity import PerGroup; from diffusers import PipelineQuantizationConfig, TorchAoConfig; PipelineQuantizationConfig(quant_mapping={'transformer': TorchAoConfig(Int8WeightOnlyConfig(granularity=PerGroup(128)))})" >/dev/null 2>&1 ; then \
            echo "torchao: $v works" ; break ; \
        fi ; \
        pip uninstall -y torchao >/dev/null 2>&1 || true ; \
    done ; \
    python -c "import torchao; print('torchao', torchao.__version__)" \
      || echo "torchao unavailable; worker will serve bf16"

# Build-time import check. Everything above can succeed while the runtime is unusable —
# a diffusers/torch mismatch only surfaces when the pipeline is imported, which at runtime
# means a crash loop. Fail here instead, where the log is cheap to read.
RUN python -c "\
import torch, diffusers; \
from diffusers import Flux2KleinPipeline; \
print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| cuda', torch.version.cuda); \
assert hasattr(torch, 'accelerator'), 'torch too old for this diffusers'"

COPY pipeline.py stages.py download_weights.py handler.py server.py ./

# Bake the checkpoint only when asked. HF_HUB_OFFLINE is deliberately NOT set here — it
# would block this very download. handler.py sets it after the weights resolve.
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        HF_TOKEN="$HF_TOKEN" python download_weights.py /models/flux2-klein-4b; \
    else \
        mkdir -p /models/flux2-klein-4b && echo "weights not baked; expecting a volume or a runtime download"; \
    fi

# Bake the super-resolution weights (BSD-3, ~64MB). Tries a mirror first because the
# upstream GitHub release asset is unreliable from CI, and it is never allowed to fail the
# build: without the file the upscale stage reports itself unavailable and the worker serves
# un-upscaled images, which is a degraded feature, not a broken deployment.
RUN mkdir -p /models/realesrgan ; \
    for url in \
         https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth \
         https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth ; do \
         if curl -fsSL --retry 2 --max-time 180 -o /models/realesrgan/RealESRGAN_x4plus.pth "$url" \
            && test -s /models/realesrgan/RealESRGAN_x4plus.pth ; then \
             echo "upscaler weights baked from $url" ; break ; \
         fi ; \
       done ; \
    test -s /models/realesrgan/RealESRGAN_x4plus.pth \
      || echo "upscaler weights unavailable; upscale stage will report unavailable"

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
