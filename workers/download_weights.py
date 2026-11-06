"""Fetch the checkpoint into a target directory.

Used two ways:
  * at image build time (`--bake`), to embed the weights in the image
  * at worker boot, when MODEL_PATH is an empty network volume — the first worker in a
    data center populates it and every later worker mounts the cache instead of downloading

Run it directly:  python download_weights.py /models/flux2-klein-4b
"""

from __future__ import annotations

import os
import sys

REPO_ID = os.environ.get("HF_REPO", "black-forest-labs/FLUX.2-klein-4B")

# Everything the pipeline needs, minus training/optimiser state.
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "tokenizer*",
    "scheduler*",
    "transformer/*",
    "text_encoder/*",
    "vae/*",
]


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PATH", "/models/flux2-klein-4b")
    os.makedirs(target, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"[imotan] downloading {REPO_ID} -> {target}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=target,
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    )
    print("[imotan] weights ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
