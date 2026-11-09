"""Bake the prompt-expansion model into the image. Build-time only.

The worker's contract is "pull the image, run it". The expander's weights are the one part of
the model stack that used to be fetched lazily, on first use, from inside a request — 3GB over
the network at the worst possible moment. Fetching them here means the running container never
needs the Hub for them, and `expander._load` can insist on `local_files_only=True`.

Deliberately not `download_weights.py`: that file's allow-list is shaped for the diffusion
repo's component directories, and expanding it to cover a plain causal LM is how the 7.2GB
root duplicate got pulled in the first place. A stage with its own list is cheaper than a
pattern that has to mean two things.
"""

from __future__ import annotations

import os

REPO_ID = os.environ.get("HF_EXPAND_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TARGET = os.environ.get("IMOTAN_EXPAND_PATH", "/models/expander")

# A causal LM is flat: weights, config, tokenizer. No component subdirectories.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]


def main() -> int:
    from huggingface_hub import snapshot_download

    os.makedirs(TARGET, exist_ok=True)
    print(f"[imotan] baking expander {REPO_ID} -> {TARGET}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET,
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=ALLOW_PATTERNS,
    )
    total = 0
    for root, _dirs, files in os.walk(TARGET):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    print(f"[imotan] expander baked: {total / (1024 ** 3):.2f}GB at {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
