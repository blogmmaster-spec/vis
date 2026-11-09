"""Fetch the checkpoint into a target directory.

Used two ways:
  * at image build time (`--bake`), to embed the weights in the image
  * at worker boot, when MODEL_PATH is an empty network volume — the first worker in a
    data center populates it and every later worker mounts the cache instead of downloading

Resumable and *legible* on purpose. 13GB in flight is not an edge case for interruption: a
real deployment log shows one boot stopping at 17 of 19 files, the worker being replaced,
and the next boot starting from zero — repeatedly, because nothing recorded that the
directory was half-written. Two markers make that state readable:

  * `.imotan-weights-partial`  — written before the first byte, removed only once the
                                 snapshot completes, so an interrupted download can never
                                 be mistaken for a usable checkpoint
  * `.imotan-weights-complete` — written after `snapshot_download` returns

`snapshot_download` keeps its own `.incomplete` files, so re-running against the same
directory resumes instead of refetching. That only helps when the directory outlives the
worker, which is the network-volume case; on a bare container every boot starts cold no
matter what this file does.

Run it directly:  python download_weights.py /models/flux2-klein-4b
"""

from __future__ import annotations

import os
import sys
import time

REPO_ID = os.environ.get("HF_REPO", "black-forest-labs/FLUX.2-klein-4B")

COMPLETE_MARKER = ".imotan-weights-complete"
PARTIAL_MARKER = ".imotan-weights-partial"

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


def is_complete(path: str) -> bool:
    """True when `path` holds a complete checkpoint.

    The bare shard test is not enough once a download can be interrupted: a worker killed at
    17 of 19 files leaves plenty of `.safetensors` behind, and treating that as ready moves
    the failure to the loader, where it reads as a broken model rather than a missing file.
    Only the absence of both markers falls back to the shard test, which is what an image
    baked before these markers existed looks like.
    """
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, COMPLETE_MARKER)):
        return True
    if os.path.exists(os.path.join(path, PARTIAL_MARKER)):
        return False
    for _root, _dirs, files in os.walk(path):
        if any(name.endswith(".safetensors") for name in files):
            return True
    return False


def size_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total / (1024 ** 3)


def _snapshot(target: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=target,
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    )


def _disable_hf_transfer() -> None:
    """Turn hf_transfer off for a retry.

    The flag is read into `huggingface_hub.constants` at import time, so changing the
    environment variable after the import would be ignored and the retry would take the
    same failing path. Both are set: the env var for any child process, the constant for
    this one.
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        import huggingface_hub.constants as constants

        constants.HF_HUB_ENABLE_HF_TRANSFER = False
    except Exception:
        pass


def download(target: str) -> str:
    """Download the checkpoint into `target`, resuming if it was interrupted."""
    os.makedirs(target, exist_ok=True)
    complete = os.path.join(target, COMPLETE_MARKER)
    partial = os.path.join(target, PARTIAL_MARKER)

    # Before the first byte, so a worker killed mid-transfer leaves evidence behind.
    with open(partial, "w", encoding="utf-8"):
        pass

    # hf_transfer moves the transfer to a Rust multi-threaded downloader; the plain client
    # pulls one file at a time and is slow enough that the boot can lose its race with the
    # request's execution window. Set before the import below, which is where it is read.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    started = time.time()
    print(f"[imotan] downloading {REPO_ID} -> {target}", flush=True)
    try:
        _snapshot(target)
    except Exception as exc:
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
            # A worker that cannot fetch its weights serves nothing, so do not let an
            # accelerator pick the failure mode: fall back to the plain client.
            print(
                f"[imotan] hf_transfer failed ({type(exc).__name__}: {exc}); "
                "retrying with the plain client",
                flush=True,
            )
            _disable_hf_transfer()
            _snapshot(target)
        else:
            raise

    elapsed = time.time() - started
    print(
        f"[imotan] weights ready in {elapsed:.1f}s, {size_gb(target):.1f}GB at {target}",
        flush=True,
    )
    try:
        os.remove(partial)
    except OSError:
        pass
    with open(complete, "w", encoding="utf-8"):
        pass
    return target


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PATH", "/models/flux2-klein-4b")
    download(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
