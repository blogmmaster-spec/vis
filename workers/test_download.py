"""Regression tests for checkpoint-completeness detection. No GPU, no network, no model.

This is the check that decides whether a worker loads weights or downloads them, and it
ran the wrong way in a live deployment: a boot killed at 17 of 19 files left plenty of
`.safetensors` on disk, the next boot called that ready, and the failure surfaced much
later as a model that would not load. The markers exist so that state is not guessable.

Run with:  python3 workers/test_download.py
"""

from __future__ import annotations

import os
import shutil
import sys

import download_weights as dw

# Scratch space next to this file rather than the system temp dir: the latter is not
# writable under every sandbox, and a test that cannot create its fixture fails for a
# reason that has nothing to do with the code under test.
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".test_download_scratch")

CASES = [
    # name, files to create, expected is_complete()
    ("empty directory", [], False),
    ("missing directory", None, False),
    # Baked before the markers existed: shards on disk are all the evidence there is.
    ("legacy baked shard", ["model-00001-of-00002.safetensors"], True),
    # Interrupted: shards present, but the partial marker says the snapshot never finished.
    ("interrupted download", ["model-00001-of-00002.safetensors", dw.PARTIAL_MARKER], False),
    # Finished: the marker is authoritative even with no shard of that name.
    ("completed download", [dw.COMPLETE_MARKER], True),
    # The complete marker wins: it is only written after a snapshot returns, so the shards
    # on disk came from a finished download. A stray partial marker cannot occur in practice
    # — `download()` only runs when `is_complete()` is already False — and if one is there,
    # the files are still the completed ones.
    ("complete then partial", [dw.COMPLETE_MARKER, dw.PARTIAL_MARKER], True),
    ("unrelated files", ["config.json", "tokenizer.json"], False),
]


def main() -> int:
    failures = 0
    width = max(len(case[0]) for case in CASES)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    try:
        for index, (name, files, want) in enumerate(CASES):
            path = os.path.join(SCRATCH, str(index))
            if files is not None:
                os.makedirs(path)
                for filename in files:
                    with open(os.path.join(path, filename), "w", encoding="utf-8"):
                        pass
            got = dw.is_complete(path)
            ok = got == want
            print("{0} {1:<{2}}  complete {3} (want {4})".format(
                "ok  " if ok else "FAIL", name, width, got, want))
            if not ok:
                failures += 1
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    print()
    # The root single-file checkpoint is a 7.2GB duplicate of `transformer/`; matching it
    # cost that much on every cold boot. Cheap to assert, invisible if it regresses.
    bare = "*.safetensors"
    covered = bare in dw.ALLOW_PATTERNS
    print("{0} root duplicate excluded  '*.safetensors' in ALLOW_PATTERNS: {1} (want False)".format(
        "FAIL" if covered else "ok  ", covered))
    if covered:
        failures += 1

    if failures:
        print(f"{failures} of {len(CASES) + 1} cases failed")
        return 1
    print(f"{len(CASES) + 1} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
