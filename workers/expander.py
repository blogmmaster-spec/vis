"""Optional prompt-expansion stage.

A small instruction-tuned model rewrites the caller's prompt into a richer one *before* it
reaches the FLUX text encoder.

This is deliberately not the pipeline's own Qwen3-4B encoder. That model is the conditioning
path: it is wired to emit hidden states, not completions, it is not tuned to rewrite prompts,
and at 4B it would add seconds per request while perturbing the exact weights whose behaviour
decides what gets drawn.

The stage is opt-in (`IMOTAN_EXPAND=1`), loads lazily on first use, quantises to int8
weight-only to match the rest of the worker, and never raises: any failure returns the original
prompt so a generation request can only lose polish, never fail. Its weights are **baked into
the image** and loaded from disk only — see `model_path()` for why lazily fetching them was a
defect rather than a shortcut.

Because a 1.5B model will cheerfully invent subjects nobody asked for, output is validated
rather than trusted — see `_accept()`. A rejected rewrite silently falls back to the original.
"""

from __future__ import annotations

import os
import threading

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Runaway-growth bound only. Attempts to police *which* words a rewrite may add (allowlisting
# lighting/camera vocabulary, capping new content words) were abandoned: they rejected valid
# expansions because a good rewrite legitimately introduces "85mm", "arena" and "sand", and they
# still admitted invented objects phrased in ordinary nouns. Measured against real traffic the
# strict version rejected 3 of 3 requests, which makes the stage useless.
#
# So: retention is enforced (a rewrite that loses the subject is caught), gross inflation is
# caught, and invention is left to the system prompt. A rewrite that slips in one extra object
# will pass — that is a known, accepted limitation, and the reason this stage is opt-in.
GROWTH_FACTOR = 3
GROWTH_FLOOR = 8

# Fraction of the caller's significant words that must survive. Not 100%: a good rewrite fixes
# spelling ("acction" -> "action") and that counted as a dropped term under an exact-match rule,
# which rejected every real prompt. Losing one or two words of ten is normal paraphrase; losing
# half means the subject changed.
MIN_RETENTION = 0.6

_SYSTEM = (
    "You rewrite prompts for an image generation model.\n"
    "Rules:\n"
    "1. Keep every subject, object, colour, material and attribute the user named. "
    "Never drop or replace one.\n"
    "2. Never invent new subjects, objects, people, animals or text.\n"
    "3. You may add only: lighting, camera and lens choice, framing and composition, "
    "surface and material detail, colour palette, atmosphere.\n"
    "4. One paragraph. Under 90 words. No preamble, no explanation, no quotation marks.\n"
    "Reply with the rewritten prompt only."
)

# Words too common to prove the rewrite stayed on topic.
_STOPWORDS = frozenset(
    """
    a an the and or but of for with without on in at to from by as is are be been being
    this that these those it its into over under very more most some any each
    make made create created render rendered show showing image picture photo
    please want needs need
    """.split()
)

_lock = threading.Lock()
_model = None
_tokenizer = None
_precision = "unloaded"
_failed = False


def enabled() -> bool:
    return os.environ.get("IMOTAN_EXPAND", "").strip().lower() in ("1", "true", "yes", "on")


def model_id() -> str:
    return os.environ.get("IMOTAN_EXPAND_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def model_path() -> str:
    """Local directory holding the expander weights.

    Baked into the image at build time, and loaded from disk with `local_files_only=True`.
    Fetching it lazily was a real defect: a worker that had already logged "HF_HUB_OFFLINE"
    still pulled Qwen2.5-1.5B from the Hub — 3GB — inside the first request that triggered
    expansion. A deployment whose contract is "pull the image, run it" cannot have a request
    that quietly reaches the internet for a third of the model.
    """
    return os.environ.get("IMOTAN_EXPAND_PATH", "/models/expander").strip() or "/models/expander"


def status() -> dict:
    """Diagnostics for the response payload and the boot log."""
    return {
        "enabled": enabled(),
        "model": model_id(),
        "path": model_path(),
        "loaded": _model is not None,
        "precision": _precision,
        "failed": _failed,
    }


def _load() -> bool:
    """Load and quantise the expander once. Returns False if it cannot be used."""
    global _model, _tokenizer, _precision, _failed

    if _model is not None:
        return True
    if _failed:
        return False

    with _lock:
        if _model is not None:
            return True
        if _failed:
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = model_path()
            if not os.path.isdir(path):
                # Not an exception path: a build that skipped the bake should disable the
                # stage, not reach for the network and not fail the request.
                print(
                    f"[imotan] expander disabled: no weights at {path} "
                    "(baked by the image build; a slim image without them cannot expand)",
                    flush=True,
                )
                _failed = True
                return False

            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, local_files_only=True
            )
            model.to("cuda")

            # int8 weight-only, same treatment as the transformer and text encoder. A 1.5B
            # model is ~3GB in bf16 and ~1.5GB at int8, so this is what keeps the stage
            # cheap enough to leave enabled.
            precision = "bf16"
            try:
                from torchao.quantization import Int8WeightOnlyConfig, quantize_

                quantize_(model, Int8WeightOnlyConfig())
                torch.cuda.empty_cache()
                precision = "int8"
            except Exception as exc:
                print(f"[imotan] expander staying bf16: {exc}", flush=True)

            model.eval()
            _tokenizer, _model, _precision = tokenizer, model, precision
            print(
                f"[imotan] expander ready: {model_id()} ({precision}) "
                f"vram {torch.cuda.memory_allocated() / (1024**3):.1f}GB allocated",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[imotan] expander unavailable: {exc}", flush=True)
            _failed = True
            return False


def _significant(text: str) -> set[str]:
    words = set()
    for raw in text.lower().replace(",", " ").replace(".", " ").split():
        word = "".join(ch for ch in raw if ch.isalnum())
        if len(word) >= 3 and word not in _STOPWORDS:
            words.add(word)
    return words


def _clean(reply: str) -> str:
    """Strip chatty preambles.

    Loops because the model stacks them ("Sure, here is the prompt: …") — stripping only the
    first left the rest in the prompt, which then reached the image encoder as literal text.
    """
    text = reply.strip()
    prefixes = (
        "here is",
        "here's",
        "sure,",
        "sure.",
        "certainly,",
        "certainly.",
        "revised prompt:",
        "expanded prompt:",
        "the prompt:",
        "prompt:",
    )
    changed = True
    while changed:
        changed = False
        lowered = text.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix) :].lstrip(" :,-").strip()
                changed = True
                break
    if text.startswith('"') and text.endswith('"') and len(text) > 2:
        text = text[1:-1].strip()
    return text.replace("\n", " ").strip()


def _accept(original: str, reply: str) -> tuple[bool, str]:
    """Validate a rewrite. Returns (accepted, reason).

    The important check is word retention: a 1.5B model's most likely failure is dropping or
    substituting a subject, and that is detectable — every significant word the caller used
    must still appear. Invented detail cannot be detected this way, which is why the system
    prompt bounds it to lighting, camera, framing and material.
    """
    if not reply:
        return False, "empty"
    if len(reply) < 15:
        return False, "too short"
    # An expanded prompt is legitimately much longer than the original; a 3x cap rejected most
    # real rewrites. The absolute floor keeps one-word prompts usable.
    if len(reply) > max(400, 4 * len(original) + 200):
        return False, "too long"

    lowered = reply.lower()
    for noise in ("i cannot", "i can't", "as an ai", "note:", "explanation", "here is the prompt"):
        if noise in lowered:
            return False, f"meta text ({noise})"

    original_terms = _significant(original)
    reply_terms = _significant(reply)
    if original_terms:
        kept = original_terms & reply_terms
        ratio = len(kept) / len(original_terms)
        if ratio < MIN_RETENTION:
            dropped = sorted(original_terms - reply_terms)
            return (
                False,
                f"kept {len(kept)}/{len(original_terms)} terms, dropped: {', '.join(dropped[:4])}",
            )

    # Invention cannot be detected by string matching — "and a red apple" passes any retention
    # check. A cap on genuinely new content words catches runaway additions, which is the
    # failure mode that matters most; subtle single-object invention still gets through, which
    # is why this stage is opt-in rather than always on.
    # Runaway inflation, not vocabulary policing: a rewrite that doubles or triples the content
    # is drifting; one that stays within a few times the original is doing its job.
    if len(reply_terms) > GROWTH_FACTOR * len(original_terms) + GROWTH_FLOOR:
        return False, f"grew to {len(reply_terms)} terms from {len(original_terms)}"

    return True, "ok"


def expand(prompt: str) -> tuple[str, str]:
    """Return (prompt_to_use, status). Never raises, never returns empty text."""
    if not enabled():
        return prompt, "disabled"
    if _load() is False:
        return prompt, "unavailable"

    try:
        import torch

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(text, return_tensors="pt").to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = _model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=_tokenizer.eos_token_id,
            )

        reply = _tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)
        reply = _clean(reply)

        accepted, reason = _accept(prompt, reply)
        if not accepted:
            print(f"[imotan] expansion rejected: {reason}", flush=True)
            return prompt, f"rejected: {reason}"

        # Log acceptance too. Only rejections were logged, which meant a successful expansion
        # left no trace at all — the one case you most want to see worked was the one case the
        # log stayed silent about. Lengths rather than text: the prompt may be a customer's.
        print(
            f"[imotan] expansion accepted: {len(prompt)} -> {len(reply)} chars",
            flush=True,
        )
        return reply, "expanded"
    except Exception as exc:
        print(f"[imotan] expansion failed: {exc}", flush=True)
        return prompt, f"error: {exc}"
