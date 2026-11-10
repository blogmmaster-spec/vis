# Imotan Vision 1 — image worker (RunPod Serverless)

FLUX.2 **[klein] 4B** (distilled, 4 steps) with the launch pipeline: prompt shaping, aspect
ratios, best-of-2 selection, super-resolution, vectorisation, and JPEG/WebP/PNG output — all
behind one OpenAI-compatible HTTP contract.

**No fine-tuning is required to deploy this.** The checkpoint is a finished Apache-2.0 model.
Training (the 5.2B expansion in `IMAGE-MODEL.md`) is applied later by pointing `MODEL_PATH` at a
new checkpoint. Nothing in this worker changes.

---

## Why there is a shim

RunPod's native contract is `POST /runsync` with `{"input": {...}}`. An OpenAI-shaped
`POST /v1/images/generations` is what image clients and marketplaces expect. `server.py`
translates, so the worker has one public contract.

```
POST /v1/images/generations
        │
        ▼
   server.py ──▶ handler.generate_payload
                   ├─ pipeline.build_prompt        (intent → style suffix, or none)
                   ├─ pipeline.dimensions_for      (aspect ratio → ~1MP, /16 aligned)
                   ├─ best-of-N + technical_score  (rejects blurred / flat candidates)
                   ├─ stages.upscale               (Real-ESRGAN, optional)
                   ├─ stages.vectorize             (vtracer → SVG, optional)
                   └─ pipeline.encode              (jpeg / webp / png)
```

`handler.py` also answers `/runsync` if you deploy a queue-based endpoint for manual tests.

---

## Build

```bash
cd workers/image-worker

# A: weights baked in (default) — fast cold starts, ~20GB image
docker build -t <registry>/imotan-vision-1:klein4b .

# B: small image, weights downloaded once into a network volume — ~8GB image
docker build --build-arg BAKE_WEIGHTS=0 -t <registry>/imotan-vision-1:klein4b-slim .

docker push <registry>/imotan-vision-1:klein4b
```

If the checkpoint repo is ever gated, pass `--build-arg HF_TOKEN=...`.

### Weight provisioning — pick one

| | Image | First boot on a host | Later boots | Push size |
|---|---|---|---|---|
| **A. Baked** | ~20GB | pull 20GB | fast, local NVMe | 20GB |
| **B. Network volume** | ~8GB | pull 8GB + one ~16GB download into the mount | mounts the cache | 8GB |
| **C. Download every boot** | ~8GB | pull + ~16GB download | **~16GB again, every new host** | 8GB |

**Do not ship C.** With `active workers: 0`, every new host re-downloads the weights and your
first request of the day waits for all ~16GB. It only looks acceptable after FlashBoot has
warmed a host once.

Choose **B** when the image push is the bottleneck (slow uplink) or when you want to iterate on
the Dockerfile without re-uploading weights each time:

1. Create a network volume (≥30GB) **in a data center that has 3090/A5000 stock**.
2. Mount it at `/models`.
3. Set `MODEL_PATH=/models/flux2-klein-4b` — the handler downloads into it on the first boot
   and every later worker mounts the populated cache (`_ensure_weights` in `handler.py`).
4. Pin the endpoint to that data center. A volume outside the GPUs' data center cannot mount.

`IMOTAN_ALLOW_DOWNLOAD=0` disables the runtime download and fails fast instead — use it on the
baked image so a mis-set path surfaces as an error rather than a ~16GB surprise.

A download is flagged before it starts (`.imotan-weights-partial`) and stamped when it
finishes (`.imotan-weights-complete`). Without that, a worker killed at 17 of 19 files leaves
enough `.safetensors` behind for the next boot to call the directory ready, and the failure
moves to the loader where it reads as a broken model. With it, an interrupted boot is
recognised and resumed — which only helps if the directory outlives the worker, i.e. on a
mounted volume. Two more things make the download survivable inside a request's execution
window: `HF_TOKEN` lifts the Hub's anonymous rate limit, and `hf-transfer` (in requirements)
downloads many chunks at once instead of one file at a time. If hf_transfer errors, the
download retries on the plain client rather than failing the boot.

### Nothing is fetched at runtime

The deployment contract is "pull the image, run it", so every artifact the worker needs is in
the image or on the volume:

| Artifact | Where it comes from |
|---|---|
| FLUX.2 klein weights | baked (`BAKE_WEIGHTS=1`) or a mounted volume |
| prompt-expansion LM (~3GB) | baked unconditionally — see below |
| upscaler (~64MB) | baked |
| CTA font | bundled in `fonts/` |

The expander's weights used to be fetched lazily from the Hub on first use, which put a 3GB
download inside a customer request — and the `HF_HUB_OFFLINE` guard did not stop it:
`huggingface_hub` reads that variable into `constants` at import time, and `handler` imports
diffusers long before the weights resolve, so setting the env var later was inert. Both are
fixed: the build bakes those weights, `expander._load` insists on `local_files_only=True`, and
`_go_offline()` sets the env var *and* the constant.

`HF_HUB_OFFLINE` is set by the handler **after** the weights resolve, so a baked worker never
touches the network while a volume-backed one still can.

The build bakes ~64MB of upscaler weights in both shapes — small enough that it is never worth
deferring.


---

## Deploy

**Endpoint type: load-balancing** (recommended). It routes straight to a worker with no queue
and serves the HTTP contract above. The image `CMD` already runs
`uvicorn server:app --host 0.0.0.0 --port 8000` — expose port **8000**.

| Setting | Value | Why |
|---|---|---|
| GPU | RTX 3090 or A5000 (24GB), 1 per worker | 4B at bf16 uses ~13GB |
| GPU priority | list both | availability under demand spikes |
| `max workers` | **1–2** | a burst must not fan out into N cold starts you pay for |
| `idle timeout` | 5–10s | at low traffic a warm idle worker is pure burn |
| `active workers` | **0** | no customers yet — see `IMAGE-MODEL.md` §2.2 |
| FlashBoot | **on** | ~3s restore instead of ~25s boot |
| Execution timeout | 120s | covers best-of-2 plus a slow queue |

The worker loads and warms the model **at boot**, before serving. That is what gives FlashBoot
a warm process to snapshot; without it every cold start pays the full load.

### Environment

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `/models/flux2-klein-4b` | baked path; repoint to upgrade the model |
| `IMOTAN_STEPS` | `4` | declare as an enum `[4]` in listings, never a range |
| `IMOTAN_DTYPE` | `int8` | `bf16` forces the unquantised path without a rebuild |
| `IMOTAN_EXPAND` | *(off)* | `1` enables the prompt-expansion stage |
| `IMOTAN_EXPAND_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | expander weights; int8 loaded on first use |
| `IMOTAN_EXPAND_PATH` | `/models/expander` | where the build baked those weights; loaded with `local_files_only=True` |
| `IMOTAN_BEST_OF` | `2` | candidates per requested image, capped at 3 |
| `IMOTAN_TEXT_BEST_OF` | `4` | candidates per image when the prompt quotes text to render; capped at 8, inside a 16-render budget |
| `IMOTAN_COMPOSE_TEXT` | `1` | draw the quoted string instead of generating it; `0` restores generate-and-verify |
| `IMOTAN_FORMAT` | `jpeg` | default when the request omits `format` |
| `IMOTAN_QUALITY` | `90` | JPEG/WebP quality |
| `IMOTAN_UPSCALE_MODEL` | `/models/realesrgan/RealESRGAN_x4plus.pth` | baked |
| `IMOTAN_UPSCALE_MAX_PIXELS` | `16777216` | hard cap on upscaled output (~4K) |
| `IMOTAN_WORKER_TOKEN` | *(empty)* | if set, requires `Authorization: Bearer <token>` |

---

## Precision

**Verified state of the shipped image:**

| | |
|---|---|
| torch | `2.8.0+cu128` |
| diffusers | `0.42.0.dev0` (git main) |
| torchao | `0.17.0` — 0.18 requires torch ≥ 2.11 |
| transformer | **int8 weight-only** |
| text encoder | **bf16** (see below) |
| VRAM at load | **11.46GB** |
| VRAM peak, 2 candidates @ 1184×880 | **13.76GB** |
| GPU total | 23.5GB |

The Dockerfile tries torchao 0.18 → 0.15 and keeps the first version that survives the *current*
API check — a config object wrapped in a quant mapping, which is what `handler.py` uses at
runtime. If none work, no torchao is installed and the worker serves bf16.

Every boot logs measured per-component size and dtype, plus VRAM:

```
[imotan] transformer: 3.85GB int8
[imotan] text_encoder: 7.53GB bfloat16
[imotan] vae: 0.16GB bfloat16
[imotan] vram after load: 11.5GB allocated, 11.9GB reserved, 11.7GB peak, 23.5GB total
```

Read those rather than the config label — the label reports what was requested, the lines
report what actually loaded.

**Why the text encoder is not quantised.** The encoder is Qwen3-4B (~8GB in bf16) and is the
largest single resident component, so it is the obvious next saving. But listing it in the
diffusers `quant_mapping` makes `from_pretrained` die inside huggingface_hub argument
validation — the component is a transformers model and diffusers re-enters the hub loader for
it — which crash-loops the worker. Doing it properly means loading the encoder separately with
`transformers.TorchAoConfig` and assigning `pipe.text_encoder` afterwards.

Why int8 and not fp8: fp8 *compute* needs Ada or Hopper, so on a 3090 or A5000 it is emulated
and slower than bf16. int8 weight-only dequantises inside the matmul, so bf16 compute is
retained while half the weight bytes move.

---

## Request contract

```json
{
  "prompt": "a matte black speaker on sand",
  "aspect_ratio": "4:3",
  "n": 1,
  "format": "jpeg",
  "seed": 42,
  "upscale": 2
}
```

| Field | Accepts | Notes |
|---|---|---|
| `prompt` | string, required | shaped, never rewritten |
| `aspect_ratio` | `1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `2:3`, `3:2`, `auto` | ~1MP target |
| `size` | `"1344x768"` | explicit override, wins over `aspect_ratio` |
| `n` | 1–4 | each image gets its own best-of-N selection |
| `steps` | 1–16 | default 4 (the distilled schedule) |
| `format` | `png`, `jpeg`, `webp`, `svg` | defaults to `jpeg`; `svg` runs the vectoriser |
| `upscale` | `2`, `4`, `true` | `true` = 2×; skipped if unavailable |
| `seed` | integer | deterministic per candidate |
| `style` | string | folded into the prompt, not sent upstream |
| `include_images` | `true` (default), `false` | `false` drops `data[].b64_json` and keeps the whole `imotan` report — see below |

**Diagnosing without the pixels.** A PNG's base64 is megabytes, and it is returned first, so
every diagnostic in the response sits below a wall of it — which makes the report unreadable
in a console or a log. `"include_images": false` returns the same `imotan` block with
`data[]` reduced to `media_type`/`revised_prompt` and `imotan.images_omitted: true`, so an
empty `data` cannot be mistaken for a failed generation:

```json
{ "prompt": "A banner with a \"Get Started for Free\" button", "intent": "text", "n": 1, "include_images": false }
```

That is the shape to read candidate scores and `text_check.coverage` from when a text
request comes back wrong.

**Why JPEG is the default.** It is the format every consumer, print pipeline and legacy client
already accepts — the least surprising thing to hand back when nobody asked for a specific
encoding. The trade is alpha: JPEG has none, so `encode()` converts to RGB. A caller that needs
transparency must ask for `png` or `webp` explicitly, and the vectoriser path returns SVG.

### Response

```json
{
  "created": 1793000000,
  "data": [{ "b64_json": "…", "media_type": "image/jpeg", "revised_prompt": "…" }],
  "imotan": {
    "model": "…", "width": 1184, "height": 880, "steps": 4, "format": "jpeg",
    "intent": "text", "best_of": 2, "upscale": 2,
    "text_precision": "int8-wo",
    "prompt_expansion": { "status": "expanded", "used": true },
    "text_check": { "enabled": true, "expected": ["Get Started for Free"] },
    "stages_available": { "upscale": true, "vectorize": true },
    "selection": [{
      "index": 0, "selected_seed": 42, "score": 0.91,
      "text_check": { "coverage": 1.0, "extra": 0.0, "score": 1.0 },
      "candidates": [
        { "seed": 42, "score": 0.91, "technical": 0.55,
          "text_check": { "coverage": 1.0, "extra": 0.0, "score": 1.0 } },
        { "seed": 43, "score": 0.72, "technical": 0.61,
          "text_check": { "coverage": 1.0, "extra": 0.5, "score": 0.875 } }
      ],
      "stages": [{ "stage": "upscale", "output": "2368x1760" }], "notes": []
    }]
  }
}
```

### Text rendering

Quote every string you want rendered, and nothing else:

```json
{ "prompt": "A banner with a \"Get Started for Free\" button, bright energetic palette", "intent": "text" }
```

**By default the quoted string is drawn, not generated.** The model paints the artwork; the
CTA is set in a bundled font over the result, so the spelling comes from a font file rather
than from a sampler. `text_check.composed: true` says that happened.

Why it changed: measured, 4-step klein treats a quoted string as a hint rather than a contract.
A run quoting "Get Sorted for Free" returned `coverage` 0.75 on all four candidates — one word
of four substituted — and an earlier run quoting "Get Started for Free" came back reading "Get
Sturted". Changing the quoted word did not change the render. Across eight samples on two
prompts, zero were correct, so best-of-N was buying nothing and the pool is no longer spent on
it: composed requests draw `IMOTAN_BEST_OF` candidates for picture quality, not sixteen for
luck. `IMOTAN_COMPOSE_TEXT=0` restores the generate-and-verify path.

What the two paths guarantee is different, and the response says which one you got:

| | `composed: true` | `composed: false` |
|---|---|---|
| Spelling | structural — glyphs come from the font | probabilistic — OCR said it matched |
| Placement | fixed band, plate for contrast | wherever the model put it |
| Look | flat display type | klein's sculpted lettering |

- Quoted spans are the contract. `text_check.expected` echoes them back.
- A prompt that quotes lettering is never sent through prompt expansion. The expander is
  instructed to emit "no quotation marks", so a rewrite removes the exact markers that
  `build_prompt` and the scorer read: `intent` falls back to photo, the *only the text "…",
  no headline, sharp legible spelling* suffix is never appended, and the model invents
  lettering freely. That used to depend on whether the 1.5B rewrite passed validation, so
  the same prompt spelled correctly or not depending on the expander's mood. The response
  says when this happened: `prompt_expansion.status: "skipped"`, `reason: "quoted_text"`.
- For `intent: "text"` the shaping suffix stops being a compliment ("legible text") and
  becomes a constraint: *only the text "…", no other text, no additional words, no headline*.
  Unconstrained, klein invents a headline and misspells it.
- When OCR is available, best-of-N ranks text candidates on `coverage` versus `extra`
  (lettering it was never asked for) instead of on edge energy alone, so a clean frame
  outranks a sharper one with a garbled headline. `text_check.enabled` says whether that
  happened; `false` means the technical score alone decided.
- `coverage` is a **word-exact** measure, not a character edit distance. A string can differ
  by two characters and still be wrong ("get sturted for free" scores 0.750, not 0.947):
  the per-line character ratio is capped by the share of the expected words that actually
  appear, so a misspelt word cannot hide behind the length of the rest of the line.
  A correctly spelt string wrapped over two lines still scores 1.0 — the lines are also
  scored joined together, and words are matched against a budget so one line cannot claim
  the same word twice.
- `text_check.lines` is what the detector actually read, capped at 8 lines. Coverage alone
  cannot tell "the model cannot spell" from "the model wrote a more likely phrase instead",
  and those are different fixes — a different prompt versus a different pipeline. A measured
  four-candidate run on `"Get Sorted for Free"` scored 0.75 on every sample, i.e. one word of
  four; only the lines say which word was substituted.
- `intent: "text"` also spends the render budget on spelling. 4-step klein misspells
  per-sample, not per-prompt — the same seedless prompt yields "Sturted", "Sturtere",
  "Stated" — so the only font-free lever is width: `IMOTAN_TEXT_BEST_OF` (default `4`)
  candidates per image, within a hard ceiling of 16 renders per request. This
  is why a text request is slower than a photo request. Set `IMOTAN_TEXT_BEST_OF=1` to
  spend nothing on retries and take the first sample, or `IMOTAN_BEST_OF=1` to disable
  selection everywhere.
- Short strings win. Long unquoted copy is where 4-step klein fails, and that is a model
  limit no suffix fixes.
- Sampling raises the odds; it does not guarantee spelling. If a caller needs the string
  pixel-exact every time, composite the type over the render instead of asking the model
  to draw it — that is the only wrapped-guarantee path, at the cost of flat type rather
  than klein's sculpted lettering.

### Tests

```bash
python3 workers/test_ocr.py       # 12 cases, no GPU, no OCR engine, needs pillow only
python3 workers/test_download.py  # 7 cases, no GPU, no network
```

The suite pins `score_lines` against detector output that actually shipped — the
"Get Sturted for Free" button, the headline-and-button frame where both copies are
misspelt — plus the shapes that are easy to regress: a correct string split over two
lines, a correct string drawn twice, an invented headline, and no text at all. CI runs
it before building, so a scoring regression never reaches a container.

---

## Stages

### Super-resolution

Real-ESRGAN x4plus (BSD-3), loaded once and cached. Requested with `"upscale": 2|4`.

- Output is capped by `IMOTAN_UPSCALE_MAX_PIXELS`; exceeding it is an error, not a silent downscale.
- If weights or `spandrel` are missing, the stage is **skipped** and `notes` says why. It never
  substitutes Lanczos and calls it upscaling — "looks like 4K" is not what this sells.

### Vectorisation

`vtracer`, chosen via `"format": "svg"`. Two presets:

| Mode | Chosen when | Good for |
|---|---|---|
| `graphic` | logos, posters, icons, flat art | genuinely good — clean paths |
| `photo` | photographs, portraits, products | traces, but a traced photo is not vector art |

The response marks `"fidelity": "traced"` so a caller can tell. Do not advertise SVG as a
general capability; declare it only for the output classes where tracing holds up.

---

## What this is not ready for

**Content moderation is a blocklist, not a filter.** `pipeline.check_prompt` exists so internal
testing cannot produce obvious prohibited content. It is not moderation.

Before this endpoint is public — and before any marketplace listing — replace it with a real
classifier on both prompt and output. The FLUX.2 model card ships NSFW/CSAM filters in its
inference code and explicitly encourages deployers to run them; that is the intended baseline.

**Native 4K.** The model generates at 1MP; `upscale: 4` produces a 4K *file* through real
super-resolution, which is a different claim from native 4K synthesis. Describe it accordingly.

---

## Upgrading the model

Nothing above changes except the checkpoint:

1. Bake the new weights at a new `MODEL_PATH` in a new image tag.
2. Deploy as a second endpoint; keep this one live.
3. Shadow-test against the current endpoint on the 200-prompt eval set.
4. Cut traffic over; keep the old endpoint listed until the new one proves uptime.

---

## Cost

At $0.69/hr serverless and ~2s per candidate (4 steps, ~1MP), two candidates cost roughly
**$0.0008** per served image before encoding. Two hundred thousand images a month is ~$330 of
compute; upscaling adds ~$0.0002 at 2×.
