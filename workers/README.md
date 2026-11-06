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
| **B. Network volume** | ~8GB | pull 8GB + one 13GB download into the mount | mounts the cache | 8GB |
| **C. Download every boot** | ~8GB | pull + 13GB download | **13GB again, every new host** | 8GB |

**Do not ship C.** With `active workers: 0`, every new host re-downloads the weights and your
first request of the day waits for all 13GB. It only looks acceptable after FlashBoot has
warmed a host once.

Choose **B** when the image push is the bottleneck (slow uplink) or when you want to iterate on
the Dockerfile without re-uploading weights each time:

1. Create a network volume (≥30GB) **in a data center that has 3090/A5000 stock**.
2. Mount it at `/models`.
3. Set `MODEL_PATH=/models/flux2-klein-4b` — the handler downloads into it on the first boot
   and every later worker mounts the populated cache (`_ensure_weights` in `handler.py`).
4. Pin the endpoint to that data center. A volume outside the GPUs' data center cannot mount.

`IMOTAN_ALLOW_DOWNLOAD=0` disables the runtime download and fails fast instead — use it on the
baked image so a mis-set path surfaces as an error rather than a 13GB surprise.

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
| `IMOTAN_DTYPE` | `int8` | `int8` = int8 weight-only; `bf16` = full precision |
| `IMOTAN_BEST_OF` | `2` | candidates per requested image, capped at 3 |
| `IMOTAN_FORMAT` | `jpeg` | default when the request omits `format` |
| `IMOTAN_QUALITY` | `90` | JPEG/WebP quality |
| `IMOTAN_UPSCALE_MODEL` | `/models/realesrgan/RealESRGAN_x4plus.pth` | baked |
| `IMOTAN_UPSCALE_MAX_PIXELS` | `16777216` | hard cap on upscaled output (~4K) |
| `IMOTAN_WORKER_TOKEN` | *(empty)* | if set, requires `Authorization: Bearer <token>` |

---

## Precision

`IMOTAN_DTYPE` controls how the transformer is loaded.

| Value | Weight footprint | Speed on 3090/A5000 | Notes |
|---|---|---|---|
| `int8` (default) | ~4GB (from ~8GB bf16) | usually **faster** | int8 weight-only via torchao |
| `bf16` | ~8GB | baseline | full precision |

**int8 weight-only, not fp8.** fp8 *compute* requires Ada or Hopper; on Ampere it is emulated
and slower than bf16. int8 weight-only dequantises inside the matmul, so it keeps bf16 compute
while moving half the weight bytes — and diffusion at 1MP is bandwidth-bound, so that is where
the win is.

Whole-pipeline VRAM at bf16 is ~13GB (transformer + text encoder + VAE). int8 typically lands
around 9–10GB, because the text encoder is not quantised by default. That leaves real headroom
on a 24GB card for best-of-2 and 2× upscaling.

The active precision is reported in every response as `imotan.precision` (`int8-wo` or `bf16`),
so you can confirm from the output what the worker actually loaded rather than trusting the env.

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
    "intent": "product", "best_of": 2, "upscale": 2,
    "stages_available": { "upscale": true, "vectorize": true },
    "selection": [{ "index": 0, "selected_seed": 42, "score": 0.71,
                    "candidates": [{ "seed": 42, "score": 0.71 }, { "seed": 43, "score": 0.58 }],
                    "stages": [{ "stage": "upscale", "output": "2368x1760" }], "notes": [] }]
  }
}
```

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
