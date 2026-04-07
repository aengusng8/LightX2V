# LightX2V — Compact Project Knowledge

## What Is This Project?

LightX2V is a **lightweight video generation framework** built on top of **Wan2.2 diffusion models** (14B params). It supports text-to-video, image-to-video, and **pose-driven video animation** (wan-animate). The codebase uses custom CUDA kernels, FP8 quantization (GGUF), and SageAttention for efficient inference on consumer GPUs (e.g., RTX 4090, RTX 6000 Pro).

## Team & Repo

- **Upstream repo**: `ItsThanhTung/LightX2V` (origin)
- **Our fork**: `aengusng8/LightX2V`
- **Contributors**: itsthanhtung (Tung), hungnguyen2611 (Hung), aengusng8 (Duc Anh)

## Project Structure

```
lightx2v/                  # Core library
  models/
    runners/
      default_runner.py    # Base runner: run_main() loop, segment processing
      wan/
        wan_animate_runner.py  # Animate-specific: chunking, prepare_input, VAE encode
    schedulers/
      wan/scheduler.py     # Noise init (prepare_latents), diffusion solver (DPM)
    networks/wan/          # DiT transformer, adapters
  infer.py                 # Entry point: python -m lightx2v.infer

configs/wan22/             # JSON configs
  wan_animate.json         # Standard animate (no replace)
  wan_animate_replace_4090.json  # Replace mode (replace_flag: true, FP8 quantized)

scripts/wan22/
  run_wan22_animate.sh            # Standard animate inference
  run_wan22_animate_replace.sh    # Animate + relight (our best pipeline)
  run_wan22_animate_lora.sh       # LoRA variant
  run_wan22_animate_gradio.sh     # Gradio demo

tools/
  preprocess/
    preprocess_data.py     # CLI entry for preprocessing
    process_pipepline.py   # Main orchestrator: ProcessPipeline class
    lbm_relight_utils.py   # LBM relighting model loading & inference
    utils.py               # Composition, wavelet fusion, affine transforms
    sam_utils.py           # SAM2 segmentation (image & video predictors)
  convert/                 # GGUF conversion, FP8 quantization

DWT_IDWT/                  # Haar wavelet transform layers (for relight fusion)
MimicMotion/               # Reference repo: ICML 2025, proven long-video chunking
app/                       # Gradio web UI
```

## Wan-Animate Pipeline (Standard)

```
Reference Image + Driving Video
  -> Preprocess: extract pose, face, retarget
  -> Inference: DiT diffusion (4 steps, DPM solver)
  -> VAE decode -> output video
```

**Script**: `scripts/wan22/run_wan22_animate.sh`
**Config**: `configs/wan22/wan_animate.json`

## Wan-Animate + Relight Pipeline (Our Best)

### Why Relight?

When replacing a person in a video (e.g., studio photo -> outdoor scene), the reference person's lighting doesn't match the driving video. Without relighting, the result looks "pasted on". Relight harmonizes the lighting in preprocessing so the animation model sees a naturally-lit reference.

### Relight Base Model: LBM (Latent Bridge Matching)

- **Paper**: ICCV 2025, [github.com/gojasper/LBM](https://github.com/gojasper/LBM)
- **Weights**: `jasperai/LBM_relighting` on HuggingFace
- **Architecture**: VAE encoder -> Bridge Matching diffusion (1-4 steps) -> VAE decoder
- **Key property**: Bridge matching starts from source latent (not noise), so it works in 1-4 steps. Learns to produce natural relighting without explicit target lighting input.

### Full Relight Flow

```
1. COMPOSE: Affine-warp reference person onto driving video frame[0]
   - SAM2 segments person mask
   - Pose-based scale + translate alignment (neck anchor)
   - Alpha blend onto first frame

2. LBM RELIGHT: Harmonize lighting of composed image
   - VAE encode -> bridge matching (1-4 steps) -> VAE decode
   - Output has correct lighting but soft/blurry details (VAE bottleneck)

3. WAVELET FUSE (Haar DWT/IDWT):
   - Decompose both relighted and original into frequency bands
   - Take LL (low-freq color/lighting) from RELIGHTED image
   - Take LH, HL, HH (high-freq edges/textures) from ORIGINAL image
   - IDWT reconstruct -> correct lighting + sharp details

4. INVERSE AFFINE: Map fused result back to reference image space
   -> Output: reference_wavelet_merged.png

5. INFERENCE: Feed relighted reference to wan-animate (model unchanged)
   + src_pose.mp4, src_face.mp4, src_bg.mp4, src_mask.mp4
```

**Script**: `scripts/wan22/run_wan22_animate_replace.sh`
**Config**: `configs/wan22/wan_animate_replace_4090.json` (`replace_flag: true`)

### Key Relight Hyperparameters

| Param | Default | Meaning |
|-------|---------|---------|
| `--lbm_relight_steps` | 4 | Diffusion steps (more = better quality, slower) |
| `--lbm_fuse_mode` | wavelet | `wavelet` preserves details, `alpha` is simpler blend |
| `--lbm_wavelet_level` | 1 | DWT decomposition depth (1-2) |

## Current Long Video Problem: Chunking

### How Current Chunking Works

From `wan_animate_runner.py`:

```
target_video_length = 77 frames per chunk
refert_num = 1 frame overlap (!)
```

- **Sequential processing**: Chunk 0 -> Chunk 1 -> Chunk 2 -> ...
- **Minimal overlap**: Only 1 frame from previous chunk used as conditioning
- **Independent noise**: `scheduler.reset()` per chunk
- **Hard stitch**: Remove overlap frame, concatenate

### The Problem

**Autoregressive error accumulation**: Each chunk only gets 1 frame of context from the previous chunk. Errors compound:

```
Chunk 0: [good] --1 frame--> Chunk 1: [ok] --1 frame--> Chunk 2: [bad] --1 frame--> Chunk 3: [terrible]
```

- First 5s: good quality
- Each subsequent 5s: progressively worse
- By 20s: completely degraded

### Independent Chunking (No Overlap)

When chunks are generated independently (no autoregressive conditioning), each 5s segment has very good quality. BUT there's visible flickering at chunk boundaries since frames aren't temporally coherent across boundaries.

### Proven Solution: MimicMotion's Approach (ICML 2025)

Reference implementation in `MimicMotion/mimicmotion/pipelines/pipeline_mimicmotion.py`.

**Key differences from current approach:**

| | Current LightX2V | MimicMotion |
|---|---|---|
| Overlap | 1 frame | 6 frames |
| Processing | Sequential (chunk by chunk) | All chunks per denoising step |
| Blending | Hard cut | Triangular weighted averaging |
| Reference | Previous chunk's last frame | Frame 0 in EVERY chunk |
| Noise | Independent per chunk | Repeated pattern across all frames |

**MimicMotion's algorithm:**

```python
# Pre-compute chunk indices with large overlap
# Frame 0 (reference) included in EVERY chunk
indices = [[0, *range(i+1, min(i+tile_size, num_frames))]
           for i in range(0, num_frames-tile_size+1, tile_size-tile_overlap)]

# Triangular weight: low at edges, peak at center
weight = (arange(tile_size) + 0.5) * 2.0 / tile_size
weight = minimum(weight, 2 - weight)

# At EACH denoising step, process ALL chunks and blend
for timestep t in diffusion_steps:
    noise_pred = zeros(all_frames)
    noise_cnt = zeros(all_frames)
    for chunk_indices in all_chunks:
        pred = unet(latents[chunk_indices], t)
        noise_pred[chunk_indices] += pred * weight
        noise_cnt[chunk_indices] += weight
    noise_pred /= noise_cnt                    # weighted average
    latents = scheduler.step(noise_pred, t, latents)  # single update for ALL frames
```

**Why this works**: Instead of generating chunks sequentially and stitching, it processes ALL chunks at every denoising step and blends their noise predictions. Overlapping frames get soft-blended contributions from multiple chunks. No error accumulation.

### Implementation Plan

Adapt MimicMotion's strategy into `wan_animate_runner.py` and `default_runner.py`:

1. Increase overlap to 6-8 frames (new config param `tile_overlap`)
2. Include frame 0 (reference) in every chunk
3. Replace sequential `run_segment()` loop with per-step all-chunk blending
4. Use triangular weighting for overlap regions
5. Pre-generate noise for entire video, slice per chunk (progressive latent fusion)

Main files to modify:
- `lightx2v/models/runners/default_runner.py` — core loop
- `lightx2v/models/runners/wan/wan_animate_runner.py` — chunk logic
- `configs/wan22/wan_animate_replace_4090.json` — add tile_overlap config

---

## TODO

- [ ] Run current relight + wan-animate pipeline to get baseline results (quality & timing)
- [ ] Implement MimicMotion-style chunking (triangular blending, all-chunks-per-step)
- [ ] Benchmark long video (20s) quality with new chunking vs baseline
