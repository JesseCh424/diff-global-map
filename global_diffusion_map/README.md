# Global Diffusion Map (Scaffold)

This is a minimal scaffold to restart the `global_diffusion_map` project. It includes:

- `tools/` CLI wrappers for training and testing (MMDetection3D style)
- `plugin/` a lightweight plugin package with default runtime config and thin wrappers
- `plugin/configs/global_diffusion/minimal_example.py` sample config referencing the local plugin
- `scripts/` a sample shell script showing how to run

## Quick Start

- Install dependencies (example):
  - Create/activate env (see repo root docs or `maptracker_env.yml`).
  - Install MMCV/MMDetection/MMDetection3D compatible with PyTorch/CUDA.
- Run a dry command to verify imports:

```
python tools/train.py plugin/configs/global_diffusion/minimal_example.py \
  --work-dir work_dirs/minimal
```

Note: The example config is non-functional by design (placeholder `DummyModel`). Replace `model`/`data` with real definitions before training.

## Data Preparation (Argoverse2)

- Ensure AV2 dataset is mounted (e.g., `../dataset/av2` with `train/ val/ test/`).
- Generate map infos (newsplit) using the copied MapTracker converter:

```
bash scripts/gen_av2_infos.sh ../dataset/av2
```

Outputs: `../dataset/av2/av2_map_infos_{train,val}_newsplit.pkl`

## Structure

```
global_diffusion_map/
├── plugin/
│   ├── configs/
│   │   ├── _base_/default_runtime.py
│   │   └── global_diffusion/minimal_example.py
│   ├── core/apis/
│   │   ├── __init__.py
│   │   └── test.py
│   ├── datasets/builder.py
│   └── __init__.py
├── tools/
│   ├── test.py
│   └── train.py
├── scripts/
│   ├── gen_av2_infos.sh
│   └── train_minimal.sh
```

## Notes
- `plugin_dir` is set to `global_diffusion_map/plugin` in configs; the CLI dynamically imports it.
- The wrappers in `plugin/core/apis` forward to MMDetection3D APIs to avoid drift.
- Add your actual models, datasets, and pipelines under `plugin/` and update configs accordingly.
- Aggregated products live under `maptracker/work_dirs/aggregated_scene_vectors/<split>` (predictions) and `maptracker/work_dirs/agg_gt_vector/<split>` (GT). Keep them out of git; regenerate as needed via the commands below.

## Guidance Training (PolyDiffuse)

Start the vector‑only guidance stage using the provided script. Decide caps (M points per polyline, num_queries instances) and write them to a stats JSON the loaders will read.

1) Decide caps (example forces M=30, num_queries=64):

```
printf '{"M":30,"num_queries":64,"class_budget":{"0":11,"1":30,"2":23}}' > global_diffusion_map/work_dirs/av2_stats.json
```

2) Launch guidance on 4 GPUs (official‑aligned params):

```
CUDA_VISIBLE_DEVICES=0,1,2,8 \
AV2_STATS_JSON=global_diffusion_map/work_dirs/av2_stats.json \
AV2_M_OVERRIDE=30 AV2_NUM_QUERIES_OVERRIDE=64 \
BATCH=128 BATCHGPU=32 LR=2e-4 TICK=1 SNAP=5 DURATION=0.1 \
bash -x global_diffusion_map/scripts/dist_train_guide.sh
```

Outputs:
- Run directory: `global_diffusion_map/work_dirs/guide/`
- Snapshot: `network-snapshot.pth` (use as `--guide_ckpt` when starting denoise)

## Denoise Training (PolyDiffuse)

Train the raster‑conditioned denoising stage using the guidance snapshot. Ensure the caps JSON matches guidance (M=30, num_queries=64).

Current training defaults (AV2 old split)
- Targets (vectors): unsimplified static GT under `maptracker/work_dirs/static_gt_vector/av2_oldsplit/<split>/<scene>.pkl`.
- Conditioning (raster): 10 rendered GT (no augmentation) under `maptracker/work_dirs/rendered_gt/av2_oldsplit/<split>/<scene>/10_render_gt.png`.

Prerequisites
- Guidance snapshot: `global_diffusion_map/work_dirs/guide/.../network-snapshot.pth`
- Pretrained MapTR/MapTracker ckpt (image backbone + encoder init):
  - `maptracker/work_dirs/pretrained_ckpts/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth`
- Caps JSON: `global_diffusion_map/work_dirs/av2_stats.json` with `M=30`, `num_queries=64`

Launch (4 GPUs; global batch 48 = 4×12)

```
CUDA_VISIBLE_DEVICES=0,1,2,8 \
AV2_STATS_JSON=global_diffusion_map/work_dirs/av2_stats.json \
BATCH=48 BATCHGPU=12 LR=6e-4 TICK=1 SNAP=10 DUMP=500 WORKERS=4 DURATION=5 \
P_MEAN=-0.5 P_STD=1.5 SIG_DATA=1.0 LAMBDA_DIR=2e-3 \
GUIDE_CKPT=global_diffusion_map/work_dirs/guide/<run>/network-snapshot.pth \
PRETRAINED_CKPT=maptracker/work_dirs/pretrained_ckpts/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth \
bash -x global_diffusion_map/scripts/train.sh
```

Notes
- The dataset and model use `M/num_queries` from `AV2_STATS_JSON`; the decoder is configured with `num_verts=M` for per‑poly decoding.
- Conditioning rasters are letterboxed to a fixed canvas `(1024, 1024)` after optional downscale (`cond_max_side=1024`) to ensure batchable shapes.
- To enforce 10 as condition in your config, set `data.train.use_condition='10'` (and keep the same letterbox options in inference for parity).

## Proposal Conditioning (Option B)

We add aggregated vectors (MapTracker proposals) as a vector‑condition to the denoiser. The loss stays GT‑only (no `z_init`), avoiding drift toward external proposal statistics.

### Implementation

- ProposalEncoder + FiLM fusion in MapTR head:
  - `poly-diffuse/projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py`
  - Per‑vertex (x,y) → per‑query feature with masked pooling; add class embedding; produce `gamma,beta` to modulate the per‑vertex tokens (FiLM or additive bias).
- Detector plumbs proposals into the head:
  - `poly-diffuse/projects/mmdet3d_plugin/maptr/detectors/maptr.py`
- Training loop passes proposals when available and logs a small stat:
  - `poly-diffuse/src/training_loops/denoise_training_loop.py`
- Inference forwards proposals in `global_diffusion_map/tools/infer_av2.py`.

### Data, Caps and Alignment

- Caps from `global_diffusion_map/work_dirs/av2_stats.json` (e.g., `M=30`, `num_queries=64`, with `class_budget`).
- Training uses packed proposals (`global_diffusion_map/work_dirs/packed_proposals/<split>/<scene>.npz`). They are auto‑prepared at launch by `global_diffusion_map/tools/run_train.py` if the directories exist.
- Inference performs class‑aware matching + permutation inline (use `--stable-permutation`) before ProposalEncoder.
- Conditioning rasters (10/11) are letterboxed with identical policy in training and inference (downscale to `cond_max_side`, paste top‑left to `cond_fixed_size`).

### How to Use

- Train (denoise):
  - Keep GT vectors as targets; use 11 (or 10) as raster condition; proposals are vector‑condition.
  - Ensure `work_dirs/av2_stats.json` matches caps; let `tools/run_train.py` auto‑prepare packed proposals for train/val.
- Infer:
  - Provide aggregated vectors dir (`--agg-pred-dir`), semantic/10/11 roots, and enable `--stable-permutation` when blending proposal + guide.

### Quick Viz & Quant Checks

- Raster alignment (pre/post letterbox):
  ```
  python -u global_diffusion_map/tools/check_cond_alignment.py \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --bounds-pred maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid \
    --cond-filename 10_render_gt.png \
    --scenes <SCENE...> \
    --out-root global_diffusion_map/viz/cond_align \
    --cond-max-side 1024 --cond-fixed-size 1024 1024
  ```
- Proposal condition effectiveness (loss delta with/without proposals):
  ```
  python -u global_diffusion_map/tools/eval_proposal_encoder.py \
    --config global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_official_base.py \
    --ckpt <denoise_snapshot.pth> \
    --guide-ckpt global_diffusion_map/ckpts/guide/network-snapshot_m30q64.pth \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --proposal-root global_diffusion_map/work_dirs/packed_proposals/av2_oldsplit/val \
    --batches 4
  ```
- Proposal conditioning strength overlay (Top‑K by |gamma−1|):
  ```
  python -u global_diffusion_map/tools/viz_proposal_condition.py \
    --config global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_official_base.py \
    --ckpt <denoise_snapshot.pth> \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --proposal-root global_diffusion_map/work_dirs/packed_proposals/av2_oldsplit/val \
    --scenes <SCENE1> <SCENE2> \
    --out-root global_diffusion_map/viz/prop_cond \
    --cond-filename 10_render_gt.png \
    --cond-max-side 1024 --cond-fixed-size 1024 1024 \
    --topk 10
  ```

### Tips

- To switch training condition to `10` (un‑augmented render), set `data.train.use_condition='10'` in config; keep the same letterbox options in inference for parity.

## Data Preparation: Aggregated MapTracker HD Maps

The global diffusion model uses **aggregated per-scene MapTracker predictions** as upstream input for refinement. This section describes how to prepare the input data.

### Overview

The diffusion model refines global maps by taking aggregated local predictions from MapTracker and improving their quality, consistency, and completeness. The preparation pipeline consists of:

1. **MapTracker Inference**: Generate frame-by-frame local map predictions
2. **Global ID Assignment**: Track and assign global IDs across frames
3. **Scene Aggregation**: Merge predictions per scene into global vectors
4. **Visualization** (optional): Generate per-scene merged images

### Prerequisites

- Completed MapTracker training and inference (see `../maptracker/README.md`)
- MapTracker config file (e.g., `maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py`)
- Inference results: `submission_vector.json` in work_dirs

### Step 1: Assign Global IDs Across Frames

Use MapTracker's tracking script to compute global instance IDs across consecutive frames:

```bash
cd /workspace/mrt/maptracker

# For AV2 old split
python tools/tracking/prepare_pred_tracks.py \
    plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --result_path work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/submission_vector.json \
    --cons_frames 5 \
    --thr 0.4

# For AV2 new split
python tools/tracking/prepare_pred_tracks.py \
    plugin/configs/maptracker/av2_newsplit/maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune.py \
    --result_path work_dirs/maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune/submission_vector.json \
    --cons_frames 5 \
    --thr 0.4
```

**Output**: `pos_predictions_5.pkl` containing frame-level predictions with global IDs

**Parameters**:
- `--cons_frames`: Number of consecutive frames to match (typically 5)
- `--thr`: Score threshold to filter low-confidence predictions (typically 0.4)

### Train Split Inference (AV2 old split)

When running inference on the AV2 old split TRAIN set (to later aggregate per‑scene inputs for training), we save outputs to a dedicated folder to avoid mixing with val:

```
# 4 GPUs on devices 4–7; save semantic masks in submission
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash maptracker/tools/dist_test.sh \
  maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
  maptracker/work_dirs/pretrained_ckpts/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth \
  4 \
  --work-dir maptracker/work_dirs/maptracker_av2_oldsplit_train_infer \
  --eval --eval-options save_semantic=True \
  --cfg-options data.test.ann_file=./datasets/av2/av2_map_infos_train.pkl data.test.seq_split_num=1
```

Outputs (train split): `maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/`
- `submission_vector.json`
- `pos_predictions.pkl` (frame metas)

You can then run tracking and scene aggregation on these train outputs the same way as for val.

### Step 2: Export Per-Scene Aggregated Vectors

Group predictions by scene and merge vectors using MapTracker's built-in merging logic:

```bash
cd /workspace/mrt/maptracker

# For AV2 old split
python tools/tracking/export_scene_grouped.py \
    plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --pred-pkl work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl \
    --out-dir work_dirs/aggregated_scene_vectors/av2_oldsplit \
    --simplify 0.5 \
    --overwrite

# For AV2 new split
python tools/tracking/export_scene_grouped.py \
    plugin/configs/maptracker/av2_newsplit/maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune.py \
    --pred-pkl work_dirs/maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl \
    --out-dir work_dirs/aggregated_scene_vectors/av2_newsplit \
    --simplify 0.5 \
    --overwrite
```

**Output**: One pickle file per scene (`<scene_name>.pkl`) in the output directory

**Data Format**:
```python
# Each scene pickle contains merged vectors and the ego trajectory:
{
    0: [np.ndarray],   # crossings (Nx2)
    1: [np.ndarray],   # dividers (Nx2)
    2: [np.ndarray],   # boundaries (Nx2)
    "car_trajectory": [
        {"frame_idx": 0, "center": [x, y], "yaw_deg": float},
        ...
    ],
}
# All coordinates are expressed in meters in the last frame's ego frame.
```

**Parameters**:
- `--simplify`: Line simplification tolerance in meters (0.5 recommended)
- `--overwrite`: Overwrite existing outputs

### Step 3: Crop Static GT with Union ROI

After aggregation, derive the static ground-truth labels by clipping the AV2 log
map with the union of the prediction ROIs. The helper below calls
`AV2MapExtractor` to obtain the full map once, unions all per-frame ROIs in the
last-frame ego frame, and crops every vector accordingly:

```bash
cd /workspace/mrt/maptracker

SCENE=0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2
python tools/tracking/crop_static_gt_from_predictions.py \
    plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --pred-pkl work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl \
    --scene-id "$SCENE" \
    --out-dir work_dirs/static_gt_vector/av2_oldsplit
```

The output lives under `work_dirs/static_gt_vector/<split>/<scene>.pkl` and
matches the aggregated schema (labels 0/1/2 plus `"car_trajectory"`). Vectors
are strictly masked by the union ROI and expressed in the last-frame ego frame.
The cropper uses Shapely bounds in the correct order `(minx, miny, maxx, maxy)`
to size the extractor ROI, which fixes tall-scene truncation.

### Step 4: Aggregate Ground-Truth (GT) Scenes

To evaluate or train against ground truth, aggregate the MapTracker GT tracks with the same logic used for predictions.

```bash
cd /workspace/mrt/maptracker

python tools/tracking/export_gt_scene_grouped.py \
    plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --gt-tracks datasets/av2/av2_map_infos_val_gt_tracks.pkl \
    --out-dir work_dirs/agg_gt_vector/av2_oldsplit \
    --simplify 0.5 \
    --overwrite
```

Tips:
- Launch long runs in the background if desired:
  ```bash
  nohup python tools/tracking/export_gt_scene_grouped.py ... \
       > work_dirs/agg_gt_vector/av2_oldsplit.log 2>&1 &
  ```
- Each GT scene pickle mirrors the prediction format (`0/1/2` arrays + `"car_trajectory"`), so downstream code can swap between sources transparently.

### Step 5: One‑Shot Comparison Visualization (01/04/05)

To compare aggregated predictions (01), aggregated GT (04) and static GT crops (05)
on identical bounds per scene:

```bash
cd /workspace/mrt/maptracker

python tools/visualization/viz_scene_compare.py \
  --aggregated-pred work_dirs/aggregated_scene_vectors/av2_oldsplit \
  --aggregated-gt   work_dirs/agg_gt_vector/av2_oldsplit \
  --static-gt       work_dirs/static_gt_vector/av2_oldsplit \
  --out-root        viz/av2_old \
  --bounds-pkl work_dirs/aggregated_scene_vectors/av2_oldsplit/{scene}.pkl \
               work_dirs/agg_gt_vector/av2_oldsplit/{scene}.pkl \
  --dpi 60
```

### Train Split (AV2 old) — Full Preparation

We prepare the train split analogously to val, and keep outputs separate.

1) Inference and tracking on train (saves to a dedicated work dir):

```
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash maptracker/tools/dist_test.sh \
  maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
  maptracker/work_dirs/pretrained_ckpts/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth \
  4 \
  --work-dir maptracker/work_dirs/maptracker_av2_oldsplit_train_infer \
  --eval --eval-options save_semantic=True \
  --cfg-options data.test.ann_file=./datasets/av2/av2_map_infos_train.pkl data.test.seq_split_num=1

python maptracker/tools/tracking/prepare_pred_tracks.py \
  maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
  --result_path maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/submission_vector.json \
  --cons_frames 5 --thr 0.4
```

2) Aggregate per‑scene predictions (train):

```
python maptracker/tools/tracking/export_scene_grouped.py \
  maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
  --pred-pkl maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/pos_predictions_5.pkl \
  --out-dir maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
  --simplify 0.5 --overwrite
```

3) Crop static GT per scene using union ROI (train):

```
for s in $(ls -1 maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train/*.pkl | xargs -n1 basename | sed 's/.pkl$//'); do \
  python maptracker/tools/tracking/crop_static_gt_from_predictions.py \
    maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --pred-pkl maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/pos_predictions_5.pkl \
    --scene-id "$s" \
    --out-dir maptracker/work_dirs/static_gt_vector/av2_oldsplit/train \
    --aggregated-pred-dir maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
    --map-info ./datasets/av2/av2_map_infos_train.pkl; \
done
```

4) Simplify GT static vectors (train), pkls only:

```
python maptracker/tools/tracking/simplify_static_gt.py \
  --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/train \
  --aggregated-pred maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
  --aggregated-gt   maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
  --out-pkl-root    maptracker/work_dirs/static_gt_vector_simp/av2_oldsplit/train \
  --no-png --no-2hop --angle-tiny-deg 4.0 --angle-deg 10.0 --dist-eps 1.5 \
  --pair-merge-eps 2.0 --pair-merge-angle-deg 60.0 --dpi 60 \
  --scenes $(ls -1 maptracker/work_dirs/static_gt_vector/av2_oldsplit/train/*.pkl | xargs -n1 basename | sed 's/.pkl$//')
```

Outputs
- Aggregated pred (train): `maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train` (694)
- Static GT (train): `maptracker/work_dirs/static_gt_vector/av2_oldsplit/train` (694)
- Simplified GT static (train): `maptracker/work_dirs/static_gt_vector_simp/av2_oldsplit/train` (694)

Note: the simplifier supports `--no-png` to skip rendering; we keep only pkls for training.

## Aggregated Semantic Raster (08)

We generate a per‑scene semantic raster aligned with 01/04/05 using a sparse‑majority vote
across per‑frame masks, followed by a uniform big‑splat and continuity‑aware overlap (cyan) between
ped‑crossing and boundary. This serves as the conditioning raster for inference.

Batch for val split (semantic 08 only):

```bash
# Outputs -> maptracker/work_dirs/semantic/<scene>/08_agg_semantic.png
bash maptracker/tools/tracking/batch_gen_semantic_val.sh \
  > maptracker/work_dirs/semantic/batch_gen_val.log 2>&1 &

# Notes: OVERWRITE=1 by default; pass SCENE_FILE to process a subset efficiently.
```

Rendered‑GT parity (same 08 method, drawn from static GT vectors):

```bash
# Outputs -> maptracker/work_dirs/rendered_gt/av2_oldsplit/val/<scene>/08_agg_semantic.png
CONNECT_M=2.0 \
bash maptracker/tools/tracking/batch_gen_rendered_gt_08.sh \
  > maptracker/work_dirs/rendered_gt/av2_oldsplit/val/batch_gen_08.log 2>&1 &
```

Default parameters (08): match 05 canvas (`--match-05`), min‑votes=1, min‑area=8, no denoise, inner ring 0.6 m,
boundary proximity 1.0 m, density k=2/min=5, close=1 px, connect circle 2.0 m, thin=1 px,
uniform splat width 0.5 m.

## Augmented Rendered GT (11)

We produce a fast, realistic conditioning raster from the rendered GT canvas by adding unit‑scale outline jaggedness and sparse erosion/bulge “bubbles”. This mirrors artifacts seen in aggregated semantics while keeping perfect canvas/bounds alignment with 05/08.

- Script: `maptracker/tools/tracking/augment_rendered_gt_realistic.py` (fast‑only; ROI crop + downscaled SDF; applies to ped, boundary, and divider).
- Output: per‑scene `11_gt_aug.png` under the same folder as `10_render_gt.png`.
- Defaults:
  - Unit/outline: `--unit-px 20`, `--noise-amp-units 1.0` (divider 0.7), `--noise-sigma-units 1.2`, `--noise-block-units 2.0`
  - ROI/SDF: `--roi-pad-units 2.0`, `--sdf-downscale 0.33`
  - Bubbles: `--bubble-prob 0.002`, `--bubbles-per-mpx 80`, `--bubble-rmin-units 0.3`, `--bubble-rmax-units 1.0`, `--bubble-erosion-frac 0.9`

Example (val, five scenes):

```bash
python maptracker/tools/tracking/augment_rendered_gt_realistic.py \
  --val-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
  --scenes 0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2 0b86f508-5df9-4a46-bc59-5b9536dbde9f \
           0b5142c1-420b-3fea-9e98-b87327ae22c6 0bae3b5e-417d-3b03-abaa-806b433233b8 \
           0c3bad78-9f1e-395d-a376-2eb7499229fd
```

Notes:
- 11 preserves 05/08 canvas and overlap color (cyan for ped×boundary).
- Typical runtime: ~30–90s per tall scene; tune aggressiveness via `--unit-px`, `--noise-amp-units`, or bubble parameters if needed.

## Training Dataset Prep (AV2 old): 10/11/08

Goal: produce per‑scene 10_render_gt.png, 11_gt_aug.png, and 08_agg_semantic.png for the train split, with canvases aligned to 05/08 like the val split.

Inputs
- Static GT (cropped): `maptracker/work_dirs/static_gt_vector/av2_oldsplit/train/<scene>.pkl`
- Aggregated pred vectors (bounds): `maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train/<scene>.pkl`
- Aggregated GT vectors (optional bounds): `maptracker/work_dirs/agg_gt_vector/av2_oldsplit/train/<scene>.pkl`
- Train submission + pos pkl for semantics: `maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/{submission_vector.json,pos_predictions_5.pkl}`

Step 0 — Generate 05 for train (only 05)

```bash
TRAIN_STATIC=maptracker/work_dirs/static_gt_vector/av2_oldsplit/train
TRAIN_LIST=/tmp/train_scenes.txt
ls -1 "$TRAIN_STATIC"/*.pkl | xargs -n1 basename | sed 's/\.pkl$//' > "$TRAIN_LIST"

python maptracker/tools/visualization/viz_scene_compare.py \
  --aggregated-pred maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
  --aggregated-gt   maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
  --static-gt       "$TRAIN_STATIC" \
  --out-root        maptracker/viz/av2_old/train \
  --scene-list      "$TRAIN_LIST" \
  --bounds-pkl      maptracker/work_dirs/static_gt_vector/av2_oldsplit/train/{scene}.pkl \
  --only 05 --dpi 60
```

Step 1 — Aggregate train semantics to 08 (aligned to 05)

```bash
CONFIG=maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py
SUB=maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/submission_vector.json
POS=maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/pos_predictions_5.pkl
OUT=maptracker/work_dirs/semantic/train

python -u maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
  "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
  --out-dir "$OUT" --scene-list "$TRAIN_LIST" \
  --png --match-05 --min-votes 1 --min-area 8 --no-denoise \
  --thickness06-px 2 \
  --overlap-ring-mode inner --overlap-edge-m 0.6 \
  --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 \
  --overlap-close-px 1 --overlap-boost-m 0.4 \
  --overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 \
  --overlap08-connect-m 2.0 --overlap08-thin-px 1 \
  --splat08-m 0.5 --splat08-block 1 \
  --viz-root maptracker/viz/av2_old/train
```

Step 2 — Render 10 for train (align to 08 shape; bounds from aggregated pred + GT if available)

```bash
OUT10=maptracker/work_dirs/rendered_gt/av2_oldsplit/train
mkdir -p "$OUT10"
while read -r s; do
  python -u maptracker/tools/tracking/render_gt_to_10.py \
    --static-root     maptracker/work_dirs/static_gt_vector/av2_oldsplit/train \
    --aggregated-pred maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
    --aggregated-gt   maptracker/work_dirs/agg_gt_vector/av2_oldsplit/train \
    --semantic-root   maptracker/work_dirs/semantic/train \
    --out-root        "$OUT10" \
    --scenes "$s"
done < "$TRAIN_LIST"
```

Step 3 — Augment to 11 for train (fast‑only)

```bash
while read -r s; do
  python maptracker/tools/tracking/augment_rendered_gt_realistic.py \
    --val-root maptracker/work_dirs/rendered_gt/av2_oldsplit/train \
    --scenes "$s"
done < "$TRAIN_LIST"
```

Outputs
- 08: `maptracker/work_dirs/semantic/train/<scene>/08_agg_semantic.png`
- 10: `maptracker/work_dirs/rendered_gt/av2_oldsplit/train/<scene>/10_render_gt.png`
- 11: `maptracker/work_dirs/rendered_gt/av2_oldsplit/train/<scene>/11_gt_aug.png`

Notes
- The 10 renderer uses aggregated pred/gt for bounds and 08 (or 05 fallback) for pixel shape, matching val.
- If train aggregated GT is missing for some scenes, 10 still succeeds using aggregated pred and logs an informational warning.

## Selective Comparison Rendering

`maptracker/tools/visualization/viz_scene_compare.py` can render only specified panels via:

```
--only {01,04,05,08}
```

Example to generate 01 and 05 only for a scene list (train):

```
python maptracker/tools/visualization/viz_scene_compare.py \
  --aggregated-pred maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
  --aggregated-gt   maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
  --static-gt       maptracker/work_dirs/static_gt_vector/av2_oldsplit/train \
  --out-root        maptracker/viz/av2_old/train \
  --scene-list      /tmp/train_scenes.txt \
  --bounds-pkl maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train/{scene}.pkl \
               maptracker/work_dirs/static_gt_vector/av2_oldsplit/train/{scene}.pkl \
  --only 01 05 --dpi 60
```

## Next Steps

- Generate augmented rendered GT static maps (same canvas/bounds as 05; preserve overlap coloring) to approximate aggregated semantics for training conditions.
- Integrate PolyDiffuse (hd‑mapping branch) as the MVP denoiser:
  - Targets: simplified GT vectors; Conditioning: augmented rendered GT rasters (train) / aggregated semantic rasters (08) at inference.
  - Fixed points per polyline (M) with padding/masks.


Outputs per scene folder under `viz/av2_old/<scene>/`:
- `01_agg_pred_direct.png`
- `04_agg_gt_direct.png`
- `05_static_gt.png`

All three share identical axis limits derived from the aggregated sources.

### Batch Crop + Viz (150 scenes, old split)

Run the full static GT crop for all scenes present in the aggregated prediction
directory, then render 01/04/05 together:

```bash
cd /workspace/mrt/maptracker

PRED_PKL=work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl
AGG_PRED_DIR=work_dirs/aggregated_scene_vectors/av2_oldsplit
OUT_STATIC=work_dirs/static_gt_vector/av2_oldsplit

nohup bash -lc '
  for p in ${AGG_PRED_DIR}/*.pkl; do 
    scene=$(basename "$p" .pkl);
    echo "[crop] $scene";
    python tools/tracking/crop_static_gt_from_predictions.py \
      plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
      --pred-pkl "$PRED_PKL" \
      --scene-id "$scene" \
      --out-dir "$OUT_STATIC" \
      --aggregated-pred-dir "$AGG_PRED_DIR"; 
  done;
  echo "[viz] rendering 01/04/05...";
  python tools/visualization/viz_scene_compare.py \
    --aggregated-pred "$AGG_PRED_DIR" \
    --aggregated-gt   work_dirs/agg_gt_vector/av2_oldsplit \
    --static-gt       "$OUT_STATIC" \
    --out-root        viz/av2_old \
    --bounds-pkl ${AGG_PRED_DIR}/{scene}.pkl work_dirs/agg_gt_vector/av2_oldsplit/{scene}.pkl \
    --dpi 60;
' > "$OUT_STATIC"/batch_crop_and_viz.log 2>&1 &
```

Monitor logs at `work_dirs/static_gt_vector/av2_oldsplit/batch_crop_and_viz.log`.

You can inspect the results with the official MapTracker viewer (reconstructs from `pos_predictions_5.pkl`) or the lightweight helper that reads the aggregated pickle directly.

**Official MapTracker visualization** (matches the training team’s PNG output):

```bash
cd /workspace/mrt/maptracker

python tools/visualization/vis_global.py \
    plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
    --data_path work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl \
    --out_dir viz/av2_old/scene_merged_png \
    --option vis-pred \
    --per_frame_result 0 \
    --simplify 0.5 \
    --dpi 20 \
    --overwrite 1 \
    --bounds_pkl \
      work_dirs/aggregated_scene_vectors/av2_oldsplit/<scene_name>.pkl \
      work_dirs/agg_gt_vector/av2_oldsplit/<scene_name>.pkl
```

`--bounds_pkl` derives shared axis limits from the aggregated pickles so the ROI (and car sprite) stay aligned between prediction and GT renders without hard-coding coordinates. Reuse the same flag for `--option vis-gt`.

**Static GT viewer** (renders the `05_static_gt.png` companion using the same
canvas and car trajectory):

```bash
python tools/visualization/render_static_maptracker_style.py \
    --static-root work_dirs/static_gt_vector/av2_oldsplit \
    --scene-list tmp_scene.txt \
    --out-dir viz/av2_old \
    --bounds-pkl \
      work_dirs/aggregated_scene_vectors/av2_oldsplit/<scene_name>.pkl \
      work_dirs/agg_gt_vector/av2_oldsplit/<scene_name>.pkl
```

This lighter renderer reads the union-cropped GT directly and plots the full car
trajectory, making it easy to compare against `01`–`04` without replaying
per-frame predictions.

## Semantic Aggregation (08)

We provide a sparse‑majority semantic aggregator aligned to 01/04/05 axes, with
continuity‑aware overlap (cyan) where ped‑crossing (blue) edges meet boundary (green).

- 06 (optional): base aggregation for continuity (not saved by default; `--save-06`)
- 08_agg_semantic.png: rendered class map with overlap continuity:
  - 06 continuity overlap: ped edge ring (inner|outer) near boundary by Euclidean distance, validated by boundary density, small closing, clipped to ped.
  - 08 post‑connect: circular closing in meters (default `overlap08_connect_m=2.0`), optional thinning, then final cyan boost.

Helper scripts:
- Per‑scene/interactive: `maptracker/tools/tracking/aggregate_semantic_scene_sparse.py`
- Batch over val split (150 scenes): `maptracker/tools/tracking/batch_gen_semantic_val.sh`

Defaults (tunable):
- `--match-05 --min-votes 1 --min-area 8 --no-denoise --thickness06-px 2 --splat08-m 0.5 --splat08-block 1`
- Overlap continuity: `--overlap-ring-mode inner --overlap-edge-m 0.6 --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 --overlap-close-px 1 --overlap-boost-m 0.4`
- 08 post‑connect: `--overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 --overlap08-connect-m 2.0 --overlap08-thin-px 1`

Batch over rendered GT (same 08 method for parity):

```
CONNECT_M=2.0 \
SCENE_FILE=scenes_val.txt \
bash maptracker/tools/tracking/batch_gen_rendered_gt_08.sh \
  > maptracker/work_dirs/rendered_gt/av2_oldsplit/val/batch_gen_08.log 2>&1 &
```

The script builds one process and reuses parsed inputs via `--scene-list` to avoid repeatedly loading `submission_vector.json`.

Outputs are written under `maptracker/work_dirs/semantic/<scene>/`.

You can also copy precomputed 08 into `viz/` using `viz_scene_compare.py --semantic-root maptracker/work_dirs/semantic`.

### Performance

- Use `--scene-list` to process many scenes in a single aggregator process and avoid re‑parsing submission_vector.json per scene.
- The batch scripts (`batch_gen_semantic_val.sh`, `batch_gen_rendered_gt_08.sh`) build a scene list and call the aggregator once; they skip scenes that already have 08.

### Expected Directory Structure

After completing data preparation:

```
maptracker/work_dirs/
├── aggregated_scene_vectors/
│   ├── av2_oldsplit/
│   │   ├── scene_001.pkl
│   │   ├── scene_002.pkl
│   │   └── ...
│   └── av2_newsplit/
│       ├── scene_001.pkl
│       └── ...
├── agg_gt_vector/
│   ├── av2_oldsplit/
│   │   ├── scene_001.pkl
│   │   └── ...
│   └── av2_newsplit/
│       ├── scene_001.pkl
│       └── ...
└── maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/
    ├── submission_vector.json         # From MapTracker inference
    └── pos_predictions_5.pkl          # From Step 1 (global ID assignment)
```

### Preparing Ground-Truth Scenes for Training

The diffusion model is trained against *cropped* ground-truth scenes rather than merged GT polylines. The workflow is:

1. Run MapTracker inference + aggregation on **both train and val** splits to obtain per-scene prediction pickles. These define the union ROI that the model will see at training time.
2. Use the union ROI to crop the raw GT tracks (`av2_map_infos_{train,val}_gt_tracks.pkl`) into per-scene `gt_scene` pickles. Each `gt_scene` should retain the original GT vectors within the ROI (no further merging).
3. Store the cropped GT in `work_dirs/gt_scene/{train,val}/<scene>.pkl` (or a similar layout). These act as labels for the diffusion model while the aggregated prediction pickles serve as conditioning inputs.

This separation (prediction aggregation + GT cropping) ensures the model learns from the authentic map geometry while still conditioning on realistic MapTracker predictions.

### Simplified GT Static Vectors (for training labels)

To reduce the gap between high‑fidelity GT geometry and agg_pred complexity, we simplify GT vectors before resampling for diffusion training while keeping 05 renders perfect for conditioning.

- Script: `maptracker/tools/tracking/simplify_static_gt.py`
- Simplifies per‑scene cropped static GT using tiny‑turn drop and pair‑merge of close consecutive points (guard corners).
- Saves per‑scene simplified pkls and renders `09_static_gt_simp.png` aligned to 05 axes.

Canonical command (val split, 150 scenes):

```
python maptracker/tools/tracking/simplify_static_gt.py \
  --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
  --aggregated-pred maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit \
  --aggregated-gt   maptracker/work_dirs/agg_gt_vector/av2_oldsplit \
  --out-pkl-root    maptracker/work_dirs/static_gt_vector_simp/av2_oldsplit/val \
  --viz-root        maptracker/viz/av2_old \
  --png-name        09_static_gt_simp.png \
  --no-2hop --angle-tiny-deg 4.0 --angle-deg 10.0 --dist-eps 1.5 \
  --pair-merge-eps 2.0 --pair-merge-angle-deg 60.0 --dpi 60 \
  --scenes $(ls maptracker/work_dirs/static_gt_vector/av2_oldsplit/val/*.pkl | xargs -n1 basename | sed 's/.pkl$//')
```

Outputs:
- Simplified vectors (pkls): `maptracker/work_dirs/static_gt_vector_simp/av2_oldsplit/val/<scene>.pkl`
- Previews: `maptracker/viz/av2_old/<scene>/09_static_gt_simp.png`

Reserved folder for rendered GT rasters (if needed later): `maptracker/work_dirs/rendered_gt/av2_oldsplit/val/`.

### Usage in Diffusion Model

The aggregated scene vectors will be used as:
1. **Input condition**: Noisy/incomplete predictions to be refined
2. **Training data**: Ground truth comes from static city maps or manual annotations
3. **Evaluation**: Compare refined outputs against aggregated predictions

See the main repository `AGENTS.md` and `AGGREGATION_WORKFLOW.md` for the complete pipeline overview.

### Merging Strategy

The aggregation uses MapTracker's native merging functions:
- **Crossings (label 0)**: Convex hull-based merging
- **Dividers (label 1)**: Interpolation-based merging
- **Boundaries (label 2)**: Interpolation-based merging

All merging logic is from `maptracker/tools/visualization/vis_global.py`

### Semantic 08 aggregation (aligned to 05)

Use `maptracker/tools/tracking/aggregate_semantic_scene_sparse.py` with `--match-05` and split‑aware bounds from aggregated vectors to ensure the 08 canvas matches 05/01. Enable cyan overlap (ped×boundary) and set a reasonable thickness.

Recommended defaults (medium thickness):
- `--thickness06-px 2`
- `--overlap-08-from-06 --overlap-ring-mode inner --overlap-edge-m 0.6 --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 --overlap08-close-px 1 --overlap08-dilate-px 0 --overlap08-connect-m 2.0 --overlap08-thin-px 0 --overlap-boost-m 0.4`
- `--splat08-m 0.4` (meters)
- `--bounds-pkl maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/<split>/{scene}.pkl`
- `--match-05 --viz-root maptracker/work_dirs/semantic/<split>` (or your folder containing `<scene>/05_static_gt.png`)

Parallel batch (train/val):

```
SPLIT=valid WORKERS=8 nohup bash maptracker/tools/tracking/batch_gen_semantic_split_parallel.sh \
  > maptracker/work_dirs/semantic/batch_valid.log 2>&1 &

SPLIT=train WORKERS=8 nohup bash maptracker/tools/tracking/batch_gen_semantic_split_parallel.sh \
  > maptracker/work_dirs/semantic/batch_train.log 2>&1 &
```
