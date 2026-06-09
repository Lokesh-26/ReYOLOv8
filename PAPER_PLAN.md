# Paper Plan: Recurrent Event-Based Object Detection for Industrial Environments

**Last updated**: 2026-06-08 (full audit)
**Target venue**: ICRA 2027 (deadline: Sep 15, 2027)
**RAL viability**: requires a solid method contribution beyond baseline analysis

---

## 1. Research Question

> How do spatial resolution, temporal context, model architecture, data composition, and class
> ambiguity affect recurrent event-based object detection in dense industrial environments?

**Dataset**: MTEvent — a DVXplorer stereo event-camera dataset (640×480 native, 75 scenes,
17 object classes, factory/warehouse setting), published at CVPRW 2025. MTEvent is an existing
published dataset; this work uses it for a systematic experimental study.

**Contribution at this stage**: a reproducible experimental study, not a new dataset.

---

## 2. The 17 Classes

```
0=wooden_pallet,  1=small_klt,       2=big_klt,          3=blue_klt,
4=amazon_luggage, 5=ikea_dammang_bin, 6=ikea_vesken_trolley, 7=ikea_sortera_bin,
8=ikea_drona_grey, 9=ikea_drona_blue, 10=ikea_knallig_box, 11=ikea_moppe_drawer,
12=ikea_labbsal_basket, 13=ikea_ivar_box, 14=ikea_skubb_case, 15=ikea_samla_box, 16=human
```

**Key challenge**: small_klt / big_klt / blue_klt differ primarily in colour. Event cameras
detect brightness changes, not colour. These three classes are nearly indistinguishable in event
data (normalised box width: small_klt ≈ 0.129, blue_klt ≈ 0.124).

---

## 3. Models

| Model | Architecture | Temporal | Notes |
|-------|-------------|----------|-------|
| YOLOv8s C1 | CNN, no recurrence | Per-frame | Non-recurrent baseline at 320×256 |
| ReYOLOv8s | CNN + ConvLSTM | Clip-based | Main model |
| RVT | Transformer + recurrence | Sequence | Architecture comparison |

**Clip notation**: C1 = clip_length=1, C5 = clip_length=5, C11 = clip_length=11,
C21 = clip_length=21.

**Initialization note**: All 640×480 ReYOLOv8s runs used `pretrained: false`
(architecture YAML only, random init). All 320×256 runs used the Gen1 pretrained checkpoint
(`weights/gen1/reyolov8s_gen1_rps.pt`). The 640×480 vs 320×256 comparison is therefore NOT
controlled for initialization.

---

## 4. Dataset Splits

- **Train**: scenes 9–75, excluding val scenes
- **Val**: scenes 14, 26, 33, 35
- **Test**: scenes 1–2
- No right-camera bag for scene 73 (skipped in stereo runs)

**Corrected 640×480 preprocessing** (fixed 2026-06-04): Original 640×480 train H5 was built
from wrong event bags → val mAP=0 throughout training. Fixed by extracting the first 5 channels
from the stereo train H5 (verified to match the val H5 for the same scenes). Script:
`scripts/fix_640x480_train.py`. Broken H5 backed up as `mtevent_train.h5.broken_bak`.
All current 640×480 runs use the corrected dataset.

**Dataset directory**: `preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/`
- train H5: verified correct
- val H5: 4238 frames
- train frames: 19369

---

## 5. Complete Experiment Audit (as of 2026-06-08)

### 5.1 Completed and Valid

| Run directory | Res | Clip | Ch | Data | Init | Best AP50 | Best ep | Checkpoint |
|--------------|-----|------|----|------|------|-----------|---------|------------|
| mtevent_baseline_yolov8s_5ch | 320×256 | C1 | 5 | left | scratch | **0.2604** | 13 | weights/best.pt |
| mtevent_17cls_v2_nofrozen | 320×256 | C11 | 5 | left | gen1-pt | **0.2780** | 19 | weights/best.pt |
| mtevent_17cls_leftonly_c21 | 320×256 | C21 | 5 | left | gen1-pt | **0.2858** | 18 | weights/best.pt |
| mtevent_17cls_combined_c212 | 320×256 | C21 | 5 | L+R | gen1-pt | **0.3067** | 24 | weights/best.pt |
| mtevent_stereo_10ch_320x256_c11 | 320×256 | C11 | 10 | stereo | 640-ft* | **0.3166** | 60 | weights/best.pt |
| mtevent_640x480_fixed_c11 | 640×480 | C11 | 5 | left | scratch | **0.5093** | 73 | weights/best.pt |
| mtevent_stereo_10ch_640x480_c11_v22 | 640×480 | C11 | 10 | stereo | scratch | **0.5063** | 111 | weights/best.pt |
| mtevent_640x480_fixed_c21 | 640×480 | C21 | 5 | left | scratch | **0.5274** | 107 | weights/best.pt |

*stereo_320x256_c11 was initialized from the 640×480 stereo C11 best checkpoint — not a clean
from-scratch or gen1-pretrained start.

**Note on KLT ablation** (custom eval, all-points AP — absolute values lower than training eval):
| Model | 17-class AP50* | 15-class AP50* | Gain |
|-------|---------------|---------------|------|
| mtevent_640x480_fixed_c21 | 0.4515 | 0.5084 | +0.057 |
| mtevent_640x480_fixed_c11 | 0.4434 | 0.5044 | +0.061 |
*From eval_merged_klt.py — uses different AP implementation than training eval.
The absolute values are lower, but the gain (+0.057/+0.061) is valid for ablation comparison.

### 5.2 Currently Running

| Run directory | Res | Clip | Ch | Data | Tmux | PID | Epoch | Latest AP50 |
|--------------|-----|------|----|------|------|-----|-------|-------------|
| mtevent_640x480_clsw_c21 | 640×480 | C21 | 5 | left | post_640_queue | 996317 | 27/150 | 0.486 (ep25) |
| mtevent_stereo_640x480_c21 | 640×480 | C21 | 10 | stereo | train_640_scratch | 995551 | 44/150 | 0.482 (ep26) |

**CRITICAL NOTE on mtevent_640x480_clsw_c21**: This run has IDENTICAL hyperparameters to
`mtevent_640x480_fixed_c21` (same fl_gamma=1.5, same cls=3.31, same model arch, same data).
No per-class BCE weighting is implemented anywhere in train.py (commented out as TODO).
This run is effectively a second training run of C21, useful for variance estimation but NOT
a distinct class-weighting ablation.

**NOTE on mtevent_stereo_640x480_c21**: This is a NEW stereo C21 run, not previously documented
in the memory. It uses stereo 640×480 data (10ch) and is showing improving mAP50 (0.581 at
ep41, trending up). This was not mentioned in earlier session notes.

### 5.3 Never Started (despite claims in earlier session notes)

| Planned run | Reason claimed | Actual status |
|-------------|---------------|---------------|
| mtevent_640x480_c1 | "already running" | No directory, no process — NOT STARTED |
| mtevent_640x480_c5 | "already running" | No directory, no process — NOT STARTED |
| mtevent_640x480_medium_c21 | "epoch 34/150" | No directory, never existed |

These runs do not exist anywhere on disk. They must be launched after completing this audit.

### 5.4 Invalid / Failed

| Run | Problem |
|-----|---------|
| mtevent_640x480_c11_v2 | Completed 150 epochs, mAP50=0 throughout — INVALID (data bug) |
| mtevent_17cls_combined_c21 | Has args.yaml and weights dir only — no results.csv, crashed before training |
| mtevent_stereo_10ch_640x480_c11 | No results.csv — crashed or was superseded by _v2/_v22 |
| mtevent_stereo_10ch_640x480_c11_v2 | No results.csv — superseded by _v22 |

### 5.5 RVT Results (INCOMPARABLE to ReYOLOv8 AP50)

| Run ID | Data | Res | Metric | Value | Source |
|--------|------|-----|--------|-------|--------|
| pu432wr0 | left | 320×256 | COCO mAP (0.5:0.95) | ~0.19 | checkpoint filename |
| jk20t51u | L+R | 320×256 | COCO mAP (0.5:0.95) | ~0.20 | checkpoint filename |

- RVT reports COCO mAP (0.5:0.95), not AP50. **Direct comparison with ReYOLOv8 AP50 is invalid.**
- All local RVT training logs are empty (0 bytes) — metrics were only in W&B (disabled).
- AP50 values (0.366, ~0.356) cited in earlier plan are NOT verified from any local file.
- Unified RVT re-evaluation using the same evaluator as ReYOLOv8 is required before any
  architecture comparison.

---

## 6. Active Run Verification

### 6.1 mtevent_640x480_clsw_c21

- **Tmux session**: post_640_queue
- **PID**: 996317 (main process)
- **GPU**: device 0, ~13.7 GB
- **Epoch**: 27/150 as of audit
- **Data**: `configs/vtei_mtevent_640x480.yaml` → corrected 640×480 dataset ✓
- **Model**: `ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml` (from scratch, no pretrained)
- **Clip**: length=21, stride=5 ✓
- **Channels**: 5 ✓
- **imgsz**: 640 ✓
- **Batch**: 4 ✓
- **fl_gamma**: 1.5 (focal loss, same as fixed_c21 — no per-class weighting)
- **Training status**: healthy, progressing normally
- **Log path**: `runs/train/mtevent_640x480_clsw_c21.log`
- **Output dir**: `runs/train/mtevent_640x480_clsw_c21/`
- **Latest checkpoint**: `runs/train/mtevent_640x480_clsw_c21/weights/last.pt`
- **Best checkpoint**: `runs/train/mtevent_640x480_clsw_c21/weights/best.pt`

### 6.2 mtevent_stereo_640x480_c21

- **Tmux session**: train_640_scratch
- **PID**: 995551 (main process)
- **GPU**: device 0, ~13.9 GB
- **Epoch**: 44/150 as of audit
- **Data**: `configs/vtei_mtevent_stereo_640x480.yaml` → stereo 10ch 640×480 ✓
- **Model**: `ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml` (from scratch)
- **Clip**: length=21, stride=5
- **Channels**: 10 ✓
- **imgsz**: 640 ✓
- **Batch**: 4 ✓
- **Latest val AP50**: 0.482 (best so far at ep26), still early in training
- **Training status**: healthy, progressing normally
- **Log path**: `runs/train/mtevent_stereo_640x480_c21.log`
- **Output dir**: `runs/train/mtevent_stereo_640x480_c21/`

---

## 7. Core Temporal-Context Study (640×480, left-only, ReYOLOv8s)

The central comparison requires four controlled runs varying only clip length:

| Config | Clip | Stride | Status | Best AP50 | Run directory |
|--------|------|--------|--------|-----------|---------------|
| C1 | 1 | 1 | **NOT STARTED** | — | mtevent_640x480_c1 |
| C5 | 5 | 3 | **NOT STARTED** | — | mtevent_640x480_c5 |
| C11 | 11 | 5 | Complete ✓ | **0.5093** | mtevent_640x480_fixed_c11 |
| C21 | 21 | 5 | Complete ✓ | **0.5274** | mtevent_640x480_fixed_c21 |

**Training commands for missing runs:**

```bash
# C1 — single-timestep baseline (minimal temporal context)
WANDB_MODE=disabled /home/loki/anaconda3/envs/reyolov8/bin/python train.py \
  --model ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml \
  --data configs/vtei_mtevent_640x480.yaml \
  --hyp configs/default_gen1.yaml \
  --device 0 --batch 4 --imgsz 640 --epochs 150 \
  --channels 5 --clip_length 1 --clip_stride 1 --freeze 0 \
  --name mtevent_640x480_c1

# C5
WANDB_MODE=disabled /home/loki/anaconda3/envs/reyolov8/bin/python train.py \
  --model ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml \
  --data configs/vtei_mtevent_640x480.yaml \
  --hyp configs/default_gen1.yaml \
  --device 0 --batch 4 --imgsz 640 --epochs 150 \
  --channels 5 --clip_length 5 --clip_stride 3 --freeze 0 \
  --name mtevent_640x480_c5
```

**Fixed across comparison**: architecture (ReYOLOv8s), initialization (from scratch), dataset
(corrected 640×480 left-only), 5-channel 50ms voxel grid, optimizer (SGD, Optuna best params),
640×480 input, batch=4, epochs=150, seed=0.

**Note on C1**: With clip_length=1, the model processes one timestep at a time. The ConvLSTM
hidden state is still technically maintained across timesteps; C1 does not bypass the recurrent
module but limits it to one frame of context per forward pass. It is a minimal-context baseline,
not a strictly non-recurrent model.

Do NOT launch another C1 or C5 if either run already exists or is running. Verify first.

---

## 8. Experiments and Status Summary

### 8.1 Must-have for paper

| What | Why | Status |
|------|-----|--------|
| C1 at 640×480 | Lower bound for temporal context | NOT STARTED — must launch |
| C5 at 640×480 | Intermediate point in context curve | NOT STARTED — must launch |
| Unified evaluation of all 640×480 models | Consistent AP50/per-class metrics | Not done — evaluation pipeline needed |
| RVT re-evaluation at AP50 | Fair architecture comparison | Not done — requires eval script |
| KLT ablation (run eval_merged_klt.py) | Verify/update 17→15 class results | Script ready, not re-run recently |

### 8.2 Currently running (do not interrupt)

| What | Why | ETA |
|------|-----|-----|
| mtevent_640x480_clsw_c21 | Second C21 run (variance estimate); functionally C21 repeat | ~100 more epochs |
| mtevent_stereo_640x480_c21 | Stereo C21 at 640×480 | ~100 more epochs |

### 8.3 Complete — do not rerun

- mtevent_640x480_fixed_c21 (C21, best AP50=0.5274)
- mtevent_640x480_fixed_c11 (C11, best AP50=0.5093)
- mtevent_stereo_10ch_640x480_c11_v22 (stereo C11, best AP50=0.5063)
- mtevent_17cls_leftonly_c21 (C21 at 320×256, best AP50=0.2858)
- mtevent_17cls_combined_c212 (C21 L+R at 320×256, best AP50=0.3067)
- mtevent_17cls_v2_nofrozen (C11 at 320×256, best AP50=0.2780)
- mtevent_baseline_yolov8s_5ch (C1 non-recurrent at 320×256, best AP50=0.2604)

### 8.4 Optional (do not queue automatically)

| Experiment | Fills gap |
|-----------|-----------|
| ReYOLOv8m C21 at 640×480 | Architecture size effect; requires batch=2 |
| RVT retraining at 640×480 | Resolution-controlled arch comparison |
| TC loss ablation (default_gen1_tc.yaml) | Method contribution test |
| Stereo C11 full re-run from scratch | Controlled stereo vs monocular at C11 |
| 25 ms / 100 ms accumulation window | Event density effect |
| Additional seed repeats | Variance estimation |

Only launch an optional experiment when it fills a specific gap in a planned paper table or
resolves an identified ambiguity.

---

## 9. Unified Evaluation Pipeline

### 9.1 Requirements

All models must be evaluated on the same split using the same protocol:
- **Split**: val (for development); test (scenes 1–2) for final paper table only
- **Dataset**: `preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/` (corrected)
- **Class mapping**: 17-class (0–16) as standard; 15-class (KLT merged) as ablation
- **Box format**: xywh normalised → xyxy for IoU
- **Confidence threshold**: 0.001 (as in eval_merged_klt.py)
- **IoU threshold**: 0.45 (as in eval_merged_klt.py) for AP50 comparison
- **AP implementation**: all-points interpolation (consistent across models)

### 9.2 Required output per model

- Overall AP50 and mAP50–95
- Per-class AP50
- Per-class instance count
- Confusion matrix
- Precision/recall at operating point

### 9.3 Models to evaluate

| Model | Checkpoint | Eval config |
|-------|-----------|-------------|
| C1 at 640×480 | (pending) | clip=1, stride=1 |
| C5 at 640×480 | (pending) | clip=5, stride=3 |
| C11 at 640×480 | mtevent_640x480_fixed_c11/weights/best.pt | clip=11, stride=5 |
| C21 at 640×480 | mtevent_640x480_fixed_c21/weights/best.pt | clip=21, stride=5 |
| Stereo C11 at 640×480 | mtevent_stereo_10ch_640x480_c11_v22/weights/best.pt | clip=11, stride=5, ch=10 |
| Stereo C21 at 640×480 | mtevent_stereo_640x480_c21/weights/best.pt (when done) | clip=21, stride=5, ch=10 |
| C21 at 320×256 | mtevent_17cls_leftonly_c21/weights/best.pt | clip=21, stride=5, imgsz=320 |
| C21 L+R at 320×256 | mtevent_17cls_combined_c212/weights/best.pt | clip=21, stride=5, imgsz=320 |
| YOLOv8s C1 at 320×256 | mtevent_baseline_yolov8s_5ch/weights/best.pt | clip=1, stride=1, imgsz=320 |

### 9.4 RVT evaluation gap

RVT was evaluated using COCO mAP (0.5:0.95). The only locally available metric values are
embedded in checkpoint filenames (pu432wr0: 0.19, jk20t51u: 0.20). The mAP50 values previously
cited (0.366, ~0.356) are NOT verified from any local file.

To make a valid architecture comparison, both RVT and ReYOLOv8 must be evaluated with the same
AP50 evaluator on the same split. Until this is done, present RVT results separately and note
they use a different evaluation protocol.

### 9.5 Script: eval_merged_klt.py

Evaluates C21 and C11 (640×480) for 17-class and 15-class (KLT-merged) AP50.
Location: `scripts/eval_merged_klt.py`
Models hardcoded: `mtevent_640x480_fixed_c21/weights/best.pt`, `mtevent_640x480_fixed_c11/weights/best.pt`
Output: printed to stdout — no JSON/CSV saved.

**TODO**: Add JSON/CSV output to this script. Add C1, C5 entries once trained.

---

## 10. Planned Paper Tables and Figures

### Table 1 — Temporal context at 640×480 (core result)

| Model | Clip | Stride | AP50 (val) | Δ vs C1 |
|-------|------|--------|------------|---------|
| ReYOLOv8s C1 | 1 | 1 | PENDING | baseline |
| ReYOLOv8s C5 | 5 | 3 | PENDING | +? |
| ReYOLOv8s C11 | 11 | 5 | 0.5093 | +? |
| ReYOLOv8s C21 | 21 | 5 | 0.5274 | +? |

### Table 2 — Resolution comparison (matched clip length)

| Model | Resolution | Clip | AP50 |
|-------|-----------|------|------|
| ReYOLOv8s C11 | 320×256 | C11 | 0.2780 |
| ReYOLOv8s C11 | 640×480 | C11 | 0.5093 |
| ReYOLOv8s C21 | 320×256 | C21 | 0.2858 |
| ReYOLOv8s C21 | 640×480 | C21 | 0.5274 |

Note: 640×480 runs used random init; 320×256 runs used Gen1 pretrained. Comparison is not
controlled for initialization. Report this limitation.

### Table 3 — Data composition at 640×480

| Model | Data | Clip | AP50 |
|-------|------|------|------|
| ReYOLOv8s | left-only | C11 | 0.5093 |
| ReYOLOv8s | left-only | C21 | 0.5274 |
| ReYOLOv8s | stereo (10ch) | C11 | 0.5063 |
| ReYOLOv8s | stereo (10ch) | C21 | PENDING (mtevent_stereo_640x480_c21) |

### Table 4 — KLT ambiguity ablation

| Model | 17-class AP50* | 15-class AP50* | Gain |
|-------|---------------|---------------|------|
| C21 640×480 | 0.4515 | 0.5084 | +0.057 |
| C11 640×480 | 0.4434 | 0.5044 | +0.061 |
*Custom AP evaluator; absolute values lower than training-time eval. The gain is valid.

### Table 5 — Architecture comparison (requires unified evaluation)

| Model | Resolution | AP50 | Note |
|-------|-----------|------|------|
| RVT best | 320×256 | TBD (unified) | Currently 0.19 COCO mAP from ckpt |
| ReYOLOv8s C21 | 320×256 | 0.2858 | AP50, different eval |
| ReYOLOv8s C21 | 640×480 | 0.5274 | AP50 |

Direct comparison of RVT COCO mAP and ReYOLOv8 AP50 is invalid. Mark this table
as requiring unified re-evaluation.

### Figure 1 — AP50 vs Clip Length curve (C1/C5/C11/C21 at 640×480)

Requires C1 and C5 results.

### Figure 2 — Per-class AP50 heatmap across clip lengths

### Figure 3 — KLT confusion matrix (17-class vs 15-class)

### Figure 4 — Training curves for all 640×480 runs

---

## 11. Analysis Plan (Without Additional Training)

### 11.1 Temporal context
When C1 and C5 are complete: compare C1→C5→C11→C21 AP50 curve. Identify saturation point.
Per-class breakdown: which classes gain/lose as clip grows. Do not pre-assume the outcome.

### 11.2 Resolution
C11 320×256 vs C11 640×480: +0.231 AP50 gain. Partially confounded by initialization
(pretrained vs scratch). Report both factors explicitly. Do not attribute entire gain to resolution.

### 11.3 KLT ambiguity
~5.7% mAP is recovered by merging small/big/blue KLT into one class. Interpretation: colour
blindness of event sensors causes ~6% AP penalty on these three classes.
Use cautious wording: "consistent with a colour-blindness penalty; other confounding factors
(scale similarity, occlusion) cannot be excluded."

### 11.4 Stereo vs monocular
Stereo C11 (0.5063) vs monocular C11 (0.5093): naive 10-channel concatenation shows no gain.
Do not conclude stereo is ineffective — only that naive concatenation is not sufficient.
Stereo C21 pending; compare when complete.

### 11.5 Data composition (320×256)
L+R combined C21 (0.3067) vs left-only C21 (0.2858): +7.0% from doubling training data.
Comparison is controlled (same init, same config). This is a data-augmentation baseline.

### 11.6 mtevent_640x480_clsw_c21
This run uses identical hyperparameters to fixed_c21. On completion:
- If AP50 is similar to fixed_c21: treat as variance estimate (second seed)
- Do NOT present as a class-weighting ablation — no weighting is implemented

---

## 12. Reproducibility Checklist

### Environment
- Python: `/home/loki/anaconda3/envs/reyolov8/bin/python` (torch 2.8, CUDA 12.8)
- GPU: NVIDIA RTX 5090 (32 GB)
- Dataset preprocessing: `/usr/bin/python3` (system python with rosbag)

### Dataset
- Source: MTEvent (CVPRW 2025, published dataset — do not redistribute)
- Raw zips: `/mnt/2tb/MTevent/` (340 GB)
- Left extracted: `/mnt/2tb/MTevent_extracted_min/scene{1..75}/`
- Right extracted: `/mnt/2tb/MTevent_extracted_right/scene{1..75}/`
- Preprocessed (corrected): `preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/`
- Preprocessing script: `scripts/mtevent_to_reyolo_h5.py` (fixed 2026-03-08: numeric sort)
- Fix script: `scripts/fix_640x480_train.py` (fixed 2026-06-04: wrong bag assignment)

### Known dataset issues
1. **String-sort bug** (fixed 2026-03-08): `mtevent_to_reyolo_h5.py` used string sort on
   scene numbers → wrong scene-label pairing. The 0.329 mAP50 result from the workshop paper
   used buggy data. All current runs use fixed preprocessing.
2. **640×480 train H5 bug** (fixed 2026-06-04): Original 640×480 train H5 built from
   different event bags than the val H5 → mAP=0 throughout training. Fixed by extracting
   first 5 channels from the stereo train H5 (which shares bags with the val set).

### Training configuration
- Hyperparams: `configs/default_gen1.yaml` (Optuna best: lr0=0.00181, box=10.87, cls=3.31)
- TC loss config: `configs/default_gen1_tc.yaml` (sets tc=0.1, same everything else)
- Seed: 0

---

## 13. Immediate Next Actions

**Priority order**:

1. **Launch C1 at 640×480** (GPU is occupied by two runs sharing a single GPU — wait for one
   to finish before launching, or assess if batch=4 permits two concurrent runs).
   Verify first that `runs/train/mtevent_640x480_c1/` does not exist.

2. **Launch C5 at 640×480** — same constraint.

3. **Add CSV/JSON output to eval_merged_klt.py** to save per-class AP results.

4. **Write unified evaluation script** for all 640×480 checkpoints (AP50, per-class, confusion
   matrix). Save outputs to `runs/eval/unified_640x480_YYYYMMDD/`.

5. **Re-run eval_merged_klt.py** once C1 and C5 checkpoints exist, and update KLT ablation
   table with those additional points.

6. **Verify RVT AP50** by running a unified evaluator on `pu432wr0` and `jk20t51u` checkpoints.

7. **When mtevent_stereo_640x480_c21 completes**: evaluate with unified script and add to
   stereo comparison table.

8. **When mtevent_640x480_clsw_c21 completes**: evaluate, confirm it matches fixed_c21
   (same hyperparams), and decide how to present it in the paper (variance estimate).

**After C1 and C5 complete**: assemble Table 1, Figure 1, and begin writing the results section.

---

## 14. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| C1/C5 GPU time unavailable | Low | Medium | Queue after one running job finishes |
| clsw_c21 result confounds paper story | Confirmed | Low | Present as repeat C21, not cls-weighted |
| RVT AP50 re-evaluation infeasible | Medium | Medium | Present separate tables, mark incomparable |
| stereo_c21 result lower than mono | Medium | Low | Shows naive stereo is hard, not a failure |
| 640×480 init difference confounds resolution analysis | Confirmed | Medium | State explicitly in paper; run gen1-pretrained 640 as optional |

---

## 15. Open Questions

1. **C1 result**: This is the most important pending number. Until C1 is evaluated, the resolution-
   vs-recurrence decomposition cannot be quantified.
2. **GPU scheduling**: With two runs occupying the GPU, when is the right time to launch C1/C5?
   Both current runs are at epochs 27 and 44 of 150 — likely 1–2 days remaining each.
3. **Venue strategy**: ICRA 2027 is feasible for a benchmark + analysis paper. RAL requires a
   method contribution. Decide on venue before investing in method development.
4. **Co-authors**: Not documented here. Who should be included?
5. **RVT AP50 recovery**: Is there a fast path to re-evaluate the existing RVT checkpoints
   using the ReYOLOv8 evaluator format?
