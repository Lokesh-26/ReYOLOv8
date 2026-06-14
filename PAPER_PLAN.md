# Paper Plan: Recurrent Event-Based Object Detection for Industrial Environments

**Last updated**: 2026-06-14 (clean temporal context study complete)
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
| mtevent_640x480_c1_clean | 640×480 | C1 | 5 | left | scratch | **0.4994** | — | weights/best.pt |
| mtevent_640x480_c5_clean | 640×480 | C5 | 5 | left | scratch | **0.5145** | — | weights/best.pt |
| mtevent_640x480_fixed_c11 | 640×480 | C11 | 5 | left | scratch | **0.5093** | 73 | weights/best.pt |
| mtevent_640x480_fixed_c21 | 640×480 | C21 | 5 | left | scratch | **0.5274** | 107 | weights/best.pt |
| mtevent_640x480_tc_c11 | 640×480 | C11 | 5 | left | scratch | **0.4706** | — | weights/best.pt |
| mtevent_stereo_10ch_640x480_c11_v22 | 640×480 | C11 | 10 | stereo | scratch | **0.5063** | 111 | weights/best.pt |

*stereo_320x256_c11 was initialized from the 640×480 stereo C11 best checkpoint — not a clean
from-scratch or gen1-pretrained start.

**Note on cls_weight**: C1 and C5 were first trained with unconditional per-class inverse-AP BCE
weighting hardcoded. Those runs (`mtevent_640x480_c1`, `mtevent_640x480_c5`) are invalid.
The `_clean` reruns disable cls_weight (default off via flag). C11/C21 were not affected.

**Temporal context study — clean results** (sequential eval via eval_merged_klt.py):

| Model | Training AP50 | KLT-eval 17-cls | KLT-eval 15-cls | Gain |
|-------|--------------|----------------|----------------|------|
| C1 clean | 0.4994 | 0.3639 | 0.4156 | +0.0517 |
| C5 clean | 0.5145 | 0.4432 | 0.5022 | +0.0590 |
| C11 | 0.5093 | 0.4434 | 0.5044 | +0.0610 |
| C21 | 0.5274 | 0.4515 | 0.5084 | +0.0569 |
| C11_tc | 0.4706 | 0.4618 | 0.5253 | +0.0635 |

Training AP50 uses clip-batch validation (hidden state reset every clip); KLT-eval uses
sequential frame-by-frame inference (hidden state carried through each scene). The TC model
appears worst in training eval but best in sequential eval — see Section 11.7.

### 5.2 Currently Running (as of 2026-06-08; verify before action)

| Run directory | Res | Clip | Ch | Data | Latest AP50 |
|--------------|-----|------|----|------|-------------|
| mtevent_640x480_clsw_c21 | 640×480 | C21 | 5 | left | 0.486 (ep25) |
| mtevent_stereo_640x480_c21 | 640×480 | C21 | 10 | stereo | 0.482 (ep26) |

**Note on mtevent_640x480_clsw_c21**: IDENTICAL hyperparameters to `mtevent_640x480_fixed_c21`.
Useful for variance estimation only; do NOT present as a class-weighting ablation.

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

## 7. Core Temporal-Context Study (640×480, left-only, ReYOLOv8s) — COMPLETE

All four controlled runs are done. Fixed across comparison: architecture (ReYOLOv8s),
initialization (from scratch), dataset (corrected 640×480 left-only), 5-channel 50ms voxel
grid, optimizer (SGD, Optuna best params), batch=4, epochs=150.

| Config | Clip | Stride | Status | Training AP50 | KLT-eval 17cls | Run directory |
|--------|------|--------|--------|--------------|----------------|---------------|
| C1 | 1 | 1 | Complete ✓ | **0.4994** | 0.3639 | mtevent_640x480_c1_clean |
| C5 | 5 | 3 | Complete ✓ | **0.5145** | 0.4432 | mtevent_640x480_c5_clean |
| C11 | 11 | 5 | Complete ✓ | **0.5093** | 0.4434 | mtevent_640x480_fixed_c11 |
| C21 | 21 | 5 | Complete ✓ | **0.5274** | 0.4515 | mtevent_640x480_fixed_c21 |
| C11+TC | 11 | 5 | Complete ✓ | **0.4706** | 0.4618 | mtevent_640x480_tc_c11 |

**Note on C1**: ConvLSTM hidden state is still present but limited to one frame of context per
forward pass. Not a strictly non-recurrent model; it is the minimal-context lower bound.

**Note on TC**: TC loss penalises frame-to-frame score/box inconsistency. Training mAP50 is
worst (0.4706) because the training validator resets hidden state every clip and misses TC
benefit. Sequential eval (KLT-eval) shows TC is best overall (0.4618 17cls / 0.5253 15cls).

---

## 8. Experiments and Status Summary

### 8.1 Must-have for paper

| What | Why | Status |
|------|-----|--------|
| Temporal context curve C1–C21 at 640×480 | Core result | **DONE** |
| KLT ablation (17→15 class) | Colour-blindness penalty quantified | **DONE** |
| Diagnostic analyses (A1–A5) | Failure mode decomposition | **DONE** |
| Unified evaluation of all 640×480 models | Consistent AP50/per-class metrics | Not done — pipeline needed |
| RVT re-evaluation at AP50 | Fair architecture comparison | Not done — requires eval script |

### 8.2 Currently running (do not interrupt; verify before action)

| What | Why | Status |
|------|-----|-----|
| mtevent_640x480_clsw_c21 | Second C21 run (variance estimate) | Running |
| mtevent_stereo_640x480_c21 | Stereo C21 at 640×480 | Running |

### 8.3 Complete — do not rerun

- mtevent_640x480_c1_clean (C1, best AP50=0.4994)
- mtevent_640x480_c5_clean (C5, best AP50=0.5145)
- mtevent_640x480_fixed_c11 (C11, best AP50=0.5093)
- mtevent_640x480_fixed_c21 (C21, best AP50=0.5274)
- mtevent_640x480_tc_c11 (C11+TC, best AP50=0.4706 training / 0.4618 sequential)
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

Training AP50 (clip-batch val); KLT-eval AP50 (sequential, all-points interpolation).

| Model | Clip | Stride | Training AP50 | KLT-eval 17cls | KLT-eval 15cls | Δ 17cls vs C1 |
|-------|------|--------|--------------|----------------|----------------|--------------|
| ReYOLOv8s C1 | 1 | 1 | 0.4994 | 0.3639 | 0.4156 | baseline |
| ReYOLOv8s C5 | 5 | 3 | 0.5145 | 0.4432 | 0.5022 | +0.0793 |
| ReYOLOv8s C11 | 11 | 5 | 0.5093 | 0.4434 | 0.5044 | +0.0795 |
| ReYOLOv8s C21 | 21 | 5 | 0.5274 | 0.4515 | 0.5084 | +0.0876 |
| ReYOLOv8s C11+TC | 11 | 5 | 0.4706 | 0.4618 | 0.5253 | +0.0979 |

Key observation: saturation between C5 and C11 in training eval; TC loss uniquely benefits
from sequential hidden-state carry (see Section 11.7).

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

small_klt/big_klt/blue_klt merged into one 'klt' class. Custom AP evaluator (all-points);
absolute values lower than training-time eval but gains are internally consistent.

| Model | 17-class AP50 | 15-class AP50 | Gain |
|-------|--------------|--------------|------|
| C1 640×480 | 0.3639 | 0.4156 | +0.0517 |
| C5 640×480 | 0.4432 | 0.5022 | +0.0590 |
| C11 640×480 | 0.4434 | 0.5044 | +0.0610 |
| C21 640×480 | 0.4515 | 0.5084 | +0.0569 |
| C11+TC 640×480 | 0.4618 | 0.5253 | +0.0635 |

The ~0.056–0.063 gain is stable across clip lengths, quantifying the colour-blindness penalty.

### Table 5 — Architecture comparison (requires unified evaluation)

| Model | Resolution | AP50 | Note |
|-------|-----------|------|------|
| RVT best | 320×256 | TBD (unified) | Currently 0.19 COCO mAP from ckpt |
| ReYOLOv8s C21 | 320×256 | 0.2858 | AP50, different eval |
| ReYOLOv8s C21 | 640×480 | 0.5274 | AP50 |

Direct comparison of RVT COCO mAP and ReYOLOv8 AP50 is invalid. Mark this table
as requiring unified re-evaluation.

### Figure 1 — AP50 vs Clip Length curve (C1/C5/C11/C21 at 640×480)

Data available. Plot training AP50 and KLT-eval AP50 on same axes to show eval-protocol effect.

### Figure 2 — Per-class AP50 heatmap across clip lengths

### Figure 3 — KLT confusion matrix (17-class vs 15-class)

### Figure 4 — Training curves for all 640×480 runs

---

## 11. Analysis Plan (Without Additional Training)

### 11.0 Diagnostic Analyses (completed 2026-06-14, script: scripts/diagnostics.py)

Five analyses on C1/C5/C11/C21/C11_tc using sequential inference (hidden state carried through
each scene). Results in `runs/diagnostics/`. Key findings:

**A1 — Localization vs object displacement** (IoU of TP pairs binned by pixel displacement):
- C1: degrades sharply 0.804 (0-2px) → 0.628 (20-40px). Fast-moving objects uncaught.
- C5/C11/C21: flat (~0.879 → 0.834). Alignment saturates at C5; extra context doesn't help further.
- Implication: C1→C5 gain is alignment, not scene understanding.

**A2 — Recall vs event density** (GT instance recall binned by local density percentile):
- Counterintuitive: recall is highest at LOW density (p0-10: ~0.84) and lowest at HIGH density
  (p90-100: ~0.32–0.57). Large static objects (easy targets) dominate low-density bins.
- TC partially helps at the highest-density bins vs C11 baseline.
- Implication: density conditioning is not the bottleneck; class difficulty is.

**A3 — FP persistence after object exit** (ghost detection):
- Only 6–12 exit-proximity predictions tracked across all 4238 val frames.
- Ghost detection is NOT a real problem in this dataset.
- TC loss ghost-suppression motivation is empirically unsupported here.

**A4 — AP by object size and clip length**:
- Small objects (area <32²): AP=0.000 for ALL models. Hard resolution ceiling.
- Medium objects drive C1→C5 gain. C11_tc has best medium AP (0.362 vs C11 0.334).
- Large objects improve with clip but plateau at C11.
- Implication: method effort on small objects is futile at this resolution.

**A5 — Per-IoU AP vs clip length** (mAP at IoU thresholds 0.30–0.90):
- C1: collapses at strict IoU (0.015 @0.90). Localization is poor.
- C5–C21: maintain 0.283–0.308 @0.90. Recurrence substantially improves tight localization.
- C11_tc is best at every IoU threshold: 0.467 @0.30 → 0.308 @0.90.
- Implication: TC improves calibration AND localization, not just recall.

**Notable per-class improvements with TC (vs C11)**:
- ikea_samla_box: 0.065 → 0.353 (+5×)
- ikea_vesken_trolley: 0.645 → 0.688

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

### 11.7 Training eval vs sequential eval discrepancy (TC loss)
Training mAP50 uses clip-batch evaluation: hidden state reset every `clip_length` frames.
Sequential KLT eval carries hidden state through each full scene (as in deployment).
TC loss trains for frame-to-frame consistency which only manifests when the hidden state is
continuously propagated. Result: TC appears worst in training eval (0.4706) but best in
sequential eval (0.4618 17cls, 0.5253 15cls).

Paper angle: "standard clip-batch evaluation under-estimates TC benefit for recurrent event
detectors evaluated sequentially." The evaluation protocol choice matters for recurrent models.

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

**Core experimental study is complete.** Temporal context curve C1→C21 + TC done.

**Priority order**:

1. **Decide on paper angle**: benchmark+analysis (ICRA 2027) or method contribution (RAL).
   Diagnostics show TC loss helps in sequential eval but training-eval score is misleading.
   Whether TC is "novel enough" for RAL is a venue judgement, not a technical one.

2. **Write results section**: Table 1 (temporal context), Table 2 (resolution), Table 4 (KLT),
   Figure 1 (AP50 vs clip length curve), Figure 2 (per-class heatmap), A1–A5 diagnostic plots.

3. **Unified evaluation script**: write script for all 640×480 checkpoints using same protocol
   as eval_merged_klt.py. Save to `runs/eval/unified_YYYYMMDD/`. Add C1_clean/C5_clean entries.

4. **Verify RVT AP50** using unified evaluator on `pu432wr0` and `jk20t51u` checkpoints.

5. **When stereo C21 and clsw_c21 complete**: evaluate, add to stereo/variance tables.

6. **Optional**: larger TC clip (C21+TC) if paper story benefits from it. Do not launch blindly.

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

1. **Venue strategy**: ICRA 2027 is feasible for a benchmark + analysis paper. RAL requires a
   method contribution. TC loss is a candidate but its evaluation-protocol discrepancy needs
   to be the paper's core narrative, not just a footnote.
2. **TC at C21**: Would C21+TC show stronger gain? Not yet run. Only launch if paper story needs it.
3. **RVT AP50 recovery**: Is there a fast path to re-evaluate the existing RVT checkpoints
   using the ReYOLOv8 evaluator format?
4. **Small object performance**: All models give 0.000 for small objects at 640×480. Is the
   resolution ceiling a fundamental limit or a training issue? Higher res or multi-scale heads?
5. **Co-authors**: Not documented here. Who should be included?
