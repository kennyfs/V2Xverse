# Changelog

All modifications to the original V2Xverse codebase are documented here.

---

## CP-Guard Integration

**Reference:** "CP-Guard: Malicious Agent Detection and Defense in Collaborative
Bird's Eye View Perception", AAAI 2025.

**Date:** 2026-05-27

### New files

| File | Purpose |
|------|---------|
| `opencood/defense/__init__.py` | Package init for the new defense module |
| `opencood/defense/cp_guard.py` | `compute_ccloss()` (Eq. 4) and `pasac()` (Algorithm 1) |
| `opencood/tools/inference_multiclass_cpguard.py` | CLI entry-point for offline CP-Guard evaluation |

### Modified files

#### `opencood/models/point_pillar_multiclass.py`

Added method `encode(data_dict)` to class `PointPillarMulticlass`
(inserted between `backbone_fix()` and `forward()`).

**What it does:** Runs the encoding sub-pipeline (VFE → scatter →
`normalize_pairwise_tfm` → backbone → shrink_conv / compression) and
returns the per-CAV BEV feature tensor `spatial_features_2d` together
with the normalised transformation matrix `t_matrix_norm` and
`record_len`.  The existing `forward()` is **not changed**.

**Why:** PASAC needs to call encode once and the fusion+decode step
multiple times (once per candidate collaborator subset).  Exposing
`encode()` avoids repeating the expensive voxelisation and backbone
inference for every verification round.

#### `opencood/tools/inference_utils.py`

Added function `inference_intermediate_fusion_multiclass_cpguard()`,
inserted just before `save_prediction_gt()`.

**What it does:** Orchestrates the three-phase CP-Guard inference:

1. `model.encode()` — encode all CAVs once.
2. Compute Y_0 (ego-solo sigmoid cls_preds, no fusion).
3. `pasac()` — identify benign collaborators via recursive CCLoss
   verification.
4. Re-fuse with only benign collaborators and run the full decoder
   (`cls_head` + `reg_head` + `generate_predicted_boxes`) to produce
   the output dict expected by `dataset.post_process_multiclass`.

**Why:** Keeping this in `inference_utils.py` follows the existing
pattern where all inference modes live in that module and are
referenced by the CLI scripts.

---

### Design decisions and known caveats

**Algorithm 1 sign correction**

Algorithm 1 in the CP-Guard paper has an inverted condition in the
single-CAV base case (line 8 uses `<= eps` but the physical meaning
requires `> eps`).  The group-level conditions (lines 16, 22) are
correct as written.  This implementation uses the semantically correct
condition (`CCLoss > eps → benign`) for the single-CAV case, matching
the paper's text description and the ablation table (Table 2).

**Segmentation → Detection adaptation**

The original CP-Guard paper targets BEV segmentation.  Here the CCLoss
is computed on sigmoid-normalised detection heatmaps (`cls_preds`) from
the V2X-ViT model.  The three-class detection outputs (vehicle /
pedestrian / cyclist) play the role of semantic class probability maps.
This is a direct structural analogy; the numerical CCLoss range may
differ from the paper's segmentation experiments so the default
`eps = 0.08` should be re-validated once the pipeline runs end-to-end.

**t_matrix subsetting**

`t_matrix_norm` is normalised once at scatter resolution inside
`encode()`, matching `forward()`'s call site.  For each PASAC subset
`indices`, the sub-matrix is extracted as:

```python
sub_t = t_matrix_norm[:, idx_t][:, :, idx_t, :, :]   # [1, k, k, 2, 3]
```

`V2XViTFusion` derives `L` from `pairwise_t_matrix.shape[1]`, so
passing a `[1, k, k, 2, 3]` tensor with `record_len = [k]` is fully
compatible without any padding.

---

## ROBOSAC / AmongUs FDR Study

**Reference:** "Among Us: Adversarially Robust Collaborative Perception
by Consensus", ICCV 2023.

**Date:** 2026-05-29

**Goal:** Empirically measure the False Discovery Rate (FDR) and Benign
Information Loss Rate (BILR) of ROBOSAC in an urban intersection setting
(V2Xverse / Town05) with 5 CAVs (1 ego + 4 collaborators, 1 attacker).

---

### Model checkpoint

| Item | Value |
|------|-------|
| Config file | `opencood/hypes_yaml/v2xverse/late_fusion_multiclass_config_morecav.yaml` |
| Checkpoint | `opencood/logs/codriving_morecav_finetune_2026_05_29/net_epoch_bestval_at22.pth` |
| Architecture | CenterPoint-CoDriving (intermediate fusion, multi-class) |
| Fine-tuned from | 4-CAV V2Xverse checkpoint; additional Morecav data |
| Epoch | 22 (best validation AP) |
| File size | ~32 MB |

---

### New files

| File | Purpose |
|------|---------|
| `opencood/attack/__init__.py` | Package init for the new attack module |
| `opencood/attack/feature_attack.py` | `perturb_attacker_features()` — PGD FP/FN attack on BEV feature maps |
| `opencood/defense/__init__.py` | Package init (shared with CP-Guard) |
| `opencood/defense/robosac.py` | `run_robosac()`, `compute_s()`, `_box_iou_distance()` — Algorithm 1 |
| `opencood/tools/inference_multiclass_robosac.py` | CLI for clean / no-attack baseline (C1) and ROBOSAC defence (C2) |
| `opencood/tools/inference_multiclass_attack_robosac.py` | CLI for FP-attack (C3/C4) and FN-attack (C5/C6) conditions |
| `opencood/hypes_yaml/v2xverse/late_fusion_multiclass_config_morecav.yaml` | Config for Morecav (5-CAV, LiDAR X∈[-36,36] m, Y∈[-12,12] m) |

---

### Modified files

#### `opencood/models/center_point_codriving.py`

Added two methods to class `CenterPointCoDriving`:

- **`encode(data_dict)`** — runs VFE → pillar scatter → backbone and
  returns `(raw_features, psm_single, pairwise_t_matrix, record_len)`.
  Does *not* run fusion or decode heads.
- **`fuse_decode_subset(raw_features, psm_single, pairwise_t_matrix,
  record_len, indices)`** — runs `AttenFusion` on the specified CAV
  subset, then the full decode pipeline (cls/reg heads +
  `generate_predicted_boxes`).  Returns the same output dict as
  `forward()`.

**Why:** ROBOSAC calls encode once and fuse_decode N+1 times (once for
Y₀ ego-solo, N times for candidate subsets).  Splitting avoids repeating
the expensive VFE/backbone pass per trial.

#### `opencood/tools/inference_utils.py`

Added two functions:

- **`inference_intermediate_fusion_multiclass_robosac(...)`** —
  orchestrates ROBOSAC inference: encode → compute Y₀ → `run_robosac()`
  → fuse accepted set → decode.  Returns `pred_box_tensor`,
  `gt_box_tensor`, `pred_score`, and optionally `robosac_log`.
- **`inference_intermediate_fusion_multiclass_cpguard(...)`** — CP-Guard
  inference (see CP-Guard section above).

#### `opencood/data_utils/datasets/basedataset/v2xverse_basedataset.py`

- Added reading of `train_index_file` / `validate_index_file` /
  `test_index_file` config keys for Morecav split files.
- **Critical fix:** stride is forced to 1 at inference:
  `stride = self.params.get('frame_stride', 1) if self.train else 1`
  (previously stride=2 was applied even at test time, halving the
  evaluation set).

#### `opencood/tools/train.py`

- Added gradient accumulation: reads `hypes['train_params'].get('accum_steps', 4)`.
  Actual forward passes are batched, gradients accumulated for
  `ACCUM_STEPS` steps before `optimizer.step()`.

---

### Run commands

All commands assume `conda activate v2xverse` and are run from the
V2Xverse root directory.

**C1 — No defence, no attack (upper-bound baseline)**
```bash
python opencood/tools/inference_multiclass_robosac.py \
    --model_dir opencood/logs/codriving_morecav_finetune_2026_05_29 \
    --fusion_method intermediate \
    --no_robosac
```
Output: `AP_robosac_nofilter.txt`

**C2 — ROBOSAC defence, no attack**
```bash
python opencood/tools/inference_multiclass_robosac.py \
    --model_dir opencood/logs/codriving_morecav_finetune_2026_05_29 \
    --fusion_method intermediate \
    --eta 0.2 --N_budget 7 --eps 0.3 \
    --log_frames
```
Output: `AP_robosac_defense.txt`, `frame_log_robosac.json`

**C3 — No defence + FP attack**
```bash
python opencood/tools/inference_multiclass_attack_robosac.py \
    --model_dir opencood/logs/codriving_morecav_finetune_2026_05_29 \
    --fusion_method intermediate \
    --attack_goal fp --no_robosac
```
Output: `AP_attack_fp_nofilter.txt`

**C4 — ROBOSAC defence + FP attack**
```bash
python opencood/tools/inference_multiclass_attack_robosac.py \
    --model_dir opencood/logs/codriving_morecav_finetune_2026_05_29 \
    --fusion_method intermediate \
    --attack_goal fp \
    --eta 0.2 --N_budget 7 --eps 0.3 \
    --log_frames
```
Output: `AP_attack_fp_defense.txt`, `frame_log_attack_fp_robosac.json`

**C5 — No defence + FN attack**
```bash
python opencood/tools/inference_multiclass_attack_robosac.py \
    --model_dir opencood/logs/codriving_morecav_finetune_2026_05_29 \
    --fusion_method intermediate \
    --attack_goal fn --no_robosac
```
Output: `AP_attack_fn_nofilter.txt`

**C6 — ROBOSAC defence + FN attack**
```bash
python opencood/tools/inference_multiclass_attack_robosac.py \
    --model_dir opencood/logs/codriving_morecav_finetune_2026_05_29 \
    --fusion_method intermediate \
    --attack_goal fn \
    --eta 0.2 --N_budget 7 --eps 0.3 \
    --log_frames
```
Output: `AP_attack_fn_defense.txt`, `frame_log_attack_fn_robosac.json`

---

### Result files

| Condition | AP file | Frame log |
|-----------|---------|-----------|
| C1 no-attack no-defence | `AP_robosac_nofilter.txt` | — |
| C2 no-attack + ROBOSAC | `AP_robosac_defense.txt` | `frame_log_robosac.json` |
| C3 FP-attack no-defence | `AP_attack_fp_nofilter.txt` | — |
| C4 FP-attack + ROBOSAC | `AP_attack_fp_defense.txt` | `frame_log_attack_fp_robosac.json` |
| C5 FN-attack no-defence | `AP_attack_fn_nofilter.txt` | — |
| C6 FN-attack + ROBOSAC | `AP_attack_fn_defense.txt` | `frame_log_attack_fn_robosac.json` |

All files are written to `opencood/logs/codriving_morecav_finetune_2026_05_29/`.

---

### ROBOSAC parameters (fixed across all runs)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| η (eta) | 0.2 | Assumed attacker fraction |
| N | 7 | Sampling budget (max trials per frame) |
| ε (eps) | 0.3 | IoU-based consensus distance threshold |
| p | 0.99 | Required success probability |
| s | 3 | Computed guaranteed-benign subset size: ⌊ln(1−(1−0.99)^(1/7)) / ln(0.8)⌋ = 3 |

With n\_c = 4 collaborators and s = 3: min(3, 4) = 3 → non-degenerate
sampling (C(4,3) = 4 distinct subsets possible).

---

### Attack parameters (`feature_attack.py`)

| Parameter | Value |
|-----------|-------|
| PGD steps K | 10 |
| Step size α | 2.5 × ε\_abs / K |
| Budget ε\_abs | ε × Q₉₅(non-zero \|features\|) |
| ε (relative) | 0.3 |
| FP loss | mean(σ(ŷ) × 1[ego\_conf < 0.15]) |
| FN loss | −mean(σ(ŷ) × σ(y\_ego)) |

One attacker is selected per frame (random, seed 42); all other
collaborators are benign.

---

### Key metrics defined

**FDR (False Discovery Rate / fallback rate)**
Fraction of frames where ROBOSAC exhausted all N=7 trials without
reaching consensus → fell back to ego-only perception.  Counts only
*explicit* rejections (Type 1).

**BILR (Benign Information Loss Rate)**
Fraction of benign CAV slots not included in the final accepted set,
across all frames.  Includes both Type 1 (explicit rejection via
fallback) and Type 2 (untested exclusion when consensus found early).

**TRR (True Rejection Rate)**
Fraction of frames where the attacker was successfully excluded from the
accepted set.

**FAR (False Acceptance Rate)**
Fraction of frames where the attacker was included in the accepted set
(1 − TRR).
