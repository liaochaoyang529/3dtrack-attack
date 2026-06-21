# Implementation Comparison: my_attack vs. ICCV-CFG vs. Cheng 2025

## Overview

This document compares the current integrated implementation (`my_attack/core/critical_feature_guided_attack.py`) against the three original sources:
1. **Original my_attack** (`critical_feature_guided_attack.py.bak`) — PGD-based white-box attack on siamese 3D trackers
2. **CFG-ICCV2025** (`CFG-ICCV2025/Attacker/CFG.py`) — Critical Feature Guided attack for 3D point-cloud classifiers
3. **Cheng 2025** (IEEE TCSVT 2025 paper) — Black-box explainability-guided adversarial attack for 3D object tracking

---

## 1. Attack Framework & Optimizer

| Aspect | Original my_attack | CFG-ICCV2025 | Cheng 2025 | My Implementation |
|--------|-------------------|--------------|------------|-------------------|
| **Target task** | 3D Siamese tracking (BAT/P2B) | 3D point-cloud classification | 3D object tracking (P2B/BAT/PTT/PTTR/M2-Track) | 3D Siamese tracking (BAT/P2B) |
| **Optimizer** | PGD (`sign(grad)`, fixed α=0.005) | Adam + Binary Search (`binary_max_steps` × `iter_max_steps`) | Adam, lr=0.01, decay 0.2 every 5 iters | **PGD (same as original)** |
| **Iterations** | 20 | `iter_max_steps` (typically 100) + binary search | 80 | 20 (same as original) |
| **Perturbation init** | Zero | Normal distribution N(0, 1e-3) | Normal distribution | Zero (same as original) |
| **Box/L∞ clip** | `clamp(-eps, eps)` | `lp_clip()` + normal projection | Implicit via distance loss | `clamp(-eps, eps)` (same as original) |

### Key Difference
Both CFG-ICCV2025 and Cheng 2025 use **Adam optimizer** with adaptive learning rates. My implementation retains the original my_attack's **PGD sign-gradient** optimizer, which is much coarser. This is a fundamental architectural mismatch when integrating Cheng's surface constraint and motion losses — Adam can better navigate constrained spaces, while PGD's sign quantization loses fine-grained gradient information needed for surface-following.

---

## 2. Attribution / Importance Scoring Method

| Aspect | Original my_attack | CFG-ICCV2025 | Cheng 2025 | My Implementation |
|--------|-------------------|--------------|------------|-------------------|
| **Method** | Gradient norm `‖grad‖₂` | `|grad ⊙ feature|` (gradient × activation magnitude) | **Occlusion-based (black-box)** — box-aware voxel downsampling + multi-scale fusion | `compute_importance` uses `‖grad‖₂` (original); `compute_cfg_loss` uses `|grad ⊙ feature|` |
| **Gradient source** | `search_feature` after `conv_final` | `layer2_features` (mid-layer) | **No gradients** — pure black-box query | `search_feature` for importance; `fusion_feature` for `L_cfg` |
| **Point selection** | Top-k ratio (`k_ratio=0.2`) by gradient norm | Top-k by `|grad ⊙ feature|` | Top-K=200 by fused attribution map | Top-k ratio (`k_ratio=0.2`) by gradient norm |
| **Black-box?** | No (white-box) | White-box (requires `features_grad`) | **Yes** (no model gradients) | No (white-box) |

### Key Differences & Issues

**2.1 CFG-ICCV2025 integration issue:**
- CFG-ICCV2025 computes importance as `|grad ⊙ feature|` and uses it **both** for attribution scoring **and** as a regularization loss `L_cfg = Σ|grad ⊙ feature| × 0.01`.
- In my implementation:
  - `compute_importance()` still returns `grads.norm(p=2, dim=1)` (original my_attack style), **NOT** `|grad ⊙ feature|`.
  - `compute_cfg_loss()` separately computes `Σ|grad ⊙ feature| × 0.01` and adds it to the objective.
- **Result**: The CFG "grad ⊙ feature" philosophy is only half-implemented. Point selection still uses gradient norm, not CFG's sensitivity-magnitude product.

**2.2 Cheng 2025 integration issue:**
- Cheng's attribution is **completely different**: occlusion-based, model-agnostic, no gradients.
- It uses 3000 iterations of random voxel downsampling with box-aware probability (ε=0.4) and multi-scale fusion (voxel sizes 0.2m, 0.4m, 0.6m).
- My implementation **does not implement Cheng's attribution at all**. Instead, it keeps the original gradient-based point selection.

---

## 3. Loss Functions

### 3.1 Original my_attack
```python
objective = l_adv - beta_cd * l_cd - gamma_knn * l_knn
l_adv = lambda_match * (-score_gt) + lambda_offset * center_error
```
- `l_match = -score_gt.mean()` — suppress target confidence
- `l_offset = ‖c_pred - c_gt‖₂` — increase center error
- `l_cd` — Chamfer distance (imperceptibility)
- `l_knn` — KNN local consistency

### 3.2 CFG-ICCV2025
```python
loss = cls_loss + scale_const * constrain_loss
constrain_loss = dis_loss + cfg.CFG * CFGloss + cfg.PF_loss_weight * pf_loss + input_diversity_loss
CFGloss = Σ|grad_temp * mid_feature| * 0.01
```
- `cls_loss` — CrossEntropy or Margin loss (classification target)
- `dis_loss` — Chamfer / L2 / Hausdorff distance
- `CFGloss` — Feature-gradient alignment (the key novelty)
- `PF_loss` — Point Feature loss (leave-one-out sensitivity)
- `input_diversity_loss` — Random dropout for transferability

### 3.3 Cheng 2025
```
L = α·L_ms + β·L_mg + γ·L_distance
α=1, β=0.1, γ=0.5
```
- **`L_ms`** (motion-shift loss): Shift motions of **all proposals** away from GT motion, weighted by confidence `-‖M(B_prev, R_i) - M(B_prev, B_gt)‖₂² · log(1-C_i)`
- **`L_mg`** (motion-gap loss): Narrow gap between high-confidence and low-confidence proposals' motions
- **`L_distance`** (Chamfer distance): Imperceptibility constraint
- **No score suppression** — Cheng focuses purely on spatiotemporal motion distortion

### 3.4 My Implementation
```python
objective = (
    l_adv                           # original my_attack
    + lambda_cfg * L_cfg            # CFG-ICCV2025
    + lambda_ms * L_ms              # Cheng 2025
    + lambda_mg * L_mg              # Cheng 2025
    - beta_cd * l_cd                # original my_attack
    - gamma_knn * l_knn             # original my_attack
)
l_adv = lambda_match * (-score_gt) + lambda_offset * center_error
```

### Key Differences & Issues

**3.1 Loss combination mismatch:**
- I **added** `L_cfg`, `L_ms`, `L_mg` to the original objective, but **kept** `l_match = -score_gt`.
- Cheng 2025 **does not use score suppression at all**. Their attack is purely motion-based.
- The combination creates competing objectives: `l_match` pulls toward score suppression, while `L_ms`/`L_mg` pull toward motion distortion. In PGD with only 20 iterations, these may interfere destructively.

**3.2 `L_ms` / `L_mg` implementation gap:**
- My `_compute_motion_loss()` requires `B_prev` (previous frame bbox).
- The runner (`run_cfg_attack.py`) passes `B_prev=None`, so `L_ms = L_mg = 0` **in all current tests**.
- Cheng's motion loss requires proposals from the tracker output. My implementation extracts proposals from `end_points["estimation_boxes"]` but the confidence-based weighting may not match Cheng's exact formulation (they rank by confidence scores `C^S`, I use `softmax(proposal_logits * temperature)`).

**3.3 `L_cfg` scale issue:**
- CFG-ICCV2025 uses `scale_const * constrain_loss` where `scale_const` is dynamically adjusted via binary search.
- My implementation uses a fixed `lambda_cfg=0.5` multiplier with no dynamic balancing. The `L_cfg` magnitude may dominate or be negligible relative to `l_adv`.

---

## 4. Geometric Constraints

| Aspect | Original my_attack | CFG-ICCV2025 | Cheng 2025 | My Implementation |
|--------|-------------------|--------------|------------|-------------------|
| **Constraint type** | Global L∞ + Chamfer + KNN | Normal vector projection (`offset_proj`) + Lp clip | **Hard surface constraint** — 8-NN polynomial fit `z=ax²+by²+cxy+dx+ey+f` | **Hard surface constraint** (Cheng style) + original Chamfer/KNN |
| **When applied** | After each PGD step (clip) | After each Adam step (project) | After each optimization step (recompute z from polynomial) | After each PGD step (recompute z) |
| **Fitting method** | N/A | N/A | Least squares (Algorithm 1) | **Analytical least squares** (`torch.linalg.lstsq`) |

### Key Differences & Issues

**4.1 Surface constraint over-restriction in PGD:**
- Cheng 2025 uses **Adam** (80 iters) to explore the constrained manifold. Adam's smooth gradients allow the optimizer to find effective perturbations within the surface constraint.
- My **PGD sign-gradient** (20 iters) is too coarse for this. The surface constraint restricts z-movement, but PGD's `sign(grad)` + fixed step size cannot effectively search the reduced 2D (x,y) perturbation space. This explains why `surface_constraint=True` **worsened** attack performance in tests.

**4.2 Pre-computation vs. per-iteration fitting:**
- Cheng's Algorithm 1 fits surfaces **once** for the clean points, then uses those polynomials throughout optimization.
- My implementation pre-computes `surface_coeffs` before the attack loop, which matches Cheng's approach.
- However, Cheng also mentions "voxels are subjected to random movement and rotation" during attribution — this is for the **black-box explainability** phase, not the attack phase.

---

## 5. Code-Level Detailed Differences

### 5.1 `_forward_with_intermediate`

**Original my_attack:**
```python
return end_points, search_feature
```

**My implementation:**
```python
return end_points, search_feature, fusion_feature
```
- Added `fusion_feature` to support CFG-ICCV2025's `L_cfg` computation.
- **Issue**: `search_feature` (after `conv_final`) has no valid gradient flow in BAT due to `conv_final` being non-differentiable in some paths. `fusion_feature` (after `xcorr`) has valid gradients.

### 5.2 `compute_importance`

**Original my_attack:**
```python
scores = grads.norm(p=2, dim=1)
```

**My implementation (unchanged for point selection):**
```python
scores = grads.norm(p=2, dim=1)  # Same as original
```

**What CFG-ICCV2025 actually does:**
```python
# In CFG.py:
grad_temp = torch.autograd.grad(out, y, grad_outputs=torch.ones_like(out))[0]
# Importance is implicitly |grad * feature| via CFGloss
```

**What Cheng 2025 actually does:**
- No gradients. Attribution via 3000 iterations of random subset evaluation.

### 5.3 `main_attack_loop` structure

**Original my_attack (~120 lines):**
- Simple: objective → grad → PGD step → mask → history

**My implementation (~180 lines):**
- Added surface pre-computation block
- Added `L_ms`, `L_mg`, `L_cfg` computation
- Added surface constraint application after PGD step
- More complex but same PGD core

**CFG-ICCV2025 (~200 lines):**
- Binary search outer loop + Adam inner loop
- Dynamic `scale_const` adjustment
- Normal projection + Lp clip
- Much more sophisticated optimization

**Cheng 2025 (Algorithm 2):**
- Pre-compute attribution map (3000 iterations, separate from attack)
- Select top-K=200 critical points
- Adam optimization (80 iters) with lr decay
- Per-step surface reprojection

---

## 6. What Was Implemented Correctly vs. Incorrectly

### ✅ Correctly Implemented
1. **Chamfer distance** — matches all three sources
2. **KNN consistency loss** — matches original my_attack and Cheng's geometric intuition
3. **`L_cfg` formula** — `Σ|grad ⊙ feature| × 0.01` matches CFG-ICCV2025's `CFGloss`
4. **Surface polynomial fitting** — `z = ax² + by² + cxy + dx + ey + f` with 8-NN matches Cheng's Algorithm 1
5. **Motion loss formulas** — `L_ms` and `L_mg` equations match Cheng 2025 paper (Eq. 13, 14)
6. **Point selection mask** — `target_mask` (seg_label > 0.5) restricts perturbation to foreground points, consistent with Cheng's "search area enclosed by bounding box"

### ❌ Incorrectly Implemented / Missing
1. **Point selection attribution** — Still uses gradient norm instead of:
   - CFG's `|grad ⊙ feature|`
   - Cheng's occlusion-based attribution map
2. **Optimizer mismatch** — Kept PGD instead of Adam. This is the **root cause** of surface constraint failure.
3. **`B_prev` not passed** — `L_ms` and `L_mg` are always zero in current runner.
4. **Missing CFG-ICCV2025 components**:
   - Input diversity loss (random dropout for transferability)
   - PF_loss (point feature leave-one-out sensitivity)
   - Binary search for `scale_const`
   - Normal vector projection (`offset_proj`)
5. **Missing Cheng 2025 components**:
   - Occlusion-based explainability (3000 subset evaluations)
   - Box-aware voxel downsampling
   - Multi-scale fusion (0.2m/0.4m/0.6m voxel sizes)
   - Adam optimizer with lr decay
6. **Loss competition** — `l_match = -score_gt` (my_attack) competes with `L_ms`/`L_mg` (Cheng). Cheng's paper has **no score suppression** — it purely attacks motion.

---

## 7. Root Cause of Test Failures

| Test Result | Root Cause |
|-------------|-----------|
| `score_drop < 0` (score increases) | `l_match = -score_gt` is too weak vs. geometry constraints. In 20 PGD iterations, sign-gradient cannot effectively minimize a smooth loss like `-score_gt`. |
| Surface constraint makes attack worse | PGD + hard z-constraint = only x,y can move. Sign-gradient cannot navigate this 2D manifold effectively. Adam is required. |
| `L_ms = L_mg = 0` | `B_prev` not passed from runner. Motion losses are disabled. |
| Cross-model transfer (BAT→M2-Track) fails | Coordinate systems differ (`sample_bb` vs `ref_box`). No coordinate alignment implemented. |

---

## 8. Recommendations for Fix

### Option A: Fix within current PGD framework (minimal change)
1. **Remove `l_match = -score_gt`** — let `l_offset` + `L_cfg` + `L_ms`/`L_mg` drive the attack
2. **Remove surface constraint** — in PGD it does more harm than good
3. **Pass `B_prev` from runner** — enable motion losses
4. **Increase `lambda_cfg`** — make `L_cfg` the dominant regularizer
5. **Switch `compute_importance` to CFG style**:
   ```python
   scores = (grads * features.detach()).abs().sum(dim=1)  # CFG style
   ```

### Option B: Full Cheng 2025 integration (recommended for correctness)
1. **Implement occlusion-based attribution** (3000 iterations, voxel downsampling)
2. **Switch optimizer to Adam** (80 iters, lr=0.01, decay 0.2/5iters)
3. **Keep surface constraint** — it works correctly with Adam
4. **Remove `l_match = -score_gt`** — Cheng doesn't use it
5. **Add `B_prev` passing** — required for motion losses
6. **Use top-K=200** instead of `k_ratio`

### Option C: Full CFG-ICCV2025 integration
1. **Switch to Adam + Binary Search** framework
2. **Add input diversity loss** and **PF_loss**
3. **Add normal vector projection** (`offset_proj`)
4. **Use `|grad ⊙ feature|` for point selection**

---

*Generated: 2026-05-17*
*Files compared:*
- `my_attack/core/critical_feature_guided_attack.py.bak` (original)
- `my_attack/core/critical_feature_guided_attack.py` (current)
- `CFG-ICCV2025/Attacker/CFG.py` (ICCV-CFG)
- `Cheng 等 - 2025 - Black-Box Explainability-Guided Adversarial Attack for 3D Object Tracking.pdf` (Cheng 2025)
