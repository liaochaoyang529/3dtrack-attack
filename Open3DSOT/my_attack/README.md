# Critical Feature Guided Attack (CFG-Attack)

This folder provides a PyTorch implementation of **Critical Feature Guided Attack** for Siamese 3D trackers (P2B/BAT-style PointNet++ backbone).

## Implemented Functions

- `compute_importance(features, loss)`
- `select_critical_points(scores, k_ratio)`
- `attack_step(points, gradients, weights, alpha, eps, delta)`
- `chamfer_distance(pc1, pc2)`
- `main_attack_loop(model, input_dict, c_gt, target_mask, attack_cfg)`

Core file:
- `my_attack/core/critical_feature_guided_attack.py`

Compatibility wrapper:
- `my_attack/critical_feature_guided_attack.py`

## One-command Run

```bash
python3 /workspace/Open3DSOT/Open3DSOT/my_attack/run_cfg_attack.py \
  --cfg /workspace/Open3DSOT/Open3DSOT/cfgs/BAT_Car.yaml \
  --checkpoint /workspace/Open3DSOT/Open3DSOT/pretrained_models/bat_kitti_car.ckpt \
  --split train \
  --batch_size 1 \
  --iters 20 \
  --eps 0.05 \
  --alpha 0.005 \
  --k_ratio 0.2
```

## Outputs

Saved under `my_attack/outputs`:
- `*_S_adv.npy`: adversarial search point cloud
- `*_delta.npy`: perturbation
- `*_report.json`: score drop and center-error change over iterations

## Script Index (整理)

- `core/`: 攻击核心算法，例如 CFG attack。
- `scripts/`: 命令行运行入口，例如 `run_cfg_attack.py` 和 `start.sh`。
- `evaluation/`: 全数据集攻击评估脚本。
- `analysis/`: 成功/失败特征分布分析脚本。
- `visualization/`: 攻击样本可视化脚本和 MATLAB 辅助文件。
- `stagea/`: M2Track Stage A 数据导出逻辑。
- `feature_direction/`: failure-direction / feature-center 攻击验证工具。
- `outputs/`: 实验输出文件，保持不移动。

Legacy wrappers kept at `my_attack/*.py`:
- `run_cfg_attack.py`
- `eval_attacked_full.py`
- `eval_attacked_full_m2track.py`
- `analyze_backbone_feature_distribution.py`
- `analyze_backbone_feature_distribution_m2track.py`
- `visualize_20_attacked_samples.py`
- `visualize_20_random_nonconsecutive.py`
- `visualize_20_random_nonconsecutive_m2track.py`
- `stagea_exporter.py`
- `feature_direction_test_utils.py`

## Notes

- Attack perturbs **search points only**.
- Perturbation is restricted to target-region points (`seg_label`).
- Geometric constraints include Chamfer and KNN-local consistency.
