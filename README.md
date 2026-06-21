# 3DTrack Attack

This repository packages an Open3DSOT-based research workspace for authorized 3D single object tracking robustness experiments. It includes the original Open3DSOT tracker code used by the experiments, plus in-tree attack/evaluation code under `Open3DSOT/my_attack`.

The main tested workflows are BC-guided no-GT black-box attacks against M2Track and BAT on KITTI Car.

## Repository Layout

```text
Open3DSOT/
  main.py                         # Open3DSOT train/test entry point
  cfgs/                           # tracker configs for M2Track/BAT/P2B
  datasets/                       # dataset loaders
  models/                         # tracker models
  utils/                          # geometry and metrics utilities
  pointnet2/                      # project PointNet++ helpers
  Pointnet2_PyTorch/              # PointNet++ CUDA extension source
  my_attack/
    configs/                      # attack configs
    core/                         # attack primitives and fast tracker evaluators
    evaluation/                   # attack/evaluation entry points
    scripts/                      # batch/sharded experiment helpers
    analysis/                     # result merge and analysis helpers
    ppo_attack/                   # BC/PPO policy training utilities
```

Large local artifacts are intentionally excluded from Git:

```text
Open3DSOT/testing/
Open3DSOT/training/
Open3DSOT/val/
Open3DSOT/pretrained_models/
Open3DSOT/my_attack/outputs/
*.ckpt, *.pt, *.pth, *.pkl
```

Keep datasets, checkpoints, generated adversarial examples, and logs outside version control.

## Environment

The experiments in this workspace were run from the AutoDL `uv` virtual environment at the repository root:

```text
/workspace/Open3DSOT/.venv
```

Current environment snapshot:

```text
uv: 0.9.9
Python: 3.8.20
Python executable: /workspace/Open3DSOT/.venv/bin/python
pip module: not installed in the venv; use uv pip
torch: 2.3.0+cu121
CUDA used by torch: 12.1
GPU visible to torch: yes, 1 device
pytorch-lightning: 1.3.8
torchmetrics: 0.4.1
pointnet2_ops: 3.0.0
numpy: 1.24.4
scipy: 1.10.1
pandas: 1.1.5
protobuf: 5.29.5
pomegranate: 0.14.8
pyquaternion: 0.9.9
Shapely: 1.7.1
tensorboard: 2.14.0
nuscenes-devkit: 1.1.9
PyYAML: 6.0.3
```

To reproduce the same style of environment on AutoDL, create the venv from the repository root and install packages with `uv pip`:

```bash
cd /workspace/Open3DSOT
uv venv .venv --python 3.8.20
source .venv/bin/activate
```

Install PyTorch CUDA 12.1 wheels first:

```bash
uv pip install \
  torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install the Open3DSOT/runtime dependencies. The checked-in `Open3DSOT/requirement.txt` is the original project requirement file; this workspace uses newer compatible versions for several packages, so install the known working core set explicitly:

```bash
uv pip install \
  easydict==1.9 \
  numpy==1.24.4 \
  pandas==1.1.5 \
  scipy==1.10.1 \
  protobuf==5.29.5 \
  pomegranate==0.14.8 \
  pyquaternion==0.9.9 \
  pytorch-lightning==1.3.8 \
  torchmetrics==0.4.1 \
  PyYAML==6.0.3 \
  Shapely==1.7.1 \
  tensorboard==2.14.0 \
  tqdm==4.61.1 \
  nuscenes-devkit==1.1.9 \
  scikit-learn==1.3.2 \
  matplotlib==3.7.5 \
  open3d==0.19.0
```

Build the vendored PointNet++ CUDA op inside the same `uv` environment:

```bash
cd /workspace/Open3DSOT/Open3DSOT/Pointnet2_PyTorch/pointnet2_ops_lib
uv pip install -e .
```

Then return to the project root for all experiment commands:

```bash
cd /workspace/Open3DSOT/Open3DSOT
```

Quick environment check:

```bash
/workspace/Open3DSOT/.venv/bin/python - <<PY
import torch
import pytorch_lightning
import torchmetrics
import numpy
import scipy
import pandas
import pointnet2_ops

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("pytorch_lightning", pytorch_lightning.__version__)
print("torchmetrics", torchmetrics.__version__)
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__)
print("pointnet2_ops", pointnet2_ops.__version__)
PY
```

Conda can also work, but the documented/reproduced setup for this workspace is the AutoDL `uv` environment above.

## Data And Checkpoints

For KITTI tracking, prepare the dataset in the standard Open3DSOT/KITTI layout:

```text
/path/to/kitti_tracking/
  calib/
  label_02/
  velodyne/
```

Most local experiments in this workspace used a prepared path like:

```text
Open3DSOT/testing/
```

Place tracker checkpoints under:

```text
Open3DSOT/pretrained_models/
  mmtrack_kitti_car.ckpt
  bat_kitti_car.ckpt
```

Place the BC point-ranker checkpoint under a local output directory, for example:

```text
Open3DSOT/my_attack/outputs/point_ranker_bc_1024_e10/best.pt
```

Those files are required to reproduce the attack runs, but they are not committed because they are large/generated artifacts.

## Sanity Check: Tracker Evaluation

M2Track KITTI Car:

```bash
python main.py \
  --cfg cfgs/M2_track_kitti.yaml \
  --checkpoint pretrained_models/mmtrack_kitti_car.ckpt \
  --test
```

BAT KITTI Car:

```bash
python main.py \
  --cfg cfgs/BAT_Car.yaml \
  --checkpoint pretrained_models/bat_kitti_car.ckpt \
  --test
```

If your dataset is not in the path encoded in the YAML config, edit the `path` field in the corresponding config file.

## BC-Guided No-GT Attack

The main strict fast M2Track command used for small KITTI Car experiments is:

```bash
python my_attack/evaluation/eval_progressive_diffusion_attack_v2_bc_nogt.py \
  --cfg cfgs/M2_track_kitti.yaml \
  --checkpoint pretrained_models/mmtrack_kitti_car.ckpt \
  --attack_cfg my_attack/configs/refbox_m2_original_params.yaml \
  --policy_checkpoint my_attack/outputs/point_ranker_bc_1024_e10/best.pt \
  --out_dir my_attack/outputs/m2_bc_guided_v2_fast_compat_strict_3seq20 \
  --data_path /path/to/kitti_tracking_or_prepared_testing \
  --split test \
  --max_sequences 3 \
  --sequence_start 0 \
  --sequence_count -1 \
  --max_frames_per_sequence 20 \
  --bc_top_k 3 \
  --patch_candidate_k 2 \
  --candidate_directions +x,-x,+y,-y \
  --disable_fake_points \
  --disable_drop_ops \
  --disable_score \
  --regularization_mode source_cover \
  --fast
```

The default no-GT failure thresholds in `my_attack/configs/refbox_m2_original_params.yaml` are:

```yaml
iou_failure_threshold: 0.01
center_error_failure_threshold: 4.0
score_failure_threshold: null
```

Outputs are written to the chosen `--out_dir`, including:

```text
summary.json
per_frame.jsonl
adv_npz/
```

## Vectorized Evaluator

A vectorized evaluator is available:

```bash
python my_attack/evaluation/eval_progressive_diffusion_attack_v2_bc_nogt_vectorized.py \
  --cfg cfgs/M2_track_kitti.yaml \
  --checkpoint pretrained_models/mmtrack_kitti_car.ckpt \
  --attack_cfg my_attack/configs/refbox_m2_original_params.yaml \
  --policy_checkpoint my_attack/outputs/point_ranker_bc_1024_e10/best.pt \
  --out_dir my_attack/outputs/m2_bc_vectorized_3seq20 \
  --data_path /path/to/kitti_tracking_or_prepared_testing \
  --split test \
  --max_sequences 3 \
  --sequence_start 0 \
  --sequence_count -1 \
  --max_frames_per_sequence 20 \
  --bc_top_k 3 \
  --patch_candidate_k 2 \
  --candidate_directions +x,-x,+y,-y \
  --disable_fake_points \
  --disable_drop_ops \
  --disable_score \
  --regularization_mode source_cover \
  --vectorized_sequences 4 \
  --vectorized_max_batch 64 \
  --strict_equivalence
```

Important: multi-sequence vectorization can change floating-point/batch behavior and may not be strictly identical to the sequential fast path. Use the sequential command above for strict reproducibility. Use vectorized runs as an experimental acceleration path and compare `per_frame.jsonl` before reporting strict results.

## Sharded Full-Test Runs

For full test evaluation without changing attack logic, prefer sequence-level sharding. Different sequences are independent, so this is the safest way to use multiple GPUs or machines.

The sharding wrapper launches `N` normal evaluator processes over disjoint sequence ranges. Put evaluator arguments after `--`:

```bash
python my_attack/scripts/run_sharded_bc_nogt_eval.py \
  --out_dir my_attack/outputs/m2_fulltest_sharded \
  --num_processes 4 \
  --cuda_visible_devices 0 \
  -- \
  --cfg cfgs/M2_track_kitti.yaml \
  --checkpoint pretrained_models/mmtrack_kitti_car.ckpt \
  --attack_cfg my_attack/configs/refbox_m2_original_params.yaml \
  --policy_checkpoint my_attack/outputs/point_ranker_bc_1024_e10/best.pt \
  --data_path /path/to/kitti_tracking_or_prepared_testing \
  --split test \
  --max_sequences -1 \
  --sequence_start 0 \
  --sequence_count -1 \
  --max_frames_per_sequence -1 \
  --bc_top_k 3 \
  --patch_candidate_k 2 \
  --candidate_directions +x,-x,+y,-y \
  --disable_fake_points \
  --disable_drop_ops \
  --disable_score \
  --regularization_mode source_cover \
  --fast
```

The wrapper merges successful shards automatically into:

```text
my_attack/outputs/m2_fulltest_sharded/merged/
```

To merge manually:

```bash
python my_attack/analysis/merge_sharded_bc_nogt_eval.py \
  --shard_root my_attack/outputs/m2_fulltest_sharded \
  --out_dir my_attack/outputs/m2_fulltest_merged
```

## Optional Reward Plateau Stop

The BC no-GT evaluator contains an optional reward plateau early stop. It is disabled by default because it changes the search stopping behavior and can weaken attacks.

Enable it only for speed/ablation experiments:

```bash
--reward_early_stop \
--reward_lambda_iou 10.0 \
--reward_patience 6 \
--reward_min_improvement 0.02 \
--reward_warmup_steps 12
```

Do not use it when strict comparability with the baseline BC attack is required.

## Notes For Reproducibility

- Run commands from `Open3DSOT/`.
- Keep dataset paths and checkpoints local.
- Record `summary.json`, `per_frame.jsonl`, command lines, thresholds, and checkpoint names for each experiment.
- For strict comparisons, avoid changing `bc_top_k`, candidate directions, operators, no-GT thresholds, score usage, or early-stop settings.
- BAT/P2B can be sensitive to batch shape; prefer the sequential strict path unless explicitly testing batching behavior.

## Security And Use

This code is for authorized robustness research on 3D single object tracking systems. Do not use it against systems or data you do not have permission to evaluate.
