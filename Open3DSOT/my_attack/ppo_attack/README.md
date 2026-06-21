# Supervised Attack-Score Weight Policy

This directory contains an isolated scaffold for training a dynamic no-score
attack-score weight policy.  The supervised model is intended to warm-start a
later PPO fine-tuning stage.

The intended workflow is:

1. Run `export_v2_teacher_dataset.py` on the training split.  It first collects
   v2 teacher candidates, then writes the strongest/stealthiest selected subset.
2. Use `bc_pretrain.py` to train `WeightActorCritic` with candidate ranking.
3. Use the saved checkpoint to initialize a future PPO actor-critic.

The policy predicts only score weights:

- positive no-GT attack score weights:
  - `pred_drift`
  - `yaw_drift`
  - `drift_consistency`
- negative no-GT attack score weights:
  - `chamfer_distance`
  - `avg_point_displacement`
  - `fake_point_ratio`
  - `removed_point_ratio`
  - `local_density_diff`

The policy does not directly output an attack type.  Attack types are chosen
indirectly by scoring the generated candidates and selecting the highest score.

Supervised records are step-level JSONL rows with `obs`, `candidates`,
`best_candidate_index`, and the selected candidate's attack/stealth scores.
The teacher label is produced with v2 GT metrics, but the model inputs and
candidate features do not include GT IoU, GT center error, `score`, or
`score_drop`.

Example export:

```bash
PYTHONPATH=/workspace/Open3DSOT/Open3DSOT \
/workspace/Open3DSOT/.venv/bin/python \
Open3DSOT/my_attack/ppo_attack/export_v2_teacher_dataset.py \
  --cfg Open3DSOT/cfgs/M2_track_kitti.yaml \
  --checkpoint Open3DSOT/pretrained_models/mmtrack_kitti_car.ckpt \
  --split train \
  --max_steps 6 \
  --select_top_k 2000 \
  --raw_jsonl Open3DSOT/my_attack/outputs/ppo_attack/v2_teacher_raw.jsonl \
  --out_jsonl Open3DSOT/my_attack/outputs/ppo_attack/v2_teacher_selected.jsonl
```

By default, selected samples prefer coming from the same sequence when that
sequence has enough high-quality records.  Use `--no_prefer_same_sequence` to
select globally by score.

For cross-model data, export raw selected records separately for BAT and
M2Track, then keep only records whose selected candidate succeeds on both:

```bash
PYTHONPATH=/workspace/Open3DSOT/Open3DSOT \
/workspace/Open3DSOT/.venv/bin/python \
Open3DSOT/my_attack/ppo_attack/select_cross_model_records.py \
  --bat_jsonl Open3DSOT/my_attack/outputs/ppo_attack/bat_teacher_selected.jsonl \
  --m2track_jsonl Open3DSOT/my_attack/outputs/ppo_attack/m2_teacher_selected.jsonl \
  --top_k 1000 \
  --out_jsonl Open3DSOT/my_attack/outputs/ppo_attack/cross_model_train.jsonl \
  --paired_out_jsonl Open3DSOT/my_attack/outputs/ppo_attack/cross_model_pairs.jsonl
```

This selector matches records by `sequence_id/frame_id/step`, requires both
models' selected candidates to satisfy `attack_success=True`, ranks by minimum
attack effect across the two models and mean stealth, and still prefers samples
from one sequence by default.
