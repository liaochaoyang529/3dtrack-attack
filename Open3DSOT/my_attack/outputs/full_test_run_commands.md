# Full Test Run Commands

Working directory for both commands:

```bash
cd /workspace/Open3DSOT/Open3DSOT
```

## M2Track BC no-GT fast full test

Note: `m2_bc_guided_v2_fast_compat_strict_testing_full` does not contain a
`summary.json` or `run.log`. This command is reconstructed from the matching
`m2_bc_guided_v2_fast_compat_strict_3seq20/summary.json` configuration and the
full-test `per_frame.jsonl`.

```bash
python my_attack/evaluation/eval_progressive_diffusion_attack_v2_bc_nogt.py \
  --cfg cfgs/M2_track_kitti.yaml \
  --checkpoint pretrained_models/mmtrack_kitti_car.ckpt \
  --attack_cfg my_attack/configs/refbox_m2_original_params.yaml \
  --policy_checkpoint my_attack/outputs/point_ranker_bc_1024_e10/best.pt \
  --out_dir my_attack/outputs/m2_bc_guided_v2_fast_compat_strict_testing_full \
  --data_path /workspace/Open3DSOT/Open3DSOT/testing \
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

Observed output coverage:

- Output: `my_attack/outputs/m2_bc_guided_v2_fast_compat_strict_testing_full`
- Frames: `2770`
- Sequences: `24`
- Clean success / adv success / drop: `70.43 / 4.26 / 66.18`
- Clean precision / adv precision / drop: `83.84 / 3.18 / 80.66`

## BAT BC no-GT fast full test

This command is recovered from
`bc_guided_v2_sourcecover_no_fake_no_drop_fast_noscore_topk3_patch2_dir4_testing_full/summary.json`.

```bash
python my_attack/evaluation/eval_progressive_diffusion_attack_v2_bc_nogt.py \
  --cfg cfgs/BAT_Car.yaml \
  --checkpoint pretrained_models/bat_kitti_car.ckpt \
  --attack_cfg my_attack/configs/refbox_m2_original_params.yaml \
  --policy_checkpoint my_attack/outputs/point_ranker_bc_1024_e10/best.pt \
  --out_dir my_attack/outputs/bc_guided_v2_sourcecover_no_fake_no_drop_fast_noscore_topk3_patch2_dir4_testing_full \
  --data_path /workspace/Open3DSOT/Open3DSOT/testing \
  --split test \
  --max_sequences -1 \
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

Observed output coverage:

- Output: `my_attack/outputs/bc_guided_v2_sourcecover_no_fake_no_drop_fast_noscore_topk3_patch2_dir4_testing_full`
- Frames: `6424`
- Sequences: `120`
- Clean success / adv success / drop: `64.75 / 8.47 / 56.29`
- Clean precision / adv precision / drop: `78.10 / 8.54 / 69.56`
