import argparse
import json
import os

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import get_model
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap
import sys
sys.path.insert(0, '/workspace/Open3DSOT/Open3DSOT')
from my_attack.evaluation.eval_attacked_full_m2track import (
    attack_one_frame_m2track, build_model, evaluate_one_sequence_attacked, load_yaml, parse_args
)


def main():
    parser = argparse.ArgumentParser('Quick M2Track eval')
    parser.add_argument('--cfg', type=str, default='cfgs/M2_track_kitti.yaml')
    parser.add_argument('--checkpoint', type=str, default='pretrained_models/mmtrack_kitti_car.ckpt')
    parser.add_argument('--split', type=str, default='test_tiny')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_sequences', type=int, default=2)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--k_ratio', type=float, default=0.2)
    parser.add_argument('--beta_cd', type=float, default=0.1)
    parser.add_argument('--gamma_knn', type=float, default=0.1)
    parser.add_argument('--knn_k', type=int, default=8)
    parser.add_argument('--out_json', type=str, default='my_attack/outputs/m2track_quick.json')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_data = load_yaml(args.cfg)
    cfg_data.update(vars(args))
    cfg_data.setdefault('preloading', False)
    cfg_data.setdefault('preload_offset', -1)
    cfg_data.setdefault('train_type', 'train_motion')
    cfg = EasyDict(cfg_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, args.checkpoint, device)

    test_data = get_dataset(cfg, type='test', split=args.split)
    # Only take first N sequences
    test_data.dataset.tracklet_anno_list = test_data.dataset.tracklet_anno_list[:args.max_sequences]
    test_data.dataset.tracklet_len_list = test_data.dataset.tracklet_len_list[:args.max_sequences]
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    success = TorchSuccess()
    precision = TorchPrecision()

    # Clean eval
    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    for batch in test_loader:
        sequence = batch[0]
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):
            this_bb = sequence[frame_id]['3d_bbox']
            if frame_id == 0:
                results_bbs.append(this_bb)
            else:
                data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)
                with torch.no_grad():
                    candidate_box = model.evaluate_one_sample(data_dict, ref_box=ref_bb)
                results_bbs.append(candidate_box)
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        success_clean(torch.tensor(ious, device=device))
        precision_clean(torch.tensor(distances, device=device))

    success_clean_score = float(success_clean.compute().detach().cpu().item())
    precision_clean_score = float(precision_clean.compute().detach().cpu().item())

    # Attacked eval
    for batch in test_loader:
        sequence = batch[0]
        ious, distances = evaluate_one_sequence_attacked(model, sequence, args)
        success(torch.tensor(ious, device=device))
        precision(torch.tensor(distances, device=device))

    success_score = float(success.compute().detach().cpu().item())
    precision_score = float(precision.compute().detach().cpu().item())

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        'clean_success': success_clean_score,
        'clean_precision': precision_clean_score,
        'attacked_success': success_score,
        'attacked_precision': precision_score,
        'success_drop': success_clean_score - success_score,
        'precision_drop': precision_clean_score - precision_score,
        'cfg': args.cfg,
        'max_sequences': args.max_sequences,
        'attack': {
            'iters': args.iters,
            'eps': args.eps,
            'alpha': args.alpha,
            'k_ratio': args.k_ratio,
            'beta_cd': args.beta_cd,
            'gamma_knn': args.gamma_knn,
            'knn_k': args.knn_k,
        },
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('=== M2Track Quick Evaluation Done ===')
    print(f'Clean success:   {success_clean_score:.6f}')
    print(f'Clean precision: {precision_clean_score:.6f}')
    print(f'Attacked success:   {success_score:.6f}')
    print(f'Attacked precision: {precision_score:.6f}')
    print(f'Success drop:   {success_clean_score - success_score:.6f}')
    print(f'Precision drop: {precision_clean_score - precision_score:.6f}')
    print(f'saved:     {args.out_json}')


if __name__ == '__main__':
    main()
