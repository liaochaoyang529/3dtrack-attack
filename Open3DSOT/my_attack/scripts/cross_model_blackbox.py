import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

sys.path.insert(0, '/workspace/Open3DSOT/Open3DSOT')
from datasets import get_dataset, points_utils
from datasets.data_classes import PointCloud
from models import get_model
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap

from my_attack.core.critical_feature_guided_attack import (
    AttackConfig,
    main_attack_loop,
)


def load_yaml(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def build_model(cfg, checkpoint, device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser('Cross-model black-box: BAT attack -> M2Track eval')
    parser.add_argument('--source_cfg', type=str, default='cfgs/BAT_Car.yaml')
    parser.add_argument('--source_ckpt', type=str, default='pretrained_models/bat_kitti_car.ckpt')
    parser.add_argument('--target_cfg', type=str, default='cfgs/M2_track_kitti.yaml')
    parser.add_argument('--target_ckpt', type=str, default='pretrained_models/mmtrack_kitti_car.ckpt')
    parser.add_argument('--split', type=str, default='test_tiny')
    parser.add_argument('--max_sequences', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--k_ratio', type=float, default=0.2)
    parser.add_argument('--lambda_cfg', type=float, default=0.5)
    parser.add_argument('--beta_cd', type=float, default=0.1)
    parser.add_argument('--gamma_knn', type=float, default=0.1)
    parser.add_argument('--knn_k', type=int, default=8)

    parser.add_argument('--out_json', type=str, default='my_attack/outputs/cross_model_bb.json')
    return parser.parse_args()


def attack_current_frame_with_bat(bat_model, curr_xyz, prev_xyz, cfg_bat, device, args):
    """Use BAT to attack current frame points, treating prev frame as template."""
    bsz = curr_xyz.shape[0]
    n_curr = curr_xyz.shape[1]
    n_prev = prev_xyz.shape[1]

    # Build a synthetic siamese input for BAT
    # Template: prev frame points, padded/truncated to template_size
    template_size = getattr(cfg_bat, 'template_size', 512)
    search_size = getattr(cfg_bat, 'search_size', 1024)

    # Regularize prev points to template_size
    prev_np = prev_xyz[0].detach().cpu().numpy()
    if prev_np.shape[0] >= template_size:
        idx = np.random.choice(prev_np.shape[0], template_size, replace=False)
        template_points = prev_np[idx]
    else:
        idx = np.random.choice(prev_np.shape[0], template_size, replace=True)
        template_points = prev_np[idx]
    template_points = torch.from_numpy(template_points).float().unsqueeze(0).to(device)

    # Regularize curr points to search_size
    curr_np = curr_xyz[0].detach().cpu().numpy()
    if curr_np.shape[0] >= search_size:
        idx = np.random.choice(curr_np.shape[0], search_size, replace=False)
        search_points = curr_np[idx]
    else:
        idx = np.random.choice(curr_np.shape[0], search_size, replace=True)
        search_points = curr_np[idx]
    search_points = torch.from_numpy(search_points).float().unsqueeze(0).to(device)

    # Dummy box_label and seg_label centered at origin
    box_label = torch.zeros((1, 4), dtype=torch.float32, device=device)
    seg_label = torch.ones((1, search_size), dtype=torch.float32, device=device) * 0.5
    target_mask = torch.ones((1, search_size), dtype=torch.bool, device=device)

    input_dict = {
        'template_points': template_points,
        'search_points': search_points,
    }
    if getattr(cfg_bat, 'box_aware', False):
        # Dummy box cloud distances (bc_channel=9 for BAT)
        bc_channel = getattr(cfg_bat, 'bc_channel', 9)
        input_dict['points2cc_dist_t'] = torch.zeros((1, template_size, bc_channel), dtype=torch.float32, device=device)

    attack_cfg = AttackConfig(
        eps=args.eps,
        alpha=args.alpha,
        iters=args.iters,
        k_ratio=args.k_ratio,
        lambda_match=1.0,
        lambda_offset=1.0,
        lambda_cfg=args.lambda_cfg,
        beta_cd=args.beta_cd,
        gamma_knn=args.gamma_knn,
        knn_k=args.knn_k,
        surface_constraint=False,
    )

    result = main_attack_loop(
        model=bat_model,
        input_dict=input_dict,
        c_gt=box_label[:, :3],
        target_mask=target_mask,
        attack_cfg=attack_cfg,
    )
    adv_search = result['S_adv']
    delta = adv_search - search_points
    return delta


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load source model (BAT)
    cfg_bat = EasyDict(load_yaml(args.source_cfg))
    cfg_bat.update(vars(args))
    cfg_bat.setdefault('preloading', False)
    cfg_bat.setdefault('preload_offset', -1)
    bat_model = build_model(cfg_bat, args.source_ckpt, device)

    # Load target model (M2Track)
    cfg_m2 = EasyDict(load_yaml(args.target_cfg))
    cfg_m2.update(vars(args))
    cfg_m2.preloading = False
    cfg_m2.preload_offset = -1
    cfg_m2.train_type = 'train_motion'
    cfg_m2.path = '/workspace/Open3DSOT/Open3DSOT/testing'
    m2_model = build_model(cfg_m2, args.target_ckpt, device)

    # Load dataset for M2Track
    test_data = get_dataset(cfg_m2, type='test', split=args.split)
    test_data.dataset.tracklet_anno_list = test_data.dataset.tracklet_anno_list[:args.max_sequences]
    test_data.dataset.tracklet_len_list = test_data.dataset.tracklet_len_list[:args.max_sequences]
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_adv = TorchSuccess()
    precision_adv = TorchPrecision()

    for batch in test_loader:
        sequence = batch[0]
        clean_results_bbs = []
        adv_results_bbs = []

        for frame_id in range(len(sequence)):
            this_bb = sequence[frame_id]['3d_bbox']
            if frame_id == 0:
                clean_results_bbs.append(this_bb)
                adv_results_bbs.append(this_bb)
            else:
                data_dict, ref_bb = m2_model.build_input_dict(sequence, frame_id, clean_results_bbs)

                # Clean eval
                with torch.no_grad():
                    clean_box = m2_model.evaluate_one_sample(data_dict, ref_box=ref_bb)
                clean_results_bbs.append(clean_box)

                # Build adv input for M2Track
                points = data_dict['points'].to(device)
                n_total = points.shape[1]
                n_half = n_total // 2
                prev_xyz = points[:, :n_half, :3]
                curr_xyz = points[:, n_half:, :3]
                curr_feat = points[:, n_half:, 3:]

                # Use BAT to compute perturbation on current frame
                try:
                    delta = attack_current_frame_with_bat(bat_model, curr_xyz, prev_xyz, cfg_bat, device, args)
                    adv_curr = curr_xyz + delta
                except Exception as e:
                    print(f'BAT attack failed: {e}, using clean points')
                    adv_curr = curr_xyz

                adv_points = points.clone()
                adv_points[:, n_half:, :3] = adv_curr.detach()

                adv_dict = {'points': adv_points}
                if 'candidate_bc' in data_dict:
                    adv_dict['candidate_bc'] = data_dict['candidate_bc']

                with torch.no_grad():
                    adv_box = m2_model.evaluate_one_sample(adv_dict, ref_box=ref_bb)
                adv_results_bbs.append(adv_box)

            clean_overlap = estimateOverlap(this_bb, clean_results_bbs[-1], dim=m2_model.config.IoU_space, up_axis=m2_model.config.up_axis)
            clean_accuracy = estimateAccuracy(this_bb, clean_results_bbs[-1], dim=m2_model.config.IoU_space, up_axis=m2_model.config.up_axis)
            success_clean(torch.tensor([clean_overlap], device=device))
            precision_clean(torch.tensor([clean_accuracy], device=device))

            adv_overlap = estimateOverlap(this_bb, adv_results_bbs[-1], dim=m2_model.config.IoU_space, up_axis=m2_model.config.up_axis)
            adv_accuracy = estimateAccuracy(this_bb, adv_results_bbs[-1], dim=m2_model.config.IoU_space, up_axis=m2_model.config.up_axis)
            success_adv(torch.tensor([adv_overlap], device=device))
            precision_adv(torch.tensor([adv_accuracy], device=device))

    clean_s = float(success_clean.compute().detach().cpu().item())
    clean_p = float(precision_clean.compute().detach().cpu().item())
    adv_s = float(success_adv.compute().detach().cpu().item())
    adv_p = float(precision_adv.compute().detach().cpu().item())

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        'clean_success': clean_s,
        'clean_precision': clean_p,
        'adv_success': adv_s,
        'adv_precision': adv_p,
        'success_drop': clean_s - adv_s,
        'precision_drop': clean_p - adv_p,
        'max_sequences': args.max_sequences,
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('=== Cross-Model Black-Box (BAT -> M2Track) ===')
    print(f'Clean success:   {clean_s:.6f}')
    print(f'Clean precision: {clean_p:.6f}')
    print(f'Adv success:     {adv_s:.6f}')
    print(f'Adv precision:   {adv_p:.6f}')
    print(f'Success drop:    {clean_s - adv_s:.6f}')
    print(f'Precision drop:  {clean_p - adv_p:.6f}')
    print(f'saved: {args.out_json}')


if __name__ == '__main__':
    main()
