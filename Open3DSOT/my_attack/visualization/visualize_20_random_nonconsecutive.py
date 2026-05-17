import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from easydict import EasyDict

from datasets import get_dataset
from models import get_model
from my_attack.critical_feature_guided_attack import AttackConfig, main_attack_loop


def load_yaml(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser('Visualize attacked non-consecutive random samples')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--k_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_match', type=float, default=1.0)
    parser.add_argument('--lambda_offset', type=float, default=1.0)
    parser.add_argument('--beta_cd', type=float, default=0.1)
    parser.add_argument('--gamma_knn', type=float, default=0.1)
    parser.add_argument('--knn_k', type=int, default=8)
    parser.add_argument('--proposal_temperature', type=float, default=10.0)

    parser.add_argument('--out_dir', type=str, default='/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/vis_strength200_20samples_nonconsecutive')
    return parser.parse_args()


def build_model(cfg, checkpoint, device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def save_vis_png(clean_points, adv_points, delta, target_mask, out_path, title_suffix=''):
    pert = np.linalg.norm(delta, axis=1)

    fig = plt.figure(figsize=(14, 4.8), dpi=130)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(clean_points[:, 0], clean_points[:, 1], clean_points[:, 2], s=4, c='royalblue', alpha=0.85)
    ax1.set_title('Clean Search')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(adv_points[:, 0], adv_points[:, 1], adv_points[:, 2], s=4, c='crimson', alpha=0.85)
    ax2.set_title('Adversarial Search')
    ax2.set_axis_off()

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    p = ax3.scatter(adv_points[:, 0], adv_points[:, 1], adv_points[:, 2], s=7, c=pert, cmap='turbo', alpha=0.95)
    if target_mask is not None and target_mask.sum() > 0:
        tpts = adv_points[target_mask]
        ax3.scatter(tpts[:, 0], tpts[:, 1], tpts[:, 2], s=9, c='lime', alpha=0.35)
    ax3.set_title('Perturbation Magnitude')
    ax3.set_axis_off()
    cb = fig.colorbar(p, ax=ax3, fraction=0.03, pad=0.02)
    cb.set_label('||delta||_2', rotation=90)

    linf = np.max(np.abs(delta))
    l2m = np.mean(np.linalg.norm(delta, axis=1))
    fig.suptitle(f'Attack iters=200 | {title_suffix} | L_inf={linf:.4f}, mean L2={l2m:.4f}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def choose_nonconsecutive_indices(total, k, seed):
    rng = random.Random(seed)
    chosen = []
    candidates = list(range(total))
    rng.shuffle(candidates)

    for idx in candidates:
        if all(abs(idx - c) > 1 for c in chosen):
            chosen.append(idx)
            if len(chosen) == k:
                break

    if len(chosen) < k:
        raise RuntimeError(f'Cannot find {k} non-consecutive indices from total={total}')

    return sorted(chosen)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg_data = load_yaml(args.cfg)
    cfg_data.update(vars(args))
    cfg_data.setdefault('preloading', False)
    cfg_data.setdefault('preload_offset', -1)
    cfg = EasyDict(cfg_data)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, args.checkpoint, device)

    dataset = get_dataset(cfg, type='train_siamese', split=args.split)
    total = len(dataset)
    pick_idxs = choose_nonconsecutive_indices(total, args.num_samples, args.seed)

    attack_cfg = AttackConfig(
        eps=args.eps,
        alpha=args.alpha,
        iters=args.iters,
        k_ratio=args.k_ratio,
        lambda_match=args.lambda_match,
        lambda_offset=args.lambda_offset,
        beta_cd=args.beta_cd,
        gamma_knn=args.gamma_knn,
        knn_k=args.knn_k,
        proposal_temperature=args.proposal_temperature,
    )

    with open(os.path.join(args.out_dir, 'selected_indices.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(str, pick_idxs)))

    for i, ds_idx in enumerate(pick_idxs):
        sample = dataset[ds_idx]

        input_dict = {
            'template_points': torch.from_numpy(sample['template_points']).unsqueeze(0).float().to(device),
            'search_points': torch.from_numpy(sample['search_points']).unsqueeze(0).float().to(device),
        }
        if 'points2cc_dist_t' in sample:
            input_dict['points2cc_dist_t'] = torch.from_numpy(sample['points2cc_dist_t']).unsqueeze(0).float().to(device)

        c_gt = torch.from_numpy(sample['box_label'][:3]).unsqueeze(0).float().to(device)
        target_mask = torch.from_numpy(sample['seg_label'] > 0.5).unsqueeze(0).to(device)

        attacked = main_attack_loop(
            model=model,
            input_dict=input_dict,
            c_gt=c_gt,
            target_mask=target_mask,
            attack_cfg=attack_cfg,
        )

        clean = input_dict['search_points'][0].detach().cpu().numpy()
        adv = attacked['S_adv'][0].detach().cpu().numpy()
        delta = attacked['delta'][0].detach().cpu().numpy()
        tmask = target_mask[0].detach().cpu().numpy().astype(bool)

        out_png = os.path.join(args.out_dir, f'sample_{i:03d}_idx{ds_idx}.png')
        save_vis_png(clean, adv, delta, tmask, out_png, title_suffix=f'dataset_idx={ds_idx}')
        print(f'saved {out_png}')

    print(f'done, total saved: {len(pick_idxs)}')
    print('selected indices:', pick_idxs)


if __name__ == '__main__':
    main()
