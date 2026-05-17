import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict

from datasets import get_dataset
from models import get_model


def load_yaml(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser('M2Track attacked random non-consecutive visualization')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--k_ratio', type=float, default=0.3)
    parser.add_argument('--beta_cd', type=float, default=0.05)
    parser.add_argument('--gamma_knn', type=float, default=0.05)
    parser.add_argument('--knn_k', type=int, default=8)

    parser.add_argument('--out_dir', type=str, default='/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/vis_m2track_car_strength200_20samples_nonconsecutive')
    return parser.parse_args()


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


def chamfer_distance(pc1, pc2):
    d = torch.cdist(pc1, pc2, p=2) ** 2
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


def build_knn_reference(points, k):
    pair = torch.cdist(points, points, p=2)
    vals, idx = torch.topk(pair, k=k + 1, dim=-1, largest=False, sorted=True)
    return idx[:, :, 1:], vals[:, :, 1:]


def knn_consistency_loss(adv_points, clean_points, knn_idx, clean_knn_dists):
    bsz, n, _ = adv_points.shape
    k = knn_idx.shape[-1]
    gather_idx = knn_idx.unsqueeze(-1).expand(bsz, n, k, 3)
    adv_neighbors = torch.gather(adv_points.unsqueeze(1).expand(bsz, n, n, 3), 2, gather_idx)
    adv_center = adv_points.unsqueeze(2)
    adv_dists = (adv_neighbors - adv_center).norm(p=2, dim=-1)
    return (adv_dists - clean_knn_dists).abs().mean()


def save_vis_png(clean_points, adv_points, delta, target_mask, critical_mask, out_path, title_suffix=''):
    pert = np.linalg.norm(delta, axis=1)

    if target_mask is not None and np.any(target_mask):
        clean_obj = clean_points[target_mask]
        adv_obj = adv_points[target_mask]
        pert_obj = pert[target_mask]
    else:
        clean_obj = clean_points
        adv_obj = adv_points
        pert_obj = pert

    clean_center = clean_obj.mean(axis=0, keepdims=True)
    clean_obj_c = clean_obj - clean_center
    adv_obj_c = adv_obj - clean_center

    fig = plt.figure(figsize=(16, 4.8), dpi=130)

    # Original current-frame cloud + important points overlay
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(clean_points[:, 0], clean_points[:, 1], clean_points[:, 2], s=3, c='lightgray', alpha=0.35)
    if target_mask is not None and np.any(target_mask):
        ax1.scatter(clean_points[target_mask, 0], clean_points[target_mask, 1], clean_points[target_mask, 2], s=7, c='deepskyblue', alpha=0.8)
    if critical_mask is not None and np.any(critical_mask):
        ax1.scatter(clean_points[critical_mask, 0], clean_points[critical_mask, 1], clean_points[critical_mask, 2], s=16, c='yellow', edgecolors='k', linewidths=0.25, alpha=0.95)
    ax1.set_title('Original + Important Points')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.scatter(clean_obj_c[:, 0], clean_obj_c[:, 1], clean_obj_c[:, 2], s=9, c='royalblue', alpha=0.95)
    ax2.set_title('Clean Object (Centered)')
    ax2.set_axis_off()

    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3.scatter(adv_obj_c[:, 0], adv_obj_c[:, 1], adv_obj_c[:, 2], s=9, c='crimson', alpha=0.95)
    ax3.set_title('Adversarial Object (Centered)')
    ax3.set_axis_off()

    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    pcol = ax4.scatter(adv_obj_c[:, 0], adv_obj_c[:, 1], adv_obj_c[:, 2], s=11, c=pert_obj, cmap='turbo', alpha=0.95)
    ax4.set_title('Object Perturbation Heatmap')
    ax4.set_axis_off()
    cb = fig.colorbar(pcol, ax=ax4, fraction=0.03, pad=0.02)
    cb.set_label('||delta||_2', rotation=90)

    linf = np.max(np.abs(delta))
    l2m = np.mean(np.linalg.norm(delta, axis=1))
    n_imp = int(np.sum(critical_mask)) if critical_mask is not None else 0
    fig.suptitle(f'M2Track attack iters=200 | {title_suffix} | important={n_imp} | L_inf={linf:.4f}, mean L2={l2m:.4f}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_model(cfg, checkpoint, device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def attack_m2track_sample(model, sample, device, eps=0.1, alpha=0.01, iters=200, k_ratio=0.3, beta_cd=0.05, gamma_knn=0.05, knn_k=8):
    points = torch.from_numpy(sample['points']).unsqueeze(0).float().to(device)  # [1, 2048, 5]
    candidate_bc = torch.from_numpy(sample['candidate_bc']).unsqueeze(0).float().to(device)
    seg_label = torch.from_numpy(sample['seg_label']).unsqueeze(0).to(device)     # [1, 2048]
    c_gt = torch.from_numpy(sample['box_label'][:3]).unsqueeze(0).float().to(device)

    n_total = points.shape[1]
    n_half = n_total // 2

    clean_curr_xyz = points[:, n_half:, :3].detach()  # current frame points only
    clean_curr_feat = points[:, n_half:, 3:].detach()

    target_mask = (seg_label[:, n_half:] > 0).float()
    if target_mask.sum().item() < 1:
        target_mask = torch.ones_like(target_mask)

    delta = torch.zeros_like(clean_curr_xyz)
    knn_idx, knn_dist = build_knn_reference(clean_curr_xyz, knn_k)
    critical_mask_last = target_mask.bool().clone()

    for _ in range(iters):
        adv_curr_xyz = (clean_curr_xyz + delta).detach().requires_grad_(True)

        adv_points = points.clone()
        adv_points[:, n_half:, :3] = adv_curr_xyz
        adv_points[:, n_half:, 3:] = clean_curr_feat

        inp = {'points': adv_points, 'candidate_bc': candidate_bc}
        out = model(inp)

        est_center = out['estimation_boxes'][:, :3]
        center_err = (est_center - c_gt).norm(p=2, dim=1).mean()

        seg_logits = out['seg_logits'][:, :, n_half:]  # [1,2,1024]
        fg_prob = F.softmax(seg_logits, dim=1)[:, 1, :]
        score_gt = (fg_prob * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1.0)

        l_adv = center_err - score_gt.mean()
        l_cd = chamfer_distance(adv_curr_xyz, clean_curr_xyz).mean()
        l_knn = knn_consistency_loss(adv_curr_xyz, clean_curr_xyz, knn_idx, knn_dist)
        objective = l_adv - beta_cd * l_cd - gamma_knn * l_knn

        grad = torch.autograd.grad(objective, adv_curr_xyz, retain_graph=False, create_graph=False)[0]
        if grad is None or grad.abs().sum().item() < 1e-12:
            grad = F.normalize(adv_curr_xyz - c_gt.unsqueeze(1), p=2, dim=-1, eps=1e-12)

        # Importance scores and critical-point selection
        imp = grad.norm(p=2, dim=-1) * target_mask
        if imp.abs().sum().item() < 1e-12:
            imp = target_mask
        n_pts = imp.shape[1]
        k = max(1, int(round(n_pts * k_ratio)))
        idx = torch.topk(imp, k=k, dim=1, largest=True, sorted=False).indices
        critical_mask = torch.zeros_like(imp, dtype=torch.bool)
        critical_mask.scatter_(1, idx, True)
        critical_mask = critical_mask & (target_mask > 0)
        critical_mask_last = critical_mask.detach().clone()

        # update only critical points
        delta = delta + alpha * critical_mask.float().unsqueeze(-1) * grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = delta * target_mask.unsqueeze(-1)

    adv_curr_xyz = (clean_curr_xyz + delta).detach()
    return (
        clean_curr_xyz[0].cpu().numpy(),
        adv_curr_xyz[0].cpu().numpy(),
        delta[0].cpu().numpy(),
        target_mask[0].cpu().numpy().astype(bool),
        critical_mask_last[0].cpu().numpy().astype(bool),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg_data = load_yaml(args.cfg)
    cfg_data.update(vars(args))
    cfg_data.setdefault('preloading', False)
    cfg_data.setdefault('preload_offset', -1)
    cfg_data.setdefault('train_type', 'train_motion')
    cfg_data['use_augmentation'] = False
    cfg = EasyDict(cfg_data)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, args.checkpoint, device)

    dataset = get_dataset(cfg, type='train_motion', split=args.split)
    total = len(dataset)
    pick_idxs = choose_nonconsecutive_indices(total, args.num_samples, args.seed)

    with open(os.path.join(args.out_dir, 'selected_indices.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(str, pick_idxs)))

    for i, ds_idx in enumerate(pick_idxs):
        sample = dataset[ds_idx]
        clean, adv, delta, tmask, cmask = attack_m2track_sample(
            model, sample, device,
            eps=args.eps, alpha=args.alpha, iters=args.iters, k_ratio=args.k_ratio,
            beta_cd=args.beta_cd, gamma_knn=args.gamma_knn, knn_k=args.knn_k,
        )

        out_png = os.path.join(args.out_dir, f'sample_{i:03d}_idx{ds_idx}.png')
        save_vis_png(clean, adv, delta, tmask, cmask, out_png, title_suffix=f'dataset_idx={ds_idx}')
        print(f'saved {out_png}')

    print(f'done, total saved: {len(pick_idxs)}')
    print('selected indices:', pick_idxs)


if __name__ == '__main__':
    main()
