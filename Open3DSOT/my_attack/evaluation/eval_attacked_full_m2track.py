import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset, points_utils
from datasets.data_classes import PointCloud
from models import get_model
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap


def load_yaml(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser('M2Track attacked full-dataset evaluation')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--k_ratio', type=float, default=0.3)
    parser.add_argument('--beta_cd', type=float, default=0.05)
    parser.add_argument('--gamma_knn', type=float, default=0.05)
    parser.add_argument('--knn_k', type=int, default=8)

    parser.add_argument('--out_json', type=str, default='/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/m2track_attacked_full_metrics_iters100.json')
    return parser.parse_args()


def build_model(cfg, checkpoint, device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


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


def attack_one_frame_m2track(model, data_dict, this_bb, ref_box, args):
    device = model.device
    points = data_dict['points'].detach().clone().to(device)           # [1,2048,5]
    candidate_bc = data_dict.get('candidate_bc', None)
    if candidate_bc is not None:
        candidate_bc = candidate_bc.detach().clone().to(device)

    n_total = points.shape[1]
    n_half = n_total // 2

    clean_curr_xyz = points[:, n_half:, :3].detach()
    clean_curr_feat = points[:, n_half:, 3:].detach()

    # GT center in ref-box coordinate (same coordinate as current points in input_dict)
    gt_box_ref = points_utils.transform_box(this_bb, ref_box)
    c_gt = torch.tensor(gt_box_ref.center, dtype=torch.float32, device=device).unsqueeze(0)

    # target mask on current-frame points
    curr_xyz_np = clean_curr_xyz[0].detach().cpu().numpy()  # [N,3]
    curr_pc = PointCloud(curr_xyz_np.T)
    target_mask_np = points_utils.get_in_box_mask(curr_pc, gt_box_ref).astype(np.float32)
    target_mask = torch.from_numpy(target_mask_np).to(device).unsqueeze(0)
    if target_mask.sum().item() < 1:
        target_mask = torch.ones_like(target_mask)

    delta = torch.zeros_like(clean_curr_xyz)
    knn_idx, knn_dist = build_knn_reference(clean_curr_xyz, args.knn_k)

    for _ in range(args.iters):
        adv_curr_xyz = (clean_curr_xyz + delta).detach().requires_grad_(True)

        adv_points = points.clone()
        adv_points[:, n_half:, :3] = adv_curr_xyz
        adv_points[:, n_half:, 3:] = clean_curr_feat

        inp = {'points': adv_points}
        if candidate_bc is not None:
            inp['candidate_bc'] = candidate_bc
        out = model(inp)

        est_center = out['estimation_boxes'][:, :3]
        center_err = (est_center - c_gt).norm(p=2, dim=1).mean()

        seg_logits = out['seg_logits'][:, :, n_half:]  # [1,2,1024]
        fg_prob = F.softmax(seg_logits, dim=1)[:, 1, :]
        score_gt = (fg_prob * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1.0)

        l_adv = center_err - score_gt.mean()
        l_cd = chamfer_distance(adv_curr_xyz, clean_curr_xyz).mean()
        l_knn = knn_consistency_loss(adv_curr_xyz, clean_curr_xyz, knn_idx, knn_dist)
        objective = l_adv - args.beta_cd * l_cd - args.gamma_knn * l_knn

        grad = torch.autograd.grad(objective, adv_curr_xyz, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None or grad.abs().sum().item() < 1e-12:
            grad = F.normalize(adv_curr_xyz - c_gt.unsqueeze(1), p=2, dim=-1, eps=1e-12)

        imp = grad.norm(p=2, dim=-1) * target_mask
        if imp.abs().sum().item() < 1e-12:
            imp = target_mask
        n_pts = imp.shape[1]
        k = max(1, int(round(n_pts * args.k_ratio)))
        idx = torch.topk(imp, k=k, dim=1, largest=True, sorted=False).indices
        critical_mask = torch.zeros_like(imp, dtype=torch.bool)
        critical_mask.scatter_(1, idx, True)
        critical_mask = critical_mask & (target_mask > 0)

        delta = delta + args.alpha * critical_mask.float().unsqueeze(-1) * grad.sign()
        delta = torch.clamp(delta, -args.eps, args.eps)
        delta = delta * target_mask.unsqueeze(-1)

    adv_points = points.clone()
    adv_points[:, n_half:, :3] = (clean_curr_xyz + delta).detach()
    adv_points[:, n_half:, 3:] = clean_curr_feat

    out_dict = {'points': adv_points}
    if candidate_bc is not None:
        out_dict['candidate_bc'] = candidate_bc
    return out_dict


def evaluate_one_sequence_attacked(model, sequence, args):
    ious = []
    distances = []
    results_bbs = []

    for frame_id in range(len(sequence)):
        this_bb = sequence[frame_id]['3d_bbox']
        if frame_id == 0:
            results_bbs.append(this_bb)
        else:
            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)
            adv_input = attack_one_frame_m2track(model, data_dict, this_bb, ref_bb, args)

            with torch.no_grad():
                candidate_box = model.evaluate_one_sample(adv_input, ref_box=ref_bb)
            results_bbs.append(candidate_box)

        this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        ious.append(this_overlap)
        distances.append(this_accuracy)

    return ious, distances


def main():
    args = parse_args()
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
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    success = TorchSuccess()
    precision = TorchPrecision()

    for batch in tqdm(test_loader, desc='M2Track Attacked Eval', total=len(test_loader)):
        sequence = batch[0]
        ious, distances = evaluate_one_sequence_attacked(model, sequence, args)
        success(torch.tensor(ious, device=device))
        precision(torch.tensor(distances, device=device))

    success_score = float(success.compute().detach().cpu().item())
    precision_score = float(precision.compute().detach().cpu().item())

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        'success': success_score,
        'precision': precision_score,
        'cfg': args.cfg,
        'checkpoint': args.checkpoint,
        'split': args.split,
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

    print('=== M2Track Attacked Full Evaluation Done ===')
    print(f'success:   {success_score:.6f}')
    print(f'precision: {precision_score:.6f}')
    print(f'saved:     {args.out_json}')


if __name__ == '__main__':
    main()
