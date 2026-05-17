import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from datasets.data_classes import PointCloud
from datasets import points_utils
from models import get_model
from my_attack.core.critical_feature_guided_attack import AttackConfig, main_attack_loop
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap


def load_yaml(file_name: str) -> Dict:
    with open(file_name, "r", encoding="utf-8") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser("Evaluate attacked full dataset precision/success")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--k_ratio", type=float, default=0.2)
    parser.add_argument("--lambda_match", type=float, default=0.0)
    parser.add_argument("--lambda_offset", type=float, default=5.0)
    parser.add_argument("--lambda_cfg", type=float, default=0.5)
    parser.add_argument("--lambda_ms", type=float, default=1.0)
    parser.add_argument("--lambda_mg", type=float, default=0.1)
    parser.add_argument("--beta_cd", type=float, default=0.1)
    parser.add_argument("--gamma_knn", type=float, default=0.1)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--surface_constraint", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay_step", type=int, default=20)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)

    parser.add_argument("--out_json", type=str, default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/attacked_full_metrics.json")
    return parser.parse_args()


def build_model(cfg: EasyDict, checkpoint: str, device: torch.device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def _build_target_from_gt(
    data_dict: Dict[str, torch.Tensor],
    this_bb,
    ref_bb,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    search_box = points_utils.transform_box(this_bb, ref_bb)
    c_gt = torch.tensor(search_box.center, dtype=torch.float32, device=device).unsqueeze(0)

    search_points = data_dict["search_points"][0].detach().cpu().numpy()  # [N,3]
    search_pc = PointCloud(search_points.T)
    target_mask_np = points_utils.get_in_box_mask(search_pc, search_box).astype(np.float32)
    target_mask = torch.from_numpy(target_mask_np).to(device).unsqueeze(0).bool()

    return c_gt, target_mask


def evaluate_one_sequence_clean(model, sequence):
    ious = []
    distances = []
    results_bbs = []

    for frame_id in range(len(sequence)):
        this_bb = sequence[frame_id]["3d_bbox"]
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

    return ious, distances


def evaluate_one_sequence_attacked(model, sequence, attack_cfg: AttackConfig):
    ious = []
    distances = []
    results_bbs = []

    for frame_id in range(len(sequence)):
        this_bb = sequence[frame_id]["3d_bbox"]

        if frame_id == 0:
            results_bbs.append(this_bb)
        else:
            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)
            c_gt, target_mask = _build_target_from_gt(data_dict, this_bb, ref_bb, model.device)

            if target_mask.sum().item() > 0:
                attacked = main_attack_loop(
                    model=model,
                    input_dict=data_dict,
                    c_gt=c_gt,
                    target_mask=target_mask,
                    attack_cfg=attack_cfg,
                )
                data_dict_adv = dict(data_dict)
                data_dict_adv["search_points"] = attacked["S_adv"].detach()
            else:
                data_dict_adv = data_dict

            with torch.no_grad():
                candidate_box = model.evaluate_one_sample(data_dict_adv, ref_box=ref_bb)
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
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    cfg = EasyDict(cfg_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)

    test_data = get_dataset(cfg, type="test", split=args.split)
    if args.max_sequences > 0:
        test_data.dataset.tracklet_anno_list = test_data.dataset.tracklet_anno_list[:args.max_sequences]
        test_data.dataset.tracklet_len_list = test_data.dataset.tracklet_len_list[:args.max_sequences]
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    attack_cfg = AttackConfig(
        eps=args.eps,
        iters=args.iters,
        k_ratio=args.k_ratio,
        lambda_match=args.lambda_match,
        lambda_offset=args.lambda_offset,
        lambda_cfg=args.lambda_cfg,
        lambda_ms=args.lambda_ms,
        lambda_mg=args.lambda_mg,
        beta_cd=args.beta_cd,
        gamma_knn=args.gamma_knn,
        knn_k=args.knn_k,
        surface_constraint=args.surface_constraint,
        lr=args.lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
    )

    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_adv = TorchSuccess()
    precision_adv = TorchPrecision()

    for batch in tqdm(test_loader, desc="Eval", total=len(test_loader)):
        sequence = batch[0]

        # Clean eval
        ious_c, distances_c = evaluate_one_sequence_clean(model, sequence)
        success_clean(torch.tensor(ious_c, device=device))
        precision_clean(torch.tensor(distances_c, device=device))

        # Attacked eval
        ious_a, distances_a = evaluate_one_sequence_attacked(model, sequence, attack_cfg)
        success_adv(torch.tensor(ious_a, device=device))
        precision_adv(torch.tensor(distances_a, device=device))

    clean_s = float(success_clean.compute().detach().cpu().item())
    clean_p = float(precision_clean.compute().detach().cpu().item())
    adv_s = float(success_adv.compute().detach().cpu().item())
    adv_p = float(precision_adv.compute().detach().cpu().item())

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        "clean_success": clean_s,
        "clean_precision": clean_p,
        "attacked_success": adv_s,
        "attacked_precision": adv_p,
        "success_drop": clean_s - adv_s,
        "precision_drop": clean_p - adv_p,
        "attack": {
            "eps": args.eps,
            "iters": args.iters,
            "k_ratio": args.k_ratio,
            "lambda_match": args.lambda_match,
            "lambda_offset": args.lambda_offset,
            "lambda_cfg": args.lambda_cfg,
            "beta_cd": args.beta_cd,
            "gamma_knn": args.gamma_knn,
            "knn_k": args.knn_k,
            "surface_constraint": args.surface_constraint,
            "lr": args.lr,
            "lr_decay_step": args.lr_decay_step,
            "lr_decay_gamma": args.lr_decay_gamma,
        },
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_sequences": args.max_sequences,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== Full Sequence Evaluation Done ===")
    print(f"Clean success:    {clean_s:.6f}")
    print(f"Clean precision:  {clean_p:.6f}")
    print(f"Attacked success: {adv_s:.6f}")
    print(f"Attacked precision: {adv_p:.6f}")
    print(f"Success drop:     {clean_s - adv_s:.6f}")
    print(f"Precision drop:   {clean_p - adv_p:.6f}")
    print(f"saved:            {args.out_json}")


if __name__ == "__main__":
    main()
