import argparse
import os
from typing import Dict

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import get_model
from my_attack.critical_feature_guided_attack import (
    AttackConfig,
    dump_attack_report,
    main_attack_loop,
)


def load_yaml(file_name: str) -> Dict:
    with open(file_name, "r", encoding="utf-8") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser("Batch evaluation for CFG attack")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=50)

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--k_ratio", type=float, default=0.2)
    parser.add_argument("--lambda_match", type=float, default=0.0)
    parser.add_argument("--lambda_offset", type=float, default=5.0)
    parser.add_argument("--lambda_pseudo_offset", type=float, default=5.0)
    parser.add_argument("--lambda_best_suppress", type=float, default=1.0)
    parser.add_argument("--lambda_margin", type=float, default=0.5)
    parser.add_argument("--lambda_score_suppress", type=float, default=0.5)
    parser.add_argument("--lambda_cfg", type=float, default=0.5)
    parser.add_argument("--beta_cd", type=float, default=0.1)
    parser.add_argument("--gamma_knn", type=float, default=0.1)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--pred_mask_threshold", type=float, default=0.5)
    parser.add_argument("--pred_mask_min_points", type=int, default=1)
    parser.add_argument("--use_gt_objective", action="store_true", default=False)
    parser.add_argument("--surface_constraint", action="store_true", default=False)

    parser.add_argument("--out_dir", type=str, default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs")
    parser.add_argument("--out_prefix", type=str, default="batch_eval")
    return parser.parse_args()


def build_model(cfg: EasyDict, checkpoint: str, device: torch.device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def prepare_batch_for_siamese(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v

    required = ["template_points", "search_points", "box_label", "seg_label"]
    for key in required:
        if key not in out:
            raise KeyError(f"Missing required key in batch: {key}")

    input_dict = {
        "template_points": out["template_points"].float(),
        "search_points": out["search_points"].float(),
    }
    if "points2cc_dist_t" in out:
        input_dict["points2cc_dist_t"] = out["points2cc_dist_t"].float()

    c_gt = out["box_label"][:, :3].float()
    target_mask = out["seg_label"] > 0.5
    return input_dict, c_gt, target_mask


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_data = load_yaml(args.cfg)
    cfg_data.update(vars(args))
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    # Auto-select data path based on split
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.cfg)))
    if args.split.lower() in ("valid", "val"):
        cfg_data["path"] = os.path.join(base_dir, "val")
    elif args.split.lower() in ("test", "testing"):
        cfg_data["path"] = os.path.join(base_dir, "testing")
    else:
        cfg_data.setdefault("path", os.path.join(base_dir, "training"))
    cfg = EasyDict(cfg_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, args.checkpoint, device)

    dataset = get_dataset(cfg, type="train_siamese", split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    attack_cfg = AttackConfig(
        eps=args.eps,
        alpha=args.alpha,
        iters=args.iters,
        k_ratio=args.k_ratio,
        lambda_match=args.lambda_match,
        lambda_offset=args.lambda_offset,
        lambda_pseudo_offset=args.lambda_pseudo_offset,
        lambda_best_suppress=args.lambda_best_suppress,
        lambda_margin=args.lambda_margin,
        lambda_score_suppress=args.lambda_score_suppress,
        lambda_cfg=args.lambda_cfg,
        beta_cd=args.beta_cd,
        gamma_knn=args.gamma_knn,
        knn_k=args.knn_k,
        pred_mask_threshold=args.pred_mask_threshold,
        pred_mask_min_points=args.pred_mask_min_points,
        use_gt_objective=args.use_gt_objective,
        surface_constraint=args.surface_constraint,
    )

    all_clean_err = []
    all_adv_err = []
    all_score_drop = []

    print(f"Running attack on up to {args.max_samples} samples...")
    for i, batch in enumerate(loader):
        if i >= args.max_samples:
            break

        input_dict, c_gt, target_mask = prepare_batch_for_siamese(batch, device)

        result = main_attack_loop(
            model=model,
            input_dict=input_dict,
            c_gt=c_gt,
            target_mask=target_mask,
            attack_cfg=attack_cfg,
        )

        clean_err = result["clean_center_error"].mean().item()
        adv_err = result["adv_center_error"].mean().item()
        score_drop = result["score_drop"].mean().item()

        all_clean_err.append(clean_err)
        all_adv_err.append(adv_err)
        all_score_drop.append(score_drop)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{args.max_samples} samples")

    # Statistics
    clean_err_mean = np.mean(all_clean_err)
    adv_err_mean = np.mean(all_adv_err)
    score_drop_mean = np.mean(all_score_drop)
    success_rate = np.mean([adv > clean for adv, clean in zip(all_adv_err, all_clean_err)])

    print("\n=== Batch Attack Evaluation ===")
    print(f"Samples evaluated:     {len(all_clean_err)}")
    print(f"Clean center error:    {clean_err_mean:.4f} ± {np.std(all_clean_err):.4f}")
    print(f"Adv center error:      {adv_err_mean:.4f} ± {np.std(all_adv_err):.4f}")
    print(f"Center error increase: {adv_err_mean - clean_err_mean:.4f}")
    print(f"Score drop:            {score_drop_mean:.4f} ± {np.std(all_score_drop):.4f}")
    print(f"Success rate (err↑):   {success_rate * 100:.1f}%")

    # Black-box simulation: re-evaluate adv sample with no_grad
    print("\n=== Black-box Simulation (no_grad re-eval) ===")
    bb_adv_err = []
    bb_clean_err = []
    from my_attack.core.critical_feature_guided_attack import _forward_with_intermediate, _compute_tracking_terms
    for i, batch in enumerate(loader):
        if i >= args.max_samples:
            break
        input_dict, c_gt, target_mask = prepare_batch_for_siamese(batch, device)

        # Clean eval (no_grad)
        with torch.no_grad():
            input_dict_clean = dict(input_dict)
            ep_clean, _, _ = _forward_with_intermediate(model, input_dict_clean)
            score_gt_clean, c_pred_clean, _, _ = _compute_tracking_terms(ep_clean, c_gt, target_mask)
            clean_err = (c_pred_clean - c_gt).norm(p=2, dim=1).mean().item()

        # Generate attack in white-box mode (WITH grad)
        result = main_attack_loop(
            model=model,
            input_dict=input_dict,
            c_gt=c_gt,
            target_mask=target_mask,
            attack_cfg=attack_cfg,
        )
        adv_search = result["S_adv"]

        # Re-evaluate adversarial sample in black-box mode (no_grad)
        with torch.no_grad():
            input_dict_adv = dict(input_dict)
            input_dict_adv["search_points"] = adv_search
            ep_adv, _, _ = _forward_with_intermediate(model, input_dict_adv)
            score_gt_adv, c_pred_adv, _, _ = _compute_tracking_terms(ep_adv, c_gt, target_mask)
            adv_err = (c_pred_adv - c_gt).norm(p=2, dim=1).mean().item()

        bb_clean_err.append(clean_err)
        bb_adv_err.append(adv_err)

    print(f"Clean center error:    {np.mean(bb_clean_err):.4f}")
    print(f"Adv center error:      {np.mean(bb_adv_err):.4f}")
    print(f"Center error increase: {np.mean(bb_adv_err) - np.mean(bb_clean_err):.4f}")
    print(f"Black-box success rate: {np.mean([a > c for a, c in zip(bb_adv_err, bb_clean_err)]) * 100:.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, f"{args.out_prefix}_batch_report.json")
    import json
    with open(report_path, "w") as f:
        json.dump({
            "samples": len(all_clean_err),
            "clean_center_error_mean": float(clean_err_mean),
            "adv_center_error_mean": float(adv_err_mean),
            "center_error_increase_mean": float(adv_err_mean - clean_err_mean),
            "score_drop_mean": float(score_drop_mean),
            "success_rate": float(success_rate),
            "bb_clean_center_error_mean": float(np.mean(bb_clean_err)),
            "bb_adv_center_error_mean": float(np.mean(bb_adv_err)),
            "bb_success_rate": float(np.mean([a > c for a, c in zip(bb_adv_err, bb_clean_err)])),
        }, f, indent=2)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
