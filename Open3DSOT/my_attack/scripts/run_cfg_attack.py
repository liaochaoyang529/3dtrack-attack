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
    parser = argparse.ArgumentParser("Critical Feature Guided Attack for Siamese 3D tracking")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--k_ratio", type=float, default=0.2)
    parser.add_argument("--lambda_match", type=float, default=1.0)
    parser.add_argument("--lambda_offset", type=float, default=1.0)
    parser.add_argument("--beta_cd", type=float, default=0.1)
    parser.add_argument("--gamma_knn", type=float, default=0.1)
    parser.add_argument("--knn_k", type=int, default=8)

    parser.add_argument("--out_dir", type=str, default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs")
    parser.add_argument("--out_prefix", type=str, default="cfg_attack")
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
    cfg = EasyDict(cfg_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, args.checkpoint, device)

    # Attack on Siamese sampled batches to keep template/search pair and seg mask directly available.
    dataset = get_dataset(cfg, type="train_siamese", split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    batch = next(iter(loader))
    input_dict, c_gt, target_mask = prepare_batch_for_siamese(batch, device)

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
    )

    result = main_attack_loop(
        model=model,
        input_dict=input_dict,
        c_gt=c_gt,
        target_mask=target_mask,
        attack_cfg=attack_cfg,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    s_adv_path = os.path.join(args.out_dir, f"{args.out_prefix}_S_adv.npy")
    delta_path = os.path.join(args.out_dir, f"{args.out_prefix}_delta.npy")
    report_path = os.path.join(args.out_dir, f"{args.out_prefix}_report.json")

    np.save(s_adv_path, result["S_adv"].detach().cpu().numpy())
    np.save(delta_path, result["delta"].detach().cpu().numpy())
    dump_attack_report(report_path, result)

    clean_err = result["clean_center_error"].mean().item()
    adv_err = result["adv_center_error"].mean().item()
    score_drop = result["score_drop"].mean().item()

    print("=== Critical Feature Guided Attack Done ===")
    print(f"Clean center error: {clean_err:.6f}")
    print(f"Adv center error:   {adv_err:.6f}")
    print(f"Center error +:     {adv_err - clean_err:.6f}")
    print(f"Score drop:         {score_drop:.6f}")
    print(f"Saved S_adv:        {s_adv_path}")
    print(f"Saved delta:        {delta_path}")
    print(f"Saved report:       {report_path}")


if __name__ == "__main__":
    main()
