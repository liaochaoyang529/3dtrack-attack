import argparse
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
from my_attack.critical_feature_guided_attack import AttackConfig, main_attack_loop
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

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--k_ratio", type=float, default=0.2)
    parser.add_argument("--lambda_match", type=float, default=1.0)
    parser.add_argument("--lambda_offset", type=float, default=1.0)
    parser.add_argument("--beta_cd", type=float, default=0.1)
    parser.add_argument("--gamma_knn", type=float, default=0.1)
    parser.add_argument("--knn_k", type=int, default=8)

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


def evaluate_one_sequence_attacked(model, sequence, attack_cfg: AttackConfig):
    ious: List[float] = []
    distances: List[float] = []
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
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

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

    success = TorchSuccess()
    precision = TorchPrecision()

    for batch in tqdm(test_loader, desc="Attacked Eval", total=len(test_loader)):
        sequence = batch[0]
        ious, distances = evaluate_one_sequence_attacked(model, sequence, attack_cfg)
        success(torch.tensor(ious, device=device))
        precision(torch.tensor(distances, device=device))

    success_score = float(success.compute().detach().cpu().item())
    precision_score = float(precision.compute().detach().cpu().item())

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out = {
        "success": success_score,
        "precision": precision_score,
        "attack": {
            "eps": args.eps,
            "alpha": args.alpha,
            "iters": args.iters,
            "k_ratio": args.k_ratio,
            "lambda_match": args.lambda_match,
            "lambda_offset": args.lambda_offset,
            "beta_cd": args.beta_cd,
            "gamma_knn": args.gamma_knn,
            "knn_k": args.knn_k,
        },
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "split": args.split,
    }

    import json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== Attacked Full Evaluation Done ===")
    print(f"success:   {success_score:.6f}")
    print(f"precision: {precision_score:.6f}")
    print(f"saved:     {args.out_json}")


if __name__ == "__main__":
    main()
