import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AttackConfig:
    eps: float = 0.05
    alpha: float = 0.005
    iters: int = 20
    k_ratio: float = 0.2
    lambda_match: float = 1.0
    lambda_offset: float = 1.0
    beta_cd: float = 0.1
    gamma_knn: float = 0.1
    knn_k: int = 8
    proposal_temperature: float = 10.0


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """Compute symmetric Chamfer distance per sample.

    Args:
        pc1: [B, N, 3]
        pc2: [B, M, 3]

    Returns:
        Tensor [B]
    """
    dists = torch.cdist(pc1, pc2, p=2) ** 2
    min_12 = dists.min(dim=2).values.mean(dim=1)
    min_21 = dists.min(dim=1).values.mean(dim=1)
    return min_12 + min_21


def compute_importance(features: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    """Compute per-point importance from feature gradients.

    features: [B, C, N]
    returns: [B, N] normalized to [0,1]
    """
    grads = torch.autograd.grad(
        loss, features, retain_graph=True, create_graph=False, allow_unused=True
    )[0]
    if grads is None:
        grads = torch.zeros_like(features)
    scores = grads.norm(p=2, dim=1)
    max_vals = scores.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    return scores / max_vals


def select_critical_points(scores: torch.Tensor, k_ratio: float) -> torch.Tensor:
    """Select top-k ratio critical points.

    scores: [B, N]
    returns: bool mask [B, N]
    """
    bsz, n = scores.shape
    k = max(1, int(round(n * k_ratio)))
    topk_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask


def attack_step(
    points: torch.Tensor,
    gradients: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
    eps: float,
    delta: torch.Tensor,
) -> torch.Tensor:
    """One weighted PGD ascent step with box projection.

    points: [B, N, 3]
    gradients: [B, N, 3]
    weights: [B, N]
    delta: [B, N, 3]
    """
    step = alpha * weights.unsqueeze(-1) * gradients.sign()
    delta = delta + step
    delta = torch.clamp(delta, min=-eps, max=eps)
    return delta


def _knn_consistency_loss(
    adv_points: torch.Tensor,
    clean_points: torch.Tensor,
    knn_idx: torch.Tensor,
    clean_knn_dists: torch.Tensor,
) -> torch.Tensor:
    """Keep local neighborhood distances close to clean cloud.

    adv_points/clean_points: [B, N, 3]
    knn_idx: [B, N, K]
    clean_knn_dists: [B, N, K]
    returns scalar
    """
    bsz, n, _ = adv_points.shape
    k = knn_idx.shape[-1]

    gather_idx = knn_idx.unsqueeze(-1).expand(bsz, n, k, 3)
    adv_neighbors = torch.gather(
        adv_points.unsqueeze(1).expand(bsz, n, n, 3), 2, gather_idx
    )
    adv_center = adv_points.unsqueeze(2)
    adv_dists = (adv_neighbors - adv_center).norm(p=2, dim=-1)
    return (adv_dists - clean_knn_dists).abs().mean()


def _build_knn_reference(clean_points: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Exclude self-neighbor by taking topk+1 and dropping the first.
    pair = torch.cdist(clean_points, clean_points, p=2)
    knn_vals, knn_idx = torch.topk(pair, k=k + 1, dim=-1, largest=False, sorted=True)
    return knn_idx[:, :, 1:], knn_vals[:, :, 1:]


def _forward_with_intermediate(model, input_dict: Dict[str, torch.Tensor]):
    """Forward for P2B/BAT while exposing search branch intermediate features."""
    template = input_dict["template_points"]
    search = input_dict["search_points"]

    m = template.shape[1]
    n = search.shape[1]

    template_xyz, template_feature, sample_idxs_t = model.backbone(template, [m // 2, m // 4, m // 8])
    search_xyz, search_feature, sample_idxs = model.backbone(search, [n // 2, n // 4, n // 8])

    template_feature = model.conv_final(template_feature)
    search_feature = model.conv_final(search_feature)
    if search_feature.requires_grad:
        search_feature.retain_grad()

    if model.__class__.__name__.lower() == "bat":
        template_bc = input_dict["points2cc_dist_t"]
        pred_search_bc = model.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1)).transpose(1, 2)
        sample_idxs_t = sample_idxs_t[:, : m // 8, None]
        template_bc = template_bc.gather(
            dim=1,
            index=sample_idxs_t.repeat(1, 1, model.config.bc_channel).long(),
        )
        fusion_feature = model.xcorr(
            template_feature,
            search_feature,
            template_xyz,
            search_xyz,
            template_bc,
            pred_search_bc,
        )
        estimation_boxes, estimation_cla, vote_xyz, center_xyzs = model.rpn(search_xyz, fusion_feature)
        end_points = {
            "estimation_boxes": estimation_boxes,
            "vote_center": vote_xyz,
            "pred_seg_score": estimation_cla,
            "center_xyz": center_xyzs,
            "sample_idxs": sample_idxs,
            "estimation_cla": estimation_cla,
            "vote_xyz": vote_xyz,
            "pred_search_bc": pred_search_bc,
        }
    else:
        fusion_feature = model.xcorr(template_feature, search_feature, template_xyz)
        estimation_boxes, estimation_cla, vote_xyz, center_xyzs = model.rpn(search_xyz, fusion_feature)
        end_points = {
            "estimation_boxes": estimation_boxes,
            "vote_center": vote_xyz,
            "pred_seg_score": estimation_cla,
            "center_xyz": center_xyzs,
            "sample_idxs": sample_idxs,
            "estimation_cla": estimation_cla,
            "vote_xyz": vote_xyz,
        }

    return end_points, search_feature


def _compute_tracking_terms(
    end_points: Dict[str, torch.Tensor],
    c_gt: torch.Tensor,
    target_mask: torch.Tensor,
    proposal_temperature: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # score map on sampled points (search branch)
    score_map = end_points["estimation_cla"]
    ns = score_map.shape[1]
    sample_idxs = end_points["sample_idxs"][:, :ns].long()

    sampled_target_mask = target_mask.gather(1, sample_idxs)
    sampled_target_mask = sampled_target_mask.float()

    score_prob = torch.sigmoid(score_map)
    score_gt = (score_prob * sampled_target_mask).sum(dim=1) / sampled_target_mask.sum(dim=1).clamp_min(1.0)

    boxes = end_points["estimation_boxes"]  # [B, K, 5]
    proposal_logits = boxes[:, :, 4]
    proposal_weight = torch.softmax(proposal_logits * proposal_temperature, dim=1)
    c_pred = (proposal_weight.unsqueeze(-1) * boxes[:, :, :3]).sum(dim=1)

    return score_gt, c_pred, sample_idxs, sampled_target_mask


def _to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main_attack_loop(
    model,
    input_dict: Dict[str, torch.Tensor],
    c_gt: torch.Tensor,
    target_mask: torch.Tensor,
    attack_cfg: Optional[AttackConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Run critical feature guided attack on search point cloud.

    Required input_dict keys for P2B: template_points, search_points
    Optional BAT key: points2cc_dist_t
    """
    if attack_cfg is None:
        attack_cfg = AttackConfig()

    device = next(model.parameters()).device
    input_dict = _to_device_batch(input_dict, device)
    c_gt = c_gt.to(device).float()
    target_mask = target_mask.to(device).bool()

    clean_search = input_dict["search_points"].detach()
    delta = torch.zeros_like(clean_search)

    knn_idx, clean_knn_dists = _build_knn_reference(clean_search, k=attack_cfg.knn_k)

    clean_eval_input = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in input_dict.items()}
    with torch.no_grad():
        clean_ep, _ = _forward_with_intermediate(model, clean_eval_input)
        clean_score_gt, clean_c_pred, _, _ = _compute_tracking_terms(
            clean_ep, c_gt, target_mask, proposal_temperature=attack_cfg.proposal_temperature
        )
        clean_center_error = (clean_c_pred - c_gt).norm(p=2, dim=1)

    history = []

    for step in range(attack_cfg.iters):
        adv_search = (clean_search + delta).detach().requires_grad_(True)
        adv_input = dict(input_dict)
        adv_input["search_points"] = adv_search

        end_points, search_features = _forward_with_intermediate(model, adv_input)
        score_gt, c_pred, sample_idxs, sampled_target_mask = _compute_tracking_terms(
            end_points, c_gt, target_mask, proposal_temperature=attack_cfg.proposal_temperature
        )

        # Attack objective: suppress target response and increase center error.
        l_match = -score_gt.mean()
        l_offset = ((c_pred - c_gt).norm(p=2, dim=1)).mean()
        l_adv = attack_cfg.lambda_match * l_match + attack_cfg.lambda_offset * l_offset

        l_cd = chamfer_distance(adv_search, clean_search).mean()
        l_knn = _knn_consistency_loss(adv_search, clean_search, knn_idx, clean_knn_dists)

        objective = l_adv - attack_cfg.beta_cd * l_cd - attack_cfg.gamma_knn * l_knn

        importance_sampled = compute_importance(search_features, objective)
        n_orig = adv_search.shape[1]
        ns = importance_sampled.shape[1]

        sampled_idx = sample_idxs[:, :ns]
        weights_orig = torch.zeros(adv_search.size(0), n_orig, device=device)
        for b in range(adv_search.size(0)):
            weights_orig[b, sampled_idx[b]] = importance_sampled[b]

        # Fallback when feature-gradient importance is degenerate: use sampled score magnitude.
        if weights_orig.abs().sum().item() < 1e-12:
            score_fallback = torch.sigmoid(end_points["estimation_cla"]).detach()
            for b in range(adv_search.size(0)):
                weights_orig[b, sampled_idx[b]] = score_fallback[b]

        weights_orig = weights_orig * target_mask.float()
        if weights_orig.abs().sum().item() < 1e-12:
            weights_orig = target_mask.float()

        critical_mask = select_critical_points(weights_orig, attack_cfg.k_ratio)
        attack_weights = weights_orig * critical_mask.float()

        grad_points = torch.autograd.grad(
            objective, adv_search, retain_graph=False, create_graph=False, allow_unused=True
        )[0]
        if (grad_points is None) or (grad_points.abs().sum().item() < 1e-12):
            # Input-grad fallback: move target points away from GT center.
            grad_points = F.normalize(adv_search - c_gt.unsqueeze(1), p=2, dim=-1, eps=1e-12)

        delta = attack_step(
            points=adv_search,
            gradients=grad_points,
            weights=attack_weights,
            alpha=attack_cfg.alpha,
            eps=attack_cfg.eps,
            delta=delta,
        )

        # Enforce perturbation only in target area.
        delta = delta * target_mask.unsqueeze(-1).float()

        history.append(
            {
                "step": step,
                "l_adv": float(l_adv.detach().item()),
                "l_match": float(l_match.detach().item()),
                "l_offset": float(l_offset.detach().item()),
                "l_cd": float(l_cd.detach().item()),
                "l_knn": float(l_knn.detach().item()),
                "score_gt": float(score_gt.mean().detach().item()),
                "delta_linf": float(delta.detach().abs().max().item()),
            }
        )

    adv_search = (clean_search + delta).detach()
    adv_eval_input = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in input_dict.items()}
    adv_eval_input["search_points"] = adv_search

    with torch.no_grad():
        adv_ep, _ = _forward_with_intermediate(model, adv_eval_input)
        adv_score_gt, adv_c_pred, _, _ = _compute_tracking_terms(
            adv_ep, c_gt, target_mask, proposal_temperature=attack_cfg.proposal_temperature
        )
        adv_center_error = (adv_c_pred - c_gt).norm(p=2, dim=1)

    out = {
        "S_adv": adv_search,
        "delta": delta,
        "clean_score_gt": clean_score_gt,
        "adv_score_gt": adv_score_gt,
        "score_drop": clean_score_gt - adv_score_gt,
        "clean_center_error": clean_center_error,
        "adv_center_error": adv_center_error,
        "center_error_increase": adv_center_error - clean_center_error,
        "history": history,
    }
    return out


def dump_attack_report(path: str, result: Dict[str, torch.Tensor]) -> None:
    serializable = {
        "clean_score_gt": result["clean_score_gt"].detach().cpu().tolist(),
        "adv_score_gt": result["adv_score_gt"].detach().cpu().tolist(),
        "score_drop": result["score_drop"].detach().cpu().tolist(),
        "clean_center_error": result["clean_center_error"].detach().cpu().tolist(),
        "adv_center_error": result["adv_center_error"].detach().cpu().tolist(),
        "center_error_increase": result["center_error_increase"].detach().cpu().tolist(),
        "history": result["history"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
