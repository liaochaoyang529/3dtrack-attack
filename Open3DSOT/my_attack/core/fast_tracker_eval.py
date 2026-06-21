"""单前向 + 批量的 tracker 评估，用于 BC 引导无 GT 攻击的推理加速。

本模块在 **不改变攻击决策** 的前提下，去掉原 no-GT 评估路径里的两处冗余开销：

1. 双前向冗余：原 ``_predict_box_metrics`` 每次查询会跑两遍 ``model(...)``
   （一遍取 box，一遍取 score）。对 matching 类 tracker（P2B/BAT），score 就在
   与 box 同一份 ``estimation_boxes`` 输出里，所以一次前向即可同时得到二者。
2. top-k 串行前向：同一搜索步的 top-k 候选原本逐个（batch=1）前向；它们共享
   同一 template / ref box，仅 search 点云不同，可以 stack 成一次批量前向。

在 ``model.eval()`` 下两者与原路径数值等价：BatchNorm 使用 running stats、
PointNet++ 的最远点采样按样本独立且确定，因此「批量前向」与「逐样本前向」
逐位一致，「同一份输出解析 box+score」与「两次前向分别取」也一致。

当 tracker 输出不是 matching 风格的 ``estimation_boxes`` 张量时（如 M2Track 的
部分输出形态），``supports_fast_path`` 返回 False，调用方应回退到原始逐候选路径，
从而保证兼容。
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from datasets import points_utils
from utils.metrics import estimateAccuracy, estimateOverlap


def _box_yaw(box) -> Optional[float]:
    """与原 eval 脚本一致的 yaw 解析，失败时返回 None。"""
    try:
        return float(box.orientation.radians * box.orientation.axis[-1])
    except Exception:
        return None


@torch.no_grad()
def supports_fast_path(model, input_dict: Dict[str, torch.Tensor]) -> bool:
    """探测当前 tracker 是否输出 matching 风格的 estimation_boxes [B, P, >=5]。

    只有这种输出能从一次前向里同时拿到 box（前 4 列）与 score（第 4 列）。
    """
    out = model(input_dict)
    boxes = out.get("estimation_boxes") if isinstance(out, dict) else None
    return isinstance(boxes, torch.Tensor) and boxes.dim() == 3 and boxes.shape[-1] >= 5


def _stack_inputs(input_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """把若干 batch=1 的 input_dict 沿 batch 维拼成一个批量 input_dict。

    同一帧内，除 ``search_points``（被攻击替换）外，其余分支（template、box-cloud
    等）在所有候选间相同，因此直接 cat 即可，结果与逐个前向等价。
    """
    ref = input_dicts[0]
    batched: Dict[str, torch.Tensor] = {}
    for key, value in ref.items():
        if torch.is_tensor(value):
            batched[key] = torch.cat([d[key] for d in input_dicts], dim=0)
        else:
            # 非张量字段（标量/元信息）在同帧候选间一致，沿用第一个。
            batched[key] = value
    return batched


def _parse_box_and_score(model, estimation_row: np.ndarray, ref_box):
    """从单行 estimation_boxes [P, >=5] 解析候选框与 score。

    与 ``BaseModel.evaluate_one_sample`` + ``infer_tracking_score`` 完全一致：
    取第 4 列得分最高的 proposal，用其前 4 维 offset 还原世界坐标框；
    score 取该行第 4 列的最大值（即被选中 proposal 的得分）。
    """
    best_idx = int(estimation_row[:, 4].argmax())
    offset = estimation_row[best_idx, 0:4]
    candidate_box = points_utils.getOffsetBB(
        ref_box,
        offset,
        degrees=model.config.degrees,
        use_z=model.config.use_z,
        limit_box=model.config.limit_box,
    )
    score = float(estimation_row[:, 4].max())
    return candidate_box, score


@torch.no_grad()
def forward_tracker_batch(
    model,
    input_dicts: List[Dict[str, torch.Tensor]],
    ref_box,
) -> List[Tuple[object, float]]:
    """对一批 input_dict 做一次批量前向，返回每个候选的 (box, score)。

    单次 ``model(batched)`` 同时给出 box 与 score（解决双前向冗余），
    且 top-k 候选共用一次前向（解决串行前向）。
    """
    if not input_dicts:
        return []
    batched = _stack_inputs(input_dicts)
    out = model(batched)
    boxes = out["estimation_boxes"]  # [B, P, >=5]
    boxes_np = boxes.detach().cpu().numpy()
    results: List[Tuple[object, float]] = []
    for row in boxes_np:
        results.append(_parse_box_and_score(model, row, ref_box))
    return results


def make_batch_clean_reference_eval_fn(
    model,
    clean_input: Dict[str, torch.Tensor],
    ref_bb,
) -> Tuple[Callable[[Dict[str, torch.Tensor]], Dict], Callable[[List[Dict[str, torch.Tensor]]], List[Dict]]]:
    """构造无 GT 评估器（单前向 + 批量版本）。

    与原 ``_make_clean_reference_eval_fn`` 语义一致：先用一次前向得到当前对抗
    轨迹输入的「干净参考框」，候选按相对该参考框的偏移打分，而非用 GT。

    返回 ``(single_eval_fn, batch_eval_fn)``：
    - ``single_eval_fn(input_dict) -> metrics``：供 clean reference 等单点查询使用。
    - ``batch_eval_fn(list_of_inputs) -> list_of_metrics``：供 top-k 批量查询使用。
    两者产出的 metrics 字段与原 ``tracker_eval_fn`` 完全一致。
    """
    clean_box, clean_score = forward_tracker_batch(model, [clean_input], ref_bb)[0]

    def _metrics_from(candidate_box, score: float) -> Dict:
        iou = estimateOverlap(
            clean_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis
        )
        center_error = estimateAccuracy(
            clean_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis
        )
        return {
            "iou": float(iou),
            "center_error": float(center_error),
            "score": score,
            "clean_reference_score": clean_score,
            "pred_center": np.asarray(candidate_box.center).astype(float).tolist(),
            "pred_wlh": np.asarray(candidate_box.wlh).astype(float).tolist(),
            "pred_yaw": _box_yaw(candidate_box),
        }

    def batch_eval_fn(candidate_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict]:
        results = forward_tracker_batch(model, candidate_inputs, ref_bb)
        return [_metrics_from(box, score) for (box, score) in results]

    def single_eval_fn(candidate_input: Dict[str, torch.Tensor]) -> Dict:
        return batch_eval_fn([candidate_input])[0]

    return single_eval_fn, batch_eval_fn
