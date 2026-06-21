"""No-GT attack evaluation with black-box score disabled.

这个入口用于验证 BAT 这类模型上 ``score_drop`` 是否误导候选选择。
攻击阶段仍然只使用 tracker 的预测框变化，不使用 GT；区别是黑盒候选
评估时把 ``score`` 置为 None，因此 no-GT 核心里的 ``score_drop`` 恒为 0。

最终 succ/pre 评估仍然调用原有 GT metric，只用于实验统计。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from my_attack.core.progressive_diffusion_attack_nogt import (
    DriftState,
    ProgressiveAttackConfig,
    run_progressive_attack,
)
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval


def _box_yaw(box) -> Optional[float]:
    try:
        return float(box.orientation.radians * box.orientation.axis[-1])
    except Exception:
        return None


def evaluate_input_blackbox_noscore(model, input_dict: Dict, ref_bb) -> Tuple[Dict, object]:
    """黑盒候选评估：只返回预测框，不返回 tracker score。"""

    candidate_box = base_eval.candidate_from_model(model, input_dict, ref_bb)
    return {
        "score": None,
        "pred_center": np.asarray(candidate_box.center).astype(float).tolist(),
        "pred_wlh": np.asarray(candidate_box.wlh).astype(float).tolist(),
        "pred_yaw": _box_yaw(candidate_box),
    }, candidate_box


def evaluate_one_sequence_attacked(
    model,
    sequence,
    attack_cfg: ProgressiveAttackConfig,
    out_dir: str,
    sequence_id: int,
    max_frames: int = -1,
):
    """和普通 no-GT evaluator 相同，但候选攻击阶段禁用 score_drop。"""

    # 复用普通 no-GT evaluator 的实现框架，只临时替换它的黑盒评估函数。
    from my_attack.evaluation import eval_progressive_diffusion_attack_nogt as nogt_eval

    original = nogt_eval.evaluate_input_blackbox
    nogt_eval.evaluate_input_blackbox = evaluate_input_blackbox_noscore
    try:
        return nogt_eval.evaluate_one_sequence_attacked(
            model, sequence, attack_cfg, out_dir, sequence_id, max_frames
        )
    finally:
        nogt_eval.evaluate_input_blackbox = original


def main():
    base_eval.ProgressiveAttackConfig = ProgressiveAttackConfig
    base_eval.DriftState = DriftState
    base_eval.run_progressive_attack = run_progressive_attack
    base_eval.evaluate_one_sequence_attacked = evaluate_one_sequence_attacked
    base_eval.main()


if __name__ == "__main__":
    main()
