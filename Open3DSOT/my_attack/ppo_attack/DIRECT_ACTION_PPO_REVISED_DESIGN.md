# Direct-Action PPO 攻击策略 —— 修订版设计方案

本文档是对原始 "PPO direct-action" 设计方案的代码级评审与修订。结论先行：**原方案方向正确、目标清晰，但对"推理瓶颈在哪里"有一处误判**；在核实真实瓶颈后，方案不仅成立，而且收益比原设想更大。下文给出修订后的可执行设计。

相关代码：
- `my_attack/core/progressive_diffusion_attack_v2_bc_fast.py`（BC fast 主流程）
- `my_attack/evaluation/eval_progressive_diffusion_attack_v2_bc_nogt.py`（BC fast 评估入口）
- `my_attack/ppo_attack/export_v2_teacher_dataset.py`（`generate_candidates` 候选生成）
- `my_attack/ppo_attack/point_policy.py`（`PointAttackRanker`，现有 BC 网络）
- `my_attack/ppo_attack/eval_online_point_ranker_policy_bat.py`（在线 argmax 选 1 的评估）
- `my_attack/core/progressive_diffusion_attack_v2.py`（攻击算子 + `CloudState`）

---

## 1. 真实瓶颈核实（这是整个方案是否值得做的前提）

原方案写道推理流程是「候选生成 -> BC rank -> 查询 top_k -> tracker metrics 选 best」，要把它砍成「policy 直接选 1 -> tracker 查 1 次」。核实代码后确认：

### 1.1 BC fast 确实是"top_k 次 tracker forward"

`progressive_diffusion_attack_v2_bc_fast.py` 的 `_evaluate_bc_filtered_candidates_batched`：

1. BC selector 对全部候选打分排序，取 top_k（`bc_top_k` 默认 **5**，见 `eval_*_bc_nogt.py:86`）；
2. 对这 top_k 个候选**逐个**构造输入并**各跑一次 tracker forward**（`:134` 的 `batch_eval_fn`，在评估脚本 `:210` 被设为 `None`，退化为 batch=1 逐个前向）；
3. 用 tracker 返回的 IoU / center_error 计算分数选 best（`:156` `_metric_attack_score`）。

主循环 `:276` 走 `max_noise_steps` 步，`:322-327` 攻击成功才 `break`。因此：

```
每帧 tracker forward 次数 ≈ failure_step × bc_top_k
```

`max_noise_steps` 各配置：保守 `progressive_diffusion_attack.yaml`=10，激进 `refbox_m2_*`=50/100。即使平均 3 步成功、top_k=5，也是约 **15 次 forward/帧**；激进配置下更高。

### 1.2 direct-action 能省多少

direct-action policy 每步直接输出 1 个动作，不再对 top_k 候选逐个查 tracker：

| 方案 | 每帧 tracker forward | 相对 BC fast(top_k=5) |
|---|---|---|
| BC fast top_k=5 | `failure_step × 5` | 基准 |
| direct-action **方案B**（每步查 1 次拿 ref / 判终止） | `failure_step × 1` | **省 5 倍** |
| direct-action **方案A**（state 不含 ref，仅最后查 1 次评估） | `≈ 1` | 省一个数量级 |

**结论：瓶颈确实在 top_k 次 tracker forward（不是 candidate 生成的张量克隆）。原方案要砍的东西砍对了，收益是 5 倍量级。方案值得做。**

> 注意：原方案文字里"候选生成"被当成可省的成本之一，这点要修正——`generate_candidates` 的 11~37 次 `state.clone()`+算子是 ~1024 点的亚毫秒 GPU 操作，相对 tracker forward 可忽略。**真正的省时来自删掉 top_k 次 forward。**

---

## 2. 动作空间（沿用方案 B 的 11 类，但说明与现有候选的关系）

现有 `generate_candidates` 在 no-fake/no-drop 配置、默认 `patch_candidate_k=4`、8 方向下会生成约 **37** 个候选（4 个 critical_patch_jitter + 4×8 local_patch_shift + 1 progressive_noise）。方案 B 把它**收缩**为 11 类（2 patch × {jitter, ±x±y shift} + 1 noise）：

| action_id | attack_type | patch_id | direction |
|---|---|---|---|
| 0,1 | critical_patch_jitter | 0,1 | — |
| 2–5 | local_patch_shift | 0 | +x,-x,+y,-y |
| 6–9 | local_patch_shift | 1 | +x,-x,+y,-y |
| 10 | progressive_noise | — | — |

要点：
- patch_id 是**每帧动态**的（`_patch_indices` 按到中心距离排序的局部近邻，`progressive_diffusion_attack_v2.py:308-327`）。patch_id=0 在不同帧指向不同几何区域，**所以 actor 必须看点云几何才能正确选 patch_id**——这决定了网络必须保留 DGCNN encoder，不能退化成只看 scalar。
- 先不加 fake/drop/recovery，与你 no-fake/no-drop 实验一致。**但建议在阶段 5 做一次消融**：验证收缩到 11 类、且去掉 tracker 验证后，攻击成功率相对 37 候选 top_k=5 掉了多少（见 §8 sanity baseline）。

---

## 3. 网络：复用 `PointAttackRanker`，不要重写 DGCNN

原方案打算新写 `direct_action_policy.py`（DGCNN + scalar encoder + 11-way actor + critic）。**更省事且更一致的做法是继承/薄封装现有 `PointAttackRanker`**，因为它已经具备 direct-action 需要的全部能力：

- `DGCNNLiteEncoder`：6 维输入 `[current_xyz, current-clean]`、point feature + global feature（`point_policy.py:92-139`）✔
- `_gather_patch_features`：按 patch_center_idx 从 point feature gather 局部特征（`:215-226`）✔ —— 这正是"基于局部几何选 patch"的关键
- `value_head(global_feature)`：critic 已预留（`:203-207`）✔

唯一区别：现有 ranker 的候选 action 特征**来自数据里 teacher 给的 candidate 数组**；direct actor 要用**一组固定的 11 个"标准动作模板"**自己生成候选特征：

```
direct actor forward(clean, current, scalar_state):
    1. patches = _patch_indices(clean, cfg)[:2]          # 现算 2 个 patch 的 center_idx
    2. 构造 11 个动作模板的 (op_id, direction_id, patch_center_idx, ...)  # 固定规则
    3. 复用 PointAttackRanker 前向 → 11 个 logits（天然条件化于几何）
    4. logits 拼上 scalar_state 编码（可选）→ actor 分布
    5. value_head(global_feature) → V(s)
```

这样：**actor 直接输出 11 维分布、critic 复用 value_head、DGCNN 一行不用重写**。`direct_action_policy.py` 变成对 `PointAttackRanker` 的封装 + 11 动作模板生成器。

scalar state（§4）可通过一个小 MLP 编码后拼到 global_feature 上，再过 actor/critic head；这部分是对 ranker 的增量。

---

## 4. 状态输入：no-GT 推理下必须避免"为拿 state 又多查 tracker"

原方案 state 里放了 `last_ref_iou / last_ref_center_error / last_success_flag`。**但 ref metric 需要对 current_state 跑一次 tracker 才能得到**——若每步都要它，等于把想省的 forward 加回来了。两条路（建议先 A）：

- **方案 A（推荐，真省时）**：state 只放**不需要 tracker 的纯几何/扰动统计**：
  ```
  step_ratio = step_id / max_noise_steps
  last_action_id (one-hot / embedding)
  removed_point_ratio
  fake_point_ratio          # 当前恒 0（no-fake），保留接口
  changed_point_ratio
  avg_point_displacement
  ```
  这些都能从 `CloudState`（`source_idx / fake_mask / jitter_delta / patch_delta`）直接算，无需 tracker。推理真正做到「中间步 0 次 forward，最后 1 次评估」。
- **方案 B（保留 ref）**：每步查 1 次 tracker 拿 ref 并判定 early-stop。若平均 2~3 步成功，仍比 top_k=5 省 5 倍。

> 建议：训练/调试期可用方案 B（ref + early-stop 让 episode 短、reward 稠密），**部署推理切方案 A**（去掉 ref 输入，policy 已学会"何时停"则固定步数或用扰动阈值停）。两者 state 维度要兼容（A 是 B 的子集）。

---

## 5. Reward：复用 `teacher_score` 形式，保证 BC warm-start 与 PPO 目标一致

原方案的 reward 把 stealth 拆成 5 个独立负项，权重与现有 `teacher_score`（`score.py` / BC 文档 §4）**量纲完全不同**：

```
teacher_score = 10*(1-IoU) + center_error - 2.0*stealth (+10 success_bonus)
stealth = chamfer + avg_disp + 0.25*fake_ratio + 0.25*removed_ratio + 0.1*density_diff
```

BC warm-start 学的是这个偏好；若 PPO 用另一套权重，PPO 一开始就会把 warm-start 洗掉。**修订：PPO 的 step reward 直接复用 teacher_score 的形式 + step/query penalty**：

```
r_step = teacher_score(metrics)                 # 与 BC teacher 同一套权重 + success_bonus
         - 0.05 * 1                              # step penalty（每步固定）
         - 0.1  * query_count_this_step          # query penalty（每次 tracker forward 计 1）
```

补充约束：
- `center_error` 项**做 clip**（如 `min(center_error, 3.0)`），避免偶发巨值主导梯度；
- success/query 的计数口径与评估指标完全一致（每次 tracker forward = 1 query）；
- 若训练必须 no-GT，把 teacher_score 里的 IoU/center_error 换成 ref 版本（clean 预测作参考），其余不变。但**先用 GT 训练**更稳（原方案此判断正确）。

这正是 BC 文档 §10 末尾自己提出的"让 BC 目标和 PPO reward 更一致"。

---

## 6. BC warm-start：复用现有 teacher 数据，不要重采

原方案阶段 2 要"从方案 B logs 重新导出 imitation dataset"。**没必要重采**——现有 `outputs/ppo_teacher_success_stealth/`（9438 条轨迹、48548 个过滤后 step，BC 文档 §2）每个 step 的 NPZ 已存：

```
candidate_op_id / candidate_direction_id / candidate_patch_center_idx / ...
best_candidate_index            # teacher 选中的候选
candidate_teacher_score         # 连续分（可做 soft label）
clean_search_points / current_points / normalization_*
```

只需写一个映射 `(op_id, direction_id, patch_center_idx→patch_id) -> action_id ∈ [0,11)`，把 `best_candidate_index` 转成 direct-action 的标签即可：

- **硬标签**：`CrossEntropy(actor_logits, mapped_action_id)`；
- **软标签（推荐）**：把落在 11 类里的候选的 `candidate_teacher_score` 按 action 聚合 → `softmax/temperature` → KL 蒸馏（与现有 `soft_ranking_loss` 同思路，`train_point_ranker.py:311-313`），保留 PPO 需要的探索性。

映射时注意：原数据是 37 候选（4 patch×8 方向），方案 B 只取 **前 2 patch + 4 方向**。落在 11 类外的候选（patch_id≥2 或斜方向）有两种处理：(a) 丢弃该 step；(b) 把它的 teacher_score 归并到最近的合法 action。建议先 (a)（干净），数据量够。

critic 的 warm-start：可先用 teacher 的 step return（teacher_score 折算）拟合 value_head，或先不训 critic、PPO 阶段从零学 V（advantage 早期噪声大但可接受）。

---

## 7. 环境：动作执行、候选生成、BC 标注三处必须共用同一函数

原方案 `apply_action` 调 `_jitter_patch_state / _shift_patch_state / apply_noise_step`，patch 用 `_patch_indices(clean,cfg)[:2]`。**关键风险：执行动作的代码必须和 `generate_candidates` 用同一套 cfg / patch 计算 / seed 规则**，否则 BC 学的标签与 env 实际执行的动作对不上，PPO 会学偏。

`generate_candidates` 的 seed 规则（`export_v2_teacher_dataset.py:330`）：jitter seed = `cfg.seed + 1100 + step_id*97 + patch_id`。

**修订：封装统一函数**，BC 标注 / env.step / 推理三处共用：

```python
def apply_action_id(state, action_id, clean_points, cfg, step_id):
    """11 类离散动作 -> 执行对应 v2 算子，返回 next CloudState。
    patch 用 _patch_indices(clean, cfg)[:2]；方向用固定 4 向；
    seed 规则与 generate_candidates 完全一致。"""
    ...
```

env 草图（一个 episode = 一帧攻击）：

```
reset(seq_id, frame_id):
    build tracker input; adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(...)
    state = make_initial_state(clean_points)
    (方案B) clean_ref_metrics = tracker_eval(state)
    step_id = 0; return obs(state)

step(action_id):
    next_state = apply_action_id(state, action_id, clean, cfg, step_id)
    (方案B) metrics = tracker_eval(next_state)   # 每步 1 次
    (方案A) metrics = None，仅 step_id==max 或扰动阈值时评估
    r = reward(metrics, imp, step_id, query_count)
    done = attack_success or step_id>=max_steps
    state = next_state; step_id += 1
    return obs, r, done, info
```

`tracker_eval` 复用 `eval_progressive_diffusion_attack_v2` 的 `evaluate_input_against_gt` / `evaluate_state`，与 BC fast 同口径。

---

## 8. 修订后的阶段计划

| 阶段 | 内容 | 与原方案差异 |
|---|---|---|
| **0（新增）** | **零训练 sanity baseline**：用 teacher `best_candidate_index` 映射成 action_id，跑"direct-action 但动作来自 teacher 标签、不查 tracker 验证"的推理，对比 BC fast top_k=5 的 success/stealth/runtime。回答"11 类 + 去验证是否丢效果"。 | 原方案没有；**应在写任何 PPO 代码前先做**，避免动作空间砍太狠才发现 |
| 1 | action mapping + 统一 `apply_action_id()`（BC/env/推理共用） | 强调三处共用同一函数 + 同 seed |
| 2 | **复用**现有 teacher NPZ，写 `(op,dir,patch)->action_id` 映射导出 BC 数据 | 原方案要重采，改为复用 9438 条 |
| 3 | BC 预训练 direct actor（**封装 `PointAttackRanker`**，软标签 KL），checkpoint 按 `selected_success - λ*selected_stealth` 存 | 原方案新写网络 + 按 val loss 存；改为复用网络 + reward-aware 选 ckpt |
| 4 | 在线 PPO fine-tune（reward 复用 teacher_score 形式 + step/query penalty） | reward 与 BC 对齐 |
| 5 | 评估：3seq20 → full test，**含 M2Track** | 原方案只 BAT；M2 forward 更贵、收益最大 |

---

## 9. 评估口径（训练与评估必须一致）

至少对比三方：raw v2 / BC fast top_k=5(方案B) / PPO direct-action。指标：

- **runtime**、**query_count（每次 tracker forward=1）** —— 这是本方案核心卖点，务必和 reward 的 query penalty 同一口径
- success_drop / precision_drop / attack_success_rate
- imperceptibility（chamfer / avg_disp / removed_ratio / changed_ratio）
- selected action distribution（看是否塌缩到单一动作）
- 平均 failure_step
- fair clean subset（clean_iou≥0.5 的子集，与现有脚本一致）

---

## 10. 主要风险与决策点

1. **动作空间收缩的代价**（最大风险）：37 候选 + top_k=5 tracker 验证 → 11 类 + 0 验证，是"更弱动作空间 + 更弱选择"的双重削弱。**靠阶段 0 sanity baseline 先量化**；若掉太多，优先加回 patch_candidate_k 到 4 或加 directional_fake。
2. **no-GT 推理终止判据**：方案 A 中间不查 tracker，无法靠 success 提前停。需让 policy 学"何时停"（加一个 stop 动作，或固定步数 + 扰动预算）。这是 A vs B 的核心权衡点。
3. **batch 一致性**：BC fast 注释（`v2_bc_fast.py:210` 上方）指出 PointNet++ CUDA 算子 batch=K vs K×batch=1 结果有差异。direct-action 每步只 1 次 forward，天然规避此问题——这是额外好处，但意味着 PPO 训练时若用批量 env 要小心同样的数值差异。
4. **policy 分布塌缩**：11 类里 progressive_noise 可能成为"安全动作"被过度选择。靠 entropy bonus（原方案 0.01 合理）+ 监控 action distribution 缓解。

---

## 附：原方案中已经正确、保留不动的部分

- 11 类离散动作、单 categorical actor（不做多头）—— 正确，最易调
- 标准 clipped PPO + GAE，超参（gamma .99 / lambda .95 / clip .2 / lr 3e-4 / ppo_epochs 4）—— 合理
- 先 BC warm-start 再 PPO fine-tune —— 正确，不要从随机 PPO 起
- 先用 GT reward 训练、推理再 no-GT —— 正确
- 保留 step/query penalty —— 正确，是省时间的关键监督信号
- 推理 argmax 不采样、每步生成 1 个动作 1 次查询 —— 正确
