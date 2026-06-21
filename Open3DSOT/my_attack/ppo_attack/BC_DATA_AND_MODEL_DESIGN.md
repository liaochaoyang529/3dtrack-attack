# BC Point Policy 数据构造与模型设计说明

本文档说明当前用于 PPO 策略模型预训练的数据构造方式、BC 网络结构、设计理由，以及 10 epoch 训练结果。相关代码主要位于：

- `my_attack/ppo_attack/collect_v2_gt_teacher_dataset.py`
- `my_attack/ppo_attack/export_v2_teacher_dataset.py`
- `my_attack/ppo_attack/point_policy.py`
- `my_attack/ppo_attack/train_point_ranker.py`

## 1. 任务目标

目标不是直接训练一个分类器判断某个点云是否被攻击，而是训练一个可以作为 PPO 初始策略的点云策略模型。给定当前搜索区域点云状态和一组候选攻击动作，模型需要给每个候选动作打分，然后选择更可能让跟踪模型失败、同时扰动更隐蔽的攻击方案。

这里的策略动作包含两个层面：

1. 攻击算子：例如局部 patch shift、critical patch jitter、fake points、drop points、progressive noise、recovery。
2. 攻击参数：例如攻击方向、作用 patch、强度、drop ratio、fake ratio、recovery id。

因此当前 BC 预训练被设计成一个 candidate ranking 问题：每一步先由 teacher 枚举一批候选攻击动作，真实运行跟踪器并用 GT 评估每个候选动作的效果，然后 BC 网络学习把 teacher 认为更好的候选动作排到前面。

## 2. 数据从哪里来

数据由 `progressive_diffusion_attack_v2.py` 的攻击算子生成，但 teacher 选择使用 GT box 做监督。采集入口是：

```bash
Open3DSOT/my_attack/ppo_attack/collect_v2_gt_teacher_dataset.py
```

采集过程是：

1. 从 `training` split 中读取跟踪序列。
2. 对每个非首帧构造 tracker 输入。
3. 在当前攻击状态下枚举多个候选攻击动作。
4. 对每个候选动作真实调用 tracker，得到预测框。
5. 用当前帧 GT box 计算 IoU、center error、attack_success。
6. 同时计算扰动隐蔽性指标，例如 Chamfer distance、平均点位移、fake point ratio、removed point ratio、local density diff。
7. 根据 teacher score 选择当前 step 的最佳候选。
8. 把当前状态、候选动作、候选攻击后点云、teacher score 等写入 NPZ。
9. 把轨迹级索引和候选元数据写入 JSONL。

当前数据目录是：

```text
Open3DSOT/my_attack/outputs/ppo_teacher_success_stealth/
```

其中：

- `records.jsonl`：训练入口读取的轨迹索引文件。
- `raw_steps.jsonl`：原始 step 级记录。
- `summary.json`：采集统计。
- `point_npz/`：每个 step 对应的点云和候选动作张量。

当前采集统计如下：

```text
total_frames_seen:          24122
attacked_frames:            23585
step_records_collected:     141510
trajectories_collected:     23585
step_records_after_filter:  48548
trajectories_after_filter:  9438
records_written:            9438
max_steps:                  6
stealth_lambda:             2.0
success_bonus:              10.0
require_success:            true
max_stealth_score:          0.25
```

训练脚本会进一步从 `records.jsonl` 中展开所有 step，并过滤：

- `attack_success = true`
- `selected_stealth_score <= 0.25`

所以最终用于 BC 训练的 step 是“teacher 选中的动作攻击成功，并且隐蔽性分数不高于阈值”的数据。

## 3. 单条训练样本长什么样

一条训练样本对应一个攻击 step，而不是一个完整视频序列。它描述的是：

```text
当前点云状态 + 一组候选攻击动作 -> teacher 对候选动作的偏好分布
```

典型 NPZ 样本字段如下：

```text
clean_search_points        (1024, 3)       float32
current_points             (1024, 3)       float32
candidate_adv_points       (50, 1024, 3)   float32

candidate_op_id            (50,)           int64
candidate_direction_id     (50,)           int64
candidate_patch_center_idx (50,)           int64
candidate_strength         (50,)           float32
candidate_patch_ratio      (50,)           float32
candidate_drop_ratio       (50,)           float32
candidate_fake_ratio       (50,)           float32
candidate_recovery_id      (50,)           int64

candidate_teacher_score    (50,)           float32
best_candidate_index       scalar          int64
obs                        (21,)           float32
normalization_center       (3,)            float32
normalization_scale        scalar          float32
template_points            (512, 3)        float32
```

其中最重要的是：

- `clean_search_points`：未攻击的搜索区域点云。
- `current_points`：经过前面若干攻击 step 后的当前点云。
- `candidate_adv_points`：每个候选动作执行后的点云，当前训练没有直接输入这个字段，但它被保存下来，方便后续训练 transition model、critic、diff reward 或做可视化分析。
- `candidate_*`：候选动作的结构化描述。
- `candidate_teacher_score`：teacher 对每个候选动作的连续评分。
- `best_candidate_index`：teacher score 最大的候选动作索引。

候选攻击算子固定为 6 类：

```text
0 critical_patch_drop
1 critical_patch_jitter
2 directional_fake_points
3 local_patch_shift
4 progressive_noise
5 recovery
```

每个候选动作不是只保存一个离散类别，而是保存“算子 + 几何位置 + 方向 + 强度参数”。这是因为攻击策略真正要学的是几何条件下的动作选择，而不是只学某个全局类别偏好。

## 4. teacher score 如何定义

teacher score 在 `my_attack/ppo_attack/score.py` 中定义：

```text
score = 10 * (1 - IoU) + center_error - stealth_lambda * stealth
```

如果候选动作已经让 tracker 攻击成功，则额外加：

```text
+ success_bonus
```

当前数据采集使用：

```text
stealth_lambda = 2.0
success_bonus = 10.0
```

隐蔽性分数为：

```text
stealth =
    chamfer_distance
  + avg_point_displacement
  + 0.25 * fake_point_ratio
  + 0.25 * removed_point_ratio
  + 0.1  * local_density_diff
```

这个设计的含义是：

- IoU 越低越好，因为跟踪框越偏离 GT。
- center error 越大越好，因为目标中心漂移越明显。
- attack_success 直接给大 bonus，保证 teacher 优先选择能让跟踪失败的动作。
- stealth 越大越差，因此要扣分。

因此 teacher 不是单纯选择“扰动最大”的动作，而是在攻击效果和隐蔽性之间做加权权衡。

## 5. 为什么这样构造数据

### 5.1 使用 step 级 candidate ranking，而不是单一动作标签

如果只保存 teacher 最终选中的动作，训练会变成普通多分类。但这个任务里很多候选动作的效果可能非常接近，尤其是局部点云攻击中，不同 patch、方向、强度之间往往不是非黑即白。

所以当前保存所有候选动作及其 `candidate_teacher_score`，训练时使用 soft label：

```text
target = softmax(candidate_teacher_score / temperature)
```

这样模型学到的是 teacher 的偏好分布，而不是硬性要求它只模仿 top-1 动作。

优点是：

- 保留候选动作之间的相对优劣。
- 减少 GT teacher 噪声对训练的影响。
- 更适合作为 PPO 初始化，因为 PPO 后续仍然需要探索，不应该一开始就变成过度确定的 deterministic policy。

### 5.2 保存 clean 和 current 两份点云

模型输入不是只用当前点云，而是用：

```text
[current_xyz, current_xyz - clean_xyz]
```

也就是每个点有 6 维：

```text
x, y, z, dx, dy, dz
```

这么做的原因是攻击策略不仅要理解物体当前几何，还要知道当前点云相对干净点云已经被改动到了什么程度。

例如：

- 某个局部区域已经被 shift 过，就不应该继续无脑叠加强扰动。
- fake point ratio 已经偏高时，应当倾向于更隐蔽的局部 shift 或 recovery。
- 如果当前点云和 clean 点云差异很小，但 tracker 已经接近失败，可以选择低强度动作继续推进。

这个输入设计给模型显式提供“攻击状态”的信息，而不是让模型从单帧点云中隐式猜测。

### 5.3 保留完整 1024 点

跟踪模型的搜索点云使用 1024 点，因此当前 10 epoch 训练使用：

```text
--max_points 0
```

表示不下采样，完整使用 1024 点。

这对攻击策略很重要，因为局部攻击的有效区域可能只占很小一部分。如果下采样过重，候选 patch center 以及局部几何上下文可能被破坏，模型会更难判断“攻击哪里”。

### 5.4 按 sequence 做稳定划分

训练/验证/测试不是随机按 step 打散，而是用：

```text
job_name:local_sequence_id
```

的稳定 hash 做 split。当前划分为：

```text
train: 41940
val:    3611
test:   2997
```

这样可以降低同一条跟踪序列的相邻帧同时出现在 train 和 test 的泄漏风险。对跟踪任务来说，相邻帧高度相关，如果随机按 step 划分，测试指标会偏乐观。

## 6. BC 网络结构

当前 BC 网络定义在：

```text
my_attack/ppo_attack/point_policy.py
```

核心类是：

```python
PointAttackRanker
```

整体结构是：

```text
clean/current 点云
    -> normalize_points
    -> DGCNNLiteEncoder
    -> point_features + global_feature

候选动作
    -> op embedding
    -> direction embedding
    -> scalar encoder
    -> action encoder

候选 patch_center_idx
    -> 从 point_features 中 gather 局部 patch feature

global_feature + patch_feature + action_feature
    -> MLP scorer
    -> candidate_logits
```

同时网络还包含一个：

```text
value_head(global_feature) -> value
```

这个 value head 当前 BC 训练中还没有使用，但它是为后续 PPO 预留的 critic/value 初始化接口。

## 7. 点云编码器：DGCNN-lite / EdgeConv

点云 encoder 是轻量版 DGCNN：

```python
DGCNNLiteEncoder(
    input_channels=6,
    hidden_channels=(64, 128, 256),
    k=edge_k,
    global_dim=512,
)
```

每个 EdgeConv block 做：

```text
1. 在特征空间中为每个点找 kNN
2. 构造 edge feature: [x_i, x_j - x_i]
3. 用共享 1x1 Conv/MLP 编码边特征
4. 对邻居维度 max pool
```

当前训练使用：

```text
edge_k = 12
```

最后把不同层的点特征 concat，再投影到 512 维 point feature。全局特征由：

```text
global max pool + global mean pool
```

拼接得到，因此 global feature 是 1024 维。

## 8. 从归纳偏置角度理解这个设计

你希望模型“对任何类别都通用”，因此不能选择过强类别先验或过强模板结构先验的模型。当前 BC 网络的归纳偏置可以理解为中等偏弱，但仍然保留了点云几何任务必须的局部结构偏置。

### 8.1 相比 PointNet：局部几何偏置更强

PointNet 对每个点独立编码，然后全局池化。它的优点是类别先验弱、置换不变性强，但问题是局部几何关系弱。对于“攻击哪里”这个问题，只知道单点坐标和全局统计是不够的。

局部攻击依赖的信息包括：

- 局部点密度是否异常。
- patch 是否位于物体边界、角点、稀疏区域。
- 某个点附近的形状结构是否对 tracker 特征敏感。
- 局部扰动是否容易被看出来。

EdgeConv 显式建模 `x_j - x_i`，所以比 PointNet 更容易理解局部邻域结构。

### 8.2 相比强 Transformer：类别/数据偏置更弱，样本效率更高

如果直接用大型 point transformer，模型容量更强，能学更复杂的跨点关系，但也更容易在当前数据规模下学到：

- Car/Pedestrian 的形状先验。
- 某个 tracker、某个数据集的固定失败模式。
- teacher 搜索过程中的偶然偏差。

这和“通用类别策略模型”的目标不完全一致。DGCNN-lite 的偏置更克制：它强制模型从局部几何邻域和全局池化统计中做判断，而不是给它过强的全局关系建模能力。

### 8.3 EdgeConv 的偏置适合攻击策略

EdgeConv 的关键偏置是：

```text
局部相对几何重要
```

这正好对应点云攻击策略的核心问题：

- drop 哪个 patch。
- jitter 哪个 patch。
- shift 哪个局部区域。
- fake points 应该朝哪个方向制造漂移。
- 当前扰动是否已经破坏局部密度。

也就是说，模型不是只看到“这是车”或“这是行人”，而是看到“这个局部几何结构如果被动，会如何影响 tracker”。

### 8.4 为什么还要保留全局 feature

攻击动作虽然作用于局部，但效果是通过 tracker 的整体预测框体现的。只看 patch 局部不够，因为同一个局部结构在不同全局位置上意义不同：

- 物体前端和后端被 shift，对中心漂移方向的影响不同。
- 边界点和内部点的隐蔽性不同。
- 稀疏物体和密集物体允许的扰动幅度不同。

所以 scorer 输入同时包含：

```text
global_feature + patch_feature + action_feature
```

这让模型在“全局形状/状态”和“局部攻击位置”之间做条件判断。

### 8.5 为什么使用候选动作条件化，而不是直接输出所有动作参数

当前网络不是一次性回归攻击参数，而是对候选动作打分。这样做的归纳偏置更适合当前阶段：

- 动作空间由已有 v2 攻击算子约束，不会生成非法攻击。
- BC 学习难度从连续控制降低为候选排序。
- teacher 可以真实评估每个候选动作，监督信号更可靠。
- PPO 后续可以在这个候选动作空间上继续优化。

这相当于把“几何动作生成”的一部分先交给可解释的攻击算子库，把神经网络用于学习“什么时候选哪个算子、哪个位置、哪个方向、哪个强度”。

这个设计的偏置比端到端连续动作策略强，但比手写规则弱，适合作为 PPO 预训练阶段。

## 9. BC 是什么，训练用来做什么

BC 是 Behavior Cloning，行为克隆。它是一种模仿学习方法：给定专家或 teacher 的状态-动作数据，训练策略网络模仿 teacher 的行为。

在这里：

- 状态是当前点云攻击状态。
- 动作是一组候选攻击动作中的选择。
- teacher 是带 GT 评估的 v2 attack search。
- 标签不是单一 hard label，而是 teacher score 转成的 soft label 分布。

训练目标是：

```text
让 PointAttackRanker 的 candidate_logits 接近 teacher 的 candidate preference distribution
```

具体 loss 是：

```text
KLDiv(
    log_softmax(model_logits),
    softmax(candidate_teacher_score / temperature)
)
```

当前使用：

```text
temperature = 2.0
```

temperature 越高，soft label 越平滑，模型不会只盯着 top-1；temperature 越低，训练越接近 hard top-1 imitation。

BC 训练在整体 PPO 流程中的作用是初始化策略。直接从随机策略开始 PPO，探索空间太大，而且很多随机攻击动作要么无效，要么扰动明显。BC 先让策略学会 teacher 的基本攻击偏好，使 PPO 起点更接近“能攻击成功且相对隐蔽”的区域。之后 PPO 再根据真实 rollout reward 继续优化。

换句话说：

```text
BC 负责把策略带到合理区域；
PPO 负责在合理区域内继续探索和超过 teacher。
```

## 10. 当前训练结果

10 epoch 训练命令使用完整 1024 点：

```bash
/workspace/Open3DSOT/.venv/bin/python Open3DSOT/my_attack/ppo_attack/train_point_ranker.py \
  --records_jsonl Open3DSOT/my_attack/outputs/ppo_teacher_success_stealth/records.jsonl \
  --output Open3DSOT/my_attack/outputs/point_ranker_bc_1024_e10/best.pt \
  --epochs 10 \
  --batch_size 4 \
  --workers 2 \
  --max_points 0 \
  --edge_k 12 \
  --temperature 2.0 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --max_stealth_score 0.25 \
  --require_success
```

输出：

```text
checkpoint: Open3DSOT/my_attack/outputs/point_ranker_bc_1024_e10/best.pt
report:     Open3DSOT/my_attack/outputs/point_ranker_bc_1024_e10/best.json
```

数据划分：

```text
train: 41940
val:    3611
test:   2997
```

测试集结果：

```text
loss:              0.39025
top1:              0.14748
top3:              0.18719
regret:            2.72360
selected_success:  0.86019
selected_stealth:  0.09882
oracle_success:    1.00000
oracle_stealth:    0.07767
```

这里需要注意，top1 不高不一定表示模型完全失败。原因是 teacher soft label 很平滑，很多候选动作分数接近，temperature=2.0 后目标分布更分散。因此 loss 和 `selected_success / selected_stealth` 比单纯 top1 更能反映当前模型是否有用。

但是当前 10 epoch 版也暴露出一个问题：checkpoint 是按 `val loss` 保存的，而不是按最终攻击目标保存的。和 2 epoch 版相比，10 epoch 的 KL loss 略低，但 selected stealth 更高。这说明单纯拟合 teacher soft distribution 并不必然等价于得到最隐蔽的攻击策略。

后续更合理的模型选择指标应该改成类似：

```text
policy_score = selected_success - lambda * selected_stealth - alpha * regret
```

或者在 loss 中额外加入 success/stealth 感知项，使 BC 预训练目标和 PPO 最终 reward 更一致。

## 11. 当前设计的局限与下一步

当前设计适合作为 PPO 前的第一版 BC 初始化，但还有几个明确限制：

1. 网络现在只用 `clean_search_points` 和 `current_points`，没有直接用 `template_points`。如果后续发现跨类别泛化不足，可以考虑加入 template-search cross encoding。
2. `candidate_adv_points` 已保存但未输入网络。未来可以让模型比较 action 前后的局部几何变化，增强对隐蔽性的判断。
3. checkpoint 选择标准还是 KL loss，不是攻击 reward。下一步应改成 reward-aware validation。
4. 当前 teacher 仍来自离散候选集合，PPO 如果要学习连续强度或连续方向，需要在 action head 上进一步扩展。
5. 当前 BC 是 ranker，不是完整 actor-critic。`value_head` 已预留，但需要用 rollout return 或 teacher value 进一步训练 critic。

总体上，当前数据构造和 BC 网络的设计思想是：用 GT teacher 产生高质量候选排序监督，用较弱但保留局部几何建模能力的 DGCNN-lite 编码点云状态，用候选动作条件化打分来约束动作空间，最终为 PPO 提供一个不随机、可攻击、相对隐蔽的初始策略。
