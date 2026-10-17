# TTT-Discover 算法流程梳理（面向 VERL 复现）

本文基于 `discover/` 代码实现做流程级梳理，目标是为 `recipe/ttt_math_construct/` 的持续复现提供一份可直接对照的“运行闭环说明”。

## 1. 一句话总览

TTT-Discover 的核心闭环是：

1. 从状态池（PUCT）选父状态；
2. 对每个父状态并行采样 `k` 个候选程序（group rollout）；
3. 沙箱执行并计算 reward/raw_score，回写状态池；
4. 组内用 `entropic_adaptive_beta` 计算优势；
5. 用 on-policy importance-sampling 目标更新 LoRA；
6. 保存 checkpoint 与 sampler 状态，进入下一步。

---

## 2. 关键对象（代码中的抽象）

- `State`: 单个“可复用中间解”，包含 `construction`、`code`、`value`、`parents`、`observation`。
- `Environment`: 用当前 `State` 生成 prompt，执行模型输出代码并给出 `VerifyResult`。
- `StateSampler` / `PUCTSampler`: 维护状态池、PUCT 统计量、采样与剪枝。
- `TrajectoryGroup`: 同一个父状态下的 `k` 条 rollout（用于组内优势计算）。
- `RLDataset`: 每 step 产出一批父状态，每个父状态再扩成一个 `group`。

---

## 3. 训练主循环（逐 step）

主入口：`ttt_discover/discovery.py::discover()` -> `ttt_discover/rl/train.py::main()`

每个训练 step 的逻辑可抽象为：

1. **采样父状态**
   - `dataset.get_batch()` 调用 `sampler.sample_states(groups_per_batch)`。
   - 每个父状态构造一个 `EnvGroupBuilder`，并复制为 `group_size=k` 个环境实例。

2. **组内 rollout（同父多采样）**
   - 对每个 `EnvGroupBuilder` 并发调用 `do_group_rollout()`。
   - 每条轨迹通常是 1-turn：`initial_observation -> policy采样 -> env.step -> done`。
   - `env.step` 内部完成代码解析、执行、验收、reward 计算。

3. **状态池回写**
   - 成功样本：`sampler.update_states([child], [parent], save=False)`。
   - 失败样本：`sampler.record_failed_rollout(parent)`（只更新访问统计，不加新节点）。
   - 批次末尾 `dataset.flush(step=...)`，统一做 top-k 与落盘。

4. **组内优势计算**
   - 对每个 group 的总回报向量 `r_1...r_k` 计算 advantage（默认 `entropic_adaptive_beta`）。
   - 常量回报组可选过滤（`remove_constant_reward_groups`）。

5. **构造 token 级训练数据**
   - 将轨迹拼成 `Datum`：
     - `target_tokens`（左移）；
     - `logprobs`（采样时logp）；
     - `advantages`（同一轨迹动作 token 共用同一标量优势）；
     - `mask`（仅 action token 为 1，prompt token 为 0）。

6. **可选 KL 惩罚（对 base model）**
   - 计算 `logp_current - logp_base`，将惩罚项直接加到优势上。

7. **参数更新**
   - 默认 `loss_fn="importance_sampling"`，`num_substeps` 可拆分 minibatch。
   - 训练后保存权重并生成新的 sampling client（确保 on-policy）。

8. **checkpoint 与恢复**
   - 定期保存训练状态 + 采样器状态。
   - 恢复训练时会同时恢复 sampler 到同一步，避免“模型步数”和“状态池步数”错位。

---

## 4. PUCT 状态采样（核心公式）

`ttt_discover/tinker_utils/sampler.py` 的打分：

`score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n_i)`

其中：

- `Q(i)`：若访问过则用 `m_i`（该节点历史可达的最好子节点值），否则用当前 `value_i`；
- `P(i)`：按 `value` 排名得到的 rank prior；
- `n_i`：节点（及祖先）访问计数；
- `T`：全局展开计数；
- `scale`：当前池中 value 范围（避免 bonus 尺度失真）；
- `c`：`puct_c`。

实现细节：

- 多样本采样时启用 **lineage blocking**，避免同一祖先链/子树同时被选中。
- 更新时对每个 parent 只保留 top-k 子节点（`topk_children`）。
- 去重键通常来自 construction（或 code），防止重复状态污染 buffer。
- buffer 超限时保留初始种子 + 高 value 节点。

---

## 5. Entropic Adaptive-Beta 优势（TTT-Discover 默认）

对同一父状态的组内回报 `r`：

1. 先解 beta（目标 KL 预算）：
   - 定义 `q_beta(i) ∝ exp(beta * r_i)`；
   - 通过二分搜索使 `KL(q_beta || Uniform) ≈ delta`（默认 `delta=ln(2)`）。

2. 再算 leave-one-out 形式优势：

`e_i = exp(beta * (r_i - max(r)))`

`Z_i = (sum_j e_j - e_i) / (k - 1)`

`w_i = e_i / (Z_i + eps)`

`A_i = w_i - 1`

这比简单 `r_i - mean(r)` 更强调“相对同组其他样本的优势比”，并通过自适应 beta 稳定探索/利用平衡。

---

## 6. 环境与奖励接口的实际语义

TTT-Discover 的环境不是传统 MDP 长轨迹，而是“**一次提案 -> 一次外部程序评测**”：

- 模型输出的是可执行代码；
- reward evaluator 在沙箱/子进程中执行代码，校验合法性；
- 返回：
  - `reward`（训练信号）；
  - `raw_score`（任务真实指标，如 C5）；
  - `result_construction`（供状态复用）；
  - `stdout/msg`（回灌到下轮 prompt）。

最关键的一点：**状态池使用的是“可复用结果对象”而不是纯文本轨迹**，这使得 test-time RL 变成“树搜索 + 策略改进”的组合。

---

## 7. 与当前 `recipe/ttt_math_construct` 的对照映射

你当前目录已经基本对齐了 TTT-Discover 的骨架：

- `dataset.py` 对应 Discover 的 `SingleProblemDataset + Environment.get_question`。
- `state_pool.py` 对应 Discover 的 `PUCTSampler`（含 lineage blocking/top-k/prune）。
- `sampler.py` 对应 Discover 的“rollout 后回写状态池”逻辑。
- `erdos_reward.py` 对应 Discover 的 `SandboxRewardEvaluator.get_reward` 风格接口。
- `verl/trainer/ppo/core_algos.py` 中 `ttt_entropic_adaptive` 对应 Discover 的 `entropic_adaptive_beta`。

建议持续校准的关键点：

1. **组定义一致**：同一父状态的 `k` 个 rollout 必须作为同一 group 算优势；
2. **失败样本计数**：失败 rollout 也要更新 parent 访问计数（否则 PUCT 偏差）；
3. **状态 value 语义**：最小化任务要统一成“higher is better”的内部值（如 `value=-raw_score` 或 `reward=1/c5`）；
4. **去重与剪枝**：construction 去重 + per-parent top-k + 全局 buffer 上限；
5. **on-policy 更新**：每步更新后要用新权重继续采样，避免陈旧策略数据。

---

## 8. 复现时可直接使用的简化伪代码

```text
init model(theta), state_pool H, puct_stats (n,m,T)
for step in 1..S:
    parents = sample_with_puct(H, B)
    groups = []
    for p in parents:
        trajs = []
        for j in 1..k:
            y = model.generate(prompt_from(p))
            out = sandbox_eval(y, p)
            if out.success:
                child = make_state(out, parent=p)
                H.add(child)
            else:
                record_failed_visit(p)
            trajs.append((y, out.reward, out.raw_score, p))
        groups.append(trajs)
    A = entropic_adaptive_adv(groups.rewards)
    D = build_token_level_data(groups, A)
    theta = on_policy_update(theta, D)
    H.prune_and_flush(step)
```

---

## 9. 结论（给后续开发的落点）

TTT-Discover 不是“普通 PPO 换个 reward”，而是三件事耦合：

- **状态复用搜索（PUCT）**；
- **组内相对优势（adaptive entropic）**；
- **代码执行反馈闭环（construction 可继承）**。

后续在 `recipe/ttt_math_construct` 里迭代时，优先保证这三件事的语义一致性，性能通常会比单纯调学习率/采样温度更敏感。

