# Standard Anchor R2 四小时实施计划

> Plan ID：`STD-ANCHOR-R2-4H`
> 上位计划：`STD-VISUAL-SPLIT-4H4H` 的任务一
> 任务名：`standard-anchor-r2`
> 计划时长：4 小时，由平台任务页控制
> 父模型：`model.ckpt-visionfull-28401.pkl`
> 目标 schedule：`visual_anchor_anneal_v2`
> 功能分支：`codex/anchor-r2`
> 状态：本地静态实现完成；评估入口回归已复现，必须先完成 P0 入口闭环，再进行
> Torch 张量测试、平台短启动和四小时长训

## 0. P0：评估入口闭环优先

在 Anchor R2 的任何短启动、固定 seed 评估或正式长训前，必须修复并核验 Camera 任务
的实际入口。`log-597491-18490378.zip` 已证明旧链路会把
`Unitree-Go2-Velocity-Camera` 自动回退为 `lbc_loco`：aisrv 初始化 `[LBC-Loco]`
并按错误评估 ID 查找视觉包，learner 子进程又重复覆盖为 `lbc_loco`。该轮的
`completed=0`、`timeout=16` 不是策略训练或 timeout 行为证据。

P0 验收要求：

1. 最终生效的评估配置显式包含
   `[env_conf].policy_entry = "visual_policy_optimization"`；平台任务页若覆盖 TOML，
   以容器回读的最终配置为准。
2. 主进程和环境子进程都调用 `agent_ppo.conf.conf._infer_stage_for_eval`，并在日志中
   出现显式入口选择和 `Stage: visual_policy_optimization`；不得出现
   `inferred from TOML task_name`、`Stage: lbc_loco` 或 `[LBC-Loco]`。
3. 同步后回读容器中的 `conf.py`、`agent.py` 和最终 TOML SHA256；客户端的
   `bundle verified` 只能证明传输，不证明平台实际加载了该版本。
4. 不修改或依赖平台会覆盖的 `server/isaac_env/base_env.py`；入口兼容逻辑必须留在
   `agent_ppo.conf.conf`。任一证据缺失只记录 warning，不能把评估结果用于晋级或宣布
   训练有效。

这是一项实验流程优先级，不是算法运行时硬门禁；NaN/Inf、权重损坏和 checkpoint 写入
失败仍按本卡既定必要停止规则处理。

## 1. 唯一目标

本任务只完成视觉学生的分段解冻、Critic 适配和 action/latent anchor 退火。视觉
学生始终独立执行环境动作；冻结的 S0 只提供同一状态上的 action/latent 参考，不参与
动作选择。

本任务不承担：

- zero、低速、转向或横移 command 泛化；
- 地形重配、严格均匀难度 reset 或上楼样本加权；
- 步态纠正或新增 reward；
- 深度增强、动力学随机化、摩擦随机化、观测噪声或外力 push；
- ONNX/Jetson 导出；
- 第二个四小时 `standard-command-r1` 的渐进 command 调度。

任务一没有通过固定评估时，任务二不得启动。

## 2. 开始实施前的工作区规则

当前工作树已有大量未提交修改和未跟踪文件。执行 Agent 必须：

1. 先执行 `git status --short --branch` 并把清单记录到实施报告；
2. 不清理、不移动、不暂存 `.zcode/`、
   `archive/代码存档/hjcnew蒸馏1_10288/` 或其他任务外文件；
3. 不使用 `git reset --hard`、`git checkout --`、`git add .`；
4. 逐文件吸收现有 R3 修改，只替换与 Anchor R2 冲突的部分；
5. 训练逻辑改造与历史文件清理分别提交，不能混成一个不可审查提交；
6. 不直接在 `main` 开发或推送。

建议提交顺序：

```text
1. feat(server): add four-hour visual anchor r2 schedule
2. test(server): cover anchor r2 resume and worker bridge
3. chore(server): prune inactive training configurations
4. docs: freeze anchor r2 execution contract
```

如果不需要拆成四个提交，至少也必须将“训练功能”与“历史文件删除”分成两个提交。

## 3. 训练域固定值

配置必须恢复 `visionfull-28401` 的实际源域：

```toml
[env]
num_envs = 256
episode_length_s = 25

[terrain]
mode = "standard"
num_rows = 10
num_cols = 20
difficulty_range = [0.0, 1.0]
curriculum = true
max_init_terrain_level = 9

[terrain.standard_wall]
enabled = true

[terrain.standard.pyramid_slope]
proportion = 0.20

[terrain.standard.pyramid_slope_inv]
proportion = 0.20

[terrain.standard.pyramid_stairs]
proportion = 0.20

[terrain.standard.pyramid_stairs_inv]
proportion = 0.40

[terrain.standard.maze]
proportion = 0.0
```

地形语义固定为：

```text
pyramid_stairs      = 上楼
pyramid_stairs_inv  = 下楼
```

command 必须使用 Isaac 原生重采样器，不经过自定义 bucket：

```toml
[commands]
resampling_time = [6.0, 10.0]

[commands.ranges]
lin_vel_x = [0.3, 1.3]
lin_vel_y = [-0.2, 0.2]
ang_vel_yaw = [-0.3, 0.3]

[commands.limit]
lin_vel_x = [0.3, 1.3]
lin_vel_y = [-0.2, 0.2]
ang_vel_z = [-0.3, 0.3]

[commands.buckets]
enabled = false

[commands.worker_progressive]
enabled = false
```

不得保留当前 R3 的 `[300,300]` fallback、`vx=[0,1.3]` 或 enabled progressive
scheduler。任务一日志应明确打印“native S0 command sampler active”，而不是
`[worker command] enabled`。

`commands.ranges.ang_vel_yaw` 与 `commands.limit.ang_vel_z` 是平台历史 API 对同一
Z 轴角速度 command 使用的两个键名，不是 yaw/Y 轴语义混用。执行时保持这两个既有
键名，不自行重命名配置 schema。

保持关闭：

```text
camera.depth_camera.augmentation.enabled = false
domain_rand.enable_domain_rand = false
domain_rand.randomize_friction = false
domain_rand.push_robots = false
noise.add_noise = false
```

Anchor R2 **冻结任务开始时当前 R3 的完整 reward surface**。不得删除任何
`[rewards.*]` 配置块，也不得修改 reward 权重、参数、函数或触发条件。当前工作树已经
包含以下 command 兼容行为，它们是 Anchor R2 的冻结基线，不是本任务新增项：

```text
track_lin_vel_xy = 3.0
track_ang_vel_z = 1.0
forward_velocity = 0
trot_gait 仅在 command 模长 > 0.1 时生效
joint_position_penalty.stand_still_scale = 2.0
```

`server/agent_ppo/feature/reward_process.py` 以 command 模长 gating 已存在的版本作为冻结
基线；活动 TOML 固定 `stand_still_scale=2.0`。实施后用测试确认没有继续增删 reward 或
改值。“冻结完整 surface”不等于只保留上面五项；TOML 中已有的其他 reward 必须原样
保留。

## 4. 四小时墙钟调度

### 4.1 调度表

| 墙钟时间 | Actor | LSTM + output | CNN | Critic | Action anchor | Latent anchor | 标签 |
|---|---|---|---|---|---:|---:|---|
| 0--45 min | 冻结 | 冻结 | 冻结 | 训练 | `1.00` | `0.25` | `anchorcritic` |
| 45--90 min | 训练 | 冻结 | 冻结 | 训练 | `1.00 → 0.75` | `0.25` | `anchoractor` |
| 90--180 min | 训练 | 训练 | 冻结 | 训练 | `0.75 → 0.35` | `0.25 → 0.10` | `anchoranneal` |
| 180--240 min | 训练 | 训练 | 冻结 | 训练 | `0.35` | `0.10` | `anchorfinal` |

表中所有 `A → B`（如 `1.00 → 0.75`、`0.25 → 0.10`）都表示在该 phase 的墙钟区间内
**按墙钟线性插值**，不是 phase 起点瞬间的阶梯跳变。每个 phase 起点的值等于上一 phase
终点的值，保证 anchor 曲线连续；`anchorfinal` 起点 `0.35/0.10` 与 `anchoranneal`
终点严格相等。

CNN 在整个任务中永久 `eval()`、`requires_grad=false`，且不进入任何 optimizer group。
表中保留 CNN 列只是要求每个 phase 都重复核验这一事实，不表示 CNN 会随 phase 切换。

配置用小时传入算法时，对应：

```text
anchor_schedule_hours = [0.0, 0.75, 1.5, 3.0]
action_anchor_schedule = [1.00, 1.00, 0.75, 0.35]
latent_anchor_schedule = [0.25, 0.25, 0.25, 0.10]
anchor_phase_labels = [
  anchorcritic,
  anchoractor,
  anchoranneal,
  anchorfinal,
]
anchor_phase_end_hours = [0.75, 1.5, 3.0]
```

phase 区间采用左闭右开语义：

```text
[0.0, 0.75) = anchorcritic
[0.75, 1.5) = anchoractor
[1.5, 3.0) = anchoranneal
[3.0, +inf) = anchorfinal
```

最后一个 phase 没有独立 end knot 是有意设计。`t >= 3.0h` 时 Action/Latent anchor
分别钳制在 `0.35/0.10`，不得外推，也不得回落到 legacy phase。

TOML 继续使用当前 Agent 已实现的分钟键，不能直接把上述 `*_hours` 名称写进 TOML：

```toml
anchor_schedule_minutes = [0.0, 45.0, 90.0, 180.0]
anchor_phase_end_minutes = [45.0, 90.0, 180.0]
```

`Agent._init_visual_ppo()` 负责除以 60 后传给算法。执行 Agent 必须同时测试 TOML
分钟值和算法小时值，避免键名改变后悄悄传入 `None`。

算法插值核对点：

| 时间 | Action | Latent | Phase |
|---:|---:|---:|---|
| 0 min | 1.00 | 0.25 | `anchorcritic` |
| 45 min | 1.00 | 0.25 | `anchoractor` |
| 67.5 min | 0.875 | 0.25 | `anchoractor` |
| 90 min | 0.75 | 0.25 | `anchoranneal` |
| 135 min | 0.55 | 0.175 | `anchoranneal` |
| 180 min | 0.35 | 0.10 | `anchorfinal` |
| 240 min | 0.35 | 0.10 | `anchorfinal` |

### 4.2 学习率

```text
Critic：0--45 min = 3e-4；45 min 后 = 1e-4
Actor：1e-5
LSTM + output：5e-6
CNN：无 optimizer group，始终 requires_grad=false
```

`std`/`log_std` 与 Actor 使用相同的冻结状态和学习率组。

`StandardVisualPPOConfig` 的 class fallback 固定为 Actor `1e-5`、LSTM `5e-6`、
Critic `1e-4`。`critic_warmup_learning_rate=3e-4` 只存在于本任务活动 TOML，并且只在
`anchorcritic` 由 `_set_phase_learning_rates()` 选用；缺少 warmup 键时回退到 Critic
fallback `1e-4` 并打印 warning，不得静默采用 `3e-4`。

### 4.3 墙钟语义

- 平台任务页控制实际四小时；`max_iterations` 只保留高安全上限；
- 使用独立 `anchor_session_elapsed_hours` 推进调度；
- `elapsed_training_hours` 只记录累计训练，不得决定 Anchor R2 phase；
- 同一 `visual_anchor_anneal_v2` bundle 中断恢复时续接 anchor session clock；
- 从 S0、R3、R1 或其他 schedule 开始时，anchor session clock 必须从 0 开始；
- live LSTM hidden 不写入 checkpoint，恢复后 reset 环境并清零学生和 S0 hidden。

## 5. 逐模块修改规格

### 5.1 活动 TOML

文件：

```text
server/agent_ppo/conf/train_env_conf_standard_visual_policy_optimization.toml
```

修改：

1. 文件标题改为 Anchor R2 四小时任务；
2. 写入第 3 节的 terrain、command、randomization 和 reward 固定值；
3. `run_name = "standard-anchor-r2"`；
4. `schedule_mode = "visual_anchor_anneal_v2"`；
5. `initial_parent_model_id = 28401`；
6. 写入第 4 节的 schedule、phase 和学习率；
7. `task_end_hours = 4.0` 仅供记录和旧字段兼容，不负责结束任务；
8. `save_interval_minutes = 10.0`；
9. `resume_first_save_minutes = 2.0`；
10. `anchor_checkpoint_minutes = [45.0, 90.0, 180.0, 230.0]`。

230 分钟的额外保存用于规避平台在 240 分钟附近直接回收任务，不能只依赖正好
240 分钟的边界回调。

### 5.2 StageConfig

文件：

```text
server/agent_ppo/conf/conf.py
```

修改：

- `Config.CURRENT` 继续为 `StandardVisualPPOConfig`；
- 更新注释，明确当前活动任务是 Anchor R2，而不是 R3；
- 将 `StandardVisualPPOConfig.ckpt_name` 更新为不误导的
  `model.ckpt-anchorfinal`，但 Visual PPO 实际保存仍以当前 phase label 为准；
- class fallback 学习率固定为 Actor `1e-5`、LSTM `5e-6`、Critic `1e-4`；
- `critic_warmup_learning_rate=3e-4` 只从活动 TOML 读取；缺失时使用 Critic fallback
  `1e-4` 并打印 warning；
- 不新增第二个并行 StageConfig，避免活动入口再次分叉。

### 5.3 AlgorithmVisualPPO

文件：

```text
server/agent_ppo/algorithm/algorithm_visual_ppo.py
```

需要修改的符号：

```text
_configure_anchor_schedule
_configure_command_schedule
_set_trainable_phase
_set_phase_learning_rates
save_training_bundle
load_training_bundle
```

具体要求：

1. 当前代码尚不存在 `visual_anchor_anneal_v2` 分支；必须在
   `_configure_anchor_schedule` 中显式新增该 mode，并继续执行完整性、单调性、
   knot/phase 数量和边界范围校验；
2. `_configure_command_schedule` 对该 mode 固定返回 `target_probability=0.0`，不要求
   progressive command 数据；
3. `_set_trainable_phase` 实现：

```text
anchorcritic：Actor/std=false，RNN/output=false，CNN=false，Critic=true
anchoractor： Actor/std=true， RNN/output=false，CNN=false，Critic=true
anchoranneal：Actor/std=true， RNN/output=true， CNN=false，Critic=true
anchorfinal： Actor/std=true， RNN/output=true， CNN=false，Critic=true
```

4. `_set_phase_learning_rates` 在 `anchorcritic` 使用 Critic warmup LR，其他 phase 使用
  正常 Critic LR；
5. 不让历史 `anchor_anneal_v1` 或 R3 标签隐式匹配 Anchor R2；
6. 训练质量指标只产生 warning 和诊断 checkpoint，不因 hard termination、KL、anchor
   MSE 或动作幅度自动中止任务或永久冻结 schedule；
7. NaN/Inf、严格 shape/key 不兼容、训练权重损坏或 checkpoint 写入失败仍立即失败；
8. `anchor_inference()` 保持冻结 S0 `eval()` 和 `requires_grad=false`；
9. optimizer 参数集合必须只包含 Actor/std、RNN/output 和 Critic，CNN 与 S0 不得进入。

`visual_anchor_anneal_v2` 必须是显式分支，不能依赖当前“未知 mode 走
legacy_three_phase_v1”的 fallback。支持集合至少明确包含：

```text
legacy_three_phase_v1
anchor_anneal_v1
visual_recovery_split_v1
visual_anchor_anneal_v2
```

新 mode 缺少完整 schedule 时必须报配置错误；不能静默生成
`rlcritic/rlactor/rlfull`。同时更新 baseline safety 的 mode 白名单，否则 resume 后会
使用错误的历史 baseline 语义。

新增 `warning_only_safety = true` 配置，并让 Anchor R2 的 hard termination、KL、
anchor MSE 和动作幅度阈值只更新计数、打印 warning、请求诊断保存；不得设置
`actor_updates_paused=true`、`anchor_schedule_frozen=true` 或回滚内存快照。历史 mode
保留原行为，避免改变旧 checkpoint 的恢复语义。

### 5.4 checkpoint 保存与恢复

仍在：

```text
server/agent_ppo/algorithm/algorithm_visual_ppo.py
```

`save_training_bundle()` 在兼容现有 `kaiwu_train_v1` 的前提下增加：

```text
training_state.anchor_session_elapsed_hours
training_state.trainable_modules = {
  cnn: false,
  actor: bool,
  lstm: bool,
  critic: true,
}
training_state.schedule_mode = visual_anchor_anneal_v2
training_state.run_name = standard-anchor-r2
training_state.source_parent_model_id = 28401
```

保留 `anchor_elapsed_hours` 作为旧 reader 兼容别名；两者写入相同值。loader 读取优先级：

```text
anchor_session_elapsed_hours
→ anchor_elapsed_hours
→ 0.0
```

**checkpoint schema 不为 periodic 保存 deadline 扩展**：保存去重（§5.8）的"下次 periodic
deadline"只存活于单次进程生命周期，不写入也不从 checkpoint 恢复。断点恢复后刻意重新
执行 `resume_first_save_minutes`（2 分钟）首存，随后按 10 分钟 cadence；即使落在上次
boundary 保存后的 10 分钟窗口内也视为安全保存，不算重复保存缺陷。

以下所有读写点必须一起迁移，不能只改 checkpoint codec：

```text
Agent 传入/日志字段
visual_ppo_workflow 的 resume_anchor_elapsed_h
workflow 每轮写回算法的 session clock
anchor checkpoint boundary index
phase/weight 计算
monitor telemetry
save_training_bundle
load_training_bundle
```

恢复分三种情况（`load_mode` 字段用于 §6 启动日志，三值固定见 §6）：

| 输入 | `load_mode` | 恢复内容 | 重置内容 |
|---|---|---|---|
| 精确 S0 `visionfull-28401` | `s0` | VisionEncoder、Actor；复制为冻结 S0 anchor | Critic、optimizer、std、所有时钟、安全窗口；使用当前配置 seed 初始化 RNG |
| 同 mode Anchor R2 bundle | `anchor_resume` | VisionEncoder、Actor、Critic、std、optimizer、RNG、anchor clock | live hidden |
| 其他 Visual PPO schedule | `schedule_migration` | 只把 VisionEncoder、Actor 当作新父策略，并打印 warning | Critic、optimizer、保存的 RNG、anchor clock、安全状态；使用当前配置 seed 初始化 RNG |

任务一首次运行不得读取 `latest`，也不得让同 ID 的历史 `vis*` 文件抢在
`visionfull-28401` 前面。身份、文件名和内嵌 ID 差异只打印 warning；网络 key/shape
不兼容仍不能 partial load。

RNG 语义固定为：**S0 anchor 没有独立 RNG 状态**——冻结 S0 仅以 `eval()` 推理，前向不
消耗 RNG，因此不存在"S0 RNG"概念。只有同一 `schedule_mode=visual_anchor_anneal_v2` 的
断点续训（`anchor_resume`）才恢复 checkpoint 里保存的 Python/NumPy/Torch/CUDA 进程级
RNG；`s0` 首训和 `schedule_migration` 都忽略 bundle RNG，并从活动 TOML 的
`[env_conf].seed` 重新初始化进程级 Python、NumPy、Torch、CUDA RNG。不得新增第二套 seed
配置。三种路径均不恢复 live LSTM hidden，加载完成后必须 reset 环境并清零学生与 S0
hidden。

### 5.5 checkpoint 标签与候选顺序

文件：

```text
server/agent_ppo/checkpoint_io.py
```

新增独立常量：

```text
VISUAL_ANCHOR_R2_PHASE_LABELS = (
  anchorcritic,
  anchoractor,
  anchoranneal,
  anchorfinal,
)
```

全部满足：

```regex
^model\.ckpt-[a-z]+-[0-9]+\.[^.]+$
```

同一平台 ID 的 Anchor R2 resume 顺序：

```text
anchorfinal → anchoranneal → anchoractor → anchorcritic
```

候选 ID 规则必须先于标签优先级执行：显式数字选择器只允许搜索文件名数字 ID 完全相同
的候选，任何情况下都不得跨 ID fallback。若文件名 ID 与 bundle 内嵌
`platform_model_id` 不同，继续使用操作者选中的文件但打印两者和醒目 warning。
`latest` 只供评估和诊断便利使用，任务一训练入口禁止使用；`latest` 解析日志必须同时
打印：

```text
requested_selector=latest
resolved_filename_id=<文件名数字 ID>
bundle_platform_model_id=<内嵌 ID 或 unknown>
```

**ID 约束必须在 `checkpoint_io.py` 候选层实现，不能只依赖 Agent**。具体四条规则：

1. **显式数字 ID**：`visual_rl_checkpoint_candidates` / `visual_eval_checkpoint_candidates`
   先按解析出的文件名数字 ID 过滤（只保留与请求 ID 完全相等的），**再**按标签优先级
   排序；ID 不等的候选在任何情况下都不进入排序列表。
2. **`latest`**：先解析目录内 Anchor R2 标签下的最大文件名数字 ID，**然后只在该 ID 内**
   按标签排序；不得把次大 ID 的更高优先标签排在前面。
3. **覆盖范围**：`visual_rl_checkpoint_candidates` 和 `visual_eval_checkpoint_candidates`
   两个函数都必须执行上述规则；`_latest_visual_rl_checkpoint_candidates` 只是 `latest`
   路径的薄封装，核心 ID 解析仍由这两个公开候选函数承担。
4. **Agent 职责边界**：Agent 只负责传入 selector（显式 ID 或 `latest`）、选择候选函数
   返回的命中文件、打印 selector / 文件名 ID / bundle ID 三类字段；**不得**在 Agent
   层自行 glob 或跨 ID 补候选。

需要同步修改的候选函数不止标签常量：

```text
visual_rl_checkpoint_candidates
_latest_visual_rl_checkpoint_candidates
visual_eval_checkpoint_candidates
相关 discovered glob/allowed_labels 过滤
```

否则新文件虽然能保存并通过正则，却不会被 resume 或 Camera eval 自动发现。

首次 `28401` 加载仍必须将 `visionfull-28401` 放在所有历史同 ID 文件之前。Camera
评估候选先尝试 Anchor R2 标签，再尝试被取代 R3、历史 R1/RL 和 Stage-4 vision
标签；保留旧标签 parser 以兼容已经生成的模型，不因清理历史 TOML 删除 loader 兼容。

### 5.6 Agent 初始化、加载和保存

文件：

```text
server/agent_ppo/agent.py
```

修改：

- 构造器把 Anchor R2 schedule 和学习率完整传给算法；
- 初始 S0 选择日志改为 Anchor R2，打印 requested ID、实际路径、bundle ID、父血缘和
  SHA256；
- 保持 `prefer_source_parent=True` 只用于首次 `28401`；
- 显式数字选择器只搜索同一文件名 ID；请求 `latest` 时按第 5.5 节打印 selector、解析后
  文件名 ID 和 bundle ID；
- `save_model()` 继续只生成一个 phase 训练包，不额外生成 `rlfull`、`locomotion` 或
  `lbc_loco` 同 ID 副本；
- 保存日志打印 phase、两类 anchor、anchor session clock、可训练模块、平台 ID 和
  SHA256；
- Camera eval 成功日志必须分别打印操作者请求选择器、解析后的文件名 ID 和 bundle ID，
  不能用 bundle 内嵌 ID 覆盖操作者请求字段。

### 5.7 平台 worker 边界

> **实施基线修正（2026-07-25）**：本节原假设 `progressive_command_scheduler.py`
> 源码已存在于活动树。实际核查 `codex/anchor-r2` 分支基线 `430b791`：该源文件
> **不存在**（磁盘与 git 历史均无），仅有未跟踪的 `.pyc`；活动源码对
> `apply_progressive_command`/`read_worker_timeouts`/`write_runtime_state` 的引用为
> **零**。因此本节"重命名"前提不成立——没有旧模块要重命名，也没有耦合要解。
>
> **最终实施范围**：不新建 `worker_runtime_bridge.py`。Anchor R2 的 timeout
> 发布由 `base_env.py` 现有的 `time_outs` 读取逻辑（通用 truncation 处理）承担，
> command override 由 TOML 关闭 `commands.worker_progressive.enabled=false` 与
> `commands.buckets.enabled=false` 保证。第二个四小时若需要 progressive command，必须
> 作为独立功能实现并独立验证，不能预埋进本轮。

本轮只要求：

1. 活动 TOML 的两个 command override 开关都为 `false`；
2. `agent_ppo` 不包含或 import 不存在的 progressive scheduler/worker bridge；
3. Agent 启动日志打印 `command_sampler=native`、`resampling_time`、ranges 和两个开关；
4. timeout/truncation 语义继续由平台 worker 提供，不能把模板
   `reached max length` 日志解释为 hard termination；训练 rollout 不读取其
   `infos["all_done"]`，因此该全局帧标记不结束 Anchor R2 workflow；
5. 第二个四小时的 command telemetry 与调度器留给其独立功能分支。

提交结束时以下扫描必须无活动源码命中：

```bash
rg -n 'progressive_command_scheduler|apply_progressive_command' \
  server/agent_ppo server/tests --glob '*.py'
```

不得仅仅把 `worker_progressive.enabled=true` 配成全 source，因为那仍会改写 command，
不等于使用 S0 原生训练域。

### 5.8 Visual PPO workflow

文件：

```text
server/agent_ppo/workflow/visual_ppo_workflow.py
```

修改：

- 日志从 `single-run R3` 收敛为通用 schedule 日志；
- 使用 `anchor_session_elapsed_hours` 推进 phase；
- 监控 phase 映射加入 `anchorcritic/actor/anneal/final`；
- 删除本任务无意义的 `target_p` 决策逻辑，仍可记录固定 `0.0`；
- 10 分钟 cadence 与 45/90/180/230 分钟 milestone 是两类保存触发器；前三个 milestone
  是 phase boundary，230 分钟是回收前保存点；
- 若两个触发器发生在同一 workflow loop 或相距 `delta <= 60.0s`（严格定义，含边界），只
  调用一次 `save_model()`；`delta > 60.0s` 时各保存一次。boundary 保存同时满足该次
  cadence 时，下一次 periodic 保存从 boundary 实际保存时刻顺延 10 分钟；
- 该去重规则**只在单次进程生命周期内有效**：periodic deadline 不写入 checkpoint、不跨
  resume 维护。断点恢复后刻意重新执行 `resume_first_save_minutes`（2 分钟）首存，随后按
  10 分钟 cadence；即使落在上次 boundary 保存后的 10 分钟窗口内也视为安全保存，不算
  重复保存缺陷（这是恢复安全保存的刻意设计）；
- 性能 warning 触发时额外保存诊断包但不结束任务；
- 不在代码内以四小时墙钟退出，平台负责终止；
- `max_iterations` 到达仍执行最终保存并退出。

### 5.9 不允许依赖 base_env

> **实施基线修正（2026-07-25，最终）**：本地 `base_env.py` 必须恢复为操作者提供的
> 平台版本并保持字节一致，不在其上叠加 R3 或 Anchor R2 功能。平台再次覆盖该文件时，
> 行为仍一致；Anchor R2 的业务逻辑全部位于 `agent_ppo` 和 TOML。
>
> **本轮实施范围**：`base_env.py` 恢复并锁定为操作者提供的平台版本
> `sha256=75ebdaf6888e94262598a26db1586b2598cb474422e3382b6bba9e96ddbb6e67`，不在其上
> 增加 Anchor R2 逻辑；command 控制完全靠 TOML
> （`commands.buckets.enabled=false` + `commands.worker_progressive.enabled=false`）；
> `agent_ppo/` 活动代码不 import `isaac_env.base_env`（静态测试守卫）。旧平台 worker
> 调用的 `_infer_stage_for_eval` 兼容入口位于 `agent_ppo/conf/conf.py`；深度预处理通过
> `agent_ppo/conf/depth_config.py` 按活动 Stage TOML 缓存解析，不依赖环境私有字段。
> 原列符号
> `reset_root_state_uniform_level_sampled` / `commands.progressive` /
> `terrain_type_x_level logging` 在 `430b791` 基线本就 0 处命中，无需处理。

文件：

```text
server/isaac_env/base_env.py
```

本任务不得新增或依赖其中的：

```text
reset_root_state_uniform_level_sampled
commands.progressive
command bucket telemetry
timeout truncation patch
terrain_type_x_level logging
```

当前文件以操作者附件为唯一基准，SHA256 测试负责防止后续修改漂移；不要从分支 HEAD
或 R3 版本恢复它。

增加静态契约测试：活动 `agent_ppo` 文件不得 import `isaac_env.base_env`；只允许读取
`base_env.py` 字节计算 SHA256，不得通过其源码符号证明 Anchor R2 功能存在。

同步 manifest 可以继续包含 `isaac_env`，但 Anchor R2 的验收不能依赖该文件上传后
仍被平台保留。

## 6. 启动日志契约

`load_mode` 字段固定三值，分别对应 §5.4 的三种恢复路径：

```text
load_mode=s0                  # 精确 visionfull-28401 首训
load_mode=anchor_resume       # 同 mode Anchor R2 bundle 续训
load_mode=schedule_migration  # 其他 Visual PPO schedule 迁移
```

`resume_anchor_session_h` 与 `load_mode` 必须一致：`s0`/`schedule_migration` 为
`0.000`，`anchor_resume` 为 checkpoint 里恢复的实际值。

平台启动后三分钟内必须看到（以下为首训 `s0` 路径示例）：

```text
[VisualPPO] run=standard-anchor-r2
schedule=visual_anchor_anneal_v2
requested_parent_id=28401
loaded_path=/data/pre_model/ckpt/model.ckpt-visionfull-28401.pkl
loaded_sha256=<64 hex>
load_mode=s0
resume_anchor_session_h=0.000
phase=anchorcritic
trainable={cnn:false, actor:false, lstm:false, critic:true}
action_anchor=1.000
latent_anchor=0.250
[VisualPPO] configured environment contract: command_sampler=native, ...
depth_augmentation_enabled=False
```

首次 10 分钟内必须看到一个探活兼容的：

```text
model.ckpt-anchorcritic-<平台ID>.pkl
```

以下情况只 warning，不阻断 rollout：

- bundle 内没有 `platform_model_id`；
- bundle ID 与请求 ID 不同；
- lineage parent unknown；
- `reached max length` 平台模板日志；
- hard termination、KL、anchor MSE 或动作幅度超出诊断阈值。

以下情况必须停止：

- S0 缺少 VisionEncoder/Actor 必要 key；
- shape 不兼容；
- 模型参数或 loss 出现 NaN/Inf；
- checkpoint 文件写入失败或写后为空；
- optimizer 参数包含 CNN 或 S0 anchor。

## 7. 历史文件清理计划

清理目的是让 `server/agent_ppo/conf/` 只保留当前代码真正可达的训练入口，不是重写
Git 历史。精确旧版本继续由 Git commit/tag 和 `shared/分析记录/` 保存。

### 7.1 必须保留

```text
server/agent_ppo/conf/train_env_conf_standard_locomotion.toml
server/agent_ppo/conf/train_env_conf_standard_standard_ref_distill.toml
server/agent_ppo/conf/train_env_conf_standard_lbc_loco.toml
server/agent_ppo/conf/train_env_conf_standard_visual_policy_optimization.toml
```

它们分别对应 `conf.py` 中四个实际 StageConfig。以下也保留：

```text
server/agent_ppo/conf/conf.py
server/agent_ppo/conf/monitor_builder.py
server/agent_ppo/conf/__init__.py
shared/分析记录/
archive/
checkpoint_io.py 中的历史 checkpoint 标签兼容
```

`server/agent_ppo/conf/train_env_conf_standard_stair_inv_finetune.toml` 暂时保留为审计项。
它虽无当前 StageConfig，但近期分析记录仍把它描述为实际训练配置；在确认其唯一事实已
迁入历史索引前，不得由本实施任务删除，也不得描述为当前活动入口。

### 7.2 已在工作树删除，正式保留删除状态

```text
server/agent_ppo/conf/train_env_conf_standard_standard_distill_1.toml
server/agent_ppo/conf/train_env_conf_standard_standard_distill_2_stair.toml
server/agent_ppo/conf/train_env_conf_standard_standard_distill_3_action.toml
server/agent_ppo/conf/train_env_conf_standard_standard_visual_ppo_1_heading.toml
server/agent_ppo/conf/train_env_conf_standard_standard_visual_ppo_2_d5.toml
server/agent_ppo/tests/test_standard_distill_1_config.py
server/agent_ppo/tests/test_standard_distill_2_stair.py
server/agent_ppo/tests/test_standard_distill_3_action.py
server/agent_ppo/tests/test_standard_visual_ppo_1_heading.py
server/agent_ppo/tests/test_standard_visual_ppo_2_d5.py
```

不得为了“恢复测试覆盖”把这些失败路线重新加回活动树。

### 7.3 删除已证明无活动入口的 Standard TOML

```text
server/agent_ppo/conf/train_env_conf_standard_hjcnew10288_lbc_loco.toml
```

该文件没有对应 `StageConfig`，且用途已由当前 LBC 配置和 Git 历史覆盖。删除前在
Changelog/实施报告中记录最后所在 commit，之后
通过 Git 历史回看，不移动到新的运行时目录。

### 7.4 删除全部 38 个不可达 Track TOML

当前 `conf.py` 没有任何 Track StageConfig，以下文件不能被活动 loader 选择：

```text
train_env_conf_track_nav.toml
train_env_conf_track_navgate.toml
train_env_conf_track_navj1.toml
train_env_conf_track_navj2.toml
train_env_conf_track_navj3.toml
train_env_conf_track_navj4.toml
train_env_conf_track_navj8.toml
train_env_conf_track_navj9.toml
train_env_conf_track_navnogate.toml
train_env_conf_track_navopt5b.toml
train_env_conf_track_navopt5debug.toml
train_env_conf_track_navslope.toml
train_env_conf_track_navspeed.toml
train_env_conf_track_navstable_a.toml
train_env_conf_track_navstable_hard.toml
train_env_conf_track_navstable.toml
train_env_conf_track_navx7bridgea.toml
train_env_conf_track_navx7bridgeb.toml
train_env_conf_track_navx7bridgec.toml
train_env_conf_track_navx7bridged.toml
train_env_conf_track_navx7bridgee.toml
train_env_conf_track_navx7bridgef.toml
train_env_conf_track_navx7bridgeg.toml
train_env_conf_track_navx7bridgeh.toml
train_env_conf_track_navx7nav1.toml
train_env_conf_track_navx7nogate.toml
train_env_conf_track_navx7score1.toml
train_env_conf_track_navx7train1.toml
train_env_conf_track_navx8align.toml
train_env_conf_track_navx8d1.toml
train_env_conf_track_navx8d2.toml
train_env_conf_track_navx8d2r.toml
train_env_conf_track_navx8d4.toml
train_env_conf_track_p22_lbc_loco.toml
train_env_conf_track_p22_ref_distill.toml
train_env_conf_track_track_lbc_loco_d2.toml
train_env_conf_track_track_lbc_loco.toml
train_env_conf_track_track_ref_distill.toml
```

删除必须显式列出文件，不能执行未来可能误伤新入口的宽泛 `rm *track*`。

### 7.5 暂不批量删除

本轮暂不批量删除：

```text
server/docs/*.md
server/agent_ppo/feature/ 中尚未完成死代码证明的 Track helper
server/agent_ppo/tests/test_hard_start_replay.py
server/agent_ppo/tests/test_j9_fixed_lr.py
server/agent_ppo/tests/test_st9_opt3_d2.py
```

这些文件虽然不是当前入口，但仍包含尚未汇总的评估证据或通用接口测试。后续若继续
清理，应先把唯一事实汇总到一个 `shared/分析记录/历史实验索引.md`，再单独删除，不能
与 Anchor R2 训练改造一起扩大范围。

### 7.6 清理后的硬检查

清理完成后：

```bash
find server/agent_ppo/conf -maxdepth 1 -type f -name '*.toml' -print | sort
```

必须只输出第 7.1 节的四个活动 TOML和一个待审计保留 TOML。随后执行：

```bash
rg -n 'train_env_conf_track_|standard_distill_[123]|standard_visual_ppo_[12]|hjcnew10288|stair_inv_finetune' \
  server/agent_ppo server/tests --glob '*.py'
```

活动 Python 代码和活动测试中不得再有命中。README、Changelog、`server/docs` 和 shared
历史说明若仍提及，必须明确标注“已从活动树删除，
通过 Git 历史恢复”，不能给出可直接运行的本地路径。

同步客户端 dry-run 必须显示删除候选；实际同步前人工核对这些候选全部属于本节名单。
不得启用对 `archive/` 或未知容器文件的远程删除。

只有 dry-run 返回的远程删除候选集合与第 7.2--7.4 节的显式名单完全相等时，才允许：

```bash
python3 local_sync_client.py --skip-unchanged --delete
```

多一个或少一个候选都应停止，不得为了让命令通过而扩大删除范围。

## 8. 测试修改

主测试文件：

```text
server/tests/test_visual_policy_optimization.py
server/tests/test_vision_distill_smoke.py
```

### 8.1 配置测试

替换 R3 断言，验证：

- 任务名、schedule mode、父 ID；
- 256 环境、25 秒 episode；
- `curriculum=true`、墙开启、`20/20/20/40/0`；
- 原生 command `[6,10]` 和三轴源域；
- 两种自定义 command scheduler 均 disabled；
- schedule knot、phase、LR 和保存边界；
- 所有随机化关闭；
- 活动 TOML 的完整 `[rewards.*]` 块与实施前快照一致；
- `reward_process.py` 与实施前快照一致，既不新增 gating，也不删除已有 gating。

删除当前测试中对 `base_env.py` 的
`reset_root_state_uniform_level_sampled` 源码断言，改为断言 Anchor R2 活动路径不依赖
该符号。

### 8.2 算法状态测试

至少覆盖：

1. 0/45/67.5/90/135/180/240 分钟的权重插值和左闭右开 phase；
2. 每个 phase 的 `requires_grad`；
3. 每个 optimizer group 的 LR；
4. optimizer 参数 ID 与 CNN/S0 参数 ID 不相交；
5. 性能阈值只产生 warning/诊断，不终止任务；
6. NaN/Inf 仍失败。

另加一项：构造 `visual_anchor_anneal_v2` 时必须实际采用配置 schedule；未知 mode 或
缺少 knot 必须失败，不能返回 legacy `rl*` phase。

### 8.3 checkpoint 测试

至少覆盖：

- 四种文件名全部通过平台探活正则；
- 初始 28401 精确优先；
- 同 mode round-trip 恢复 Actor/LSTM/Critic/std/optimizer/RNG/clock；
- 精确 S0 首训忽略 bundle RNG，使用 `[env_conf].seed`，clock=0；
- schedule 不同只加载 VisionEncoder/Actor，忽略保存的 RNG并使用 `[env_conf].seed`，
  clock=0，Critic/optimizer 不继承；
- `anchor_session_elapsed_hours` 优先、旧 `anchor_elapsed_hours` fallback；
- live hidden 不恢复；
- Camera eval 只加载视觉 Encoder 和 Actor，不加载 Critic/S0。

候选测试必须从临时目录真实创建四类 Anchor R2 文件，分别验证显式 ID、`latest` 和
Camera eval 都能发现，并验证首次 `28401` 仍优先 `visionfull-28401`。显式 ID 测试必须
同时放入更高优先标签但不同数字 ID 的文件，证明不会跨 ID fallback；`latest` 测试必须
断言 selector、文件名 ID 和 bundle ID 三个日志字段彼此独立。

保存测试必须覆盖 `delta <= 60.0s`（含边界）与 `delta > 60.0s` 的全部四种情况，断言
保存次数：

| periodic 与 boundary 间隔 | 期望 `save_model()` 调用次数 |
|---|---:|
| `0s`（同一 loop） | 1 |
| `30s` | 1 |
| `60s`（边界，`delta == 60.0`） | 1 |
| `60.001s`（刚过边界） | 2 |

boundary 去重后下一 periodic deadline 应为 boundary 实际保存时刻加 10 分钟。该去重规则
只在单次进程生命周期内有效，断点恢复后 2 分钟首存是刻意设计，测试不得把它判定为
"重复保存缺陷"。

### 8.4 平台 worker 边界测试

至少覆盖：

- 两个 command override 开关均为 `false`；
- 活动源码不 import `progressive_command_scheduler`、`worker_runtime_bridge` 或
  `isaac_env.base_env`；
- `base_env.py` SHA256 与操作者提供的平台版本一致；
- `_infer_stage_for_eval` 兼容入口存在；
- 深度预处理解析器从活动 TOML 读到 `augmentation.enabled=false`；
- 启动日志包含原生 command 与深度增强配置，不对 command 做运行时改写。

### 8.5 本地命令

不依赖本机安装 pytest；优先使用标准库 unittest：

```bash
cd server
python3 -m compileall -q agent_ppo
python3 -m unittest discover -s tests -p 'test_visual_policy_optimization.py'
python3 -m unittest discover -s tests -p 'test_vision_distill_smoke.py'
python3 - <<'PY'
import pathlib, tomllib
for path in pathlib.Path('agent_ppo/conf').glob('*.toml'):
    with path.open('rb') as stream:
        tomllib.load(stream)
    print(path)
PY
```

真实 torch tensor、optimizer round-trip 和 recurrent rollout 测试必须在平台同镜像或
具备 torch 的环境执行。本地 skip 不能描述成 tensor 测试通过。

## 9. 文档同步

同一 PR 更新：

```text
server/README.md
server/CHANGELOG.md
shared/interfaces/server-deploy-contract.md
shared/README.md
shared/分析记录/2026-07-25_Standard视觉学生双四小时训练计划.md
```

要求：

- 将 R3 八小时入口标为历史；
- 当前活动入口只写 `standard-anchor-r2`；
- 契约登记新 schedule、clock 和 phase 标签；
- 明确训练包仍 `deployable=false`；
- 记录清理范围明细：**38 个 Track TOML + 5 个已删除的旧 Standard TOML（§7.2）+ 1 个
  hjcnew10288 TOML（§7.3）= 合计 44 个 TOML**；`standard_stair_inv_finetune.toml` 不计入
  清理（保留审计）；
- 不把历史文件删除描述为删除 Git 历史；
- 不修改 deploy 目录或声称部署制品已经生成。

## 10. 平台短启动与四小时放行

本实施卡包含两个不同交付状态：完成代码、配置、文档和静态测试后，可以声明
“静态实现完成”；只有再完成本节 3--10 分钟平台短启动，才能声明“可以开始四小时
长训”。短启动是放行验证，不是正式训练的一部分。

### 10.1 同步前

```bash
cd server
python3 local_sync_client.py --check-local
python3 local_sync_client.py --dry-run --skip-unchanged
```

核对：

- 上传清单包含本任务修改文件；
- 远程删除清单只包含第 7 节明确删除项；
- `isaac_env/base_env.py` 不再是任务功能依赖；
- 没有 archive、checkpoint、Cookie、token、缓存或日志。

### 10.2 短启动

先运行 3--10 分钟，不直接消耗四小时。确认：

- 精确加载 28401；
- command 使用原生 S0 域；
- phase 为 `anchorcritic`，只有 Critic 更新；
- loss、梯度和权重有限；
- 首个 `anchorcritic` checkpoint 保存并能重新加载；
- 平台模型列表能探活该文件。

短启动成功后再创建正式四小时任务。短启动产物只作加载/保存验证，不作为训练父模型；
正式任务仍从 28401 重新开始。

## 11. 四小时评估与退出标准

优先评估 180、210、230 分钟附近候选，不默认选最后一个。固定 seed 明确为
`0, 1, 2, 3`；`28401` 基线与每个 Anchor R2 候选必须使用完全相同的任务配置和这四个
seed，不能用不同 seed 的均值互相比对。

通过条件：

```text
总体得分 >= 28401 的 95%
上楼总分相对 28401 下降 <= 3 分
上楼前进距离相对 28401 下降 <= 3 分
下楼总分相对 28401 下降 <= 3 分
坡面总分相对 28401 下降 <= 3 分
hard termination 相对 28401 增加 <= 1 个百分点
无静止策略、动作爆炸、视觉输入失效或持续拒绝登阶
```

每个候选都必须下载固定 seed 上下楼视频。训练 loss、anchor MSE 或最后一个 checkpoint
不能替代固定评估。

通过后登记唯一的 `standard-command-r1` 父 checkpoint、平台 ID、文件名、SHA256、
评估配置和结果。未通过则继续冻结 `visionfull-28401`，不得启动第二个四小时。

## 12. 完成定义

### 12.1 静态实现完成

以下条件全部满足时，执行 Agent 可以宣称“Anchor R2 静态实现完成”，但还不能放行四
小时长训：

- 活动 TOML 与本卡完全一致；
- schedule、模块冻结、LR、保存、恢复和保存去重测试通过；
- command override 已关闭，timeout/truncation 继续由平台 worker 管理；
- 活动代码不依赖平台 `base_env.py` 修改；
- 历史配置清理与训练功能分开提交，已证明清理项的活动入口引用扫描通过；
- README、Changelog、checkpoint 契约和执行卡一致；
- 未把任何本地静态测试 skip 描述成平台 tensor 验证通过。

### 12.2 四小时长训放行

只有同时满足以下条件，执行 Agent 才能宣称“Anchor R2 可以开始四小时长训”：

- 第 12.1 节静态实现条件全部满足；
- 平台短启动成功加载 28401、完成一次 Critic update、保存并恢复
  `anchorcritic` checkpoint；
- 短启动实际日志证明 native S0 command、phase 和 trainable modules 与
  第 6 节契约一致；
- 正式任务明确重新从 `visionfull-28401` 启动，短启动 checkpoint 未被选为父模型。
