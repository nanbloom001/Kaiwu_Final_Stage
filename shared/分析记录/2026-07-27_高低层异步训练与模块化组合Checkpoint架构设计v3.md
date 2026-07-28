# 高低层异步训练与模块化组合 Checkpoint 架构设计 v3

- 日期：2026-07-27
- 分支：`codex/hier-nav-dagger`
- 性质：下一阶段目标架构与实施契约（本文档不代表代码已实现）
- 前置文档：`2026-07-26_下一阶段高低层分层架构设计.md`
- MVP 收敛版：`2026-07-26_高低层架构设计v2_MVP与实施契约.md`
- 决策背景：低层先达到“基本可控”，高层可先训练；之后高低层允许交替或并行迭代，不因某一个低层 checkpoint 或固定指令词表被结构性绑定。
- 本次修订：在高层内加入具有固定物理语义的 `ResponseAdapter`；先用 `standard-com_34728` 在 Standard 模式做连续指令扩展，并用同一批轨迹同步预热低层与 ResponseAdapter，但两套损失、优化器和梯度路径严格隔离。

## 0. 一句话结论

**采用“两套独立主网络、一个高层内置 ResponseAdapter、一个模块化组合 checkpoint、声明式单/复合训练 scope、部署时独立导出两个 ONNX”的架构。**

- 低层网络负责 `depth + proprio + exec_cmd3 -> joint12`。
- 高层网络负责 `depth + goal + command-response 历史 + capability -> target_cmd3`。
- 高低层不共享可变的中间特征；高层不再直接依赖低层 CNN 的 `cnn_feat32` 语义。
- 高层输出长期固定为连续 `cmd3=[vx,vy,wz]`，低层能力扩展通过 capability profile 和训练适配解锁，不改高层输出维度。
- 一个 checkpoint 同时保存高低层权重、各自的 optimizer/scheduler/RNG/训练计数/血缘。
- Standard 低层训练可只实例化低层；高层模块作为冻结透明载荷保留，不进 GPU、不进 optimizer，保存时摘要必须逐位不变。
- `standard-com_34728` 的下一阶段采用 `low_level_and_response_adapter` 复合 scope：同一批 Standard rollout 同时更新低层与 ResponseAdapter，但禁止任何一方损失跨模块反传。
- 高层训练必须实例化低层作为冻结执行器，但只恢复和更新高层训练状态。
- 同一组合包可分别导出 `loco.onnx` 与 `nav.onnx`；经兼容性门禁后允许只更新其中一层。

## 1. 目标、语义与非目标

### 1.1 必须满足的目标

1. 平台每个训练任务仍只使用一个 `preload_model_id` 和一个 checkpoint 文件。
2. 这个文件可包含完整高低层模块；训练任务必须声明精确的单模块或复合 scope，且每个 optimizer 的参数集合必须与其声明完全一致。
3. 高层、低层和 ResponseAdapter 各自保存可独立 resume 的训练状态，不共用一个 `current_iteration` 或一套 optimizer。
4. Standard 模式可从完整组合包中只加载低层运行模块和低层训练状态，继续低层 RL。
5. 高层可在新低层上先做零样本评估，再选择只训 response adapter 或完整高层。
6. 低层的指令范围、地形能力和网络内部可继续扩展，不强制高层改输出 head。
7. 真机发布时可从同一包独立导出高低层 ONNX，同时防止未经验证的错误混搭。
8. 高层要通过低层的实际速度、时延和响应历史适应低层版本与 sim2real 差异，不只记住一个模型 ID。

### 1.2 “解耦”的准确含义

解耦不等于高层对低层行为完全无感。高层仍然必须在低层构成的闭环环境中训练。本文对解耦的定义是：

- 低层内部可以改网络权重、训练奖励、地形课程和 command 支撑范围。
- 只要稳定外部契约不变，高层网络结构、ONNX 输入输出维度和 command 语义就不改。
- 低层行为变化后，先许高层依靠 response adapter 零样本适应；不足时只做短适配训练，而不是必然从随机高层重训。
- 低层破坏稳定接口时，明确升级 major contract，拒绝把破坏性变化伪装成兼容更新。

### 1.3 首版非目标

- 不在真机上开放无限制在线梯度更新。首版仅允许 recurrent hidden 的快速适应，权重适配由真机日志离线训练完成。
- 不承诺高层能拯救不可控的低层。低层必须先通过最低可控门。
- 不再把规则 Oracle 动作标签作为最终高层训练目标。Oracle/DAgger 可保留为 smoke 基线和安全引导，最终高层使用闭环 RL。

## 2. 总体结构

```text
                                       高层 5Hz
 depth ------------------------> NavigationEncoder ---- nav_feat32 ------------------+
 goal / cmd / velocity / IMU / capability ----------------> nav_nonvisual36 ---------|
                                                                                     v
 response_observation32 --> ResponseAdapter(GRU64) --> response_profile16 --> HighLevelPolicy
                                                                                     |
                                                                                     v
                                                                continuous target_cmd3
                                                                              |
                                               CommandMapper + safety + slew   |
                                                                              v
                                                   exec_cmd3 (50Hz) -----------+
                                                                              |
                                       低层 50Hz                               v
 depth --> LocomotionEncoder(depth, proprio45[6:9]=exec_cmd) --> latent32 --> Actor77
                                                                              |
                                                                              v
                                                                          joint12
                                                                              |
                                                                              v
                                                    robot/env --> measured_velocity3
                                                                    + valid/age
                                                                    + IMU feedback
```

高层和低层各自拥有独立的视觉编码器：

- `LocomotionEncoder` 为低层步态服务，可随低层训练升级。
- `NavigationEncoder` 为高层导航服务，可从现有 CNN 权重初始化，但不再引用低层运行时中间特征。
- 两者可以初始权重相同，但 digest、optimizer、lineage 必须独立。

## 3. 网络结构 v3

### 3.1 低层网络

首版保持 Standard Actor77 外部契约：

```text
depth[180,320,1]
  -> locomotion CNN -> cnn_feat32

proprio45
  = ang_vel3
  + projected_gravity3
  + exec_cmd3
  + joint_pos12
  + joint_vel12
  + last_action12

[cnn_feat32 | proprio45]
  -> LSTM(2 layers, hidden=64)
  -> Linear(64,32)
  -> L2Norm
  -> latent32

[proprio45 | latent32]
  -> Actor MLP(77 -> 512 -> 256 -> 128 -> 12)
  -> joint_action12
```

低层内部未来允许变更，但必须保持对高层可见的稳定语义：

- 输入速度指令仍为机体坐标系 `exec_cmd3=[vx,vy,wz]`。
- 低层控制频率默认 50Hz。
- 输出仍为 12 维原始关节动作，关节顺序、scale、offset、clip 由部署契约固化。
- 低层内部 latent 维度可只在 major contract 升级时修改；高层不使用该 latent，因此不会因 latent 语义漂移直接失效。

### 3.2 高层视觉编码器

首版使用与现有 `simple_cnn` 同形的独立副本：

```text
depth[180,320,1] -> NavigationEncoder CNN -> nav_feat32
```

初始化顺序：

1. 首选复制已验证的视觉 CNN 权重。
2. 独立注册 `modules.high_level.navigation_encoder`。
3. 高层 warm-up 可先冻结该 CNN，闭环 RL 稳定后再以小学习率解冻。
4. 任何时候都不再从 `modules.low_level` 取中间 feature 作为高层输入。

### 3.3 低层响应适配器

ResponseAdapter 是**高层网络内部、低层外部**的系统辨识层。它不读取低层 CNN/LSTM latent，只根据实机可获得的 command-response 历史预测低层未来怎样响应。

每个高层 tick 组装固定的 `response_observation32`：

| 字段 | 维度 | 语义 |
|---|---:|---|
| `previous_target_cmd3` | 3 | 上一次高层目标指令 |
| `exec_cmd3` | 3 | 当前实际写入低层的 slew 后指令 |
| `measured_velocity3` | 3 | 机体系 `[vx,vy,wz]` 实测反馈 |
| `velocity_valid` | 1 | 速度测量是否有效 |
| `velocity_age` | 1 | 距最新有效测量的秒数，归一化后输入 |
| `ang_vel3` | 3 | IMU 机体角速度 |
| `projected_gravity3` | 3 | IMU 姿态重力向量 |
| `capability_profile15` | 15 | 当前低层有效指令域、活动维度与 slew 能力 |
| **合计** | **32** | 固定部署接口 |

图内再计算 `tracking_error3=exec_cmd3-measured_velocity3`，因此时序骨干实际接收 35 维物理量。首版骨干采用小型 MLP + GRU，不采用 CNN；这些数据是一维时序信号，没有需要卷积提取的空间结构：

```text
response_observation32
  -> compute tracking_error3
  -> concat35
  -> LayerNorm
  -> Linear(35,64) + SiLU
  -> GRU(input=64, hidden=64, layers=1)
  -> LayerNorm
  -> multi-head physical prediction
  -> response_profile16
```

`response_profile16` 不使用可任意旋转的无约束 latent，而使用固定物理语义：

| 输出 | 维度 | 语义 |
|---|---:|---|
| `pred_velocity_0p2s` | 3 | 维持当前执行条件时 0.2 秒后的机体速度 |
| `pred_velocity_0p6s` | 3 | 0.6 秒后的机体速度 |
| `pred_velocity_1p0s` | 3 | 1.0 秒后的机体速度 |
| `pred_pose_delta_1p0s` | 3 | 未来 1 秒机体系 `[dx,dy,dyaw]` |
| `stuck_probability` | 1 | 指令存在但速度/位移响应不足的概率 |
| `velocity_log_sigma_1p0s` | 3 | 1 秒速度预测的不确定度 |
| **合计** | **16** | 固定高层语义接口 |

固定物理语义是允许 Adapter 独立更新的关键：若输出任意 `response_z32`，Adapter 权重改变后 latent 坐标系也可整体旋转，即使预测能力变好，也会让尚未同步训练的高层 Actor 输入语义漂移。

适配器的职责是用最近 1-2 秒的 command-response 历史显式估计：

- 低层版本的加速、制动和转向增益。
- 控制延迟与指令死区。
- 地面打滑和真机执行器衰减。
- 速度测量来源变化、过期和丢失。

部署时不更新网络权重；只通过 GRU hidden 在数秒内适应。

训练 Adapter 时可使用仿真真值速度、位姿和终止/接触信息构造监督标签，但这些只进入 loss，不得进入 Adapter 输入或高层 Actor 输入。

### 3.4 capability profile

高层不输入无物理含义的 model ID，而输入 15 维可解释能力向量：

```text
active_mask3
cmd_min3
cmd_max3
slew_up3
slew_down3
= capability_profile15
```

非矩形组合约束不塞进固定向量，由结构化 `command_envelope` 描述：

```text
envelope_type = box_v1 | piecewise_union_v1
parameters = {...}
```

- `box_v1` 是长期目标：低层在稠密连续 cmd3 长方体或经明确裁剪的连续域上训练。
- `piecewise_union_v1` 只用于迁移现有 34728 这类非矩形指令支撑域。
- 34728 首次迁移可设 `active_mask=[1,0,1]`，固定 `vy=0`；未来低层通过验收后再打开 `vy`。

### 3.5 高层策略网络

高层 policy 输入固定为 84 维：

```text
nav_feat32
+ nav_nonvisual36
+ response_profile16
= 84
```

网络：

```text
input84
  -> NavigationMemory LSTM(input=84, hidden=64, layers=2)
  -> actor_mean3
  -> actor_log_std3
  -> tanh-squashed Gaussian
  -> CommandMapper
  -> physical target_cmd3=[vx,vy,wz]
```

约束：

- PPO 必须使用 squashed distribution 的正确 log-prob 修正，禁止“高斯采样后硬 clamp，仍用原高斯 log-prob”。
- `active_mask=0` 的维度不进入当前阶段的 log-prob/entropy，执行值固定为合同中性值（当前为 0）。
- 对 `piecewise_union_v1` 不使用非可逆硬投影冒充正确 policy density。迁移期优先限制活动维度和使用已验证的连续子域；长期通过重训低层建立连续支撑域。

### 3.6 高层 critic

高层使用 asymmetric privileged critic，允许额外看到：

- 真值位置与速度。
- height scan、接触力、地形类型。
- 仿真动力学随机化参数。
- 本 episode 选用的低层版本和 capability profile。

Actor 禁止直接看到这些真机不可得信息。本文不再使用 Oracle 动作标签限制 Actor 最终策略。

## 4. 稳定运行时接口

### 4.1 高层 -> 低层 command 接口

| 字段 | 定义 |
|---|---|
| 语义 | `target_cmd3=[vx,vy,wz]` |
| 单位 | `m/s, m/s, rad/s` |
| 坐标系 | 机体 body frame，`+x` 前、`+y` 左、`+wz` 左转 |
| 高层频率 | 5Hz（默认每 10 个低层帧决策一次） |
| 低层频率 | 50Hz |
| 执行链 | `target_cmd -> finite check -> capability envelope -> slew -> exec_cmd` |
| 注入位置 | `proprio45[6:9] = exec_cmd3` |
| 延迟语义 | 当帧低层先发布 joint；本 nav tick 新目标从下一个 50Hz 帧生效 |
| 急停 | 安全状态机可绕过普通 slew 直接写 zero；不依赖 policy 自行学会 |

### 4.2 低层/机器人 -> 高层 response 接口

```text
measured_velocity3 = [body_vx, body_vy, body_wz]
velocity_valid1
velocity_age1
feedback_source1
```

`feedback_source` 首版只进诊断和日志，不进 Actor 的 `nav_nonvisual36`。Actor 通过统一语义的速度值、`valid` 和 `age` 处理来源切换，避免对特定真机传感器编号过拟合。

真机来源顺序为：

1. SportMode 速度与 yaw speed，在 freshness 时限内使用。
2. UWB 滤波机体平面速度 + IMU yaw rate。
3. 两者都无效时 `velocity_valid=0`，数值置中性值，不允许 Actor 把无效零值当成真实静止。

训练侧禁止 Actor 使用仿真完美 base velocity，却在部署时切换为有噪声、丢帧和延迟的 Sport/UWB 链。训练必须通过同语义测量适配器产生反馈。

ResponseAdapter 首版所有输入在实机均可获得：

- `previous_target_cmd3`、`exec_cmd3` 来自控制软件自身状态。
- `ang_vel3`、`projected_gravity3` 来自 IMU。
- `measured_velocity3/valid/age` 来自现有 SportMode/UWB+IMU 反馈链。
- `capability_profile15` 来自 bundle manifest 与部署配置，不依赖额外传感器。

其中 `vx/vy` 是状态估计值，不是 IMU 直接测量值。部署验收必须确认 SportMode 速度坐标系并统一转换为 body frame，同时把 UWB 的噪声、滤波延迟、丢帧和 freshness 语义复现在训练测量链中。仿真真值速度、位姿、接触和动力学参数只可用于标签或 privileged critic。

首版不把 `joint_pos12/joint_vel12/last_action12` 加进 Adapter；它们虽在实机可得，但会增强 Adapter 对具体低层步态实现的绑定。只有 32 维接口不足以区分打滑与步态相位时，才在兼容升级中增加可选的 gait-summary 分支。

### 4.3 训练侧 FeedbackEmulator 契约

仿真引擎能直接提供完美的机体速度、角速度、姿态和位姿，但这些不是实机测量语义。训练侧必须维护两条严格隔离的数据路径：

```text
simulation true state
  ├─ privileged branch
  │    -> ResponseAdapter future labels
  │    -> asymmetric critic / metrics
  │
  └─ measurement branch
       -> FeedbackEmulator
       -> measured_velocity3
          velocity_valid1
          velocity_age1
          ang_vel3
          projected_gravity3
          feedback_source1 (diagnostic only)
       -> ResponseAdapter / HighLevel Actor
```

仿真真值速度固定定义为：

```text
true_velocity3 = [
  robot.data.root_lin_vel_b.x,
  robot.data.root_lin_vel_b.y,
  robot.data.root_ang_vel_b.z
]
true_ang_vel3 = robot.data.root_ang_vel_b
true_projected_gravity3 = robot.data.projected_gravity_b
```

`FeedbackEmulator` 是每环境独立的有状态训练组件，不是可学习网络，也不导出到 ONNX。输入、内部状态和输出契约如下：

```text
inputs:
  true_velocity3
  true_ang_vel3
  true_projected_gravity3
  simulation_time
  episode_reset
  feedback_profile
  rng

per-env state:
  delayed_sample_queue
  filtered_velocity3
  last_valid_sample_time
  dropout/source state

outputs:
  measured_velocity3
  velocity_valid1
  velocity_age_raw_s1
  ang_vel3
  projected_gravity3
  feedback_source1
```

`feedback_profile` 至少描述：

- SportMode 路径的速度噪声、yaw-speed 噪声、发布频率、延迟和 freshness timeout。
- UWB 路径的平面速度噪声、低通时间常数、延迟、丢帧、hold 和 freshness timeout。
- 来源优先级、来源切换概率和完全失效窗口。
- IMU 三轴陀螺仪的噪声、偏置和更新频率。
- 姿态/四元数噪声以及由此计算 `projected_gravity3` 的一致方法。
- `age_clip_s`、归一化和无效值编码。

模型实际输入：

```text
velocity_age = min(max(velocity_age_raw_s, 0), age_clip_s) / age_clip_s
```

从未收到有效样本或当前反馈无效时：

```text
measured_velocity3 = [0,0,0]
velocity_valid = 0
velocity_age = 1
```

因此 Actor/Adapter 必须结合 `valid` 解读中性零值，不能把它理解成机器人真实静止。`feedback_source` 保留原始枚举用于日志和分桶，不进入首版 Actor。

硬约束：

1. `root_lin_vel_b/root_ang_vel_b/projected_gravity_b` 不得直接拼入 ResponseAdapter 或 HighLevel Actor 输入；必须经过 FeedbackEmulator 的速度/IMU 测量分支。
2. 仿真真值只允许出现在标签生成、privileged critic 和诊断指标的显式 allowlist。
3. Standard 预热和高层 RL 必须调用同一个 `FeedbackEmulator` 实现与同一字段归一化函数。
4. `feedback_profile` 和实现版本必须写入 feedback contract，其 digest 写入 checkpoint；exact resume 时不匹配必须硬停或显式 warm-start。
5. live delay queue、filter 和 freshness 状态不写 checkpoint；resume 后随 episode reset 清空。
6. 完美真值输入只允许作为离线 ablation 上界，不得作为可发布训练配置。

### 4.4 高层非视觉输入 v3

`nav_nonvisual[36]` 布局固定如下：

| slice | 字段 | 维度 |
|---|---|---:|
| `[0:4]` | goal4 | 4 |
| `[4:7]` | previous_target_cmd3 | 3 |
| `[7:10]` | exec_cmd3 | 3 |
| `[10:13]` | measured_velocity3 | 3 |
| `[13:14]` | velocity_valid | 1 |
| `[14:15]` | velocity_age | 1 |
| `[15:18]` | ang_vel3 | 3 |
| `[18:21]` | projected_gravity3 | 3 |
| `[21:36]` | capability_profile15 | 15 |

`tracking_error3` 在图内由 `exec_cmd3 - measured_velocity3` 计算，避免训练/部署重复实现产生差异。

同一份 `nav_nonvisual36` 同时供高层 Actor 使用，并在图内抽取其中除 `goal4` 外的字段组成 `response_observation32`。训练、导出和部署不得维护第二套字段顺序。

### 4.5 reset 契约

episode reset 必须同帧清零：

- loco LSTM `h/c`。
- navigation LSTM `h/c`。
- response adapter GRU hidden。
- `previous_target_cmd3`、`exec_cmd3`、slew 状态。
- 速度历史有效标记。
- FeedbackEmulator 的 delay/filter/source/freshness 状态。

所有 live hidden 和 per-env 控制状态均不写入 checkpoint；resume 后强制 reset 环境。

## 5. 训练设计

### 5.1 低层启动高层训练的最低可控门

低层不需要训到最优，但必须满足：

- `zero` 可稳定停车。
- `vx` 响应方向正确且大致单调。
- 左右 `wz` 都可控，无系统性单侧失稳。
- command-to-velocity 响应延迟有界。
- 在基础地形和安全指令域上的硬终止率低于预设门限。
- 至少存在一个可完成基础导航的连续 `[vx,wz]` 子域。

门限数值需由 34728 的固定 seed 评估补齐，未取得数据前不在本文伪造数值。

### 5.2 高层训练

最终路线为 5Hz recurrent continuous PPO：

```text
冻结低层执行器
+ 独立 NavigationEncoder
+ ResponseAdapter
+ HighLevel Actor/Critic
+ K=10 低层步的 semi-MDP reward 聚合
```

奖励主体：

```text
+ goal progress
+ goal success
+ heading/progress consistency
- elapsed time
- hard termination / collision
- unstable posture
- energy proxy
- command rate / excessive slew
- small command tracking error penalty
```

command tracking 误差只是辅助项，不得主导奖励；否则策略可能以持续输出 zero 取得最小跟踪误差。

高层 RL 与 Standard 预热复用完全相同的 command/response 闭环，唯一允许变化的是 `target_cmd3` 的生产者：

```text
HighLevel Actor target_cmd3
  -> finite check -> capability envelope -> slew -> exec_cmd3
  -> frozen low level -> simulated robot
  -> shared FeedbackEmulator
  -> measured_velocity3 / valid / age / IMU observation
  -> ResponseAdapter + HighLevel Actor
```

不得为高层 RL 单独实现第二套反馈张量或归一化。

### 5.3 低层版本与动力学随机化

高层不应只在一个低层 checkpoint 上训练。每个 episode 可从低层池中抽样一个并在该 episode 内固定：

```text
L0 = 34728 或其安全迁移版
L1 = 楼梯修复版
L2 = 转向扩域版
L3 = 横移解锁版
+ 低层训练过程中的代表快照
```

同时随机化：

- 摩擦、质量、重心、执行器增益与扭矩。
- command gain、死区、一阶滞后、执行延迟和控制周期抖动。
- 深度噪声、空洞、帧龄和相机外参小偏差。
- 速度反馈来源、噪声、延迟、失效与 freshness。
- UWB 噪声、低通延迟、丢帧与 hold。

平台只能预加载一个包，因此多低层权重池若需在同一任务中真实抽样，可作为可选训练载荷写入：

```text
training_assets.low_level_bank[]
```

该权重池只供高层训练，不进入部署 release。首版可先用当前低层 + 响应随机化，不把权重池作为 P0 阻塞。

### 5.4 `standard-com_34728` 指令扩展与 ResponseAdapter 同步预热

下一阶段不先让高层 Actor 接管控制。以 `standard-com_34728` 为父模型，在 Standard 模式下由系统 command sampler 只产生 `target_cmd3`，再经过与未来高层完全相同的 envelope/slew 得到 `exec_cmd3` 并交给低层执行；同一批 rollout 同时用于：

1. 用 Standard PPO 继续训练低层，使其覆盖更稠密、更连续的指令域。
2. 用真实 command-response 序列训练 ResponseAdapter，使其在高层 RL 开始前已能预测当前低层的响应。

此时 `previous_target_cmd3` 的来源就是 Standard 系统 command sampler 的上一条目标命令；接口语义与未来高层 Actor 输出完全相同，因此预热阶段不需要伪造一个高层网络。

```text
Standard command sampler target_cmd3
  -> finite check -> capability envelope -> slew -> exec_cmd3
  -> trainable low level -> simulated robot
  -> shared FeedbackEmulator
  -> measured_velocity3 / valid / age / IMU observation
  -> trainable ResponseAdapter
```

Standard 与高层 RL 的差异仅为：

| 项目 | Standard 联合预热 | 高层 RL |
|---|---|---|
| `target_cmd3` 生产者 | 系统 command sampler | HighLevel Actor |
| 低层参数 | PPO 更新 | 默认冻结 |
| Adapter 参数 | auxiliary loss 预热 | 冻结或小学习率适配 |
| command envelope/slew | 共用 | 共用 |
| FeedbackEmulator | 共用 | 共用 |
| 真值使用边界 | 标签/critic/metrics | 标签/critic/metrics |

任务配置使用显式复合 scope：

```text
checkpoint_mode = extend_low_with_response
train_scope = low_level_and_response_adapter
optimizer_restore = exact_resume | warm_start
preserve_inactive_modules = true
```

这里的“同步训练、同时更新梯度”定义为**同一任务、同一批环境轨迹、同一 outer iteration 内分别执行两个 optimizer step**，而不是把两个网络连成一条可互相反传的联合损失：

```text
low_level PPO loss
  -> optimizer.low_level
  -> modules.low_level.*

response auxiliary loss
  -> optimizer.response_adapter
  -> modules.high_level.response_adapter
     + modules.high_level.response_prediction_head

stop_gradient:
  low-level action / hidden / weights
  command sampler output
  privileged future labels
```

必须满足：

- 两个 optimizer 的参数集合精确、互斥；交集非空硬失败。
- 低层 PPO backward 后 Adapter 的梯度必须为零；Adapter backward 后低层梯度必须为零。
- Adapter loss 不改变低层策略，低层 PPO loss 也不借 Adapter 获取捷径。
- 高层 NavigationEncoder、Actor、Critic 若已存在则冻结透明保留；若尚不存在也不阻塞本阶段。
- 两套 optimizer、scheduler、AMP scaler、RNG、iteration 与 gradient-step 计数分别保存。

ResponseAdapter 监督目标从同一环境序列的未来帧构造：

```text
L_response =
  w_v * SmoothL1(pred_velocity_{0.2,0.6,1.0s}, future_velocity)
  + w_pose * SmoothL1(pred_pose_delta_1.0s, future_body_pose_delta)
  + w_stuck * BCE(stuck_probability, stuck_label)
  + w_nll * GaussianNLL(pred_velocity_1.0s, log_sigma)
```

- reset、终止后跨 episode 的未来帧、缺失标签和超出 horizon 的尾部全部 mask。
- 三个 future horizon 的语义都是“从当前时刻起保持当前 `exec_cmd3` 时的响应”。Standard sampler 必须插入足够的 command hold 校准窗口；若 horizon 内目标命令改变，则只保留改变前仍成立的短 horizon，其余监督项 mask，禁止让 Adapter 猜不可观测的未来随机命令。
- 训练输入必须经过部署语义的反馈生成器；仿真真值只用于监督标签。
- `stuck` 正负样本需重加权或分层采样，防止模型仅输出“永不卡住”。
- Adapter 使用最近 on-policy 序列；若使用 replay，样本必须记录 low-level module digest/iteration，禁止无标识混合多个低层行为版本。
- 低层仍快速变化时先降低 Adapter loss 权重；达到最低可控门后再 ramp 到正常值。
- 指令扩展阶段结束后，冻结最终低层并做一个短 Adapter 尾部校准，使其收敛到最终低层而不是训练中途版本的平均响应。

### 5.5 指令扩展课程

`standard-com_34728` 的扩展遵循“先稠密化已有安全域，再扩大边界，最后增加新自由度”：

1. 在已验证的 `[vx,wz]` 支撑内做稠密连续采样，补充小角速度、平滑 ramp、加减速、zero/短驻留和左右对称样本。
2. 在基础地形门禁保持通过后，逐级扩展转向速度、刹停响应和联合 `[vx,wz]` 区域。
3. `vy` 首阶段仍由 `active_mask=[1,0,1]` 锁为零；只有纯横移和恢复能力通过独立门禁后才解锁。
4. 解锁 `vy` 后先纯横移，再加入 `vx/vy/wz` 联合采样，最终收敛到可由 `box_v1` 或简单可微 envelope 描述的连续域。
5. 每次扩域都保留旧 command bucket 与楼梯/基础地形回归，禁止为新速度范围牺牲已有稳定性而不被发现。

具体指令上下限由 `standard-com_34728` 固定 seed 能力评估和安全门禁确定；本文只固定课程顺序，不在缺少评估证据时伪造最终范围。

### 5.6 Adapter 预热完成门

进入高层闭环 RL 前至少检查：

- 三个预测 horizon 的速度 MAE、1 秒位姿增量误差和不确定度校准。
- `stuck` 的 precision/recall/AUROC，而不只看类别不平衡下的 accuracy。
- 按 zero、直行、左右转、加速、减速、feedback source/age 分桶的误差。
- 新旧 command 区域覆盖率和低层本身的 PPO/稳定性指标。
- 联合训练期间低层与 Adapter 的跨模块梯度均为零。

这些门限需用首轮基线数据确定。Adapter 预热不要求完美预测，但必须显著优于“恒定零速度”“直接复制 exec_cmd”这两种朴素基线，才有资格作为高层输入。

## 6. 组合 checkpoint v2 schema

### 6.1 顶层结构

保留平台识别的 `format="kaiwu_train_v1"`，将 `schema_version` 升为 2。旧 schema 1 只读兼容，加载后转换为内存中的 v2 结构，新任务不再写回 schema 1。

```text
{
  "format": "kaiwu_train_v1",
  "schema_version": 2,
  "bundle_kind": "hierarchical_control_v3",
  "platform_model_id": "<platform injected>",

  "contracts": {...},
  "modules": {...},
  "optimizers": {...},
  "training_states": {...},
  "lineage": {...},
  "digests": {...},
  "capabilities": {...},
  "training_assets": {...},
  "release_eligibility": {...}
}
```

`checkpoint_io.is_kaiwu_train_bundle()` 当前只接受 `schema_version==1`，实施时必须改为显式支持 `{1,2}`，不得以改文件名或伪造顶层 format 绕过迁移。

旧 schema 1 低层包的迁移规则：

- `modules.vision_encoder` 迁移为 `modules.low_level.locomotion_encoder`。
- `modules.low_level.actor_state_dict` 迁移为 `modules.low_level.actor.state_dict`。
- 能识别的 low-level optimizer/training state 迁移到对应独立 section；不能识别时标记 `weights_only`，不伪造可 exact resume 状态。
- 旧包不含 high-level 是合法的 low-only v2 内存形态；`resume_low` 可继续低层，`bootstrap_high` 可之后补建高层。
- 首次保存 schema 2 前必须打印迁移报告：原 format/schema、识别到的模块、丢弃的旧字段、新 module digests 和 optimizer 恢复级别。

### 6.2 modules

```text
"modules": {
  "low_level": {
    "contract_version": "low_level_v2",
    "locomotion_encoder": {
      "class_name": "VisionEncoder",
      "spec": {...},
      "state_dict": {...}
    },
    "actor": {
      "class_name": "Actor77Sequential",
      "spec": {...},
      "state_dict": {...}
    },
    "critic": {
      "class_name": "...",
      "spec": {...},
      "state_dict": {...}
    },
    "action_distribution": {...},
    "capability_profile": {...}
  },

  "high_level": {
    "contract_version": "high_level_continuous_v1",
    "component_status": "adapter_only|complete",
    "navigation_encoder": {
      "class_name": "NavigationEncoder",
      "spec": {...},
      "state_dict": {...}
    },
    "response_adapter": {
      "class_name": "CommandResponseAdapter",
      "spec": {...},
      "state_dict": {...}
    },
    "response_prediction_head": {
      "class_name": "PhysicalResponsePredictionHead",
      "spec": {...},
      "state_dict": {...}
    },
    "actor": {
      "class_name": "ContinuousHighLevelPolicy",
      "spec": {...},
      "state_dict": {...}
    },
    "critic": {
      "class_name": "PrivilegedHighLevelCritic",
      "spec": {...},
      "state_dict": {...}
    }
  }
}
```

规则：

- 每个顶层模块组必须有 `contract_version`；组内每个可学习 leaf module 必须有 `class_name`、`state_dict`、`spec`。
- `component_status=adapter_only` 只允许出现在 P1.5 预热包中，此时 `response_adapter/head` 必须存在，而 NavigationEncoder/Actor/Critic 可以尚未创建；`bootstrap_high` 负责补齐并改为 `complete`。`resume_high/evaluate_full/export_high/export_release` 遇到非 `complete` 必须硬停。
- 不再使用顶层单一 `trainable=true/frozen=true` 当作运行真值。可训范围由当前任务配置决定，保存时将实际 `train_scope` 记录为证据。
- 低层 critic 和高层 critic 均可在部署导出时忽略。

### 6.3 独立 optimizers

```text
"optimizers": {
  "low_level": {
    "algorithm": "ppo",
    "state_dict": {...},
    "bound_module_digest": "...",
    "valid_for_companion_digest": null
  },
  "high_level": {
    "algorithm": "recurrent_continuous_ppo",
    "state_dict": {...},
    "bound_module_digest": "...",
    "valid_for_companion_digest": "<low-level digest or pool digest>"
  },
  "response_adapter": {
    "algorithm": "auxiliary_system_identification",
    "state_dict": {...},
    "bound_module_digest": "...",
    "trained_with_low_level_digest": "...",
    "scheduler_state": {...},
    "amp_scaler_state": {...}
  }
}
```

优化器恢复分三级：

- `exact_resume`：活动模块 digest、对端 companion digest/低层池 digest、schedule 均匹配，恢复 optimizer/scheduler/RNG。
- `warm_start`：只恢复权重，重建 optimizer/scheduler；适用于高层换到新低层后适配。
- `weights_only`：只作迁移初始化，所有训练状态清零。

高层在 `H0+L0 -> H0+L1` 之后虽然权重未变，其旧 optimizer 默认不作 `exact_resume`，因为 companion 低层行为已变；默认用 `warm_start`。

`low_level_and_response_adapter` 的 `exact_resume` 只有在低层 optimizer/state、Adapter optimizer/state 和 `compound_schedule_phase` 三者都完整且相互匹配时成立；缺少任一项都必须显式降级为 `warm_start`，不能只恢复其中一套 Adam 动量却继续联合 schedule。

### 6.4 独立 training_states

```text
"training_states": {
  "global": {
    "last_active_scope": "low_level|high_level|response_adapter|low_level_and_response_adapter|none",
    "bundle_revision": 17,
    "last_transition": "...",
    "compound_schedule_phase": null,
    "global_rng_state": {...}
  },

  "low_level": {
    "stage_type": "standard_visual_ppo",
    "schedule_mode": "...",
    "current_iteration": 0,
    "iteration_semantics": "completed_outer_iterations_v1",
    "total_env_steps": 0,
    "gradient_steps": 0,
    "scheduler_state": {...},
    "curriculum_state": {...},
    "rng_state": {...},
    "last_trained_at": "...",
    "trained_from_module_digest": "..."
  },

  "high_level": {
    "stage_type": "hier_nav_continuous_ppo",
    "schedule_mode": "...",
    "current_iteration": 0,
    "iteration_semantics": "completed_nav_rollouts_v1",
    "total_env_steps": 0,
    "total_nav_ticks": 0,
    "gradient_steps": 0,
    "scheduler_state": {...},
    "rng_state": {...},
    "trained_against_low_level_digests": ["..."],
    "trained_against_pool_digest": "...",
    "last_trained_at": "..."
  },

  "response_adapter": {
    "stage_type": "standard_command_response_pretrain",
    "current_iteration": 0,
    "total_env_steps": 0,
    "total_response_sequences": 0,
    "gradient_steps": 0,
    "auxiliary_loss_state": {...},
    "scheduler_state": {...},
    "rng_state": {...},
    "trained_against_low_level_digests": ["..."],
    "last_trained_at": "..."
  }
}
```

不允许再用一个顶层 `current_iteration` 同时表示高层、低层和 Adapter 训练进度。

### 6.5 lineage 与 digests

```text
"lineage": {
  "bundle_parent_sha256": "...",
  "low_level": {
    "source_bundle_sha256": "...",
    "source_platform_model_id": "...",
    "parent_module_digest": "..."
  },
  "high_level": {
    "source_bundle_sha256": "...",
    "source_platform_model_id": "...",
    "parent_module_digest": "..."
  },
  "composition": {
    "composed": false,
    "composer_version": null,
    "low_source_sha256": null,
    "high_source_sha256": null
  }
},

"digests": {
  "low_level_module": "sha256(...) ",
  "high_level_policy_module": "sha256(navigation_encoder+actor+critic) ",
  "response_adapter": "sha256(response_adapter+response_prediction_head) ",
  "high_level_release_module": "sha256(high_level_policy_module+response_adapter) ",
  "command_contract": "sha256(...) ",
  "feedback_contract": "sha256(...) ",
  "feedback_emulator_profile": "sha256(profile+normalization+implementation_version) "
}
```

摘要规则：

- 训高层时：低层 module digest 必须保持不变，漂移硬失败。
- 训低层时：`high_level_policy_module`、`response_adapter` 和对应训练状态 digest 必须保持不变，漂移硬失败。
- 训 response adapter 时：高层其他模块和低层均必须不变。
- 训 `low_level_and_response_adapter` 时：只允许 low-level digest、response-adapter/head digest 及各自训练状态改变；NavigationEncoder、HighLevel Actor/Critic 及其训练状态必须逐位不变。
- 当前代码低层 digest 漂移是 warning-only；v3 不允许这种行为用于冻结模块。

### 6.6 冻结透明载荷

Standard 低层单独训练时不需实例化高层网络：

```text
materialized modules on GPU:
  modules.low_level.*

opaque preserved payload on CPU:
  modules.high_level
  optimizers.high_level
  training_states.high_level
```

保存时：

1. 重新序列化更新后的低层。
2. 原样写回高层透明载荷。
3. 高层权重、optimizer 和 training state 的结构摘要必须与加载时一致。
4. 禁止因 Standard trainer 未构造高层类就删除 `modules.high_level`。

这是“Standard 只加载低层”的正式定义：只实例化低层，但完整包中的高层仍被保留。

`low_level_and_response_adapter` 是另一种明确模式：只实例化 `modules.low_level.*`、`response_adapter` 和具有物理输出语义的 prediction head；NavigationEncoder、HighLevel Actor/Critic 仍作为透明载荷。不得因为 Adapter 位于 `modules.high_level` 命名空间就把整个高层加载到 GPU。

## 7. 加载和训练状态机

### 7.1 配置字段

平台顶层仍只保留：

```text
preload_model = true
preload_model_id = <single platform model id>
preload_model_dir = <single directory>
```

业务侧新增：

```text
checkpoint_mode =
  bootstrap_high
  resume_low
  extend_low_with_response
  resume_high
  adapt_high
  evaluate_full
  export_low
  export_high
  export_release
  weights_only

train_scope = low_level | high_level | response_adapter | low_level_and_response_adapter | none
optimizer_restore = exact_resume | warm_start | reset
preserve_inactive_modules = true
```

加载分支不再依赖“注入 ID 是否等于 `low_level_parent_model_id`”的隐式判断。当前 `Agent._load_nav()` 的父包/恢复二选一必须替换为 mode-driven loader。

### 7.2 加载矩阵

| 模式 | GPU 实例化 | 恢复权重 | 恢复 optimizer/state | 透明保留 | 允许训练 |
|---|---|---|---|---|---|
| `bootstrap_high` | 高+低 | 低层 + 已预热 Adapter（若存在） | 低层只读；补建 NavigationEncoder/Actor/Critic | Adapter 训练状态保留 | 高层 |
| `resume_low` | 低 | 低层 | 低层 optimizer/state | 高层 section（若已存在） | 低层 |
| `extend_low_with_response` | 低层 + Adapter/head | 低层；Adapter 若存在则恢复、否则新建 | 低层与 Adapter 各自恢复或 warm-start | 高层其余 section | 低层 + Adapter，梯度隔离 |
| `resume_high` | 高+低 | 高+低 | 高层 optimizer/state；低层只读 | 低层训练状态 | 高层 |
| `adapt_high` | 高+低 | 高+低 | response adapter 重建或 warm-start | 其他训练状态 | adapter/head |
| `evaluate_full` | 高+低 | 高+低 | 不恢复 optimizer | 全部 | 无 |
| `export_low` | CPU 低层 | 低层 | 无 | 其他 section | 无 |
| `export_high` | CPU 高层 | 高层 | 无 | 其他 section | 无 |
| `export_release` | CPU 高+低 | 高+低 | 无 | 全部 | 无 |

### 7.3 加载硬门

1. 单一预加载文件的 format/schema/bundle_kind 必须匹配。
2. `checkpoint_mode` 要求的 module section 必须存在；例如 `resume_high` 缺 high-level 硬停。`resume_low` 允许低层独立包，`bootstrap_high` 允许 high-level 缺失并新建。`extend_low_with_response` 必须有低层，但允许从 `standard-com_34728` 的 low-only 包新建 Adapter/head。
3. `exact_resume` 所需 optimizer/scheduler/training state 缺失硬停；`warm_start` 可显式允许缺失。
4. 每个 state_dict 用 `strict=True` 校验 class/spec/key/shape/有限性。
5. 冻结模块的参数集合不得与 optimizer 参数集合相交。
6. optimizer 参数集合必须精确等于 `train_scope` 允许的参数集合。
7. 复合 scope 下必须分别检查两个 optimizer 的参数集合互斥，并在两次 backward 后断言对端模块没有梯度。
8. 训练前记录所有冻结 module digest，每次保存前重算并硬比较。
9. 不能因 checkpoint 内包含 high-level 就禁止 Standard 低层加载；Standard loader 必须理解 schema v2 的透明保留语义。

## 8. 异步迭代工作流

### 8.1 从 `standard-com_34728` 启动联合预热

```text
standard-com_34728 (schema 1 or low-only schema 2)
  -- migrate/bootstrap ResponseAdapter -->
L0 + A_random
  -- Standard command expansion, joint rollout / isolated optimizers -->
L1 + A1
  -- freeze L1, short Adapter tail calibration -->
L1 + A2
  -- bootstrap NavigationEncoder + HighLevel Actor/Critic -->
H0(A2) + L1
```

首次加载旧包时：

- 严格迁移并恢复 34728 低层权重；若旧 optimizer/state 可识别则按规则恢复，否则显式 `warm_start`。
- 新建 `response_adapter`、`response_prediction_head` 及其独立 optimizer/training state。
- 未存在的 NavigationEncoder、HighLevel Actor/Critic 不伪造权重；到 `bootstrap_high` 阶段再创建。
- 首次保存即输出 schema 2 组合包，记录父模型 ID、原包 SHA256、迁移报告及 low/adapter 独立 digest。

### 8.2 交替训练（平台单 preload 下的默认路线）

```text
L0 low-only bundle
  -- bootstrap_high --> H0 + L0

H0 + L0
  -- resume_low --> H0 + L1

H0 + L1
  -- evaluate_full --> zero-shot compatibility report
  -- adapt_high     --> H1 + L1   (if needed)

H1 + L1
  -- resume_low --> H1 + L2
  ...
```

每一次任务仍只使用一个预训练包，每一次输出仍是一个完整包。

### 8.3 Standard 只继续低层

输入：`H0+L0` 完整包。

```text
checkpoint_mode = resume_low
train_scope = low_level
optimizer_restore = exact_resume | warm_start
preserve_inactive_modules = true
```

Standard 训练进程：

- 只构造 locomotion encoder/actor/critic。
- 只恢复 low-level optimizer/scheduler/RNG/training state。
- 高层作为 CPU 透明载荷，不运行、不反序列化成网络对象。
- 输出 `H0+L1`，高层权重和训练状态不变。

### 8.4 高层在新低层上适配

先执行零样本评估：

```text
H0 + L1 -> fixed seeds / full track / response diagnostics
```

分级处理：

- 达标：直接发布或进入更广测试，不训练。
- 轻度回退：只训 response adapter + command head。
- 中度回退：训练完整高层，NavigationEncoder 先冻结再解冻。
- 接口破坏：拒绝适配，升级 contract major version。

### 8.5 真正并行训练

从同一基线包并行启动：

```text
Task A: H0 + L0 -> H1 + L0
Task B: H0 + L0 -> H0 + L1
```

平台单 preload 不能在下一任务中同时加载两个父包，但用户已确认模型包可重新导入平台，因此使用离线 composer 组成：

```text
H1 from Task A
+ L1 from Task B
-> compatibility validation
-> H1 + L1 composed bundle
-> import as a new platform model
```

## 9. bundle composer

### 9.1 输入与输出

```text
compose_hier_bundle \
  --high-from <bundle-A> \
  --low-from <bundle-B> \
  --out <bundle-H1-L1> \
  --optimizer-policy reset-incompatible
```

composer 只做可复现的结构化组合，不做字符串替换或 checkpoint 改名。

### 9.2 组合前硬校验

1. 高层 command contract major version 与低层 command contract major version 相同。
2. `cmd3` 单位、坐标系、控制频率和 `proprio[6:9]` 语义相同。
3. 低层 capability envelope 覆盖高层 release policy 声明的必需域，或高层明确支持该 capability profile。
4. 反馈字段、速度来源、valid/age 归一化相同。
5. 两个输入包的 module digest 与 lineage 可验证。
6. 合并后所有 state dict 有限，class/spec/key/shape 严格匹配。

### 9.3 optimizer 处理

- 低层 optimizer 可从 low source 保留。
- 高层 optimizer 只有在其 `valid_for_companion_digest` 包含新低层 digest 时才可保留。
- 默认将组合后的高层 optimizer 标记为 `warm_start_required`，防止用面向 L0 的 Adam 动量直接在 L1 上 exact resume。
- composer 必须写入双父包 SHA256、module digest、工具版本和组合时间。

## 10. 独立 ONNX 导出与部署

### 10.1 `loco.onnx`

从组合包中只读取：

```text
modules.low_level.locomotion_encoder
modules.low_level.actor
```

目标清洁接口：

```text
inputs:
  depth[1,180,320,1]
  proprio[1,45]
  loco_h[2,1,64]
  loco_c[2,1,64]

outputs:
  joint[1,12]
  loco_h_out[2,1,64]
  loco_c_out[2,1,64]
```

`exec_cmd3` 在图外写入 `proprio[6:9]`。首个迁移版若为了复用现有 C++ runner 保留历史 8 入 8 出占位端口，必须标记为 `loco_compat8_v1`，不得与目标清洁接口混同。

### 10.2 `nav.onnx`

从组合包中只读取：

```text
modules.high_level.navigation_encoder
modules.high_level.response_adapter
modules.high_level.response_prediction_head
modules.high_level.actor
```

接口：

```text
inputs:
  depth[1,180,320,1]
  nav_nonvisual[1,36]
  nav_h[2,1,64]
  nav_c[2,1,64]
  adapter_h[1,1,64]

outputs:
  target_cmd[1,3]
  nav_h_out[2,1,64]
  nav_c_out[2,1,64]
  adapter_h_out[1,1,64]
  response_profile[1,16]       # 固定物理语义诊断输出，可在 release profile 中关闭
```

图内计算 response tracking error 与 capability mapping，图外只负责 finite check、安全状态机、slew 和最终 `exec_cmd`。

为训练单测和离线回放可额外导出独立 Adapter 图，但它不是默认真机上的第三个网络：

```text
inputs:
  response_observation[1,32]
  adapter_h[1,1,64]

outputs:
  response_profile[1,16]
  adapter_h_out[1,1,64]
```

发布时 Adapter 与 prediction head 内嵌于 `nav.onnx`，因此仍保持“低层一个 ONNX、高层一个 ONNX”的部署结构。

### 10.3 部署时序

```text
every 50Hz frame:
  1. sample sensors and current velocity feedback
  2. write exec_cmd into proprio[6:9]
  3. run loco.onnx
  4. publish joint action on time
  5. if frame % 10 == 0:
       assemble nav_nonvisual36
       run nav.onnx in remaining frame budget
       update target_cmd
  6. step safety envelope and slew for next frame
```

高层超时不得阻塞当帧关节动作发布。超时时保持旧 target/exec；连续超时或输出非有限时进入降级状态机。

### 10.4 release manifest

```text
release/
  loco.onnx
  nav.onnx
  manifest.json
  deploy.yaml
  sha256sums.txt
```

`manifest.json` 至少包含：

```text
source_bundle_sha256
platform_model_id
low_level_module_digest
high_level_policy_module_digest
response_adapter_digest
high_level_release_module_digest
command_contract_version
feedback_contract_version
training_feedback_emulator_profile_digest
capability_profile
loco_onnx_sha256
nav_onnx_sha256
exporter_versions
compatibility_report_sha256
tested_pair = {high_digest, low_digest}
```

默认发布同一包导出的配对。只更新其中一个 ONNX 时，必须有新的 compatibility report 和 `tested_pair`，不允许手工替换文件而沿用旧 manifest。

## 11. 兼容性分级

### Level A：接口与行为域内兼容

- command/feedback contract 不变。
- 新低层 response 位于高层训练随机化或低层池覆盖内。
- 零样本评估达标。

处理：高层无需训练，可直接组合后进入发布门。

### Level B：接口兼容，行为变化或能力扩展

- 指令范围扩展、响应更快/更慢，或新增 `vy`。
- 高层结构和 ONNX I/O 不改。

处理：先训 response adapter/head，必要时完整高层 warm-start；新能力不会被旧高层自动利用，需要训练解锁。

### Level C：破坏性接口变化

- command 单位/坐标系改变。
- 反馈字段、归一化或频率改变。
- action12 顺序或控制语义改变。
- 高层必需输入不再真机可得。

处理：升级 major contract，旧高层组合硬拒绝，重新实施训练/部署原子更新。

## 12. 测试与验收门

### 12.1 checkpoint codec

- schema 1 -> schema 2 只读迁移。
- schema 2 full round-trip：所有 module/state/optimizer/lineage 不丢失。
- `resume_low` 后高层 section 的权重、optimizer、training state 摘要逐位不变。
- `resume_high` 后低层 module digest 逐位不变。
- `extend_low_with_response` 后只允许低层、Adapter/head 及其各自训练状态改变，高层其余摘要逐位不变。
- 任何冻结参数泄漏进 optimizer 必须使测试失败。
- 复合 scope 的两个 optimizer 参数集合必须互斥；分别 backward 后对端梯度必须为空或严格为零。
- optimizer exact/warm/weights-only 三条恢复语义分别测试。

### 12.2 composer

- 同契约 H1+L1 组合成功，血缘与双父 SHA 正确。
- command/feedback major version 不同硬拒绝。
- capability 不覆盖时硬拒绝或明确要求 high-level adaptation，不允许静默通过。
- 组合后高层 optimizer 按 companion digest 规则保留或失效。

### 12.3 平台 smoke

1. 单一导入的 `H0+L0` 包在 Standard 模式下 `resume_low`，完成一次真实 low-level update 和 checkpoint 发布。
2. 下载新包，验证高层摘要不变、低层摘要改变、两套训练计数独立。
3. 从 `standard-com_34728` 以 `extend_low_with_response` 跑一次最小真实 update，验证低层和 Adapter/head 摘要都改变、两套 optimizer/state 均可恢复、高层其余模块不变或仍合法缺失。
4. 将联合预热包 exact resume 一次，验证 low/adapter 的独立计数和联合 schedule phase 连续。
5. 将该包以 `resume_high` 重新导入，完成一次高层 rollout/update/save。
6. eval 入口严格加载同一包两层权重并输出 module digests。
7. composer 产物重新导入平台，走通 `evaluate_full`。

### 12.4 ONNX 与部署

- Python -> ONNX 多帧数值 parity，覆盖三套 recurrent state 回喂。
- 混频 golden-vector parity：50Hz loco、5Hz nav、一帧延迟、slew、reset、反馈过期。
- 分别单独导出 loco/nav，验证只读指定 module section。
- manifest 错配、ONNX SHA 不一致、contract 不一致必须在首次 inference 前硬停。
- Jetson 影子模式测试 nav/non-nav 帧 p50/p95/max、跳帧率、实际 Hz 和速度反馈有效率。

### 12.5 ResponseAdapter

- 对齐 0.2/0.6/1.0 秒 horizon，验证 command 变化、reset、timeout 和 episode 尾部 mask。
- 报告速度/位姿 MAE、Gaussian NLL/校准、`stuck` precision/recall/AUROC。
- 按 command bucket、地形、低层 digest、feedback source/valid/age 分桶，不能只给全局均值。
- 与“恒定零速度”“直接复制 exec_cmd”基线比较，防止获得看似较低但无用的平均误差。
- 使用部署语义输入跑离线回放；额外加入仿真真值输入若能大幅改善，只说明测量链仍是瓶颈，不能把真值输入带到 Actor。

### 12.6 FeedbackEmulator 与真值隔离

- 固定 seed 下验证噪声、延迟队列、低通、丢帧、来源切换和 freshness 的确定性。
- golden-vector 验证 `velocity_valid/age` 的归一化、首次无样本、过期、reset 和来源切换语义。
- schema/dataflow 测试禁止 `root_lin_vel_b/root_ang_vel_b/projected_gravity_b` 直接进入 Adapter/Actor，只允许进入 FeedbackEmulator 或明确的 label/critic/metric allowlist。
- 验证 Standard 与高层 RL 对同一 true-state/seed 序列产生逐元素一致的 feedback 张量。
- 分别记录仿真 true response 和 emulated measured response，报告误差、延迟、有效率和 age 分布。
- 与实机影子日志比较上述分布；不要求逐帧相同，但必须覆盖实机观察到的主要噪声、延迟和失效区域。
- perfect-feedback ablation 必须带不可发布标记，checkpoint/release gate 遇到该 profile 硬拒绝。

## 13. 实施阶段

### P0：契约与 codec

- 在 `shared/interfaces/server-deploy-contract.md` 增加 v3 command/feedback/checkpoint/ONNX 契约。
- 实现 schema 2 codec、模块摘要、独立 training states、透明载荷和加载矩阵。
- 先用假网络单测验证 `resume_low/resume_high`，不同时开始长训。

### P1：Standard 单包低层续训

- 让 Standard VisualPPO/command 路径能从 schema 2 完整包只实例化低层。
- 平台 smoke 证明高层透明载荷未被删除或改写。

### P1.5：34728 指令扩展与 ResponseAdapter 联合预热

- 从 `standard-com_34728` 迁移/新建 schema 2 包和 ResponseAdapter/head。
- 将 Standard sampler 的 `target_cmd3` 与 post-envelope/slew 的 `exec_cmd3` 显式拆分，并固化 previous-target 时序。
- 实现共享 `FeedbackEmulator`、feedback profile/digest、真值 allowlist 和部署同语义的 valid/age 归一化。
- 在 Standard 系统 command 驱动下扩展连续指令域，同一批 rollout 分别执行低层 PPO 与 Adapter auxiliary update。
- 实现 future-horizon 标签队列、reset/termination mask、部署语义反馈噪声链和 low-level digest 标记。
- 用参数集合、梯度、module digest 和独立训练状态测试证明两套更新同步但不耦合。
- 完成 Adapter 朴素基线对比与冻结低层后的尾部校准，再允许进入 P2。

### P2：连续高层网络与 RL

- 实现 NavigationEncoder、continuous recurrent actor/critic，并接入已预热的 ResponseAdapter。
- 复用 P1.5 的 command envelope/slew 与 FeedbackEmulator，禁止另建高层专用反馈链。
- 在平地直达小环境验证 recurrent PPO 收敛和 squashed log-prob，再接入楼梯/迷宫。

### P3：composer 与真并行训练

- 实现双父包组合、兼容性检查、optimizer 失效策略和平台重导入验证。

### P4：双导出器与影子部署

- 实现 `export_loco_from_hier_bundle` 与 `export_nav_from_hier_bundle`。
- 原子更新 `server/`、`deploy/`、共享接口契约。
- 影子模式验收 Jetson 时延和速度反馈质量，之后才允许接管。

### P5：多低层池与真机离线适配

- 加入 low-level checkpoint bank。
- 影子模式日志进入离线回放与高层适配训练。
- 保留旧低层版本回归，防止高层只适配最新版本并遗忘旧域。

## 14. 对现有 v2/MVP 的处置

现有 Nav DAgger 工作不丢弃，重新定位为：

- checkpoint、平台 lifecycle、高低层混频时序和 eval 入口的 MVP 验证基础。
- Oracle/DAgger 可作为快速 smoke、行为安全初始化和对照组。
- 不再作为最终高层动作空间与长期适配设计。

以下 v2 结论被 v3 修订：

| v2 | v3 |
|---|---|
| 高层分接低层 CNN raw32 | 高层独立 NavigationEncoder |
| 固定 10 token categorical | 固定连续 cmd3 输出 + capability profile |
| 换低层后高层从头重训 | 先零样本，再 adapter/head/full warm-start |
| 高层只看 exec/held cmd + IMU | 新增真实速度、valid/age 和 command-response 历史 |
| 无独立响应模型或任意 latent | 固定物理语义 `response_profile16`，可在 Standard 阶段先监督预热 |
| checkpoint 只保存高层 optimizer/state | 高低层各自保存完整训练状态 |
| Standard loader 视 high-level 为异类包 | Standard 可只实例化低层并透明保留高层 |
| 单任务只允许一个可训练模块 | 支持声明式复合 scope，但每套损失/optimizer/梯度严格隔离 |
| 单父包无法合并并行训练结果 | 离线 composer + 平台重导入 |

## 15. 当前代码事实与实施缺口

可复用事实：

- `algorithm_nav_dagger.py:913-983` 已能在一个 `kaiwu_train_v1` 包内保存 `vision_encoder + low_level + high_level`。
- `checkpoint_io.py:525-540` 已有低层 spec 校验。
- `checkpoint_io.py:765-799` 已有高层 contract 校验。
- `checkpoint_io.py:824-849` 已有低层 state-dict digest 先例。
- VisualPPO 已有“同 schedule 恢复 optimizer/state，异 schedule 只迁移权重”的可复用模式。
- `worker_command_bridge.py:120-146` 已能读写并校验 Standard 的三维 `base_velocity` command tensor。
- `reward_process.py:74` 已能访问仿真 `root_lin_vel_b` 真值；现有 proprio/nav 路径也能取得 `ang_vel3/projected_gravity3`。
- `deploy/sim2real_test_loco/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/src/State_VisionLoco.cpp:761-788` 已在诊断链计算 SportMode/UWB feedback 的 valid、age 和速度来源，可作为训练反馈语义的实机参照。

必须修改的阻断：

- `checkpoint_io.is_kaiwu_train_bundle()` 当前只接受 schema 1。
- `Agent._load_nav()` 当前以“ID 是否等于 low-level parent”决定首载或 resume，无显式模块加载模式。
- `load_parent_bundle()` 固定低层加载、高层随机；`load_nav_resume()` 固定恢复高层训练。
- `_load_low_level_from_bundle()` 加载后总是冻结低层，无 Standard `resume_low` 反向模式。
- Nav 包当前只保存 `optimizers.high_level_dagger` 和单一 Nav training state。
- Standard worker 当前把 schedule command 直接写入环境 command tensor，尚未形成稳定的 `target_cmd3 -> envelope/slew -> exec_cmd3` 显式接口。
- 当前训练代码没有 `FeedbackEmulator`，也没有组装 `measured_velocity3/velocity_valid/velocity_age`；仿真 root velocity 目前只能视为真值来源，不能直接替代部署语义反馈。
- 当前 Nav 输入只包含 exec/held command 与 IMU proprio，尚未接入 measured feedback；部署端 feedback 当前主要用于诊断日志，也尚未接入模型输入。
- 当前 `export_loco_onnx.py` 只接受 `format=lbc_loco`；它提示的 `export_vision_nav_onnx.py` 在当前仓库中尚不存在。

## 16. 最终决策

1. **平台单 checkpoint 不是高低层解耦的阻碍。** 文件是容器，模块、optimizer 和训练状态在容器内独立。
2. **ResponseAdapter 属于高层，但可早于高层 Actor 预热。** 输入全部采用真机可得的 command、速度估计、IMU 与 capability，输出固定物理语义 `response_profile16`。
3. **仿真真值和部署语义测量必须硬隔离。** 真值只进入 Adapter 标签、privileged critic 和指标；Adapter/Actor 只能读取共享 FeedbackEmulator 生成的 measured velocity、valid、age 和 IMU 观测。
4. **下一步从 `standard-com_34728` 做 Standard 指令扩展。** 系统 sampler 产生 target，公共 envelope/slew 产生 exec，低层学习执行，Adapter 用相同 rollout 学习 command-response。
5. **Standard 预热与高层 RL 共用同一闭环。** 唯一变化是 target 由系统 sampler 还是 HighLevel Actor 产生，后续 command 和 feedback 链不得分叉。
6. **“同时更新”不等于端到端互相反传。** 低层 PPO 与 Adapter auxiliary loss 使用独立且互斥的 optimizer；同步获取数据、同步保存状态，但梯度严格隔离。
7. **Standard 只加载低层仍是一等模式。** 当不需要预热 Adapter 时，高层作为冻结透明载荷保留，不必须实例化。
8. **高低层与 Adapter 必须各自拥有完整训练状态。** 不允许用一套 optimizer/current_iteration 模糊表示多个模块；复合任务还要保存联合 schedule phase。
9. **平台内交替训练直接使用上一个完整包。** 并行任务的结果通过 composer 组合后重新导入。
10. **部署使用两个独立 ONNX。** Adapter 内嵌 `nav.onnx`，默认同源配对，经兼容性门禁后允许单层升级。
11. **实施顺序为 checkpoint/loader -> Standard 联合预热 -> 新高层 RL。** 先证明包能恢复、两套梯度隔离且响应预测有效，再让高层 Actor 学习导航。
