# server ↔ deploy 接口契约

> 版本：`v0.3`
> 状态：已固化 checkpoint 类别与 Standard/Track 维度；深度、ONNX 和动作字段仍需随发布候选补齐
> 运行时边界：本文件只作协作契约，`server/` 与 `deploy/` 不得依赖 `shared/`

## 1. 两条模型合同不得混用

| 路线 | Actor 输入 | Goal | 当前用途 |
|---|---:|---:|---|
| Standard 低层视觉运控 | `proprio45 + latent32 = 77` | 0 | 本轮规划的低层最终合同 |
| Track 378413 | `proprio45 + latent32 + goal3 = 80` | 3 | 现有 `deploy/sim2real_test_loco` 历史部署路线 |

Standard 的 `goal0/Actor77` 与 Track 的 `goal3/Actor80` 是两份独立合同。任何 loader、
exporter 或 `deploy.yaml` 都必须先核对 task、format、goal_dim 和 Actor 第一层 shape，
不得仅凭 checkpoint 数字 ID 判断结构。

## 2. Checkpoint 类别

### 2.1 `kaiwu_train_v1`：统一训练包

用途：在训练阶段之间复用同一顶层 schema。R2 先保存 low-level；后续视觉蒸馏
增加 `modules.vision_encoder`，高低层混合增加 `modules.high_level`，不通过改名
伪装为部署制品。

必需字段（特权 DAgger 路径，R2）：

```text
format = "kaiwu_train_v1"
schema_version = 1
model_spec
modules.low_level.policy_state_dict
modules.low_level.encoder_state_dict
modules.low_level.actor_state_dict
modules.privileged_teacher.state_dict
optimizers.low_level_distill
training_state.current_iteration / total_steps / gradient_steps
training_state.dagger_phase_index / dagger_phase_iteration
training_state.student_drive_probability / safety_threshold
training_state.promotion_history / recent_iteration_metrics / rng_state
replay.capacity / size / seen / obs_fp16 / teacher_actions_fp16
phase_snapshots.entry / best / exit
lineage.parent_model_id / teacher_sha256 / config_sha256 / code_commit
capabilities.critic_trained = false
capabilities.deployable = false
```

视觉蒸馏路径（阶段 4）在上述 schema 上用 `modules.vision_encoder` +
`modules.low_level`（冻结教师副本）替代 `modules.privileged_teacher`，并改用
线性 ramp 调度字段：

```text
format = "kaiwu_train_v1"
schema_version = 1
ramp_label                         # visionteacher / visionhalf / visionfull / visionblocked
model_spec
modules.vision_encoder.state_dict  # 视觉学生（本轮训练对象）
modules.low_level.encoder_state_dict / actor_state_dict   # 冻结教师副本（daggerfull-16288）
optimizers.vision_distill
training_state.current_iteration / iteration_semantics / total_steps
training_state.ramp_probability / ramp_start_h / ramp_end_h
training_state.ramp_clock_h
training_state.safety_threshold / safety_fixed / safety_calibration_l2
training_state.lr_scheduler_state / rng_state
training_state.soft_stay_frozen / soft_stay_reason / training_status
replay.enabled = false             # 首轮不启用单帧 replay（LSTM 不适用）
lineage.parent_checkpoint_sha256 / teacher_low_level_sha256 / config_sha256 / code_commit
capabilities.uses_depth = true
capabilities.uses_height_scan_at_inference = false
capabilities.deployable = false
lstm_reset_contract                # reset mask 语义；hidden 不跨运行恢复
```

视觉路径使用：

```text
iteration_semantics = "completed_outer_iterations_v1"
```

`current_iteration` 表示已经完整完成的外层 iteration 数；每个外层 iteration
包含 `num_steps_per_env` 个环境步/视觉优化更新，但只推进一次平台 lifecycle。
`total_steps` 是累计环境样本数，不是平台模型 ID，也不是 optimizer update 数。
修复前没有 `iteration_semantics` 的视觉包按旧的零基 loop index 读取，并在恢复时
加一转换，避免重复最后一轮。

Anchor R2（阶段 5 的历史父阶段）使用相同顶层格式，并增加：

```text
format = "kaiwu_train_v1"
schema_version = 1
stage_type = "standard_visual_ppo"
model_spec = proprio45 / scan256 / depth180x320x1 / latent32 / action12 / goal0
modules.vision_encoder.state_dict
modules.low_level.actor_state_dict
modules.critic.state_dict
modules.action_distribution.std
modules.s0_anchor.vision_encoder_state_dict / actor_state_dict
optimizers.visual_ppo
training_state.current_iteration / anchor_session_elapsed_hours
training_state.elapsed_training_hours / phase_label
training_state.schedule_mode = "visual_anchor_anneal_v2"
training_state.run_name = "standard-anchor-r2"
training_state.source_parent_model_id = 28401
training_state.action_anchor_weight / latent_anchor_weight
training_state.trainable_modules = cnn/actor/lstm/critic
training_state.actor_updates_paused / pause_reason
training_state.baseline_hard_termination_rate
training_state.rng_state
lineage.s0_checkpoint_sha256
lstm_reset_contract.live_hidden_saved = false
capabilities.requires_depth = true
capabilities.requires_height_scan_for_actor = false
capabilities.deployable = false
```

阶段文件名固定为：

```text
model.ckpt-anchorcritic-<平台ID>.pkl
model.ckpt-anchoractor-<平台ID>.pkl
model.ckpt-anchoranneal-<平台ID>.pkl
model.ckpt-anchorfinal-<平台ID>.pkl
```

同一 `visual_anchor_anneal_v2` 断点恢复时以
`anchor_session_elapsed_hours` 续接阶段；`elapsed_training_hours` 只记录累计训练时长，
不得决定阶段。从 S0 或其他 schedule 迁移时，Anchor session clock、Critic、optimizer
和安全窗口从零开始。live LSTM hidden 不跨运行保存，恢复后必须 reset 环境并清零
学生和 S0 anchor hidden。评估必须在最终 Camera TOML 中显式设置
`[env_conf].policy_entry = "visual_policy_optimization"`，只读取
`modules.vision_encoder`、`modules.low_level.actor_state_dict` 与动作分布，不加载 Critic
或 S0 anchor。Camera `task_name` 只是旧 LBC 回退的依据，不能覆盖显式视觉 PPO 入口。

#### `visual_command_generalization_v1`：当前命令泛化训练恢复包

`standard-command-r1` 从 Anchor R2 固定评估选择的包迁移。它不改变 Actor77、深度输入、
LSTM reset 或部署合同；S0 只在训练时计算固定的 action/latent anchor `0.35/0.10`。包仍为
不可部署训练恢复包：

```text
format = "kaiwu_train_v1"
schema_version = 1
stage_type = "standard_visual_ppo"
model_spec = proprio45 / scan256 / depth180x320x1 / latent32 / action12 / goal0
modules.vision_encoder.state_dict
modules.low_level.actor_state_dict
modules.critic.state_dict
modules.action_distribution.std
modules.s0_anchor.vision_encoder_state_dict / actor_state_dict
optimizers.visual_ppo
training_state.current_iteration / iteration_semantics / total_steps
training_state.schedule_mode = "visual_command_generalization_v1"
training_state.run_name = "standard-command-r1"
training_state.command_session_elapsed_hours
training_state.command_target_probability
training_state.requested_source_probability / effective_source_probability
training_state.effective_target_probability
training_state.source_target_counts / target_bucket_counts / command_out_of_range_count
training_state.command_hook_status / command_hook_verification
training_state.command_hook_failure_count / command_hook_last_failure_reason
training_state.command_profile_mix_history
training_state.action_anchor_weight = 0.35
training_state.latent_anchor_weight = 0.10
training_state.rng_state
lineage.transition_parent_platform_model_id / transition_parent_sha256
lstm_reset_contract.live_hidden_saved = false
capabilities.requires_depth = true
capabilities.requires_height_scan_for_actor = false
capabilities.deployable = false
```

`command_session_elapsed_hours` 与 command hold/bucket/RNG 只在同一 schedule 的恢复中续接；
从 Anchor R2 迁移时它们清零，模型和 optimizer 仍按 transition resume 恢复。计数比例按
command 重采样事件记录，不得冒充 env-step 比例。没有公开 command setter 或读回校验失败时，
`effective_target_probability=0`，不得通过改 policy observation 伪造命令实际已写入环境。
一次失败后的有效 setter/readback 成功会恢复 live hook 状态为 `active`，同时保留累计失败
计数和最后失败原因。

文件名仅为同一 payload 的以下训练恢复包，不生成 `locomotion` 或 `lbc_loco` 同 ID 副本：

```text
model.ckpt-commandbase-<平台ID>.pkl
model.ckpt-commandblend-<平台ID>.pkl
model.ckpt-commandfull-<平台ID>.pkl
```

阶段标签只用小写英文，平台 ID 仅由框架传入。评估显式给出 ID 时，loader 只能在该 ID 的
`command*`、`anchor*` 与兼容视觉标签中排序；没有同 ID 文件只输出候选诊断并拒绝跨 ID
静默回退。训练代码不会由本地未保存文件覆盖评估模型包。`command*` 继续
`deployable=false`，部署端仍只接受经过独立导出和审查的 `lbc_loco` 制品。

#### `p15_response_adapter_v1`：P1.5 低层扩域与响应器训练包

P1.5 从 `commandfull-34728` 的 schema 1 包迁移，保持 Standard Actor77、depth、
proprio45 和 action12 外部合同不变。训练包升级到 schema 2，并加入高层内部的
adapter-only 组件；它仍是不可部署的训练恢复包：

```text
format = "kaiwu_train_v1"
schema_version = 2
stage_type = "p15_response_adapter"
model_spec = proprio45 / scan256 / depth180x320x1 / latent32 / action12 / goal0
modules.vision_encoder.state_dict
modules.low_level.actor_state_dict
modules.low_level.locomotion_encoder.state_dict
modules.low_level.actor.state_dict
modules.low_level.critic.state_dict
modules.critic.state_dict
modules.action_distribution.std
modules.high_level.component_status = "adapter_only"
modules.high_level.response_adapter.spec / state_dict
optimizers.visual_ppo
optimizers.response_adapter
schedulers.low_level
schedulers.response_adapter
training_states.global.compound_schedule_phase / session_elapsed_hours / env_steps
training_states.low_level.current_iteration / gradient_steps / skipped_nonfinite / rng_state
training_states.response_adapter.iteration / gradient_steps / skipped_nonfinite / rng_state
training_states.response_adapter.buffer.resume_policy = "completed_records_only_v2"
contracts.command / command_digest
contracts.feedback.profile / implementation.sha256 / feedback_digest
contracts.critic_transport = wire346 -> critic316 + response_aux30
lineage.low_level_state_digest
capabilities.envelope_type = "piecewise_union_v1"
capabilities.high_level_component = "adapter_only"
capabilities.deployable = false
```

运行时维度固定为：

```text
policy observation = proprio45 | scan256 | depth57600 = 57901
worker privileged wire = critic316 | response_aux30 = 346
aisrv Critic/PPO storage = critic316
response observation = active_target3 | exec3 | measured_velocity3 |
                       valid1 | age1 | IMU6 | capability15 = 32
response profile = future_velocity9 | pose_delta3 | stuck1 | log_sigma_xyz_1s3 = 16
```

`response_aux30` 只用于 Adapter 标签与审计，禁止进入低层 Critic、Actor 或 PPO
minibatch。`velocity_log_sigma` 的三个值是 1.0 秒速度的 XYZ 不确定度，不得解释成
三个 horizon 各一个标量。`measured_velocity3` 的短时语义固定为
`SportMode(vx,vy) | IMU gyro(wz)`；UWB 不进入 response observation、tracking error 或
0.2/0.6/1.0 秒预测，只允许作为独立的 3-5 秒低频卡滞证据。`capability15` 使用
`piecewise_union_profile_v1`，显式描述主 `[vx,wz]` 分支、`vy` 专项分支和父域 replay，
不得解释成三轴任意组合的矩形包络。文件标签固定为：

```text
model.ckpt-responsebase-<平台ID>.pkl
model.ckpt-responseexpand-<平台ID>.pkl
model.ckpt-responsefull-<平台ID>.pkl
model.ckpt-responsecalib-<平台ID>.pkl
```

平台 ID 与路径只能由框架注入，不生成同 ID 别名。schema 2 续训恢复 aisrv 的
低层/Adapter 权重、optimizer、scheduler、RNG、计数和已经形成标签的 records；未完成的
future history 在低层版本切换或新进程 resume 时必须清空，禁止跨 optimizer update 或新
任务 reset 拼接。完整状态标记为 `weights_optim_rng_resume_with_history_reset`；缺 scheduler、
RNG 或必要训练状态时显式降级为 `warm_start` 并告警。worker command clock 不新增 IPC，
跨任务续训时必须把 checkpoint 的 session hour 显式写入
`p15_response.command_resume_offset_hours`。该差异只告警，不成为长训单点阻断门禁。

Standard 若要从 `response*` 包继续训练，必须显式设置 `low_level_only_preload=true`；
loader 只恢复视觉 Encoder、Actor、Critic、动作方差及可选 PPO optimizer，并明确忽略
`modules.high_level`。默认 Standard candidate 顺序不得静默偏向 `response*`。

#### `p2_nav_ppo`：P2 Track 连续高层训练包

`p2nav2hsafedir` 显式使用最新验证通过的 `p2nav10hvyavoid2` 完整三轴 P2 schema 2 包执行
`p2_safe_direction_continue_warm_start`。NavigationEncoder、三轴 Actor/LSTM、Critic、
ResponseAdapter、return statistics、RNG 与兼容 optimizer moments 保留；新增
training-only `NavigationSafetyHead`，其 Adam state 从空状态开始。
本阶段 checkpoint 标签为 `safewarm`、`safefull`、`safestable`，候选优先级固定为
`safestable > safefull > safewarm >` 旧 P2 标签。同 ID 续训和评估必须先发现新标签，不能因
候选表遗漏而回退到旧 `navfull`。
旧 P2 reward-v1 包仍可作为显式 warm start：保留低层、NavigationEncoder、Actor/LSTM 与
ResponseAdapter 权重，重建 Critic、return statistics、高层 optimizer/scheduler 和 rollout；
reward-v3 P2 包按完整训练合同 exact resume；旧 P2 reward-v1/v2 包只允许按显式 warm start
重建 Critic、return statistics 和高层优化状态。平台请求 ID、文件名 ID、payload ID 和 lineage
仅用于候选优先级与追溯，不得因不一致单点拒绝；候选最终仍必须通过 stage、module、spec、shape
和 finite 校验。
低层 VisionEncoder/Actor 全程冻结；高层是会直接产生 `[vx,vy,wz]` 的动作组件，因此不能沿用
P1.5 的 `adapter_only` 评估降级路径：

Standard 低层评估是唯一例外，但必须由 `low_level_only_eval=true` 或 P2 的
`standard_low_level_only_eval=true` 且当前 terrain mode 为 Standard 显式启用。该路径只读取
`modules.low_level.locomotion_encoder/actor`，不创建或执行完整高层；未显式启用时仍硬拒绝忽略
action-producing high-level。Track 评估始终加载完整 P2。

```text
format = "kaiwu_train_v1"
schema_version = 2
bundle_kind = "hierarchical_control_v3"
stage_type = "p2_nav_ppo"
deployable = false
modules.low_level.locomotion_encoder / actor / critic / action_distribution / s0_anchor
modules.high_level.component_status = "complete"
modules.high_level.navigation_encoder / navigation_safety_head(training_only) / actor / critic / response_adapter
each learnable leaf = class_name / spec / state_dict
optimizers.high_level_actor / high_level_critic / response_adapter
schedulers.high_level_actor / high_level_critic / response_adapter
transparent_parent_optimizers / transparent_parent_schedulers
transparent_parent_training_states.global / low_level
training_states.high_level.iteration / nav_ticks / effective_training_seconds
training_states.high_level.frame_count / session_effective_seconds / lifetime_effective_seconds
training_states.high_level.actor_gradient_steps / critic_gradient_steps
training_states.global.train_scope = "high_level_and_response_adapter"
training_states.high_level.shuffle_rng_state / action_rng_state / vy_action_rng_state / neutral_rng_state
training_states.response_adapter.gradient_steps / rng_state / buffer
contracts.command / confidence / feedback / reward / training / critic_transport
lineage.source_parent_model_id / parent_checkpoint_sha256 / low_level_state_digest
curriculum_diagnostics
```

运行时维度固定为：

```text
policy observation = proprio45 | scan256 | goal4 | depth57600 = 57905
worker privileged wire = critic323 | response_aux30 | diagnostic_aux32 = 385
NavigationEncoder(depth57600) = nav_feat32
nav_nonvisual = goal4 | target3 | exec3 | measured3 | valid1 | age1 | IMU6 |
                nav_capability15 = 36
actor input = nav_feat32 | nav_nonvisual36 | response_profile16 | confidence1 = 85
critic input = critic323 | active_target3 | nav_capability15 = 341
policy action distribution = incremental diagonal tanh-squashed Gaussian heads over [vx,wz] + [vy]
mapped target = [vx, vy, wz]
vy trusted core = [-0.20,0.20], exploration hard boundary = [-0.40,0.40] from first rollout
response_aux[24] = reset boundary
response_aux[25] = terminal reason code (0 none / 1 success / 2 failure / 3 timeout)
response_aux[28:30] = terrain column / row
diagnostic_aux[30:55] = gait duty / swing / air / frequency / slip / valid
diagnostic_aux[55:58] = pre-step terrain column / row / goal distance
diagnostic_aux[58] = max non-foot contact force over the latest 0.2 seconds
diagnostic_aux[59] = terminal-safe current Track segment from root_pos_w.x (-1 invalid)
diagnostic_aux[60] = gait ContactSensor name/shape mapping valid
diagnostic_aux[61] = full-body collision mapping valid
```

当前训练合同固定为 `run_name=p2nav2hsafedir`、本轮 `target_effective_seconds=7200` 和三段 Track
`pyramid_slope_inv -> pyramid_stairs_inv -> open_entry_maze`。导航
训练使用 128 环境、20 个静态 difficulty column 与 `curriculum=false`；列 difficulty 为
`col/20=0.00..0.95`，reset 不重采样 column，出生 row 在两个非迷宫段随机。出生 row、当前物理段
和难度列是三个独立字段。
`p2_track_reward_v9_safe_direction_teacher` 只在 5 Hz transition 结算。局部正负 progress 与
new-best 不再直接进入 PPO；稠密导航信号改为
`gamma_frame^duration * phi_after - phi_before`，其中
`phi=2*(episode_start_distance-best_distance)`，所有 terminal 的 `phi_after=0`。因此历史最佳距离
允许迷宫绕行，但失败或超时会结清此前势函数，前两段进展不能补贴未完成 episode。timeout 为
`-22.5`、hard failure 为 `-25`、success 为 `+50`。保留非足端 collision 与基于单调 best-distance frontier 的 stagnation，删除
可刷取的 recovery bonus。`predictive_collision_risk` 只读取部署可得的归一化深度和三轴
target command，使用 left/center/right 三个方向与 upper/middle/lower 三个高度带的 20% 稳健
低分位；通过上下带一致性抑制缓坡和仅下部接近的台阶误报，再按 `[vx,vy,wz]` 的预测运动方向
平滑选择扇区，单 tick 最低 `-0.02`。零/无效深度按 5m 超量程处理，零命令严格不处罚，也不奖励远离障碍；旧中央
ROI 风险只保留 shadow diagnostic。`gait_symmetry` 的 PPO 权重为零，四足 excess 仅用于诊断。
training-only SafetyHead 从 `nav_feat32` 预测三方向 risk；标签只由 scanner wall risk 与
`height_scan256` 地形连续性生成，不含 goal/heading。错过明显安全替代方向时才产生渐进负奖励，
scanner 或 height scan 无效、静止、全部堵塞或方向近似等价时严格为零。网格按
`ordering=xy` 解释：低 row 是 body `-y` 右侧，高 row 是 body `+y` 左侧；对外向量顺序仍固定为
`[left,center,right]`。安全选向正确率只在非并列且 `safe_gap` 超过 margin 的条件样本上计算。
Track/Standard eval 与 exporter 忽略该 Head。
Adapter/UWB/heading 或地形门控不得进入 PPO reward。terminal outcome、命令链、速度、步态和
碰撞诊断必须使用首次 done 前冻结快照；reset 后数据只属于下一 episode，terminal tracking 无效。
两小时 LR/entropy 阶段、session/lifetime 时钟、50 Hz `frame_count`、return statistics、gait baseline 与 training contract digest 是 exact
resume 必需状态。

平台标准评估回调只转发 policy observation，不转发 critic group。`p2_nav_eval` 因此在 worker
侧把 `response_aux30` 写入 P2 未消费的 `scan256` 前 30 槽，并在下一槽写入固定 transport
marker；总维度仍为 57905。aisrv 只在 marker 存在时拆出 aux30，并构造仅供
`AlgorithmP2NavPPO.frame_begin()` 运输的零值 critic323 前缀。该 eval-only wire 不进入 Critic，
不改变训练 observation、网络输入、checkpoint 或部署接口。缺 marker 时拒绝推理，禁止把真实
height scan 误读成反馈字段。

P2 为 ResponseAdapter 保留 P1.5 的 `piecewise_union_profile_v1` capability 语义；
Actor/Critic 使用独立的 active-mask/min/max/slew capability15。两种 15 维向量虽然维度相同，
禁止互换。`adapter_confidence` 只由部署可得的 velocity valid/age、Adapter log-sigma 和
超出可信核心域的幅度计算；训练态 10% neutral dropout 不进入 eval。

P2 文件标签固定为：

```text
model.ckpt-navwarm-<平台ID>.pkl
model.ckpt-navadapt-<平台ID>.pkl
model.ckpt-navfull-<平台ID>.pkl
```

平台路径和数字 ID 仅由无参数 `agent.save_model()` 的框架 wrapper 注入。exact resume
必须同时具备完整高层模块、三套 optimizer/scheduler、独立 action/shuffle/neutral/Adapter
RNG、计数和兼容 completed-record buffer；缺失或非有限状态必须拒绝伪装成 exact resume。
live hidden、未完成 rollout、
未完成 future history和 per-env terrain 状态不保存，resume 后强制 reset。当前 deploy
目录没有 P2 高层导出器或 runtime，因此 `component_status="complete"` 仍必须按
`deployable=false` 拒绝部署。

平台 wrapper 在未启用 trajectory recorder 时会丢弃公开 `truncated/time_outs`。P2 因此以
worker-owned `response_aux[24:25]` 恢复 reset 和终止原因；reset 且不是 success/failure
时按 timeout 处理。当前 wrapper 不能提供 terminal critic observation，所以 timeout 的
`continuation_mask=0` 且 `bootstrap_mask=0`，禁止使用 auto-reset 后新 episode 的 observation
为旧 transition bootstrap。若未来平台显式提供 terminal observation，才允许恢复 timeout
单次 bootstrap。

`p2_nav_eval` 使用 `evaluate_full` 的模块-only 装配：只实例化低层 Encoder/Actor、
NavigationEncoder、高层 Actor 与 ResponseAdapter；Critic、PPO storage、ResponseBuffer、
optimizer 和 scheduler 均不创建、不恢复。首次推理前必须由平台 runtime 的
`eval_model_dir/eval_model_id`（缺失时回退包内 `configure_app.toml`）优先选择同 ID P2 文件，
同 ID 不存在时允许发现同目录的 P2 候选并告警后继续结构校验；没有
`evaluate_full_modules_only` 加载结果时拒绝随机参数评分。模块权重仍按
`class_name/spec/state_dict` 严格校验。
P1.5 父包顶层的 legacy `action_distribution` 与 `s0_anchor` 在 P2 首存时迁移进
`modules.low_level` 的规范 leaf，不能因 P2 低层只做 inference 就丢弃；这是后续
`resume_low` 恢复动作方差和 S0 anchor 的必要透明状态。

#### `p3_standard_joint`：Standard 高低层联合恢复包

P3 从完整 P2 schema-2 包显式 warm start，在单个 Standard+Camera 任务内维护两条独立
recurrent rollout 时间轴。外部 policy observation 与 P2 相同，低层仅通过输入适配器删除
goal4；这不是低层网络结构变化：

```text
policy observation = proprio45 | scan256 | goal4 | depth57600 = 57905
low input adapter  = proprio45 | scan256 | depth57600 = 57901
low PPO storage    = proprio45 | frozen_cnn_feat32 = 77 (training-only)
high actor input   = nav_feat32 | nav_nonvisual36 | response16 | confidence1 = 85
low action         = joint12 at 50Hz
high action        = [vx,vy,wz] at 5Hz
environment step   = joint12 (platform contract unchanged)
training worker wire = critic323 | response+diagnostic62 | p3_training_extra108 = 493

The P3.5 training-only tail retains the original runtime/gait/command/Sim2Real
fields and appends joint-acc12, contact force/onset/over-threshold-duration14,
mapping-valid flags, and Push event/delta/age/active/telemetry fields.
These fields are excluded from policy observations,
evaluation inputs, exported models, and deployment interfaces.
```

平台会覆盖 `isaac_env/base_env.py`，P3 不依赖该文件的本地补丁。低层恢复阶段由 P3 worker
采样器通过公开 `base_velocity` tensor 写入并 readback 2-8 秒三轴命令，低层 observation、worker
reward 与实际执行保持一致；高层开始拥有命令后低层
冻结，worker 低层 reward 不进入任何低层更新。P3 的
`_p3_goal_positions` 是训练 worker 私有局部目标，不替换平台 Standard scorer 的任务定义。
局部目标使用相对真实出生点的径向里程碑：M1 `1.3-1.6m`、M2 `2.5-2.9m`；M3 使用
`terrain_width/2-0.1m` 的平台完成公式。默认 8m 地块对应 proxy `3.90m`、控制目标 `3.93m`
与边界 `4.00m`。M1/M2 进入 0.5m 只结算一次事件奖励，不终止或 reset Standard episode；
M3 达到 proxy 后命令归零等待平台 scorer。正常晋级保持方向，timeout 只允许相对 episode
初始方向左右 30/60 度重规划。正式 Standard success 仍由平台 scorer 产生；proxy/platform
一致率和 joint success 仅作监控，不能反向改变 scorer。

```text
format = "kaiwu_train_v1"
schema_version = 2
stage_type = "p3_standard_joint"
deployable = false
modules.low_level.locomotion_encoder / actor / critic / p3_critic / p3_anchor / action_distribution
modules.high_level.navigation_encoder / navigation_safety_head(training_only)
modules.high_level.actor / critic / response_adapter
optimizers.low_level / high_level_actor / high_level_critic / response_adapter
training_states.low_level / high_level / response_adapter / global
training_states.low_level.gait_baseline / mirror_rng_state / mirror_training_fraction (training_only)
training_states.low_level.depth_fault / memory_auxiliary / camera_timing / memory_training_fraction (training_only)
training_states.low_level.p35_reward_baseline (training_only)
training_states.global.session_effective_seconds / lifetime_effective_seconds
training_states.global.low_updates / high_updates / compound_schedule_phase
training_states.global.p3_anchor_digest / low_level_version
training_states.global.domain_randomization_phase / domain_randomization_realized
training_states.global.runtime_m3_contract
contracts.p3_standard_joint.name = "p35_low_speed_gait_push_v1"
```

当前两小时 P3.5 session 的阶段标签固定为 `gaitfixcalib`、`repair`、`pushwarm`、
`pushfull`、`stable`。候选文件缺失时可继续查找配置父包；一旦选中文件，格式、模块、
spec/shape、optimizer exact-resume 状态或有限值不兼容必须停止。平台模型 ID、文件名 ID 和
lineage 只用于选择、告警与追溯，不得形成单点硬门禁。live hidden、未完成 rollout、未完成
future history和每环境局部目标状态不保存，resume 后统一 reset。

本轮父包选择固定为任务 `235689` 的最终 `stairfinal` checkpoint（模型 ID `1013548`）；实际文件
SHA256 与模块 digest 必须由加载器在下载并解析真实制品后记录。训练命令中所有含 `vx` 的类型
共用 `0.10-0.35/0.35-0.70/0.70-1.00m/s` 三档和 `55/30/15` 概率。ResponseAdapter replay
按 session 阶段使用 `50/25/25`、`60/25/15`、`75/15/10` 的最新/近期/父 records 比例；最后
15 分钟低层冻结不代表 Adapter 冻结，仍每 rollout 更新一次。

当前 7200 秒 session 只执行低层 recurrent PPO 和 ResponseAdapter：`0-900s gaitfixcalib`、
`900-4500s repair`、`4500-5400s pushwarm`、`5400-6300s pushfull`、
`6300-7200s stable`。低层 rollout 与 TBPTT 均为 128 帧；低层 CNN、高层
NavigationEncoder/Actor/Critic/SafetyHead 全程冻结，高层 rollout、PPO 和 optimizer/scheduler
step 均不执行，`high_updates` 必须保持 0。P3.5 另外冻结低层 Actor body/std，只允许 LSTM、
RNN output、最终 action head 和 Critic 更新；`stable` 阶段低层全部冻结。从旧 P3 合同 warm
start 时保留高层模块及 optimizer 状态，仅重置本轮 session clock、rollout 与 live hidden。

训练期 depth contract 不改变模型输入布局。每环境 `near_clip` 保持父包合同；冻结 CNN feature32
通过随机 phase 的 30Hz capture、50Hz hold 和 10 帧 FP16 队列模拟相机时序，主动延迟上限
150ms，150-250ms 只作 shadow。完整 128 帧 sequence 继续使用父包末段 50% 深度故障强度。
clean 路径使用冻结父模型 anchor，fault/delayed 路径只允许 memory auxiliary 更新低层 LSTM、
RNN output 与最终 action head；eval/export/deploy 忽略这些 training-only state，且不运行人工增强。
memory auxiliary 的选择 mask 是像素故障与实际交付 feature age 大于零的并集，必须分别报告
fault-only、delay-only 与交集占比，不能把纯 sample-and-hold/延迟帧排除在教师监督之外。

同一个 P3 包支持两种独立评估入口，二者共用同一份候选发现与结构验证器
（`p3_standard_joint_eval_candidates` + `validate_p3_eval_bundle`），绝不回退
P2/LBC/随机权重：

- **`p3_standard_eval`（Standard+Camera）**：只从 P3 包加载
  `modules.low_level.locomotion_encoder` + `modules.low_level.actor`（冻结
  VisionEncoder + Actor77）。policy observation 为低层合同 57901（proprio45 |
  scan256 | depth57600，goal0），输出 12 维关节动作；使用平台 Standard 原生命令、
  reward、termination 与 scorer。不创建/执行高层 Actor、NavigationEncoder、
  ResponseAdapter、SafetyHead、Critic、optimizer 或训练 buffer，不使用 P3 局部目标、
  goal4 或训练命令覆盖。
- **`p3_track_eval`（Track+Camera）**：从同一 P3 包加载完整层级：冻结低层
  VisionEncoder/Actor、NavigationEncoder、三轴 Actor/LSTM、ResponseAdapter。
  policy observation 保持 57905，高层 5Hz / 低层 50Hz，高层发布 `[vx,vy,wz]`，低层
  最终输出 12 维关节动作。reset/termination/timeout 正确清空 recurrent hidden，
  保留已验证的 Track goal-reached→scorer 链（完成数不再恒为 0）。
SafetyHead/Critic/optimizer/scheduler/训练 buffer 一律不创建。

P3 训练专属步态状态不改变 57901 低层输入、12 维动作或部署接口。镜像训练只在 joint order、
action scale、PD stiffness/damping、effort limit 和 `contact_forces` 足端映射均明确左右一致时启用；
异常只关闭 training-only mirror/gait 奖励并告警。交叉落脚使用完整 root quaternion 逆旋转后的
body-y；P2 共享诊断继续报告接触帧平均滑移速度，P3 私有 tail 只在 stance 结束帧报告一次累计
世界 XY 滑移距离，训练奖励不会在后续窗口重复扣分。镜像合同固定为
`FL<->FR`、`RL<->RR`，`vy/wz` 与左右相关轴取反，Hip 按机械轴交换并取反、Thigh/Calf 仅交换；
depth 水平翻转，scan 置换由真实 lateral ray 坐标生成。前 15 分钟采集 4 类地形 × 3 类运动健康
基线；样本不足按同地形、全局逐级回退，全局仍不足时对应接触/交叉/饥饿奖励归零。镜像和步态
baseline/RNG 仅写入 training state，`p3_standard_eval` 与 `p3_track_eval` 必须忽略它们。
worker command sampler 位于独立环境 worker，当前公开 transport 不回传其 live RNG/hold 状态；
checkpoint 明确记录 `seeded_fresh_after_environment_reset`，不得宣称该部分 exact resume。

#### `p4_nav_ppo`：冻结 P3.5 低层的 Track 导航鲁棒包

P4 Maze 强化当前父包选择为 `p4nav2h_1256446-F`（平台模型 ID `1256446`，checkpoint SHA256
`8aae0892664f2f263949f7e5f9f3b53ecd2936b44e80f3783091579bcf3d3461`）。该 ID只用于候选选择；
请求/配置 ID 不可用时只允许唯一结构兼容 P4 discovery，多个候选必须报歧义。选中包仍须通过
stage、模块 spec、tensor shape 与有限值验证后才能 warm start。

P4 不改变部署形状：worker policy 仍为 57905、低层输入 57901、Actor77、joint12，高层 Actor85。
R2 训练 wire 为 `p4_worker_wire_v2=507`：稳定 P3.5 的 493 维之后仅附加未裁剪米制 goal2 与
wall-stuck diagnostics12；这些字段不进入 Actor/Critic。Track eval wire 仍为 385，Standard/Track
评估和部署 I/O 不变。低层 CNN/LSTM/Actor/std/Critic
全程冻结；P4 只更新 NavigationEncoder、高层 Actor/LSTM、新建 Critic、SafetyHead 和
ResponseAdapter。低层可复用 delivered frame 对应的 feature32，但 recurrent hidden 必须在每个
50Hz tick 使用当前 proprio 推进；高层每 5 个低层帧决策一次，即 10Hz，32 个高层 tick 必须拆成
两个 TBPTT16 序列。P2 及旧 P4 默认 5Hz 合同保持不变，只有新
`p4_maze_attack10hz_v1` checkpoint 可以 exact resume。

高层 normalized tanh-Gaussian 与 log-prob 不变，物理映射版本固定为
`p4_capability_action_mapper_v1`：`vx=[0,effective_max_vx]`、`vy=+-0.30`、`wz=+-0.90`。
checkpoint 与 Track eval 必须携带并核验 mapper 版本；`p2_legacy_action_mapper_v1` 只用于父策略
逐值迁移验证。GoalBelief v3 直接取代旧 Actor Goal 链，训练时只从 P4 tail 读取未裁剪米制目标，
并用 SportMode `vx/vy` 与 IMU `wz` 传播；episode reset 立即重建，普通 segment 变化不伪装成目标
变化。critic、reward、terminal 和 scorer 仍读取干净真值。反馈 xy 无效时不使用伪零平移，过程
方差放大四倍；短跳变不会立即接管，持续五次一致的 5Hz 测量可重捕获。历史目标 stale 时保留
0.05 freshness 下限并限制 `vx<=0.20, |vy|<=0.10, |wz|<=0.25`；从未获得有效目标时停止平移。

共享相机链只缓存 raw/captured/delivered depth 与冻结低层 CNN feature，不缓存最终 recurrent
latent。clean teacher 每 rollout 从当前高层策略刷新，使用相同 Goal/capability/Adapter/reset 与
独立 hidden；训练 storage 只增加 clean action mean 和 delay-only/fault-only/overlap mask。
人工 fault、clean teacher、SafetyHead、Critic、Push wrapper、optimizer 和 storage 均不进入
eval/export。

Push pulse 由 EventManager 原函数 wrapper 回传的 worker step 去重；`push_epoch` 变化或
`seconds_since_push<horizon` 会使 Adapter 0.2/0.6/1.0 秒标签失效，pose/stuck 使用 1.0 秒合同。
Adapter replay 必须先同时核验 record schema、低层 digest、feedback digest、capability/mapper 与
observation/label layout，再尝试 50/25/25；record 必须携带用于重建训练 observation 的实际
response capability15，兼容池不足不得混入旧合同 record。P3 父包历史 completed record 缺逐记录
contract 时，只允许在 parent warm replay 路径按 aux/label shape、有限值、horizon/sequence、来源
digest 格式和序列内部 `(digest,iteration)` 一致性做 `legacy_parent_structural_v1` 迁移。历史记录允许
来自父训练的旧低层版本，不伪装为当前冻结低层 exact 数据；迁移记录保留 P3/P2 capability，当前 P4 记录
缺 contract 仍拒绝，且该迁移不得用于 exact resume。父包 current/earlier lineage 必须保持独立
连续池，不能把 32 条记录对半切成小于 `burn-in8+sequence16` 的不可采样窗口。`safety_cap` 从 delivered depth 与当前
exec 命令的 0.8 秒 slew 弧线计算，Track eval 不加载 SafetyHead 也必须执行同一限速合同。

P4 Maze 八小时 reward 合同为 `p4_maze_reward_v3_10hz_route`。predictive collision raw 按父项
放大 1.25 倍并限制到不低于 `-0.03/tick`；missed-safe 仅在 scanner 有效、正在运动且存在明显
安全替代时结算，方向 gap 归一化后按 0-30 分钟 `0->0.015`、30 分钟-2 小时
`0.015->0.040`、2-6 小时保持 `0.040`、6-8 小时 `0.040->0.030`。predictive、missed-safe 和
yaw-cancellation 合计按比例限制到不低于 `-0.06` 的原始 5Hz 等效尺度，再按实际 duration 缩放。
连续 tick 项统一乘 `duration_frames/10`，因此 10Hz 下每个正常 tick 为原权重 0.5 倍；success、
failure、timeout、stuck reset 等事件 impulse，按米计算的 route excess 和每次决策 command-rate
不缩放。body collision 保持每个 10Hz tick 的完整约束，明确作为更强的碰撞治理。成功 impulse
为 `+200`；卡墙候选持续 1 秒后从 `-0.02` 线性加深到 `-0.10`，安全方向内的目标偏好最多
`-0.04`，额外路程按 `-0.05/m` 且每 tick 最多 0.20m。旧 `frontier_stagnation` 的 PPO 权重为零，
只保留 shadow 诊断；Goal 跳变/丢失故障继续使用独立 `goal_fault_multiplier`。

P4 Maze checkpoint 合同为 `p4_maze_attack10hz_v1`：10 分钟只读诊断使用独立
`session_wall_seconds`，不计入正式 `session_effective_seconds=28800`；诊断结束的 rollout 边界是
正式训练时钟零点。诊断期除 clean/live 对照外，使用独立 RNG 对 10% clean depth 生成轻故障
shadow；该 shadow 只用于 detached 线性探针和动作差异诊断，不进入 Actor observation、PPO
reward 或 rollout storage；fault 鲁棒性验收只使用预留的 held-out probe 环境。P4 启动必须核验
运行时 segment 语义为单段 `maze`，实际三轴 slew 与完整 command contract 一致。exact resume
恢复 wall/training clock origin、clean/fault 累计统计、探针、fault RNG 和已选择分支，并同时核验
完整 training/reward/command/stuck/camera 合同。阶段优先级为
`mazefinal > mazehard > mazeattack > mazefull > mazeprobe > mazediag > legacy P4`。
`p4_standard_eval` 只加载低层；`p4_track_eval` 加载
低层、NavigationEncoder、高层 Actor 与 Adapter，training-only SafetyHead 允许为 `None`；公共
推理路径在该模块缺失时只清零风险诊断，不得实例化随机 Head 或中断首帧。正式平台任务配置
29700 秒（8 小时 15 分钟），覆盖 600 秒诊断、28800 秒梯度训练和 300 秒 rollout/保存/退出余量；
工作流达到有效训练目标后自行结束。完整八小时任务若因
平台中断后续训，必须 exact
resume session/lifetime、optimizer、return statistics、Goal/相机/速度 RNG 与 Push 阶段；live hidden、
EventManager timer、pending rollout 在新环境中重建。

P4 R2 的 `p4_stuck_reset_v1` 只在 worker bridge 中维护世界坐标约束与非足端接触证据，不修改
平台 BaseEnv。active 前必须在线确认既有 `nav_stuck_timeout` term、`time_out=true`、公开
get/set/readback 和 `dt=0.02s`。当前 Maze 八小时合同以 7 秒确认，reason=4 结算 `-15` terminal
impulse 并回收该 episode 未结算的 frontier potential；因此完整 terminal tick 不保证恰好 -15，
但必须通过 `stuck_terminal_episode_return_mean/nonnegative_rate` 验证 reset episode 的整体收益为负。
reason=4 使用 `bootstrap_mask=continuation_mask=0`，同 tick 不重复 collision、predictive collision、
missed-safe 或 stagnation。live tracker、recurrent hidden、pending rollout 和未完成 Adapter history
不进入 checkpoint，resume 后统一 reset。

Track physical segment index 与指标语义分离。运行时仍按世界坐标产生 `0..track_length-1`，指标层
必须根据当前 TOML `sub_terrains` 映射到稳定的 `slope_inv/stairs_inv/maze` bucket。单段
`open_entry_maze` 的 physical index 0 必须报告为 Maze；未配置的稳定 bucket 保持零，不能用固定
`0=slope` 解释污染 Adapter 与 Track 面板。

P3 评估候选标签优先级为 `stable > pushfull > pushwarm > repair > gaitfixcalib`，随后兼容
`stairfinal/stairrobust/stairadapt/stairwarm/staircalib` 及
旧 `highslow/highadapt/adaptercalib/lowfull/lowmedium/lowmild/lowbase`；无同 ID P3 文件时只允许
唯一 discovery，多个候选明确报歧义。请求 ID、
文件名标签与 lineage 不一致只告警；选中文件的反序列化、必需模块、spec/shape 或非有限值
错误必须硬失败，禁止继续用随机参数评分。

每个阶段边界先保存 checkpoint，再通过平台公开 `env.reset(config)` 开始新 episode并清理 live
recurrent 状态；该调用不会重建平台 create-once 的 Isaac 环境。friction `[0.65,1.25]`、base mass
`+-0.4kg`、restitution `[0,0.05]`、noise `0.35` 在首次 reset 装配。`push_robot` EventTerm 同时
以零 XY 速度保留；75 分钟通过 EventManager `get_term_cfg/set_term_cfg/reset` 切到
`+-0.05m/s`，90 分钟切到 `+-0.08m/s`，间隔 12-18 秒。wrapper 必须先调用 Isaac 原
`push_by_setting_velocity`，再记录推前/推后的真实 root velocity delta；配置值不得冒充实测值。
checkpoint session 时间经 reset `usr_conf` 传给 worker，断点恢复立即恢复对应 Push 阶段。
未验证的 COM、PD、action gain/delay 不属于运行合同。低层 optimizer 成功后先推进 low-level
digest/version 并清除未完成 future history，再执行一次 Adapter update。

文件数字 ID 完全使用开悟框架传入值，不由业务代码从 iteration 或父模型 ID
计算。R2 每次保存同一 payload 的阶段文件和评估兼容别名：

```text
model.ckpt-daggerzero-<平台ID>.pkl
model.ckpt-daggerquarter-<平台ID>.pkl
model.ckpt-daggerhalf-<平台ID>.pkl
model.ckpt-daggerthreequarter-<平台ID>.pkl
model.ckpt-daggerfull-<平台ID>.pkl
model.ckpt-locomotion-<平台ID>.pkl
```

阶段标签只用小写字母，数字只出现在平台 ID 中。该训练包依赖仿真
`height_scan256` 且 `deployable=false`，deploy 不得接受。Standard eval 从
`modules.low_level.policy_state_dict` 加载学生；视觉 LBC 从 low-level
encoder/actor 建立冻结教师。

旧 raw bridge checkpoint 只能显式 weight-only 导入；旧
`behavior_distill_v2` 和 `privileged_loco_teacher_v1` 作为历史格式保留，不得
仅改 `format` 字段冒充新 schema。

### 2.2 `privileged_loco_teacher_v1`：历史视觉教师封装

```text
format = "privileged_loco_teacher_v1"
schema_version = 1
encoder_state_dict
actor_state_dict
model_spec = proprio45 / scan256 / latent32 / action12 / goal0
source_teacher_sha256
bridge_checkpoint_sha256
source_iteration / source_training_status / platform_model_id
critic_trained = false
deployable = false
```

它是历史独立封装，不是 Jetson 制品。R2 后新产物优先使用
`kaiwu_train_v1.modules.low_level`；进入视觉蒸馏前仍需比较固定 seed、上下楼
视频和失败率，并记录选用的 checkpoint ID 与 SHA256。

### 2.3 `lbc_loco`：可部署视觉策略候选

只有完成 depth→latent32 视觉蒸馏并保存为 `format="lbc_loco"` 的 Standard
checkpoint，才允许进入现有 Standard exporter 的部署审查。至少需要：

- `vision_encoder_state_dict`；
- `teacher_actor_state_dict`；
- `goal_dim=0`；
- Actor 第一层 `[512,77]`、输出 `[12,128]`；
- 与同一训练产物配套的配置、commit 和 SHA256。

`lbc_loco` 只表示结构可导出，不自动表示相机、ONNX、真机安全或比赛能力已验收。

阶段 4 起，视觉训练恢复包统一为 `kaiwu_train_v1`（§2.1 视觉路径，含
`modules.vision_encoder`）。`lbc_loco` 顶层 key 格式保留给后续部署导出制品：
训练期只保存一个带 `vision*` 标签的 `kaiwu_train_v1` 文件，不再同时复制
`lbc_loco` 或冻结教师的 `locomotion` 副本。未来 `lbc_loco` exporter 会改读
`kaiwu_train_v1.modules.vision_encoder` 并单独导出部署制品。

### 2.4 其他格式

- `visual_ppo`：当前 Standard 部署导出器尚不支持，不能改名冒充 `lbc_loco`；
- Track `lbc_loco`：若 `goal_dim=3`，Actor 必须为 80-D，不能交给 Actor77 exporter；
- Git LFS pointer：不是可用 checkpoint/ONNX 制品。

## 3. STD-DAGGER-R2 的冻结边界

本轮只改 `server/` 的 flat301→Actor77 特权桥接及 loader 契约：

- 已审计的原始源教师 SHA256 为
  `d5999461f00c4634bdea0648e46baac9fba34e621eaa586fe26f64f68953eeed`，但 R2 运行时不将它
  作为字节级加载门禁；预训练教师身份由操作者确认；
- 教师输入 301，学生输入仍含特权 scan，输出 action12；
- 不接入 depth、Goal、UWB；
- 不修改部署 exporter、ONNX、C++ 或 `deploy.yaml`；
- R2 产物不得复制到 `deploy/` 或登记为“可部署”。

## 4. 发布候选必须原子固化的字段

进入可部署 PR 时，server、deploy 和本文件必须在同一 PR 中补齐：

- checkpoint 文件名、来源分支/commit、父 checkpoint、SHA256；
- `format/schema_version/task/goal_dim` 和关键 state-dict shape；
- proprio45 的字段顺序、command `[6:9]` 的单位/裁剪/训练范围；
- depth 尺寸、裁剪、归一化、无效值、帧率、LSTM reset 和相机内外参；
- Goal/UWB 表示、坐标转换、归一化、滤波、失效和停车语义（仅 Track）；
- ONNX 输入/输出名、shape、hidden state、opset 和文件 SHA256；
- action12 的关节顺序、scale、clip、控制频率和急停恢复；
- `deploy.yaml` 与训练 TOML 的逐字段映射；
- Python→wrapper→ONNX 多帧误差、离线回放、吊起和平地真机证据。

当前 378413 checkpoint 在 loco/st9 部署目录中的历史 SHA 一致；ST7、standard
部署树缺失的真实制品继续以各目录 `ARTIFACTS.md` 为准，不在本契约中虚构可用性。
