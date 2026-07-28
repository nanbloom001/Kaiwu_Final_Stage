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
