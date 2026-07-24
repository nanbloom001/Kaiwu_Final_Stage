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
