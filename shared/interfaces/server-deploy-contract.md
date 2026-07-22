# server ↔ deploy 接口契约

> 版本：`v0.2`
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

### 2.1 `behavior_distill_v2`：训练恢复文件

用途：把 flat301 Standard 教师桥接为模块化特权策略，并精确恢复训练。

必需字段：

```text
format = "behavior_distill_v2"
schema_version = 1
model_spec
student_model_state_dict
teacher_model_state_dict
optimizer_state_dict
current_iteration / total_steps / gradient_steps
platform_model_id / checkpoint_role
dagger_phase_index / dagger_phase_iteration / student_drive_probability
safety_threshold / gate_history / recent_iteration_metrics
rng_state
teacher_sha256 / config_sha256 / code_commit
critic_trained = false
training_status
```

R1 文件名采用 `platform_model_id = 10288 + current_iteration`，确保平台探活不会
把预训练教师 10288 误判为最新学生。R1 TOML 原有的常规定时 checkpoint 使用
`model.ckpt-<platform_model_id>.pkl`；通过阶段闸门的完整桥接副本使用
`model.ckpt-bridge-<platform_model_id>.pkl`；blocked 文件不得发布教师制品。

该文件依赖仿真 `height_scan256`，`deployable=false`（语义上），部署端不得接受。
旧 raw bridge checkpoint 只能作为显式 weight-only 输入，不能伪装成精确 resume。

### 2.2 `privileged_loco_teacher_v1`：视觉蒸馏教师

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

它是后续 Standard LBC 的冻结教师，不是 Jetson 制品。server LBC loader 兼容该
封装与历史 `encoder.*`/`actor.*` raw 权重；新产物优先使用封装格式。如最新
R1 checkpoint 已标记 `blocked`，LBC loader 不得在 `latest` 模式下静默回退到
更旧的 teacher；需人工审核后显式选择已通过闸门的 teacher ID。

### 2.3 `lbc_loco`：可部署视觉策略候选

只有完成 depth→latent32 视觉蒸馏并保存为 `format="lbc_loco"` 的 Standard
checkpoint，才允许进入现有 Standard exporter 的部署审查。至少需要：

- `vision_encoder_state_dict`；
- `teacher_actor_state_dict`；
- `goal_dim=0`；
- Actor 第一层 `[512,77]`、输出 `[12,128]`；
- 与同一训练产物配套的配置、commit 和 SHA256。

`lbc_loco` 只表示结构可导出，不自动表示相机、ONNX、真机安全或比赛能力已验收。

### 2.4 其他格式

- `visual_ppo`：当前 Standard 部署导出器尚不支持，不能改名冒充 `lbc_loco`；
- Track `lbc_loco`：若 `goal_dim=3`，Actor 必须为 80-D，不能交给 Actor77 exporter；
- Git LFS pointer：不是可用 checkpoint/ONNX 制品。

## 3. STD-BRIDGE-R1 的冻结边界

本轮只改 `server/` 的 flat301→Actor77 特权桥接及 loader 契约：

- 源教师 SHA256：`d5999461f00c4634bdea0648e46baac9fba34e621eaa586fe26f64f68953eeed`；
- 教师输入 301，学生输入仍含特权 scan，输出 action12；
- 不接入 depth、Goal、UWB；
- 不修改部署 exporter、ONNX、C++ 或 `deploy.yaml`；
- R1 产物不得复制到 `deploy/` 或登记为“可部署”。

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
