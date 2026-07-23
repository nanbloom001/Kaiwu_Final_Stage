# STD-BRIDGE-R1 执行卡

> Experiment ID：`STD-BRIDGE-R1`
> 总阶段：`STD-DEPTH-5STAGE` 的阶段 2 第一轮
> 状态：历史 R1 设计稿；已由
> [`STD-DAGGER-R2`](./2026-07-23_STD-DAGGER-R2执行卡.md) 取代，不再作为当前启动依据
> 基线：`main@ba175ca`
> 实现分支：`codex/standard-bridge-r1`
> 禁止事项：本卡未授权深度视觉训练、PPO、Goal/UWB/Track 或部署端改动

## 1. 单一目标

把原始复赛 Standard 10288 的扁平行为迁移到可拆分的特权策略：

```text
T0: proprio45 + height_scan256 = 301 → flat Actor → action12

T1: height_scan256 → 512 → 256 → latent32
    proprio45 + latent32 = 77 → 512 → 256 → 128 → action12
```

R1 仍然读取仿真特权 `height_scan256`，不读取深度相机，因此不是可部署模型。
本轮只有教师 action 监督；原 flat 教师没有 latent32 可供监督。

## 2. 冻结输入与证据

| 项目 | 冻结值 |
|---|---|
| 源文件 | `archive/代码存档/复赛_standard/ckpt/model.ckpt-10288.pkl` |
| 源 SHA256（审计参考） | `d5999461f00c4634bdea0648e46baac9fba34e621eaa586fe26f64f68953eeed` |
| 原始 key 类型 | raw `std`、`actor.*`、`critic.*`，无 `encoder.*` |
| Actor 第一层 | `[512,301]` |
| Critic 第一层 | `[512,316]` |
| Actor 输出层 | `[12,128]` |
| R1 配置 | `server/agent_ppo/conf/train_env_conf_standard_standard_bridge_r1.toml` |
| 配置 SHA256 | `799699b37f12c1e6639a66d4a9cf665828a386d41ac8701ccb7c901f0c84269f` |

旧本地 `archive/代码存档/hjcnew蒸馏1_10288/` 仅作为审计证据，不进入分支。
其中 checkpoint `model.ckpt-locomotion-10288.pkl` 的 SHA256 为
`eb6da58646f7d4a437791de5ae32c6df0061c753e725f7089d233e229aa97a27`，
Actor 第一层为 `[512,77]`；它是约 10 分钟桥接产物，不是原始教师。由于该目录
仍为未跟踪用户资料，本卡不登记不稳定的目录级哈希，也不提交重复源码或 checkpoint。

实现只借鉴历史节点：

- `07d10bc`：flat301 到 latent32 的行为蒸馏方向；
- `0197e5f`：逐环境 DAgger、教师冻结与阶段日志；
- `st9-opt3-d1-vision-distill`、D3/D4/D5：只作历史对照，不作 R1 起点。

## 3. 训练域

- 并行环境：1024；每 iteration 24 步；总预算 5000 iterations；
- 命令：`vx=[0.3,1.3]`、`vy=[-0.2,0.2]`、`wz=[-0.3,0.3]`；
- 重采样：`[6,10]` 秒；
- 地形：坡面 20%、反坡 20%、楼梯 20%、反楼梯 40%、迷宫 0%；
- 完整地形难度，`max_init_terrain_level=9`；
- 首轮关闭 domain randomization、摩擦随机化、观测噪声和外力推动；
- 保留源奖励与速度课程表只为固定环境定义，R1 不使用 reward/PPO 更新。

命令观测超出记录包络时只打印一次 warning，对应环境行的训练权重置 0，
不用 OOD 教师标签反向传播，也不中断整个任务。

## 4. 单任务 DAgger 调度

| iteration | 学生驱动概率 |
|---:|---:|
| 0--1499 | 0% |
| 1500--2249 | 25% |
| 2250--2999 | 50% |
| 3000--3749 | 75% |
| 3750--4999 | 100% |

driver mask 每个环境独立抽取。每阶段末使用两个连续 50-iteration 在线窗口
生成质量诊断，阈值为：

- action cosine `≥0.98`；
- normalized action MSE `≤0.10`；
- NaN/Inf 和教师 OOD 率为 0；
- 学生安全接管率 `≤10%`；
- 硬终止率相对上一阶段增加不超过 1 个百分点。

阈值不满足时打印 warning、记入 checkpoint，仍按预定计划进入下一学生驱动比例。
它们是人工评估依据，不是平台训练门禁。阶段 0 结束后，动作安全阈值固定为
`max(0.25, 2 × P95_action_L2)`。

每步样本权重：正常 `1.0`，教师安全接管 `0.25`，学生 NaN/Inf 或非 timeout
硬终止 `0`。学生非有限值时由教师无条件接管；其梯度行不会进入反向传播。

## 5. Checkpoint 与后续接口

平台探活取目录内最大的文件名数字。为避免预训练教师 `10288` 压过 R1 的
`1--5000` 真实 iteration，文件名使用：

```text
platform_model_id = 10288 + current_iteration
```

checkpoint payload 内的 `current_iteration` 仍保持真实 `1--5000`，DAgger 调度和
恢复一律读取 payload，不从文件名反推。第一轮完整 rollout 后立即保存
`model.ckpt-10289.pkl`；当前 `save_interval=100`，每 100 iteration 常规定时保存。
这不是额外划分的“每 100 轮恢复阶段”。`conf/configure_app.toml` 的
`dump_model_freq` 是框架 Learner 调用 `learn()` 时的落模频率；R1 在自定义
workflow 中直接更新学生，因此 R1 的可控定时保存以本节的
`save_interval` 为准，不假设框架参数会代替这个调用。

常规定时 checkpoint（内容完整，需要时可用于续训）：

```text
model.ckpt-<platform_model_id>.pkl
format = "behavior_distill_v2"
schema_version = 1
```

它包含学生、冻结教师副本、optimizer、iteration、DAgger 阶段、质量诊断历史
（为兼容 schema 仍使用 `gate_history` 字段名）、RNG、
源教师 SHA、配置 SHA、代码 commit 和 `training_status`，可做单文件精确恢复。
旧 raw checkpoint 只允许显式 weight-only 导入。

每个 DAgger 阶段边界都额外发布候选制品：

```text
model.ckpt-bridge-<platform_model_id>.pkl   # 完整恢复 checkpoint
model.ckpt-teacher-<platform_model_id>.pkl  # 视觉蒸馏教师
format = "privileged_loco_teacher_v1"
schema_version = 1
deployable = false
critic_trained = false
```

阈值超限的制品在 `source_training_status` 中记录 `phase_warning` 或
`completed_with_warnings`，但不生成 `blocked` 文件，也不阻断保存。标签只用
单个小写单词，确保平台的 `[a-z]*` 探活正则可识别。关键映射为：

| current_iteration | platform_model_id |
|---:|---:|
| 1 | 10289 |
| 1500 | 11788 |
| 2250 | 12538 |
| 3000 | 13288 |
| 3750 | 14038 |
| 5000 | 15288 |

后续 LBC loader 可严格读取该封装，也兼容历史 `encoder.*`/`actor.*` 权重。
进入视觉蒸馏前，应根据诊断和固定评估结果显式记录选用的
`model.ckpt-teacher-<id>.pkl`；文件存在不代表已自动晋级。
部署端继续只接受 `format="lbc_loco"`；本轮不修改 exporter、ONNX 或 Jetson 代码。

## 6. 启动与恢复

在腾讯平台选择**原始** 10288 checkpoint 作为预加载模型，使用本分支代码，从
`server/` 作为项目根运行：

```bash
cd server
python train_test.py
```

本分支的 `conf/configure_app.toml` 已明确设置
`preload_model=true`、`preload_model_dir=/data/pre_model/ckpt` 和首轮默认
`preload_model_id=10288`。平台任务页若把所选父模型提前传给 `Agent.load_model`
则优先使用该制品；否则 workflow 从上述目录查找最新的结构兼容 checkpoint。
`10288` 没有写死在 workflow 中，因此恢复 R1 时仍可由平台选择新的
`behavior_distill_v2` checkpoint，而不会退回原始教师。

预训练教师由操作者手动确认。上表 SHA 只用于事后审计，不作为
平台 rollout 启动门禁。启动日志应显示：

- stage 为 `standard_bridge_r1`；
- 教师已成功加载，日志中记录的来源与 SHA 仅作追溯；
- 观测为 301，教师 Actor 输入 301，学生 Actor 输入 77；
- `teacher_frozen=true` 且 optimizer 仅含学生 encoder+actor；
- 配置 SHA 和代码 commit 已记录。
- `learner_proxy` 的 sample 成功数可以保持 0；R1 在自定义 workflow 中直接
  使用 `env.step` 更新，应看 `[BehaviorDistill] iter=...`、loss 和 checkpoint。

恢复时只能选择结构兼容的 `behavior_distill_v2` 文件；文件名数字是平台 ID，
不是内部 iteration。配置 SHA 变化、阶段/iteration 不一致或历史
`training_status="blocked"` 只会 warning，阶段按 payload 的 `current_iteration` 重算。

## 7. 最终比较与放行

原始 T0 评估可以与 R1 训练并行，但最终必须使用相同地形、命令、seed 和指标。
R1 学生评估必须 100% student-drive，不能误用 checkpoint 中的 flat 教师。

放行要求：

- 综合完成率不少于 T0 的 95%；
- 上楼、下楼完成率各自下降不超过 5 个百分点；
- 摔倒率增加不超过 1 个百分点；
- checkpoint 中途恢复后 DAgger 阶段不漂移；
- 教师参数训练前后最大绝对差为 0；
- side artifact 能被后续 LBC loader 严格加载。

在线 DAgger 质量阈值仅是诊断，不等于本节放行。没有平台视频、分地形指标和恢复证据时，
状态只能写“静态就绪”或“诊断”，不得晋级视觉蒸馏。
