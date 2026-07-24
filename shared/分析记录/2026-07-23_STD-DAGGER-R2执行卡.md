# STD-DAGGER-R2 执行卡

> 任务名：`std-dagger-r2`
> 分支：`codex/standard-dagger-r2`
> 起点：`codex/standard-bridge-r1-minimal@7d48cba`
> 平台时长：3 小时
> 内部预算：6000 iterations
> 状态：平台长跑、最终模型评估与视频人工检查已完成

## 1. 唯一目标

从原始复赛 Standard 10288 的 flat301 教师重新训练一个模块化特权学生：

```text
教师：proprio45 + height_scan256 → action12

学生：height_scan256 → 512 → 256 → latent32
      proprio45 + latent32 → 512 → 256 → 128 → action12
```

R2 仍读取仿真特权 `height_scan256`，不读取深度图，不可部署。它只负责把
10288 的行为迁移为后续视觉蒸馏可复用的 Actor77 低层结构。

## 2. 训练域

- `num_envs=1024`，`num_steps_per_env=24`，`learning_rate=3e-4`；
- 按 `archive/代码存档/复赛_standard` 当前训练入口
  `StairInvFineTuneConfig` 保持教师代码域：
  `vx=[0.3,1.3]`、`vy=[-0.2,0.2]`、`wz=[-0.3,0.3]`；
- 命令重采样 `[6,10]` 秒；初始范围、hard limit 和四级 velocity curriculum
  均为同一范围；
- 地形比例：坡面 20%、反坡 20%、上楼 20%、下楼 40%、迷宫 0%；
- 首轮关闭 domain randomization、观测噪声和外力推动；
- 教师永久 `eval()`、冻结；optimizer 只包含学生 encoder 和 actor。

证据边界：10288 的 `kaiwu.json` 只记录 `train_step=10288` 和约 3 小时训练，
checkpoint 本身不含历史命令配置。本卡采用的是 `复赛_standard` 归档代码当前
实际选中的训练入口，不把 HJC 视觉/桥接配置当作原教师配置。

## 3. 单任务自适应 DAgger

| 标签 | 学生请求比例 | 最少 iterations | 最多 iterations |
|---|---:|---:|---:|
| `daggerzero` | 0% | 900 | 1500 |
| `daggerquarter` | 25% | 600 | 900 |
| `daggerhalf` | 50% | 600 | 900 |
| `daggerthreequarter` | 75% | 600 | 900 |
| `daggerfull` | 100% | 1800 | 训练至 6000 |

每达到最少预算后，每 50 iterations 检查最近两个连续窗口。正常晋升要求：

- action cosine `≥0.98`；
- normalized action MSE `≤0.10`；
- 非有限值率和教师 OOD 率为 0；
- 安全接管率 `≤0.10`；
- 硬终止率相对上一阶段增加不超过 0.01。

前四阶段到最大预算仍未通过时，从阶段最佳有效点恢复；若无最佳点则恢复阶段
入口。随后强制晋升并记录 `promotion_mode="forced"`，最终状态为
`completed_with_warnings`。最晚晋升仍为 `daggerfull` 留出 1800 iterations。

## 4. 污染保护

- 学生非有限动作或与教师动作 L2 差异超阈值时，所请求的学生环境由教师接管；
- 0% 阶段结束后固定安全阈值：
  `max(0.25, 2 × P95_action_L2)`；
- 正常样本权重 1.0，安全接管样本 0.25，非有限、命令越界和硬终止样本为 0；
- 容量 8192 的 FP16 reservoir 只保存权重 1.0 的
  `observation301 + teacher_action12`；
- 更新损失为 75% 当前访问状态 + 25% reservoir 回放；
- 每阶段保存 entry、best 和 exit 学生/optimizer 快照；
- 教师参数变化、学生权重非有限或 checkpoint 写入失败才硬停止。

日志必须同时显示请求比例、有效学生比例和安全接管率，不能把请求的 100%
描述成实际 100%。

## 5. Checkpoint

统一训练包：

```text
format = "kaiwu_train_v1"
schema_version = 1
```

包含低层完整策略、encoder、actor、冻结教师、optimizer、iteration、DAgger
阶段与晋升历史、reservoir、RNG、模型/观测契约和血缘。预留
`modules.vision_encoder` 与 `modules.high_level` 扩展位，但 R2 不创建这两个模块。

平台 ID 只使用框架传入值，不按 iteration 人工计算。每次保存同 ID 的两份内容
相同的文件：

```text
model.ckpt-<阶段英文>-<平台ID>.pkl
model.ckpt-locomotion-<平台ID>.pkl
```

阶段英文固定为 `daggerzero`、`daggerquarter`、`daggerhalf`、
`daggerthreequarter`、`daggerfull`。它们只含小写字母，符合平台探活正则。
`save_interval=100`，因此第 100 iteration 必须出现首个 `daggerzero` 文件。

Standard 评估入口从 `modules.low_level.policy_state_dict` 加载学生，不加载 flat
教师。后续视觉 LBC 可从同一包的 low-level encoder/actor 建立冻结教师。
Jetson 仍只接受另行导出的 `lbc_loco`/ONNX；本训练包 `deployable=false`。

## 6. 平台验收

先检查第 100 iteration：

- `train_global_step` 增长；
- `daggerzero` 和 `locomotion` 同 ID 文件均存在；
- payload 为 `kaiwu_train_v1`；
- 日志 iteration、loss、cosine、reservoir size 持续变化。

6000 iterations 完成后必须有 `daggerfull`。随后用至少 3 个固定 seed 做相同
命令、相同地形的教师/学生对照，并下载视频核对上楼、下楼、转向和摔倒原因。
未完成视频验收前，不进入深度视觉蒸馏。

## 7. 实际平台结果

训练任务 `standard-dagger-r2`（任务 ID `234080`）完成 6000 iterations，
累计 147,456,000 个环境步，实际运行约 2 小时 59 分。最终日志：

```text
training_status = completed_with_warnings
daggerfull_iterations = 1800
requested_student_ratio = 1.0
effective_student_ratio ≈ 0.998
safety_takeover_rate ≈ 0.002
normalized_action_mse ≈ 0.000939
action_cosine ≈ 0.9998
teacher_parameter_diff = 0
```

四次阶段晋升均为 `forced`，并在晋升前恢复 `phase_entry`。这说明最终模型结果
可用，但自动晋升状态机没有按原计划形成正常晋升链。后续视觉 DAgger 不复用
“到最大预算后仍强制升档”的规则；未通过时应停留并保存 `blocked`。

最终模型 `standard-dag_16288` 的视频评估为 4/4 完成、0 timeout、0 abnormal，
平均分 64.19；`pyramid_stairs` 为 66.76，`pyramid_stairs_inv` 为 61.63。
人工查看视频后确认基本上下楼表现没有明显异常，因此模型能力满足进入深度视觉
蒸馏的条件。

下载目录中的：

```text
model.ckpt-daggerfull-16288.pkl
model.ckpt-locomotion-16288.pkl
```

SHA256 均为：

```text
5bfd5d7c75162ba82f0e7d9514713747ef5d7c360d336fc50546088f5a3fc67c
```

两者是同一 payload 的兼容副本。后续血缘只登记 `daggerfull` 文件。
