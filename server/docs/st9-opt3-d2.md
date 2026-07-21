# ST9-Opt3-D2

## 目标

从 ST9-Opt3-D1 视觉学生继续蒸馏，解决 latent 小误差经冻结 Actor 放大为动作误差，以及纯教师轨迹无法覆盖学生闭环偏移状态的问题。

## 血缘

- 代码父提交：仓库最新 `main`。
- 模型父 checkpoint：ST9-Opt3-D1 视觉学生。
- 教师：D1 checkpoint 内冻结的 ST7-Opt3 Actor 与高度扫描编码器。
- 输出命名继续使用 `model.ckpt-track-lbc-loco-<id>.pkl`，保持平台探活和评估兼容。

训练启动时启用 `require_student_resume = true`。如果平台只加载原 ST7 教师 checkpoint，任务会直接失败，禁止随机初始化学生冒充 D2 续训。

## 唯一训练链路变化

1. 损失改为 `0.5 * latent SmoothL1 + 0.1 * cosine distance + 1.0 * raw action SmoothL1`。
2. 教师编码器和 Actor 冻结，只优化 VisionEncoder 与其两层 LSTM。
3. 教师和学生 Actor 使用同一份 proprio、goal，动作损失比较加入默认关节角之前的 12 维原始 Actor 输出。
4. DAgger 按环境独立采样：训练前 20% 学生驱动 50%，中间 40% 为 75%，最后 40% 为 100%。教师仍在每个状态提供标签。
5. TBPTT 序列长度从 8 增至 16，rollout 设为 32 步；hidden/cell 仅在 done 环境清零，并在片段结束后 detach。
6. 完整赛道难度权重调整为 `[0.07, 0.07, 0.08, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15]`，不使用困难段瞬移出生。
7. 深度散点 dropout 降为 3%，块空洞帧概率降为 20%，最大面积降为 10%，持续 2 至 3 帧；高斯噪声和轻微图像姿态扰动保留。

网络结构、观测顺序、latent 维度、Actor、PPO/奖励及部署接口均未修改。

## 启动检查

日志必须包含：

- `stage=TrackLBCLocoD2Config`
- `student_checkpoint=<D1 checkpoint path>`
- `sequence_length=16`
- `student_drive_ratios=[0.5, 0.75, 1.0]`
- `loss=0.5*smooth_l1_latent + 0.1*cosine + 1.0*smooth_l1_action`
- 一次 `[ST9-Opt3-D2 LossScale]`，打印三项未加权损失
- `teacher_max_abs_diff=0`

如 `assert_student_ready` 报错，应更正平台预训练模型选择，不要关闭硬检查。

## 评估建议

先确认干净仿真明显超过 D1，再单独恢复更强视觉破坏。正式对比继续使用相同 128 环境、`level=[0,3,6,9]` 配置，并分别记录 L9 完成、`bad_orientation`、`base_contact` 和迷宫恢复表现。
