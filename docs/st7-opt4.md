# ST7-Opt4：危险姿态角速度提前约束

## 父模型

- 必须从 `ST7-Opt3-30min` 继续训练。
- 平台创建任务时选择评估 `595729` 对应 checkpoint，代码中不硬编码模型路径。
- 不从 Opt2B、ST9 视觉蒸馏或其他 Opt4 候选模型继续。

## 视频前置确认

使用 Opt3-30min 独立运行两次 L9 视频评估：

```toml
[env]
num_envs = 16

[terrain]
level = [9]

[video]
save_mp4 = true
```

同时关闭环境 noise 和 goal noise。只在异常仍以侧翻、向前扑为主时使用 Opt4；若主要失败已变为迷宫卡墙，转入迷宫恢复分支。

## 唯一实验变量

`dynamic_tilt_risk` 的角度阈值、总权重和最大惩罚不变：

- `soft_roll = 0.12`
- `hard_roll = 0.30`
- `soft_pitch = 0.22`
- `hard_pitch = 0.55`
- `weight = -0.30`
- `max_penalty = 2.0`

仅提前角速度阈值：

- roll rate：`0.60/1.80 -> 0.50/1.60 rad/s`
- pitch rate：`0.80/2.20 -> 0.65/1.90 rad/s`

仅增强风险组合中的角速度项：

```python
risk = (
    roll_risk.square()
    + 0.8 * pitch_risk.square()
    + 0.35 * roll_rate_risk.square()
    + 0.30 * pitch_rate_risk.square()
)
```

风险值为正，由 TOML 中的负权重转为惩罚，因此公式内必须使用加号。

## 保持不变

- ST7-Opt3 UWB goal 小噪声参数和实现。
- 原有 reward、termination、地形、速度与 PPO 参数。
- foot clearance、base height、网络结构和观测结构。
- 不引入 UWB 跳变/丢包、地形速度新逻辑或深度增强。

## 训练与评估

- 训练 60 分钟，保留 20/30/40/60 分钟 checkpoint。
- 前 10 分钟检查 `dynamic_tilt_risk`、完成率、yaw rate、动作柔顺度、value loss 和 policy loss。
- 每个候选先运行两次 128 env 正式评估，再对最佳候选运行两次 L9 视频评估。

晋升标准：

- 干净总分 `>= 63.0`
- 总完成 `>= 253/256`
- L9 完成 `>= 62/64`
- 两轮 L9 视频中侧翻+前扑 `<= 1`
- 能耗分和时间分各自下降不超过约 `0.7`
