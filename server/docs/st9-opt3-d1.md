# ST9-Opt3-D1 Track 视觉蒸馏

## 阶段目标

- 教师：平台创建任务时选择 `ST7-Opt3-30min` checkpoint（评估任务 `595729`），代码不硬编码模型路径。
- 任务：在当前 Track 模型结构上蒸馏视觉学生，只优化 `VisionEncoder`。
- 保留：ST7-Opt3 UWB goal 小噪声、Opt2B `dynamic_tilt_risk`、当前 reward/地形/速度配置。
- 新增：每环境独立的连续块状深度空洞，保留散点 dropout、高斯噪声、旋转和纵向平移。

## 真实调用链

```text
conf/algo_conf_legged_robot_competition_26.toml
  -> agent_ppo.workflow.train_workflow.workflow
  -> Config.CURRENT.algorithm == "lbc_loco"
  -> agent_ppo.workflow.lbc_workflow.workflow
  -> feature.__getattr__("PolicyObservationProcess")
  -> LBCObservationProcess.process
     -> default_observation: proprio(45) + height_scan(256)
     -> GoalNoiseAugmenter.apply: shared noisy goal(3)
     -> nav_observation_utils.depth_camera_image: depth(180x320x1)
  -> AlgorithmLBC._split_obs
  -> teacher_encoder(height_scan) / VisionEncoder(depth + proprio)
  -> same frozen teacher_actor(proprio + latent + shared goal)
```

## 修改文件

- `agent_ppo/conf/conf.py`：正式入口切换到已有 `TrackLBCLocoConfig`。
- `agent_ppo/conf/train_env_conf_track_track_lbc_loco.toml`：复制 ST7-Opt3 完整 Track 配置，只追加 LBC、深度增强和已有标定相机配置。
- `agent_ppo/feature/lbc_observation_process.py`：在实际 LBC 观测链中生成一次 noisy goal，不覆盖 `proprio[6:9]`。
- `agent_ppo/feature/goal_noise.py`：复用 ST7-Opt3 实现，集中配置解析并导出监控指标。
- `agent_ppo/feature/policy_observation_process.py`：改用共用配置解析器，PPO 行为不变。
- `agent_ppo/feature/depth_block_dropout.py`：新增每环境、每块独立的持久状态机。
- `agent_ppo/feature/nav_observation_utils.py`：把块状空洞接到进入 `VisionEncoder` 前的真实深度链，并防止 `NaN/Inf`。
- `agent_ppo/workflow/lbc_workflow.py`：启动硬检查、shape/配置日志、教师权重周期差异检查和新监控数据。
- `agent_ppo/conf/monitor_builder.py`：增加 goal、depth 和 distillation 面板。
- `agent_ppo/agent.py`：修正 LBC 评估路径对已移除 `_build_actor_input` 的调用。

## 关键契约

- noisy goal 只在 `LBCObservationProcess` 中生成一次，拼入 flat observation 一次。Teacher/student 均从 `AlgorithmLBC._split_obs()` 的同一 `obs["goal"]` 取值。
- `env.goal_positions` 不写回；reward、termination 和 critic 继续使用 clean ground truth。
- 块状空洞状态为 `[num_envs, max_blocks]`，记录 top/left/width/height/TTL；通过 `episode_length_buf` 回退只清理 reset 环境。
- 遮挡面积使用实际 union mask 计算，每环境不得超过 15%。
- teacher encoder/actor 保持 `eval` 且 `requires_grad=False`；optimizer 参数必须与 `VisionEncoder` 参数集完全相等。
- 训练期周期检查 `teacher_max_abs_diff == 0`，环境终止时只清理对应 LSTM hidden/cell。

## D1 配置

- 散点 dropout：`0.05`
- 高斯噪声：`0.02`
- 旋转：`±5°`
- 纵向平移：`±6 px`
- 块生成帧概率：`0.40`
- 每环境块数：`1..2`
- 块持续：`2..4` 帧
- 最大实际遮挡面积：`15%`
- 整帧黑屏、旧帧保持、edge dropout、旧 square hole、blur：关闭

## 验证

```bash
python3 -m py_compile \
  agent_ppo/agent.py \
  agent_ppo/conf/conf.py \
  agent_ppo/conf/monitor_builder.py \
  agent_ppo/feature/depth_block_dropout.py \
  agent_ppo/feature/goal_noise.py \
  agent_ppo/feature/lbc_observation_process.py \
  agent_ppo/feature/nav_observation_utils.py \
  agent_ppo/feature/policy_observation_process.py \
  agent_ppo/workflow/lbc_workflow.py

git diff --check
```

已做 TOML 结构对照：移除 D1 专属段并恢复 `num_envs` 后，与 `train_env_conf_track_nav.toml` 的 `goal_noise`/reward/地形/速度配置完全一致。

本地 Python 环境不含 Isaac Lab/PyTorch，因此状态机多帧 tensor 验证和教师 checkpoint shape 验证必须在平台的 20–30 分钟短任务中完成。启动硬检查会在教师未加载、shape 不匹配或 optimizer 包含教师参数时直接报错。

## 短任务观测点

- 创建任务时明确选择 `595729` 对应的 ST7-Opt3-30min checkpoint。
- 确认日志的 workflow、stage、TOML、teacher source 和 shape 均正确。
- `goal_noise_*` 与 `depth_block_dropout_*` 必须非恒 0。
- `teacher_max_abs_diff` 始终为 `0`。
- `latent_mse` 总体下降，`cosine_similarity` 总体上升，`grad_norm` 无爆炸。
