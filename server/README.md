# server/

训练运行时目录。腾讯开悟平台上传/同步以本目录为项目根。

## 内容

| 路径 | 说明 |
|---|---|
| `agent_diy/` | DIY agent 框架（基线参考） |
| `agent_ppo/` | PPO + LBC + 行为蒸馏主代码（活动基线） |
| `conf/` | 算法/应用配置 + 同步脚本 `tongbu.py` |
| `isaac_env/` | Isaac Lab 环境封装 |
| `docs/` | 实验笔记（navj8/j9、st7-opt2a/2b/3 等） |
| `kaiwu.json` | 开悟平台项目元信息 |
| `local_sync_client.py` | 本地-平台同步客户端（同步 `agent_diy`/`agent_ppo`/`conf`） |
| `train_test.py` | 训练入口 |
| `.vscode/launch.json` | 调试配置（用 `${workspaceFolder}`，打开 `server/` 即可生效） |

## 活动基线

- **起始基线**：`codex/st7-opt3`（提交 `9b0b3df`，ST7 Opt3 UWB goal noise）
- **当前活动入口**：`StandardDistill3ActionConfig`（STD-D3A），必须从 D2-40min 视觉学生续训；在 D2 楼梯分布上把 raw-action 模仿权重提高到 1.0，并使用教师驱动的干净轨迹修复上下楼动作对齐。
- `StandardDistill1Config` 与 `StandardDistill2StairConfig` 均保留为独立可复现阶段；D3A 不修改网络、latent、LSTM、教师或相机外参。
- 301-D 扁平旧模型的行为桥接仅作为 `StandardRefDistillConfig` 备用；ST9-Opt3-D2 的动作感知、闭环 DAgger 与时序实现继续保留。
- TrackNav 保留 Opt3 基线 + J9 通用学习率接口修复 + Opt4 角速度保护；`navopt5debug` 与 `navopt5b` 作为独立可复现实验阶段保留，不是默认入口。
- 含 ST7-Opt2B `dynamic_tilt_risk`、ST7-Opt3 `goal_noise`、行为蒸馏/LBC 机制（在 `codex/st7-opt2a` 并入主干）。

## 运行约定

- **必须从 `server/` 目录运行**。`local_sync_client.py` 与 `conf/tongbu.py` 以 cwd / `--root` / `IDE_SYNC_ROOT` 为相对根，同步 `agent_diy`/`agent_ppo`/`conf`。
- `.vscode/launch.json` 使用 `${workspaceFolder}/train_test.py`--用 VS Code 打开 `server/` 即自适应，无需改路径。
- 不引入指向 `../shared/` 或 `../archive/` 的运行时引用。

## 变更记录

见 [`CHANGELOG.md`](./CHANGELOG.md)。
