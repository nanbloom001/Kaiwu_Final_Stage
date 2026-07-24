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
| `local_sync_client.py` | 本地-平台同步客户端（同步 `agent_diy`/`agent_ppo`/`conf`/`isaac_env`） |
| `train_test.py` | 训练入口 |
| `.vscode/launch.json` | 调试配置（用 `${workspaceFolder}`，打开 `server/` 即可生效） |

## 活动基线

- **当前功能分支入口**：`StandardVisualPPOConfig`
  （`visual_policy_optimization`），从冻结的
  `model.ckpt-visionfull-28401.pkl` 初始化深度视觉 Encoder 与 Actor77，
  使用 recurrent PPO 改善停止、低速、转向和 command 跟踪。
- 单次平台任务按墙钟执行 30 分钟 Critic 预热、60 分钟 Actor 微调和
  90 分钟 LSTM/output-head 微调；CNN、S0 anchor 和部署接口始终冻结。任务页
  控制实际 3 小时时长，workflow 约每 10 分钟请求一次 checkpoint。
- Actor 只使用 `proprio45 + depth180x320x1 + LSTM state`；训练 Critic 可使用
  `height_scan` 等特权状态。Camera 评估只加载视觉 Encoder 与低层 Actor，
  不加载 Critic 或 S0 anchor。
- 阶段 2 的 R2（`daggerfull-16288`）和阶段 4 的视觉蒸馏
  （`visionfull-28401`）已作为父阶段合入 `main`。R2 仍是不可部署的
  height-scan 教师，阶段 4 的 `visionfull-28401` 是当前视觉回滚基线。
- D1–D5 是已被阶段 4 重建路线取代的历史实验。现存历史 TOML、测试和归档
  Tag 仅用于复盘，不是当前入口；其 checkpoint、奖励、学习率、锚定衰减和
  训练调度不得作为当前阶段参数来源。
- 所有训练恢复包继续使用 `kaiwu_train_v1`，且
  `capabilities.deployable=false`；当前部署导出器只接受单独审查生成的
  `lbc_loco` 制品。
- 文件数字 ID 完全由平台注入。Stage 5 保存
  `rlcritic`/`rlactor`/`rlfull` 纯字母标签，不能从 iteration 人工计算 ID。
- TrackNav 保留 Opt3 基线 + J9 通用学习率接口修复 + Opt4 角速度保护；`navopt5debug` 与 `navopt5b` 作为独立可复现实验阶段保留，不是默认入口。
- 含 ST7-Opt2B `dynamic_tilt_risk`、ST7-Opt3 `goal_noise`、行为蒸馏/LBC 机制（在 `codex/st7-opt2a` 并入主干）。

## 运行约定

- **必须从 `server/` 目录运行**。`local_sync_client.py` 与 `conf/tongbu.py` 以 cwd / `--root` / `IDE_SYNC_ROOT` 为相对根，同步 `agent_diy`/`agent_ppo`/`conf`/`isaac_env`。
- 同步服务和本地客户端必须通过 `IDE_SYNC_TOKEN` 或客户端 `--token` 使用同一个随机共享值；仓库不保存 Token 或网页 Cookie。浏览器代理 Cookie 只通过环境变量、CLI、本地缓存或交互输入提供。
- `.vscode/launch.json` 使用 `${workspaceFolder}/train_test.py`--用 VS Code 打开 `server/` 即自适应，无需改路径。
- 不引入指向 `../shared/` 或 `../archive/` 的运行时引用。

同步前可先做完全离线检查：

```bash
python local_sync_client.py --check-local
```

Cookie 失效时用 `--refresh-cookie` 强制跳过缓存和旧兼容值重新录入；
`--dry-run` 仍会连接腾讯代理并读取 `/health`、`/manifest`，但不写远程。
当前在线状态必须标记为“待刷新 Cookie 验证”，不能由离线检查推断已经可同步。

当前阶段的 S0 冻结、三小时训练和 S0/S1 验收见
[`../shared/分析记录/2026-07-24_Standard视觉学生独立化与发布冻结计划.md`](../shared/分析记录/2026-07-24_Standard视觉学生独立化与发布冻结计划.md)。

## 变更记录

见 [`CHANGELOG.md`](./CHANGELOG.md)。
