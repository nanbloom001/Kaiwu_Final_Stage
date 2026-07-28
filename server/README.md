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
| `local_sync_client.py` | 本地-平台同步客户端（同步 `agent_diy`/`agent_ppo`/`conf`/`isaac_env`，但显式排除平台拥有的 `isaac_env/base_env.py`） |
| `train_test.py` | 训练入口 |
| `.vscode/launch.json` | 调试配置（用 `${workspaceFolder}`，打开 `server/` 即可生效） |

## 活动基线

- **当前功能分支入口**：`P15ResponseConfig`（`p15_response`），任务名
  `p15resp8h`。它从 `commandfull-34728` 恢复完整低层 PPO 状态，CNN 继续冻结，
  前 7 小时联合训练 Actor/LSTM/Critic 与独立 ResponseAdapter，最后 1 小时只校准
  Adapter；S0 action/latent anchor 固定为 `0.35/0.10`。worker 传输 346 维
  privileged wire，aisrv 在进入 PPO 前拆成 Critic316 与独立 aux30。
  Adapter rollout 为 80 帧，future horizon 不跨低层 optimizer update；GRU 使用 8 帧
  burn-in 和逐环境 episode reset。Terrain 使用静态 0-9 难度覆盖，原生距离 curriculum
  保持关闭。
- 本任务的 P0 是评估入口闭环：平台最终的
  `tools/eval/conf/eval_env_conf.toml` 必须显式设置
  `policy_entry = "visual_policy_optimization"`，并由 aisrv 和 learner 同时记录
  VisualPPO stage/loader。P0 smoke 未取得前，任何评估分数都只是入口诊断数据，不能
  用于选择模型或宣称本任务可用。
- 平台任务页控制实际 8 小时时长，workflow 首次约 2 分钟、之后按墙钟约每 10 分钟
  请求一次 checkpoint，并在课程边界额外保存；
  `task_end_hours` 只是训练包元数据，`max_iterations` 只是高安全上限。训练恢复包使用
  `responsebase`/`responseexpand`/`responsefull`/`responsecalib` 纯字母标签，仍为
  `kaiwu_train_v1` 且 `deployable=false`。
- Actor 只使用 `proprio45 + depth180x320x1 + LSTM state`；训练 Critic 可使用
  `height_scan` 等特权状态。Camera 评估只加载视觉 Encoder 与低层 Actor，
  不加载 Critic 或 S0 anchor。
- 阶段 2 的 R2（`daggerfull-16288`）和阶段 4 的视觉蒸馏
  （`visionfull-28401`）已作为父阶段合入 `main`。R2 仍是不可部署的
  height-scan 教师，阶段 4 的 `visionfull-28401` 是当前视觉回滚基线。
- D1–D5 和 TrackNav 是已退出活动 loader 的历史实验。对应不可达 TOML 与仅服务
  这些配置的测试已移除；复盘依据保留在 Git 历史、Changelog、文档和归档 Tag，
  其 checkpoint、奖励、学习率、锚定衰减和训练调度不得作为当前阶段参数来源。
- 所有训练恢复包继续使用 `kaiwu_train_v1`，且
  `capabilities.deployable=false`；当前部署导出器只接受单独审查生成的
  `lbc_loco` 制品。
- 文件数字 ID 完全由平台注入，不能从 iteration 人工计算。Anchor R2 的
  `anchor*` 与上一阶段的 `command*` 标签只用于父阶段；P1.5 使用
  `responsebase`/`responseexpand`/`responsefull`/`responsecalib`。
- `response_observation32` 的实时速度只使用 SportMode `vx/vy` 与 IMU `wz`；UWB 不进入
  0.2/0.6/1.0 秒 Adapter 输入。跨任务 resume 只恢复已完成 records，未完成 future
  history 始终清空。Standard 读取 response 包必须显式启用 `low_level_only_preload`。
- TrackNav 历史实验不再提供活动 `StageConfig` 或可直接启动的 TOML；需要恢复时
  必须从 Git 历史新建独立实验分支并重新验证。
- 含 ST7-Opt2B `dynamic_tilt_risk`、ST7-Opt3 `goal_noise`、行为蒸馏/LBC 机制（在 `codex/st7-opt2a` 并入主干）。

## 运行约定

- **必须从 `server/` 目录运行**。`local_sync_client.py` 与 `conf/tongbu.py` 以 cwd / `--root` / `IDE_SYNC_ROOT` 为相对根，同步 `agent_diy`/`agent_ppo`/`conf`/`isaac_env`；二者都会拒绝上传或删除平台拥有的 `isaac_env/base_env.py`。
- 同步服务和本地客户端必须通过 `IDE_SYNC_TOKEN` 或客户端 `--token` 使用同一个随机共享值。本地客户端会在 CLI 参数与已导出的环境变量均未提供时，自动读取 `<同步根>/conf/.env` 中的 `IDE_SYNC_TOKEN`；优先级为 `--token`、环境变量、`.env`。仓库不保存 Token 或网页 Cookie。浏览器代理 Cookie 只通过环境变量、CLI、本地缓存或交互输入提供。
- `.vscode/launch.json` 使用 `${workspaceFolder}/train_test.py`--用 VS Code 打开 `server/` 即自适应，无需改路径。
- 不引入指向 `../shared/` 或 `../archive/` 的运行时引用。

同步前可先做完全离线检查：

```bash
python local_sync_client.py --check-local
```

Cookie 失效时用 `--refresh-cookie` 强制跳过缓存和旧兼容值重新录入；
`--dry-run` 仍会连接腾讯代理并读取 `/health`、`/manifest`，但不写远程。
当前在线状态必须标记为“待刷新 Cookie 验证”，不能由离线检查推断已经可同步。

开发容器需要 GPU/PyTorch 诊断时，可在 IDE 中重启新版 `conf/tongbu.py` 后使用：

```bash
python3 container_rpc_client.py --cwd . "nvidia-smi"
```

该命令复用 `IDE_SYNC_TOKEN` 和 Cookie 缓存；远端 `cwd` 必须位于项目根目录，
并受超时和输出大小限制。它只用于开发容器诊断，不用于启动或管理平台训练任务。

hier-nav 需要验证真实 preload、rollout、TBPTT update 和 checkpoint lifecycle 时，
在开发容器的 `server/` 根目录运行可复用完整 smoke：

```bash
python3 -m agent_ppo.tools.nav_full_smoke start --num-envs 8
python3 -m agent_ppo.tools.nav_full_smoke status
python3 -m agent_ppo.tools.nav_full_smoke stop
```

启动器不设置 `KAIWU_TRAIN_TEST`，也不修改生产 TOML；`num_envs` 只通过
`NAV_FULL_SMOKE_NUM_ENVS` 在加载后的内存配置中覆盖。默认在首个包含有效样本的
TBPTT 更新后对独立进程组发送 SIGTERM，并将状态、stdout 和原子 JSONL 事件写入
`/tmp/kaiwu_nav_full_smoke/`。运行前必须确保 `configure_app.toml` 指定的父包已经
挂载到开发容器预加载目录。若事件只有 `first_update_skipped` 而没有
`first_update_complete`，说明目标标签无效，不能判定 smoke 成功。

当前命令泛化的 P0、四小时日程、保存/恢复和验收见
[`../shared/分析记录/2026-07-25_Standard命令泛化下半四小时实施计划.md`](../shared/分析记录/2026-07-25_Standard命令泛化下半四小时实施计划.md)。
P1.5 八小时扩域与响应器实现见
[`../shared/分析记录/2026-07-28_P1.5连续指令扩域与响应器八小时实施记录.md`](../shared/分析记录/2026-07-28_P1.5连续指令扩域与响应器八小时实施记录.md)。
Anchor R2 只作为父阶段记录，见
[`../shared/分析记录/2026-07-25_StandardAnchorR2四小时实施计划.md`](../shared/分析记录/2026-07-25_StandardAnchorR2四小时实施计划.md)。

## 变更记录

见 [`CHANGELOG.md`](./CHANGELOG.md)。
