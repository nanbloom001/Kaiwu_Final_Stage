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

- **当前功能分支入口**：`StandardRefDistillConfig`（`STD-DAGGER-R2`），从原始
  Standard 10288 flat301 教师重新开始，用一次 6000-iteration、3 小时任务自适应
  完成 0/25/50/75/100% 逐环境 DAgger，产出不可部署的 Actor77 特权教师。
- R2 由操作者手动确认预训练教师身份，不做字节级 SHA 门禁；仍校验教师
  key/shape、成功加载与冻结状态。命令和地形分布保持不变，首轮关闭随机化/
  噪声/push；它不读取
  depth、不执行 PPO，也不修改 Goal/UWB/Track/部署端。
- 平台任务页选择原始 10288；未提前注入时 workflow 从
  `/data/pre_model/ckpt` 显式加载 10288。后续选择 `kaiwu_train_v1` 文件可恢复
  optimizer、DAgger 阶段、reservoir 和 RNG。
- `StandardDistill1Config` 到 D5 继续作为已经尝试过的视觉路线保留；它们不是
  R1 的父 checkpoint。D1 需要 encoder-based bridge artifact，不能直接加载原始
  flat301 文件。
- R2 训练恢复格式统一为 `kaiwu_train_v1`；视觉和高层模块以后在同一 schema
  的 `modules` 下扩展。训练包不能交给当前部署导出器。
- 文件数字 ID 完全由平台注入，不再用 `10288 + iteration` 计算。每 100
  iterations 保存阶段英文文件和同 ID 的 `locomotion` 评估别名；阶段最大预算
  到期时从最佳点回退后强制晋升并写入 warning。
- TrackNav 保留 Opt3 基线 + J9 通用学习率接口修复 + Opt4 角速度保护；`navopt5debug` 与 `navopt5b` 作为独立可复现实验阶段保留，不是默认入口。
- 含 ST7-Opt2B `dynamic_tilt_risk`、ST7-Opt3 `goal_noise`、行为蒸馏/LBC 机制（在 `codex/st7-opt2a` 并入主干）。

## 运行约定

- **必须从 `server/` 目录运行**。`local_sync_client.py` 与 `conf/tongbu.py` 以 cwd / `--root` / `IDE_SYNC_ROOT` 为相对根，同步 `agent_diy`/`agent_ppo`/`conf`/`isaac_env`。
- 同步服务和本地客户端必须通过 `IDE_SYNC_TOKEN` 或客户端 `--token` 使用同一个随机共享值；仓库不保存 Token 或网页 Cookie。浏览器代理 Cookie 只通过环境变量、CLI、本地缓存或交互输入提供。
- 如需用文件保存本机/IDE 的凭据，复制 [`server/.env.example`](./.env.example) 为**未跟踪**的 `server/.env`，并在运行时显式传入 `--env-file .env`。本地与 IDE 各自保留一份同 Token 的私有文件；执行 `chmod 600 .env`，不要让同步客户端上传该文件。优先级为 CLI 参数、进程环境变量、`.env` 文件。
- `.vscode/launch.json` 使用 `${workspaceFolder}/train_test.py`--用 VS Code 打开 `server/` 即自适应，无需改路径。
- 不引入指向 `../shared/` 或 `../archive/` 的运行时引用。

同步前可先做完全离线检查：

```bash
python local_sync_client.py --check-local
```

Cookie 失效时用 `--refresh-cookie` 强制跳过缓存和旧兼容值重新录入；
`--dry-run` 仍会连接腾讯代理并读取 `/health`、`/manifest`，但不写远程。
当前在线状态必须标记为“待刷新 Cookie 验证”，不能由离线检查推断已经可同步。

R2 的完整调度、污染保护和验收见
[`../shared/分析记录/2026-07-23_STD-DAGGER-R2执行卡.md`](../shared/分析记录/2026-07-23_STD-DAGGER-R2执行卡.md)。

## 变更记录

见 [`CHANGELOG.md`](./CHANGELOG.md)。
