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

- **当前功能分支入口**：`StandardBridgeR1Config`（`STD-BRIDGE-R1`），从原始
  Standard 10288 flat301 教师开始，用一次 5000-iteration 任务完成
  0/25/50/75/100% 逐环境 DAgger，产出不可部署的 Actor77 特权教师。
- R1 由操作者手动确认预训练教师身份，不做字节级 SHA 门禁；仍校验教师
  key/shape、成功加载与冻结状态。命令和地形分布保持不变，首轮关闭随机化/
  噪声/push；它不读取
  depth、不执行 PPO，也不修改 Goal/UWB/Track/部署端。
- R1 已在 `conf/configure_app.toml` 明确启用预加载，首轮默认 ID 为 `10288`、
  目录为 `/data/pre_model/ckpt`。平台任务页若已经注入所选父模型则直接使用；
  否则 workflow 从该目录加载最新的结构兼容 checkpoint。workflow 不写死 ID，
  因而同一代码也能恢复后续 `behavior_distill_v2` checkpoint。
- `StandardDistill1Config` 到 D5 继续作为已经尝试过的视觉路线保留；它们不是
  R1 的父 checkpoint。D1 需要 encoder-based bridge artifact，不能直接加载原始
  flat301 文件。
- R1 训练恢复格式为 `behavior_distill_v2`；后续视觉 LBC 教师格式为
  `privileged_loco_teacher_v1`；二者均不能交给当前部署导出器。
- R1 的平台文件 ID 为 `10288 + current_iteration`：第一轮保存 `10289`，最终
  保存 `15288`。内部 DAgger iteration 仍为 `1--5000`；当前每 100 iteration
  常规定时保存一次，这不是额外划分的“100 轮恢复阶段”；每个 DAgger 阶段
  边界额外发布 `bridge`/`teacher` 候选，质量阈值只记录 warning，不会停训。
- TrackNav 保留 Opt3 基线 + J9 通用学习率接口修复 + Opt4 角速度保护；`navopt5debug` 与 `navopt5b` 作为独立可复现实验阶段保留，不是默认入口。
- 含 ST7-Opt2B `dynamic_tilt_risk`、ST7-Opt3 `goal_noise`、行为蒸馏/LBC 机制（在 `codex/st7-opt2a` 并入主干）。

## 运行约定

- **必须从 `server/` 目录运行**。`local_sync_client.py` 与 `conf/tongbu.py` 以 cwd / `--root` / `IDE_SYNC_ROOT` 为相对根，同步 `agent_diy`/`agent_ppo`/`conf`/`isaac_env`。
- `.vscode/launch.json` 使用 `${workspaceFolder}/train_test.py`--用 VS Code 打开 `server/` 即自适应，无需改路径。
- 不引入指向 `../shared/` 或 `../archive/` 的运行时引用。

同步前可先做完全离线检查：

```bash
python local_sync_client.py --check-local
```

Cookie 失效时用 `--refresh-cookie` 强制跳过缓存和旧兼容值重新录入；
`--dry-run` 仍会连接腾讯代理并读取 `/health`、`/manifest`，但不写远程。
当前在线状态必须标记为“待刷新 Cookie 验证”，不能由离线检查推断已经可同步。

R1 的冻结计划、必要安全检查和质量告警见
[`../shared/分析记录/2026-07-23_STD-BRIDGE-R1执行卡.md`](../shared/分析记录/2026-07-23_STD-BRIDGE-R1执行卡.md)。

## 变更记录

见 [`CHANGELOG.md`](./CHANGELOG.md)。
