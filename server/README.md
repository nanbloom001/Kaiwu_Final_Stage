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
| `model_chunk_uploader.py` | 大模型断点分片上传工具；逐片与整文件 SHA256 校验后在开发容器原子重组 |
| `train_test.py` | 训练入口 |
| `.vscode/launch.json` | 调试配置（用 `${workspaceFolder}`，打开 `server/` 即可生效） |

## 活动基线

- **当前功能分支入口**：`P2NavPPOConfig`（`p2_nav_ppo`），任务名 `p2nav2hsafedir`。
  它显式从最新验证通过的完整三轴 `p2nav10hvyavoid2` 包做
  `p2_safe_direction_continue_warm_start`：保留冻结低层、NavigationEncoder、三轴 Actor/LSTM、
  Critic、ResponseAdapter、return statistics 与兼容 Adam moments，只新增 training-only
  `NavigationSafetyHead`。平台请求 ID、包内 ID 和 lineage 不一致只告警；真正的硬错误仅限
  文件缺失、反序列化失败、必需模块/spec/shape 不兼容或非有限状态。
  Track 固定为 `pyramid_slope_inv -> pyramid_stairs_inv -> open_entry_maze`，累计有效训练
  本轮 session 目标为 7200 秒，lifetime 继承父包。terrain curriculum 关闭，128 环境静态近似
  均匀分布在 20 列（difficulty=`col/20`）；reset 保持 column，出生 row 在两个非迷宫段随机。
- P2 policy observation 为 57905，worker wire 为
  `critic323 | response_aux30 | diagnostic_aux32 = 385`；前 30 槽保持 ResponseAdapter
  合同，后 32 槽携带 1.5 秒四足窗口指标、terminal 前 row/column/goal-distance 快照、最近
  0.2 秒非足端最大接触力、按世界 X 计算的实时赛段及 gait/collision 映射有效位。步态使用
  `contact_forces` 的 sensor-local 足端列，足端速度才使用 Articulation body ID；名称或 shape
  映射异常时两项奖励自动归零并告警，不形成训练硬门禁。
  高层输入为 `nav_feat32 + nav_nonvisual36 + response_profile16 + confidence1 = 85`。
  rollout depth 在高层 tick 起点立即取得 CPU pinned FP16 独立所有权，低层 PPO storage、
  S0 anchor 和第二套 Camera 均不创建。首选 128 环境，显存不足时先降 CNN microbatch，
  96/80 环境只能通过新任务重启。
- P2 导航 reward-v9 只在 5 Hz transition 边界结算：terminal-consistent frontier potential、
  一次性成功/失败/超时、时间、crawl、command-rate、仿真真值 tracking、非足端碰撞、停滞与
  部署可得三方向深度风险。新增的 `missed_safe_direction` 使用 training-only scanner/height
  教师，只在存在明显更安全替代方向时产生负值，权重在两小时内从 0 渐进到 `-0.03`；教师不含
  goal/heading，SafetyHead 也不参与评估或部署。不奖励“碰撞后恢复”过程；
  绝对目标距离、速度突降、heading/地形门控和 Adapter 输出均不进入 PPO reward。50 Hz body contact 改为零权重监控，避免与
  5 Hz collision 重复处罚。
- 两小时 optimizer 课程按本轮有效训练秒数执行：0-30 分钟 CNN 0.15×、Actor 0.10×、
  SafetyHead 1.0×；30 分钟至 1.5 小时为 0.25×/0.15×/1.0×；最后 30 分钟为
  0.15×/0.10×/0.5×。`vy` 从首个 rollout 保持可信核心 `|vy|<=0.20` 和探索硬边界
  `|vy|<=0.40`。
  CNN 仅在 rollout 边界切换，exact resume 校验 reward/training contract、阶段、LR、entropy、
  return statistics、gait baseline 和独立 RNG。
- P2 checkpoint 标签按阶段为 `safewarm`/`safefull`/`safestable`；加载器优先级为
  `safestable > safefull > safewarm > navfull > vyadapt > vywarm > navadapt > navwarm`。
  包保存完整高层、冻结低层、
  三套 optimizer/scheduler、独立 main-action/vy-action/shuffle/neutral/Adapter RNG、completed response
  records 与 curriculum 诊断；每个可学习 leaf 固定携带 `class_name/spec/state_dict`，包保持
  `deployable=false`。首存约 5 分钟，之后按有效训练墙钟每 10 分钟无参数调用
  `agent.save_model()`，正常结束与 SIGTERM 共享幂等 final save。
  P1.5 父包的低层动作方差与 S0 anchor 会迁移进规范 low-level leaf，P2 不训练它们但也
  不丢弃，供后续独立 `resume_low` 使用。Standard 评估只能在显式低层模式下抽取规范
  `modules.low_level.locomotion_encoder/actor`；默认 Camera loader 仍拒绝忽略完整高层。
- P2 worker wire 的 aux24/25 分别携带 reset boundary 与终止原因。terminal outcome 使用
  aux55/56/57 的 reset 前 column/row/goal-distance 归因，reset 后 row/column 只计下一 episode
  起点。平台若丢失公开 timeout，
  aisrv 从该字段恢复；当前无 terminal critic observation 时 timeout 不使用 reset 后状态
  bootstrap。`p2_nav_eval` 只装配推理模块，不创建 Critic、optimizer、scheduler 或训练 buffer。
- P2 Track 评估使用独立的 `p2_nav_eval` 推理装配。平台
  `tools/eval/conf/eval_env_conf.toml` 若显式传入 `p2_nav_ppo`，评估侧会自动提升为
  `p2_nav_eval`；若平台没有转发 `policy_entry`，Track+Camera 会保留当前 P2 bootstrap
  lineage，而不是回退到旧 `nav_eval`。有效评估必须同时看到 `Stage: p2_nav_eval`、
  `eval-only assembly initialized` 和同 ID `navwarm`/`navadapt`/`navfull` 文件加载成功；
  任一证据缺失时分数无效。旧 DAgger/Nav 分支仍保持 `nav_eval`，两种高层契约不混载。
- 平台任务页控制实际 4 小时时长，workflow 首次约 5 分钟、之后按有效训练时间每 10 分钟
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
  `responsebase`/`responseexpand`/`responsefull`/`responsecalib`；P2 使用
  `navwarm`/`navadapt`/`navfull`。
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

大 checkpoint 不应塞进常规源码同步清单。使用独立分片工具上传到受限临时目录：

```bash
python3 model_chunk_uploader.py \
  /absolute/path/model.pkl \
  --remote-path agent_ppo/test_artifacts/model.pkl \
  --no-cookie-prompt
```

默认按 4 MiB 分片，使用 `2 × 4 = 8` 个最大并发请求；每片和重组后的整文件都校验
SHA256。相同片段可断点续传，目标文件只在完整校验后通过原子替换出现，成功后默认删除分片；
需要保留分片排障时显式传 `--keep-parts`。远端目标被限制在
`agent_ppo/test_artifacts/`，避免覆盖训练代码、平台配置或凭据。

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
