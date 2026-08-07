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

- **当前 P4 Maze 即时命令修复入口**：`P4NavPPOConfig`（`p4_nav_ppo`），分支
  `codex/p4-maze8h-instant-command-r4-inputfix`，任务 `p4maze8h-instant-r4-inputfix`。父包固定为
  `p4maze8h10hz_1416926` 的 `mazefinal` checkpoint；128 env、10Hz 高层、50Hz 低层、
  32-tick rollout、TBPTT16、4 PPO epochs，单段 `open_entry_maze`、20 列、课程关闭、
  120 秒 episode，完整有效训练 `28800s`，平台任务墙钟 `8.25h`。完整合同见
  [`docs/p4-maze8h-instant-command-r4.md`](./docs/p4-maze8h-instant-command-r4.md)。
- 本轮固定 `instant_hold_10hz`：每个高层 tick 的三轴 policy target 只做有限值和硬范围处理后立即
  成为 exec command，并在随后的 5 个低层帧保持；不执行 slew、零交叉、反向确认、运行时 limiter、
  near-goal rewrite 或 recovery override。checkpoint 与 exact resume 必须记录并恢复这一合同，旧
  slew P4 包只能 warm start，不能按 R4 exact resume。
- Actor85 的 capability15 末六维保持父模型训练时的变化率语义
  `up=[0.30,0.30,1.00]`、`release=[0.30,0.60,2.50]`；instant controller 的真实物理变化率
  `[10,6,18]` 只写 command contract 和诊断，不能直接写入未归一化 Actor 输入。父 Actor anchor
  仅约束旧 slew 在一个 10Hz tick 内可达且不反向的样本，避免把旧 target 动力学强加给瞬时执行策略。
- 低层、NavigationEncoder、SafetyHead 和 StuckHead 全程冻结。0-30 分钟只校准 Critic；30 分钟至
  2 小时低学习率校准 Actor 与 Adapter；2-3 小时继续 Actor 修正；3-3.5 小时收敛；根据上一轮约
  3h50 已开始同步劣化的证据，3.5 小时后固定冻结 Actor/Teacher/Anchor/Adapter，只以 `1e-5`
  更新 Critic直到 8 小时结束。父 Actor 分布作为不可变 training-only anchor，仅在低风险样本约束
  漂移，不进入评估或部署输入。
  模型 ID 只用于候选选择，文件、模块 spec、shape、有限值和冻结低层 digest 才决定兼容性。
- Maze reward 使用不可重复的历史最短距离 credit：`1.0/m`、累计上限 `+6`；failure、timeout 和
  reason4 精确回收该 episode 已发放 credit，success 不回收。
  success/failure/timeout/reason4 为 `+200/-60/-40/-75`。reason4 在 10 秒墙面接触与空间受限后
  真实 reset，按 failure 统计、双 GAE mask 为零、不得计完成。持续卡墙从 0.8 秒开始，2 秒达到
  `-0.04/tick`；碰撞 onset 为 `-0.16-0.24*severity`，持续接触为 `-0.06/tick`。
- 五方向教师只在训练期监督 Actor mean，在多个近似安全出口中以 Goal 作 tie-breaker，不改 Actor85
  或部署接口。全局 yaw-cancellation、near-goal capture 和平移 limiter 全程 shadow，实际命令不再
  被 Actor 采样后的隐式机制改写；部署可得 predictive collision 与五方向 Actor mean 教师承担学习
  信号。额外 Camera/Goal fault、Push、五段出生和低层联合训练关闭。新 checkpoint 标签为
  `instantwarm/instantadapt/instantcorrect/instantstable/instantfrozen`，旧 `loop*`/`closed*`/`credit*`
  仅保留发现和 warm-start 兼容性。

- **历史 P4 五段全赛道入口**：`P4NavPPOConfig`（`p4_nav_ppo`），分支
  `codex/p4-full8h-r2`，任务 `p4full8h-r2`，父包
  `p4maze8h10hz_1416926`。ZIP SHA256 为
  `106908c8830f4fc7989125372f6ed397366f7ca1b4aac071128dccf71add0f6c`，checkpoint
  SHA256 为 `0bf54e3c18e6450492d7c1d44d69596d79beddbebb57166432220f8cb93dd2e4`。
  训练为 128 env、10Hz 高层、冻结低层、完整 `28800s`，固定五段
  `slope -> slope_inv -> stairs -> stairs_inv -> maze`，20 列、课程关闭、120 秒 episode。
  70/60/75% episode 从整条赛道起点开始；其余 episode 按 15/15/20/20/30% 分配至五段，
  四个段内分位与 70/30 safe/hard 位置训练。困难位置必须经运行时 raycast 验证；不可验证时
  fail-closed 到同段入口，不能把“请求了随机出生”当作已经覆盖。
- 目标输入 shape 不变：P4 训练在 `d<=10m` 与旧包逐值一致，`d>10m` 改用单位方位，距离仍为
  `min(d/20,1)`。训练期 GoalBelief 使用距离相关、零均值椭圆跳变，径向最大 0.20m、切向最大
  1.0m，并在近终点平滑衰减。部署端 `UwbGoal` 已增加显式编码版本，但稳定 Actor80 调用默认仍
  使用 legacy 逐轴 clamp；本轮 checkpoint 为 `deployable=false`，未来只有匹配 P4 v2 spec 的制品
  才能启用方向保持版本。
- reason 4 使用 10 秒滑动空间窗口、非足端墙接触、真实速度和 Goal 距离确认，主动 reset
  结算 `-25`，不计成功且不与碰撞/安全惩罚重复。前两次回到同一出生桶，第三次退回同段入口，
  防止通过更靠后出生刷完成率。worker 的 quota/RNG 没有公开 checkpoint transport，fresh-process
  resume 会从确定性 seed 重新初始化出生调度；模型、optimizer、训练时钟可 exact resume，但不得
  宣称 worker 出生序列位级恢复。
- P4 正式训练要求 reset spawn hook 安装成功，active 卡滞 term 最多等待 8 个真实帧完成装配；超时、
  readback 不一致或物理 root-state 写入失败均明确停止训练。同帧 wall-stuck 优先于 success，防止
  reason4 reset 被计为完成。出生比例面板只在 reset edge 计数，难度列与 safe/hard 位置分开报告。
- 五段 frontier 使用 terminal-clawed potential，长期不产生净刷分；全局路程惩罚关闭。开阔直行
  只在 teacher 证明 `slope/slope_inv` 前方安全、Goal 新鲜、远离边界且非 junction/dead-end/contact/
  recovery 时生效，总下限 `-0.0125/tick`，不约束楼梯和 Maze 的必要绕行。Camera 仅 nominal/light，
  Push 关闭。训练 wire 为 `519 = P3 493 + P4 tail26`；eval wire 与 Actor85/三轴动作不变。

- **历史 P4 Maze 入口**：`P4NavPPOConfig`（`p4_nav_ppo`），分支
  `codex/p4-maze3h-recovery`，任务 `p4maze3h-recovery-r2`。父包为
  `p4maze8h10hz_1416926`，checkpoint SHA256
  `0bf54e3c18e6450492d7c1d44d69596d79beddbebb57166432220f8cb93dd2e4`。128 env、75 秒 episode、
  单段 `open_entry_maze`、20 个静态难度列、课程关闭，完整训练 `10800s`；平台墙钟 `3.15h`
  只额外覆盖 rollout、保存和正常退出。模型 ID和标签仅用于候选定位，结构/spec/shape/有限值才是
  加载依据。完整合同见 [`docs/p4-maze3h-recovery.md`](./docs/p4-maze3h-recovery.md)。
- 网络和部署接口不变：低层 57901/Actor77/action12、高层 Actor85，低层全程冻结；高层保持
  10Hz、32 tick rollout 和 TBPTT16。新增 training-only `Actor LSTM hidden -> stuck_logit`、
  rollout-time Actor mean 教师和从 rollout 内真实 reset 起点构造的零-hidden TBPTT16 中约 10%
  sequence mirror，均不增加 Actor observation，也不进入 eval；任意固定 PPO chunk 的中途非零
  recurrent context 不做伪镜像。
  Camera clean/live 辅助仍可更新 NavigationEncoder；privileged safe3 教师只更新 Actor/LSTM/head。
- 三轴 policy mapper 保持 `vx=[0,1.0]`、`vy=+-0.30`、`wz=+-0.90`。slew 改为
  `vx 0.30/0.30`、`vy 0.40/0.80`、`wz 1.50/3.00`，继续先归零再反向。部署可得 predictive risk
  同比例缩放完整 `(vx,vy)`、不压缩 `wz`；解除风险时 alpha 每个 10Hz tick 最多恢复 0.15。
- 近终点 `0.65-1.20m` 且 Goal 新鲜、策略方向朝向目标时只限制平移速度，不改方向、不归零、不增加
  距离正奖励。predictive、missed-safe、yaw cancellation、yaw-exit 和 goal-safe preference 共用
  原 5Hz 等效 `-0.06` 安全预算，reason 4 卡墙 terminal 单独结算 `-15` 且不计完成。
- 卡墙确认合同由 7 秒延长为 10 秒。训练标签同时要求 policy/exec 运动意图、低 true motion、墙面
  证据和 mapping valid。当前平台没有经过验证的 aisrv 到 worker 高层运动意图 transport，因此正式
  配置保持 `shadow`：worker 继续报告位移、真值速度、墙面证据和候选时长，但绝不把主动停车误判成
  物理 reset。training wire 为 `509 = P3 493 + raw goal2 + stuck diagnostics13 + raw wall term1`；只有未来公开 hook
  提供并通过 readback 后才能切到 active。

- **当前 P3 功能分支入口**：`P3StandardJointConfig`（`p3_standard_joint`），分支
  `codex/p35-gaitfix2h`，任务 `p35gaitfix2h`。128 env、25 秒 episode、课程关闭，总计
  7200 秒；父包固定为上一轮任务 235689 的最终 `stairfinal` checkpoint，模型 ID `1013548`。
  父包下载加载后必须补录文件 SHA256 和模块 digest。完整 policy observation 仍为 57905，低层输入 57901、
  Actor77、动作 12、高层输入 85 均不变。
- 仅低层 LSTM、RNN output、最终 action head、Critic 与 ResponseAdapter 更新。低层 CNN、
  Actor body/std、高层 NavigationEncoder/Actor/Critic/SafetyHead 及其 optimizer state 全程冻结，
  `high_updates=0`。阶段为 `gaitfixcalib(0-15m)`、`repair(15-75m)`、`pushwarm(75-90m)`、
  `pushfull(90-105m)`、`stable(105-120m)`；最后 15 分钟低层完全冻结、Adapter 使用 `1e-5`。
- training-only worker wire 为 `critic323 | response+diagnostic62 | p3_extra108 = 493`。新增
  joint-acc12、14 槽 contact force/onset/over-threshold duration、映射有效位和 Push
  event/delta/age/active/telemetry；这些字段不进入 policy、eval、ONNX 或部署。worker wire 拆分后
  立即回填 target/exec command，供 gait baseline、reward、Adapter 与监控使用。
- 相机训练使用冻结 CNN feature32 的每环境随机 phase 30Hz capture、50Hz hold 和 10 帧 FP16
  队列。前 15 分钟主动延迟关闭；15-75 分钟为 70% nominal、30% 40-100ms；75 分钟后为
  50% nominal、35% 40-100ms、15% 100-150ms。150-250ms 只作 shadow；像素故障维持父包末段
  50% 强度，不继续增加。eval/deploy 不启用人工增强。
- DR 固定为 friction `[0.65,1.25]`、base mass `+-0.4kg`、restitution `[0,0.05]`、noise
  `0.35`。EventManager `push_robot` 在进程创建时保留但速度为零；75 分钟后通过公开
  `get_term_cfg/set_term_cfg/reset` 切到 `+-0.05m/s`，90 分钟后切到 `+-0.08m/s`，间隔
  12-18 秒。透明 wrapper 调用 Isaac 原 `push_by_setting_velocity` 并记录真实 delta；断点 session
  时间通过 `env.usr_conf` 传给 worker，75 分钟后的 resume 不重新等待。未修改平台覆盖的
  `isaac_env/base_env.py`。
- 前 15 分钟从父策略采集 joint/posture/gait envelope，之后使用正常区间严格为零且有 cap 的真实
  推进、默认姿态、joint acceleration、undesired contact、gait responsibility 与合并姿态项。
  Push 后 0.4 秒只减半 joint-acc/gait/非关键姿态。Hip/Thigh `17.6/22Nm`、Calf
  `34.4/43Nm` 保留低权重平方软约束，不改变仿真 effort limit；强 contact quality/crossing/
  starvation、deterministic smoothing 和 action-range auxiliary 保持 shadow-only。anchor、memory、
  mirror 梯度目标为 3%/1.5%/0.5%，合计硬上限 5%。
- P3 评估使用两个显式入口：`p3_standard_eval`（Standard+Camera，低层-only，obs 57901，
  只加载 `modules.low_level.locomotion_encoder/actor`）与 `p3_track_eval`（Track+Camera，
  完整低层+NavigationEncoder+三轴 Actor+ResponseAdapter，obs 57905）。二者共用
  `p3_standard_joint_eval_candidates` 与 `validate_p3_eval_bundle`，新标签优先级
  `stable>pushfull>pushwarm>repair>gaitfixcalib`，并兼容 stair-memory/旧 P3 标签；绝不回退
  P2/LBC/随机权重；
  SafetyHead/Critic/optimizer/训练 buffer 均不创建。
- 低层恢复命令域与可部署高层保持一致，`vx=[0,1.0]`，不训练高层无法发布的负 `vx`。七桶 wire
  继续保留 reverse 槽位以兼容 checkpoint，但其采样权重固定为 0；运动类型为直行 25%、
  `vx+wz` 35%、`vx+vy` 10%、pure-yaw 8%、brake/restart 15% 和 zero 7%。所有包含 `vx` 的
  样本共用低/中/高 `0.10-0.35/0.35-0.70/0.70-1.00m/s` 与 55/30/15 分布。
- 低层步态诊断仍按正/逆坡、正/逆楼梯和运动桶报告接触、滑移、触地、交叉与饥饿，但后三项
  本轮不进入 PPO。mirror 前 45 分钟仅 shadow，之后目标梯度最多 1%；足端、关节、PD、effort
  或 action-scale 映射异常时 mirror 与步态训练项自动归零并告警，主训练不因模型 ID 停止。

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

如果容器已经启动、但 `8765` 同步服务尚未确认，可在 Codex Chrome browser
client 的 Node 会话中复用仓库自举 API。调用方必须传入已经登录腾讯开悟 IDE 的
受控 `tab`；这不是普通 shell CLI：

```js
const { startKaiwuSyncService } = await import(
  "../shared/tools/tencent_kaiwu_webide_upload.mjs"
);

const bootstrap = await startKaiwuSyncService({ tab });
if (!bootstrap.healthy) throw new Error("Kaiwu sync service is unavailable");
```

该入口只负责确保容器内固定脚本
`/data/projects/legged_robot_competition_26/conf/start_tongbu.sh` 已在
`127.0.0.1:8765` 提供 HTTP 响应：已有服务不会重复启动；没有响应时才通过
WebIDE PTY 后台启动。未携带 Token 的 `/health` 返回 `401` 仍表示服务存活，
但不证明本地同步鉴权已经通过。自举使用的临时 WebSocket token 也不是
`IDE_SYNC_TOKEN`。已有服务分支已在真实容器验证；冷启动分支仍须在端口自然空闲的
新容器中复核，不能为了测试主动停止健康服务。完整证据和限制见
[`shared/分析记录/2026-07-31_腾讯开悟WebIDE远程文件系统协议探查.md`](../shared/分析记录/2026-07-31_腾讯开悟WebIDE远程文件系统协议探查.md)。

自举成功后仍从 `server/` 执行标准同步流程：

```bash
python3 local_sync_client.py --check-local
python3 local_sync_client.py --dry-run --skip-unchanged
python3 local_sync_client.py --skip-unchanged
python3 container_rpc_client.py --cwd . "pwd"
```

前三条依次是离线范围检查、联网预览和正式同步；最后一条只验证 RPC 命令链。
正式同步需要与容器一致的 `IDE_SYNC_TOKEN`，腾讯代理 Cookie 缺失或失效时按客户端
提示使用 `--refresh-cookie`，不得把“自举成功”当作 Token/Cookie 已验证。

Cookie 失效时用 `--refresh-cookie` 强制跳过缓存和旧兼容值重新录入；
`--dry-run` 仍会连接腾讯代理并读取 `/health`、`/manifest`，但不写远程。
当前在线状态必须标记为“待刷新 Cookie 验证”，不能由离线检查推断已经可同步。

开发容器需要 GPU/PyTorch 诊断时，可在 IDE 中重启新版 `conf/tongbu.py` 后使用：

```bash
python3 container_rpc_client.py --cwd . "nvidia-smi"
```

该命令复用 `IDE_SYNC_TOKEN` 和 Cookie 缓存；远端 `cwd` 必须位于项目根目录，
并受超时和输出大小限制。它只用于开发容器诊断，不用于启动或管理平台训练任务。

开发容器运行 Isaac/PyTorch smoke 时，当前已验证的解释器入口是：

```bash
/workspace/isaaclab/_isaac_sim/python.sh -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
```

预检通过后，再用同一入口运行 smoke：

```bash
/workspace/isaaclab/_isaac_sim/python.sh agent_ppo/tools/<smoke>.py ...
```

不要直接调用普通 `python`、Conda 环境 Python，或未经预检就使用
`/workspace/isaaclab/isaaclab.sh -p`。平台不同镜像中的 `isaaclab.sh -p` 可能选择未注入
`omni.isaac.ml_archive/pip_prebundle` 的 Conda Python，表现为 `ModuleNotFoundError: torch`，
而平台训练进程本身仍可正常加载 Torch。若 `_isaac_sim/python.sh` 不存在或预检失败，应先
定位当前镜像的 Isaac Python 入口，不能把导入失败归因于模型、checkpoint 或训练代码。

大 checkpoint 不应塞进常规源码同步清单。使用独立分片工具上传到受限临时目录：

```bash
python3 model_chunk_uploader.py \
  /absolute/path/model.pkl \
  --remote-path agent_ppo/test_artifacts/model.pkl \
  --no-cookie-prompt
```

默认按 1 MiB 分片，使用 `2 × 2 = 4` 个最大并发请求；每片和重组后的整文件都校验
SHA256。相同片段可断点续传，目标文件只在完整校验后通过原子替换出现，成功后默认删除分片；
需要保留分片排障时显式传 `--keep-parts`。远端目标被限制在
`agent_ppo/test_artifacts/`，避免覆盖训练代码、平台配置或凭据。

正式创建训练任务前，可在开发容器的 `/workspace/code` 运行可复用清理工具。默认只预览：

```bash
python3 conf/container_training_cleanup.py
```

确认清单后删除测试制品、同步归档、Python/pytest 缓存和 P3 临时日志：

```bash
python3 conf/container_training_cleanup.py --apply
```

所有开发容器测试完成、模型已不再需要时，再执行训练快照精简：

```bash
python3 conf/container_training_cleanup.py --apply --remove-dev-files
```

最后一档还会删除容器副本中的 `.git`、`.vscode`、`agent_ppo/tests` 和
`agent_ppo/tools`，但始终保留正式运行源码、`conf/` 与 `conf/.env`。脚本不会清理平台拥有的
`kaiwudrl/tools`，也不会删除凭证、正式模型或训练日志。每次必须先看 dry-run 清单。

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
