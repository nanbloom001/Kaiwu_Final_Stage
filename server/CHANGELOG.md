# server CHANGELOG

本文件记录 `server/` 主线的活动变更。小调参不再创建永久分支，改为：Changelog + 独立 TOML + 实验编号/父 checkpoint/配置哈希/评估结果。

## [未发布]

- **[P4 R4 Actor 输入兼容修复]** 新入口 `p4maze8h-instant-r4-inputfix` 修复父模型 warm start 的
  capability15 分布偏移：instant controller 的真实变化率 `[10,6,18]` 不再直接写入未归一化
  Actor85 输入，策略继续看到父模型训练时的 `up=[0.30,0.30,1.00]`、
  `release=[0.30,0.60,2.50]`；真实 instant 速率只进入 command contract/诊断。父 Actor anchor
  同时限制为旧 slew 在一个 10Hz tick 内可达且不反向的样本。网络维度、三轴硬范围、
  `policy_target==exec` 和 instant hold 语义不变。

- **[P4 Maze R4 完全取消 slew]** 新入口 `p4maze8h-instant-r4` 从
  `p4maze8h10hz_1416926-mazefinal` warm start，固定单段 Maze、128 env、10Hz 高层和
  28800 秒有效训练。三轴 policy target 在每个高层 tick 只做有限值与硬动作边界处理后立即成为
  exec command，并在 5 个低层帧保持；取消 slew、零交叉、反向确认、运行时 limiter、near-goal
  rewrite 和 recovery override。`reset_live_state()` 现在保留 transition mode，避免 checkpoint
  load/resume 后静默退回旧 slew。R4 command/checkpoint 合同和新 phase 标签明确与旧 P4 slew 包
  exact-resume 不兼容。
- **[P4 R4 后段劣化保护与奖励闭环]** 依据上一轮约 3h50 已开始劣化的证据，R4 在 3 小时开始
  降低 Actor LR，3.5 小时固定冻结 Actor/Teacher/父策略 anchor/Adapter，后续只校准 Critic直到
  8 小时结束。新增不可变父 Actor 分布 anchor；failure、timeout 和 reason4 精确回收 episode 内
  已发放的 Maze new-best credit。碰撞提高到 `-0.16-0.24*severity/-0.06 persistent`，持续卡滞
  2 秒达到 `-0.04/tick`，reason4 保持 `-75` 且不计完成。
- **[P4 R4 命令与 episode 边界闭环]** 修复 instant target 原先在高层 tick 的首个 50Hz 帧仍由
  旧 exec 驱动低层策略的一帧错位；现在 tick 边界先设置瞬时命令，再把同一 exec 写入低层
  proprio、critic 与 worker aux，边界帧起连续 5 帧完全一致。R4 Track eval 现在要求 checkpoint
  的完整 command contract 与运行 profile 相等，不能把 R4 包装配到旧 full-track/slew runtime。
  wrapper 触发但 worker reason 为 0 的未知 reset 作为无奖励、无 bootstrap、无跨 episode GAE 的
  无效边界行处理，并从 Actor/Critic loss、advantage/return normalization 和辅助损失中排除，不再
  以人工零回报训练 `-V`，也不伪造 timeout/failure。Adapter completed record 合同加入 instant-command
  provenance 和 command digest；缺少该证据的旧 slew records 在 R4 replay 中明确拒绝。
- **[P4 R4 冻结期 exact-resume 防线]** 新增 3.5 小时冻结阶段的 save/resume 回归：Actor 与 Adapter
  参数及 Adam `step/exp_avg/exp_avg_sq` 在更新调用前后逐值不变，Critic 仍可更新。该检查覆盖
  fresh process exact resume，防止只把 LR 置零却继续积累 optimizer moments。
- **[P4 R4 capability 真值闭环，已由 inputfix 更正]** instant-command 的 Actor capability 与诊断
  报告真实硬映射边界 `vx<=1.0`。初版把 `[10,6,18]` 写入历史变化率槽；平台行为和父 checkpoint
  A/B 证明该未归一化输入会显著扰动旧 Actor，现已按上方 inputfix 保留父输入语义，并将真实速率
  移至 command contract/诊断。
- **[P4 卡滞恢复证据修复]** unclassified reset 不再伪造成 timeout；reason4 只接受平台
  `nav_stuck_timeout` term 的实际 readback。仅旋转 20 度不再自动清除卡滞，必须同时观察到墙接触
  EMA 至少下降 30%；候选基准跟踪持续接触期间的 EMA 峰值。foot-jam 仅作 shadow 诊断，不触发
  active reset。
- **[P4 R4 容器回归修复]** 修复 `finish_tick()` 将 `[E,1]` 的 unknown-reset mask 与 `[E]`
  的 invalid-row mask 直接组合时广播成 `[E,E]` 的问题，保证 rollout `valid_mask` 始终为
  `[E,1]`。R4 最小 smoke 同步支持单段 Maze reason4 reset、instant-command warm start、禁用的
  mirror 辅助，以及 ResponseAdapter 的 51 帧 future-label 加 24 条 recurrent record 最小历史，
  防止旧测试夹具把合法配置误报为训练故障。

- **[P4 卡墙平移脱困训练]** closed-loop Safety teacher 在训练期已确认
  “执行平移命令但真实位移/速度不足”的样本中，先约束 Actor 降低 `vx`；仅在
  `safe5` 左右侧净空存在明确差异时，再引导与安全侧同符号的 `vy`。response 卡滞标签
  同步拆分为平移与 yaw 两条证据，避免正常转动掩盖 `vx/vy` 失效。这不是运行时
  硬覆盖，接触力和特权 scan 都不进入部署 Actor；Actor/Critic 输入、wire、ONNX I/O
  和 command mapper 不变。闭环机身碰撞的 onset/persistent 值提升至
  `-0.20-0.30*severity` / `-0.08`，持续卡墙提升至 `-0.010` 到 `-0.05` 每 10Hz tick，
  reason4 终止 tick 仍去重。监控增加 recovery loss 与激活比例。
- **[P4 墙边缘前瞻避障]** 五方向 training-only Safety teacher 新增
  action-conditioned edge-clearance 引导：当 Actor 的平移朝低净空扇区运动时，
  以更窄的偏转容差同时约束速度与朝向，修复原 35 度方向容差会接受
  “从安全扇区斜切向墙边”的盲区。该项复用既有 `safe5` 特权标签，只更新
  训练期 Actor auxiliary loss；Actor/Critic 输入、训练 wire、部署接口和运行时
  command mapper 均不变。监控新增 edge loss 与激活比例，定向回归覆盖低净空
  边缘而普通方向损失为零的样本。
- **[P4 closed-loop 最终审查修复]** 五方向教师现在即使全局 Goal
  与必要局部绕行方向不一致，也会要求 Actor 对安全出口建立正确符号
  `wz`，避免用 `vy` 替代机身转向。StuckHead 的 diagnostic-only
  `no_grad` 仅限 `maze_closed_loop_v3`，历史 credit/full-track
  profile 恢复其版本合同中的辅助训练。被 v3 禁止写入 PPO 的旧方向
  reward 继续保留未清零的 shadow raw 指标，避免监控面板失去反事实证据。
  closed-loop 的 Goal tie-break mask 也允许安全度接近的岔路样本进入教师，
  Goal 失效时仍要求安全 top-1 具备明确优势。
- **[P4 Maze 八小时闭环强化 v3]** 新任务 `p4maze8h-closedloop-r3` 从
  `p4maze8h10hz_1416926-mazefinal` continuation warm start，固定单段 Maze、20 列、课程关闭、
  120 秒 episode 和 28800 秒有效训练。低层、NavigationEncoder、SafetyHead、Adapter、StuckHead
  全程冻结，只更新 Actor LSTM/action head 与 Critic；保留父权重和 Critic return statistics，但按
  新 reward/command 合同重置两套 Adam moments。
- 三轴 slew 为 `0.60/0.60/2.00` 增速、`1.20/1.20/4.00` 释放。near-goal capture、全局
  yaw-cancellation、平移 limiter 和旧方向奖励全程 shadow，避免 Actor 采样后的隐式动作改写或多个
  privileged 信号互相竞争。新增五方向 training-only Actor mean 教师，安全优先、Goal 只在安全值
  差不超过 0.10 的出口中 tie-break，方向/速度/yaw 比例为 0.55/0.10/0.35、梯度硬上限 3%。
- success/failure/timeout/reason4 改为 `+200/-60/-40/-75`。active reason4 使用 10 秒滑动窗口、
  非足端墙接触、真实低运动和 Goal 距离确认；不计完成且不 bootstrap。历史最短距离奖励降为
  `1.0/m`、每 episode 上限 `+6`；碰撞 onset/persistent 和持续卡墙惩罚加强。新 checkpoint 标签
  reset 采用显式 wall-clock 课程：前 30 分钟 shadow/12s，之后 active/12s，2 小时后 active/10s；
  checkpoint 保存恢复 offset。新标签 `loopstable > looptrain > loopadapt > loopwarm` 已加入发现
  优先级；Actor85、Critic输入、训练/评估 wire 和部署接口不变，未修改平台覆盖的 `BaseEnv`。

- **[P4 profile 合同与监控路由修复]** P4 checkpoint 的 command/reward/training 元数据现在按
  `training_profile` 生成：Maze credit-repair 使用 `p4_maze_credit_repair_v1`，五段训练使用
  `p4_full_track_v2` 和 `p4_full_track_reward_v2_potential_straight`。exact resume 同时校验
  profile 对应的 `train_scope`；历史误标成 Maze 合同的 full-track 包只能结构化 warm start，
  不再伪装 exact resume。监控构建器读取活动 P4 TOML 的 profile，full-track 会真实装配
  `P4全赛道诊断` 面板，Maze credit-repair 不加载该组。新增两种 profile 的 save/load 与最终
  dashboard 构建回归。历史误标 full-track 包若包含 StuckHead leaf 与 Adam group，会按稳定组名
  一并迁移并返回独立 `full_track_legacy_contract_warm_start` disposition；较旧且同时没有 leaf/group
  的包保持初始化 StuckHead。两类 legacy warm start 都统一重置 session/live state，并输出明确的
  full-track 降级告警。

- **[P4 creditwarm 空辅助批反传修复]** 首阶段 Actor/CNN 冻结且当前 minibatch 没有有效
  stuck 标签时，组合 Actor loss 可能是无计算图常量。公共 PPO 循环现在只在 loss 实际具有
  gradient graph 时执行 backward，并且只有真正产生 Actor 梯度时才 step optimizer 和递增
  gradient-step；Critic 更新继续执行。新增空 stuck batch 回归，防止首轮训练再次因
  `element 0 of tensors does not require grad` 停止。

- **[P4 Maze 两小时归因修复]** 新分支 `codex/p4-maze2h-credit-repair` 从
  `p4maze8h10hz_1416926` 的 `mazefinal` 权重 warm start，任务
  `p4maze2h-credit-repair` 累计完整 `7200s`。训练回到单段 Maze、20 列、课程关闭和 75 秒
  episode；低层、NavigationEncoder、SafetyHead、ResponseAdapter 全程冻结，高层 Actor 权重保留
  但 Adam state 重置，Critic/return statistics 重建，并保留 training-only StuckHead。
- 删除 terminal frontier clawback，改为每 episode 非重复的历史最短距离 credit（`2.0/m`、上限
  `+12`）。平移 limiter 改为 risk `>0.75` 才进入紧急减速，risk=1 时保留 60% 平移且不缩放
  `wz`；卡滞 reset 保持 shadow，只施加 0.8-2.0 秒渐进到 `-0.02/tick` 的持续惩罚。
  关闭额外 Camera/Goal fault、Push、五段出生、segment frontier、route-excess 和 open-straight，
  保留父模型安全组权重。新增 `credit*` checkpoint 标签、Maze credit 面板和新合同 exact-resume
  round-trip；修复 warm start 把旧 session 时钟带入新 optimizer phase，以及 P4 新 train scope 被
  继承 P2 loader 拒绝的问题。
  最终审查进一步把 `session_effective_seconds` 改为只累计 rollout collection 与 PPO update
  活跃时间，checkpoint/监控/日志耗时只计入 `session_wall_seconds`；移除未接线的 Teacher TOML
  伪旋钮，并在共享接口合同中补齐 credit-repair 的保存、评估和恢复边界。

- **[P4 五段全赛道八小时长训]** 新分支 `codex/p4-full8h-r2` 从
  `p4maze8h10hz_1416926` continuation warm start，任务 `p4full8h-r2` 完整训练 `28800s`。
  Track 改为正坡、逆坡、正楼梯、逆楼梯、开放入口迷宫五段，20 列、课程关闭、120 秒 episode；
  高层保持 10Hz/32-tick/TBPTT16，低层冻结。新增 70/60/75% 整轨起点与五段/四分位/70:30
  safe-hard quota 出生，困难位置只有通过运行时表面和机身 clearance raycast 才写入，否则退回同段
  入口。reason 4 同桶重试两次后退回同段入口，禁止通过后移出生刷完成率。
- Goal 编码保持 4 维接口：P4 训练在 10m 内逐值兼容、10m 外保存单位方位；部署 `UwbGoal`
  改为显式版本化，现有 Actor80 默认仍走 legacy 逐轴 clamp，只有未来匹配 P4 v2 spec 的可部署
  制品才显式启用方向保持编码，避免 non-deployable 训练合同改变稳定运行时。
  GoalBelief 加入近目标衰减的零均值椭圆跳变，切向最大 1m、径向最大 0.2m。五段首次推进使用
  terminal-clawed potential，全局 route-excess 关闭；开阔直行只在正/逆坡、teacher 明确开阔、
  Goal 新鲜、远离边界且非 junction/dead-end/contact/recovery 时施加小额 lateral/S-turn/path cost。
  修复了大 `|wz|` 会错误削弱 S-turn 惩罚的逃逸口。
- P4 training wire 从 509 扩为 519，增加出生段、段内分位、安全点与 reason4/fallback 诊断。
  面板独立报告出生段和终止所在段、正负 `vy/wz` 命令链、Goal clean/fault 方位、跳变径向/切向、
  五段 frontier、开阔直行和 active stuck reset。旧 P4 合同只可 warm start；新合同才可 learner
  exact resume。worker quota/RNG 在 fresh-process resume 后按 seed 重启，未伪装成位级 exact resume。
  正式训练的 reset spawn hook 和 active `nav_stuck_timeout` 现在属于启动正确性条件：出生 hook 安装
  失败立即停止，卡滞 term 允许 8 帧装配重试后失败，物理 root-state 写入异常直接抛错。同帧
  wall-stuck 优先于 success，确保真实 reason4 reset 不会获得 `+200` 或进入完成统计。出生比例改为
  reset-event 口径；L0-L9/L10-L19 只称难度列，不再误标为 safe/hard 出生位置。
  平台启动 smoke 进一步修复 worker bridge 的 P4 wire 日志在构造期间读取未初始化
  `_p4_enabled` 的顺序回归；该标志现由已经解析的 runtime stage 动态派生，P2/P4 eval 不受影响。
  同一平台链发现 spawn quota 保持在 CPU 而 worker tail 位于 CUDA，现先统一迁移 device 再组合
  519 wire，避免首次 observation 生成时发生 CPU/CUDA `torch.where` 冲突。

- **[P4 Maze 三小时卡滞恢复与转向稳定]** 新分支 `codex/p4-maze3h-recovery` 从
  `p4maze8h10hz_1416926` continuation warm start，运行 `p4maze3h-recovery-r2` 完整
  `10800s`。保持 Actor85、三轴动作、10Hz/50Hz、32-tick/TBPTT16 和冻结低层不变；将三轴
  slew 更新为 `0.30/0.40/1.50` 增速与 `0.30/0.80/3.00` 释放。新增 Actor mean 容差教师、
  training-only StuckHead、从 rollout 内任意真实 reset 起点构造的零-hidden TBPTT16 中按全部
  PPO sequence-update 约 10% 调度的 mirror、
  SafetyHead hard-positive 加权、完整平移
  vector limiter 和近终点软捕获。所有辅助梯度共用 5% 上限；privileged teacher 不反传到
  NavigationEncoder，Camera clean/live 辅助继续更新视觉编码器。
- 安全奖励将 predictive、missed-safe、yaw cancellation、yaw-exit 和 goal-safe preference 统一到
  原 5Hz 等效 `-0.06` 预算；success 保持 `+200`，reason 4 保持 `-15` 且 terminal tick 不重复
  collision/safety 惩罚。卡墙确认从 7 秒改为 10 秒，Push 全程关闭。平台暂无可信的高层运动意图
  回传，worker wire 升级为 509，额外保留独立 raw wall term，并对 active reset fail-closed，正式配置保持 shadow；面板分别统计
  候选进入、有运动/正进度证据的自然脱困、证据暂失但未确认运动的退出、候选中终止、恢复耗时
  和 60 秒/lifetime 成功脱困，不再把 reset 记为恢复；reset/completion 使用独立信号交叉检查，
  exact resume 恢复 recovery 时间戳和 lifetime 计数。
  旧 P4 checkpoint 只做
  continuation warm start：按稳定 optimizer group 名恢复 8 个旧 Actor/CNN/SafetyHead Adam 组，
  新 StuckHead 保持空 state；同合同 exact resume 严格恢复新 leaf、mirror RNG 和校准状态。
- 平台首轮任务 `p4maze3h-recovery-r1`（ID `236388`）在第一次真实 PPO 更新进入 mirror
  auxiliary 时暴露 GPU dtype 回归：CPU pinned rollout 的 FP16 depth 在未启用 AMP 的路径中直接
  输入 FP32 NavigationEncoder。mirror CNN 重算现与主 PPO 路径一致启用 CUDA autocast，并在 CPU
  测试路径显式转为 FP32；新增 FP16 mirror-depth 回归用例。失败任务只保留为诊断证据，正式替代
  任务使用 `p4maze3h-recovery-r2`。

- **[P4 Maze 10Hz 路径效率强化]** 基于 `p4nav2h_1256446-F` 将 P4 高层控制由 5Hz 提升到
  10Hz，同时保持 Actor85、三轴动作、32-tick rollout、TBPTT16、低层 50Hz 和部署接口不变。
  成功 impulse 提高到 `+200`；新增持续卡墙、目标一致的安全选向和额外路程三项小幅负奖励。
  连续 tick 奖励按 `duration_frames/10` 归一化，避免 10Hz 把原 5Hz 每秒权重翻倍；按米路程、
  每次决策 command-rate 和 terminal impulse 保持原语义。旧 slow/cruise/fast 速度档运行时状态与
  RNG 已删除，exact resume 会恢复已选择的诊断分支后重新应用对应 LR。前 10 分钟只读诊断不计入
  28800 秒梯度训练，并保留 300 秒 rollout/保存/退出余量，平台墙钟合同因此为 29700 秒
  （8 小时 15 分钟）。Push worker 在诊断阶段使用剩余诊断时间的负偏移，2 小时启用边界严格按
  有效训练时间。活动入口和 smoke 默认父包
  同步为 `1256446`；安全方向内的目标偏好使用 terminal-safe 米制真值，带噪 GoalBelief 不进入
  PPO reward。

- **[P4 Track eval-only SafetyHead 边界]** 将 `c93dd11` 已用于旧模型 ZIP 的最小保护补入
  活动 P4 算法源码。`p4_track_eval` 省略 training-only `NavigationSafetyHead` 时，公共相机教师
  诊断路径保持风险 tensor 为零，不再首帧调用 `None`；训练装配存在 SafetyHead 时仍执行原计算。
  不改变网络、checkpoint、Actor action、Goal、终止或 scorer 合同。

- **[P4 Maze 感知归因与进攻式八小时强化]** 新分支 `codex/p4-maze8h-attack` 将
  `p4maze2h-attack` 扩展为诊断后完整训练 28800 秒的 `p4maze8h-attack`。训练改为
  `mazeprobe/mazeattack/mazehard/mazefinal` 四阶段，保留 Actor85、三轴 mapper、低层冻结和
  单段 Maze。新增 10% clean-depth 轻故障只读 shadow，用于区分视觉鲁棒性与 Actor 决策问题；
  fault 在同一 5Hz 教师 tick 生成并与特征 mask 对齐，不进入 observation、reward 或 storage。
  missed-safe 使用事件归一化严重度和连续 8 小时
  权重课程，predictive collision 提升至 1.25 倍且安全组封顶 -0.06；实际
  `frontier_stagnation` 归零，仅保留 shadow。active wall-stuck terminal impulse 改为 -15，并新增
  episode return/非负率监控。单段 `open_entry_maze` 的 physical segment 0 现在按配置映射到
  Maze 指标桶，不再误记为 slope。监控条件率以 eligible event 为分母，checkpoint 同时恢复
  fault RNG/统计/分支并核验 reward 合同；风险减速事件在 terminal tick 立即作废，避免把 reset
  后变化归因给旧 episode。fault 鲁棒性仅使用 held-out probe 环境，P4 启动同时核验 Maze-only
  segment 与实际三轴 slew，exact resume 比较完整 command contract。旧 P4 仅 warm start，不增加
  模型 ID 硬门禁。

- **[P4 Maze 感知诊断与软巡航两小时强化]** 新分支 `codex/p4-maze2h-attack`
  将 P4 Track 训练入口切为 `p4maze2h-attack`：128 env、75 秒 episode、单段
  `open_entry_maze`、20 个静态难度列、课程关闭，总计 7200 秒。旧 slow/cruise/fast
  随机速度档不再限制 `vx`；Actor 仍输出完整 `vx=[0,1.0]`、`vy=+-0.30`、`wz=+-0.90`，
  Goal stale 与 safety cap 只作为执行侧 limited target 保护。新增只产生负值的软巡航项，
  在前方清晰且目标新鲜时轻微惩罚低于 `0.60m/s` 或高于 `0.75m/s` 的 policy target，
  terminal tick 不结算。P4 保存/加载合同升级为 `p4_maze_soft_cruise_v1`，新标签为
  `mazeprobe/mazefull/mazefinal`，同时旧 P4 包在结构通过但合同不同的情况下作为 warm start，
  不再误走 exact resume。监控新增 maze 感知诊断、SafetyHead 风险、Head 正确但 Actor
  选错、风险到减速链路和 zero-hidden shadow；`student_risk_*` 更名为
  `safety_head_risk_*`。
  审查修复后，10 分钟只读诊断使用独立 wall clock，正式 `session_effective_seconds`
  在诊断完成的 rollout 边界从零开始并完整累计 7200 秒；auto 分支使用训练期独立的
  `nav_feat32` 风险/场景线性探针，并以 `goal4` 探针和随机基线作对照，累计 held-out teacher
  coverage、wall AUROC/漏检率、安全方向 top-1、场景 macro-F1 和 clean/live latent cosine 后
  再选择，不再读取最后一个 tick。诊断探针及其优化器只服务分支选择，独立保存并在旧合同
  warm start 时清零。风险减速指标改为锁存风险出现时的 policy vx，并在后续 5 个高层
  tick 内判断 policy/limited target 是否下降。checkpoint 合同升级为
  `p4_maze_soft_cruise_v2_training_clock`，优先级固定为
  `mazefinal > mazefull > mazeprobe > mazediag > legacy P4`，同 ID 缺失时只允许唯一结构兼容
  P4 discovery。P4 周期保存改用 wall clock，因此 10 分钟只读诊断期间仍会在 5 分钟保存
  `mazediag`，正式训练进度与结束条件继续只看 training clock。父包明确切换为
  `p4nav8h-r3 pnavstable-1207698`。旧合同 warm start 现在保留新 session 的确定性运行时 RNG，
  不再强行向 CPU generator 恢复 CUDA 格式状态；同合同 exact resume 仍严格恢复全部 RNG。
  新增 `p4_maze_continue_smoke.py`，固定以 8 env 验证真实父包 warm start、32-tick PPO、Adapter、
  冻结低层 digest、保存和 exact resume；大规模环境测试改为显式按需执行。修复 Maze 诊断探针在
  `finish_tick()` 的 `torch.no_grad()` 上下文中反向传播失败：仅对 detached `nav_feat32/goal4`
  和四个 training-only 线性探针局部启用梯度，CNN、Actor 与 rollout 收集仍保持无图。

- **[P4 卡墙 reset 容器验证]** `MotionWallStuckTracker` 在首批真实 worker
  frame 重试 termination-manager 装配，不再把 observation 初始化期的暂时
  unavailable 状态永久保留。新增 1-env 真实 Isaac smoke，已在开发容器
  验证 `nav_stuck_timeout` 回读、物理 auto-reset 和 worker `reason=4`。下一轮
  平台 smoke 从 shadow 进入保守的 `active + 12s`；不修改平台覆盖的
  `BaseEnv`。

- **[P4 安全奖励连续 ramp 与 Adapter 父记录迁移]** 保持 predictive collision、missed-safe、
  yaw-cancellation 的最终权重及安全组 `-0.05/tick` 上限不变，将安全/鲁棒奖励倍率改为
  `0-30m=0`、`30-60m 0->0.25`、`60-120m 0.25->1.0`，消除 2 小时边界约 `0.5->1.0` 的突跳，
  并将 ramp 版本写入 training digest；Goal fault 使用独立 multiplier，奖励调度不再隐式改变目标
  噪声分布。P3 父包的 32 条 current completed Adapter records 原先因缺少
  逐记录 contract 全部被拒绝，嵌套的 32 条 earlier-lineage records 也未进入 P4；现在只对 parent warm replay 执行 shape、finite、horizon、sequence
  与来源低层 digest/序列版本一致性验证后迁移，当前 P4 缺合同记录仍严格拒绝。历史 record 不要求
  等于父包最终低层 digest，因为它们本来就是 P3 版本化 off-policy replay。父 checkpoint 的 current 与嵌套
  earlier lineage 池不再被机械对半切成不足 24 条的不可采样窗口，而按真实池采样并从可用池补齐。
  新增迁移/拒绝监控，未增加模型 ID 硬门禁。

- **[P4 20列结果与 shadow 卡滞统计修复]** P4 自定义 Track 结果面板改为完整展示
  `completed/abnormal/timeout/score` 的 L0-L19；平台 scorer 按实际 column 直接生成
  `track_l{col}`，不存在两列自动合并为一个旧 L 档。shadow wall-stuck 达到确认阈值后现在每段
  confinement 只上报一次 would-reset 事件和一次预计节省时间，避免把持续 level 信号重复累计为
  数千秒。Adapter 50/25/25 replay 面板按互斥三池的实际 batch 数计算，比例不再可能大于 1；
  lifecycle 面板改用明确的 `platform_lifecycle_callbacks`，不再被误读为任务完成数。stuck reset
  已完成当前开发容器的 termination term 与 1-env 物理 reset 验证，下一轮平台
  smoke 使用保守的 `active + 12s`。

- **[P4 checkpoint 身份与 stuck-reset resume 合同]** P3/P4 phase label 在
  stage/module/spec/shape/有限值校验通过后降为 warning-only 身份元数据，重命名但结构兼容的父包和
  评估包不再被标签单点拒绝。P4 checkpoint 现在散列并保存实际 shadow/active/disabled
  stuck-reset 配置，奖励使用同一合同中的 terminal penalty；exact resume 遇到模式、确认时长或
  其他 reset 参数漂移时明确报错。

- **[P4 R2 目标/相机热修复与墙面卡滞回收]** 任务改为 `p4nav8h-r2`，仍从
  `p3stairmem8h_1013548` 重新 warm start。修复 P4 相机把米制近裁剪阈值直接与归一化 depth
  比较的问题；GoalBelief 改为读取 P4 training-only wire 中未裁剪的米制 goal，并增加过程方差、
  五次一致测量重捕获和 stale-MAP 低速行为。P4 训练 wire 由 493 扩为 507，但 Actor85、Critic
  输入、57905 policy、385 eval wire 和低层接口均保持不变。前 30 分钟通过 `requires_grad` 真正
  冻结 NavigationEncoder 与 Actor/LSTM，避免 LR=0 时 Adam moments 仍漂移。新增 P4-only
  `MotionWallStuckTracker`：默认 shadow，通过平台既有 `nav_stuck_timeout` 的公开 term config 与
  `_nav_motion_stuck` 完成物理 reset；确认后 reason=4 使用独立 `-6` impulse、无 bootstrap且同 tick
  不重复收取 collision/predictive/stagnation。terminal 环境会冻结 507 维 wire 的 raw goal 与 stuck
  diagnostics，避免自动 reset 后的新 episode 数据污染旧 transition；相机面板明确区分 raw、
  near-clip-added 与 delivered hole。未修改平台覆盖的 `isaac_env/base_env.py`。

- **[P4 Track 导航鲁棒八小时训练]** 新增 `p4_nav_ppo` / `p4nav8h`，以操作者选择的
  `p3stairmem8h` 最终 `stairfinal` 包 `1013548` warm start，冻结完整低层并只训练 NavigationEncoder、高层
  Actor/LSTM、新 Critic、SafetyHead 与 ResponseAdapter。新增版本化
  `p4_capability_action_mapper_v1`、slow/cruise/fast 动态速度上限、GoalBelief v2、exec/true
  yaw cancellation、安全奖励同比 `-0.05/tick` cap、共享 30Hz capture/50Hz low-LSTM 状态机及
  rollout 刷新的 clean teacher。后 6 小时通过 EventManager 公共接口渐进启用温和 Push；真实
  Push pulse 以 worker step 去重，并使相交的 Adapter 0.2/0.6/1.0 秒与 pose/stuck 标签失效。
  Adapter replay 在 schema、低层/feedback/capability/mapper/布局合同兼容过滤后再尝试
  50/25/25，不足只由兼容池回填。checkpoint 新增 P4 mapper、Goal、camera、速度 RNG 与冻结
  digest，支持 P4 Standard 低层-only、Track 完整评估及两次四小时 exact resume；恢复时钟会在
  `env.reset(usr_conf)` 前回填 worker Push offset。训练 wire 保持 493、policy 57905、低层
  57901/Actor77/action12、高层 Actor85，不修改平台覆盖的 `isaac_env/base_env.py`。
  最终审计补齐合法 track segment/goal epoch 对 GoalBelief 与 yaw 历史的原子 reset、Goal 重获时间、
  2-6 小时严重相机故障与长 Goal dropout 的互斥，以及 clean/live action MAE 与 latent cosine 的
  PPO 指标汇总。Adapter record 现在携带实际 P4 response capability15，采样重建不再回退 P2
  常量；动态 safety cap 改由 delivered depth 与当前 exec 弧线计算，因此 Track eval 在忽略
  training-only SafetyHead 后仍保持与训练一致的可部署限速闭环。
  快速回归审计进一步将相机延迟改为每个 fault event 固定采样，并保证 delivered frame ID
  单调不回放旧帧；128 环境的孔洞与结构块增强改为批量张量路径。exact resume 恢复相机 RNG
  前不再由 live-buffer reset 消耗随机数；P4 监控健康合同扩展到 Goal、相机、Push、Adapter、
  reward、冻结状态和资源指标，并补齐全部 Adapter 兼容拒绝原因，避免少量样例指标掩盖面板缺项。
  开发容器首轮 Isaac smoke 进一步发现 `p4*` checkpoint phase 含数字，不满足平台探活文件名
  的纯小写字母约束；现改为唯一的 `pnavwarm/pnavrobust/pnavfull/pnavstable`，平台注入的模型 ID
  仍原样写入且不作为硬门禁。过长的 Adapter response-profile 拒绝指标同时缩短到 60 字符以内，
  避免单个非法 metric 使整套 P4 自定义面板被平台跳过。
  P4 迭代与平台 dump 现在也写入仅在 smoke 环境变量启用时生效的原子事件日志，使进程组 runner
  能在首个完整 32-tick update 与成功 checkpoint 后自动停止，不改变正式训练热路径。

- **[P3.5 两小时低速楼梯、相机时序与后期 Push]** 新分支
  `codex/p35-gaitfix2h`、任务 `p35gaitfix2h` 将 P3 训练收敛为 7200 秒低层专项修复。
  保持 policy `57905`、低层 `57901/Actor77/action12` 和高层 85 维合同不变；只训练低层
  LSTM、RNN output、最终动作 head、Critic 与 ResponseAdapter，低层 CNN、Actor body/std 和
  全部高层模块冻结，`high_updates=0`。地形调整为正逆楼梯各 35%、正逆坡各 15%，命令改为
  直行 25%、`vx+wz` 35%、`vx+vy` 10%、pure-yaw 8%、brake/restart 15%、zero 7%，不采样
  负 `vx`；worker wire 拆分后立即回填 target/exec command，避免 gait/Adapter/监控误归低速桶。
  training-only P3 extra 从 46 扩为 108，完整 wire 从 431 扩为 493，新增 joint-acc12、14 槽
  contact force/onset/duration、映射有效位和真实 Push event/delta/age/active/telemetry；字段不进入
  policy、eval、ONNX 或部署。前 15 分钟采集父策略 joint/posture/gait envelope，之后启用有
  deadzone 与 frame cap 的真实推进、默认姿态、joint acceleration、undesired contact、gait
  responsibility 和合并姿态项；Push 后 0.4 秒只减半可恢复项。原生重复 posture/joint-acc/contact
  权重置零，力矩仍使用 Hip/Thigh `17.6/22Nm`、Calf `34.4/43Nm` 的低权重平方软约束。
  冻结 CNN feature32 新增每环境随机 phase 的 30Hz capture、50Hz hold 和 10 帧 FP16 队列；
  15-75 分钟主动延迟为 70% nominal/30% 40-100ms，75 分钟后加入 15% 100-150ms，150-250ms
  只作 shadow。EventManager `push_robot` 从进程创建保留为零速度，75 分钟通过公开
  `get_term_cfg/set_term_cfg/reset` 切到 `+-0.05m/s`，90 分钟切到 `+-0.08m/s`，间隔 12-18 秒；
  包装器透明调用 Isaac 原 `push_by_setting_velocity` 并记录真实 delta。checkpoint session 时间经
  `env.usr_conf` 传给 worker，断点恢复不会重新等待 75 分钟。新增监控合同健康度、相机 age/hold、
  Push 装配/事件/幅度/恢复/条件分桶，以及 P3.5 奖励和 3% anchor、1.5% memory、0.5% mirror
  梯度比例。未修改平台覆盖的 `server/isaac_env/base_env.py`。
  独立审查后进一步统一所有运动类型的 `vx` 三档范围与 restart 采样，使用普通足端 contact onset
  放宽对应腿的 joint-acc 阈值，修正 roll/pitch 轴语义与 15/15/35/35 地形列边界。纯相机延迟帧
  现在也进入 clean-teacher memory auxiliary，并分别监控 fault-only、delay-only 与交集占比。
  Adapter 在最后低层冻结阶段仍按每 rollout 一次更新，replay 比例随阶段切换为
  `50/25/25 -> 60/25/15 -> 75/15/10`；监控合同健康度改为按真实注册、有限数据和最后更新时间计算。
  父包固定为任务 235689 的最终 `stairfinal-1013548`，SHA256 在真实制品加载后记录。
  开发容器真实父包 smoke 进一步修复 `p35_previous_run_final_warm_start` 未加入合同迁移允许列表的
  启动阻断；上一轮 P3 checkpoint 现在走显式 warm-start，而不会误入 exact-resume 合同校验。

- **[P3 楼梯半盲记忆与域随机化八小时恢复]** 新任务
  `p3std8h-stairmem-dr` 从 `p3nav8h-r1_884257-F2` 显式 warm start，保持 57901 低层输入、
  12 维动作和部署接口不变。低层 rollout/TBPTT 扩为 `128/128`，低层 CNN、高层
  NavigationEncoder/Actor/Critic/SafetyHead 全程冻结且高层 PPO 不执行；仅更新低层
  LSTM/Actor/Critic 与 ResponseAdapter。训练观测链新增每环境 `0.10-0.25m` Beta(1,4)
  近裁剪和 rollout-persistent 稀疏/块状/严重孔洞与短时黑屏，使用冻结 F2 clean 教师对 fault
  recurrent action 施加最多 3% 的 memory auxiliary，并提供 full-hidden/zero-hidden 消融指标。
  平台 `BaseEnv` 只创建一次 Isaac 环境，因此首次 reset 即使用 friction `[0.55,1.35]`、base
  mass `+-0.85kg`、restitution `[0,0.10]`、noise `0.55`，不再伪装 0.5 小时动态重建；通过
  EventManager 从进程启动启用 `0.15m/s`、`8-15s` push，确保 25 秒 episode 内能够触发。
  sparse/severe/blackout 使用 rollout 级固定空间随机秩图；severe/blackout 根据原始孔洞率只补足
  到目标无效率，避免实测 80% 原始孔洞被重复叠加成近乎全黑。
  Sim2Real 力矩项改为相对 Hip/Thigh `22Nm`、Calf `43Nm` 硬线的 80% 平方软约束，
  不硬裁剪楼梯短时发力；强 contact/crossing/starvation、deterministic smoothing 与 range
  auxiliary 均只作 shadow，PPO 仅保留低权重 prolonged-air/contact-participation 防线。
  checkpoint 合同升级为 `p3_low_stair_memory_dr_v1`，保存故障 RNG、DR/push 配置合同、
  memory state 与冻结高层 digest；恢复故障 RNG 后仅由进程首次环境 reset 采样一次 live near-clip，
  避免 exact resume 额外消耗随机数。低层吞吐指标读取实际 128 帧 storage，并继续支持
  Standard 低层-only和 Track 完整双评估。
  阶段边界 episode reset 不再重采样 near-clip；监控新增 near-clip 桶楼梯完成率和 severe/blackout
  事件数/持续时间。Push/DR 面板明确标记为配置合同，并用 telemetry-available=0 表示平台当前未
  回传实际事件和逐环境物理采样，禁止将配置上下限解释为运行实测值。删除已取消高层训练后残留的
  `short_high_adaptation_preserves` 等合同元数据。
  监控复核后进一步补齐 near-clip 完成率的逐桶 attempt 分母、12 个地形/运动 gait 样本占比、
  rollout 径向事件与 M3 边界面板；worker reset 结果明确标为 rollout 口径，不再与平台 scorer
  lifetime `completed_count` 直接比较。禁用阶段仍会预采样的故障时长改名为
  `depth_fault_planned_duration_s`，避免误解为已发生故障；P3 面板移除当前低层-only训练中不会
  产生的 `adapter_confidence`、高层 update time 和 CPU-pinned depth H2D 空指标。

- **[P3 stance 滑移单次结算与辅助训练 fail-safe]** 修复步态专项审查发现的两处训练语义问题：
  P2 共享 gait aux 的 `GAIT_SLIP_SPEED_SLICE` 恢复为接触帧平均世界 XY 足端速度（m/s），不再
  被 P3 的累计 stance 距离覆盖；P3 training-only tail 新增 completed-stance distance/event，
  接触质量基线、奖励和分桶面板只在 stance 结束帧结算一次，避免同一异常在后续 1.5 秒窗口内
  被 50Hz 重复扣分。P3 extra/wire 因此从 `42/427` 扩为 `46/431`，policy observation、评估、
  导出和部署接口不变。gait baseline 合同升级为 v3，旧错误语义状态不能 exact resume；结构
  兼容父包仍可 warm start。关节/PD/action-scale/contact 映射无效时，mirror、gait reward 与
  deterministic-mean smoothing/range auxiliary 现在统一归零，不再只关闭前两项。

- **[P3 25 分钟数据驱动的命令域、步态基线和监控修复]** 根据训练捕获
  `20260801-030946`，以 `p3nav8h-r1_884257-F2` 为父包的 P3 低层恢复不再采样高层契约无法
  发布的负 `vx`：保持七桶 checkpoint
  布局，但 reverse 权重固定为 0，并将有效样本重新分配到低速前进、前进和 brake/restart。
  步态基线升级为 v2，删除 reverse 运动桶，brake/zero 不进入健康基线；同地形回退只以 0.5
  权重训练，全局回退仅作诊断，避免 75% fallback 用跨地形阈值错误塑形。监控改用独立
  `p3_low_*`/`p3_high_*` PPO 指标，新增 KL、clip fraction、更新职责/耗时、命令桶覆盖、基线
  回退层级、机械功率分位和四项 Sim2Real 原始分量。训练专用 worker tail 从 37 扩为 42，wire
  从 422 扩为 427；新增字段不进入 policy observation、评估、导出或部署接口。奖励权重和网络
  结构保持不变。课程重新分配为 120 分钟低层、10 分钟 Critic/Adapter 校准、10 分钟 Actor
  warm-up 和 10 分钟完整高层收尾；P3 radial warm start 显式保留父包高层 Critic、Critic Adam
  moments、return statistics 和高层梯度步数，避免 30 分钟高层阶段从零重学价值函数。

- **[P3 训练侧防震荡与命令 anchor 修复]** 任务更新为
  `p3std2h30-gait-smooth-radial`，保持父包 `p3nav8h-r1_884257`、57901 低层输入和 12 维动作接口。
  修复 P3 sampler 继承旧 bucket-0 zero hold、使用全局 RNG 以及低层 transition 无条件
  `anchor_weights=1` 的训练语义错误；P3 worker tail 新增 committed command bucket/anchor 两列，
  wire 从 420 扩为 422，仅训练入口消费。低层 PPO 新增 deterministic action mean rate/jerk
  超限辅助和 `|mean|>5` 软范围约束，梯度只连接 LSTM、RNN output 与最终 action head，并和 mirror
  共用 10% PPO Actor 梯度上限。新增 raw/exec clip、关节目标 rate/jerk、15-25Hz 功率及分命令桶
  clip/anchor 面板。`push_robots` 保持关闭，不把尚未隔离的外扰混入前 15 分钟 gait baseline。
  审查后进一步修正：每个 minibatch 都重新标定 mirror/smooth/range 梯度，并按三者实际合成
  梯度向量统一裁到 PPO policy gradient 的 10%，面板不再显示过期估算；worker command 状态
  缺失或 wire 值非法时 anchor fail-safe 改为 0，禁止静默回退为全锚定。镜像装配的静态
  joint/PD/effort/action-scale 核验缓存到 worker bridge 生命周期，不再在 50Hz 每帧同步 GPU。
  P3 全 target sampler 在概率为 1 时不再调用全局 Torch RNG，保持专用 RNG 与主训练 RNG 隔离。
  最终审查补齐四项可信度修复：镜像 preflight 精确定位平台 `JointPositionAction` 并校验实际
  `0.25` scale 合同，避免多 action term 时误验无关项；axis-major 动作的逐腿 mirror error 改为
  `FL[0,4,8]/FR[1,5,9]/RL[2,6,10]/RR[3,7,11]`；步态分桶只统计有效 gait 帧，impact 改按各腿
  contact-onset 事件作分母；P3 低层 update 显式传入 coordinator 阶段，gaitcalib 不再被旧 8 小时
  schedule 重映射为 lowbase，真实冻结策略权重及 Adam moments。

- **[P3 真实父包 smoke 合同修复]** 将开发容器父包测试工具对齐当前 2.5 小时课程：分别验证
  gaitcalib 的低层 Actor 冻结与 lowbase 的低层 Actor 更新，阶段边界使用
  `5400/6000/7200s`，并允许低层恢复 rollout 按计划更新 Adapter。真实 `884257` 父包现可完成
  低层 PPO、高层 PPO、Adapter update 和 save/exact-resume 联合 smoke。

- **[P3 步态专项审查漂移修复]** 镜像/步态训练的 worker preflight 不再只检查关节名称顺序，
  现在同时核验左右 action scale、PD stiffness/damping、effort limit 和 `contact_forces` 足端映射；
  任一项不可证明对称时只关闭 training-only mirror/gait 奖励并告警，不阻断主训练。交叉落脚的
  touchdown-y 改为完整 root quaternion 逆变换后的机体系横向坐标，避免坡面/楼梯 roll/pitch
  造成误罚；滑移统计改为每次 stance 累计世界 XY 距离，并在 1.5 秒窗口取 stance 最大预算，
  不再用接触帧平均速度稀释短时严重滑移。checkpoint 补齐实际运行时 M3 半径合同的保存/恢复，
  明确 worker command sampler 因跨进程边界在环境 reset 后 fresh 初始化。面板增加正/负 `vy/wz`
  条件响应、M3 边界差和 lifetime 时钟，并将四足滑移单位改为 stance distance。

- **[P3 Standard 双评估、径向完成与步态专项 2.5 小时课程]** 新任务
  `p3std2h30-gait-radial` 从
  `p3nav8h-r1_884257` warm start。局部目标改为相对真实出生点的 M1 `1.3-1.6m`、M2
  `2.5-2.9m` 与平台 Standard M3；默认 8m 地块使用 `3.90m` proxy、`3.93m` 控制目标和
  `4.00m` 边界，最后 0.20m/0.04m 只限制向外线速度并在完成后归零。正常里程碑晋级保持方向，
  timeout 仅从原方向左右 30/60 度重规划。P3 worker transport 在 reset 行保留终止前 root pose，
  使 M3 proxy、平台 success 和 terminal transition 使用同一 episode；不修改平台覆盖的 BaseEnv。
  高层新增不可重复的 radial new-best 与 M1/M2/M3 事件奖励，Track-only safety/tracking/gait shaping
  继续归零；raw frontier clawback 改为有效 timeout 的加权均值。低层成功更新后立即推进 Adapter
  version/history 边界并执行 1/2 次 Adapter update，集中校准阶段每轮 4 次，高层后段每两轮 1 次。
  监控拆分 M1/M2、proxy/platform/joint、一致率、径向距离/历史最佳/M3 hold、低高层 reward mean
  和 Adapter attempts/applied/skipped。低层新增 4 类地形 × 3 类运动的健康父策略基线、25% 完整
  TBPTT sequence 镜像一致性，以及只在超出基线时生效的接触滑移/冲击、交叉落脚和步态饥饿
  惩罚；旧 air-time/duty/participation shaping 权重归零。镜像梯度只进入低层 LSTM、RNN 输出层
  与最终 action head，首 15 分钟 shadow/baseline 阶段冻结 Actor，CNN 始终冻结。P3 worker wire
  在训练入口扩展为 `critic323 | P2 aux62 | P3 gait/runtime35 = 420`，Track/Standard eval 仍使用各自
  原有装配且忽略 training-only gait 状态。Adapter replay 改为最新版本 50%、近期 P3 25%、父记录
  25%；`adaptercalib` 完全跳过高层 Actor backward/step，避免 Adam moments 漂移。

- **[P3 双评估入口回归修复]** 新增两个显式评估入口 `p3_standard_eval`（Standard+Camera 低层-only，
  obs 57901）与 `p3_track_eval`（Track+Camera 完整高低层，obs 57905），供同一个 P3 包 `highslow`
  等阶段文件分别评标准/赛道。修复平台任务 `599578` 中 P3 包被 `_infer_stage_from_task_name`
  回退到 `lbc_loco`、旧 LBC loader 又只搜 `responsecalib` 等候选、最终未加载任何 checkpoint 即
  尝试 `exploit()` 并报 `'NoneType' object is not subscriptable` 的装配链。两个入口共用
  `p3_standard_joint_eval_candidates` 与 `validate_p3_eval_bundle`：P3 标签优先级
  `highslow>highadapt>adaptercalib>lowfull>lowmedium>lowmild>lowbase`，无同 ID 时唯一 discovery，
  多候选明确报歧义，绝不回退 P2/LBC/随机权重；选中文件结构/有限值错误硬失败。Standard 只装
  VisionEncoder+Actor77，Track 只装低层+NavigationEncoder+三轴 Actor+ResponseAdapter，
  两者都不创建 Critic/SafetyHead/optimizer/scheduler/训练 buffer。Track 继续复用 P2
  eval transport 与 terminal-return bridge，`goal_reached` 进入平台 scorer 完成数不再恒为 0。
  Track 首次平台评估（任务 `599777`）发现 `P2WorkerBridge._resolve_config()` 的 stage 闸门
  不含 `p3_track_eval`，worker 首次 env reset 即在 `p2_response_aux()` 报
  `P2 response aux requested while bridge is disabled`；现已将 `p3_track_eval` 纳入共享
  feedback/gait transport 启用集合并从 `[p3_standard_joint]` 读配置（`_terminal_safe_root_pose`
  与 curriculum probe 保持 P2 eval 语义不变）。
- **[P3 15 分钟 lifecycle 回归修复]** P3 自定义 workflow 现与已验证的 Nav 边界一致：每个
  成功 `env.step()` 完成 observation/terminal/storage 处理后调用一次平台 lifecycle no-op，低层
  80 帧和高层 320 帧路径均覆盖；失败 step 不推进。新增成功/失败回调面板，普通回调异常只告警，
  checkpoint 写入失败仍中止。修复平台任务 `235452` 中 `succ_cnt=0`、5 分钟业务包已写出但
  平台约 15 分钟仍发送外部 SIGTERM 的回归。
- **[训练任务创建前容器清理工具]** 新增 `conf/container_training_cleanup.py`。工具默认 dry-run，
  普通模式只清理测试制品、同步归档、Python/pytest 缓存与 P3 临时文件；显式
  `--remove-dev-files` 才删除容器副本的 `.git`、`.vscode`、测试和 smoke 工具。根目录身份校验、
  允许删除范围和 `.env` 保护均为硬安全边界，避免再次靠手工命令误删凭证或平台运行目录。
- **[P3 正式环境数调整]** 真实开发容器完成 64/128 环境对照 smoke；128 环境在
  80 帧低层 PPO、32-tick 高层 PPO、Adapter update 和 save/exact-resume 全链路中
  峰值 GPU reserved 约 1.61 GiB（约 32.9%），端到端样本吞吐比 64 环境高约
  31%。P3 正式配置因此从 64 调整为 128 环境，其余 rollout/network 合同不变。
- **[P3 平台托管 BaseEnv 边界修正]** 撤回全部 `server/isaac_env/base_env.py` 修改。低层恢复阶段
  改用平台原生 2-8 秒三轴命令，保证低层 observation、worker reward 与实际执行一致；高层接管
  命令后冻结低层，只训练高层 PPO 与 Adapter。末段标签由 `jointslow` 改为 `highslow`，不再声称
  存在无法闭环的同步低层更新。P3 高层明确关闭
  Track SafetyHead/predictive collision/missed-safe、gait、body-collision、tracking 和 stagnation shaping；
  local success/timeout 使用 +8/-1，timeout 回收 frontier potential。目标改为 64 次 rejection sampling，
  严格保持 1.5-2.8m 与地块 1m 内边界；`local_abs>3.2m` 只保留诊断，不冒充平台 reset。
- **[P3 Sim2Real、恢复与显存]** 四阶段 DR 仅使用平台公开支持的摩擦、base added mass 和显式
  observation noise；删除无效的 COM、PD、action gain/delay 与按环境 push 声明。低层新增有界 torque EMA/peak 与 action rate/jerk 代价。
  低层 PPO storage 改存 77 维 `proprio+frozen CNN feature`，在线/导出 observation 仍保持 57901。
  checkpoint 保存 immutable anchor/digest、低层版本和实际 DR phase；exact
  resume 校验 anchor 与 Adapter record 的低层 digest/version，候选 discovery 多解时明确报歧义。
- **[P3 审查与性能收口]** 所有阶段职责切换都在 checkpoint 后通过公开 `env.reset()` 开始新
  episode，不依赖 BaseEnv 补丁；仅 0.5h/2h/3.5h 改变 DR 参数。Adapter 校准阶段跳过低层 critic、
  log-prob、anchor 与 PPO storage 写入，高层阶段按每两次 PPO update 更新一次 Adapter。命令、地形
  和奖励统计改为 GPU 端累计并在 rollout 末一次性搬运，移除逐帧/逐 tick 的同步点；P3 TOML 显式
  关闭未使用的旧 `worker_progressive` bridge。平台原生命令现按环境跟踪命令 epoch，避免异步
  resample 被 Adapter 误当成同一 target；高层阶段删除永远不可达的低层联合 storage/update
  分支和逐帧 no-op lifecycle 调用。低层 storage 指标覆盖 observation、critic、action、return、
  anchor 与 recurrent hidden 等全部 tensor buffer。exact resume 额外校验阶段标签、训练状态阶段与
  session 时钟一致；真实 smoke 将 Adapter interval 临时设为 1，生产配置仍保持每两轮更新一次。
  高层 10 帧窗口内提前结束的环境会在余下帧保持零命令，禁止旧 episode 命令进入 reset 后的新
  episode；Standard/joint-success 计数改为 GPU tensor 累积、rollout 末统一搬运，移除每个 50Hz
  frame 的两次 `.item()` 同步。低层版本号现仅在至少一个 PPO minibatch 完成 optimizer step 后
  推进；全部非有限更新被跳过时保留当前 digest/version 与 Adapter unfinished history。

- **[P3 Standard 高低层联合恢复八小时]** 新增 `p3_standard_joint` 复合入口与
  `p3std8h-sim2real` 配置，从完整 P2 `p2nav2h-r2_648278` 包执行结构校验后的 warm start。
  单任务按 rollout 边界依次训练低层 Actor/LSTM/Critic、集中校准 ResponseAdapter、适配高层
  PPO；高层开始训练后低层保持冻结。低层 CNN、动作方差和高层/低层
  optimizer 参数集合保持隔离。P3 policy transport 继续为 57905，低层只通过输入适配器删除
  goal4 得到 57901，不改变低层网络结构；高层 Actor85 与三轴命令合同不变。新增 1.5-2.8m
  私有局部目标，目标限制在 8m 地块 1m 内边界，0.6m 到达只结算高层奖励并重采样，不终止
  Standard episode；平台 Standard scorer 仍是正式成功口径。checkpoint 增加 P3 阶段标签、
  低/高层 optimizer、独立计数、session/lifetime 时钟和 exact-resume，模型 ID/lineage 仅用于
  候选选择与追溯，不作为单点硬门禁。
- **[P3 运行时接线修复]** P3 的 observation 复用 P2 response transport 时，worker bridge
  原先只允许 P2 stage，真实 critic observation 会因 aux bridge 未启用而直接失败；现将
  `p3_standard_joint` 纳入共享反馈/步态 transport，并关闭不适用于 Standard 的 Track curriculum
  worker probe。P3 使用独立 workflow 驱动 50Hz 低层与 5Hz 高层 rollout，补齐训练、推理、
  save/load、SIGTERM final save 与真实父包 save/resume 闭环。监控不再复用 Track 课程、赛段和
  迷宫面板，只展示 P3 阶段、局部目标、命令链、步态、Adapter 与资源指标。
- **[P3 recurrent 模式交接与 Adapter LR 修复]** rollout 收集显式将共享低层 ActorCritic 置于
  `eval`；进入低层 PPO replay 时显式恢复 Actor/Critic/LSTM 的训练模式，同时让冻结 CNN 的
  BatchNorm 统计和 S0 anchor 保持 `eval`。这修复了早期开发版高层推理后进入低层 replay 时 CUDA 报
  `cudnn RNN backward can only be called in training mode` 的交接错误。Adapter optimizer 也按
  P3 阶段启用真实 LR：低层恢复阶段 `3e-5`、集中校准 `2e-4`、高层阶段 `1e-5`，避免
  update 计数前进但参数不变。

- **[P2 容器 smoke 合同校正]** `p2_continue_smoke.py` 改为使用专用
  `INITIAL_VY_LOG_STD=-1.1` 校验新增 `vy` head，避免把正确的二维到三维迁移
  误报为旧 `vx/wz` 初始值 `-0.7`。同步补齐 rollout 新增的
  `safety_target/safety_valid` 和当前 CNN 训练阶段必需的 CPU FP16 depth，并确认
  SafetyHead BCE 在真实 checkpoint smoke 中有限。非 CUDA 更新路径在 CNN forward 前
  将 FP16 depth 转为 FP32；GPU 正式训练仍保持 FP16 storage 和 AMP。

- **[P2 两小时提前安全选向与吞吐优化]** 新任务 `p2nav2hsafedir` 从最新验证通过的
  `p2nav10hvyavoid2` 完整包继续训练 2 小时。保留三轴 Actor、Critic、NavigationEncoder、
  Adapter、return statistics 和兼容 Adam moments，只新增 training-only SafetyHead。
  `nav_scanner` 现在区分 finite hit、全 `+Inf` 合法 no-hit 与坏 ray；P2 在有效率不足时跳过
  教师损失/奖励，DAgger Oracle 继续严格失败。纯安全教师只使用三方向墙体风险和
  `height_scan256` 地形连续性，不混入 goal/heading；错过明显更安全方向的负奖励在两小时内
  从 0 渐进到 `-0.03` 上限，保留部署可得深度风险与真实碰撞项。Track 改为 20 列、关闭课程，
  统计合同升级为 3×20。PPO epoch 改为读取配置并记录逐 epoch KL/clip/entropy；高层 depth 在
  tick 起点立即取得 CPU FP16 所有权，低层帧不再复制完整 observation/aux，命令桶和赛段桶改为
  GPU `scatter_add_` 聚合。后续审查修正 `ordering=xy` 的物理左右语义：低 row 是 body `-y`
  右侧，高 row 是 body `+y` 左侧；异常 height scan 也会使教师标签失效。checkpoint 发现顺序
  补齐 `safestable > safefull > safewarm`，并提供显式 `low_level_only_eval`，让 Standard 从完整
  P2 包只抽取冻结低层，默认路径仍拒绝静默丢弃高层。安全选向率改为只在非并列、确有明显
  安全替代的条件样本上统计。评估与 Standard 导出忽略 SafetyHead。

- **[P2 reward-v8 terminal-consistent frontier potential]** 根据上一轮 100 分钟训练中
  success 吞吐基本持平、timeout 增量 `18→46`、迷宫 stuck `17%→52%` 的趋势，删除直接进入 PPO
  的 `positive_progress/negative_progress/new_best`。改用
  `gamma_frame^duration * phi_after - phi_before`，其中 `phi=2*(episode_start_distance-best_distance)`，
  failure/timeout/success 的 terminal potential 固定为零，使坡和楼梯的局部进展无法补贴迷宫失败，
  同时历史最佳距离允许必要绕行。timeout 从 `-15` 调整为 `-22.5`，缩小与 hard failure `-25`
  的套利空间；success 保持 `+50`。新增 potential before/after 和 terminal clawback 面板。

- **[开发容器模型分片上传]** 新增 `model_chunk_uploader.py`：大 checkpoint 使用可配置的
  分片并发和 bundle 内 GET 并发上传，每片独立重试与 SHA256 校验，支持按远端 manifest
  断点续传；容器端按固定顺序写入 `.uploading` 临时文件，整文件大小/SHA256 正确后才原子
  替换目标，成功后默认清理分片。远端路径强制限制在 `agent_ppo/test_artifacts/`，模型、Cookie
  和 Token 均不进入常规源码同步清单。
  默认并发调整为 `2×2`，单片从 4 MiB 收紧到 1 MiB；同时兼容
  `scope=agent_ppo` 返回的相对 manifest 路径，避免已完成模型被误判为缺失并
  重复上传。

- **[P2 reward-v7 定向墙体风险与 terminal 统计]** 任务改为 `p2nav10hvyavoid2`。
  预测碰撞从中央 ROI 升级为 left/center/right 三方向与 upper/middle/lower 三高度带的 20%
  稳健低分位，按三轴目标运动方向平滑选区，并用上下带平整度降低缓坡及仅下部台阶误报；
  单 tick 封顶 `-0.02`，旧风险只作 shadow diagnostic。该值根据上一轮 100 分钟数据校准：旧
  `-0.04` 中央 ROI 项虽占总负奖励约 6.5%，但末段真实接触率仍升高，因此本轮优先修正方向和
  坡面误报，只保留旧上限的一半。`vy` 初始 `log_std` 收紧为 `-1.1`，
  但动作范围仍从首轮完整开放到 `±0.40m/s`。gait PPO 权重归零，四足 excess 继续监控。
  首次 terminal 前的 target/exec/measured/true/gait/collision 会冻结到旧 transition，reset 后的新
  episode 不再污染命令桶、赛段、碰撞 trace 或 tracking；terminal tracking 明确无效。碰撞 onset
  改为 rollout 窗口数量和发生率，面板增加三方向 wallness/risk 与 legacy-v2 对比。

- **[P2 command-v2 三轴追加十小时训练]** 新任务 `p2nav10hvy` 从完整二维 P2
  `p2nav8h-r2_291713` 显式 warm start，保留原 NavigationEncoder、LSTM、`[vx,wz]` head、
  ResponseAdapter 与兼容 Adam moments，新增零均值 `vy` head 并重建 Critic/return statistics。
  动作分布、tanh Jacobian、rollout、slew、reward、Adapter 分组和监控统一升级为
  `[vx,vy,wz]`；`vy` 从首个 rollout 即开放可信核心 `±0.20m/s` 与探索硬边界 `±0.40m/s`，
  不再使用动作范围墙钟课程。新增 head 仍使用独立 LR 与 RNG，旧 CUDA RNG 无法注入不同设备
  generator 时只记录 fresh-seed 降级，不形成模型 ID/载荷单点门禁。checkpoint 区分本轮
  session 与继承 lifetime，新增 `frame_count` 保持 exact resume 的 50Hz/5Hz 相位；评估严格要求
  command-v2 三轴合同，旧二维 evaluator/exporter 不得静默补零。

- **[P2 非有限奖励与 reset 步态窗口修复]** 5 Hz reward settlement 现在逐环境校验
  goal distance、duration、terminal reason、frame safety、命令及 gait/collision 输入；异常行
  写入有限零奖励、保持旧 best distance、清空有状态脱困/碰撞历史并标记整轮跳过，避免 NaN
  进入 GAE 和 return statistics。四足窗口不再把 auto-reset 边界帧写入 ring 后又将分母清零，
  消除下一帧 duty factor 可能大于 1 的错配。checkpoint 约束同时澄清为：模型 ID/lineage
  仅告警，候选文件缺失时允许回退；一旦选中文件，反序列化或模块结构错误仍必须停止，禁止
  静默替换 exact resume 或配置父包。新增 `tools/p2_continue_smoke.py`，用旧 P2 navfull 包
  覆盖 warm start、独立 Critic 重建、PPO/Adapter 更新、保存和 exact resume 的完整续训闭环；
  开发容器已用 `model.ckpt-navfull-291713.pkl` 跑通。

- **[P2 ContactSensor 映射与实时赛段归类修复]** 四足窗口不再把 Articulation
  全局 body ID 当作 ContactSensor 局部列，也不再选择任意首个带 air-time 的 sensor；运行时固定
  选择 `contact_forces`，按 FL/FR/RL/RR 精确名称分别建立 robot/sensor 两套索引，并校验名称唯一性
  和 `current_air_time/net_forces_w` shape。映射异常时 gait 与 body-collision reward 立即归零并
  warning，训练继续。三段面板不再把出生 `terrain_levels` 当作当前位置，而按 Track 世界 X
  边界生成 terminal-safe `current_segment`；出生 row、难度 column 和当前段分别保留。Adapter
  分段 MAE 改用 current segment，父/旧 records 以 `-1` 排除。worker transport 扩为
  `critic323 | response_aux30 | diagnostic_aux32 = 385`，reward/training contract 升至 v4。
  通用平台 EnvMonitor 的旧 `completed_count_track_l*` 仍不作为 P2 权威口径，且未修改受保护的
  `isaac_env/base_env.py`。

- **[P2 Track 评估成功终止回传修复]** 正常复赛 Track 评估依赖 Gymnasium
  `terminated/truncated -> RslRlVecEnvWrapper dones -> BaseEnv single-life -> BaseScorer`
  的既有链路；`p2nav8h-r2_291713` 的 worker 已在 auto-reset 帧识别
  `goal_reached`，但当前平台 wrapper 会在部分 P2 success 行丢失公开 done，导致机器人传送回
  起点、worker 统计成功而正式 `completed=0`。P2 observation 初始化现在为当前环境安装幂等
  terminal-return adapter，将 aux24 reset 与 aux25 success/failure/timeout 只做 OR 合并回
  原生 `terminated/truncated`，从而继续复用已验证 scorer，不修改平台 `base_env.py`、不新增
  scorer，也不检查模型 ID。平台复测 `598827` 证明首版把 adapter 安装在 observation process
  无 env 构造期，实际没有包装真实 ManagerBasedRLEnv；现改为在 ObservationBridge 已绑定真实
  env 的首次 `process()` 开头幂等安装，早于 RSL wrapper 的第一次正式 step，并增加一次安装和
  最多八次 terminal merge 的有限诊断。原生 done 永不被清除或重分类；初始 reset 的 reason=0
  不会被计为 episode。真实容器 1-env Track+Camera 闭环已验证
  `native_hard=0 -> merged_hard=1 -> goal_reached -> completed=1 -> score=99.99`；
  完整模型的平台重新评估仍待执行，证据分层见 Bug 台账。

- **[P2 reward-v3 避障与脱困]** 在保持 progress/new-best/success 主体不变的前提下，新增三个
  5 Hz 高层项：非足端接触力 collision（首次按严重度 `-0.08~-0.20`、持续贴墙 `-0.03`）、
  基于单调 best-distance frontier 的 3 秒停滞惩罚（`-0.015` 递增并封顶 `-0.06`），以及
  frontier 单 tick 真正推进至少 `0.08m` 后的一次性 `+0.15` recovery。Recovery 具有 5 秒冷却、
  每 episode 最多两次，reset/terminal 不发放；来回摆动、低命令和后退重进不能刷取。worker
  transport 扩为 `critic323 | response_aux30 | diagnostic_aux29 = 382`，新增最近 0.2 秒非足端
  最大接触力，前 58 槽语义不变。50 Hz `undesired_contacts` 改为零权重监控，避免双罚；速度
  突降不进入 reward。P2 自定义面板新增“避障与脱困”三项贡献。

- **[P2 自定义指标上报修复]** `p2_nav_ppo_workflow` 不再依赖平台 monitor 的
  `get_pids()` 注册列表，而与通用 workflow 一致使用当前进程 PID 调用 `put_data()`。
  修复注册列表为空或接口不可用时 P2 损失、奖励分解、命令链、Adapter、课程和性能面板全部
  无数据、仅 EnvMonitor 原生运动 Reward 可见的问题。上报异常不终止八小时训练，但改为每次
  既有一分钟上报周期输出明确 warning，不再静默吞掉。

- **[P2 监控 line 面板上限修复]** 平台每个 line 面板最多接受 20 个指标，原五个
  `vx分桶N命令链` 各含 24 个指标，导致整份用户监控配置被 learner 跳过。现按语义拆为
  `前进链` 和 `转向链`，每个 12 项，保留全部 target/exec/true 指标。P2 回归测试新增
  每面板 `<=20` 的静态断言，定向回归 `87 passed`。开发容器使用平台原生
  `MonitorConfigBuilder` 完整构建成功；邻近 P1.5/checkpoint 回归 `108 passed`。

- **[P2 父包身份与步态基线修复]** `p2nav8h` 默认父包改为
  `p15resp8h-r1_37953-F` / `responsecalib-37953`。P2 checkpoint 选择改为请求 ID 优先、配置
  父包与同类文件发现兜底；请求 ID、文件名 ID、payload/lineage 不一致只输出告警，不再形成
  单点阻断，模块/spec/shape/finite 等兼容性校验仍保持硬失败。gait baseline 升级为训练开始前
  即生效的版本化 37953 固定 envelope，禁止用正在训练的 P2 策略在线改写，并把逐腿步频差
  纳入封顶 `-0.04` 的非劣化约束和监控；Adapter 域/赛段样本占比仅统计有效 future horizon。
  宿主 P2 核心为 `60 passed`，扩展回归为 `238 passed, 3 subtests passed`；真实 37953 父包在
  故意传入请求 ID `88888` 时仍完成 `bootstrap_high`，实际 lineage 记录为 `37953`。

- **[P2 三段逆向 Track 八小时 reward-v2 长训]** 将活动任务升级为 `p2nav8h`，固定
  `pyramid_slope_inv -> pyramid_stairs_inv -> open_entry_maze`、128 环境和 28800 秒累计
  有效训练。默认从 P1.5 `responsecalib-37953` 建立高层；已有 reward-v2 P2 包继续按完整训练
  合同 exact resume，旧 P2 reward-v1 仍可显式 warm start 并重建 Critic、return statistics、
  高层 optimizer/scheduler 和 rollout。
  导航奖励移到 5 Hz，删除持续可领取的绝对距离/航向类正奖励和复杂条件门控，改为非对称
  进度、新纪录、一次性 success/failure/timeout、恒定时间成本、crawl 死区、小 command-rate、
  仿真真值 tracking 与封顶 `-0.04` 的 1.5 秒 gait 非劣化约束。50 Hz 仅保留轻量姿态、能耗、
  body contact 和两个零权重 curriculum compatibility term；Adapter、UWB、障碍评分和地形
  类型均不参与 PPO reward。
  worker transport 扩为 `critic323 | response_aux30 | diagnostic_aux28 = 381`，保持前 30 槽
  Adapter 合同不变，追加四足 duty/swing/air/frequency/slip 和 reset 前 column/row/goal-distance。
  P2 curriculum outcome 现在按 terminal 前快照归因，首次 episode 起点也计入累计；成功 term
  同时兼容 `active_terms` 与 `_term_names`，避免平台 manager 版本差异把完成误记为失败；合法
  worker reason 现在统一决定 hard/timeout 分类，二者严格互斥。terminal tick 结算前不再提前
  清空 episode 历史最佳距离，避免失败/超时帧把普通 tick 进度重复计作 `new_best`。
  八小时 LR/entropy 课程覆盖 0/10/20 分钟、2/6/8 小时，CNN 只在 rollout 边界解冻；checkpoint
  新增八小时训练合同、return statistics、gait baseline、aux 维度和 encoder digest，并保持
  三套模块/optimizer/scheduler、独立 RNG 与 completed Adapter records。
  P2 面板新增 reward-v2 守恒分解、20 个 `vx x |wz|` 桶的 count/share/推进/跟踪/命令链/结果/
  gait 数据、三段起点条件统计、逐腿 gait 和 Adapter 分 row/核心域/外沿域误差。逐环境 reset
  日志仍按分钟聚合。宿主 P2 核心回归当前 `60 passed`；除两个无关旧模块外的训练端扩展回归
  为 `238 passed, 3 subtests passed`。容器真实父包 smoke、128 环境完整
  rollout/backward、平台 builder 和正式八小时任务尚未执行。

- **[P2 Track 连续高层 PPO 两小时训练]** 新增 `p2nav2h` / `p2_nav_ppo`：从
  `responsecalib-37953` 启动，冻结低层视觉 Encoder/Actor，训练独立
  NavigationEncoder、二维 tanh-squashed Gaussian recurrent Actor、recurrent Critic，
  并以独立梯度路径低学习率更新 ResponseAdapter。高层动作只建模 `[vx,wz]`，mapper
  插入 `vy=0`；探索硬边界为 `vx=[0,1.25]、|wz|<=1.0`，可信核心域只用于 Adapter
  confidence 衰减，不裁剪策略探索。前 10 分钟 CNN optimizer 组保持零学习率，之后按
  conv1/conv2/conv3+fc 分层解冻。
  `configure_app.toml`、训练 TOML 和 `Config.CURRENT` 三处入口统一指向 P2，避免首次
  `load_conf()` 前的 worker/工具误读 P1.5 默认维度。
  新增 32-tick、TBPTT16 的高层专用 recurrent PPO storage 和 variable-duration GAE；
  worker 通过 aux24/25 传输 reset 与 success/failure/timeout，修复 wrapper 丢失
  `truncated/time_outs` 后旧新 episode 被拼接的问题。当前平台不提供 terminal critic
  observation，因此 timeout 明确使用 no-bootstrap fallback，且 GAE 不跨 episode。
  解冻前只保存 85 维高层特征，
  解冻后深度以 CPU pinned FP16 存放并按 CNN microbatch 搬运，低层 PPO storage、
  S0 anchor 和第二套 Camera 均不创建。实现 128 环境优先、64->32 frame microbatch
  降级与显存 allocated/reserved 峰值遥测。
  Track 固定为 `pyramid_slope -> open_entry_maze`，开启平台通用 curriculum，并新增
  row/column/reset/outcome 的批量结构化探针；20 分钟语义异常只告警。奖励只迁移目标
  接近、方向/速度投影、距离、成功、时间、终止、姿态、能耗和 body contact，三项高层
  command/tracking/stuck penalty 每个 5 Hz tick 只计一次；tracking penalty 使用该
  transition 结束时的 exec/feedback，避免把上一条指令的响应错误归因给新动作。P2 面板
  同时报告目标/执行命令、左右转向、反馈有效率、Adapter confidence/MAE/NLL/标签有效率、
  curriculum 累计值，以及 H2D、各 optimizer update、env-step、吞吐和显存指标。
  checkpoint 使用 schema 2 的 `navwarm`/`navadapt`/`navfull` 标签，完整保存冻结低层、
  动作型高层、在线 Adapter、三套 optimizer/scheduler、独立 action/shuffle/neutral/Adapter
  RNG、有效训练秒数、completed
  records 和课程诊断，并在 CPU 透明字段保留父低层 optimizer/scheduler/训练状态，保持
  `deployable=false`。包补齐 `bundle_kind`、实际 train scope 和所有可学习 leaf 的
  `class_name/spec/state_dict`；P2 eval 改为模块-only 装配，不创建 Critic、optimizer、
  scheduler 或 ResponseBuffer。非有限 policy/value transition 会执行零指令、清洗存储、
  清零 recurrent hidden 并跳过当轮 PPO，但 Adapter 和长训继续。
  P1.5 父包的动作方差与 S0 anchor 同时迁移进规范 low-level leaf，避免 P2 首存后无法再
  独立恢复低层训练状态。首存约五分钟，之后每十分钟及 CNN 解冻边界请求无参数平台归档；
  正常结束/SIGTERM final save 幂等。静态预审额外修复了
  解冻后 `[N,180,320,1]` depth 无法直接写入 `[N,57600]` storage 的必现 shape 错误；
  lifecycle 自动 dump 写失败现在只告警并安排 60 秒归档重试，不再提前终止两小时任务。
  开发容器 full-stack preflight 进一步确认平台配置校验器禁止在 `[terrain.track]` 中出现
  `sub_terrains_random`，即使其值为 `false`；生产 TOML 已删除该字段，固定赛道顺序继续由
  `sub_terrains=["pyramid_slope","open_entry_maze"]` 表达。第二轮 preflight 又确认平台在
  `terrain.curriculum=true` 时会同时启用依赖 `track_lin_vel_xy` / `track_ang_vel_z` 的原生
  command curriculum；P2 现注册两个 `weight=0.0` 的 compatibility-only term，使平台可以
  解析配置但不产生奖励、也不扩张原生 fallback command，Track 难度列课程继续生效。两项修复
  均不修改平台 `base_env.py`。
  课程探针不再在 Isaac 首次 Track reset 前锁定 zero-filled `terrain_types`；显式列初始化
  标记可用时等待全部环境完成，旧平台则等待首个非零 episode length，并兼容 manager 仅暴露
  `_term_names` 的版本。当前已完成宿主与开发容器 P2 专项 `35 passed`、按当前分支实际测试
  清单执行的 Nav/P1.5/P2 邻近回归 `209 passed`、Python 编译、TOML 解析与 diff 检查。
  真实 37953 已在开发容器完成 128 环境 Track/Camera reset、32 个高层 tick、低层冻结
  inference、Actor/Critic/Adapter update、运行 checkpoint、SIGTERM final candidate 和 CUDA
  exact resume；十分钟 CNN 解冻规格另以 128 环境、450 MiB pinned depth、64-frame microbatch
  完成完整 backward，无 OOM。平台正式训练任务、模型列表发布和评估仍待验证。
  P2 自定义监控从 53 个单指标图和 7 个无仓库侧生产者的旧 Track 计分图，重组为训练收敛、
  奖励贡献、导航成效、控制反馈、运动安全、响应预测、课程诊断和性能资源八组共 45 个多曲线
  面板。除 rollout 回报/价值/优势、动作探索标准差和三套学习率外，现直接展示十项平台
  `reward_*` 贡献，并新增成功/失败/超时率、目标推进效率、target/exec/measured/true 速度链、
  feedback source/age、可信核心域外探索、倾斜/横向漂移和低层动作饱和只读遥测；旧
  `completed_count_track_l*` 等 P2 空面板不再注册。aisrv curriculum reset 明细从逐帧逐环境
  INFO 改为每分钟聚合成功/失败/超时、行列变化和最多 12 个样本，累计指标与 checkpoint 语义
  不变。宿主相关 P2/Nav 回归 `64 passed`；当前 8 组 45 面板包含 110 个去重 metric key，平台
  原生 builder 与前端显示仍需在下一次任务复核。
  Track 评估兼容路由同时修复：显式 `p2_nav_ppo` 在 eval 模式自动转换为 `p2_nav_eval`；平台
  未转发 `policy_entry` 时，Track+Camera 根据当前 P2 bootstrap lineage 选择 P2 纯推理装配，
  不再无条件进入旧 `nav_eval` 并只搜索 `navbc/navdagger/navfull`。旧 Nav lineage 保持原入口；
  checkpoint schema、模块 spec、有限值和“未加载不得评分”门禁均未放宽。平台 eval 只转发
  policy observation 的限制通过 eval-only scan 槽位携带 `response_aux30 + marker` 兼容，
  维度仍为 57905；首次推理优先使用 runtime 评估目录中的同 ID P2 包，ID 不一致只告警，
  选中的文件仍须通过完整 P2 模块契约后才能进入确定性前向。

- **[P1.5 连续指令扩域与响应器联合训练]** 新增 `p15resp8h` / `p15_response`：
  从 `commandfull-34728` 在 Standard + Camera、128 环境继续训练，CNN 冻结，前 7 小时
  联合更新低层 Actor/LSTM/Critic 与独立 `ResponseAdapter`，最后 1 小时只校准
  Adapter。worker 以 5 Hz 独立采样 `[vx,wz]` 目标、50 Hz slew 写入 live command；
  课程按 0.5/1.25/2/7 小时墙钟扩域并持续回放父指令域。跨进程 privileged wire 为
  `critic316 | response_aux30 = 346`，aisrv 在进入 Critic/PPO storage 前立即拆分，
  Actor observation 保持 57901、Critic/storage 保持 316。新增版本化
  `FeedbackEmulator`、future-label buffer、四足逐腿统计和三项轻量步态保持 penalty；
  前两小时把已注册的 physics-material event 固定为 `[1.0,1.0]`，之后仅将摩擦扩为
  `[0.6,1.2]`，质量、push 和强增强保持关闭。
  checkpoint 升级为 `kaiwu_train_v1/schema_version=2`，同时保存低层、adapter-only
  高层组件、双 optimizer/scheduler/RNG、独立训练计数、bounded warm-resume buffer 与
  实际 command/feedback 配置 digest，并记录 `FeedbackEmulator` 实现 SHA256；fresh/resume
  均从第 10 分钟开始按 10 分钟墙钟周期保存，课程边界不再插入额外模型。
  预审修复后 rollout 调整为 80 帧，ResponseAdapter 使用 8 帧 burn-in、逐环境 reset mask
  和同一 low-level version 序列；版本切换与进程 resume 都会清空未完成 future history，
  已完成 records 可继续回放。短时反馈固定为 SportMode `vx/vy` + IMU `wz`，UWB 退出
  0.2/0.6/1.0 秒输入；terrain 距离 curriculum 关闭并增加 level×command histogram。
  capability15 改为显式 piecewise-union 编码。schema2 缺 scheduler/RNG 时不再冒充完整
  resume，而是 warning-only 降级为 warm start；新增显式 low-level-only loader。
  逐腿 air/swing 指标改用有效样本分母，family telemetry 分开报告重采样事件与按帧占用。
  graceful final save 增加一次短重试，SIGTERM 会链式调用平台已有 handler 后进入幂等收尾。
  开发容器 full-smoke 启动器保留调用时的 server 根目录，避免 IDE 中 `agent_ppo`
  符号链接解析到 `/workspace/code` 后丢失项目内 `kaiwudrl` 与 `tools`。
  容器 smoke 进一步发现 TOML 的 80 帧 rollout 未被 Agent 采用：运行时仍继承 StageConfig
  的 48 帧，导致 51 帧 future-label history 在每次低层更新时被清空。现已在
  `P15ResponseConfig` 固定 80 帧，并增加 append/history/version-reset/有效 horizon 遥测。
  full-smoke 收尾改为按 `--runtime-dir` 扫描 `/proc` 中的所有已证明子进程组，
  避免框架另建进程组后父 PID 先退出、工具却误报已完全停止；该逻辑已在开发容器以
  两个独立临时进程组完成 `/proc` 集成验证，停止后无匹配进程残留，支持回归集
  `165 passed`。
  首次平台任务在 256 环境 reset 时于 ray-caster 初始化暴露 CUDA illegal-memory；P1.5
  的 80 帧 rollout 使批量帧预算从父阶段 `256x48=12288` 增至 `256x80=20480`。
  生产环境数因此固定为 128，使预算降至 `128x80=10240`，同时保留 future-label、
  burn-in 和 TBPTT 合同；不修改平台 `base_env.py`、scanner、相机或地形。
  正式训练不设置表现门禁；非有限 minibatch、command 写入失败和保存失败按既定宽松策略
  跳过/告警/重试，不主动提前结束八小时任务；iteration-cap 最终保存失败也会在 60 秒后
  再试一次。真实 34728 已在开发容器完成 8 环境 GPU rollout、低层 PPO、
  Adapter 更新、schema2 保存与完整进程 resume；平台短 smoke 与八小时长训待执行。

- **[P1.5 adapter-only Camera 评估修复]** P1.5 schema2 将辅助
  `ResponseAdapter` 保存为 `modules.high_level.component_status="adapter_only"`，此前
  Camera 评估兼容入口把任何 `modules.high_level` 都误判为 hier-nav 动作策略，导致
  `responsecalib-37953` 在低层权重加载前硬停止。LBC-Loco Camera 路径现在要求
  `stage_type=p15_response_adapter`、精确的 adapter-only 容器结构和
  `CommandResponseAdapter(32,64,16)` spec；辅助 Adapter 权重本身不加载，视觉编码器与
  low-level Actor 则在有限值检查后继续以 `strict=True` 加载。真正 Nav 高层、伪造
  adapter-only 标记或包含额外动作策略字段的包仍拒绝静默降级。
  checkpoint schema、训练状态、Standard/Track low-level-only 续训和候选文件名均不改变。
- **[Nav 运行中平台归档恢复]** 平台任务 `234786` 的最终模型
  `navbc-52313` 证明现有 checkpoint 名称、三模块内容和最终归档链均有效；运行中的
  `/data/ckpt` 文件不会单独生成前端所需的 `id_list`、`kaiwu.json` 和 ZIP。Nav workflow
  因此恢复历史成功阶段使用的无参数 `agent.save_model()`，在完整 TBPTT iteration 边界每
  五分钟请求一次用户可见平台归档，路径与数字 ID 仍只由平台 wrapper 注入；正常结束和
  SIGTERM final save 保持独立幂等。`dump_model_freq=3600` 继续只负责内部运行 checkpoint。
  自动 dump 诊断同时改为计入已加载模型 ID，父 `34728` 的首个真实边界现在正确显示为
  1272 次 callback 后的 `36000`，不再在会话 callback=3600 输出假边界。平台任务
  `234805` 已完成运行中端到端验证：无参数周期保存获得
  `/data/user_ckpt_dir/...navbc-39528.pkl`，24 秒后生成平台 ZIP，且模型 `39528`
  已在网页模型列表可见；紧接着的 `/data/ckpt/...navbc-39600.pkl` 仅是独立的
  运行恢复 checkpoint。

- **[Nav 监控面板名称兼容性]** 将三个控制链面板的中文显示名从含 `/` 的
  `requested/effective`、`worker/exec` 表述改为平台字符白名单内的纯中文短名称。
  英文面板 ID 和指标 key 保持不变，避免 monitor builder 配置校验失败后整份 Track/Nav
  自定义面板被跳过。

- **[工作区归档与平台 base env 复原]** 活动树中的
  `isaac_env/base_env.py` 已恢复为平台原始版本，SHA256 为
  `75ebdaf6888e94262598a26db1586b2598cb474422e3382b6bba9e96ddbb6e67`；此前写在该
  文件中的 command bucket、continuous-training 和深度配置补丁不再保留，相关能力必须
  继续由 `agent_ppo` 扩展点实现。历史实验源码和分析记录纳入 Git，checkpoint、ZIP、
  ONNX、视频、缓存和 `.env` 继续仅作本地制品并由精确规则忽略。

- **[Nav 保存失败分类与同步路径加固]** Nav 平台 lifecycle 回调现在会区分
  普通回调异常和 `CheckpointSaveError`：前者仍记录后继续，后者代表
  checkpoint 序列化、写入或非空校验失败，必须终止训练，避免长训在无可恢复
  模型的情况下继续。优雅退出的 final save 仍保持 best-effort，不覆盖平台原始
  退出原因。IDE 同步服务的 workspace 校验改为真实路径校验，阻止符号链接
  越界和通过别名绕过 `isaac_env/base_env.py` 保护；同时显式兼容平台将
  `agent_diy`、`agent_ppo`、`conf` 映射到 `/workspace/code` 的固定目录结构，映射目录
  内再次越界的符号链接仍会被拒绝。根目录同时忽略本地
  `.worktrees/` 与 `.zcode/`，避免嵌套工作树和工具状态被误暂存。

- **[hier-nav 控制链可观测性]** Nav DAgger 新增 student/oracle/requested/
  effective token 分布、驻留状态、Oracle 规则分支、worker/held/exec command、
  低层 action 与真实速度的对齐统计。首个 nav tick 输出完整控制链快照，
  后续日志可区分高层 token 没生效、scheduler 驻留、低层无响应和
  worker reward 仍读取另一套 command 等场景。切换响应按一个 nav period
  后的 action/vx 变化统计；action 覆盖全部 token 切换，vx 只统计目标 vx
  确实变化的切换，避免把纯转向误报为速度无响应。reset 会取消未完成样本。

- **[hier-nav 终止指标口径修正]** aisrv 只能确认 non-timeout termination，
  不再将其命名为 hard failure 或 completed episode；真实完成/异常/超时仍以
  Track scorer 指标为准。soft-stay 继续使用历史的每环境帧风险分母，
  episode 归一化比例只用于面板，不改变既有阈值数量级。身份与 digest
  元数据仍按当前宽松策略 warning-only，结构、非有限权重和写盘失败仍硬停。

- **[hier-nav 平台模型发布生命周期修复]** 经可成功完成的 Track LBC 归档核验，平台
  模型 ID 和 `dump_model_freq` 按 `BaseAgent.learn()` 回调推进，而不是按 Nav 的 outer
  iteration 或业务侧 `save_model()` 写盘推进。历史 LBC 的 `10000 × 24 = 240000`
  个低层帧与 `282409 - 42399 = 240010` 个平台步基本一致；失败 Nav 则从父 ID
  `34728` 仅推进到 `34788`，与约 60 个 outer iteration 一致。Nav workflow 现于
  每个成功低层批量帧完成后调用一次 lifecycle callback，TBPTT 梯度更新频率不变；
  生产 `dump_model_freq` 调整为 3600，按当前 128 环境 Camera Track 吞吐约每五分钟
  触发一次平台标准 checkpoint。常规 checkpoint 完全交由平台发布，删除 10 轮首存、
  5 分钟业务侧墙钟保存和 30 轮兜底；正常结束、平台正常停止与 SIGTERM 共用一次
  幂等 final best-effort 保存。新增成功/失败 callback、累计低层步、累计环境帧和
  下次 dump 距离 telemetry；Nav 保存日志会区分 `/data/ckpt` 运行中 checkpoint 与
  `/data/user_ckpt_dir` 最终平台归档候选，并打印 size、SHA256、距离上次保存的墙钟
  时间和 lifecycle 数。单次回调失败记录后继续且不计入发布进度。完整链路 smoke 使用
  测试专用 160 回调 dump，避免 per-frame lifecycle 下每帧重复落盘。该更正取代下方
  旧条目的周期自定义保存策略，不改网络、checkpoint schema、标签或平台拥有的
  `isaac_env/base_env.py`。

- **[hier-nav Track 生命周期与监控修复]** 首轮平台 Track smoke 证明
  `nav_dagger` 能真实更新，但 256 环境任务在首次常规 checkpoint 前收到外部
  SIGTERM，且通用 monitor builder 显示 Standard/PPO 风格面板。正式配置固定为
  128 环境；checkpoint 改为本会话 10 个 outer iteration 后首存、之后每 5 分钟
  墙钟保存，并保留每 30 iteration 的兜底。自定义 monitor builder 现在显式展示
  Track 0-9 难度的完成/失败/超时和总分/能耗/姿态/时间分，以及 Nav DAgger 的
  CE、Oracle 模仿率、目标有效率、进度与终止指标。平台拥有的
  `isaac_env/base_env.py` 不参与本修复。任务 `234739` 进一步证明自定义 Track/Nav
  面板已成功追加，但平台默认环境模板仍按 `Config.CURRENT` 的字面 Standard fallback
  注册；本专用分支现将静态默认改为 `NavDaggerConfig`，使监控注册期和运行期都从
  `task_type=track` 开始。运行时 `configure_app.toml` 选择及一致性校验继续保留。

- **[hier-nav checkpoint 与完整 smoke 可诊断性]** checkpoint 候选仍按平台请求 ID
  精确选择，但包内 `platform_model_id`、lineage、父 ID 和低层 digest 缺失或不一致
  改为醒目 warning，不再阻断结构兼容的操作者选定包。反序列化失败、必需模块缺失、
  high/low-level 契约或 state-dict key/shape 不兼容、非有限张量仍硬停止。保存前若
  低层 digest 漂移，会记录 warning 并把实际重算 digest 写入新包，避免传播虚假
  lineage。新增 `agent_ppo.tools.nav_full_smoke`，用 test-only 环境变量在内存中将
  `num_envs` 临时改为 1-256，生产 TOML 不落盘修改；支持 `start/status/stop`、独立
  进程组、首个有效 TBPTT update 自动停止。`NAV_SMOKE_EVENT_LOG` 启用单次
  `O_APPEND` JSONL 事件，消除多进程 stdout 交错对自动判定的影响。首个 nav tick
  及每轮同时记录 Oracle goal 有效率、goal3 幅度和 goal4 freshness；全无效目标只
  产生 `first_update_skipped`，不能被 smoke 误判为完成训练。

- **[hier-nav 开发容器启动链修复]** 使用真实 `commandfull-34728` 在开发容器运行
  完整 `train_test.py`，确认开悟 local-wrapper 会在 preload 之前先调用一次
  `save_model(id=0)`。nav 现在跳过这次尚无父血缘的 bootstrap save，不写随机包，
  随后仍由正常 preload 建立低层状态。worker 的 nav scanner 解析改为依据平台
  `AnisotropicGridPatternCfg` 推导矩形网格，支持当前 `21x13=273` rays，并校验
  pattern metadata 与 tensor 总数；不再把非平方的合法 scanner 误判为环境错误。
  修复后的开发容器 `train_test.py` 已在 47.11 秒内成功；该入口强制单环境且不执行
  真实 preload/nav workflow，所以正式任务的父包加载、256 环境和首个训练 update
  仍需单独 smoke，不把工具级成功扩大解释为完整训练成功。
  随后使用隔离启动器关闭 test 模式、临时降为 8 环境，完整运行到 14 个 outer
  iteration（18808 env steps / 1888 nav ticks）；Adam state 已建立，第 10 轮
  `ce=1.6981, top1=0.133, grad=0.700, nonfinite=0`。最终临时包的 34728 lineage、父包
  SHA256 与低层 digest 均现场核验一致，证明真实 preload、rollout、TBPTT update 和
  checkpoint lifecycle 已走通；停止后远端 TOML 已按哈希恢复为 256 环境。
  checkpoint 身份/digest 门禁的 warning-only 收敛与可复用 smoke 启动器已在后续
  修改中实现，见上条。

- **[开发容器诊断 RPC]** 现有 `conf/tongbu.py` 复用唯一
  `IDE_SYNC_TOKEN` 新增 `exec_b64_v1`，可在项目根内执行有超时和
  输出上限的诊断命令；不引入第二套 admin 鉴权。新增
  `container_rpc_client.py` 复用本地 `.env` 与腾讯代理 Cookie，并新增
  `nav_init_gpu_probe.py` 在开发容器真实 GPU 中逐步构造高低层模块。
  RPC cwd 经纯词法归一化后必须位于工作区内，同时修复了
  `Path.absolute()` 不折叠 `..` 的旧路径边界。

- **[hier-nav 启动生命周期诊断]** 针对平台任务 `navdagger-r1`
  只完成 Agent 构造、但没有 checkpoint 身份日志、`NavDAgger` iteration 或
  `env.reset` 证据的启动停滞，在 Agent 进入/离开 `BaseAgent.__init__`、
  平台 `load_model` 进入/完成、通用 workflow 分发、nav workflow 进入及
  `env.reset` 前后增加一次性 `LifecycleProbe` 日志，并继续覆盖首帧 command
  注入、冻结视觉编码器/低层 Actor、高层 CNN/LSTM、Oracle、tick buffer、
  首次 `env.step`、首次 TBPTT 更新、平台 lifecycle callback 及首次保存。
  在 aisrv 仅完成三套网络打印、未到 `AlgorithmNavDagger ready` 的平台证据后，
  构造期进一步拆分为三模块逐个 `.to(device)`、架构校验、冻结、Adam 创建、
  optimizer 参数集合断言和 Oracle 创建；每个边界打印进程角色、耗时、模块设备
  与无同步的 CUDA allocated/reserved 统计。
  checkpoint 目录清单采用 best-effort 读取，诊断本身不会因模型池并发替换
  文件而阻断加载。该变更不改训练、checkpoint 或平台 `base_env.py` 行为；
  用于下一次 smoke 通过最后一条成功日志把停点定位到单一边界。

- **[hier-nav MVP 审查修复]** 训练 stage 现在从
  `conf/configure_app.toml [app].policy_entry` 在拼接 stage-specific TOML 路径前选择，
  正式训练固定 `nav_dagger` + 低层父 `34728`；Track+Camera 评估即使平台
  未转发 `policy_entry`，worker 也会推导 `nav_eval`，aisrv 被强制送入
  `lbc_loco` 时会按同 ID nav 包升级为完整高低层装配。`nav_scanner` 在
  worker observation process 内压缩为前/左/右墙特权特征，critic 契约扩为
  323 维，Oracle 会朝更开阔侧确定性转向；缺 sensor/结构错误直接停止，合法的 no-hit
  ray 按开阔空间处理，不再
  静默训练无法走迷宫的教师。删除无消费者的 `[terrain.level_mix]`，改用
  `num_parallel_tracks=10` 原生轨道并打印实际 level/type histogram。nav checkpoint
  现严格锁定词表、输入切片、网络维度、UWB 测量链、slew/clamp 和 zero
  急停语义；resume 保留包内真实低层父血缘。soft-stay 的 hard/timeout 比例改为按
  已完成 episode 计算，不再被环境帧数稀释。全程不修改平台拥有的
  `isaac_env/base_env.py`。

- **[Camera 评估 checkpoint 严格加载]** 平台即使因 Camera task 强制走
  `lbc_loco` 入口，也会使用与 VisualPPO 相同的
  `visual_eval_checkpoint_candidates()` 同 ID 顺序（`command* -> anchor* ->
  rl* -> vision*`）选择训练包。LBC eval 记录完整候选、选中绝对路径、SHA256、
  bundle ID 和 lineage，只加载 `vision_encoder` 与 low-level Actor；非
  `kaiwu_train_v1`、缺失/冲突的 `platform_model_id`/`lineage.platform_model_id`、
  state_dict/model spec 不兼容或反序列化异常都会终止评估。未成功加载时 inference
  显式拒绝运行，禁止以随机初始化参数产生评分；不再创建或回退同 ID 的
  `lbc-loco` 别名，且不修改平台拥有的 `isaac_env/base_env.py`。

- **[Standard 命令泛化与评估入口修复]** 新增
  `standard-command-r1` / `visual_command_generalization_v1`：从固定评估选出的 Anchor
  R2 视觉学生继续训练，CNN 冻结，Actor/LSTM/Critic 训练，S0 action/latent anchor 固定为
  `0.35/0.10`。运行使用 256 环境、四小时平台墙钟、28401 source command 域与按墙钟
  `0→50→100%` 混入 target profile 的 command scheduler；target 包含 zero、低速/正常
  前进、前进+yaw、纯 yaw 和横移。保存只生成 `kaiwu_train_v1` 的
  `commandbase`/`commandblend`/`commandfull` 文件，数字 ID 完全由平台注入，训练包仍
  `deployable=false`。command scheduler 已从无法穿过进程代理的 aisrv 迁入 Isaac worker，
  由 `LBCObservationProcess`/`PolicyObservationProcess` 与 `CriticObservationProcess` 在
  `default_observation()` 前调用同一个 env-owned 幂等 bridge。bridge 只通过公开
  `command_manager.get_command("base_velocity")` 取得 live tensor，写入后立即回读，并用
  `common_step_counter` 保证 policy/critic 同一步只调度一次；`episode_length_buf == 0` 的环境
  会重新采样。写入失败时先恢复原生命令并禁用本任务的自定义调度，恢复或回读也失败才硬停止。
  每个新任务的 worker ramp 都从 0 分钟开始，checkpoint 不保存或恢复每环境 command、hold、
  bucket 或 worker RNG。aisrv 只从 policy observation `[6:9]` 计算 anchor 权重，不再输出伪
  effective 指标。全程不修改平台拥有的 `base_env.py`，不访问私有 `_command`；原生 sampler
  固定为合法的 `[300,300]` 防止短周期覆盖。血缘/ID 差异、动作幅度、KL、timeout 和
  hard termination 都只记录 warning。P0 评估入口与 command writer 的平台 smoke
  **尚待验证**：最终评估 TOML 必须显式选择 `visual_policy_optimization`，且 aisrv/learner
  不得出现旧的 `CommandAdapter command_hook=unavailable`。同步端现精确保护平台拥有的
  `isaac_env/base_env.py`，拒绝写入或删除该文件，同时继续同步其余 `isaac_env` 文件。

- **[Standard Anchor R2 视觉学生退火]** 将
  `visual_policy_optimization` 阶段，从冻结的
  `visionfull-28401`（`S0_visual_lbc`）初始化视觉编码器与 Actor，不复用
  D4/D5 或 R3 的配置、checkpoint 和调度。一次 4 小时平台任务依次执行
  45 分钟 Critic 预热、45 分钟 Actor 微调、90 分钟 Actor/LSTM 联合退火和
  60 分钟低锚定稳定；CNN 与冻结 S0 始终不进入 optimizer。训练采用 48 步
  recurrent rollout、16 步 TBPTT、原生 28401 command/terrain 域和 warning-only
  漂移诊断。`kaiwu_train_v1` 恢复包新增独立 Anchor 会话时钟，并使用
  `anchorcritic`/`anchoractor`/`anchoranneal`/`anchorfinal` 探活标签；Camera 评估
  仍只加载视觉编码器与低层 Actor。平台提供的 `base_env.py` 保持固定 SHA256，
  旧环境需要的评估推导和深度预处理配置在 `agent_ppo` 内兼容。D1–D5、HJC 旧
  LBC 和无活动 `StageConfig` 的 Track TOML 及其专用测试已删除；历史证据继续
  保留在 Git 历史、Changelog、分析文档和归档 Tag。真实 PyTorch tensor smoke、
  平台预加载及首个 checkpoint 保存仍须在 3–10 分钟平台启动中确认。

- **[Standard 视觉长训计数修复]** 视觉 LBC 的平台 lifecycle 从每个 inner
  environment step 调用一次改为每个完整 outer iteration 调用一次，恢复与上一阶段
  一致的 iteration / 平台模型 ID 语义；checkpoint 新增
  `iteration_semantics=completed_outer_iterations_v1`，并兼容修复前的零基视觉包。
  定时发布改为 225 outer iterations（按平台实测约 9.9 分钟），
  `max_iterations=20000` 仅作高安全上限，平台任务页控制 10 小时，余弦学习率
  独立按 14000 iterations 衰减。教师预热延长到 30 分钟，4.5 小时线性 ramp
  在第 5 小时结束，随后保留约 5 小时纯学生训练。Standard LBC 显式启用
  `continuous_training`，避免全局 `frame_no` 超过单 episode 的 1250 帧后持续误报
  `all_done`；底层逐环境 auto-reset 和评估终止语义不变。

- **[Standard 深度视觉蒸馏]** 阶段 4：用 D435i 深度图替换特权 `height_scan256`，
  训练 `VisionEncoder(CNN+LSTM)` 蒸馏视觉能力到冻结的 Actor77。分支
  `codex/depth-vision-distillation` 基于 `codex/standard-dagger-r2@5c5b5ab`
  （R2 已含 `kaiwu_train_v1` 低层加载、base_env 修复和视觉路径核心改动），
  并从 `origin/main` 恢复 R2 误删的 10 个 `agent_ppo/tests/` 文件。核心改动：
  (1) **视觉训练包 codec**：新增 `kaiwu_train_v1 + modules.vision_encoder` 格式
  （`save_vision_bundle`/`load_vision_bundle`），与现有 `lbc_loco` 部署导出格式
  分离；checkpoint 文件名用纯字母 ramp 标签（`visionteacher`/`visionhalf`/
  `visionfull`/`visionblocked`）满足探活正则。(2) **线性 ramp DAgger**：学生驱动
  比例从 0 线性 ramp 到 100%（约 4.5h），取代原计划的 6 阶段离散分档；无强制晋升，
  质量恶化时 soft-stay 冻结当前比例（不回退、不升档），恢复后继续 ramp。(3) **单次
  forward**：`prepare_vision_update` 缓存 teacher/student latent+action，动作选择和
  三路损失复用同一结果，避免学生驱动时 LSTM 对同一观测推进两次。(4) **三路损失**：
  `0.5*SmoothL1(latent) + 0.1*(1-cos) + 1.0*SmoothL1(action)`，action_loss 梯度穿过
  冻结 teacher_actor 回传到 vision_encoder。(5) **首轮不启用 replay**：LSTM 单帧
  replay 会用错误 hidden 历史算 latent；序列回放（8-16 帧 + burn-in）留作独立消融。
  (6) **LSTM 不跨运行恢复**：每次启动 reset 环境 + 清零 hidden，checkpoint 只存
  `lstm_reset_contract`。(7) **父文件显式优先**：`vision_parent_candidates` 让
  `daggerfull` 显式排在 `locomotion` 之前，并从视觉续训候选中排除低层父文件。
  (8) **长训恢复状态**：视觉包增加 ramp clock、LR scheduler、安全阈值/
  预热样本和 RNG；workflow 从已完成 iteration 继续，不再覆盖为 0。
  (9) **深度增强显式化**：评估无条件关闭随机增强，Standard 首轮通过
  `[camera.depth_camera.augmentation] enabled=false` 保持干净控制变量。配置：
  `num_envs=256`（规则上限）；指令域/地形对齐父模型（`vx=[0.3,1.3]`、maze=0%、
  `max_init_terrain_level=9`）；关闭 domain_rand/noise/push 隔离感知迁移。
  **相机外参对齐 21.22° 标准值**：`offset_pos=[0.339871,0.034697,0.075010]` /
  `offset_rot=[0.982631,-0.007085,0.184337,-0.020153]`（pitch≈21.22°），与仓库
  9 处 config + CHANGELOG + docs（standard-distill-1/3 明确标注 "calibrated
  21.22-degree"）一致；lbc_loco.toml 此前用的 ~11.16° 值（`05734d7` 带入）与
  其余视觉阶段冲突，本次统一到 21.22°。本阶段 `deployable=false`，不动 deploy；
  评估 loader 只读 `modules.vision_encoder`，不读 height_scan。平台短训验证
  （步骤 6）与 10h 长训待执行。

- **[Standard 特权网络结构蒸馏]** 从已验证可启动的 minimal lifecycle 重新构建 10288
  flat301→Actor77 桥接。单次 6000-iteration 任务自适应完成
  0/25/50/75/100% 逐环境 DAgger；加入动作安全接管、1/0.25/0 样本权重、
  8192 条 FP16 reservoir、75/25 当前/回放损失和阶段 entry/best/exit 回退。
  新训练包统一为 `kaiwu_train_v1`，保存纯英文阶段文件与同 ID
  `locomotion` 别名；数字 ID 只使用平台注入值。平台完成 6000 iterations，
  最终 `daggerfull` 有效学生比例约 99.8%，视频评估 4/4 完成。四次升档均为
  forced，因此模型能力通过但晋升状态机不复用。后续视觉阶段只登记
  `daggerfull-16288`，并改用失败后停留而非强制升档。
- **[STD-BRIDGE-R1]** 以原始复赛 Standard 10288 flat301 checkpoint 为唯一
  行为教师。预训练教师身份改由操作者手动确认，不再以字节级 SHA
  不一致阻断 rollout；仍保留 key/shape 校验、成功加载要求和冻结教师证明。新增逐环境
  0/25/50/75/100% 单任务 DAgger、安全接管、终止后样本权重和两窗口质量诊断。
  `behavior_distill_v2` 保存学生、冻结教师、optimizer、RNG、DAgger 阶段和配置/
  代码血缘；同时导出不可部署的 `privileged_loco_teacher_v1`，供后续视觉 LBC
  严格加载。平台文件 ID 从父教师 `10288` 继续递增，第一轮立即落探活 checkpoint，
  常规定时保存调整为每 100 iteration 一次，并非新增一套 100 轮阶段；每个
  DAgger 阶段边界额外发布单标签 `bridge`/`teacher` 候选。action/终止/OOD
  阈值只产生 warning 并写入 checkpoint，不再生成 `blocked` 文件或中断后续比例；
  命令越界行权重置零。首轮保持源命令/地形分布并关闭随机化、噪声和 push；
  `configure_app.toml` 明确启用 `/data/pre_model/ckpt` 预加载并以 `10288` 为
  首轮默认 ID；平台注入 checkpoint 优先，未注入时 workflow 从该目录加载最新
  兼容制品，且每个 iteration 调用一次平台 lifecycle callback。平台训练与闭环
  验收待执行。
- **[本地同步可诊断性]** `local_sync_client.py` 新增零网络 `--check-local` 和
  `--refresh-cookie`；Cookie 优先级改为显式值、缓存、旧兼容值、交互输入。
  代理拒绝时打印实际来源，且只在拒绝的是缓存 Cookie 时删除缓存。
  `--dry-run` 继续连接远程但不写入。真实在线验证仍待刷新 Cookie。
- **[STD-D3A]** 从 D2-40min 视觉学生继续 LBC，保持 D2 地形分布与相机外参，
  将动作模仿权重从 `0.2` 提高到 `1.0`，关闭 student-drive，并以
  `2e-4` 学习率在教师驱动的干净轨迹上修复楼梯动作对齐。训练命令按环境
  覆盖 `0.45-0.70 m/s`，阶段内关闭 domain randomization、观测噪声和深度
  增强。`require_student_resume=true` 继续阻止误加载教师或随机初始化学生；
  输出描述性文件与 `model.ckpt-<id>.pkl` 探活别名。
- **[STD-D2-Stair]** 从当前 Standard 视觉学生继续 LBC，只调整训练数据分布：
  `max_init_terrain_level=4`，上/下坡各 5%，上楼梯 30%，下楼梯 60%，
  maze 0%。网络、latent、LSTM、损失、相机外参、深度增强、命令、随机化和
  student-drive 与 D1 完全一致。新增 `require_student_resume=true` 作为预加载
  硬检查，不改变正确续训时的优化行为。输出仍为探活兼容的
  `model.ckpt-standard-<id>.pkl` 与 `model.ckpt-<id>.pkl`。
- **[standard-distill-1，历史]** 该视觉 LBC 实际需要已经完成桥接的 77-D
  `ActorCriticEncoder` 教师；原始复赛 10288 文件已核验为 flat301，不能直接
  进入 LBC。旧 HJC 约 10 分钟产物包含
  `encoder.* / actor.* / critic_encoder.* / critic.*`，LBC 只拆分前两组。训练命令覆盖
  低速前后、横移和双向转动，开启 Standard 地形课程、摩擦随机化、观测
  噪声和深度增强，第一阶段关闭外部 push。相机安装外参为
  `offset_pos=[0.339871,0.034697,0.075010]`、
  `offset_rot=[0.982631,-0.007085,0.184337,-0.020153]`。模型同时保存
  `model.ckpt-standard-<id>.pkl` 与探活兼容别名 `model.ckpt-<id>.pkl`。
- **[checkpoint 兼容修复]** LBC 预加载按平台探活规则识别同一 ID 的带标签
  文件名与任意扩展名，例如 `model.ckpt-hjcnew-10288.pkl`。若候选是结构兼容
  的 Camera/LBC checkpoint，则恢复学生继续蒸馏；否则按平面教师拆分。
  `goal_dim` 和三组 state dict 必须与当前阶段完全一致，不兼容模型仍会拒绝。
- **[遗留路线收敛]** 将 `codex/st7-opt5-debug` 的 Opt5/Opt5-debug/Opt5B
  困难段出生模块、独立 TOML、监控、契约测试和实验文档迁入 `server/`；
  `Config.CURRENT` 保持 ST9-Opt3-D2，不启用 hard-start。补入 ST9-Opt3-D1
  的原始配置与实验文档；D1 的相机、goal-noise、深度增强和蒸馏基础已由
  D2 实现吸收。源提交继续由 annotated tags 保存，远程活动分支统一收敛到
  `main`。
- **[ST9-Opt3-D2]** 从 ST9-Opt3-D1 视觉学生继续训练：迁移 D1 的相机观测、UWB goal 小噪声与持续块状深度空洞链路；蒸馏目标改为 `0.5*latent SmoothL1 + 0.1*cosine + 1.0*raw-action SmoothL1`，教师网络保持冻结。DAgger 改为按环境独立采样的 50%/75%/100% 三阶段学生驱动，TBPTT 序列长度改为 16，高难度完整赛道占比提高，并减轻深度空洞增强。新增 D1 学生 checkpoint 强制续训检查、损失量级日志、监控面板和契约测试。网络、观测顺序、checkpoint 命名与部署接口不变。父 checkpoint = ST9-Opt3-D1 视觉学生；训练与评估结果待平台补录。
- 仓库从扁平结构迁移至 `server/` 目录（2026-07-21）。源基线 = `codex/st7-opt3`（`9b0b3df`）。
- **[J9 接口修复移植]** 源 `codex/stage3j9-fixed-learning-rate`（`68222ad`）。修复 PPO 构造器静默忽略 `schedule`/学习率上下界的 latent bug：`AlgorithmPPO` 现显式接收 `min_learning_rate`/`max_learning_rate`（默认 `1e-5`/`1e-2` 保自适应原行为），`_update_learning_rate` 改用配置界而非硬编码，新增 `_validate_fixed_lr()` 在 `learn()` 入口与出口双查（算法 LR + 所有 optimizer 组）。**策略决策**：仅移植通用接口修复，`TrackNavConfig` 保持 Opt3 自适应 `1.5e-5`（未提升 J9 的固定学习率实验进基线）；J9 固定学习率实验以 `train_env_conf_track_navj9.toml` 作记录，需要时设 `schedule="fixed"` 即生效。
- **[Opt4 移植]** 源 `codex/st7-opt4-angular-rate`（`5f8764d`）。仅调 `dynamic_tilt_risk` 角速度阈值（roll-rate `0.60/1.80->0.50/1.60`、pitch-rate `0.80/2.20->0.65/1.90`）与风险组合权重（`0.25->0.35` roll_rate²、`0.20->0.30` pitch_rate²）。保留 Opt3 goal noise + Opt2B 结构。
- 新增 `docs/st7-opt4.md`（从源分支移植实验笔记）。

## 基线

- **ST7-Opt3**（`9b0b3df`，2026-07-20）：UWB goal noise。Actor 接收几何一致的 UWB 式 goal 噪声（75% 环境的 bearing/distance jitter + per-episode bias），Critic/奖励/完成检查继续用干净真值。父 = ST7-Opt2B（`595693`）。
- 含 ST7-Opt2B `dynamic_tilt_risk`（连续倾斜风险惩罚）、行为蒸馏/LBC 机制（`codex/st7-opt2a` 并入）。

## 仓库迁移状态

- 阶段 3 已完成：`deploy/` 通过保留第二父的 subtree merge 导入，四套目录均有 `ARTIFACTS.md`。
- 阶段 4 已完成：378413、分析报告和旧部署包分别纳入 `archive/` 与 `shared/`。
- 阶段 5：通过 PR merge commit 合入 `main`，随后按分支登记执行带远程 SHA 复核的分支收敛。
- 训练长跑与 Jetson 真机验证属于后续模型发布验收，不作为本次仓库布局合并的阻断条件。
- **[P4 全赛道监控平台校验修复]** 五段出生覆盖和出生位置四分位面板移除平台不接受的中文括号，标题缩短到 20 字符以内；新增 P4 面板标题字符集和长度回归，避免单个非法标题导致平台跳过整份自定义监控配置。
