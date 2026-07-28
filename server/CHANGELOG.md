# server CHANGELOG

本文件记录 `server/` 主线的活动变更。小调参不再创建永久分支，改为：Changelog + 独立 TOML + 实验编号/父 checkpoint/配置哈希/评估结果。

## [未发布]

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
