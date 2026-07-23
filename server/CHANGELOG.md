# server CHANGELOG

本文件记录 `server/` 主线的活动变更。小调参不再创建永久分支，改为：Changelog + 独立 TOML + 实验编号/父 checkpoint/配置哈希/评估结果。

## [未发布]

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
  `daggerfull` 显式排在 `locomotion` 之前，不靠偶然排序命中父模型。配置：
  `num_envs=256`（规则上限）；指令域/地形对齐父模型（`vx=[0.3,1.3]`、maze=0%）；
  关闭 domain_rand/noise/push 隔离感知迁移。**外参待验证**：当前 11.16° 值来自
  `05734d7` 恢复 HJC 最小路径，非单纯误改；21.22° 值在仓库另有 9 处 config/docs
  使用，启动长训前需操作者凭原始标定 + 四元数约定 + 真机安装证据确认。本阶段
  `deployable=false`，不动 deploy；评估 loader 只读 `modules.vision_encoder`，
  不读 height_scan。平台短训验证（步骤 6）与 10h 长训待执行。

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
