# Bug 修复台账

> 维护规则：本文件是仓库级 Bug 根因与修复知识库，长期追加，不按训练阶段另起一份。
> `server/CHANGELOG.md` 记录“改了什么”，本文件记录“为什么坏、如何证明修好、如何防复发”。
> 最近更新：2026-07-27。

## 1. 使用方法

每次开始 Bug 修复前，先用报错原文、模块名、任务名、模型 ID、checkpoint 标签和平台现象
搜索本文件。相似症状不等于相同根因，尤其要先区分：

- 本地文件与容器文件是否一致；
- 容器磁盘上的新文件与运行中 Python 进程已加载的旧模块是否一致；
- 平台拥有并会在启动时覆盖的文件，与仓库可持久同步的文件是否一致；
- 训练 checkpoint、评估 checkpoint 和部署制品是否属于同一种格式；
- 本地测试、容器静态检查、平台 smoke、完整评估和真机验证分别完成到哪一级；
- warning 是根因、伴随症状，还是另一个进程退出后的二次故障。

### 1.1 状态定义

| 状态 | 含义 |
|---|---|
| 调查中 | 现象已确认，但根因或修复尚未闭环 |
| 代码已修复待验证 | 已修改代码，但尚未取得足够运行证据 |
| 本地已验证 | 静态检查/单元测试通过，尚未完成平台运行 |
| 平台已验证 | 容器真实启动或训练日志证明修复生效 |
| 评估已验证 | 同配置评估、视频或真机证据进一步证明行为正确 |
| 已回滚 | 修复无效或副作用不可接受，已恢复到明确基线 |

上传成功、HTTP 200、日志暂时消失、测试 skip、训练 loss 下降或单次总分接近，都不能越级
替代更高层级的验证。

### 1.2 新条目模板

```markdown
## BUG-YYYYMMDD-NNN：标题

- 状态：调查中 / 代码已修复待验证 / 本地已验证 / 平台已验证 / 评估已验证 / 已回滚
- 影响：训练 / 评估 / checkpoint / 同步 / 部署 / 文档
- 首次发现：任务名、模型 ID、时间；未知则写未知
- 症状：关键日志原文、错误分数或文件现象
- 根因：代码路径、状态机或平台边界；未确认时明确写“假设”
- 排除项：曾怀疑但已经证伪的方向
- 修复：文件、符号和行为变化
- 验证：本地、容器、平台、评估分别列出；未执行必须写未执行
- 防复发：测试、日志、哈希、门禁或操作规范
- 血缘：commit/PR、父模型、checkpoint、SHA256
- 回滚：最短可执行回滚方法
- 再遇检查：三到五步最短排查顺序
```

## 2. 历史恢复条目

以下内容由 Git 历史、`server/CHANGELOG.md`、既有复盘文档、平台日志和 2026-07-22 至
2026-07-26 的连续排障记录恢复。较早条目若缺少模型 SHA，会明确标为未知，不补造数据。

## BUG-20260721-001：PPO 构造器静默忽略学习率 schedule 和上下界

- 状态：本地已验证，已进入历史主线。
- 影响：PPO 学习率策略和固定学习率实验可信度。
- 症状：配置声明 `schedule="fixed"` 或自定义最小/最大学习率，但算法仍使用硬编码自适应上下界；
  表面日志中的初始 LR 正确，后续 optimizer param group 可能已经漂移。
- 根因：`AlgorithmPPO` 构造器没有把 `schedule/min_learning_rate/max_learning_rate` 完整传入并
  持久使用，更新函数又引用硬编码范围。
- 修复：构造器显式接收学习率上下界，`_update_learning_rate` 使用实例配置；固定模式在
  `learn()` 入口和出口同时校验算法 LR 与全部 optimizer param group。通用接口由提交
  `4bcc0f0` 移植，审查修复为 `904d79d`。
- 验证：J9 fixed-LR 契约测试覆盖；主线 Track 配置仍保持原 Opt3 自适应策略，未把失败实验参数
  提升为默认值。
- 防复发：任何 optimizer schedule 都必须测试构造参数、一次 learn 前后值和所有 param group。
- 血缘：`4bcc0f0`、`904d79d`、合并提交 `97ac193`。
- 回滚：回滚实验配置而不回滚通用接口修复；不得恢复硬编码上下界。
- 再遇检查：active schedule → algorithm.learning_rate → optimizer groups → learn 前后断言。

## BUG-20260722-001：把 flat301 教师误当作 latent32 教师

- 状态：平台已验证。
- 影响：Standard 视觉蒸馏启动、网络结构和教师加载。
- 症状：原始复赛 Standard `10288` 无法直接作为 `proprio45 + latent32 -> Actor77` 的
  LBC 教师；state dict key/shape 与现有视觉 LBC 入口不匹配。
- 根因：`10288` 实际是 `proprio45 + height_scan256 = 301` 维 flat Actor，而不是带
  `encoder.* + actor.*` 的模块化 77 维策略。训练路线跳过了结构桥接。
- 排除项：不是简单改 checkpoint 文件名、放宽 `strict=False` 或补一个缺失 key 就能解决；
  这样只会掩盖结构语义不一致。
- 修复：先新增 flat301→latent32/Actor77 行为蒸馏桥接，再以模块化学生作为后续视觉教师；
  相关提交为 `07d10bc`，后续由 Standard DAgger R2 完成结构迁移。
- 验证：R2 完成 6000 iterations，`daggerfull-16288` 有效学生驱动约 99.8%，固定视频
  评估 4/4 完成。
- 防复发：checkpoint loader 校验模型格式、关键 state dict 和输入维度；训练计划必须写明
  flat301、Actor77、Actor80 的边界。
- 血缘：父模型 `10288`；桥接结果 `daggerfull-16288`；具体制品 SHA 未在本条恢复。
- 回滚：回到冻结的 `10288`，重新执行结构桥接，不得把视觉阶段 checkpoint 当结构桥接输入。
- 再遇检查：先打印 checkpoint keys → 核对输入维度 → 核对 teacher/actor/encoder 模块 →
  再检查文件名和加载路径。

## BUG-20260722-002：带标签 checkpoint 未被预加载发现

- 状态：本地已验证，历史平台路径已使用。
- 影响：LBC 续训和教师预加载。
- 症状：文件实际存在，例如 `model.ckpt-hjcnew-10288.pkl`，但 loader 只查找旧的固定文件名，
  导致误判“模型不存在”或错误落入随机初始化/其他候选。
- 根因：候选生成没有覆盖平台探活允许的标签文件名和扩展名，也没有坚持同 ID 选择。
- 修复：扩展 checkpoint candidate 规则，识别同 ID 的带标签文件；结构兼容的 Camera/LBC
  文件恢复视觉学生，否则按 flat 教师路径加载。相关提交 `8a4905f`。
- 验证：候选顺序与同 ID 约束由 checkpoint 单元测试覆盖；历史视觉蒸馏已使用该路径。
- 防复发：新增任何保存标签时必须同时补 candidate-order 测试；禁止通过“latest”跨 ID 猜测。
- 血缘：commit `8a4905f`；具体父 checkpoint 视任务而定。
- 回滚：显式选择已知同 ID 文件，不能回退到无身份校验的目录扫描。
- 再遇检查：列出目录 → 解析请求 ID → 打印完整候选 → 打印 selected path → 校验 bundle ID。

## BUG-20260722-003：模块化教师加载遗漏 encoder/误装载 critic

- 状态：本地已验证，历史训练已采用。
- 影响：Standard 视觉蒸馏教师初始化。
- 症状：checkpoint 包含 `encoder.* / actor.* / critic_encoder.* / critic.*`，旧加载逻辑
  不能正确拆出视觉蒸馏所需的低层 encoder 与 actor，或把 critic 侧键混入教师。
- 根因：loader 没有按训练角色划分 state dict，只按宽泛前缀或平面模型假设加载。
- 修复：明确只提取 encoder 与 actor，忽略 critic 侧键，并为模块/shape 不兼容保留失败路径；
  相关提交 `193deae`。
- 验证：`test_encoder_teacher_checkpoint_ignores_critic_side_keys` 等回归测试覆盖。
- 防复发：训练恢复包、特权教师包和部署包使用不同契约；不能用 `strict=False` 作为兼容策略。
- 血缘：commit `193deae`。
- 回滚：回到已确认的模块化 HJC/bridge 教师，不使用无法解释 key 语义的 checkpoint。
- 再遇检查：打印 format → 模块列表 → 每组 key 数 → shape → 实际加载模块。

## BUG-20260723-001：同步包遗漏 `isaac_env`，训练无法进入真实 workflow

- 状态：平台已验证；后续平台边界见 BUG-20260725-003。
- 影响：训练启动、环境配置和本地到容器同步。
- 症状：Agent、Algorithm、Workflow 文件都在，但平台长期没有 BehaviorDistill iteration；
  minimal 包只有 `isaac_env/__init__.py`，缺少环境实现。
- 根因：早期同步范围只有 `agent_diy`、`agent_ppo`、`conf`，没有 `isaac_env`。训练包不是
  只有模型代码，环境 wrapper 和配置应用逻辑同样是运行入口。
- 修复：当时恢复完整环境目录并把 `isaac_env` 纳入同步范围，commit `3c2390a`。
- 验证：修复后平台连续打印 `BehaviorDistill iter=30/40/50...`，MSE 下降、cosine 接近 0.99。
- 防复发：同步 manifest 固定显示四个同步目录；项目根必须是 `server/`；平台拥有的
  `base_env.py` 后续改为精确排除，而非排除整个 `isaac_env`。
- 血缘：commit `3c2390a`；minimal 起点后续为 `7d48cba`。
- 回滚：不能回到“三目录同步”；若平台文件归属变化，应调整精确保护列表。
- 再遇检查：看 sync dirs → 看 remote manifest → 查环境入口文件 → 查运行日志中的 config path。

## BUG-20260723-002：HTTP 200 空响应被当成上传成功或失败，远端状态不确定

- 状态：平台已验证。
- 影响：同步可靠性，曾在 `algorithm_ppo.py`、`base_env.py` 等文件上出现。
- 症状：`{'ok': True, 'empty_response': True, 'status': 200}`；客户端无法判断代理丢了响应体、
  写入未完成，还是远端文件已经正确落盘。
- 根因：旧客户端把 HTTP 状态/响应体当成最终事实，没有回读目标文件。
- 修复：空响应后调用远端 read，比较本地与远端 SHA256；只有哈希一致才成功。提交
  `d2bf115`。
- 验证：同步单元测试覆盖空响应后哈希一致与不一致；真实平台同步验证通过。
- 防复发：任何上传 transport 都必须以远端最终哈希为准，不能只信 200；dry-run 必须打印
  具体待覆盖文件。
- 血缘：commit `d2bf115`。
- 回滚：不允许回滚到无回读校验的上传实现。
- 再遇检查：看 HTTP status → 远端 read → SHA256 → 第二次 dry-run 应为 0 文件。

## BUG-20260723-003：自定义蒸馏绕开开悟 lifecycle，步数和模型池不更新

- 状态：平台已验证。
- 影响：`train_global_step`、平台模型 ID、定时保存和模型列表。
- 症状：BehaviorDistill loss 正常下降，但训练步数停在预加载模型附近，模型池 ID 不推进，
  保存列表没有预期模型。
- 根因：自定义 workflow 直接 `env.step` 和 optimizer update，没有经过普通 PPO 的
  `Agent.learn()` lifecycle；业务代码又一度尝试人工计算模型数字 ID。
- 修复：每个完整 outer iteration 完成内部更新后调用一次 no-op `agent.learn(None)` 作为
  框架 lifecycle callback；保存时不手工传数字 ID，由平台注入。提交 `7d48cba`。
- 验证：平台 `train_global_step` 与模型池恢复推进，minimal bridge 可以保存模型。
- 防复发：iteration 语义固定为“一个 rollout/一轮内部更新后的 outer iteration”；自定义
  workflow 必须单测 callback 次数；禁止 `parent_id + iteration`。
- 血缘：commit `7d48cba`。
- 回滚：回到 `7d48cba` 的 lifecycle 模式，不回到手工 ID。
- 再遇检查：训练 loss → lifecycle callback 次数 → `train_global_step` → save_model 入参 →
  平台模型列表。

## BUG-20260723-004：checkpoint 标签含数字或命名不符合平台探活正则

- 状态：平台已验证。
- 影响：checkpoint 被平台发现、模型 ID 解析和任务列表展示。
- 症状：`dagger25`、`dagger-25` 等名称可能不匹配平台阶段标签 `[a-z]*`，或数字被误识别
  为模型 ID。
- 根因：把阶段比例直接编码成文件名数字，没有按平台探活语法设计。
- 修复：阶段标签改为纯小写英文，例如 `daggerzero/quarter/half/threequarter/full`、
  `visionteacher/half/full`、`anchorcritic/...`；实际比例写入 checkpoint 字段。
- 验证：五阶段文件名通过 probe regex 测试，R2 产出 `daggerfull-16288`。
- 防复发：所有新标签加入 `validate_probe_filename` 测试；数字只允许出现在平台注入的末尾 ID。
- 血缘：相关提交 `fa15aca` 及后续 R2/视觉集成提交。
- 回滚：改名为纯英文阶段标签并保留 checkpoint 内的数值状态。
- 再遇检查：正则 → 文件名最后数字 → bundle ID → 平台列表。

## BUG-20260724-001：视觉 LBC iteration 在 inner step 上推进，保存与训练时长快约 24 倍

- 状态：平台已验证。
- 影响：视觉长训计数、17000/18000/19000 等模型 ID、保存节奏和 LR 调度。
- 症状：约 5 分钟日志从 17000 到 18000，迭代明显快于上一阶段；内部 checkpoint 日志存在，
  平台模型列表却不一定同步出现。
- 根因：平台 lifecycle 在每个 inner environment step 调用一次，而配置/上一阶段把一个完整
  outer iteration 作为一个训练步。`num_steps_per_env=24` 放大了计数。
- 修复：每个完整 outer iteration 只调用一次 lifecycle；checkpoint 写入
  `iteration_semantics=completed_outer_iterations_v1`；旧零基 iteration 兼容转换；保存间隔
  按实测 outer iteration 调整为约 10 分钟。提交 `049f6b1`（等价集成提交 `97fe7a9`）。
- 验证：视觉长训计数恢复到 outer iteration 语义，相关 smoke/unit tests 通过。
- 防复发：日志同时打印 outer iteration、inner steps、wall-clock 和模型 ID；保存间隔以墙钟
  复核，不能只看整数步数。
- 血缘：commit `049f6b1`；视觉训练包 `kaiwu_train_v1`。
- 回滚：恢复到修复后的 outer lifecycle，不使用修复前产生的 iteration 推断训练进度。
- 再遇检查：单 iteration 耗时 → `num_steps_per_env` → lifecycle 调用位置 → 保存 ID 间距。

## BUG-20260724-002：长训被全局 `frame_no > 1250` 误判为 episode 完成

- 状态：平台已验证；实现边界后来迁移到 workflow。
- 影响：长时间训练持续性和 `reached max length` 日志。
- 症状：训练超过 1250 帧后重复出现
  `Episode done trigger: reached max length, frame_no=..., max_length=1250`。
- 根因：`frame_no` 是一次 `env.reset()` 后的全局 workflow 计数，不是每个 Isaac 子环境的
  episode length。底层已经逐环境 auto-reset，把全局 frame 当 episode 会持续误报 `all_done`。
- 修复：首轮在 Standard LBC 配置 `continuous_training=true`，训练时不因全局 frame 结束，
  评估仍保留单 episode 结束语义；确认平台会恢复原始 `base_env.py` 后，将活动实现迁到训练
  workflow，直接忽略训练态 `infos['all_done']`，不再依赖修改平台文件。
- 验证：单元测试断言训练 rollout 不读取该全局标记；平台长训继续推进。日志本身仍可能由平台
  原始环境打印，但不再作为训练退出条件。
- 防复发：区分 per-env `terminated/truncated`、底层 auto-reset 和全局 workflow frame；不要用
  “日志仍出现”单独判断训练停止。
- 血缘：`049f6b1` 及当前 `train_workflow.py` 未发布修复。
- 回滚：保留 workflow 侧兼容；不得再次依赖定制 `base_env.py` 持久化。
- 再遇检查：看 iteration 是否继续 → 看 hard termination → 看 rollout 是否 break → 再看日志。

## BUG-20260724-003：视觉续训没有完整恢复时钟、调度器和 RNG 状态

- 状态：本地已验证，平台视觉续训已采用。
- 影响：断点续训的 ramp、LR、安全阈值和可复现性。
- 症状：续训加载了网络和 optimizer，但 `ramp_clock_h` 归零、安全阈值重新标定、LR scheduler
  从头开始，可能把已退火模型当首训继续。
- 根因：旧视觉 checkpoint 只关注权重，缺少训练状态；某些 legacy 分支还会把部分加载误标为
  `resume_loaded=True`。
- 修复：`kaiwu_train_v1` 增加并恢复 ramp/session clock、LR scheduler、安全阈值与标定样本、
  RNG、iteration 和 resume mode；LSTM hidden 按明确契约在新环境 reset 后清零，不跨运行恢复。
- 验证：保存—恢复、scheduler 和 legacy iteration 测试覆盖；本机无 PyTorch 时相关测试必须
  显式显示 skip，不能声称 tensor 测试通过。
- 防复发：checkpoint schema 新字段必须有 round-trip 和旧格式 fallback 测试；`resume_loaded`
  只能表示相应状态真的恢复。
- 血缘：commit `1961272`/集成 `3219238` 与后续视觉长训修复。
- 回滚：若状态不兼容，必须明确按 schedule migration/S0 启动，不能静默伪装完整续训。
- 再遇检查：load mode → restored iteration → session clock → LR → safety_fixed → RNG 日志。

## BUG-20260724-004：深度相机外参在配置间漂移为约 11.16°

- 状态：本地已验证，视觉长训采用统一配置。
- 影响：深度视觉输入分布和 Sim2Real 一致性。
- 症状：`lbc_loco.toml` 中外参与其他九处配置/文档不一致，pitch 约 11.16°，而确认标准为
  21.22°。
- 根因：从历史 minimal/HJC 路线带入了旧外参，后续配置复制没有统一校验。
- 修复：统一为 `offset_pos=[0.339871,0.034697,0.075010]`、
  `offset_rot=[0.982631,-0.007085,0.184337,-0.020153]`，并同步文档和测试。
- 验证：配置扫描与静态测试通过；长训实际采用该外参。
- 防复发：相机外参作为跨训练/部署契约管理，不允许仅改某个 TOML。
- 血缘：commit `4a658d2`（等价 `0f307ed`）。
- 回滚：回到标定的 21.22° 标准，不使用来源不明的历史四元数。
- 再遇检查：打印 active config path → offset → 四元数顺序 → pitch → 部署标定。

## BUG-20260724-005：同步凭据来源混乱，`.env` 未自动加载或两端 token 不一致

- 状态：本地已验证，真实同步已验证。
- 影响：本地同步启动和容器 `unauthorized`。
- 症状：`missing token: pass --token or set IDE_SYNC_TOKEN`，或本地已有 Cookie 仍返回
  `HTTP 400 unauthorized`。
- 根因：同步 token 与腾讯网页登录 Cookie 是两层认证；旧脚本的 token 来源/根目录不稳定，
  本地和容器也可能使用不同值。Cookie 有效不能弥补 sync token 不匹配。
- 修复：客户端按 CLI → 环境变量 → `server/conf/.env` 加载 `IDE_SYNC_TOKEN`，只打印来源和
  截断哈希；服务端启动脚本读取同一键；Cookie 使用独立缓存和刷新逻辑。禁止把真实凭据写入
  受版本控制源码。
- 验证：dotenv 不执行 shell 的单元测试通过；真实同步输出 token source 后成功连接。
- 防复发：`.env` 必须被 `.gitignore`；日志不打印 token；排障先区分 401 proxy 与 400 sync
  unauthorized。
- 血缘：commit `94cae90`/`b0ed556` 及当前同步工具未发布改动。
- 回滚：删除错误的本地 `.env` 并在两端生成同值，不回退到硬编码源码凭据。
- 再遇检查：token source → 截断 SHA → 容器 server source → Cookie source → HTTP 层级。

## BUG-20260724-006：POST 经开悟网关丢失，逐文件全量上传耗时且不可靠

- 状态：平台已验证。
- 影响：同步性能和大文件可靠性。
- 症状：约 110 个文件每次上传约 70 秒；POST 可能返回 200 空 body，`base_env.py` 等大文件
  出现“失败”但远端状态不明。
- 根因：开悟 IDE 网关对 POST body 转发不稳定，旧客户端又没有先做并发 manifest 哈希差异。
- 修复：先读取远端 manifest 做增量 SHA 比较；变更文件通过 GET chunk/bundle 并发传输；bundle
  完成后校验文件数量、大小和 SHA；第二次 dry-run 必须为 0。
- 验证：6 文件同步约 0.6 秒，74 文件约 10.8 秒，单文件修复约 0.3 秒；同步后哈希复核为 0
  待覆盖文件。
- 防复发：保留 GET transport、增量 manifest、结束回读和失败重试；禁止仅因 POST 理论上更快
  就切回未验证 transport。
- 血缘：当前 `server/local_sync_client.py`、`server/conf/tongbu.py` 未发布改动。
- 回滚：使用已验证的 GET chunk 模式，不使用无最终哈希的旧 POST。
- 再遇检查：manifest 数 → overwrite 列表 → transport → bundle verified → 第二次 dry-run。

## BUG-20260724-007：同一平台 ID 保存阶段文件和兼容别名，下载包出现两个模型

- 状态：本地已验证；新 VisualPPO/Camera eval 路径待平台完整复验。
- 影响：checkpoint 下载、候选选择、模型身份和操作者判断。
- 症状：一次平台保存后下载目录同时包含阶段模型和 `locomotion`/`lbc-loco` 兼容模型；操作者
  无法判断后续应该使用哪个，额外里程碑保存日志也不一定出现在平台模型列表。
- 根因：旧实现为同一平台 ID 既写真实训练 bundle，又复制一个兼容别名来适配旧 loader；平台
  lifecycle 只登记标准保存回调，业务侧额外写文件不能保证被模型池登记。
- 修复：VisualPPO 每次保存只写一个带纯英文阶段标签的 `kaiwu_train_v1` 训练包；候选 loader
  直接支持 `command/anchor/rl/vision` 标签，不再依赖同 ID 别名；Camera LBC eval 也禁止创建
  `lbc-loco` 别名。保存仍由平台 lifecycle 注入数字 ID。
- 验证：单元测试断言 VisualPPO save 不使用 `shutil.copyfile`/side alias，Camera eval 候选覆盖
  所有阶段标签；平台重新下载复验尚未执行。
- 防复发：兼容性应由 loader 的显式候选规则实现，不应复制同一权重制造多个身份相似文件。
- 血缘：当前 `checkpoint_io.py`、`agent.py`、VisualPPO workflow 未发布改动。
- 回滚：回到唯一训练包；旧模型需要兼容时只扩展读取，不新增写入别名。
- 再遇检查：同 ID 文件清单 → 每个文件 format/SHA → save callback 次数 → loader candidate 顺序。

## BUG-20260725-001：Anchor/Command 模式仍校验旧 anchor phase boundary，启动即失败

- 状态：本地已验证，后续平台任务可启动。
- 影响：VisualPPO 初始化。
- 症状：`ValueError: Anchor phase boundary falls outside the anchor schedule`，随后 monitor proxy
  出现大量 Broken pipe。
- 根因：新 schedule mode 没有使用旧 Anchor phase knots，但构造器仍无条件校验旧 phase
  boundary。Broken pipe 是主进程初始化退出后的二次症状，不是根因。
- 修复：按 `schedule_mode` 分流配置和校验；command 模式使用固定 anchor 与 command phase，
  只有 Anchor 模式校验 anchor knots/boundaries。
- 验证：Python/TOML 测试通过，后续启动越过该构造阶段。
- 防复发：每种 schedule mode 都必须有最小构造测试；日志错误按最早 traceback 排序。
- 血缘：当前未发布 VisualPPO 工作树；具体 commit 未生成。
- 回滚：使用与当前 mode 相匹配的配置，不通过删除所有校验绕过。
- 再遇检查：第一条 traceback → schedule_mode → knots → phase ends → monitor 错误是否仅为伴随。

## BUG-20260725-002：`resampling_time=1000000` 超出平台配置范围，Agent 尚未创建

- 状态：平台已验证到配置校验阶段。
- 影响：命令 scheduler 训练启动。
- 症状：`resampling_time[0/1]=1000000.0 out of valid range [0.0,300.0]`，aisrv/learner 同时退出。
- 根因：为了阻止原生命令重采样使用了超大哨兵值，但平台在创建 Agent 前有严格 TOML 范围
  校验，代码根本没有机会接管命令。
- 修复：fallback 改为合法 `[300,300]`；真实 1.5–10 秒保持时间由经过验证的 agent-side
  scheduler 管理。writer 不可用时退回原生 source 域。
- 验证：配置校验通过，Agent 能创建；writer 的运行验证属于后续独立条目。
- 防复发：配置必须先过平台 schema，不能用 schema 外数值表达“禁用”。
- 血缘：当前活动 TOML 未发布改动。
- 回滚：保留合法 300 秒 fallback；若 agent writer 不可用，不伪造 target 指标。
- 再遇检查：错误发生层级 → schema 范围 → active TOML → scheduler 是否已实际启动。

## BUG-20260725-003：平台启动覆盖 `isaac_env/base_env.py`，本地修复反复失效

- 状态：平台已验证，长期边界已固化。
- 影响：episode lifecycle、command buckets、深度配置和同步可信度。
- 症状：本地/容器手工复制后短暂存在，重启容器又恢复旧版本；下载代码包中 `isaac_env`
  只有 `__init__.py`；日志 line 2635 仍是平台旧实现。
- 根因：`base_env.py` 是平台 bootstrap 拥有并在启动时重建的文件，不是普通用户代码。同步成功
  只证明某个时刻磁盘写入，不证明下一次启动仍会保留。
- 修复：把训练逻辑迁移到 `agent_ppo`；同步客户端、服务端、单文件、bundle 和删除入口全部精确
  保护 `isaac_env/base_env.py`，同时保留其余 `isaac_env` 文件同步。
- 验证：单元测试覆盖所有 mutation endpoint 的拒绝；真实 dry-run 不再列出该文件。
- 防复发：功能不得依赖修改平台文件；允许读取平台初始 wrapper 结构，但要有兼容测试和运行日志。
- 血缘：当前同步工具和 VisualPPO 工作树未发布；平台基线 SHA256
  `75ebdaf6888e94262598a26db1586b2598cb474422e3382b6bba9e96ddbb6e67`。
- 回滚：回滚 `agent_ppo` 功能时也不能重新开放 base_env 上传。
- 再遇检查：容器重启前后 SHA → sync protected list → 功能是否仍依赖 base_env diff。

## BUG-20260725-004：Camera 评估被平台切到 `lbc_loco`，指定视觉模型没有加载

- 状态：本地已验证，待平台重新评估。
- 影响：Camera 模型评估和所有由错误评分做出的晋级判断。
- 首次发现：平台请求模型 ID `30531`；日志包 `log-597573-18493013.zip`。
- 症状：`_find_vision_eval_ckpt()` 在 `torch.load()` 前失败，没有任何 checkpoint 成功加载日志，
  程序仍以未正确初始化模型完成评估并得到 38.45；该分数无效。`visionteacher-0` 是启动时保存，
  不是被加载的 30531。
- 根因：Camera task 被平台强制映射到 `lbc_loco`，当前 LBC eval 回归到旧
  `vision_checkpoint_candidates()`，丢失 requested model ID 和 bundle identity 校验；失败后 inference
  没有 hard stop。
- 修复：兼容平台 `lbc_loco` 入口但使用 `visual_eval_checkpoint_candidates()`，同 ID 顺序
  `command* -> anchor* -> rl* -> vision*`；只加载 `vision_encoder` 与 low-level Actor；严格校验
  `platform_model_id`/lineage，打印候选、selected path、SHA256、bundle ID、lineage；任何查找、
  反序列化、结构或身份失败都终止评估；不创建额外 `lbc-loco` 别名。
- 验证：本地 `88 tests OK (skipped=20)`、compileall、diff check；平台重新评估尚未执行。
- 防复发：Camera 强制 LBC 入口为永久兼容路径；禁止失败后用随机初始化参数继续评分。
- 血缘：参考已成功的 `standard-cmd_29611.zip`；当前修复未提交。
- 回滚：评估失败就停止并回到已知可加载的 29611/28401，不接受无加载证据的分数。
- 再遇检查：requested ID → candidate list → selected path → SHA/bundle/lineage → loaded modules。

## BUG-20260725-005：原生 source fallback 错误包含 `vx=0`

- 状态：本地已验证，活动 TOML 已同步。
- 影响：命令泛化前三十分钟和 writer 不可用时的训练分布。
- 症状：计划要求 source `vx=[0.3,1.3]`，但 fallback 一度使用 `[0.0,1.3]`，即使 target writer
  不可用也会提前混入零速/低速，无法隔离变量。
- 根因：混淆了原生 sampler 的 `commands.ranges` 与两套 profile 并集的 `commands.limit`。
- 修复：`ranges.lin_vel_x=[0.3,1.3]` 保持原始 source；`limit.lin_vel_x=[0.0,1.3]` 只作为自定义
  scheduler target 的合法边界；vy/wz 保持 source 与 target 并集一致。
- 验证：TOML 测试断言 ranges 与 limit 分离；同步后远端哈希一致。
- 防复发：启动日志同时打印 native ranges、global limit、requested/effective target probability。
- 血缘：当前活动 TOML未发布改动。
- 回滚：writer 不可用时明确 source-only，不把 limit 当 native range。
- 再遇检查：active config → ranges → limit → hook status → effective bucket counts。

## BUG-20260725-006：零速样本与前进距离课程机制耦合，高难度样本可能被饿死

- 状态：本地已验证；关闭课程后的长训分布待平台验证。
- 影响：terrain difficulty 分布、零速训练和楼梯能力保持。
- 症状：命令 profile 含大量零速时，基于 episode 前进距离判断晋升的 curriculum 无法区分
  “正确停车”和“没有能力前进”；零速环境长期不能晋升，训练样本偏向低难度。timeout 也可能被
  误读为失败，进一步干扰难度分布判断。
- 根因：课程晋升指标假设每个环境都被要求前进，而新 command 域包含 zero/pure-yaw/lateral；
  任务目标与 curriculum 的成功代理不再一致。
- 修复：命令泛化/恢复阶段设置 `curriculum=false`，256 环境在 reset 时覆盖 0–9 难度并记录
  level histogram；难度余数采用长期统计均匀，不新增依赖平台 `base_env.py` 的严格配额器。
- 验证：活动 TOML 和静态测试确认 curriculum 关闭、num_envs=256；真实 terrain×difficulty
  histogram 和长训楼梯保持仍待平台日志确认。
- 防复发：改变 command 语义时必须审查 curriculum 的晋升代理；timeout 与 hard termination
  分开统计，零速 timeout 不能直接算摔倒。
- 血缘：当前 `standard-command-r1` 工作树，父模型计划为 28401。
- 回滚：若固定均匀分布造成训练不稳，回到明确的静态难度上限或按 command-aware 指标重新设计，
  不直接恢复旧的距离 curriculum。
- 再遇检查：curriculum flag → level histogram → command bucket×level → timeout/hard-term 分离 →
  分难度完成率。

## BUG-20260726-001：aisrv `CommandAdapter` 无法穿过跨进程环境代理

- 状态：最终 worker 侧修复已完成本地静态验证，容器 PyTorch 测试与平台 smoke 待执行。
- 影响：目标 command 是否真正进入环境 reward getter、policy observation 和 critic observation；
  错误实现会使 target profile 永远不生效，或造成 reward 与观测中的 command 不一致。
- 症状：`aisrv [CommandAdapter] verified command writer unavailable; target commands are not applied
  and effective_target_probability=0.0`，以及 `command_hook=unavailable`。
- 最终根因：`workflow`/`CommandAdapter` 位于 aisrv，所持 `env` 是跨进程代理；真实 Isaac
  `command_manager` 位于环境 worker。aisrv 递归搜索 `env/_env/_gym_env/unwrapped` 不能跨越进程
  边界。此前用本地 fake `Robot._gym_env` 对象证明递归可达，只证明了同进程测试拓扑，不能证明
  平台架构可达，因此“补 `_gym_env` edge”是错误修复路线，已废弃。
- 排除项：不能只在 aisrv 改 policy/critic observation，否则 policy 看到的新 command 与 worker
  reward 使用的原 command 分离；不能访问 command term 私有 `_command`；不能修改会被平台覆盖的
  `isaac_env/base_env.py`；继续增加 wrapper 属性候选也不会解决进程边界。
- 最终修复：删除 aisrv `CommandAdapter` 初始化和 observation patch。新增 env-owned
  `WorkerCommandBridge`，由实际视觉入口 `LBCObservationProcess`、普通
  `PolicyObservationProcess` 和 `CriticObservationProcess` 在 `default_observation()` 前调用。
  bridge 复用纯 Torch `CommandSchedule`，通过公开
  `command_manager.get_command('base_velocity')` 写 live tensor并立即回读；
  `common_step_counter` 保证 policy/critic 任一先调用时同一步只调度一次，
  `episode_length_buf == 0` 只重采样 reset 环境。aisrv 改为从真实 policy observation `[6:9]`
  计算 S0 anchor 权重，不再报告无法观测的 effective source/target 指标。
- 恢复与失败语义：每个训练任务创建全新的 worker scheduler，ramp 从 0 分钟开始；checkpoint
  不保存或恢复每环境 command、hold、bucket、worker RNG 或 scheduler pending state。模型、critic、
  optimizer 与训练 RNG 仍按既有训练包恢复。写入或回读失败时先恢复调用前的原生命令并禁用本任务
  自定义调度；只有原 command 也无法恢复或回读时硬停止，禁止在无法证明一致时继续训练。
- 本地验证：worker 测试覆盖 policy-first/critic-first、同 step 幂等、部分 reset、未到期保持、
  原生覆盖后的重新发布、getter 返回副本、写入异常、恢复失败、配置关闭和 observation-derived
  anchor 映射；静态测试禁止 aisrv `CommandAdapter`、私有 `command_manager._command` 和 checkpoint
  scheduler state。当前本机无 PyTorch 的测试显式 skip；容器测试与 runtime smoke 仍是验收项。
- 预期平台证据：启动出现 `WorkerCommandBridge status=active`，command tensor shape 正确；每 500
  step 的 `readback_error_max`、`policy_error_max`、`critic_error_max` 均不超过 `1e-5`。前 30 分钟
  `requested_target_probability=0`，30 分钟后 effective target samples 与对应 bucket 计数开始增长；
  日志不得再出现 `CommandAdapter command_hook=unavailable`。
- 防复发：command 的唯一运行时 owner 是环境 worker；跨进程代理不得通过递归属性猜测能力。
  测试必须覆盖实际 observation 扩展入口和 live readback，不能用同进程 fake wrapper 代替平台证据。
  同步后必须重启 aisrv/learner，运行中 Python 不会热加载新模块。
- 血缘：分支 `codex/visual-command-generalization`；当前未提交；父模型计划为 28401。
- 回滚：关闭 `[commands.worker_progressive].enabled` 即保留原生 source sampler；不得回滚到 aisrv
  adapter、observation-only patch 或平台 `base_env.py` 修改。
- 再遇检查：确认新进程 → worker bridge active → tensor shape → readback/policy/critic error →
  30 分钟前后 target probability → effective sample/bucket counts。

## BUG-20260727-001：hier-nav 阶段不可达、Oracle 不会绕墙且契约未闭环

- 状态：本地已验证，平台 S0/Oracle-only smoke 待执行。
- 影响：hier-nav 训练入口、Camera 评估、Oracle 标签质量、难度分布、checkpoint
  续训语义和 soft-stay 安全诊断。
- 症状：`train_env_conf_track_nav_dagger.toml` 虽声明 `policy_entry=nav_dagger`，训练仍从
  `Config.CURRENT=StandardVisualPPOConfig` 拼接其他 TOML；Camera 被平台送入 `lbc_loco`
  时 nav 组合包不可评；Oracle 遇到正前墙只会降速直行；TOML 声明的
  `[terrain.level_mix]` 实际无任何消费者。
- 根因：训练 TOML 路径包含 stage name，因此文件内的 `policy_entry` 不可能反向
  选择自己；旧 eval 回退只识别 loco/visual 命名空间；critic obs 只有近场
  height scan 与 goal，worker 的 `nav_scanner` 没有跨进程送到 Oracle；历史配置
  表只有注释，平台 loader 从未实现 level_mix。同时 checkpoint 只校验少量字段，
  hard termination 又错用 env frames 作分母。
- 修复：`conf/configure_app.toml [app].policy_entry` 作为 stage-specific TOML 之前的唯一
  训练 bootstrap，本分支固定 `nav_dagger`/`34728`；Track+Camera 无显式入口时推导
  `NavEvalConfig`，aisrv 的强制 LBC 路径发现同 ID nav 包后内存升级到完整 nav
  装配。`NavCriticObservationProcess` 把 nav scanner 压成
  `[available,front,left,right]` 四个特权分数，critic 由 319 扩为 323 维；Oracle
  在前墙阻塞时朝墙分数更低侧 creep 转向，平局固定向左，配合 2s dwell
  避免摇摆。删除假 level_mix，改用 10 条原生并行轨道与实测 histogram。
  checkpoint 额外锁定输入切片、词表名称/顺序、网络维度、UWB 测量链、
  clamp/slew/zero 语义；resume 以包内 `low_level_parent_model_id` 为真实血缘。
  soft-stay 改用 `hard_events/completed_episodes`与 `timeouts/completed_episodes`，无完成
  episode 的窗口不伪造 0% 死亡率。
- 排除项：不修改平台会覆盖的 `server/isaac_env/base_env.py`；不在 aisrv 搜索
  worker 私有对象；不通过复制同 ID 别名解决 Camera 评估；不保留无消费者的
  TOML 字段伪造可配性。
- 验证：新回归测试覆盖训练 bootstrap、Track+Camera 无显式入口、强制 LBC
  升级防线、nav scanner 缺失硬停、左/右/平局避障、checkpoint 契约篡改、
  父血缘恢复和 episode 分母。本地 nav 套件 `70 passed, 3 subtests passed`；平台
  Track+Camera+256 已能进入真实更新；Oracle-only 首轮为 0 完成/256 超时，未通过
  迷宫完成率门禁。后续按 BUG-20260727-003 固定 128 环境继续验证组合 checkpoint 评估。
- 防复发：stage 选择必须在 stage TOML 路径拼接前有日志；新增 TOML 表必须能指向
  实际消费函数；Oracle-only 迷宫完成率是 DAgger 开训前门禁；任何改变
  `nav_contract.py` 推理语义的修改都必须使旧包在 resume/eval 阶段显式拒绝。
- 血缘：分支 `codex/hier-nav-dagger`；低层父 `command-34728`；本次修复尚未提交。
- 回滚：回到 `command-34728` 低层评估基线，不运行未通过 Oracle-only 门禁的
  DAgger；不回滚到修改 `base_env.py`、旧 LBC 默认评 nav 或无契约检查的加载路径。
- 再遇检查：stage bootstrap 日志 → 实际 TOML 路径/维度 → nav scanner 特权可用率 →
  Oracle-only 迷宫完成率 → checkpoint selected/SHA/lineage/contract → episode outcome 分母。

## BUG-20260727-002：hier-nav Agent 已创建但 workflow 未进入

- 状态：已修复，并在开发容器 8 环境完整训练链验证；256 环境正式平台任务仍按
  独立 smoke/长训验收，不把 8 环境结果扩大解释为吞吐验证。
- 影响：hier-nav DAgger 首轮 Track+Camera 训练无法产生 rollout、高层更新或
  checkpoint。
- 任务与证据：平台任务 `navdagger-r1`（task ID `234657`，低层父
  `command-34728`），2026-07-27 01:25 启动。日志明确出现
  `Stage: nav_dagger, task_type: track`、正确的
  `train_env_conf_track_nav_dagger.toml`、冻结 VisionEncoder/Actor77、可训练
  HighLevelPolicy 以及 `AlgorithmNavDagger ready`；learner 也打印
  `preload_model_file success ... id is 34728`。但至少 12 分钟内只有
  `learner_proxy send sample stat, succ_cnt is 0, error_cnt is 0`，且搜索不到
  `[nav] first-load low-level parent`、`[NavDAgger] start`、`reset ok`、`iter=1`
  或 traceback。
- 当前判断：停点位于 Agent 构造开始之后、用户 nav workflow 第一条
  日志之前。根因尚未定案；需要区分 `BaseAgent.__init__` 未返回、平台
  没有调用自定义 `load_model`、模型池首次同步未完成、以及 workflow 根本未
  分发四种可能。
- 第二轮平台证据：任务 `navdagger-r1` / `234666` 中 learner
  在约 `3.3s` 内打印 `agent_init before_base_agent` 与 `complete`；aisrv
  已打印 VisionEncoder、LowLevelActor 和 HighLevelPolicy，但没有
  `AlgorithmNavDagger ready`、`agent_init before_base_agent`、`load_model`或 workflow
  日志，持续 8 分钟且无 ERROR。因此停点已缩小到 aisrv 的
  `AlgorithmNavDagger.__init__`，排除 `BaseAgent`、checkpoint、workflow、
  `env.reset` 和平台 `base_env.py`。
- 已排除：日志中的 `<tools.base_env.base_env.Robot>` 是平台正常 wrapper
  类名，不能证明仓库 `server/isaac_env/base_env.py` 生效或造成卡死；环境进程
  已打印 `isaac_env start`，但尚无 `env.reset` 调用证据。`succ_cnt=0` 在
  VisualPPO/DAgger 直接更新模式下也可正常，必须结合专用 iteration 日志判断。
  `preload_ratio=1` 与上一轮成功 VisualPPO 配置相同，不是本轮新引入变量。
- 诊断修改：`agent.py` 在 `BaseAgent.__init__` 前后、`load_model`
  进入/候选清单/完成及首次 lifecycle 回调打印一次性
  `LifecycleProbe`；`train_workflow.py` 区分 workflow 入口、nav 模块 import 和
  实际分发；`nav_dagger_workflow.py` 继续标记 wrapper、配置、
  `env.reset`、首帧 `env.step`、首次 TBPTT 更新和平台 callback；
  `algorithm_nav_dagger.py` 将首帧再分成 per-env 状态、command 注入、
  冻结 VisionEncoder/Actor、高层 CNN/LSTM、Oracle 和 tick buffer 边界。
  日志均为一次性；checkpoint 清单为 best-effort，不会因平台并发
  替换文件而阻断真正加载。
  针对 `234666` 的新边界，构造器内新增 VisionEncoder/Actor/HighLevel
  逐个 `.to(device)`、架构校验、冻结、Adam、optimizer 参数断言和
  Oracle 的 before/after 探针，同时记录 aisrv/learner 角色、耗时和 CUDA
  allocated/reserved。这些边界用于区分平台 aisrv 特有的 CUDA 上下文、
  Adam 延迟导入或后续断言停点；架构不匹配应立即抛错，
  `NavOracle()` 无重操作，概率最低。
- 开发容器实测：复用 `IDE_SYNC_TOKEN` 的 `exec_b64_v1` 已经重启并实际
  执行。容器为 RTX 4090 D 5GB / PyTorch `2.7.0+cu128`。单进程构造中
  VisionEncoder 首次 `.to(cuda)` 约 `0.025s`，三模块重复 `.to(cuda)` 均约
  `0.001s`，唯一显著步骤是首次 Adam 构造约 `2.917s`；两个并发
  独立进程均在约 `0.76s` 完成。因此已排除模型大小、普通 CUDA
  搬运、显存不足和普通并发；当前最可疑是平台 aisrv 特有的进程
  初始化上下文与首次 optimizer 延迟导入之间的交互，待新探针 smoke 定案。
- 真实父包联调修正：使用用户下载的 `commandfull-34728`（checkpoint SHA256
  `0ce3b485053faa5da37c2b0ad792ad414ec6f5a482208c728cee65487d019d0f`）运行完整
  `train_test.py` 后，确认 Adam/CUDA 不是根因。开悟 local-wrapper 的真实顺序是先以
  `id=0` 调用 `save_model()`，再执行 `preload_model_file()`；nav 在父包尚未加载时对
  首存执行严格 digest 校验，导致 aisrv helper 线程提前退出。修复为只跳过这个
  `id=0 + low_level_state_digest 未初始化` 的无效 bootstrap save，不写随机 checkpoint；
  父包加载后的正常保存仍保持原行为。修复后启动继续进入 Track+Camera env reset。
- 第二个实测根因：平台当前 `nav_scanner` 是各向异性网格
  `size=(2.5,2.0), resolution_x=0.2, resolution_y=0.1, ordering="xy"`，真实输出
  `(N,273,3)`，flatten 布局为 `21(y) x 13(x)`。旧 helper 只接受 143 或平方网格，
  ObservationManager 构造时硬失败。现按真实 `pattern_cfg` 推导并校验矩形 shape，
  同时保留 143/256/平方兼容；17 项本地与开发容器测试通过。修复后的官方
  `train_test.py` 在开发容器 47.11 秒完成并明确打印 `Train test succeeded`，已覆盖
  Agent 构造、bootstrap save 跳过、Track+Camera 仿真创建、273-ray scanner 解析和
  deploy config 生成。该入口会设置 `KAIWU_TRAIN_TEST=1`、强制 `num_envs=1`，且日志
  证明它不调用真实 preload/nav workflow，因此该结果不能替代正式任务对 34728 加载、
  256 环境 reset、首帧 rollout 或首个 update 的验证。
- 后续 checkpoint 校验决策（用户明确，尚未在本轮扩大修改）：身份、父模型 ID、
  lineage 和 digest 不应默认成为训练硬门禁，应打印醒目 warning 并继续使用操作者
  选择的结构兼容包。硬失败只保留无法反序列化、缺少执行所需模块、state dict
  key/shape 无法装载或张量非有限等真正不可运行的结构问题。启动链跑通后应单独审计
  nav/visual/eval loader 中现有身份与 digest 门禁，避免与本轮 lifecycle/env 修复耦合。
- 本地验证：完整 nav + RPC/同步测试集曾通过（`95 passed,
  3 subtests passed`）；本次又按当前文件清单重跑全部九个 nav 测试文件，结果为
  `73 passed, 3 subtests passed`。相关运行文件和测试 `py_compile`、`git diff --check`
  均通过；探针/RPC 服务文件已同步且远端 hash 回读一致。
- 开发容器完整链验证：在隔离启动器中关闭 `is_train_test` 和 checkpoint 早退，临时
  将 nav TOML 从 256 环境降为 8 环境，真实启动 modelpool、learner、aisrv、Isaac
  worker、34728 preload 与 `nav_dagger_workflow`。最终临时包记录
  `current_iteration=14`、`total_env_steps=18808`、`total_nav_ticks=1888`，Adam state 有
  10 个参数条目；第 10 轮日志为 `ce=1.6981, top1=0.133, grad=0.700,
  nonfinite=0`。包内 `source_parent_model_id=low_level_parent_model_id=34728`、父 SHA256
  为 `0ce3b485...d019d0f`，低层 digest `512ca94f...168aae` 与真实父包现场计算完全
  一致，证明 preload、rollout、TBPTT 更新、optimizer 和 checkpoint lifecycle 均已
  走通。停机后远端 TOML 已恢复 `num_envs=256` 且哈希与启动前一致，进程组无残留。
- 可复用诊断收尾：临时启动器已固化为
  `server/agent_ppo/tools/nav_full_smoke.py`，通过 `NAV_FULL_SMOKE_NUM_ENVS` 只覆盖
  内存配置，不再备份/改写/恢复生产 TOML。`start/status/stop` 使用独立进程组，默认
  只在首个 `valid_ticks>0` 的 TBPTT update 后停止；全无效段只写
  `first_update_skipped`，不能冒充成功。
- 多进程日志误判：aisrv、learner、worker 共享 stdout 时行片段可能交错，曾使普通
  grep 看起来像停在 Algorithm 构造，实际 Python 栈已经进入 `env.step()`。新增
  `NAV_SMOKE_EVENT_LOG` 的单行 JSONL 通道；每条事件由一次 `O_APPEND` 写入，记录
  `agent_ready -> preload -> workflow/reset -> first_frame/nav_tick -> first_update ->
  iteration/checkpoint/fatal`。平台原日志保留供人工阅读，自动 smoke 判定只消费 JSONL。
- RPC 使用边界：长时间训练不得占用一条 RPC HTTP 长连接；代理/浏览器链路会先于
  训练进程断开，断开不等于训练失败。RPC 只执行短命令启动、`status` 或 `stop`；
  后台进程组、PID、事件和 stdout 均由容器本地 runtime 目录持有。
- goal 初始化时序：构造期 `nav_probe` 看到 `goal_positions=ABSENT` 只说明 reset 尚未
  建立目标，不能推出训练标签始终无效。开发容器有效 CE/梯度已证明 reset 后目标存在；
  现在首个真实 nav tick 明确打印 `oracle_valid_count/ratio`、`goal3_abs_mean/max` 和
  `goal4_fresh_count/ratio`，iteration 以实际 count 聚合 `goal_valid_rate`。零有效目标
  继续跳过 optimizer update，不伪造标签。
- 下次 smoke 最短判断表：缺 `agent_init complete` 则卡在 BaseAgent；有构造完成但
  无 `load_model enter` 则平台没有调用自定义 loader；有 load 完成但无
  `train_workflow enter` 则是平台 workflow/model-pool 调度层；有 workflow 但无
  `nav_env_reset begin` 则是配置/分发层；有 reset begin 无 `reset ok` 才进入
  环境 worker/平台 base env 调查。如果已进入首帧，则按
  `per_env_state_ready -> command_injected -> vision_encoder -> low_level ->
  cnn -> high_level -> oracle -> buffer_add -> env_step -> first_update ->
  lifecycle_callback` 的最后成功标记定位，不再笼统归因到
  Camera 或 `base_env.py`。
- 血缘：分支 `codex/hier-nav-dagger`；真实父包 `commandfull-34728` SHA256
  `0ce3b485053faa5da37c2b0ad792ad414ec6f5a482208c728cee65487d019d0f`；完整 smoke
  使用临时 checkpoint 验证后已清理，未登记为平台发布模型。
- 回滚：删除 `LifecycleProbe` 不影响任何权重或训练状态；在根因定案前
  不得改回或同步平台拥有的 `isaac_env/base_env.py`。

## BUG-20260727-003：Nav 首存晚于平台退出窗口且监控面板误导为 Standard

- 状态：本地已验证，平台 30 分钟 smoke 待执行。
- 影响：hier-nav 长训存活、checkpoint 可恢复性和 Track 训练判读。
- 首次发现：`navdagger-r1`，task ID `234680`，2026-07-27。
- 症状：Nav DAgger 已运行到 iteration 50，CE/梯度均正常，但首次常规保存计划在
  iteration 90（实测约 20 分钟）；平台约 15 分钟发出外部 SIGTERM，只有退出收尾包
  `navbc-34780`。前端显示 Standard/PPO 风格面板，容易误判实际地形；后端日志实际为
  `EnvMonitor [task=track]`，并在首个 120 秒窗口报告 `timeout=256`。
- 根因：保存节奏使用 outer iteration 整数估算墙钟，环境数和 Camera 吞吐变化后首存漂移到
  平台退出窗口之后。`agent_ppo/conf/monitor_builder.py` 又是通用 PPO/Standard 风格，只展示
  value/policy/entropy loss 和线速度奖励，没有显式声明 Track 分难度结果或 Nav DAgger 指标。
  平台发出 SIGTERM 的内部看门狗条件不可见，因此“不发布 checkpoint 导致退出”仍是待平台
  A/B 验证的强假设，不冒充已证明根因。
- 排除项：环境实际不是 Standard；日志、配置和 `EnvMonitor [task=track]` 已证明 Track 生效。
  `base_env.py` 会在任务启动前由平台覆盖，不能用本地版本差异解释历史 LBC 与本轮 Nav 的差异。
  Nav SIGTERM handler 只响应外部信号并收尾保存，不会主动发送 SIGTERM。
- 修复：生产配置固定 `num_envs=128`。Nav workflow 在本会话完成 10 个 outer iteration 后首存，
  此后每 5 分钟按 `time.monotonic()` 保存，并保留每 30 iteration 兜底；数字 ID 继续由平台
  注入。monitor builder 显式注册 Track level 0-9 的完成/失败/超时、四项分数，以及 CE、
  top1、分歧率、目标有效/新鲜率、进度和终止率，不再以 Standard PPO 面板作为主视图。
- 平台复核：任务 `234739` 的页面已经出现 `Track 赛道结果(7)` 与 `Nav DAgger(12)`，证明
  `agent_ppo/conf/monitor_builder.py` 已被加载；同时仍保留平台预置的 Standard 地形组。
  后端继续打印 `EnvMonitor [task=track]`，因此不是环境模式回退。对比历史 Track LBC 后确认，
  当前 `Config.CURRENT` 的字面 fallback 仍是 `StandardVisualPPOConfig`，而 Track 选择只在后续
  bootstrap 执行；平台默认面板注册早于该动态切换。专用分支现把字面默认改为
  `NavDaggerConfig`，运行时 bootstrap 和 eval override 保持不变。若新任务仍合并 Standard
  默认组，即可定案为平台外层模板不可由代码包删除，而不是继续修改环境或自定义 builder。
- 验证：配置和纯保存触发测试覆盖 128 环境、首存、墙钟和 iteration 兜底；monitor 源契约测试
  覆盖 Track/Nav 指标。本地测试结果记录在本次交付；平台仍需运行超过 6000 frame 且至少
  30 分钟的 smoke，确认 10 iteration 首存、5 分钟周期保存、Track 面板和无 15 分钟 SIGTERM。
- 防复发：长训保存以墙钟为主、iteration 为兜底；日志必须打印保存 reason/iteration；新的
  Track stage 必须在 monitor builder 显式登记 Track 结果指标。不得用训练 loss 下降替代
  Oracle-only 完成率门禁，本任务的 `0 completed / 256 timeout` 仍需独立修复 Oracle。
- 血缘：分支 `codex/hier-nav-dagger`；低层父 `command-34728`；平台收尾包 `navbc-34780`。
- 回滚：恢复 Nav TOML 和 workflow 的旧保存参数即可；不得通过修改平台 `base_env.py` 回滚。
- 再遇检查：实际 task_type → 首个 checkpoint 时间 → 周期保存 reason → 6000 frame 后存活 →
  Track completed/timeout 分档 → 外部 SIGTERM 来源。

## BUG-20260727-004：Nav lifecycle 按 outer iteration 推进导致平台模型长期不可见

- 状态：本地已验证，平台待验证。
- 影响：hier-nav 平台训练步、模型 ID 推进、模型池自动发布与长训可恢复性。
- 首次发现：Track Nav 任务从父模型 `command-34728` 运行约 20 分钟，workflow 多次打印
  自定义 checkpoint 已保存，但平台模型列表长期没有新模型；相关失败模型 ID 只推进至
  `34738`、`34758`、`34783`、`34788`。
- 症状：Nav 每个 outer iteration 实际执行 160 个低层批量帧，但平台 ID 每轮只增加 1。
  `dump_model_freq=1000` 因此需要 1000 个 outer iteration 才触发平台自动发布。业务侧
  每 5 分钟调用 `save_model()` 可以写出或打包文件，但不能证明模型池已经登记，造成日志
  显示“保存成功”而前端始终无模型。
- 根因：`nav_dagger_workflow.py` 把仅用于平台 lifecycle 的 `agent.learn(None)` 放在
  160 帧 outer loop 之后，每轮只调用一次。平台框架按 `BaseAgent.learn()` 回调计数推进
  train step 和 `dump_model_freq`，而 Nav 的真实梯度更新由
  `finish_nav_sequence_update()` 独立完成。可成功训练和保存的 Track LBC 归档提供了直接
  证据：父 ID `42399`、最终 ID `282409`，差值 `240010`；配置为 10000 iteration、
  每轮 24 帧，理论回调数 `10000 × 24 = 240000`，两者基本一致。失败 Nav 从 `34728`
  到 `34788` 的约 60 步也与运行约 60 个 outer iteration 对齐。
- 排除项：checkpoint 的 `navbc/navdagger/navfull` 标签符合平台纯字母探活规则；成功 Track
  LBC 与失败 Nav 都使用 Camera、128 环境、120 秒 episode，并都会出现
  `succ_cnt=0` 和 `reached max length`，因此这些不是本次发布停滞的根因。平台拥有的
  `isaac_env/base_env.py` 未参与本修复。BUG-20260727-003 中“不发布导致外部 SIGTERM”
  仍不是已证明的平台看门狗因果关系，本条只修复已经由计数证据确认的发布生命周期。
- 修复：每个成功 `env.step()` 在观测、终止处理和可能的 TBPTT 更新完成后调用一次
  `agent.learn(None)`；删除 outer iteration 末尾的额外 callback。失败的 `env.step()`
  不推进 lifecycle。单次 lifecycle callback 异常按操作者决策记录后继续，成功和失败
  分开计数，失败不计入发布进度。
- 2026-07-27 更正：平台任务已能跨过 15 分钟后，用户进一步要求把常规自动保存控制在
  约 5 分钟，并减少身份门禁。生产 `dump_model_freq` 从 `1000` 调整为 `3600`，因为当前
  128 环境 Camera Track 实测约 `160` 个 lifecycle callback / `13.46s`，`3600` 次约
  `5.0` 分钟。首个周期若落在 4-6 分钟外只记录偏差，不在运行中动态改频率。
- 保存策略更正：BUG-20260727-003 引入的 10 iteration 首存、5 分钟墙钟保存和 30
  iteration 兜底从当前 Nav workflow/TOML 删除。平台自动发布成为唯一常规发布路径；
  正常结束、平台正常停止与 SIGTERM 共用一次幂等 final best-effort 保存，仅用于退出
  恢复。保存日志必须区分 `/data/ckpt` 运行中 checkpoint 与 `/data/user_ckpt_dir` 最终
  平台归档候选，并打印平台 ID、文件大小、SHA256、距离上次保存的墙钟时间和 lifecycle
  数。不得用运行中 `/data/ckpt` 文件存在替代平台模型列表登记证据；单机任务运行中前端
  不实时展示模型时，最终以正常停止后的 `/data/user_ckpt_dir` 和模型列表为准。
- 校验策略更正：`platform_model_id`、lineage、父模型 ID、low-level digest 和历史身份
  元数据不一致为 warning-only；结构兼容的操作者指定包可以继续训练/续训/保存。硬停止
  仅保留给文件写入/反序列化失败、必需模块缺失、state-dict key/shape 不兼容、非有限
  权重或无法生成有效非空 checkpoint。生产保存后只做文件存在、非零大小与 SHA256 检查；
  完整 round-trip 放到测试和 resume smoke。
- 修改范围：`agent_ppo/workflow/nav_dagger_workflow.py`、Nav stage/TOML、
  `agent_ppo/agent.py` 注释、`conf/configure_app.toml` 说明、完整链路 smoke 和 Nav 测试。
  网络结构、TBPTT、checkpoint schema、phase 标签和评估 loader 不变。
- 本地验证：修改文件通过 `py_compile`；`pytest -q agent_ppo/tests/test_nav_*.py`
  通过 102 项测试及 3 个子测试，覆盖逐帧 callback、失败环境帧、callback 异常继续、
  TBPTT 次数、3600 回调边界、最终保存、stage/metrics、smoke 和 checkpoint；
  `git diff --check` 通过。容器同步、平台 20-30 分钟 smoke、最终模型列表、resume、
  评估与真机验证均未执行。
- 防复发：测试必须证明 N 个成功低层批量帧恰产生 N 次 lifecycle callback、outer loop 无
  额外调用、失败环境帧不推进、TBPTT 更新次数不变，并覆盖 999/1000/1001 callback
  历史边界及 3599/3600/3601/7200 当前边界。日志持续上报 callback 成功/失败数、累计
  步数、累计环境帧、下次 dump 距离和保存文件事实。
- 血缘：分支 `codex/hier-nav-dagger`；低层父 `command-34728`；对照归档
  `track蒸馏_282409`，父模型 `42399`、最终模型 `282409`。关联 commit/PR：尚未提交；
  当前 checkpoint SHA256：本修复不新增 checkpoint，不适用。
- 遗留风险：平台自动发布仍需真实任务确认；回调成功只能证明用户代码返回，不能单独证明
  模型池同步完成。若 callback 持续失败，训练会继续但平台发布进度停滞，必须以 failure
  telemetry 和模型列表共同判断。
- 回滚：恢复 outer iteration 单次 callback 与旧周期保存配置；不得修改
  `isaac_env/base_env.py`。再次遇到时最短检查路径：父/当前模型 ID 差值 → lifecycle
  callback 计数 → `dump_model_freq` 边界日志 → 平台模型列表 → final save 是否仅在退出出现。

## BUG-20260727-005：Nav lifecycle 容错吞掉 checkpoint 写入失败

- 状态：本地已验证，平台待验证。
- 影响：Nav 五分钟自动发布、长训可恢复性和模型列表可见性。
- 症状：`agent.learn(None)` 的平台 lifecycle 回调被 `except Exception` 统一记录后
  继续。如果 `dump_model_freq` 边界内的 `save_model()` 因磁盘、序列化、路径或
  非空校验失败，也会被当成普通回调失败吞掉，训练可能在没有任何新 checkpoint
  的情况下继续。
- 根因：业务容错和制品持久化共用了无类型的异常通道；Nav `save_model()` 没有把
  checkpoint 写入/验证错误标记为不可恢复。
- 排除项：`navbc/navdagger/navfull` 文件名和 `dump_model_freq=3600` 不是此问题；
  普通 lifecycle 抖动仍允许记录后继续，SIGTERM/final save 仍是 best-effort。
- 修复：在 `checkpoint_io.py` 新增 `CheckpointSaveError`；`Agent.save_model()` 将 Nav 的
  标签/路径校验、bundle 写入和文件大小验证错误转换为该类型。Nav workflow
  遇到该异常时记录 `checkpoint_save_failure` 并立即重抛；其他回调异常保持原容错。
- 验证：`test_nav_lifecycle_publication.py` 新增 checkpoint 失败在第 2 次回调立即
  终止的回归测试，并保留普通 `RuntimeError` 继续训练的反例。相关文件
  `py_compile` 通过；Nav、checkpoint 与同步相关回归集累计
  `184 passed, 1 deselected, 27 subtests passed`。被单独选出的实际 HTTP
  bundle 往返测试在允许本地回环端口后复测为 `1 passed`。
- 防复发：lifecycle 测试必须同时覆盖“普通失败继续”和“保存失败停止”；
  任何新的 checkpoint writer 都必须使用专用异常而非裸 `RuntimeError`。
- 血缘：分支 `codex/hier-nav-dagger`；父模型 `command-34728`；关联 commit/PR 尚未生成；
  本修复不生成 checkpoint，模型 ID/SHA256 不适用。
- 遗留风险：平台 wrapper 是否原样传播用户异常仍需真实任务验证。回滚时可移除
  专用异常分支，但不得恢复吞掉 checkpoint 写入失败的行为。再遇最短路径：
  lifecycle failure 原始类型 → 是否进入 `Agent.save_model()` → 写盘路径/剩余空间 → 是否终止。

## BUG-20260727-006：IDE 同步路径可经符号链接越界或绕过保护文件

- 状态：本地已验证。
- 影响：开发容器同步/RPC 的路径边界，以及平台拥有的 `isaac_env/base_env.py`。
- 症状：`Workspace.resolve()` 只用文本 `abspath` 折叠 `..`，工作区内指向外部的
  symlink 仍可通过 `relative_to()`；指向 `isaac_env` 的别名也会让保护检查看到
  `alias/base_env.py` 而不是真实的 `isaac_env/base_env.py`。
- 根因：路径边界和保护列表都在 lexical path 上判定，没有在现有父目录符号链接
  解析后的 canonical path 上判定。
- 修复：`Workspace` 初始化时解析项目根，每个请求目标用
  `Path.resolve(strict=False)` 解析现有父目录，然后再检查是否位于根目录内；保护文件
  比对因此也使用真实相对路径。根 `.gitignore` 同时加入 `/.worktrees/` 和
  `/.zcode/`，避免 65 MB 本地工作树和工具状态被误提交。
- 验证：新增外部 symlink 被拒绝、`isaac_env` symlink 别名仍拒绝
  `base_env.py` 写入的测试；原有普通 `isaac_env/overlay.py` 写入/删除测试保留。
  相关 `py_compile` 通过；Nav、checkpoint 与同步相关回归集累计
  `184 passed, 1 deselected, 27 subtests passed`；实际 HTTP bundle 往返测试
  单独复测 `1 passed`。
- 防复发：任何新的 read/write/delete/bundle/exec 路由必须统一走
  `Workspace.resolve()`；保护路径测试必须包含 `..`、URL 编码、路径分隔符和 symlink 别名。
- 血缘：分支 `codex/hier-nav-dagger`；不涉及模型/checkpoint；关联 commit/PR 尚未生成，
  SHA256 不适用。
- 遗留风险：开发容器若依赖项目根内指向根外的特殊挂载 symlink，新边界会拒绝；
  应将这类挂载设为 `--root` 本身，不应放宽任意子路径。回滚只能回到另一个真实路径
  实现，不得回到 lexical-only 检查。再遇最短路径：原始 path → canonical path →
  根目录边界 → 保护相对路径。

## 3. 已知高频误判

以下现象可能伴随真实 Bug，但不能单独作为根因：

- `learner_proxy sample succ_cnt=0`：自定义蒸馏直接更新时可能正常；应看专用 iteration、loss、
  global step 和 checkpoint。
- `monitor_proxy ... Broken pipe`：通常是 aisrv/learner 已因更早 traceback 退出后的二次错误；
  应先找时间最早的异常。
- `Episode done: reached max length`：平台原始环境可能仍打印；要结合 outer iteration 是否继续、
  workflow 是否 break、hard termination 和 per-env reset 判断。
- checkpoint 文件出现在 `/data/ckpt/...`：只证明内部写盘，不保证平台模型列表已登记，也不证明
  评估 loader 会选中它。
- 同步显示 `bundle verified`：证明容器磁盘文件一致，不证明运行中的 Python 已重新 import。
- 总分接近教师：不能替代分地形指标、固定 seed 和视频；无 checkpoint 加载证据的分数一律无效。
- warning-only 身份提示：候选文件仍按平台请求 ID 精确选择；包内 ID/lineage/digest
  属追溯元数据，缺失或不一致只告警。文件不存在、反序列化失败、必需模块缺失、
  state-dict key/shape 或网络/输入契约不兼容、非有限张量仍必须 hard stop；未成功
  加载模块时评估不得继续评分。

## 4. 固定排障顺序

遇到训练/评估回归时按以下顺序检查，避免先调奖励或网络：

1. **任务输入**：实际任务名、训练/评估 TOML、`policy_entry`、预加载模型 ID。
2. **文件一致性**：本地 hash、dry-run overwrite 清单、同步完成后第二次 dry-run、平台保护文件。
3. **进程版本**：同步后是否真正重启 aisrv/learner，启动日志中的 config path、schedule mode、
   source parent 和代码特征是否为新版本。
4. **checkpoint 身份**：requested ID、所有 candidates、selected path、SHA256、format、bundle ID、
   lineage、实际 loaded modules。
5. **环境契约**：num_envs、terrain/curriculum、command ranges/limits/resampling、camera、noise/randomization。
6. **训练 lifecycle**：outer iteration、inner steps、global step、save callback、平台模型 ID。
7. **运行数据**：requested/effective command、bucket counts、action/KL、hard termination、timeout。
8. **能力验收**：同配置固定 seed、分地形分项、视频；最后才讨论奖励和网络修改。

## 5. 资料来源

- [`server/CHANGELOG.md`](../../server/CHANGELOG.md)
- [`2026-07-23_Standard桥接蒸馏昨夜至今迭代复盘.md`](./2026-07-23_Standard桥接蒸馏昨夜至今迭代复盘.md)
- [`2026-07-24_main归档与Standard视觉主线合并记录.md`](./2026-07-24_main归档与Standard视觉主线合并记录.md)
- [`2026-07-25_StandardAnchorR2四小时实施计划.md`](./2026-07-25_StandardAnchorR2四小时实施计划.md)
- [`2026-07-25_Standard命令泛化下半四小时实施计划.md`](./2026-07-25_Standard命令泛化下半四小时实施计划.md)
- [`server/tests/test_local_sync_client.py`](../../server/tests/test_local_sync_client.py)
- [`server/tests/test_visual_policy_optimization.py`](../../server/tests/test_visual_policy_optimization.py)
