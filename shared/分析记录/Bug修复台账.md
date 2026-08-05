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
- 2026-07-27 更正与收口：分支工作树中仍残留过一份含 command bucket、
  continuous-training 和深度配置的已提交定制版本。现已将活动
  `server/isaac_env/base_env.py` 恢复为归档原始文件，逐字节 SHA256 为
  `75ebdaf6888e94262598a26db1586b2598cb474422e3382b6bba9e96ddbb6e67`；功能实现继续留在
  `agent_ppo`。状态为本地已验证，平台文件保护的既有平台验证结论不变。

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
- 2026-07-27 平台证据再更正：任务 `234786` 运行中连续写出
  `navbc-36000/39600/...`，任务结束后才生成可下载的 `navbc-52313` ZIP。最终包包含规范
  `ckpt/id_list`、`ckpt/kaiwu.json` 与同一三模块 checkpoint，证明 `dump_model_freq`
  的 `/data/ckpt` 写盘和前端用户归档是两条生命周期。当前方案因此恢复无参数、无自定义
  ID/path 的五分钟 `agent.save_model()` 平台归档请求；`dump_model_freq` 只保留为运行恢复。
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

- 状态：平台已验证（开发容器同步链）。
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
- 2026-07-27 更正：实际开悟容器不是单一真实根。同步根为
  `/data/projects/legged_robot_competition_26`，但 `agent_diy`、`agent_ppo`、`conf`
  是指向 `/workspace/code/<同名目录>` 的平台固定符号链接，`isaac_env` 则仍在原根内。
  因此“把特殊挂载设为唯一 `--root`”不适用于该平台；上述 canonical-only
  修复会在 manifest 阶段误报 `path escapes root: agent_diy`。现实现保留根内逻辑路径，
  另行核验 canonical target：只允许三个固定顶层目录映射到
  `/workspace/code/<同名目录>`，映射目录内的任意二次越界仍拒绝，canonical alias
  命中 `isaac_env/base_env.py` 时仍禁止写入。新增 mapped read/write/manifest 正例和
  mapped 内部越界负例。同步回归 `27 passed`，相关 Python 编译和
  `git diff --check` 通过。容器内新服务 SHA256 为
  `f0ee115b4b526e011b41c7f5707ee1902007877e91568908e4a01d83747796ea`，按原根目录重启后真实
  `local_sync_client.py --dry-run` 成功读取四个目录，清单为 103 个本地文件、7 个
  待更新文件、0 个删除候选，不再出现 `path escapes root`。该 dry-run 未上传业务文件。

## BUG-20260727-007：Nav 高低层控制链缺少可区分的统计口径

- 状态：本地已验证，平台待验证。
- 影响：hier-nav DAgger 训练监控、高层控制是否真正传到低层的故障定位。
- 首次发现：分支 `codex/hier-nav-control-observability`；平台任务和模型 ID 不适用。
- 症状：原面板只能看到 CE、超时和模糊的 episode 终止数，无法区分
  Oracle 没有生成合理 token、驻留/调度压制了指令、低层未响应，还是 worker
  reward 仍在读取另一套 command。`terminated` 又被误命名为 hard failure，
  它实际也可能是 Track 成功退出。
- 根因：aisrv 中的 `NavScheduler.inject()` 只改写 policy/critic observation，
  不会反向改写 worker `command_manager`；原有统计没有同时保留 worker command、
  exec command、真实速度和 action 响应，也没有 Track scorer 与 aisrv 终止口径边界。
- 排除项：本修复不修改 `isaac_env/base_env.py`，不新增 aisrv→worker IPC，不调整
  Oracle 规则、奖励、DAgger ramp 或网络。
- 修复：`algorithm_nav_dagger.py` 新增四组 token 分布、driver/驻留、
  worker/held/exec command、真实 vx、tracking error、action 幅度/变化/非有限与
  token 切换后一个 nav period 的 action/vx 响应。`nav_oracle.py` 记录现有规则
  分支和 goal/墙体统计，不改 token 输出。workflow 把内部指标改名为
  `ended_episode_count` 与 `non_timeout_termination_*`，面板保留 Track scorer 原始指标。
- 验证：target commit `f7145da` 曾通过 94 项 Nav 测试；合并与误报修正后
  完整 `agent_ppo/tests/test_nav_*.py` 为 `116 passed, 3 subtests passed`，
  修改 Python 文件 `py_compile` 与 `git diff --check` 通过。容器同步、平台
  smoke、评估和真机均未执行。
- 防复发：回归测试覆盖 ramp=0 Oracle driver、四组 token 比例、worker/exec 对齐、
  非有限 action、Oracle 分支、切换响应的稀疏加权聚合、reset 取消样本以及
  终止指标命名。最短检查路径：Oracle token/mode → requested/effective →
  dwell → worker/exec/actual vx → action response → Track scorer。
- 合并审查更正：纯 yaw 或相同 vx 的 token 切换本来就不要求 vx 变化；
  `switch_response_no_velocity_ratio` 现只在目标 vx 确实变化的样本上计算，
  action 响应仍覆盖所有 token 切换，避免误报低层失联。
- 血缘：target commit `f7145da`；合并 commit 待生成；父模型、checkpoint 和
  SHA256 不适用。
- 遗留风险：所有新指标仍需平台 monitor 实际上报验证。身份与 digest
  元数据不一致按当前策略 warning-only；结构不兼容、非有限权重、写盘失败
  仍必须硬停。回滚可撤销 `f7145da` 的诊断指标，不触碰 `base_env.py`。

## BUG-20260728-001：P1.5 预审发现目标 epoch、物理随机化与 NLL 语义漂移

- 状态：本地已验证，开发容器与平台待验证。
- 影响：`p15_response` future-label 有效性、父步态保持、前两小时控制变量和
  ResponseAdapter 不确定性训练。
- 症状：本轮尚未提交的平台前置实现中，smooth target 每 5 Hz 改变时没有同步增加
  `command_epoch`，future-label buffer 会把变化中的目标误判为 horizon 内恒定；
  `[domain_rand] enable_domain_rand=true` 又未显式关闭缺省为 true 的 base-mass
  randomization；摩擦边界只修改 event config，没有证明 startup event 会重新执行；
  Adapter 还把 `velocity_log_sigma_xyz_1s3` 错当成三个 horizon 的标量 sigma。
- 根因：command destination、实际 active target 和 label epoch 三者未分开；TOML 只关闭
  已显式列出的 randomization 项，没有核对平台 `base_env.py` 的缺省值；运行时摩擦切换
  只改配置对象而未调用 event manager；NLL 广播维度虽然 shape 合法，但与 16 维输出的
  固定物理语义不一致。
- 修复：smooth target 每次实际变化都推进 epoch，step/急停只在 target 真变化时推进；
  transition family 固定为 zero/start/brake，并记录 `vx-only/wz-only/both` 均衡事件。
  P1.5 TOML 将总开关保持开启以保留原生 `physics_material` event，但启动范围固定为
  `[1.0,1.0]`，并显式关闭 base mass 与 push；2 小时后由 worker 将摩擦扩为
  `[0.6,1.2]` 并实际调用 event manager，失败只告警并每 500 步重试。
  NLL 只作用于 1.0 秒 XYZ 速度头；三个 horizon 单独报告 MAE，并增加 zero/copy-exec
  baseline。父模型支持域内的 zero、pure-yaw 和 lateral 也恢复 S0 anchor 权重。
- 其他修正：schema 2 补齐低层 scheduler、gradient/skipped 计数恢复；response buffer
  返回逐记录 low-level digest/iteration；reset 与 step 两条 workflow 路径都立即拆分
  wire346，Critic/storage 始终只见 316。monitor builder 按 `policy_entry` 选择 P1.5 或
  原 Nav 面板，避免新阶段覆盖旧入口。
- 验证：P1.5 本地测试当前 `26 passed`；完整 agent 回归（排除两个已知失效旧 API
  测试）为 `146 passed, 3 subtests passed`，VisualPPO 回归为
  `57 passed, 24 subtests passed`。真实父文件在 CPU 完成 schema1 首载与 schema2
  save/resume：父内记录 iteration `6327`，低层 digest 与 Adapter 参数恢复一致，输出包
  13,780,365 bytes。GPU 容器 rollout/update 与平台 smoke 尚未完成，不能提升为平台已验证。
- 防复发：回归测试覆盖课程边界、独立轴采样、smooth epoch、父域 anchor、346 拆分、
  bounded buffer、双训练状态、FeedbackEmulator 可复现性、摩擦 event 调用、保存失败重试
  与 monitor 中文名称。正式任务继续只把结构/写盘不可用作为外部不可恢复故障；训练指标
  不得变成阶段暂停或提前停止门禁。
- 血缘：分支 `codex/p15-response-adapter`；父包
  `commandfull-34728`，SHA256
  `0ce3b485053faa5da37c2b0ad792ad414ec6f5a482208c728cee65487d019d0f`；关联 commit/PR、
  平台任务、输出模型 ID 与 SHA256 尚未生成。
- 遗留风险：平台 EventManager 的公开方法签名和运行中重新应用 physics material 仍需
  GPU 容器 smoke；当前开发容器旧 IDE `18005` 已返回 `WEBIDE_RECORD_NOT_FOUND`，需用户
  重新打开 IDE 后才能同步。跨任务 resume 的 worker clock 依赖 TOML offset，不新增 IPC。回滚时
  删除 P1.5 新入口与 TOML即可，不修改平台拥有的 `isaac_env/base_env.py`。
- 2026-07-28 更正：课程扩域必须覆盖主联合域中的连续低速端，expanded joint/straight
  不再保留人为 `vx>=0.05`、`|wz|>=0.03` 的孔洞；急停明确绕过 slew。跨 `num_envs`
  resume 仅丢弃 shape 不兼容的 bounded warm history，不影响模型、optimizer 或计数恢复。
  iteration-cap 最终保存失败会等待 60 秒重试一次，第二次失败只保留明确错误，仍不把
  业务代码变成提前终止门禁。schema2 现在写入实际 TOML feedback profile 与
  `FeedbackEmulator` 源文件 SHA256，不再只记录静态默认 profile。

## BUG-20260728-002：P1.5 Response 序列跨 episode/低层版本污染及反馈合同漂移

- 状态：本地与开发容器已验证，平台短 smoke 待验证。
- 影响：`p15_response` 的 0.2/0.6/1.0 秒标签、GRU hidden、跨任务 resume、Standard
  terrain 难度分布、短时速度反馈、schema2 恢复可信度和八小时任务最终保存。
- 首次发现：分支 `codex/p15-response-adapter` 的未提交预审实现；父模型
  `commandfull-34728`，生产配置 Standard + Camera、256 环境、rollout 48 帧、最长 future
  horizon 50 帧。平台任务、输出模型 ID 与 checkpoint SHA256 尚未生成。
- 症状：48 帧 rollout 必然让 50 帧标签跨过一次低层 PPO 更新；buffer 只按 done/command
  epoch mask，没有 low-level version 边界。随机 16 帧窗口从零 hidden 开始且没有 episode
  reset/burn-in。schema2 resume 还恢复未完成 history，而新任务首次 env reset 会与旧任务尾部
  拼接。`FeedbackEmulator` 可把低频 UWB 当作即时 `[vx,vy,wz]` 输入；terrain 又重新开启了
  BUG-20260725-006 已证伪的距离 curriculum。缺 scheduler/RNG 时 loader 仍沿用完整 resume
  名称。逐腿 swing/air 指标和 command family 统计也存在分母语义混淆。
- 根因：future-label 生命周期、低层 optimizer 生命周期和 episode 生命周期没有建立同一份
  版本/reset 合同；反馈 profile 把长时定位证据与短时速度反馈混在一个 source selector；
  P1.5 配置没有复用命令泛化阶段已验证的静态 difficulty 边界；恢复代码只检查模块/optimizer，
  没有报告 scheduler、RNG 和 buffer 的实际恢复级别。
- 排除项：加速 command 课程 `0.5/1.25/2/7h` 是用户最终批准方案，不回退；不修改既有 gait
  reward 权重、不新增 3-8 秒 command segment、不修改 `isaac_env/base_env.py`。UWB 过去没有
  直接充当 future true-velocity label，但它会进入当前 response observation/tracking error，
  因此仍需从短时输入移除。
- 修复：生产 rollout 改为 80 帧，buffer 在 low-level digest/iteration 变化时清空未完成
  history，采样窗口只允许单一版本；GRU 增加 8 帧 burn-in 与逐环境 reset mask。checkpoint
  只恢复已经形成标签的 records，未完成 history 永久采用
  `completed_records_only_v2`。短时 `measured_velocity3` 固定为 SportMode `vx/vy` + IMU
  gyro `wz`，source 仅 0/1；UWB 仅保留为独立长时卡滞证据。capability15 改为显式
  `piecewise_union_profile_v1`。terrain `curriculum=false`，新增
  terrain family×level×command family 直方图。schema2 完整恢复标记为
  `weights_optim_rng_resume_with_history_reset`，缺 scheduler/RNG 等状态 warning-only 降级
  `warm_start`；Standard 读取 response 包必须显式 `low_level_only_preload=true`。
- 运行时修正：重采样 family 与按帧 active family 分开统计；air/swing 指标使用各腿有效样本
  分母。摩擦切换增加 event config readback、attempt/failure telemetry，仍在失败后每 500 步
  重试。P1.5 graceful final save 失败短重试一次；SIGTERM wrapper 链接已有平台 handler 后
  进入幂等 final save，继续复用 BUG-20260727-009 已平台验证的无参数
  `agent.save_model()` → `/data/user_ckpt_dir` → ZIP → 网页模型列表链。开发容器首启还发现
  monitor 标题 `P1.5动态响应训练` 的句点不在平台白名单，已改为 `P15动态响应训练` 并加入
  源码回归断言；该错误只会跳过自定义面板，不会停止训练。
- 修改文件：`response_aux_buffer.py`、`response_adapter.py`、`algorithm_p15_response.py`、
  `algorithm_visual_ppo.py`、`feedback_emulator.py`、`p15_contract.py`、
  `p15_command_schedule.py`、`p15_worker_bridge.py`、`visual_ppo_workflow.py`、`agent.py`、
  P1.5 TOML、checkpoint candidate、接口/实施记录和专项测试。
- 当前验证：修改 Python 文件 `py_compile`、TOML 解析和 `git diff --check` 已通过。
  开发容器 P1.5 专项测试 `38 passed`；Nav + P1.5 + hard-start 支持回归集
  `164 passed`。自定义 monitor 已在容器日志确认“User custom monitor config loaded
  successfully”。首次真实 GPU smoke 还证明 Standard + Camera、wire346 拆分、
  34728 preload、低层 PPO update 与 schema2 保存链可运行，但暴露 Adapter 空转。
  80 帧 StageConfig 修正后的二次 GPU smoke 和 schema2 完整进程 resume 已完成，
  详细证据见本条后续更正。
  平台短任务、八小时长训、评估和真机均未执行，不能提升为平台已验证。
- 防复发：测试必须覆盖 episode reset 后 hidden 等价于零状态重放、版本切换不混合窗口、
  resume 不恢复 unfinished history、80 帧 rollout 能在单一版本内形成 50 帧标签、UWB
  profile 变化不影响短时输入、curriculum 永久关闭、schema2 warm-start 分类、显式
  low-level-only、逐腿有效分母、摩擦 readback、SIGTERM handler 链和 final save 重试。
- 血缘：分支 `codex/p15-response-adapter`；父包 SHA256
  `0ce3b485053faa5da37c2b0ad792ad414ec6f5a482208c728cee65487d019d0f`；关联 commit/PR、
  平台任务、输出模型 ID/路径/大小/SHA256 均待生成。
- 遗留风险：80 帧 rollout 的 GPU 显存和实际吞吐、Isaac EventManager 运行时摩擦应用、
  terrain metadata API、平台已有 SIGTERM handler 行为都必须由 8-16 环境开发容器 smoke
  验证。回滚时可以恢复旧 Adapter replay，但不得恢复 48 帧跨版本标签、未完成 history
  resume、UWB 短时输入或距离 curriculum。再次遇到的最短检查路径：rollout/horizon →
  low-level version → episode reset mask → buffer resume mode → feedback source → terrain level
  histogram → save path category。
- 2026-07-28 更正：开发容器 8 环境真实 GPU smoke 表明之前“rollout 已改为
  80 帧”的结论只在 TOML 层成立。`Agent` 初始化 storage 时只读
  `P15ResponseConfig.num_steps_per_env`，运行时实际仍为继承值 48。证据 checkpoint：
  `low_gradient_steps=460`、`adapter_iteration=23`，但 `adapter_gradient_steps=0`、
  `buffer.record_steps=0`、`global.env_steps=8840`；日志中 `adapter=0.00000`、
  `adapter_mae=0.00000`。根因是 48 帧永远达不到 51 帧 future-label 门槛，而
  低层每轮更新后 digest/iteration 改变会正确清空 unfinished history。修复为
  `P15ResponseConfig.num_steps_per_env=80`，并新增 buffer append、history 峰值、version
  reset、record 与 valid-horizon 遥测。
- 2026-07-28 容器复验：修正 StageConfig 后，8 环境真实 GPU 链在
  `iter=6330` 已出现 `records=91`、`adapter_gradient_steps=3`、
  `adapter_loss=0.34020`、`adapter_mae=0.40075`；继续到 `iter=6340` 时为
  `records=391`、`adapter_gradient_steps=13`、loss `0.28751`、MAE `0.35434`。首个
  周期包 `/data/user_ckpt_dir/legged_robot_competition_26_ppo/`
  `model.ckpt-responsebase-34743.pkl`，大小 `14184143` bytes，SHA256
  `346366fbd548a5bc447a7f8ddb6a6efa46336b31e4293f37e3a85c7c947f70ea`。解析结果为
  schema 2 / `p15_response_adapter`，含全部低层、Adapter-only、S0 anchor 模块与
  `visual_ppo`/`response_adapter` 双 optimizer；`low_gradient_steps=300`、
  `adapter_gradient_steps=15`、`record_steps=451`。将该包作为 preload 后的第二次
  完整进程 resume 在 `iter=6350` 继续到 `adapter_gradient_steps=23`、loss
  `0.04697`、MAE `0.24892`，证明低层、Adapter、optimizer 计数和 bounded completed
  records 均实际恢复。容器专项测试 `38 passed`，Nav + P1.5 + hard-start
  回归 `164 passed`。测试后已恢复容器 `configure_app.toml` 原 SHA256
  `99224b5abcd377e69297eeb8d3f8fdf21cc05f10b265267e378722b23662ffb5`，并按用户
  授权删除上传分片、34728 预加载副本和 resume 临时副本。生成的
  第二次 smoke 启动会清理上一轮 `/data/user_ckpt_dir`，因此
  `responsebase-34743` 文件已不在容器；其路径、大小、SHA256 和解析结果已如实记录，
  不将它表述为仍可下载制品。

## BUG-20260728-003：开发容器 full-smoke 解析 symlink 后丢失项目运行根

- 状态：开发容器已验证。
- 影响：开发容器中 Nav/P1.5 的非 `KAIWU_TRAIN_TEST` 完整训练链 smoke；正式平台训练不走
  该工具，不受影响。
- 症状：后台 smoke 先报 `ModuleNotFoundError: No module named 'kaiwudrl'`；显式补充
  `PYTHONPATH` 后继续报 `tools/change_sample_server.sh: No such file or directory`。
- 根因：`nav_full_smoke._start()` 用 `Path(__file__).resolve().parents[2]` 作为子进程 cwd。
  开发容器的 `agent_ppo` 是指向 `/workspace/code/agent_ppo` 的 symlink，resolve 后 cwd
  离开 `/data/projects/legged_robot_competition_26`，从而丢失项目内 `kaiwudrl` namespace
  和平台 `tools` 目录。
- 排除项：34728 checkpoint SHA、P1.5 observation/optimizer、Isaac GPU 和生产 TOML 均尚未
  进入失败路径；这不是父包不兼容或 `base_env.py` 问题。
- 修复：启动器保留操作者调用时的 server 根目录，并在缺少 `train_test.py` 时明确拒绝；
  回归测试锁定传给 `subprocess.Popen` 的 cwd 等于调用 cwd。
- 验证：修复前两条原始错误已在 IDE 18005 稳定复现；修复后同一启动器已
  两次成功起动非 `KAIWU_TRAIN_TEST` 的 8 环境 Standard + Camera GPU 链，完成真实
  34728 preload、rollout、低层/Adapter update、checkpoint 与 resume。父包容器端
  SHA256 为 `0ce3b485053faa5da37c2b0ad792ad414ec6f5a482208c728cee65487d019d0f`；
  测试后已按授权删除预加载副本与临时分片。
- 防复发：任何后台 smoke 启动器都不得从 symlink-resolved `__file__` 推导平台项目根；最短
  检查路径为 child cwd → `kaiwudrl.__path__` → `tools/change_sample_server.sh` → preload。
- 血缘：分支 `codex/p15-response-adapter`；关联 commit/PR、输出模型 ID/SHA256 尚未生成。
- 遗留风险：调用者必须从 server 根启动；若未来需要任意 cwd，应新增显式 `--project-root`
  参数，不得再次从 resolved module path 猜测。回滚仅移除 smoke 工具修复，不触碰生产流程。
- 2026-07-28 补充：实测 `stop` 后父 PID 已死，但框架创建的新进程组仍可短暂存活；
  旧逻辑只等待 pidfile 中的父 PID，会误报已清理。工具现按 `--runtime-dir` 扫描
  `/proc/*/cmdline`，对同一 smoke 的所有已证明子进程组发送 TERM/KILL；并新增“父 PID
  已死、子进程仍活”回归测试。容器随后被平台回收为
  `WEBIDE_RECORD_NOT_FOUND`，所以该收尾修正当前只能标记本地验证，待下次容器 smoke 复核。
- 2026-07-28 容器复验：IDE 18005 中专项测试 `10 passed`；随后创建两个携带相同
  `--runtime-dir=/tmp/p15_proc_cleanup_check`、各自独立 session/process-group 的临时子进程，
  并把 pidfile 写成不存在的父 PID。`_owned_smoke_pids()` 在停止前识别出两个 PID，
  `_stop()` 返回 0 并终止两个进程组，停止后再次扫描结果为 `[]`。Nav + P1.5 +
  hard-start 支持回归集为 `165 passed`。该证据验证的是开发容器 smoke 工具的孤儿进程
  收尾，不替代当前正式 128 环境平台 smoke 或八小时长训。

## BUG-20260728-004：P1.5 256 环境在 ray-caster 初始化时报 CUDA illegal memory

- 状态：平台已验证（128 环境缓解方案）。
- 影响：`p15resp8h` 首次正式平台任务；环境 reset 失败，训练尚未进入 rollout、PPO 或
  ResponseAdapter 更新。
- 症状：learner 在 `gym.make()` 的 `self.sim.reset()` 阶段失败；CUDA 错误最终出现在
  `anisotropic_grid_pattern` 向 `ray_directions` 写入方向张量时，aisrv 随后只报告
  `_initialize_training_state` reset failed。关键原文为
  `RuntimeError: CUDA error: an illegal memory access was encountered`。
- 根因判断：当前标记为高概率初始化/rollout 显存压力，平台复验前不写成 scanner 逻辑
  已确认损坏。P1.5 为容纳 50 帧 future label、8 帧 burn-in 和训练序列，把低层 rollout
  从父阶段的 48 帧提高到 80 帧；256 环境使批量帧预算从父阶段已验证的
  `256x48=12288` 增至 `256x80=20480`，同时还增加 response aux、future records 和
  Adapter 状态。CUDA 异步错误可能在更早的分配/内核发生，只在 ray-caster 的下一次
  CUDA 调用处被抛出，因此栈顶不能单独证明 pattern 赋值是根因。
- 排除项：P1.5 TOML 没有覆盖 ray pattern/direction；Camera、Standard terrain grid 与
  已验证视觉父阶段一致。历史 273-ray 问题是 observation shape 解析失败，不是本次 CUDA
  illegal-memory。异常发生在 env reset，尚未使用 command scheduler、奖励、Adapter loss
  或 checkpoint。不得修改平台 `server/isaac_env/base_env.py`。
- 修复：生产 `num_envs` 从 256 固定为 128，80 帧 rollout 保持不变；批量帧预算降为
  `128x80=10240`，低于父阶段预算。地形、相机、scanner、摩擦日程、command 课程和八小时
  墙钟均不改变。配置测试同时锁定 128、80 和 10240，防止未来只恢复环境数却忘记
  future-label 对 rollout 长度的要求。
- 验证：本地 TOML 解析和 `128x80=10240` 合同检查通过，`git diff --check` 通过。修复文件
  已经精确同步到 IDE 18005；容器读取结果为 `num_envs=128`、`num_steps_per_env=80`、
  `batch_frames=10240`，配置专项测试 `5 passed`，完整 P1.5 回归 `38 passed`。开发容器
  8 环境完整训练链此前已验证；128 环境平台证据见下方复验记录。
- 2026-07-28 平台复验：任务 `p15resp8h-r1` / `234864` 使用修复后的 128 环境配置启动，
  已跨过 reset 并持续训练。05:32:51 日志为 `iter=6340`、`records=391`、
  `adapter_gradient_steps=13`、Adapter loss `0.20407`、MAE `0.26942`；05:33:08 保存
  `/data/user_ckpt_dir/.../model.ckpt-responsebase-34743.pkl`，SHA256
  `bc66279ac430c8d3fdedfa199a1d8ad53e0e1efb1a9d04070d59eb224d07edca`，05:33:30
  learner 报告平台 ZIP 复制成功。05:34:03 Standard monitor 已累计
  `completed=113, abnormal=14, timeout=8`。因此 128 环境缓解方案的平台启动、更新和保存链
  均已验证；“256x80 显存压力是唯一根因”仍保留为高概率解释，不扩写成已证明的 CUDA
  内核根因。
- 血缘：分支 `codex/p15-response-adapter`；父包 `commandfull-34728`，SHA256
  `0ce3b485053faa5da37c2b0ad792ad414ec6f5a482208c728cee65487d019d0f`；失败任务
  `234863`，成功任务 `234864`，输出模型 ID `34743`，checkpoint SHA256 见平台复验记录。
- 回滚与最短检查：若 128 仍在 reset 报同类错误，先用同包做 64 环境隔离，并设置
  `CUDA_LAUNCH_BLOCKING=1` 的开发容器诊断；随后比较 domain-rand startup event 和已验证
  visual 配置。不得通过改 scanner 几何、关闭 Camera 或降低 rollout 到 50 帧以下掩盖问题。

## BUG-20260728-005：P1.5 课程里程碑破坏十分钟模型保存节奏

- 状态：代码已修复待平台验证。
- 影响：P1.5 八小时任务的网页模型数量、保存间隔和恢复点可预测性。
- 症状：任务 `234864` 首个 bundle 于 05:33:08 保存，日志中的
  `anchor_session_h=0.032` 对应 workflow 启动后约 1.92 分钟；原配置随后每 10 分钟周期
  保存，但 30/75/120/420/470 分钟里程碑会插入额外模型，使部分相邻保存只有 5-8 分钟。
- 根因：`first_save_minutes=2`、`save_interval_minutes=10` 和独立
  `response_checkpoint_minutes` 三套触发同时存在。workflow 每次实际保存后重新以该时刻
  计算下一周期，因此里程碑不仅增加一个模型，还会整体平移后续周期。
- 修复：fresh/resume 的首存统一改为 10 分钟，常规周期保持 10 分钟，P1.5 独立里程碑
  列表清空。课程阶段仍按墙钟正常切换，由切换后的下一个周期包记录；正常结束和 SIGTERM
  的幂等最终保存继续保留，不计入常规周期。
- 验证：TOML 与配置测试锁定 `10/10/10` 分钟及空里程碑列表；本地解析、容器测试和平台
  实际间隔待修改后补录。正在运行的任务 `234864` 已在启动时加载旧配置，不支持热更新，
  因此本修复只影响后续任务，不要求中断当前成功训练。
- 回滚：恢复 2 分钟首存和课程里程碑列表；无需修改 workflow、checkpoint schema 或
  `base_env.py`。

## BUG-20260728-006：P1.5 adapter-only 包被 Camera 评估误判为 hier-nav

- 状态：本地已验证，平台重新评估待验证。
- 影响：P1.5 `response*` checkpoint 的 Standard/Track Camera 低层评估；训练、checkpoint
  schema、ResponseAdapter 状态和 low-level-only 续训数据本身未损坏。
- 首次发现：评估日志 `log-598368-18520933.zip`，请求模型 ID `37953`，命中文件
  `model.ckpt-responsecalib-37953.pkl`；评估任务 ID 未包含在下载日志中。训练血缘为
  `p15resp8h-r1`、父模型 `commandfull-34728`。
- 症状：平台把 Camera 评估入口推导为 `lbc_loco` 后，13:58:49 在真正加载权重前报
  `[LBC-Loco eval] this bundle contains modules.high_level (a hier-nav combined bundle)`，并错误
  建议改走 `nav_eval`；随后 `_exploit_lbc_loco` 正确拒绝在未加载 checkpoint 时推理，评估失败。
- 根因：P1.5 schema2 按模块化组合契约把辅助 ResponseAdapter 保存到
  `modules.high_level={component_status: adapter_only, response_adapter: ...}`。旧 Camera 防呆只
  检查 `"high_level" in modules`，没有区分“不会产生动作的 adapter-only 组件”和真正的
  hier-nav 高层动作策略，因此把合法 P1.5 包误拒绝。
- 排除项：候选选择已按请求 ID 正确优先命中 `responsecalib-37953`，不是文件名、探活、模型
  ID、Standard terrain、Camera task 或 `base_env.py` 问题。真实 checkpoint 可反序列化，
  `vision_encoder` 与 low-level Actor 均能 `strict=True` 加载并完成有限值前向；不应把该包改走
  `nav_eval`，也不应删除其 Adapter、optimizer 或训练状态来制造低层别名。
- 修复：`checkpoint_io.py` 新增最小分类函数。没有 `high_level` 时沿用历史低层评估；
  `component_status=="adapter_only"` 时不加载、不校验该未使用组件，继续严格加载视觉编码器和
  low-level Actor；其他 high-level 仍按可能的动作策略拒绝静默降级。`agent.py` 在成功日志中
  输出 `high_level=adapter_only_ignored_for_locomotion_eval`。该变化不修改候选标签、checkpoint
  schema、训练 loader 或 `low_level_only_parent_candidates()`，因此 Standard/Track 后续续训路径
  保持不变。
- 验证：`PY311test`（Python 3.11.13、PyTorch 2.11.0）运行评估/候选专项测试通过；真实
  `37953` 包分类为 adapter-only，视觉编码器与 Actor `strict=True` 加载后完成一次 CPU 前向，
  latent/action shape 为 `(1,32)/(1,12)` 且 action 全有限。评估专项为
  `27 passed, 3 subtests passed`，Nav + P1.5 + hard-start 支持回归为
  `164 passed, 3 subtests passed`，`git diff --check` 通过。开发容器同步、平台重新评估和视频
  均未执行，不能标记平台或评估已验证。
- 防复发：测试固定四条边界：adapter-only 即使不提供可用 Adapter state 也不得阻塞未使用的
  低层评估；真正 Nav state 必须拒绝；`responsecalib` 同 ID 候选必须可发现；视觉编码器和 Actor
  必须继续 `strict=True` 加载。日志必须同时给出 selected path、SHA256、bundle/lineage 和
  high-level disposition。
- 血缘与制品：分支 `codex/p15-response-adapter`，当前 P1.5 源码仍为未提交工作区，commit/PR
  待生成；原 checkpoint SHA256
  `5b3715491bd2a184f157aa37a63a598848b733b67d34b7474120a6e9e5cd6193`。制品级热修复包
  `p15resp8h-r1_37953-evalfix-v1.zip` SHA256
  `b251a4eefe56c316cc62463b9c21ad26eefdab80f305a5497812c9d4a28f404e`，checkpoint 字节未改。
- 回滚与再遇检查：回滚只需恢复旧分类调用，不修改 checkpoint；原始未热修复归档仍保留。
  再遇时依次检查 requested ID/candidate → selected path → `component_status` →
  `vision_encoder`/Actor strict load → loaded log；只有真正动作型 Nav 包才切换 `nav_eval`。
- 2026-07-28 审查更正：仅检查 `component_status="adapter_only"` 仍会让误标或恶意构造的
  Nav 动作策略绕过 Camera 防线。分类现改为接收完整 bundle，并同时要求
  `stage_type="p15_response_adapter"`、`modules.high_level` 只包含
  `component_status/response_adapter`，以及 Adapter spec 的 class/input/hidden/profile
  分别为 `CommandResponseAdapter/32/64/16`。Camera 路径仍不加载未使用的 Adapter 权重，
  但会在 `strict=True` 前分别检查 vision encoder 与 low-level Actor 权重有限性；新增伪造
  adapter-only、错误 stage 和嵌套 NaN 回归测试。该更正当前为本地代码修改，平台重新评估
  仍待验证，不能提升本条状态。
## BUG-20260727-008：Nav 控制链面板名称含斜杠导致监控配置整体跳过

- 状态：本地已验证，平台待验证。
- 影响：hier-nav Track/Nav 自定义监控面板加载；不影响训练计算、模型或 checkpoint。
- 首次发现：2026-07-27，任务名、平台任务 ID 和模型 ID 未记录。
- 症状：learner 启动时报 `Error occurred while loading user monitor config`，并列出
  `requested/effective 不一致`、`worker/exec 指令误差`、`worker/exec 指令匹配率`
  三个中文名称非法，随后 `will skip loading`。
- 根因：`agent_ppo/conf/monitor_builder.py` 的三个中文显示名包含平台白名单不接受的 `/`；
  builder 校验按整份配置失败处理，因此其他合法 Track/Nav 面板也被一并跳过。
- 排除项：英文面板 ID、metric key、指标上报逻辑、Track stage、`base_env.py` 和 checkpoint
  均与本错误无关。
- 修复：仅将三个显示名改为 `请求生效不一致`、`执行指令误差`、`执行指令匹配率`；
  英文 ID 和 metric key 不变。新增测试遍历 Track/Nav 面板规格，约束名称长度不超过 20，
  且只含中英文、数字、`*`、`-`、`_` 和空格。
- 验证：目标单元测试和 Python 编译通过；容器同步、平台 smoke、评估和真机未执行。
- 防复发：新增面板时必须通过统一字符与长度测试，不再只检查已知三个字符串。
- 血缘：分支 `codex/hier-nav-dagger`；不涉及父模型、checkpoint 或 SHA256；commit/PR 待生成。
- 回滚：恢复三个旧显示名即可，但会重新触发平台拒绝，因此仅用于定位差异。
- 再遇检查：先看 learner 的 monitor config 校验原文，再核对显示名字符与长度，最后确认
  容器已同步并重启；不要先排查训练 workflow 或模型加载。

## BUG-20260727-009：Nav 运行 checkpoint 正常但前端模型列表只在任务结束后更新

- 状态：平台已验证。
- 影响：hier-nav 训练中模型可见性、下载、阶段中途评估和长训恢复。
- 首次发现：平台任务 `234786`，父模型 `34728`，最终模型 `navbc-52313`。
- 症状：运行中日志连续记录 `/data/ckpt/model.ckpt-navbc-36000.pkl`、`39600.pkl`、
  `46800.pkl` 等非空文件，SHA256 正常且 lifecycle failure 为 0，但前端模型列表为空；
  任务正常结束后模型 `52313` 立即出现并可下载。
- 根因：运行中的 `dump_model_freq` 只触发 `/data/ckpt` checkpoint；用户可见归档还需要
  平台 wrapper 生成 `ckpt/id_list`、`ckpt/kaiwu.json` 并打包 ZIP。此前为避免业务侧重复
  保存而删除五分钟无参数 `agent.save_model()`，误把运行 checkpoint 当成前端归档发布。
- 排除项：下载包已证明 `navbc` 纯字母标签合法、4.49 MB 文件大小可接受、
  `kaiwu_train_v1/nav_dagger_v1` 可反序列化，且 `vision_encoder`、`low_level`、
  `high_level`、optimizer、父 ID 和 digest 均存在；因此不是文件名、模块缺失、schema、
  `base_env.py` 或 monitor builder 导致。
- 修复：`nav_dagger_workflow.py` 在完整 TBPTT outer iteration 边界按墙钟每五分钟调用一次
  无参数 `agent.save_model()`；不传 path/id、不创建别名，正常结束/SIGTERM final save 保持
  独立幂等。周期归档异常转换为 `CheckpointSaveError` 并停止，避免无可下载模型的长训。
  `dump_model_freq=3600` 保留为 `/data/ckpt` 运行 checkpoint。自动 dump 倒计时改为计入
  `algorithm.loaded_platform_model_id`，父 `34728` 的首边界正确为 1272 callback 后的 36000。
- 验证：最终包
  `legged_robot_competition_26-234786-ppo-52313-2026_07_27_20_20_30-27.0.21.zip`
  已由平台生成；本地 SHA256 为
  `36f5cff83884fd1f4adfe9fbc1d0b00c1ba21986aa0c3d737ec8b5dea0a6d97d`，包内 checkpoint
  `platform_model_id=52313` 且三模块齐全。本地完整 Nav 回归为
  `123 passed, 3 subtests passed`，相关 Python 编译、TOML 解析和 `git diff --check` 通过；
  容器同步和新平台 smoke 未执行，当前不得宣称运行中前端归档已验证。
- 2026-07-27 平台周期归档验证：任务 `234805`、父模型 `34728`在
  iteration 30 执行第 1 次无参数周期归档。平台注入模型 ID `39528` 和路径
  `/data/user_ckpt_dir/legged_robot_competition_26_ppo/model.ckpt-navbc-39528.pkl`；
  日志确认 `path_category=final_platform_archive_candidate`、文件大小 `4492589`
  字节、checkpoint SHA256
  `ca8ab4d42d510fa6aff8fdc20f4662cde611bfdd332b321f25ef6918fba93bea`。
  24 秒后平台打包
  `legged_robot_competition_26-234805-ppo-39528-2026_07_27_21_15_16-27.0.21.zip`
  并记录 `copy to /workspace/train/backup_model/ success`；用户已在网页模型列表确认
  `39528` 可见。这证明“运行中无参数 `agent.save_model()` →
  `/data/user_ckpt_dir` → ZIP → 前端模型列表”全链路有效。
- 同一时间窗的反例特征：周期归档后 4 秒，`dump_model_freq=3600` 又以
  ID `39600` 写入
  `/data/ckpt/legged_robot_competition_26_ppo/model.ckpt-navbc-39600.pkl`，分类为
  `running_checkpoint`，SHA256 为
  `671258ae5de0d87cbb9b381e3852ce96a3bee02236014e3dc1166f573bc36fb5`。该文件与前端归档
  `39528` 不是同一次保存；不得用 `/data/ckpt` 成功日志代替前端发布证据。
- 防复发：测试覆盖 300 秒边界、周期保存与 final 幂等隔离、周期保存异常硬停止、父模型
  ID 偏移下的自动 dump 边界，以及平台仍注入 `/data/ckpt` 时的醒目错误。日志必须同时
  打印平台注入的 path category、绝对路径和数字 ID。
- 血缘：分支 `codex/hier-nav-dagger`；父模型 `34728`；最终 checkpoint `navbc-52313`；
  commit/PR 待生成。
- 遗留风险：此前“周期归档待 smoke”已由任务 `234805` 关闭。平台打包到前端
  列表可见仍可能有数十秒异步延迟，应先查 `final_platform_archive_candidate`和
  `backup_model success` 再判定失败。最终 ZIP 还可能包含 `conf/.env` 和缓存文件；
  这不影响 checkpoint 有效性，但发布前必须另行清理远端敏感配置并轮换同步凭据。
- 回滚：删除 `platform_archive_interval_minutes` 与周期归档调用即可恢复“仅任务结束可见”；
  不得修改 checkpoint 标签、父模型或 `base_env.py`。
- 再遇检查：save path category → `id_list`/`kaiwu.json` → ZIP 是否生成 → 前端模型列表；
  不要先调 `dump_model_freq`、文件名或网络结构。

## BUG-20260728-007：P2 高层专用 storage 与训练边界初版存在解冻后崩溃和序列污染风险

- 状态：本地已验证。
- 影响：`p2nav2h` 两小时 Track 连续高层 PPO；涉及十分钟 CNN 解冻、ResponseAdapter
  future label、recurrent reset、variable-duration GAE、checkpoint exact resume、显存降级与
  平台周期归档。P1.5/Standard/Nav 既有训练入口不应受影响；平台拥有的
  `server/isaac_env/base_env.py` 未修改。
- 复现背景：分支 `codex/p2-track-nav-ppo`，基线 commit
  `b1ace13918f47776245bde13fa7fe9c972d5a9e3`，父包 `responsecalib-37953`；本地归档
  `p15resp8h-r1_37953.zip` SHA256
  `134b8388220309131d676b83f672887ded26ed2d71f942b6a0bb6a61d7feb206`。生产配置为
  Track/Camera、128 环境、32 high-level ticks、TBPTT16、固定 slope→open-maze，TOML
  SHA256 `24dd4905a41734b65d10e953b18d73c9effe8ea263d0c2936517616b93025d00`。
  当前没有平台任务 ID、P2 模型 ID 或 checkpoint SHA256。
- 预审症状与根因：初版解冻后把 Camera depth `[N,180,320,1]` 直接 `copy_` 到
  `[N,57600]` CPU FP16 storage，形状不可广播，会在十分钟边界后的首个 depth rollout
  必现失败。ResponseAdapter 初版还混用了 P2 Actor capability 与 P1.5 Adapter capability，
  done slice 和 episode-start reset 各有一帧偏移，可能让终止前 hidden 或未来标签污染新
  episode。PPO 初版按 rollout 全局归一化 advantage，并在 4-sequence CNN microbatch 层面
  混用 Actor/Critic 数据，不能满足完整 64-sequence minibatch 口径。exact resume 若缺
  scheduler/RNG/buffer 或含非有限状态也不能静默降级。
- 排除项：历史平台任务已证明无参数 `agent.save_model()` 会由 wrapper 注入
  `/data/user_ckpt_dir` 路径与数字 ID，因此 P2 的周期/最终无参数保存不是
  `path=None/id=1` Bug，不得改成 iteration ID 或业务侧自造路径。P1.5 的 128 环境
  CUDA illegal-memory 发生在 256 环境 ray-caster 初始化，不能据此预判 P2 128 环境必败；
  本轮仍必须以真实 rollout/backward 峰值决定是否重启为 96/80 环境。
- 核心修复：新增独立二维 squashed-Gaussian high-level Actor/Critic、CPU-backed P2 storage、
  `critic323|aux30` worker transport、5 Hz target/50 Hz slew、P1.5 shared FeedbackEmulator、
  terrain curriculum probe 与高层 tick reward。depth 入库前现在严格校验元素数并显式展平；
  frozen/unfrozen rollout 分别保存 nav feature 或 pinned FP16 depth。Adapter Track/parent
  records 分别使用 P2 response capability 和 P1.5 capability，future done 边界覆盖当前
  transition，episode reset mask 移到 termination 后的 observation，resume 只恢复 completed
  records并清空 unfinished history。PPO advantage 在完整 minibatch 统计，microbatch loss
  按有效 timestep 比例缩放，Actor/Critic 独立传输与 backward。timeout 只在 delta bootstrap
  一次且 continuation 为零。exact resume 严格要求高层模块、三套 optimizer/scheduler/RNG、
  计数和兼容 buffer，并对模块/optimizer tensor 做有限值检查。同 ID loader 先尝试
  `navfull/navadapt/navwarm` exact resume，仅在不存在 P2 包且 ID 为 37953 时回退
  `responsecalib` 父包；保存端在原子替换前检查六个活动模块与三套 optimizer 的有限值。
  CNN 解冻同时更新 optimizer group LR、`initial_lr`、LambdaLR `base_lrs/_last_lr`，防止
  scheduler 在首个解冻 update 后把分层学习率重新归零。
  128→96→80 的显存降级允许对 completed records 的环境维做向下切片，模型、optimizer、
  RNG 与有效训练秒数仍精确恢复；反向扩容仍拒绝伪造 exact resume。Adapter update 单独捕获
  OOM，跳过当次辅助 step并把 batch_envs 减半，不回滚或污染已完成的高层 PPO update。
  后续静态审查还发现 tracking penalty 原先在新动作刚采样、尚未执行时取样，会把上一指令
  的响应误归到新 transition；现已移至 `finish_tick()`，使用 10 个低层帧结束时的
  `exec_cmd/measured_velocity`，command-rate 仍在动作切换时计算。高层命令方向、反馈有效率、
  confidence、三项 tick penalty、Adapter 三 horizon MAE/NLL/有效率、课程累计值与完整性能
  时序现均进入 monitor metrics，所有项目只观测、不形成训练门禁。另修复 lifecycle
  自动 dump 抛 `CheckpointSaveError` 时被特殊向外重抛、违背跑满两小时策略的问题；现在
  与其他 lifecycle 异常一致只累计告警，并安排 60 秒后的无参数平台归档重试。最后将
  `Config.CURRENT` 的字面默认从 P1.5 同步为 P2，使配置 bootstrap、训练 TOML 和首次加载前
  的 stage 读取保持一致。
- 保存与运行策略：首存约五分钟、之后十分钟及 CNN 解冻边界使用无参数平台归档；
  normal end/SIGTERM 共用一次幂等 final save。训练指标、curriculum、Adapter 质量和低层 digest
  漂移只告警；非有限 minibatch 跳过 optimizer step。显存首先把 CNN microbatch 从
  64 frame 降到 32 frame并开 activation checkpointing；环境数调整只能新建 96/80 环境任务，
  不在已创建 Isaac/CUDA 进程中热改。
- 当前验证：新增 action/log-prob、GAE、row/column probe、capability 分离、future done/reset、
  CPU buffer resume、128/96/80 sequence 数、depth flatten、timeout mask、累计 CNN 解冻和
  final-save 幂等测试，并补充 transition 末端 tracking penalty 与 curriculum monitor
  flatten 回归。Python `py_compile`、两份 TOML 解析、P2 面板名称字符检查、`git diff --check`
  及同步客户端 `27 tests` 已通过。腾讯 IDE dry-run 仍返回
  `ApiUserErrors.WEBIDE_RECORD_NOT_FOUND`；宿主机无 PyTorch，因此 pytest、真实 37953 strict load、GPU
  rollout/update/save/resume、128 环境峰值、平台 smoke、模型列表与评估均未验证；状态不得
  提升为本地/平台/评估已验证。
- 防复发：CNN 解冻路径必须使用真实 Camera shape 完成至少一轮 storage add 与 PPO backward；
  same-ID parent/resume 需检查 stage/component/spec/finite；P2 Actor capability 与 Adapter
  piecewise-union capability 必须分别锁定；future history、live recurrent hidden、未完成
  rollout 和 per-env terrain state 不进入 checkpoint。运行时必须同时报告 allocated/reserved
  当前值与峰值、pinned depth bytes、H2D/rollout/update 时间。
- 遗留风险与回滚：平台 eval 是否同时向 `agent.exploit()` 提供 policy observation 与
  critic/response wire 仍需 smoke；timeout terminal observation 能否由 wrapper 精确提供也需
  容器确认。若 128 环境 OOM，按 96→80 新任务回退并从已有 P2 checkpoint 累计有效训练秒数；
  若无有效 P2 包则从 37953 重新跑满两小时。整体回滚为切回 P1.5 `responsecalib-37953`，
  不删除父包、不修改 base env、不将 P2 标记为 deployable。
- 关联：commit/PR、平台任务名 `p2nav2h` 对应的任务 ID、最终模型 ID、checkpoint SHA256
  均待完成后补录。

- 2026-07-28 复审更正（状态仍为“代码已修复待验证”）：此前本条写成“timeout 只在 delta
  bootstrap 一次”，该结论隐含平台能提供 terminal critic observation，现已确认不成立。
  `server/isaac_env/base_env.py` 在 trajectory recorder 关闭时会把 `truncated` 覆盖为全零，
  返回 infos 也不保留原始 `time_outs`；同时 auto-reset 后公开 observation 已属于新 episode。
  修复改为 worker 在现有 `response_aux30` 的 aux24 写 reset boundary、aux25 写
  `0 none/1 success/2 failure/3 timeout`，aisrv 从下一帧 wire 恢复 done/timeout。当前无 terminal
  observation 时 timeout 使用 `bootstrap_mask=0`、`continuation_mask=0`，明确禁止拿 reset
  后状态为旧 transition bootstrap；未来只有平台提供显式 terminal observation/value 后才可
  恢复单次 timeout bootstrap。
- 同轮复审还确认并修复：高层 `mean/log_std/pre_tanh/log_prob/target/value` 非有限时原实现
  只替换 target，仍会把非有限 transition 写入 storage；现在相关 env 执行 zero、所有存储字段
  清洗、recurrent hidden 清零并标记整轮 PPO 跳过，Adapter update 和两小时任务继续。Actor
  采样新增 device-local action RNG，与 sequence shuffle、neutral dropout、Adapter RNG 分开保存
  和 exact resume。P2 schema2 新包补齐 `bundle_kind`、实际 `train_scope` 以及每个可学习 leaf 的
  `class_name/spec/state_dict`；训练 resume 继续严格恢复三套 optimizer/scheduler/buffer，
  `p2_nav_eval` 则使用模块-only 装配，不再创建或恢复 Critic、optimizer、scheduler、PPO storage
  和 ResponseBuffer。
- 课程诊断原实现尝试从 aisrv 代理穿透读取 worker 的 env-owned probe，平台进程边界下会永久
  得到空快照。现由 aisrv 使用 aux24/25/28/29 独立累计 row/column movement、outcome、start 和
  histogram，并将累计值写入 checkpoint；per-env previous row/column 不保存，resume 首帧只建立
  新边界，避免跨任务伪造一次 reset 迁移。请求 ID 与 payload 内历史 `platform_model_id` 不一致
  仍遵循仓库既有身份策略只告警，实际候选继续由平台请求 ID/同 ID 文件名选择，没有升级为新的
  单点硬门禁。
- 真实 `responsecalib-37953` 结构核验进一步发现：父包的动作方差位于
  `modules.action_distribution`，S0 anchor 位于 `modules.s0_anchor`，初版 P2 payload 只重建
  `modules.low_level/high_level`，会在首存时丢弃这两组冻结状态。现将其迁移为
  `modules.low_level.action_distribution` 与 `modules.low_level.s0_anchor` 的规范 leaf；真实
  37953 已完成 CPU bootstrap、P2 schema2 保存和 exact resume，迁移后 low-level 键包含
  `action_distribution/actor/critic/locomotion_encoder/s0_anchor`，父 completed records 为 32。
- 本次新增回归覆盖 timeout aux 恢复、post-reset no-bootstrap、课程累计/恢复、终止原因优先级、
  非有限 transition 清洗、action RNG exact resume、leaf 契约 round-trip 和 eval 无训练状态；
  宿主 `PY311test` 的 P2 专项为 `33 passed`；排除仓库已知失效的旧
  `test_j9_fixed_lr.py` 与 `test_st9_opt3_d2.py` 后，训练端回归为
  `207 passed, 3 subtests passed`。真实 37953 已完成 CPU bootstrap、schema2 保存和 exact
  resume；GPU rollout/update/save/resume、128 环境显存、平台 smoke、模型列表与评估仍未执行，
  因此不得升级为“本地已验证”或更高状态。

- 2026-07-28 开发容器 full-stack preflight 补充（状态仍为“代码已修复待验证”）：使用真实
  `responsecalib-37953`、生产 P2 配置和 128 环境启动时，aisrv 在 Isaac reset 前明确拒绝配置：
  `Configuration validation failed ... [terrain.track] sub_terrains_random is not supported and must not be configured`。
  根因是生产 TOML 沿用了设计稿中的 `sub_terrains_random=false`，但当前平台配置白名单禁止该键
  存在；布尔值为 false 也不会绕过校验。现从活动 TOML 删除该字段，固定
  `pyramid_slope -> open_entry_maze` 顺序继续由 `sub_terrains` 列表表达，并增加配置回归断言
  禁止重新写入该键。首次失败未创建环境、未执行 rollout，也不是 CUDA/OOM；后续 broken-pipe
  与子进程 `signal_killed` 均为 aisrv 首错后的派生噪声。需重新同步并取得真实 reset、首轮 update、
  显存峰值和 checkpoint 证据后，才能提升验证状态。

- 2026-07-28 第二轮开发容器 full-stack preflight 补充（状态仍为“代码已修复待验证”）：删除非法
  `sub_terrains_random` 后，128 环境已完成真实 Track/Camera 创建，环境创建阶段显存约
  `3693 MiB / 5000 MiB`，但首次 reset 在平台 curriculum manager 中失败：
  `ValueError: Reward term 'track_lin_vel_xy' not found`。根因不是 OOM，也不是 P2 goal reward
  实现，而是当前平台把 `[terrain] curriculum=true` 同时解释为启用
  `terrain_levels`、`lin_vel_cmd_levels` 和 `ang_vel_cmd_levels`；后两项固定查询
  `track_lin_vel_xy` / `track_ang_vel_z`，P2 的全量 reward override 未注册它们。平台源码同时
  证明 Track 的 `terrain_levels_vel` 会升降 `terrain_types`（难度列），不是升降赛道 row。
  为保留用户批准的通用 terrain curriculum 且不修改平台 `base_env.py`，P2 TOML 现注册两个
  `weight=0.0` 的 compatibility-only tracking term。Isaac RewardManager 不删除零权重 term；
  原生 command curriculum 读取到的 episode sum 与 `0.8 * weight` 阈值均为零，严格比较
  `0 > 0` 为假，因此不扩张平台 fallback command，两项也对 P2 reward 数值贡献严格为零。
  新增配置回归断言锁定 term 名、零权重和参数。后续 broken pipe、`env.reset returned None` 与
  `signal_killed` 均为首次 reset 异常后的派生错误。仍需重新同步并取得 reset、32 tick rollout、
  PPO/Adapter update、CUDA 峰值、checkpoint save 与 exact resume 证据后升级状态。

- 2026-07-28 正式训练前开发容器验证补充（状态升级为“本地已验证”）：父包使用容器原生
  `/data/pre_model/ckpt/model.ckpt-responsecalib-37953.pkl`，SHA256
  `5b3715491bd2a184f157aa37a63a598848b733b67d34b7474120a6e9e5cd6193`。生产 128 环境
  Track/Camera 已完成真实 reset、32 个高层 tick、冻结低层 inference、Actor/Critic/Adapter
  update，lifecycle callback `320` 次且失败为 `0`；首轮 `max_memory_allocated=535753216`、
  `max_memory_reserved=645922816`，未出现 CUDA/OOM。运行 checkpoint
  `/data/ckpt/.../model.ckpt-navwarm-38400.pkl` 大小 `21033482`、SHA256
  `0943d595803e1d8d6f11e641ed00c7141f9eaf9073e09d1ff9e6d520c6e08455`；SIGTERM final
  candidate `/data/user_ckpt_dir/.../model.ckpt-navwarm-38723.pkl` 大小 `21033482`、SHA256
  `0349f936fa651822abd9e85904cc170dbfd74aecd65ff986f870f10abefb7309`。后者已用 CUDA、
  128 环境完成 `exact_resume_history_reset`，恢复 `iteration=2`、`nav_ticks=9984`、三套
  gradient step `32/32/2`、有效训练秒 `59.8843`，unfinished history 为零、completed records
  为 `32`。
- CNN 解冻规格通过独立 128 环境 probe：pinned depth `471859200` bytes（450 MiB），
  64-frame microbatch 完成 Actor/Critic/Adapter backward，`h2d_time_s=0.38294`、
  `max_memory_allocated=477448704`、`max_memory_reserved=530579456`，无需降级 microbatch 或
  activation checkpointing。最后修正 worker curriculum 探针的初始化时序：旧实现会在首次
  Track reset 前把 Isaac 的临时均匀列分配写成 initial histogram，并且只读取
  `manager.active_terms`；新实现等待显式列初始化标记，当前平台无该标记时退化为等待首个非零
  episode length，同时回退读取 `_term_names`。该问题只影响课程诊断日志，不影响实际训练；
  checkpoint 已证明实际初始分布为 row0/col0 全 `128`。宿主和容器 P2 专项均为
  `35 passed`；按当前分支实际存在的测试文件执行 Nav/P1.5/P2 邻近回归为 `209 passed`，同步后
  二次 dry-run 为零差异。容器中另外残留 6 个本地分支已不存在的旧 Standard 测试文件，它们在
  pytest 收集期引用已归档符号而失败；未将其计入当前分支回归，也未越权删除。平台正式任务、
  模型列表和评估尚未执行，状态不得提升为“平台已验证”或“评估已验证”。

## BUG-20260728-008：P2 面板碎片化且课程 reset 日志逐环境刷屏

- 状态：本地已验证。
- 影响：`p2nav2h` 的前端监控可读性和 aisrv 日志量；不改变 PPO、奖励、课程状态、模型权重、
  checkpoint schema 或平台 `base_env.py`。当前已运行的平台任务不会热加载本修复，需下一次任务
  启动时生效。
- 用户可见症状：P2 页面包含大量旧的、没有数据的 Track 计分面板，同时 2026-07-28
  `21:25:58` 至 `21:26:08` 连续出现多条单环境
  `p2_curriculum_reset_batch_aisrv`。样本均为 `old_row/new_row=0`、`old_col/new_col=0`、
  `termination_reason_code=2`；同窗口 `EnvMonitor` 报告 `completed=0, abnormal=5, timeout=0`。
- 含义与根因：reason code `1/2/3` 分别为成功/失败/超时，因此日志确实表示 row0/col0 的环境
  失败后 reset，课程没有晋级或换列，并非训练循环重新启动。噪声来自 aisrv 每个低层帧解析
  aux24/25 后立即 INFO；128 环境的 reset 分散到不同帧时，所谓 batch 经常只有一个 env。
  面板侧则把 53 个 P2 指标各建一个图，并继续注册 `completed_count_track_l0..9` 等 70 条旧 Track
  计分序列；仓库 P2 workflow 没有这些旧 key 的生产者，所以页面出现空图。
- 核心修复：`monitor_builder.py` 将 P2 重组为训练收敛、导航控制、响应预测、课程诊断和性能资源
  五组共 30 个多曲线面板，移除 P2 下的旧 Track 计分组，Nav/P1.5 独立入口保持不变。新增
  rollout reward/return/value/advantage、动作探索标准差、Actor/Critic/Adapter 学习率、累计非有限
  跳过和目标进度等只读 telemetry，使面板可以直接判断收敛、探索、导航进展和资源余量。
  `P2CurriculumAccumulator` 改为每 60 秒输出一条 `p2_curriculum_reset_summary_aisrv`，聚合
  success/failure/timeout、row/column move 和 reset 总数，只保留最多 12 个样本用于定位；累计
  histogram、monitor metrics 和 checkpoint 内容不变。
- 排除项：日志中的 code 2 不是 timeout，也不是 curriculum 自己触发的重启；旧 Track 空面板与
  奖励计算、父包 `37953`、GPU 显存和 checkpoint 保存链无关。当前失败占比是否过高仍需结合
  新面板的目标距离/进度、终止累计和指令响应判断，不能因减少日志就认定训练效果已改善。
- 验证：宿主 Python 编译、`git diff --check` 和 P2/Nav/P1.5 相关测试 `66 passed`；开发容器同组
  测试 `66 passed, 1 warning`。容器原生 `MonitorConfigBuilder.build()` 成功生成五组 30 个面板，
  中文名称和 metric expression 均通过平台 builder；同步后待执行零差异复核。正式平台新任务、
  前端面板显示和日志降频尚未验证，因此状态不能提升为“平台已验证”。
- 防复发：P2 新面板必须引用 workflow/algorithm 直接上报的 metric key；禁止重新引入仅由旧
  Standard/Nav 假定存在的 `*_track_l*` 序列。高频 per-env 事件必须先在进程内聚合，INFO 日志
  只输出窗口摘要，精细计数进入 monitor/checkpoint。
- 血缘：分支 `codex/p2-track-nav-ppo`，父包 `responsecalib-37953`；当前平台任务 ID、模型 ID、
  commit/PR 均未记录。回滚可恢复旧 `_build_p2_monitor()` 和逐事件日志，但会重新产生空面板与
  日志洪泛。
- 再遇检查：确认 `policy_entry=p2_nav_ppo` → 查看 monitor builder 加载错误 → 核对新任务是否
  包含五个 P2 group → 检查 `rollout_reward_mean/goal_progress/curriculum_failures` 是否上报 →
  确认 reset 日志为每分钟 summary，而不是逐环境 batch。

## BUG-20260728-009：P2 Track checkpoint 被旧 Nav 评估入口拒绝

- 状态：本地已验证，平台评估待验证。
- 影响：P2 `navwarm/navadapt/navfull` 模型的 Track+Camera 评估入口，以及 P2 训练效果监控；
  不修改 PPO、奖励权重、checkpoint schema、网络输入、平台 `base_env.py` 或部署端。
- 任务与证据：分支 `codex/p2-track-nav-ppo`，父包 `responsecalib-37953`，请求模型 ID
  `43393`，失败日志 `/Users/nanbloom001/Downloads/log-598611-18525725.zip`。评估任务 ID
  `598611`，关联运行号 `18525725`；训练任务名 `p2nav2h`，平台训练 task ID 未在日志包中记录。
- 用户可见症状：2026-07-28 21:33:38，aisrv 先打印
  `Override Config.CURRENT: p2_nav_ppo -> nav_eval`、`Stage: nav_eval`，随后旧 loader 报
  `[nav-eval] no same-ID nav checkpoint for id=43393`，候选仅包含
  `navfull/navdagger/navbc`。21:34:04 因 checkpoint 未加载拒绝推理，之后
  `NoneType object is not subscriptable` 与 `process_stop error_code 1` 均为派生错误。Track
  环境稍后成功创建为 `pyramid_slope -> open_entry_maze`，因此地形、CUDA、reward 和模型行为
  不是本次首错。
- 根因：`conf.py::_infer_stage_from_task_name()` 在平台 eval TOML 没有转发 `policy_entry` 时，
  对所有 Track+Camera 无条件返回历史 `NavEvalConfig`，覆盖仓库已由 `configure_app.toml`
  建立的 P2 lineage。旧 `nav_eval` 只识别 DAgger/Nav 命名空间和离散高层契约；P2 首存约五分钟
  时通常为 `navwarm-<id>.pkl`，应由 `p2_nav_eval` 的模块化连续高层装配加载。日志中的
  `existing_nav_files=[]` 只过滤旧 Nav 文件，不能证明同目录没有 P2 文件。
- 排除项：不是 checkpoint 完整性门禁“过严”。旧 Nav loader 在错误装配下拒绝未知模型是正确
  防线；直接让它接受 `navwarm` 会把连续 P2 Actor、85 维输入和 ResponseAdapter 误装进离散
  DAgger 高层，风险高于硬失败。也不允许在加载失败后用随机参数继续评分。
- 核心修复：显式 eval `policy_entry=p2_nav_ppo` 自动转换为 `P2NavEvalConfig`；无显式入口的
  Track+Camera 根据 `Config.CURRENT` bootstrap lineage 选择 `p2_nav_eval`，旧 Nav/Dagger lineage
  仍选择 `nav_eval`。P2 路径继续只搜索同 ID `navwarm/navadapt/navfull`，使用已有 eval-only
  assembly，仅创建低层、NavigationEncoder、连续 Actor 和 ResponseAdapter；Critic、optimizer、
  scheduler、PPO storage 和 ResponseBuffer 不创建。schema、module spec、shape、有限值和同 ID
  选择仍严格校验。
- 监控补强：此前五组 30 面板仍缺少十项实际奖励贡献和多个效果判据。P2 现扩展为八组 45
  面板、110 个去重 metric key，新增 `reward_approach_goal`、速度投影、航向、距离、成功、时间、
  终止、姿态、能耗、body contact，高层三项 penalty，以及 success/failure/timeout、推进速度、
  target/exec/measured/true 速度、有效反馈误差、source/age、域外探索、机身倾斜、横向漂移和动作
  饱和。新增 telemetry 只读取 rollout/aux/action，不参与 loss、reward 或阶段推进。
- 修改文件：`server/agent_ppo/conf/conf.py`、`server/agent_ppo/workflow/p2_nav_ppo_workflow.py`、
  `server/agent_ppo/conf/monitor_builder.py`、两组相关测试、`server/README.md`、
  `server/CHANGELOG.md` 与本台账。
- 验证：宿主 `PY311test` 下 P2/Nav 定向回归 `64 passed`，Python 编译和 `git diff --check`
  通过；AST 校验得到 P2 monitor `8 groups / 45 panels / 110 unique metrics`。宿主缺少平台
  `kaiwudrl.common.monitor.MonitorConfigBuilder`，因此本轮尚未取得平台原生 builder、容器同步、
  真实 `43393` 加载或 Track 评估证据，状态不得提升为“平台已验证”或“评估已验证”。
- 2026-07-28 回迁核验：将同一评估路由修复最小回迁到模型 `61633` 的代码快照
  `archive/代码存档/legged_robot_competition_26-ppo-61633`，仅更新 `agent_ppo/conf/conf.py`
  和对应路由回归测试；归档自带的 `eval_model_id=61633` 保持不变。真实 checkpoint
  `model.ckpt-navadapt-61633.pkl` 已直接核验为 `kaiwu_train_v1/schema2`、
  `stage_type=p2_nav_ppo`、`bundle_kind=hierarchical_control_v3`、完整 high-level 组件，SHA256
  为 `26aabe5f7eb5b590349fc3b46823a1934e9dc686862c506764d5ee45cbd409ed`。模块-only 加载返回
  `evaluate_full_modules_only`，CPU 确定性前向得到有限 `action[1,12]` 和有限
  `target_cmd3`；活动回归 `64 passed`、快照路由回归 `25 passed`、ZIP 完整性检查通过。
  修复包为 `legged_robot_competition_26-ppo-61633-evalfix-v1.zip`，SHA256
  `1a5ba2ff98991c837b57c7c0ed28ebd27c5e37dcdc6e0614f00169bb674373dc`；包内 checkpoint
  字节未改变，且未包含 `.env`、`__pycache__` 或 `.pyc`。容器同步和平台 Track 评估尚未执行，
  因此状态仍为“本地已验证”。
- 2026-07-28 二次更正（评估任务 `598619` / 运行号 `18526070`）：v1 路由回迁实际已生效，
  日志依次出现 `selected p2_nav_eval`、`Stage: p2_nav_eval` 和 eval-only assembly；新的首错是
  22:20:44 `P2 evaluation requires both policy observation and the worker response/critic transport`。
  平台标准 eval workflow 只转发 policy observation，worker 生成的 `critic323|response_aux30`
  critic group 没有进入 aisrv；同一日志也没有任何 `load_model`、`navadapt-61633` 或
  `evaluate_full_modules_only` 证据，因此修 transport 后还存在随机权重评分风险。修复保持
  policy 维度 57905：仅在 eval worker 把 aux30 和 marker 写入 P2 不消费的 scan 前缀，aisrv
  拆出后构造零 critic323 前缀供 `frame_begin()` 运输；训练路径不变。首次 exploit 同时从 runtime
  `eval_model_dir/eval_model_id`（必要时回退包内配置）严格加载同 ID checkpoint，未加载仍拒绝评分。
  这两项是评估链的必要一致性检查，不新增表现门禁、重复 schema 检查或自动停止策略。
  活动 P2/Nav 回归 `68 passed`，61633 快照回归 `64 passed`；真实
  `model.ckpt-navadapt-61633.pkl` 已通过修复后的 `Agent.exploit()` 自动定位并返回
  `evaluate_full_modules_only`，确定性 action `(1,12)` 与 target 均有限。v2 修复包
  `legged_robot_competition_26-ppo-61633-evalfix-v2.zip` SHA256 为
  `e5c809f747a17c9d1e2d9936ed5a660fb5d6f928ee2ac603afb339ae24367d5f`，包内 checkpoint SHA256
  仍为 `26aabe5f7eb5b590349fc3b46823a1934e9dc686862c506764d5ee45cbd409ed`。平台复评尚未执行，
  状态仍为“本地已验证”。
- 2026-07-28 `99393` 回迁核验：原始包
  `legged_robot_competition_26-ppo-99393.zip` SHA256 为
  `6c9e704edfa295cee8efedd7624636caf8fdd709202265c6afe79c97566773c1`。其六个评估相关文件与
  修复前 `61633` 快照逐字节一致，因此原样回迁 v2 的 P2 eval 路由、单流 aux30
  运输、首次 `exploit()` 同 ID 加载和对应回归测试，保留 `eval_model_id=99393`。
  快照定向回归 `64 passed`；真实 `Agent.exploit()` 输出
  `evaluate_full_modules_only`，选中 `model.ckpt-navadapt-99393.pkl`，确定性 action shape
  为 `(1,12)`，action 和 `target_cmd3` 均为有限值。修复包
  `legged_robot_competition_26-ppo-99393-evalfix-v2.zip` SHA256 为
  `fb5156916b1406ab9298ad6b67a6e5dd4e852d56911ea9605b195df87f57f305`；包内 checkpoint
  SHA256 仍为 `eb682662c113fb0aa8d015aa4ea4b8756661b81026273e54e7fae7ee465b4556`，
  且未包含 `.env`、`__pycache__`、`.pyc` 或 `.pytest_cache`。容器同步和平台 Track
  评估尚未执行，状态仍为“本地已验证”。
- 防复发：评估测试同时覆盖“P2 无显式入口保留 lineage”“显式训练入口提升为 eval-only”和
  “旧 Nav 无显式入口保持 nav_eval”。模型列表可见不等于 loader 已选对；每次评估必须依次确认
  stage、候选 namespace、selected path、load mode 和有限前向。
- 血缘与制品：commit/PR、模型 ZIP 和 SHA256 未记录；模型 ID `43393` 的实际内部标签需通过模型
  ZIP 或下一次加载日志确认，当前按训练时间高置信推断为 `navwarm`，不写成已直接核验事实。
- 回滚与最短检查：回滚本条只需恢复旧 eval 映射和新增面板/telemetry，不修改 checkpoint 字节。
  再遇时先查最早的 `Stage:` → 查看是否为 `p2_nav_eval` → 核对同 ID P2 candidate → 确认
  `evaluate_full_modules_only` → 再看环境、动作和视频；禁止从末尾 `NoneType` 反推模型损坏。

## BUG-20260729-001：P2 两小时奖励与双段统计合同不适合八小时三段长训

- 状态：本地已验证，开发容器与平台待验证。
- 影响：分支 `codex/p2-track-nav-ppo` 的 P2 高层 PPO、ResponseAdapter、课程统计、监控和
  checkpoint exact resume；默认 warm-start 父包为 P2 `99393`。不修改或上传平台拥有的
  `server/isaac_env/base_env.py`，不改变默认部署 runtime，训练包继续 `deployable=false`。
- 用户可见症状与证据：此前 `p2nav2h` 训练中课程列提升、失败数下降但完成数长期为零，末段
  姿态/能耗改善同时 `navigation_time` 更接近完整超时。旧 50 Hz 奖励允许静止持续领取绝对
  距离/航向正回报，progress 又被环境 `dt` 二次缩小；旧 scorer 还可能在 auto-reset 后用新
  `terrain_types` 给刚结束 episode 归档。新任务要求逆缓坡、下台阶、迷宫三段连续训练 8 小时，
  原 `2x10` 探针、aux30-only wire、两小时调度和旧 reward contract 均不再匹配。
- 根因：导航任务目标和低层安全 shaping 混在 50 Hz RewardManager 中，导致高层的任务进展
  信号相对弱且存在保守策略漏洞；terminal 前 column/row/goal-distance 没有稳定跨 worker wire
  传给 aisrv；CNN 解冻状态曾可在非 rollout 边界先切换；worker 成功 term 只读取
  `active_terms`，平台仅暴露 `_term_names` 时会误分类。监控只覆盖旧 reward 和少量全局均值，
  无法按命令联合域、三段 Track、逐腿 gait 或 Adapter 域别定位退化。
- 核心修复：任务改为 `p2nav8h`、28800 秒与固定
  `pyramid_slope_inv -> pyramid_stairs_inv -> open_entry_maze`。5 Hz reward-v2 使用非对称进度、
  new-best、一次性 success/failure/timeout、时间、crawl、command-rate、仿真真值 tracking 和
  封顶 `-0.04` 的 1.5 秒 gait 非劣化约束；删除绝对距离/朝向/障碍/地形门控，stall 仅监控，
  Adapter 与 UWB 不进入 PPO reward。50 Hz 只保留轻量姿态、能耗、body contact 和两个零权重
  curriculum compatibility term。
- 统计与反馈修复：worker wire 从 353 扩为
  `critic323 | response_aux30 | diagnostic_aux28 = 381`，前 30 槽保持 P1.5 Adapter 合同，追加
  四足窗口指标及 pre-step column/row/goal-distance。terminal outcome 使用 pre-step 快照，
  reset 后 row/column 只计下一 episode，首次 128 个 episode 也计入 starts；成功 term 同时兼容
  `active_terms/_term_names`。课程探针扩为 `3x10`，逐环境 reset INFO 继续按分钟聚合。
- 终止奖励边界修复：`frame_end()` 在存在待结算高层 transition 时保留旧 episode 的
  `best_goal_distance`，由 `finish_tick()` 使用 terminal-safe end distance 完成 `new_best` 结算后
  再清空，避免 terminal tick 重复发新纪录奖励。合法 worker reason `1/2/3` 统一决定
  success/failure/timeout，只有缺失或非法 reason 才回退 wrapper timeout，因此 hard 与 timeout
  mask、奖励和面板结果严格互斥。
- 训练与恢复修复：八小时 schedule 固定 0/10/20 分钟、2/6/8 小时的 CNN/Actor/Critic/
  Adapter LR 与 entropy；CNN 只在空 rollout 边界切换，旧 optimizer group 缺 `base_lr` 时按组名
  恢复。旧 P2 reward-v1 采用 `reward_v2_warm_start`，保留低层/NavigationEncoder/Actor/Adapter
  权重并重建 Critic、return statistics 和高层优化状态；reward-v2 exact resume 强制校验训练
  contract、阶段、CNN 状态、LR/entropy、return statistics、gait baseline、completed records
  和独立 RNG。首存 5 分钟，resume 后回到下一个全局 10 分钟保存边界。
- 监控修复：新增 reward 分解守恒、20 个 `vx x |wz|` 桶的 count/share、目标/执行/真值、
  tracking MAE、progress、outcome 与 gait penalty；增加三段起点条件统计、逐腿 duty/swing/air/
  frequency/slip，以及 Adapter 的 horizon/axis/baseline/coverage、分 row 和核心域/外沿域 MAE。
  当前仓库无法从公开 observation 确定机器人实时所在的物理赛道段，因此不伪造
  slope-to-stairs/maze 边界穿越率；现有 row 面板明确是 episode 起点条件统计。
- 修改文件：`server/agent_ppo/feature/p2_contract.py`、`p2_gait.py`、`p2_worker_bridge.py`、
  `p2_curriculum_probe.py`、`p2_response_buffer.py`、`p2_observation_process.py`，
  `server/agent_ppo/algorithm/algorithm_p2_nav_ppo.py`、P2 workflow/TOML/monitor/tests、
  `server/README.md`、`server/CHANGELOG.md` 和接口契约。
- 验证：宿主 `PY311test` 下 `agent_ppo/tests/test_p2_core.py` 为 `57 passed`，覆盖三段/八小时
  配置、reward 排序、非对称负进度、new-best 不可重复、gait 正常零罚与异常封顶、五个时间
  边界、rollout 边界解冻、pre-reset 归因、成功 term fallback、旧 P2 warm start、reward-v2
  exact resume、action RNG、全 rollout advantage 归一化及 Adapter 分组监控。P2/Nav/P1.5
  终止 new-best 和 success/timeout 互斥。除两个无关旧模块外的训练端扩展回归为
  `235 passed, 3 subtests passed`；Python 编译、TOML 解析和 `git diff --check` 通过。完整
  `agent_ppo/tests` 仍有旧 `test_st9_opt3_d2.py` 导入已删除 `_student_drive_probability` 的收集
  问题，以及 `test_j9_fixed_lr.py` 对当前 `AlgorithmPPO` 已删除 `_validate_fixed_lr` 的 4 个旧断言
  失败，均与本轮 P2 变更无关。真实 `99393`、小环境
  PPO/Adapter/save-resume、128 环境完整 backward、平台 monitor
  builder 和正式任务均尚未验证，不能标记为平台已修复。
- 防复发：奖励分解和必须等于 PPO storage reward；PPO reward 输入测试必须拒绝 Adapter/UWB/
  障碍/地形/朝向依赖；所有 terminal scorer 使用 reset 前快照；训练合同或 aux 维度变化必须
  同步 checkpoint/interface/test。禁止用 column 晋级、全局 reward 或平台旧
  `completed_count_track_l*` 单独证明迷宫能力。
- 血缘与回滚：当前无 commit/PR、平台 task ID、八小时模型 ID 或 checkpoint SHA256；工作区原有
  多项未提交 P2 修改，未清理或覆盖。回滚应整体恢复 reward-v1、353 wire、双段 TOML 和旧
  training contract，不能只回滚一端造成 observation/checkpoint 不兼容。最短检查路径：确认
  `run_name/track_length/wire_dim` → 查看 reward 守恒 → 核对 pre-step outcome → 检查 schedule/
  exact resume → 再看联合命令桶、三段进度、gait 与 Adapter 分组误差。
- 2026-07-29 更正：本条记录的“默认 warm-start 父包为 P2 `99393`”已被用户明确撤销；当前
  指定父包为 `p15resp8h-r1_37953-F` / `responsecalib-37953`。历史描述保留用于说明当时实现，
  新的选择和身份策略见 `BUG-20260729-002`，不得再据本条把 99393 当成默认父包。

## BUG-20260729-002：P2 模型 ID 单点门禁与在线步态基线污染

- 状态：本地已验证，开发容器、平台 smoke 与八小时训练待验证。
- 影响：分支 `codex/p2-track-nav-ppo` 的 P2 preload/eval 候选、checkpoint lineage、八小时
  gait 非劣化奖励和监控。指定父包为本地归档
  `archive/代码存档/p15resp8h-r1_37953-F.zip`，其中模型文件为
  `model.ckpt-responsecalib-37953.pkl`；归档 `kaiwu.json` 记录 train_step/model ID `37953`。
- 用户可见症状与风险：活动配置仍把 `99393` 写成默认 P2 warm start；P2 candidate 只有在平台
  请求 ID 等于配置父 ID 时才允许回退 `responsecalib`，`latest` 还会直接报错，因此平台重写、
  沿用或未正确注入 ID 时，兼容的 37953 父包也会被单点拒绝。步态保护则在 P2 前十分钟用
  正在变化的高层策略在线收集阈值，校准期间 penalty 为零，不能称为父模型基线；计算虽已上报
  逐腿 step frequency，却没有把持续快慢脚纳入 penalty。
- 根因：候选层把模型 ID 从“选择/追溯元数据”错误提升为兼容性条件；gait v1 把 CNN 冻结窗口
  等同于父基线采样窗口，没有独立的父模型离线 envelope，且 `_metrics()` 只返回 duty、swing 和
  prolonged-air 三项。
- 核心修复：`configure_app.toml` 与 P2 TOML 默认改为 `37953`。P2 loader 现在按“请求 ID P2
  优先 → 配置父包 → 同类发现候选”选择，训练和评估都允许 ID 不一致或 `latest`，并输出
  requested/selected/payload 身份告警；实际 lineage 采用 bundle 身份。ID、label、lineage、SHA
  只用于诊断，不能单点阻断；文件不存在、反序列化失败、stage/module/spec/shape 不兼容或非有限
  state 仍硬失败，避免在未加载有效策略时继续训练/评分。
- 步态修复：baseline 升级为 `p2_gait_parent_baseline_v2`，训练开始前即以版本化的
  `p15resp8h-r1_37953-F` 保守 envelope 生效，`observe()` 不再允许 Track on-policy 数据改写
  阈值；checkpoint 保存 parent label/ID/SHA、固定阈值和“非经验 P99”来源说明。新增
  `step_frequency_imbalance_hz=1.5`，与 duty/swing/prolonged-air 一起计算超限，整体仍封顶
  `-0.04/tick`；面板新增 `reward_gait_frequency_excess`。Adapter row/core/outer 样本占比也改为
  只按有效 horizon mask 统计，避免无标签 future slot 扭曲覆盖率判断。
- 审查项处理边界：P2 worker/probe 已使用 terminal 前 row/column/goal-distance，P2 自有结果为
  权威口径；平台拥有且同步脚本排除的 `server/isaac_env/base_env.py` 通用 EnvMonitor 仍只传 row
  快照，本轮不伪装成已修复，也不重新注册旧 `completed_count_track_l*` 面板。outer command 已有
  `command_core_overflow_rate`、hard-boundary rate 和 20 个联合桶，按计划只监控、不增加配额或
  表现门禁。平台默认 maze 也继续只作实验变量，不声称 L0 是简单迷宫。
- 修改文件：`server/agent_ppo/checkpoint_io.py`、`agent.py`、
  `algorithm/algorithm_p2_nav_ppo.py`、`feature/p2_contract.py`、`feature/p2_gait.py`、P2 TOML、
  `configure_app.toml`、monitor、tests、`server/README.md`、`server/CHANGELOG.md` 和接口契约。
- 本地验证：`test_p2_core.py + test_nav_stage_and_metrics.py` 为 `87 passed`；排除两个已知无关旧
  模块后的训练端回归为 `238 passed, 3 subtests passed`。Python 编译、全部 TOML 解析和
  `git diff --check` 通过。直接解包真实 `p15resp8h-r1_37953-F`，故意以请求 ID `88888`
  调用 P2 bootstrap，结果为 `bootstrap_high`，实际 `source_parent_model_id` 与
  `loaded_platform_model_id` 均为 `37953`，证明 ID 不一致未阻断且真实 schema/module 校验通过。
- 防复发：单测锁定配置父 ID 37953、跨 ID/`latest` 候选、payload 实际身份、固定 baseline 不受
  live observe 影响和纯步频失衡产生 penalty。今后新增任何 loader 时，身份检查只能改变候选
  优先级或产生告警；不得替代 stage/module/spec/shape/finite 兼容性校验，也不得成为唯一失败点。
- 血缘与回滚：当前无 commit/PR、平台 task/model ID 或新 checkpoint SHA；父归档文件 SHA 仍以
  原归档记录为准，本轮未修改模型。回滚代码可恢复旧候选和 gait v1，但会重新引入 ID 单点阻断
  与在线基线污染，不建议单独回滚。再次遇到 preload 失败时最短路径：列出 candidates/selected
  → 看结构校验首个异常 → 核对 bundle stage/spec/finite → 最后才看 requested/payload ID 告警。

## BUG-20260729-003：开发容器历史文件与测试缓存导致训练任务创建失败

- 状态：平台已验证；清理后训练任务可正常创建，无缓存容器回归已通过。
- 影响：腾讯开悟开发容器 `/data/projects/legged_robot_competition_26` 向平台创建
  `p2nav8h-r1` 训练任务时的代码快照/打包路径。不影响本地源码、父模型字节或 P2
  checkpoint loader 语义。
- 用户可见症状与原始证据：前端只提示“训练任务创建失败”。通过 Chrome Network
  直接捕获 `POST /api/v5/Competition/CreateTrainTask`，请求为 `p2nav8h-r1`、PPO、单机、
  `28800s`、父模型 ID `334205`，平台返回 HTTP `500`、`code=1102`、
  `TrainErrors.TRAIN_TASK_CREATE_FAIL`。`CheckTrainTask` 与 `GetResourceBalance` 均返回
  `code=0`；团队余额为 CPU 6 核、GPU 1 卡、并发任务 1。模型 `334205` 名为
  `p15resp8h-r1_37953-F`、状态 `success`，且已成功用于先前 `p2nav2h-r1`。
- 排除的错误方向：不是 P2 模型 ID 门禁。`CreateTrainTask` 在 learner/aisrv 容器创建、
  Python import 和 checkpoint 加载之前已失败，业务 loader 尚未执行。也不是同名任务、
  资源余额、并发限制、父模型不可用或训练时长越界。
- 根因：容器同步默认不删除远端历史文件，容器共有 188 个 sync-scope 文件，而当前
  本地合法清单为 132 个。多出 56 个历史 TOML、旧 Standard 测试、废弃模块和容器专用
  文件；同时容器测试改写了被工作区 Git 跟踪的 `__pycache__/*.pyc`，并生成
  `.pytest_cache`。平台代码快照/打包在这一污染状态下返回通用 500；清理后立即恢复任务
  创建。现有证据证明“清理整体”与恢复有因果关系，但平台未返回具体失败文件，
  因此不把某一个单独缓存文件写成已被独立证明的唯一根因。
- 修复：对远端 manifest 与本地 `local_sync_client.collect_local_files()` 同等边界做差集，
  删除 53 个确定已过期的历史配置、旧测试和废弃模块；保留容器/平台必需的
  `agent_ppo/conf/deploy.yaml`、`conf/start_tongbu.sh` 和 `isaac_env/base_env.py`。删除全部
  `.pyc/.pyo/__pycache__/.pytest_cache/.mypy_cache/.ruff_cache/*.sync-tmp`。清理后 manifest 为
  135 个文件，其中 132 个与本地合法清单一致，另外 3 个是上述容器必需文件。
  `conf/.env`、同步 Token、Cookie、模型、checkpoint 和训练日志均未删除。
- 平台验证：用户在清理后确认同一创建流程已可成功创建训练任务。这一证据将
  状态提升为“平台已验证”，但不等价于 P2 八小时训练、平台 smoke 或最终模型能力已验证。
- 容器回归：所有 pytest 均设置 `PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`。P2/Nav 定向
  回归 `87 passed`，P1.5/checkpoint 邻近回归 `108 passed`；测试后复查
  `pyc=0`、`__pycache__=0`、`.pytest_cache=0`，证明新测试方式没有重新污染平台快照。
- 防复发：容器测试统一使用 `PYTHONDONTWRITEBYTECODE=1` 与 pytest `-p no:cacheprovider`，
  不再在平台快照工作区运行会生成缓存的 `compileall`。每次正式创建任务前比对本地/远端
  manifest，只允许 3 个已知容器专用差异，并确认 cache count 为零。
- 血缘与回滚：分支 `codex/p2-track-nav-ppo`；父模型 `p15resp8h-r1_37953-F`，平台
  模型 ID `334205`。本次是容器运维清理与文档记录，不修改模型或 checkpoint；commit/PR 未建立。
  若需回滚仅能从历史版本重新同步某个已删文件，不应恢复任意缓存。再次遇到时的最短检查
  路径：`CreateTrainTask` 响应 → `CheckTrainTask/GetResourceBalance` → 父模型状态 →
  local/remote manifest 差集 → cache count → 容器 IDE record。
- 2026-07-29 复发补充：创建 `p2nav10hvyavoid` 前再次出现“训练任务创建失败”。容器根磁盘
  仅使用 19%，但真实代码挂载 `/workspace/code` 为 118 MB / 1344 files。其中
  `agent_ppo/tests/data` 遗留一个 25 MB `291713` smoke checkpoint 和 9 个约 25 MB 的上传分片，
  另有 8 组 `__pycache__`；活动代码仅在 `tmp_path` 单测中使用同名文件，不依赖这些残留。
  按用户授权删除临时 checkpoint、分片和字节码缓存，保留 `.env`、Token、平台配置、源码、
  golden data 和 `/tmp/vgpu` 平台缓存。随后对 Git 执行标准 `gc --prune=now`：1091 个、
  51.65 MiB 松散对象压缩为 1 个 45.05 MiB pack，所有可达历史和未提交工作区修改保留。
  最终 `/workspace/code` 为 65 MB / 149 files，松散对象、`__pycache__`、pytest/ruff/mypy cache、
  `.ide-sync-*` 和 `*.sync-tmp` 均为 0。当前尚未取得用户重新创建任务成功的证据，因此这次
  复发只记录为“容器已清理待平台重试”，不能把旧平台已验证结论自动套到新任务上。
- 2026-07-31 P3 复发补充：创建 `p3std8h-sim2real` 前再次出现“训练任务创建失败”。
  首轮清理删除 `/tmp/IsaacLab`、smoke 日志、sync 归档、Python/pytest 缓存和用户明确
  不再保留的 `agent_ppo/test_artifacts/p3_parent`；凭证与 `conf/.env` 保留。重试仍失败后
  发现项目根目录的 `agent_ppo/conf` 是指向 `/workspace/code` 的符号链接，根目录表面仅
  103 MB/625 files，但真实用户代码挂载仍包含 72 MB Git object database、测试和 smoke 工具。
  按用户授权删除容器内 `/workspace/code/.git`、`.vscode`、`agent_ppo/tests`、
  `agent_ppo/tools` 及所有生成缓存；本地 Git 仓库与正式训练源码不受影响。清理后
  `/workspace/code` 为约 2.2 MB/113 files，P3 `agent.py`、128-env TOML 和 `conf/.env` 均存在。
  项目根剩余约 103 MB 主要是平台自带 `kaiwudrl/tools`，不再删除，避免破坏任务启动。
  当前状态仍为“容器已清理待平台重试”；只有用户成功创建任务后才能升级验证状态。
- 2026-07-31 P3 平台验证与防复发：用户确认上述清理后训练任务已经创建成功，因此本次
  P3 复发由“待平台重试”升级为“平台已验证”。新增可复用
  `server/conf/container_training_cleanup.py`：默认 dry-run，`--apply` 清理测试制品、同步归档、
  缓存和 P3 临时文件；仅在全部容器测试完成后显式使用 `--remove-dev-files` 删除容器副本中的
  `.git`、`.vscode`、`agent_ppo/tests` 和 `agent_ppo/tools`。脚本校验 Kaiwu 根目录、限制删除范围，
  并保护 `conf/`、`conf/.env`、平台 `kaiwudrl/tools`、正式模型和训练日志。宿主安全回归覆盖
  dry-run、普通/彻底清理边界、凭证保留与错误根目录拒绝；容器 dry-run/实际执行结果见本轮
  后续验证记录。关联分支 `codex/p3-standard-joint-recovery`，commit/PR 尚未建立。
- 2026-08-01 P3 再次复发更正：用户在创建 `p3gait2h30` 时仍看到“训练任务创建失败”。
  先执行普通清理，再按已完成容器测试的边界执行 `--apply --remove-dev-files`，删除容器
  副本中的 `.git`、`agent_ppo/tests` 和 `agent_ppo/tools`，共释放 `70,861,806` bytes；其中包含
  `36,280,101` bytes 的测试父模型副本与约 `32.4 MB` Git object。`/workspace/code` 由约
  `72 MB` 降至 `2.3 MB`，磁盘使用率 `18%`、inode 使用率 `3%`，凭证、`conf/.env`、P3 训练
  TOML 和正式运行时源码均保留。项目根剩余约 `104 MB` 主要为平台固定提供的
  `kaiwudrl` 约 `73 MB` 和 `tools` 约 `30 MB`，不得为减小快照删除。
- 同次 Chrome 只读证据显示：清理后 `POST /api/v5/Competition/CreateTrainTask` 仍连续三次返回
  HTTP `500`（响应体 `92` bytes）。页面控制台另有一条前端生成 TOML 的错误：第 37 行
  `track_length =` 无值；但创建请求已实际发送至后端，因此它不能单独解释 HTTP 500。
  本次复发状态修正为“调查中”：文件过多已被排除为当前唯一根因，后续必须在重试前启用
  Network 事件捕获并读取 `CreateTrainTask` 响应正文，再区分平台快照、父模型、配置转换或
  后端临时故障。禁止继续删除 `kaiwudrl/tools/isaac_env` 等平台运行框架来试错。
- 2026-08-01 平台结果追记：用户确认 `p3gait2h30` 已成功创建并开始训练，父模型为
  `p3nav8h-r1_884257-F2`。因此本次复发从“调查中”升级为“平台已验证”。成功前实际执行的
  有效操作是删除容器副本中的 `.git`、`agent_ppo/tests`、`agent_ppo/tools` 及生成缓存；
  `agent_ppo/tests/data/model.ckpt-highslow-884257.pkl` 作为已完成容器测试的父模型副本，随
  `tests` 目录按用户授权删除。本地测试源码、本地模型归档、容器凭据、`conf/.env` 和正式运行时源码
  均未删除。由于平台 HTTP 500 没有提供可区分单个文件的根因，仍只能证明“彻底清理后重试”与
  恢复有因果关联，不把某一个 test 文件写成已被独立证明的唯一根因。平台 task ID、后续
  checkpoint/model ID 与评估结果当前未知，不做虚假记录。

## BUG-20260729-004：P2 指令联合域 line 面板超过平台 20 指标上限

- 状态：开发容器已验证，新训练任务启动日志待确认。
- 影响：`p2nav8h` learner 加载 `agent_ppo/conf/monitor_builder.py` 时的整套自定义监控。
  训练数值路径不受影响，但配置校验失败后平台会跳过所有 P2 面板，导致长训不可观测。
- 用户可见症状与原始日志：`learner_init Error occurred while loading user monitor config`，
  随后报“配置校验失败，共发现 5 个错误”，5 条均为“line 类型面板最多支持 20 个指标，
  当前 24 个”，最后 `will skip loading`。
- 根因：“指令联合域”中 5 个 `vx分桶N命令链` 同时放入四个 `|wz|` 桶的
  `target/exec/true vx` 与 `target/exec/true |wz|`，即 `4 x 6 = 24` 条曲线。仓库原测试只校验
  group/panel 名称字符合法性，没有锁定平台的单面板指标数上限。
- 修复：每个 `vx` 分桶拆成两个 line 面板：`前进链` 保留四个 `|wz|` 桶的
  target/exec/true vx，`转向链` 保留 target/exec/true `|wz|`，各 12 项。不删除、改名
  或重新聚合任何底层 metric key。`test_p2_core.py` 新增对每个 P2 panel `<=20` 的断言。
- 本地验证：`PYTHONDONTWRITEBYTECODE=1` 且禁用 pytest cache 运行
  `test_p2_core.py + test_nav_stage_and_metrics.py`，结果 `87 passed`。容器原生 builder 加载、
  配置校验和训练启动日志待验证，本条暂不标记为平台已验证。
- 容器验证：修复文件同步后，直接导入容器安装的
  `kaiwudrl.common.monitor.monitor_config_builder.MonitorConfigBuilder` 并执行完整
  `build_monitor()`，返回 `monitor_builder_ok dict`，未再出现单面板 24 指标校验错误。
  同一容器的 P2/Nav 定向回归为 `87 passed`，P1.5/checkpoint 邻近回归为
  `108 passed`。为避免重现 `BUG-20260729-003`，全部测试均禁止 bytecode 和 pytest cache，
  测试后 cache count 仍为零。
- 防复发：新增 P2 面板时必须通过单面板 `<=20` 测试；“监控指标全部存在”不等于
  “平台 builder 可接受”，正式训练前还必须在开发容器调用平台原生
  `MonitorConfigBuilder` 完整构建一次。
- 血缘与回滚：分支 `codex/p2-track-nav-ppo`，父模型 `p15resp8h-r1_37953-F` / ID
  `334205`；本轮不修改 checkpoint、奖励、观测或训练算法。回滚只需恢复五个旧面板定义，
  但会重新导致整份监控配置被平台跳过，不建议回滚。

## BUG-20260729-005：P2 自定义监控依赖空 PID 注册表且静默丢弃

- 日期：2026-07-29；状态：容器已验证，待新平台任务验证。
- 影响：任务 `p2nav8h-r1` / task `235036` 的 P2 自定义监控。训练、梯度、checkpoint 和
  EnvMonitor 原生 reward 不受影响，但损失、reward-v2 分解、命令链、Adapter、课程诊断和
  性能资源面板没有时序数据。
- 用户可见症状：监控页中只有“运动Reward”有数据，其余大量 P2 面板显示“暂无数据”；训练日志
  同时正常出现 `[P2NavPPO] iter=...`，且 EnvMonitor 每分钟报告已上报 321 个环境指标。
- 根因：P2 `_monitor_put()` 使用
  `monitor.put_data({pid: metrics for pid in monitor.get_pids()})`。该注册 PID 列表在当前 aisrv
  workflow 中为空或接口不可用时，不会提交当前进程产生的 P2 metrics；异常又被裸
  `except Exception: pass` 静默吞掉。通用 workflow 的已验证写法是
  `monitor.put_data({os.getpid(): monitor_data})`。运动 Reward 由独立的 learner EnvMonitor
  上报，因此形成“只有运动 Reward 有数据”的特征。
- 修复：P2 上报改用当前进程 `os.getpid()`，不再读取 monitor 注册 PID；函数返回成功状态。
  上报失败仍不阻断八小时训练，但在原有约一分钟调用周期输出异常类型与原因，禁止静默失败。
- 回归防线：新增测试验证 `get_pids()` 即使抛错也不会被调用，payload 使用当前 PID；另验证
  `put_data()` 异常会产生 warning 且返回失败，不向训练路径传播。
- 验证：宿主 `PY311test` 定向回归 `62 passed`；同步 bundle 校验成功，二次 dry-run 为
  `files to overwrite: 0`；开发容器 `env_isaaclab` 定向回归 `62 passed`。现有 task `235036`
  已丢失的历史 P2 时序无法补写，且运行中 Python 不会热加载同步后的 workflow；平台验证必须
  在重启新任务后确认至少 `actor_loss`、`reward_positive_progress`、`target_vx` 和
  `adapter_loss` 出现数据。
- 血缘：分支 `codex/p2-track-nav-ppo`；父模型 `p15resp8h-r1_37953-F` / ID `334205`；
  checkpoint、reward、observation 和部署契约均未修改。回滚为恢复旧 `_monitor_put()`，会重新
  造成 P2 自定义面板无数据，不建议回滚。

## BUG-20260729-006：完整赛道前缀进度掩盖迷宫卡墙且缺少可学习脱困信用

- 日期：2026-07-29；状态：代码已修复待容器与平台验证。
- 影响：`p2nav8h` reward、worker transport、checkpoint exact-resume contract 和 P2 自定义
  面板；不修改低层、Actor action 维度、Adapter 输入前 30 槽或部署 runtime。
- 用户可见症状：机器人反复通过逆坡和下台阶取得较高 positive-progress/new-best，进入迷宫后
  卡墙并持续低分，episode reset 后曲线再次跃升。抓取 `20260729-043111` 中 success 全程为零，
  progress/new-best 已占真实 reward 绝对量约 64%，说明继续放大进度只会强化容易的赛道前缀，
  不能给“碰墙前转向”和“卡墙后成功绕出”提供清晰信用。
- 根因：reward-v2 对负进度已做非对称降权，但 50 Hz body contact 混在 frame safety 中且无法在
  高层 tick 单独归因；stall 仅监控。高层能够知道卡住最终较差，却没有 collision onset、持续
  frontier 停滞或真正突破历史最远位置的独立反馈。速度突降同时会由台阶、坡面、Slew 和主动
  减速触发，不能作为可靠撞墙标签。
- 修复：reward contract 升级为 `p2_track_reward_v3`。worker 从 contact sensor 排除四足后，
  运输最近 10 个低层帧的最大非足端接触力；5 Hz collision 首次按严重度处罚 `-0.08~-0.20`，
  持续贴墙为 `-0.03`，terminal/failure 不重复罚。stagnation 使用 15 tick 前的单调
  best-distance frontier，3 秒没有至少 `0.03m` 新进展才从 `-0.015` 递增，封顶 `-0.06`，
  不按命令、地形、墙体方向或 Adapter 门控。recovery 只在已确认停滞后单 tick 将 frontier
  推进至少 `0.08m` 时发 `+0.15`，5 秒冷却、每 episode 最多两次；reset/terminal 禁止发放。
- 双罚与刷分防线：TOML 中 50 Hz `undesired_contacts` 改为零权重，仅保留平台监控；碰撞每个
  高层 tick 只结算一次。frontier 单调不允许后退再前进重复刷新，低命令不能逃逸 stagnation；
  同位移测试锁定“连续推进回报高于故意等待后 recovery”。速度改变量只允许作为诊断。
- 接口与恢复：diagnostic aux 从 28 增至 29，wire 从 381 增至 382；新增 aux58 为 0.2 秒非足端
  最大接触力，response aux30、gait30:55 和 pre-step55:58 保持原语义。reward/training contract
  升级至 v3，因此旧 reward-v1/v2 P2 包必须走已有显式 warm-start，不能伪装 exact resume；
  P1.5 父包 `p15resp8h-r1_37953-F` 不受 checkpoint ID 硬门禁。
- 监控：P2 自定义“奖励贡献”新增“避障与脱困”，报告 collision/stagnation/recovery 三项真实
  加权贡献；“卡滞与终止”同步展示三项。前端空的“导航Reward(5)”经代码检索确认不属于用户
  `agent_ppo/conf/monitor_builder.py`，而是平台 Track 基础组；本轮不修改平台拥有的
  `server/isaac_env/base_env.py`，容器 builder/平台页面验证后再确认是否可由用户配置隐藏。
- 修改文件：`p2_contract.py`、`p2_gait.py`、`p2_worker_bridge.py`、
  `algorithm_p2_nav_ppo.py`、P2 TOML、monitor、tests、README、CHANGELOG 和 server-deploy contract。
- 验证：宿主与开发容器 `env_isaaclab` 的 P2 核心/监控定向回归均为
  `93 passed`；同步后二次 manifest 为 `files to overwrite: 0`。容器原生 builder 实际
  构建 10 个 P2 自定义组，line panel 最多 12 个指标，且
  `legacy_nav_reward_present=False`；容器真实 contact sensor 运行时实测、平台
  reward 守恒和面板待验证。验收必须确认三个新项长期合计绝对贡献约 `3%-8%`，且连续推进
  高于停顿刷 recovery、collision 不统计足端、terminal 不双罚。
- 血缘与回滚：分支 `codex/p2-track-nav-ppo`，父模型 `p15resp8h-r1_37953-F` / ID `334205`。
  当前无 commit/PR/新 checkpoint SHA。回滚必须原子恢复 reward/training contract、wire dim、
  worker aux、TOML 和 checkpoint 恢复边界，禁止只回滚单项造成 381/382 transport 错配。

## BUG-20260729-007：P2 评估内部成功 reset 未回传到 Track scorer

- 日期：2026-07-29；状态：本地已验证，待平台重新评估。
- 影响：`p2nav8h-r2_291713` 的 Track/Camera 评估完成数、single-life mask、结束时机和最终
  分数。模型前向、训练 reward、checkpoint 模块和模型 ID 选择均不受影响。
- 用户可见症状：模型多次到达终点后被 Isaac Lab auto-reset 直接传送回起点，但评估没有把该
  环境标记完成；异常日志 `/Users/nanbloom001/Downloads/log-598805-18532703.zip` 最终
  `completed=0`、`completion_coeff=0`、`total_score=0`。P2 worker curriculum 汇总同期累计
  21 次 success、3 次 failure，而 BaseEnv 正式 term-debug 只收到 3 次
  `bad_orientation` 和 4 次 `time_out`，single-life 仅到 `7/16`。
- 动态基准：已知正常包 `archive/代码存档/复赛_track` 的评估日志
  `/Users/nanbloom001/Downloads/log-598816-18533115.zip` 中，16 个环境均通过公开
  `term='goal_reached'` 进入 single-life，frame 2374 达到 `16/16`，最终
  `episode_count=16`、`completion_coeff=1.0`、`total_score=73.87`、`completed=16`。
  两份日志证明 scorer 后段没有丢分；异常发生在 success reset 到公开 done 的回传边界。
- 静态对比：正常基准、活动树和模型包内 BaseEnv 的
  `dones -> newly_done -> EnvMonitor.on_step -> all_done` 主逻辑一致。模型目录
  `archive/代码存档/p2nav8h-r2_291713` 中 `p2_worker_bridge.py`、
  `p2_observation_process.py` 和 `isaac_env/base_env.py` 与修复前活动代码逐文件一致。
  P2 独有 worker aux24/25 已保留 reset 及 success/failure/timeout，但只被 P2
  rollout/recurrent 路径消费，未合并回 Gymnasium step 返回值，BaseScorer 因而完全看不到
  这些 success。
- 排除方向：不是 0.6m 阈值过严——worker 在真实 auto-reset 帧已读取
  `goal_reached=True`；周期性 `[goal_term] reached=0/16` 只是非终止时刻采样。不是模型 ID
  门禁或 checkpoint 未加载，评估已进入 `p2_nav_eval` 并完成确定性前向。也不是
  `make_json_and_done_file` 后处理漏分，因为异常 JSON 在生成时已是
  `episode_count=7/completion_coeff=0`。
- 根因：当前平台 wrapper 在部分 P2 success auto-reset 行没有向外保留 done；正常 Track
  scorer 只消费 wrapper dones。我们虽然已为 P2 训练在 aux24/25 建立权威终止恢复，却没有在
  评估环境返回边界把该信号送回平台既有 scorer，形成“两套终止语义”。
- 修复：`P2PolicyObservationProcess` 和 `P2CriticObservationProcess` 初始化时，对当前 P2
  内层环境安装一次幂等 terminal-return adapter。原始 Gymnasium step 完成、worker aux 已更新
  后，将 `reset && reason in {success,failure}` OR 到 `terminated`，将
  `reset && reason==timeout` OR 到 `truncated`。原生 flags 只增不减；reason=0 的首次 reset、
  缺失/短 aux 和非 P2 环境保持原行为。随后继续走正常复赛 Track 已验证的
  RSL wrapper、BaseEnv single-life 和 BaseScorer，不修改受同步保护的
  `server/isaac_env/base_env.py`，不创建第二套 scorer，也没有任何模型 ID 硬门禁。
- 修改文件：`server/agent_ppo/feature/p2_worker_bridge.py`、
  `server/agent_ppo/feature/p2_observation_process.py`、
  `server/agent_ppo/tools/p2_terminal_smoke.py`、
  `server/agent_ppo/tests/test_p2_core.py`、`server/CHANGELOG.md` 与本台账。
- 回归防线：覆盖 success/failure/timeout 三种恢复、原生 done 保留、初始 reset 忽略、安装
  幂等及无模型 ID 依赖。宿主 `PY311test` 执行 P2 定向测试 `71 passed`，Python 编译与
  `git diff --check` 通过。容器 Isaac 原生 step-return、term-debug 出现
  `goal_reached`、single-life 达到 `16/16` 和平台非零 completion 仍待验证，不能提前升级为
  平台已验证。
- 血缘：分支 `codex/p2-track-nav-ppo`；异常模型
  `p2nav8h-r2_291713` / `model.ckpt-navfull-291713.pkl`；正常动态基准为
  `archive/代码存档/复赛_track`。原 checkpoint SHA256 为
  `79cf02e890bdf032dcec6fdc259426e047b8e5ca7bb41521c1b7ad692da71a2a`。已将相同补丁写入冻结
  目录并生成 `archive/代码存档/p2nav8h-r2_291713-evalfix.zip`，ZIP SHA256 为
  `ee80b6bd9728d827f17e22720efe1dd9803b49270d7585d19d6da1ab606caf3b`；重新解包后 P2 定向测试
  `71 passed`，且包内 checkpoint SHA 与原权重完全一致。ZIP 已排除 `.env`、同步脚本、缓存、
  bytecode 和 `.nfs*`。当前无新 commit/PR。Arena 评估使用模型包中冻结的代码，因此旧
  `291713` 不会因活动树修改自动变化，应上传该修复 ZIP 重新评估。回滚为移除 P2
  terminal-return adapter；这会恢复 worker success 可见但 scorer `completed=0` 的缺陷，
  不建议回滚。
- 2026-07-29 更正：修复包平台评估 `/Users/nanbloom001/Downloads/log-598827-18533594.zip`
  仍为 `completed=0`、`completion_coeff=0`、`total_score=0`。worker 在 600 秒累计
  `25 success / 7 failure`，BaseEnv 正式 single-life 只收到 7 个 `bad_orientation` 和
  2 个 `time_out`，没有任何 `goal_reached`；因此首版修复的“本地已验证”仅证明纯函数和
  手工绑定 FakeEnv，不证明平台生命周期，状态退回“代码修复中”。容器源码确认
  `register_observation_processes()` 无参数构造 process，而 `ObservationBridge.wrapper(env)`
  只在运行期调用 `process()` 前绑定真实 env；首版在 `__init__` 调用安装时 `env=None`，
  `install_p2_terminal_return_bridge()` 静默返回 False，随后不再重试。第二版将幂等安装移到
  policy/critic `process()` 第一行，使 RSL wrapper 初始化 reset 的 observation 计算期间即
  包装真实 ManagerBasedRLEnv、早于首次正式 step；增加一次安装日志与最多八次 worker/native/
  merged 计数日志。模型 ID 仍只作告警，不参与安装或终止判断。第二版宿主、真实容器和平台
  证据应在下方继续追加，未得到 scorer 非零完成前不得升级为平台已验证。
- 2026-07-29 容器验证：同步后容器文件 SHA256 与本地一致，`env_isaaclab` 执行
  `agent_ppo/tests/test_p2_core.py` 为 `72 passed`。新增的轻量真实 Isaac smoke 只将测试时
  Track 缩为 `1 env × 1 column`，仍使用生产三段顺序、真实
  `Unitree-Go2-Velocity-Camera`、真实 `ManagerBasedRLEnv`、RSL wrapper、平台 BaseEnv
  single-life 与 BaseScorer；它不加载或检查任何模型 ID，也不修改生产 TOML。强制把 env0
  放到平台生成的 Track goal 后，日志得到
  `worker_success=1/native_hard=0/merged_hard=1`，随后正式
  `term='goal_reached'`、`public_terminated=True`、`public_truncated=False`、
  `single_life_done=True`、`completed=1`，BaseScorer 生成
  `episode_count=1/total_score=99.99`，工具最终打印 `[P2TerminalSmoke] PASS`。这证明第二版
  adapter 在真实平台生命周期中的安装时机与 success 回传闭环均正确；尚未使用修复后的完整
  `p2nav8h-r2_291713` 模型重新提交 Arena 评估，因此状态只能是“本地已验证”，不能写成
  “平台已验证”或“评估已验证”。测试结束后已仅清理 smoke 残留 Isaac 进程，RPC 同步服务与
  凭据未改动。
- 2026-07-29 冻结包同步：将上述第二版 `process()` 安装时机修复及 `72 passed` 回归同步到
  `archive/代码存档/p2nav8h-r2_291713`，新建
  `archive/代码存档/p2nav8h-r2_291713-evalfix-v2.zip`，ZIP SHA256 为
  `ff436b2364042f97119e4be926b9672343b690bdfc5a4557ffca4170fc70656b`。ZIP 完整性检查通过，
  包内 checkpoint SHA256 仍为
  `79cf02e890bdf032dcec6fdc259426e047b8e5ca7bb41521c1b7ad692da71a2a`，与原权重逐字节一致；
  `.env`、`tongbu.py`、`start_tongbu.sh`、缓存、bytecode 和 `.nfs*` 均未打包。旧
  `p2nav8h-r2_291713-evalfix.zip` 是已被 `598827` 证伪的首版，不得再上传评估。

## BUG-20260729-009：P2 非有限奖励污染 rollout 且 reset 步态窗口分母错位

- 日期：2026-07-29；状态：开发容器已验证，平台训练短跑待验证。
- 影响：分支 `codex/p2-track-nav-ppo` 的 5 Hz P2 reward/GAE/return statistics，以及
  auto-reset 后约 1.5 秒四足 gait 面板和非劣化 reward。低层、Actor85、ResponseAdapter 输入、
  Track 赛段契约和父包 `p15resp8h-r1_37953-F` 不变。
- 发现方式与症状：未提交变更审查发现 `finish_tick()` 只清洗 bootstrap value 与 tracking，
  `start/end_goal_distance`、frame safety、gait/collision aux、duration、terminal reason 和
  command penalty 可未经有限值检查进入 reward stack。一个 NaN/Inf 会写入 rollout，并可能把
  `best_goal_distance`、GAE 和 return statistics 变为 NaN。另发现 `P2GaitWindowProbe.step()`
  先清空 reset 环境，再把 reset 边界样本写入 ring，最后将 `env_counts` 设为零；下一帧 ring
  已含两帧而分母只有一帧，duty factor 可超过 1。尚无平台日志证明这两项已在长训触发，因此
  平台影响仍标记为待验证。
- 根因：reward settlement 缺少统一的逐环境 finite boundary；已有 `rollout_invalid` 只覆盖
  Actor/critic 部分输出，而且 `update()` 在跳过无效 rollout 前仍会计算 returns。步态窗口则把
  “reset 帧输出应为零”和“reset 帧是否计入历史”混成两套不一致处理。
- 核心修复：逐环境验证 bootstrap、goal、duration、terminal reason、frame safety、command 和
  实际参与 reward 的 gait/collision 输入；异常行的所有 reward component 统一归零，duration
  收敛到合法 `1..10`，保持旧 best distance，清空该环境 frontier/collision 状态，同时置
  `rollout_invalid` 并按环境只累计一次 `invalid_transition_count`。写入 storage 的 transition
  保持有限，因此即使整轮随后跳过，也不会先污染 GAE/return statistics。gait reset 行现在清空
  当前 ring slot、保持 `env_counts=0`，只有非 reset 行才递增分母。
- checkpoint 审查结论更正：审查一度建议“首个存在但结构不兼容时继续尝试后续候选”，这与既有
  安全边界冲突，未实施。模型 ID、payload ID 和 lineage 不一致继续只 warning；候选回退仅处理
  高优先级文件不存在。一旦选中文件，反序列化失败、必需模块缺失、state-dict key/shape 或输入
  契约不兼容仍 hard stop，禁止把损坏的 exact resume 静默替换成其他父包。相关 docstring 和
  loader 注释已明确，该边界不构成模型 ID 单点门禁。
- 修改文件：`algorithm_p2_nav_ppo.py`、`p2_gait.py`、`checkpoint_io.py`、`agent.py`、
  `test_p2_core.py`、`server/CHANGELOG.md` 和本台账。
- 本地验证：`PY311test` 下 `agent_ppo/tests/test_p2_core.py` 为 `79 passed`，新增多重非有限
  reward 输入只计一个 invalid row、storage reward/duration 有限、best distance 不污染，以及
  reset ring 与 denominator 对齐测试。模型 ID 非单点门禁测试继续覆盖“平台请求 88888 时仍包含
  配置父包 37953”；P2/Nav/P1.5 邻近回归共 `152 passed`，相关 Python 编译、12 份 TOML 解析和
  `git diff --check` 均通过。
- 容器/平台/评估：2026-07-29 已同步开发容器；`env_isaaclab` 中 P2 核心测试为
  `79 passed`。以 `p2nav8h-r2_291713-evalfix-v2.zip` 内
  `model.ckpt-navfull-291713.pkl`（checkpoint SHA256
  `79cf02e890bdf032dcec6fdc259426e047b8e5ca7bb41521c1b7ad692da71a2a`）执行真实
  reward-v2 warm start 续训 smoke：NavigationEncoder/Actor/Adapter 权重保持，Critic 与 return
  statistics 重建，完成 8 次 Actor、8 次 Critic gradient step 和 1 次 Adapter update，保存后
  exact resume 返回 `exact_resume_history_reset`。真实 1-env Track+Camera smoke 同时通过
  `goal_reached -> terminated -> completed=1 -> score=99.99`。尚未运行 128-env 完整
  rollout/backward 或平台训练任务，因此不得升级为“平台已验证”。
- 防复发、回滚和最短检查路径：reward 测试必须同时注入 goal NaN、safety Inf、gait NaN、
  collision Inf 和非法 duration，并断言 storage 全有限、invalid count 按行而非按字段计数。
  gait 测试必须断言 reset 后 ring sum/count 均为零、下一帧 duty 不超过 1。回滚时可独立恢复
  本条两个运行时修复及测试；不得移除结构兼容 hard stop。再次遇到训练无更新时先查
  `ppo_rollout_skipped_nonfinite/invalid_transition_count`，再检查 reward 分解与 gait duty 范围。
- 血缘：本次未生成 checkpoint/model ID/SHA256、commit 或 PR；工作区保留用户已有未提交 P2
  修改。

## BUG-20260729-008：P2 混用 ContactSensor 局部索引且把出生 row 当实时赛段

- 日期：2026-07-29；状态：开发容器已验证，平台训练短跑待验证。
- 影响：分支 `codex/p2-track-nav-ppo` 的四足 gait 面板、gait 非劣化 reward、5 Hz
  body-collision reward、三段 Track 面板及 Adapter 分赛段 MAE。ResponseAdapter 前 30 槽、
  Actor85、低层模型和评估 terminal bridge 不变；不修改平台拥有且同步排除的
  `server/isaac_env/base_env.py`，不引入模型 ID 或 checkpoint label 硬门禁。
- 用户可见症状与证据：八小时抓取 `shared/arena_frontend_monitor/runtime/manual_metric_recorder/
  sessions/p2nav8h` 中 FL duty 约 `0.69`，FR/RL/RR 约 `0.0012` 且后三条逐点相同；后三足
  prolonged-air 约 `98%-99%`、最长 air time 达数十秒，但同期 `true_vx` 约 `0.36m/s`。
  `reward_body_collision` 几乎恒定为 `-0.0302`，恰好接近 persistent-contact 档位，而平台
  `reward_undesired_contacts` 为零。三段面板同时长期显示
  `slope_inv_sample_share=1/stairs=0/maze=0`，与 P2 自有累计 `2788` 次 success 和整赛道评估
  能力矛盾。旧抓取没有逐帧 world X，无法事后恢复真实分段访问次数，必须修复后重新采集。
- 根因：`P2GaitWindowProbe._resolve()` 从 `robot.find_bodies(".*_foot")` 取得 Articulation
  全局 body ID，却直接索引 ContactSensor 的 `current_air_time/last_air_time` 局部 body 列；
  并无条件选取首个带 air-time 的 sensor。`_body_collision_force()` 又复用同一错误 ID 排除
  `net_forces_w` 足端列。平台镜像确认 ContactSensor 张量第二维按 sensor 自身
  `body_names` 排列，Go2 的目标 sensor 名为 `contact_forces`。另一方面 workflow 把
  `terrain_levels` 当 live row；Track 中它只是 episode 出生段，机器人沿 X 进入楼梯/迷宫时
  不更新。Adapter record 和分组也沿用了该错误字段。
- 对历史记录的更正：`BUG-20260729-006` 曾写成“worker 从 contact sensor 排除四足”，当时只
  验证了碰撞公式和 ID 恰好一致的 fake tensor，没有验证真实 sensor-local 映射；该结论被本轮
  数据与平台 API 证伪。`BUG-20260729-001` 的“公开 observation 无法确定实时物理段”也不再作为
  阻碍：本地平台镜像提供了 `TerrainExitManager.get_track_segment_index()` 的明确边界公式，
  可由 `root_pos_w.x`、`track_length` 和 terrain `size_x` 等价计算。
- 核心修复：固定选择 `scene.sensors["contact_forces"]`，按 FL/FR/RL/RR 精确名称分别建立
  `robot_foot_ids` 与 `sensor_foot_ids`；foot velocity 只用前者，air/contact/force 只用后者。
  启动打印 sensor 名称、四足名称、两套 ID、body 数量和 tensor shape。名称不唯一、足端不齐、
  tensor shape 漂移或 full-body sensor 缺失时，gait/body-collision 映射有效位置零、两项 reward
  立即归零并 warning，训练继续。碰撞失效时同步清空 0.2 秒 force history，避免残留九帧继续扣分。
- 实时赛段修复：worker 按平台 Track 居中边界
  `offset_x=-size_x*track_length/2` 和 `root_pos_w.x` 计算 `current_segment=0/1/2`，非 Track、
  配置缺失或非有限位置使用 `-1`；auto-reset 行用前一帧 segment 形成 terminal-safe 快照。
  `terrain_levels` 继续只作 `spawn_row`，`terrain_types` 继续只作 `difficulty_column`。workflow
  的 slope/stairs/maze progress、速度、结果和 gait 改按 current segment 聚合；Adapter completed
  Track records 保存 current segment，P1.5 父 records 和旧 records 统一标为 `-1`，不再混入三段
  MAE 分母。
- 接口与恢复：worker transport 从
  `critic323 | response_aux30 | diagnostic_aux29 = 382` 升至
  `critic323 | response_aux30 | diagnostic_aux32 = 385`；新增 aux59 current segment、aux60
  gait mapping valid、aux61 collision mapping valid，前 59 槽语义保持。reward/training contract
  升为 v4，旧 P2 包继续走已有显式 warm-start，不能伪装 exact resume；父包
  `p15resp8h-r1_37953-F` 的选择仍为 warning-only 身份策略。
- 修改文件：`p2_gait.py`、`p2_worker_bridge.py`、`p2_contract.py`、
  `p2_response_buffer.py`、`response_aux_buffer.py`、`algorithm_p2_nav_ppo.py`、
  `p2_nav_ppo_workflow.py`、`p2_curriculum_probe.py`、`p2_observation_process.py`、monitor、
  P2 tests、`server/README.md`、`server/CHANGELOG.md` 和 server-deploy contract。
- 本地验证：`PY311test` 下 `agent_ppo/tests/test_p2_core.py` 为 `77 passed`，覆盖非连续
  Articulation ID、乱序 sensor-local 足端列、缺足 fail-safe、collision history 立即清空、
  Track X 边界、terminal 后新 episode 位置隔离和 Adapter current-segment 分组；P2/Nav/P1.5
  邻近回归共 `150 passed`，Python 编译、全部 TOML 解析和 `git diff --check` 通过。真实
  `contact_forces` shape、开发容器 Track 短跑和平台面板仍待执行，因此不能标记平台已修复。
- 开发容器验证（2026-07-29）：真实 1-env Track+Camera 启动打印
  `sensor=contact_forces`、`robot_foot_ids=[27,28,29,30]`、
  `sensor_foot_ids=[4,11,20,27]`、`current_air_time_shape=(1,31)`，
  `gait_valid=True/collision_valid=True`，证明两套索引没有再次混用。worker 以 wire385 启动并
  打印 `track_length=3,size_x=8.0,boundaries=[-12,-4,4,12]` 与有效 initial segment histogram。
  同一 smoke 的终点闭环得到 `completed=1/score=99.99`。该单步 smoke 尚未覆盖机器人依次穿越
  三段后的面板曲线，也未覆盖 128-env 长窗口 gait 分布，因此平台训练短跑和重新抓取仍是必要
  验收项。
- 防复发与验收：fake sensor 测试必须让 robot IDs 与 sensor columns 明确不相等；启动日志必须
  显示四足名称唯一且两套 ID/shape 自洽。短跑验收要求四腿曲线不再出现三腿逐点相同或数十秒
  air time，collision 应以稀疏事件出现而非固定 `-0.03`；三段 sample share 总和接近 1 且机器人
  前进时依次出现 0/1/2，Adapter 父 records 不进入分段分母。旧
  `completed_count_track_l*` 和通用 EnvMonitor 分档仍非 P2 权威口径。
- 血缘、遗留风险与回滚：本次未生成 checkpoint/model ID/SHA、commit 或 PR，工作区保留用户原有
  未提交 P2 修改。平台 `base_env.py` 仍未传 `pre_step_terrain_types`，只影响旧通用分档面板，
  不覆盖 P2 自有 pre-reset column 统计。回滚必须原子恢复 reward/training contract、worker wire、
  observation shape、buffer metadata 和 monitor，禁止只回滚一端。再次遇到相同症状时最短路径：
  查 `[P2GaitProbe] contact_mapping` → 核对 aux60/61 → 查看 collision 稀疏性 → 核对
  current_segment validity/share → 最后再调 gait/collision 权重。

## BUG-20260729-010：P2 二维父策略扩展三轴时的动作墙钟与跨设备 RNG 单点阻断

- 日期：2026-07-29；状态：本地已验证，开发容器与平台 smoke 待验证。
- 影响：分支 `codex/p2-track-nav-ppo` 的追加十小时 P2 高层 PPO。父 checkpoint 为
  `archive/代码存档/p2nav8h-r2_291713/ckpt/model.ckpt-navfull-291713.pkl`，SHA256 沿用既有
  `79cf02e890bdf032dcec6fdc259426e047b8e5ca7bb41521c1b7ad692da71a2a`；冻结低层仍保留
  37953/F2 lineage，`p15resp8h-r1_37953-F` 仅作为 gait baseline 来源。
- 症状与计划更正：初版 command-v2 实现按 20 分钟、90 分钟和 2 小时逐级开放 `vy`，并把
  `vy_expansion_elapsed_seconds` 写入 exact-resume 合同。用户明确取消该动作范围墙钟，要求从
  首轮直接允许横移探索。真实 `291713` 本地加载还发现其 CUDA `action_rng_state` 为 16-byte
  generator state；CPU smoke 的 generator 需要 5056-byte state，直接 `set_state()` 会抛出
  `Expected a CPUGeneratorImplState ...` 并阻断本来结构兼容的 warm start。
- 根因：动作能力边界与 optimizer 学习率课程被错误绑定为同一时钟；RNG 恢复又把设备实现格式
  当成模型结构契约。另审查发现 exact resume 未保存驱动 5Hz cadence 的 `frame_count`，恢复后会
  从 frame 0 重新对齐，不能称为精确续训。
- 核心修复：删除独立 vy action curriculum/clock，command-v2 从首个 rollout 固定使用可信核心
  `|vy|<=0.20`、探索硬边界 `|vy|<=0.40`；20/90 分钟阶段只控制新增 head 与共享网络 LR/entropy，
  不裁剪动作域。保留独立 main/vy sampling RNG；warm start 在 generator state 同设备兼容时恢复，
  不兼容时保留配置种子并在 `optimizer_migration_report.rng` 记录 `fresh_seed` 原因，不能因模型 ID
  或 RNG 设备格式单点拒绝。exact resume 仍严格恢复同版本 RNG，并新增 `frame_count` 保存/校验。
  旧二维 Actor 的 mean/log-std、LSTM 与 CNN 按 shape 校验迁移；新增 vy mean 为零、log-std=-0.7；
  Critic、return/value statistics 重建。
- Reward/控制/监控：crawl、command-rate 和仿真真值 tracking 使用三轴尺度
  `[1.25,0.40,1.0]`；vy/wz 减速与反向使用快速 release-to-zero，禁止越零 overshoot。面板新增
  vy 使用率、三轴链路、联合桶、Adapter vy specialty/joint outer 与碰撞事件窗口；每个 line 面板
  继续不超过 20 指标。碰撞力同时报告 rollout mean/max，recovery bonus 已删除，避免奖励撞墙后
  恢复这一可刷取过程。
- 本地验证：`PY311test` P2 核心回归在取消动作墙钟后为 `83 passed`；排除仓库已知失效的
  J9/ST9 两个旧测试后，训练端完整回归为 `261 passed, 3 subtests passed`。真实 `291713` 包的
  command-v2 smoke 继承 lifetime `28761.0402s`，起始 limits 为 `(0.20,0.40)`，迁移 11 个旧
  Actor tensor 与 19 份 Adam state，完成 8 次 Actor、8 次 Critic gradient step 和 1 次 Adapter
  update，重新保存后 exact resume 成功。CPU smoke 对源 CUDA main/neutral RNG 明确记录
  fresh-seed 降级，Adapter/shuffle RNG 正常恢复；未伪造为 exact RNG 保持。Python 编译、12 份
  TOML、monitor AST 与 `git diff --check` 均通过。
- 修改文件：`p2_contract.py`、`p2_high_level.py`、`p2_command_controller.py`、
  `algorithm_p2_nav_ppo.py`、`p2_nav_ppo_workflow.py`、P2 TOML、monitor、checkpoint candidate、
  `test_p2_core.py`、`server/README.md`、server-deploy contract、部署制品边界、CHANGELOG 和本台账。
- 容器/平台/评估：尚未同步本轮三轴代码，尚未完成真实 Isaac 128-env rollout/backward、平台
  15-30 分钟 smoke 或新 checkpoint 评估；本地 RPC dry-run 返回
  `ApiUserErrors.WEBIDE_RECORD_NOT_FOUND`，确认 IDE 18005 记录已失效，因此不得标记开发容器/
  平台/评估已验证。
- 防复发与回滚：测试必须验证首轮 `vy` hard limit 已为 0.40、二维输出迁移前后逐字相同、新 head
  零初始化、主/vy RNG 相互独立、跨设备 RNG 只降级不阻断、frame_count exact resume 和反向 slew
  不越零。回滚必须原子恢复 action/spec/checkpoint/reward/monitor，不得只把 evaluator 改回二维或
  静默插入 `vy=0`。再次遇到 warm-start 失败时先看 module/spec/shape，再看 migration report，禁止
  首先增加 checkpoint/model ID 白名单。

## BUG-20260729-011：P2 只有接触后碰撞信用，缺少部署可得的提前避障信号

- 日期：2026-07-29；状态：本地已验证，开发容器与平台 smoke 待验证。
- 影响：`p2nav10hvy` 的 5Hz 高层 reward、reward contract/checkpoint digest 和监控。低层、
  Actor/Adapter 输入维度、三轴动作边界、Track 生成和 worker wire 不变。
- 症状与目标：现有 body collision 只能在接触后处罚，frontier stagnation 还要等待约 3 秒无新
  纪录，因此视觉策略缺少“尚未撞墙但当前速度已来不及制动”的稠密信用。目标是让模型更早改变
  路径，同时避免原始距离墙奖励诱导居中慢走、拒绝窄通道或停车刷分。
- 奖励全量审查：P2 活跃 50Hz 项只有 `flat_orientation=-0.05` 与 `energy=-5e-6`，tracking/contact
  是零权重课程兼容项；5Hz 不含绝对 goal distance、heading、速度幅值或 recovery 正奖励。
  未发现 50Hz/5Hz 重复结算、terminal/reset 双算、跨 episode best-distance 污染或撞墙恢复刷分。
  保留的观察风险是：零命令前三秒只承担 time cost，终点 impulse 相对 dense shaping 较大，以及
  `wz/vy` 的变化/跟踪成本可能让早期策略偏保守；这些均无正收益漏洞，先通过面板观察，不继续
  叠加奖励或门控。
- 核心修复：新增 `predictive_collision_risk`，每个 5Hz tick 从同一份部署可得归一化深度的
  ROI `[y30:105,x96:224]` 取 10% 稳健低分位净空。深度零值按 D435i 无效/超量程处理为 5m；
  只用正向 `vx` 计算制动距离 `0.35 + vx^2/(2*0.60)`，仅当其超过净空时按 0.45m risk band
  的平方风险处罚，单 tick 下限 `-0.04`。全量审查时发现若把 `vy/wz` 也计入中央前向 ROI 的
  制动速度，会错误处罚墙前横移和原地转向，因此最终实现明确让纯 `vy/wz` 风险为零。零命令
  风险严格为零，不奖励 clearance 增大，不使用 Adapter/UWB、terrain、heading 或接触状态，
  因此不能靠停车、摆动或深度空洞领取正奖励。
- 权重判断：该项最大 `-0.04/tick`，低于首次 body collision 最大约 `-0.20`、frontier cap
  `-0.06` 和正进度单 tick 最大 `+2.0`，用于提前引导而不取代任务进度。终点/失败/timeout
  impulse 保持 `+50/-25/-15`，未因本次避障项调整。
- 监控：新增全局 clearance/stopping-distance/risk/reward 面板，并按 slope_inv/stairs_inv/maze
  分段记录净空、风险和 penalty，专门检查下楼梯或坡面是否因相机俯视产生误触发。reward
  decomposition 继续要求与 PPO storage reward 守恒。
- 修改文件：`p2_contract.py`、`algorithm_p2_nav_ppo.py`、`p2_nav_ppo_workflow.py`、
  `monitor_builder.py`、`test_p2_core.py`、`server/README.md`、server-deploy contract、CHANGELOG
  和本台账。
- 本地验证：P2 定向测试 `86 passed`；排除已知失效旧 J9/ST9 用例后的邻近完整回归为
  `264 passed, 3 subtests passed`；Python 编译、全部训练 TOML 解析和 `git diff --check` 通过。
  定向测试覆盖远距离零处罚、中央近障碍封顶、零命令零处罚、纯横移/原地转向零处罚、图像
  外侧障碍不触发、全零深度按 5m 处理和 penalty 永不为正。使用真实
  `p2nav8h-r2_291713` checkpoint 完成 command-v2
  warm start、PPO update、Adapter update、保存和 exact resume smoke；本地 smoke 通过不等于真实相机
  分布已验证。当前尚无证据证明 ROI 在 slope/stairs/maze 上均无误触发，平台短跑前不得标记平台
  已验证。
- 2026-07-29 开发容器补充：同步后使用容器实际测试环境复跑 P2 定向测试 `86 passed`，邻近回归
  `264 passed`；该层只验证算法、奖励、monitor 和 checkpoint 契约的 CPU/stub 路径。容器镜像
  当前没有可导入的 PyTorch，因此未运行真实相机/GPU rollout，状态仍保持“平台 smoke 待验证”。
- 防复发、回滚与最短检查路径：禁止把零 depth 当 0m 障碍，禁止增加 clearance 正奖励或按
  terrain/heading 关闭风险；测试必须保持 cap、零命令和外侧噪点语义。若短跑出现楼梯前停车，
  先看三个 segment 的 clearance/risk/penalty，再调整 ROI/quantile/risk band，不能先提高 success
  或删除 collision。回滚应原子恢复 reward/training version、算法 component、面板和测试。

## BUG-20260729-012：大 checkpoint 单 bundle 上传易丢片且无法断点续传

- 日期：2026-07-29；状态：开发容器已验证；真实 PPO smoke 因容器无 PyTorch 未执行。
- 影响：开发容器中的模型上传与真实 checkpoint 继续训练 smoke；正式训练运行时和模型格式不变。
- 症状：25 MB `model.ckpt-navfull-291713.pkl` 使用单个 GET bundle 上传时，最终返回
  `bundle is missing chunks`；降低到 2 worker 后又在长传输过程中遇到
  `WEBIDE_RECORD_NOT_FOUND`。失败 bundle 没有生成目标模型，但每次只能从头重传，延长页面
  空闲时间并增加容器回收概率。
- 根因：源码同步 bundle 针对少量小文件设计，网关仍按 4 KiB GET 请求转发。单个 20 MB 以上
  bundle 会形成数千请求，任一响应丢失都会让整包失败；现有客户端没有暴露 bundle 内缺失 offset，
  无法只重传缺片。Chrome 页面交互只能维持前端活动，不能保证后端 IDE 记录不被平台回收。
- 修复：新增 `server/model_chunk_uploader.py`。本地先把模型拆成默认 4 MiB 的独立 bundle，使用
  `part_workers × chunk_workers` 可控并发，每片单独重试和 SHA256 校验；远端 manifest 中哈希一致
  的片段直接跳过。全部片段齐全后，容器按固定顺序写入 `.uploading`，整文件大小和 SHA256 一致
  才 `os.replace()` 原子发布，成功后默认清理片段。目标路径硬限制在
  `agent_ppo/test_artifacts/`，不允许覆盖源码、平台配置、凭据或任意绝对路径。
- 排除项：本轮 11 个 P2 源码文件的普通增量 bundle 已在 0.8 秒内完成并经二次 dry-run 证明
  overwrite 为 0；P2 容器测试 `86 passed` 和邻近回归 `264 passed`。因此大模型失败不是源码
  同步、reward-v6、checkpoint 反序列化或模型 ID 门禁问题。
- 修改文件：`server/model_chunk_uploader.py`、`server/tests/test_model_chunk_uploader.py`、
  `server/README.md`、`server/CHANGELOG.md` 和本台账。
- 验证：工具离线测试 `4 passed, 4 subtests passed`；既有同步测试排除沙箱禁止绑定回环端口的
  HTTP 往返用例后 `26 passed, 1 deselected`，Python 编译与 `git diff --check` 通过。容器先将
  `291713` 的 7 个 4 MiB 级片段重组，整文件 SHA256
  `79cf02e890bdf032dcec6fdc259426e047b8e5ca7bb41521c1b7ad692da71a2a` 与本地一致；新工具随后
  正确识别完整远端文件并零上传。另以 `model_chunk_uploader.py` 自身完成真实分片上传、容器原子
  重组、最终 manifest SHA 校验和自动片段清理。容器 P2 专项 `86 passed`、邻近回归
  `264 passed`。尝试运行真实 `291713` command-v2 smoke 时，容器
  `/opt/conda/envs/env_isaaclab` 报 `ModuleNotFoundError: No module named 'torch'`，且 `/opt`、
  `/workspace` 未发现 torch 安装；因此本轮不能把真实 PPO/Adapter backward 标记为容器已验证。
  本地同一 checkpoint 的 warm start/PPO/Adapter/save/exact-resume smoke 已通过。验证后按授权删除
  容器中的 25 MB 临时模型、7 个手工片段和 round-trip 文件，未删除凭据或源码。
- 防复发、回滚与最短检查路径：大于源码同步上限的模型不得加入 `local_sync_client` 常规 scope；
  遇到失败先看远端 part manifest 和每片 SHA，再看最终 merge SHA，不能仅凭 HTTP 200 判断成功。
  回滚只需删除本地工具和容器 `agent_ppo/test_artifacts/` 临时文件，不触碰训练代码与已发布模型。
  关联 commit/PR、平台任务 ID和新 checkpoint：尚未生成。
- 2026-07-31 更正：在 P3 父包 `p2nav2h-r2_648278.zip` 实测中，`scope=agent_ppo`
  manifest 返回的 key 可为 `test_artifacts/...` 相对路径，旧客户端只查询
  `agent_ppo/test_artifacts/...` 完整路径，因此已经原子合并且 SHA256 正确的文件
  仍会被误判为缺失并重复上传；中断后项目根目录留下 `.ide-sync-*.tar.gz`
  bundle。已将 manifest 查询统一为兼容 rooted/scope-relative key，最终文件与分片均
  使用同一规则；默认并发从 `2×4` 收紧为 `2×2`。后续发现 4 MiB 单片在
  `2×2` 下仍包含约 1024 个 GET chunk，首片长时间无进度，因此默认单片进一步收紧为
  1 MiB，降低单片重试成本并提高进度可见性。实测父包大小
  `24,694,018` bytes，容器 SHA256
  `cd82e6d98e32dc6c09a9341138d84d19cfd190ee1ef0c565a25a7bfd9e5bf784`，ZIP 全量校验通过。
  修复后以默认 `2×2` 参数复验，客户端在约 1.5 秒内输出
  `remote file already verified`，没有再生成分片或 bundle。随后已删除本次测试产生的
  父包 ZIP、`extracted/`、`.parts/` 和项目根目录 `.ide-sync-*.tar.gz`，未删除代码、
  凭证或平台缓存。上传器离线回归 `7 passed, 4 subtests passed`，该更正已完成开发容器验证。
  后续用 1 MiB/`2×2` 完整上传同一 24 MB 父包，24/24 分片均输出独立校验进度，
  最终原子合并 SHA256 一致；本次测试为后续 P3 smoke 保留父包，未提前删除。

## BUG-20260730-013：中央深度风险误罚坡面且 terminal 诊断混入 reset 后数据

- 日期：2026-07-30；状态：本地已验证，开发容器与平台 smoke 待验证。
- 影响：`codex/p2-track-nav-ppo` 的十小时三轴 P2 训练奖励、命令/速度分桶、碰撞与步态面板；
  父包固定为 `p2nav8h-r2_291713-F2`，模型 ID 仅用于候选选择和 lineage，不形成单点门禁。
- 症状：旧预测碰撞只读取中央前向 ROI 的 10% 低分位并仅按 `vx` 处罚，无法区分垂直墙、缓坡和
  仅下部接近的台阶，也不能为 `vy`/纯转向选择对应方向。高层 10 帧窗口中途 done 后，workflow
  继续推进 reset 后的新 episode；tick 末的 target/exec/measured/true、gait/collision 与命令桶会
  读取新 episode 数据，`collision_onset_count` 还被错误展示为全程累计。
- 根因：手工风险缺少垂直结构与方向投影；workflow 只冻结 terminal goal/segment，没有冻结首次
  done 前的命令与 worker aux。gait excess 仍通过 `reward_*` 名称展示，容易被误解为高层职责。
- 修复：reward contract 升级为 `p2_track_reward_v7_directional_wall`。深度按三个方向和三个高度带
  取 q20，用 upper/lower 一致性、upper/middle near 与 stopping gap 组合 wallness/risk，再按
  `atan2(vy,max(vx,0.05))+0.5*wz*0.8` 的方向权重平滑汇总；零/无效/远景有限且零命令严格零处罚，
  单 tick 下限 `-0.02`，旧中央风险仅 shadow。上一轮 100 分钟数据中，旧 `-0.04` 风险平均
  `-0.00875/tick`、约占总负奖励 6.5%，但 body contact 从 1.8% 升至末段约 41%，证明不能只靠
  放大旧 ROI。新权重取旧上限一半，约为正常 `0.3m/s` 推进 tick 收益的 3.3%，用于加强提前
  转向提示，同时仍显著弱于真实 collision onset。gait PPO 权重设零，excess 改为无 `reward_` 前缀的
  诊断。每个低层 step 前保存 target/exec/aux，首次 done 冻结；tick 末 terminal 行选冻结值，
  live 行选末帧值，terminal tracking 从有效统计剔除。`vy_log_std=-1.1`，动作范围仍为 `±0.40`。
- 修改文件：`p2_contract.py`、`p2_high_level.py`、`algorithm_p2_nav_ppo.py`、
  `p2_nav_ppo_workflow.py`、P2 TOML、`monitor_builder.py`、`test_p2_core.py`、接口合同、CHANGELOG
  和本台账。未修改平台托管 `server/isaac_env/base_env.py`。
- 验证：P2 定向测试与邻近 monitor 测试 `114 passed`；覆盖垂直墙高风险、线性坡低风险、仅下部
  台阶低风险、左右横移和纯转向方向选择、零命令/无效深度/远景有限零处罚、gait 权重零但 excess
  更新、terminal-safe tensor、F2 二维头逐值迁移及 `vy_log_std=-1.1`。Python 编译、TOML 解析和
  `git diff --check` 结果见本任务最终记录；尚未进行真实 F2 容器 rollout 或五分钟平台 smoke，
  不得标记平台已验证。
- 防复发、回滚与最短检查路径：测试必须同时保留 wall/slope/stairs/directional/zero-depth 六类
  样本；terminal 行不得用 reset 后 telemetry 补值。若平台短跑仍误罚坡面，先比较 legacy/v2、
  三方向 wallness/risk 和真实视频，再调 flatness/near 尺度，禁止添加 terrain/heading 门控或
  clearance 正奖励。回滚须原子恢复 reward contract、算法 component、workflow 快照和面板。
- 关联 commit/PR、新任务/checkpoint：尚未提交；任务名 `p2nav10hvyavoid2`；新 checkpoint 未生成。

## BUG-20260730-014：局部进度长期补贴未完成 episode，策略转向迷宫超时

- 日期：2026-07-30；状态：本地已验证，开发容器和平台 smoke 待验证。
- 影响：`p2nav10hvyavoid2` 的 5 Hz 高层 PPO reward settlement；网络输入、动作 shape、低层与
  Adapter 合同不变。父包仍为 `p2nav8h-r2_291713-F2`，身份元数据不形成单点硬门禁。
- 数据证据：`20260730-004400` 的 100 分钟抓取中，末段 `positive_progress+new_best≈0.574/tick`，
  success 仅约 `0.076/tick`；前后有效窗口 success 增量 `65→62`，failure `45→42`，timeout
  `18→46`。maze progress `0.176→0.111m/s`，spin `12.4%→28.4%`，stuck `17.4%→51.5%`。
- 根因：旧局部 progress/new-best 在坡和楼梯持续发放，即使最后迷宫失败也不会撤销；timeout
  `-15` 又显著轻于 hard failure `-25`，稳定卡墙直至超时可能优于继续探索终点。
- 修复：reward contract 升级为 `p2_track_reward_v8_terminal_potential`。删除进入 PPO storage 的
  positive/negative progress 与 new-best，使用 episode-relative monotonic frontier potential：
  `phi=2*(episode_start-best)`，tick reward 为
  `gamma_frame^duration*phi_after-phi_before`；success/failure/timeout 的 `phi_after=0`，通过同一
  terminal-safe duration 和 outcome 结算。timeout 改为 `-22.5`，success `+50`、failure `-25`、
  time/stagnation/collision 保持不变。面板新增每 rollout terminal 数量、episode success fraction
  和 timeout fraction，避免继续误读平台窗口 completed count。live episode start/best 不写
  checkpoint，resume 后 reset。
- 修改文件：`p2_contract.py`、`algorithm_p2_nav_ppo.py`、`monitor_builder.py`、`test_p2_core.py`、
  server-deploy contract、CHANGELOG 和本台账。
- 验证：定向测试覆盖新 frontier 产生稠密信号、相同 frontier 不可刷正分、terminal clawback、
  timeout 单次 bootstrap、terminal reset 后 episode 状态清空及 reward decomposition；最终测试数和
  静态检查见本任务结论。真实 F2 rollout 和平台 5 分钟 smoke 尚未执行。
- 防复发：完成能力必须看 success/(success+failure+timeout) 和 outcome 增量，禁止用窗口
  completed_count 或局部 progress 代替。任何新的稠密进度项必须证明 failure/timeout 不保留可刷取
  正回报。回滚需原子恢复 reward version、algorithm component、monitor 与测试。
- 关联 commit/PR/checkpoint：尚未提交；新 checkpoint 尚未生成。

## BUG-20260730-015：特权 scanner 坏帧被当开阔空间且 20 列训练仍按 10 列归因

- 日期：2026-07-30；状态：本地已验证，开发容器、平台 smoke 和行为评估待验证。
- 影响：`codex/p2-track-nav-ppo` 的提前避障教师、P2 reward、两小时续训 checkpoint、
  Track 难度统计与训练吞吐；任务名 `p2nav2hsafedir`。父包为最新验证通过的
  `p2nav10hvyavoid2` 完整包，模型 ID/标签只用于选择和 lineage，不形成单点硬门禁。
- 症状：旧 `nav_scanner_privileged_features()` 将 NaN、负无穷、混合 finite/inf triplet 全部
  `nan_to_num(0)`，与合法 no-hit 一样解释成开阔空间，无法可靠生成安全方向标签。同时
  `P2CurriculumAccumulator`/`P2TrackCurriculumProbe` 将 outcome、start 和 histogram 固定为
  `3×10`；切换 20 条 Track 后 L10–L19 会被 clamp 到 L9 或越界，面板不能证明静态均匀覆盖。
  高层 tick 的 depth 只保存 GPU view，若平台复用 observation buffer，10 个低层帧后才写入
  rollout 时可能已经变成后续帧。
- 根因：scanner 特征最初服务严格 DAgger Oracle，没有定义逐 ray 完整性；P2 复用后又缺少
  “允许缺帧但 mask 标签”的模式。课程探针继承两小时/八小时 10 列常量。workflow 与 algorithm
  对 observation 所有权没有显式区分只读 view、立即快照和延迟 storage 写入。
- 修复：scanner 仅接受 XYZ 全有限的 hit 或 XYZ 全 `+Inf` 的合法 no-hit；NaN、负无穷和混合
  triplet 判坏。`well_formed_ratio>=0.95` 且 `finite_hit_ratio>=0.80` 才令 P2 `available=1`；
  no-hit 按无墙，坏 ray 不参与均值。P2 通过显式非严格上下文跳过无效样本，DAgger 默认仍硬失败。
  新增 goal-independent 安全教师：三方向 scanner risk 与 `height_scan256` 的 q90 前向高度跳变
  共同生成 safe3；SafetyHead 只读 nav_feat32，并与 PPO 共用 NavigationEncoder forward、Actor
  optimizer 和一次 backward，不连接 Actor LSTM。错过安全方向奖励只在有明显安全替代、命令非零、
  scanner 有效时产生负值，不奖励开阔空间。Track 改为 20 列、`curriculum=false`，probe 与
  checkpoint 统计升级为 3×20，并抑制“课程未生效”的错误告警。高层 depth 在 tick 起点立即复制为
  独立 CPU FP16；terminal 只复制首次 done 行，完整 next observation 不再逐帧 clone；命令/赛段桶
  使用 `scatter_add_`。
- Warm start/checkpoint：新增 `p2_safe_direction_continue_warm_start`，保留 Actor、Critic、Adapter、
  return statistics、RNG 和旧参数组 Adam moments；SafetyHead 使用新参数和空 Adam state。新包保存
  `navigation_safety_head` 的 `class_name/spec/state_dict` 并标记 `training_only=true`。exact resume
  要求恢复该 leaf；即使配置仍写 warm-start 模式，同 reward/training contract 且含 Head 的本轮包
  也优先走 exact resume，不能重置两小时 session。Track/Standard eval 不实例化或调用它。
- 排除项：没有把 goal alignment 或 heading 塞入教师；没有增加 open-space 正奖励、corridor
  centering、转向幅值奖励或 recovery bonus；没有降低 PPO epoch、控制频率、相机分辨率或环境数。
  平台镜像确认 scanner 为 273 ray、ordering=xy，对应 21 个 lateral-y row × 13 个 forward-x col；
  该镜像证据仍需当前开发容器启动日志复核。
- 本地验证：P2、Nav observation、monitor 与 P1.5 邻近定向测试 `135 passed`。覆盖 21×13 pattern、
  合法 no-hit/坏 ray、左右中墙、坡面/普通台阶连续性、安全替代/等价方向、SafetyHead 梯度隔离、
  20 列统计、environment buffer 复用时 depth 所有权、四阶段字段、warm start 保留 Critic/return/
  optimizer moments、checkpoint exact resume 与 eval 忽略 training-only leaf。Python 编译、TOML、
  diff check 的最终结果见本任务结论。真实 128 环境 rollout/backward/save/resume 与五分钟平台 smoke
  尚未执行，不能标记平台已验证。
- 防复发、回滚与最短检查路径：scanner fixture 必须保留 open/left/front/right/no-hit/NaN/shape；
  20 列配置必须与 probe tensor shape、histogram 和 monitor 面板原子更新；视觉 rollout 不能延迟持有
  平台 observation view。若 smoke 中 `scanner_available_share` 很低，先看两分钟 scanner ratio 日志和
  真实 ray shape/order，不得通过把 `available` 固定为 1 绕过。回滚需同时恢复 reward/training
  contract、SafetyHead leaf、optimizer group、3×20 probe、TOML 和监控。
- 关联 commit/PR、新 checkpoint：尚未提交；新 checkpoint 尚未生成。

### 2026-07-30 更正：左右坐标、safe 标签发现和条件统计

- 更正原因：后续独立审查发现，上一版“左右中墙 fixture 已覆盖”的结论错误。测试使用人为 row
  编号命名左右，没有依据平台实际 ray 起点坐标，因此把错误约定固化为通过测试。平台镜像
  `sensor_patterns.py` 明确显示 `ordering=xy` 时 lateral `y` 从负到正排列；Go2 机体系 `+y` 为左，
  所以低 row 是右侧、高 row 才是左侧。
- 用户可见风险：左墙可能被写成右侧风险，SafetyHead 和 `missed_safe_direction` 会把策略推向障碍；
  新保存的 `safewarm/safefull/safestable` 又不在 P2 候选表内，同 ID 续训/评估可能找不到新包并
  回退旧 `navfull`。此外 NaN height scan 仍会形成“三向全危险”的有效标签，选向率分母包含方向
  等价样本。
- 修复：scanner 和 height 教师统一改为 low-row=right、high-row=left，对外仍输出
  `[left,center,right]`；fixture 按真实 `y<0/y>0` 构造。height 每个方向要求有限值比例和有效相邻
  diff 数，任一方向无效则整条教师标签 mask。P2 标签发现顺序补齐
  `safestable > safefull > safewarm >` 旧标签。选向验收分母改为 scanner/height 有效、运动中、
  best-safe 足够、`safe_gap>0` 且最佳方向非并列；分子为其中实际选择唯一最佳方向的样本。
- Standard 契约：新 P2 包补顶层低层 `model_spec`；仅在显式 `low_level_only_eval`，或 Standard
  terrain 下的 `standard_low_level_only_eval` 时，Camera loader 才读取规范 low-level
  `locomotion_encoder/actor`。默认路径仍拒绝静默丢弃完整高层，Track 仍加载完整 P2。
- 验证：修复后定向测试 `168 passed, 3 subtests passed`；覆盖物理坐标左右墙、NaN height mask、
  `safe*` 同 ID 优先发现、默认拒绝与显式低层评估分类。Python/TOML/diff 静态检查及更广测试见
  本任务最终结论。开发容器、真实 checkpoint round-trip、128 环境和平台 smoke 仍待验证。
- 状态：本地已验证；平台行为未验证。旧“135 passed 已覆盖左右 fixture”的文字保留作为历史，
  以上述更正为准。
- 训练时长调整：用户随后将本轮从 4 小时缩短为 2 小时；新合同为
  `p2_track_training_2h_safe_direction_v2`，阶段边界为 30 分钟和 90 分钟，session 目标 7200 秒，
  安全方向权重仍从 0 渐进到最终 `0.03`。该调整不改变网络、reward 上限、20 列地形或 PPO 规格。
- 2026-07-30 开发容器复核：最新 bundle 同步后二次 dry-run 为
  `files to overwrite: 0`；最终 P2/checkpoint 定向回归 `169 passed`，关键 Python 编译和
  8 份 TOML 解析通过。使用真实 `291713-F2` 包做 command-v2 smoke 时依次发现三个
  验证链回归：工具用 `INITIAL_LOG_STD=-0.7` 而非 `INITIAL_VY_LOG_STD=-1.1`
  校验新 `vy` head；人工 rollout 未补 `safety_target/safety_valid` 和 depth；CPU 路径在
  禁用 AMP 后仍将 FP16 depth 直接传入 FP32 CNN。前两项修复 smoke，第三项在非 CUDA
  CNN forward 前转 FP32，GPU 训练仍保持 FP16 storage + AMP。复跑通过：保留 CNN/LSTM/
  旧 `vx/wz`/Adapter，新 `vy` 为零均值且 `log_std=-1.1`，执行 8 次 PPO 更新、1 次
  Adapter 更新、保存和 `exact_resume_history_reset`。真实 Isaac Track+Camera terminal smoke 进入 headless Kit 后场景
  初始化超过 4 分钟未返回首次 reset，已只终止本轮 smoke 进程；不得因此标记
  真实 Isaac、128 环境或平台行为已验证。

## BUG-20260731-001：P3 复合训练入口缺少 lifecycle 接线且 worker transport 被禁用

- 日期：2026-07-31；状态：本地已验证，开发容器真实 checkpoint 继续训练与
  Standard+Camera reset/step 已验证，完整 P3 workflow rollout 与平台 smoke 待验证。
- 影响：分支 `codex/p3-standard-joint-recovery`，任务 `p3std8h-sim2real`，父包
  `p2nav2h-r2_648278`。P3 目标是在同一个 Standard+Camera 任务中依次恢复低层、校准
  ResponseAdapter、适配高层并做低频联合微调；包保持 `deployable=false`。
- 用户可见症状与审查证据：初版只有 P3 模块装配，通用 `workflow()` 会落入单算法 PPO 路径；
  `Agent.learn/predict/exploit/save_model/load_model` 没有完整 P3 分支，复合 coordinator 也没有
  独立 low/high rollout 与 exact resume。后续补上 observation 复用后又发现
  `P2WorkerBridge._resolve_config()` 只允许 `p2_nav_ppo/p2_nav_eval`，真实 P3 critic observation
  会在 `p2_response_aux()` 直接报 `P2 response aux requested while bridge is disabled`。
- 根因：P3 不是把现有 stage 名称换掉即可运行；它需要 50Hz 低层与 5Hz 高层两条独立 recurrent
  时间轴、各自 storage/optimizer/update 计数和阶段边界 reset。共享 response transport 的启用条件
  仍把 stage 写死为 P2，且 P3 dashboard 直接复用 Track curriculum/segment 面板，导致 Standard
  任务出现空面板或错误语义。
- 修复：新增 `p3_standard_joint_workflow.py`，分别收集 80 帧低层 rollout 与 32 tick 高层
  rollout；阶段只在完整 rollout 边界切换并 reset 环境、高低层 hidden、slew 和 future history。
  Agent 新增 P3 专用训练 no-op 回调、推理、候选加载、阶段保存和 SIGTERM/final save。
  `AlgorithmP3StandardJoint` 保存/恢复低层 critic、动作方差、optimizer、RNG、低/高层更新计数、
  session/lifetime 时钟，并区分 P2→P3 warm start 与 P3 exact resume。checkpoint 标签加入
  `lowbase/lowmild/lowmedium/lowfull/adaptercalib/highadapt/jointslow`；模型 ID/lineage 不一致
  只告警，选中文件的结构或有限值错误仍硬失败。P3 共享 worker feedback/gait transport，但禁用
  不适用的 Track curriculum worker probe。监控改为 P3 专用阶段、局部目标、三轴命令链、步态、
  Adapter 和资源面板。
- 目标与成功契约：完整 policy observation 保持 57905，高层 Actor85 不变；低层输入适配器只
  删除 goal4 得到 57901，不改变低层网络结构。私有目标从当前位置采样 1.5-2.8m，并限制在
  8m 地块的 1m 内边界；进入 0.6m 只结算一次高层奖励并在下一 observation 重采样，不终止
  Standard episode。平台 Standard scorer 仍是正式完成口径，joint success 只允许监控。
- 本地验证：`PY311test` 下 P3/P2/P1.5 定向回归 `142 passed`；修复 worker stage gate 后
  P3/P2 核心回归 `105 passed`，最终定向与邻近回归为 `171 passed`。广域测试为
  `386 passed, 31 subtests passed, 4 failed`；四个失败分别是已有 P2 smoke 工具被通用 import
  规则扫描、sandbox 禁止本地 HTTP bind、旧 LBC 配置仍为 50 iterations、nav smoke 对 cwd 的
  历史假设，均不涉及本轮 P3 文件。Python 编译、全部 server TOML 解析与 `git diff --check` 通过。
  使用真实归档 `archive/代码存档/p2nav2h-r2_648278.zip` 中
  `model.ckpt-safestable-648278.pkl` 完成 P2→P3 结构 warm start、P3 保存和 exact resume smoke；
  session 秒数从 123.0 恢复，live hidden/future history 按合同清空。尚未在 Isaac 开发容器完成
  env.reset、首个 50Hz/5Hz rollout/backward 或平台 checkpoint 注册，因此状态不能升级为平台已验证。
  2026-07-31 尝试通过现有 RPC dry-run 连接开发容器，腾讯代理返回
  `WEBIDE_RECORD_NOT_FOUND`；这是 IDE 容器记录失效，不是模型 ID、checkpoint 门禁或 P3 代码
  报错，待用户重新打开容器后继续在线 smoke。
  2026-07-31 容器重开后完成代码 bundle 同步，二次 dry-run 为
  `files to overwrite: 0`；P3 contract/schedule/worker 定向测试 `37 passed`，Python compileall 和
  10 份 TOML 解析通过。真实父包经分片上传、SHA256 和 ZIP 校验后，运行
  `p3_parent_smoke.py` 在 import 阶段报
  `ModuleNotFoundError: No module named 'torch'`；当时该失败尚未取得真实 checkpoint tensor
  加载、rollout/backward 或 Isaac reset 证据。
  同一 smoke 在本地 `PY311test` 使用同一父 checkpoint 通过，输出
  `p3_joint_warm_start:p2_safe_direction_continue_warm_start` 和
  `p3_exact_resume:exact_resume_history_reset`。
- 2026-07-31 容器验证更正：上述“容器不含 PyTorch”判断错误。PyTorch
  `2.7.0+cu128` 位于 Isaac Sim 的
  `omni.isaac.ml_archive/pip_prebundle`，必须通过 `/workspace/isaaclab/isaaclab.sh -p`
  注入依赖路径。首次包装器 smoke 仍失败，是因为命令前人为设置 `PYTHONPATH=.`
  覆盖了包装器路径；移除该覆盖后，容器使用真实
  `model.ckpt-safestable-648278.pkl` 完成 P2→P3 warm start、P3 save 和 exact resume，
  输出与本地一致，`parent_loaded=true`、low-level digest 存在、session 恢复为
  `123.0s`。该证据验证了真实 checkpoint tensor 加载与 save/resume，仍不等价于
  Isaac `env.reset()` 或 rollout/backward 已验证。
- 2026-07-31 开发容器最终 smoke：扩展 `p3_parent_smoke.py` 后，真实 648278 父包
  在 `lowbase` 阶段填充一个完整低层 recurrent rollout，执行 1 次 PPO
  backward/optimizer step，`applied_updates=1`，Actor 和 Critic 参数均发生有限更新，随后
  P3 save/exact resume 通过。独立 `p3_env_smoke.py` 通过 Isaac Lab 包装器创建 1 个
  Standard+Camera 环境；reset 返回 policy `[1,57905]`、worker wire `[1,385]`，policy、
  critic core 和 worker aux 有限值比例均为 1.0，连续 3 次 `env.step()` 成功，第 3 帧
  reward `-0.0194000751`、`terminated=false`、`truncated=false`，输出 `status=PASS`。
  工具一度因对整个 worker wire 做过严有限值断言而在 reset 后退出，已改为只对
  policy/critic core 做硬断言并单独报告 aux；同时修正了 vGPU `env.close()` 使用
  `os._exit()` 导致外层 timeout 残留的测试编排，最终只保留一个 smoke 实例。当前
  容器仍保留父包 ZIP、解压 checkpoint 和 `/data/pre_model/ckpt` 副本，待全部后续测试
  完成后再清理。本证据尚未覆盖正式 P3 workflow 的 80 帧低层 rollout、高层
  32 tick rollout 或平台 checkpoint 注册。
- 2026-07-31 平台 15 分钟回归与更正：任务 `p3nav8h-r1`（task `235452`）从
  `09:01:33` 运行至约 `09:15:41`，训练持续到 `iter=105`、`session_h=0.224`、
  `low_updates=105`，没有数值异常或主动达到 8 小时目标。`09:07:36` 已成功写出合法的
  `model.ckpt-lowbase-648278.pkl`，证明 checkpoint 标签正则和业务写盘正常；但 learner proxy
  全程报告 `succ_cnt is 0`，平台模型 ID 未离开父 ID `648278`，随后 aisrv/learner 同时收到
  外部 SIGTERM。根因是 P3 workflow 虽然让 `Agent.learn()` 对 P3 返回 lifecycle no-op，却从未
  在任何低层或高层环境帧调用它；5 分钟墙钟 `save_model()` 不能代替平台 train step、
  `dump_model_freq` 和健康看门狗。修复为每个成功 `env.step()` 完成全部帧处理后恰调用一次
  `agent.learn(None)`，低层 80 帧与高层 32×10 帧均覆盖；普通 callback 失败计数并继续，
  `CheckpointSaveError` 保持硬失败。新增 `platform_lifecycle_callbacks/failures` P3 面板与定向
  回归。状态：代码已修复待本地、开发容器和新平台任务验证；旧任务已失败，不能热修复。
- 2026-07-31 lifecycle 修复验证补充：本地从 `server/` 运行 P3 schedule 与相邻监控定向测试
  `47 passed`，Python 编译、全部 server TOML 解析和定向 `git diff --check` 通过。随后只将
  `agent_ppo/workflow/p3_standard_joint_workflow.py` 与 `agent_ppo/conf/monitor_builder.py`
  同步到开发容器，二次 dry-run 为 `files to overwrite: 0`；容器使用 PyTorch
  `2.7.0+cu128` 完成 Python 编译，并确认 lifecycle helper 连续两次调用均成功、attempt/success
  计数均为 2、failure 为 0，低层和高层 rollout 的两个调用点均存在。状态更新为“本地已验证、
  开发容器静态与 helper 行为已验证，新平台任务跨 20 分钟待验证”；该证据仍不能替代平台
  learner wrapper 的 global-step、模型登记和 watchdog 行为验证。
- 2026-07-31 P3 阶段交接补充验证：新增 `p3_joint_rollout_smoke.py`，使用真实 648278 父包在
  5h/6h/6.5h rollout 边界分别覆盖 `adaptercalib/highadapt/jointslow`，并断言 target/exec、
  pending tick、高低层 recurrent hidden 和未完成 Adapter future history 在边界 reset 后清空。
  `adaptercalib` 完成 80 个真实 Standard 低层帧和 4 次 Adapter update，只有 Adapter 参数变化；
  `highadapt` 完成 32 个高层 tick × 10 个低层帧、高层 PPO、Adapter update、save/exact resume，
  低层 Actor forward hook 记录的注入命令绝对值均值为 `0.29683`，证明高层命令实际控制了
  Standard 低层动作路径。
- 该 smoke 同时发现两个训练语义 Bug。第一，P3 调用了 Adapter update，但旧阶段调度在低层恢复
  和 `adaptercalib` 将 response optimizer LR 置零；现按阶段使用 `3e-5/2e-4/1e-5`，并用参数
  实际变化而非计数验证。第二，高层推理将共享低层模型留在 `eval`，导致 `jointslow` 低层 CUDA
  recurrent PPO 报 `cudnn RNN backward can only be called in training mode`。现规定 rollout 收集
  显式 `eval`，低层 replay 恢复 Actor/Critic/LSTM `train`，冻结 CNN 和 anchor 保持 `eval`。
  修复后 4 环境 `jointslow` 明确 PASS：高层 PPO 20 个 minibatch update、Adapter update、80 帧
  低层 recurrent PPO、save/exact resume 全部完成；低层 Actor/Critic 变化，低层 CNN/LSTM 不变，
  NavigationEncoder/高层 Actor/Critic/Adapter 均变化，所有关键 loss 有限。本地和容器 P3
  contract/schedule 回归均为 `12 passed`。
- 正式规格边界：开发 IDE vGPU 只暴露 5 GiB。64 环境 full-terrain 在模型 forward 前由 PhysX
  尝试分配 `671088640` 字节 `mGpuContactPairsDev` 时 OOM，32 环境同样在 Articulation reset 阶段
  OOM；随后 16 环境 full-terrain 在 height-scanner RayCaster 初始化触发 CUDA illegal-memory。
  这些失败均发生于 `env.reset()`，不涉及 checkpoint 门禁、PPO 或 Adapter。compact-terrain 的
  完整算法链已验证，但 64 环境正式 full-terrain 必须在训练规格 GPU 或重启后的更大显存容器做
  平台 smoke，当前不得标记该规格已验证。
- 已知边界：worker 已计算 `local_abs>3.2m` 诊断，但 2026-07-29 平台镜像显示现有 terrain-bound
  termination 只在 eval 模式启用；训练态强制 partial reset 尚未取得可靠公开 API，当前不得宣称
  已生效。原计划的分阶段质量/COM/电机/PD/action-delay/push 表目前只有配置记录，除顶层 TOML
  静态摩擦外尚无经过 runtime 证明的执行器；正式长训前必须在开发容器补实现或明确缩减合同。
- 防复发、回滚和最短检查路径：回归测试必须覆盖 P3 stage 能启用共享 worker transport、57905→
  57901 只删除 goal4、局部目标事件保留一帧、P3 标签优先级、低高 optimizer 无交集、阶段 LR 和
  P3 save/resume。启动时最短检查为：`Stage=p3_standard_joint` → worker 日志
  `runtime_stage=p3_standard_joint` → P3 parent load mode → 首个 low/high update → 同 ID P3 阶段包
  save/load。回滚只需切回 P2 `policy_entry` 与 648278 父包，不得把 P3 包交给旧二维/单层 loader。
- 关联 commit/PR/checkpoint：P2 父提交 `b7519e6`；P3 尚未提交/未创建 PR；父 checkpoint ID
  `648278`，新 P3 checkpoint 尚未生成。

### 2026-07-31 更正：P3 正式训练阻断闭环与 compact storage

- 状态：代码已修复待开发容器复核。关联分支 `codex/p3-standard-joint-recovery`，任务
  `p3std8h-sim2real`，父包 `p2nav2h-r2_648278`；新 checkpoint 尚未生成，commit/PR 尚未创建。
- 新审查证据：旧实现的 P3 低层阶段仍沿用 300 秒原生命令，worker RewardManager 不读取高层注入
  command；`jointslow` 又在高层 rollout 后采集另一段原生命令低层数据。目标采样使用 clamp，边界
  附近会缩短到 1.5m 以下；`local_out_of_bounds` 只有诊断，没有 termination。P3 高层继续结算 P2
  Track SafetyHead、predictive collision、missed-safe、gait、body collision、tracking、stagnation 和
  Track timeout impulse。低层 storage 保存 `80×64×57901 float32`，仅 observation 即约 1.10GiB。
  分阶段 DR 表大多未被 worker 消费，exact resume 也没有不可变 anchor leaf。
- 根因：P3 最初只完成模块装配，没有把 command、reward、环境随机化、rollout ownership 和
  checkpoint lineage 视为同一个原子训练合同；旧 smoke 工具仍发送 12 维 action，并在 jointslow
  额外采样低层 rollout，反而会把错误路径验证为通过。
- 修复：新增 `P3RecoveryCommandSampler`，按 55/15/15/5/5/5 比例生成 2-8 秒三轴命令，核心/扩展
  80/20，全程启用。P3-only 17 维 envelope 携带 `joint12+exec_cmd3+epoch+valid`，BaseEnv 写入并
  readback `base_velocity` 后只把 joint12 送入 Isaac；原生命令 resampling 延长到 40000 秒。
  `jointslow` 只使用同一 32×10 高层 rollout 的前 80 帧，更新顺序固定为 high→low→版本边界→
  Adapter。低层更新后立即更新 digest/version 并清除 unfinished future history。
- 目标/奖励：局部目标使用最多 64 次 rejection sampling，严格保持 1.5-2.8m 与地块 1m 内边界；
  失败采样直接触发 reset，不伪造近距离成功。动态注册 `p3_local_out_of_bounds` hard termination，
  threshold、timeout、success distance 与 seed 均从 P3 TOML 读取。P3 高层只保留 frontier/progress、
  local +8/-1、time、crawl、command-rate 与 hard failure；local timeout 回收 frontier potential。
  Track safety teacher/head loss、predictive/missed-safe、gait/body collision、tracking、stagnation 以及
  Track success/timeout impulse 在 P3 中归零，SafetyHead 权重保留且冻结。
- Sim2Real/DR：四阶段环境配置展开摩擦、全 link mass、COM、PD、action gain、0-2 帧 delay、push 与
  observation noise；startup 随机化在 0.5h/2h/3.5h 边界通过环境重建生效。低层增加 torque 0.2s
  EMA excess、instant peak、normalized action rate/jerk 的合并有界代价，单帧 raw cost 不超过 0.12；
  joint/torque mapping 无效时归零并只报警一次。
- 显存/checkpoint：低层在线输入和导出仍为 57901，但 recurrent PPO storage 只保存
  `proprio45+frozen CNN feature32=77`；CNN collection 无梯度，replay 仍训练 LSTM/Actor/Critic。
  P3 checkpoint 新增 immutable anchor leaf/digest、command sampler RNG、low version 与实际 DR phase；
  exact resume 校验 anchor 和 Adapter completed records 的 low digest/version。显式 requested P3、配置
  parent P3、显式 P2 648278 依次选择；仅在三者都不存在时允许唯一 discovery，多候选明确报歧义。
- 额外回归：低层 timeout 过去把 pre-step value 当 terminal bootstrap，现因平台不提供 reset 前
  terminal critic observation而采用零 bootstrap，避免把旧状态或新 episode 污染 Critic。P3 eval
  与 `p3_env_smoke.py` 同步改为 17 维 envelope；joint smoke 改为验证同 rollout 80 帧路径。
- 本地验证：当前阶段 P3/P2/Nav 定向回归 `140 passed`；新增严格目标距离、17 维 transport、sampler
  对称/RNG、DR phase、77 维 compact replay 数值一致和 P3 reward profile 测试。最终测试、Python
  编译、TOML、monitor schema 与 diff check 将在本轮代码审查结束后补录。开发容器真实 648278、
  64 环境 full rollout/backward/save/resume、平台 15 分钟 smoke 尚未执行，不能升级为平台已验证。
- 回滚和最短检查：回滚必须同时恢复 17 维 envelope、P3 workflow、compact storage、P3 reward
  profile、DR materialization 与 checkpoint contract，不能只改 TOML。再次遇到时依次核对启动日志
  sampler→BaseEnv command readback→worker tracking command→同 rollout low storage 77→low version reset→
  anchor exact resume；任何一步缺失都不得启动正式 8 小时训练。

### 2026-07-31 更正：平台覆盖 BaseEnv，撤回 17 维 transport 与伪随机化能力

- 状态：本地已验证，开发容器待重新同步复核。用户明确确认正式平台会覆盖
  `server/isaac_env/base_env.py`；该文件现已完整恢复到分支基线，`git diff` 为空。
- 被证伪的旧方案：上一条记录中的 P3-only 17 维 envelope、worker command readback、动态
  out-of-bounds termination、全 link mass、COM、PD、action gain/delay 与按环境 push 都依赖本地
  BaseEnv 修改，正式任务不会执行。旧记录保留作为历史，不得再引用为已实现能力。
- 替代闭环：0-5h 低层恢复直接使用平台原生 2-8 秒三轴命令，PPO observation、worker
  RewardManager、实际动作与 Adapter records 使用同一命令。6h 后高层接管低层 observation，低层
  全程冻结，只更新高层 PPO 与 Adapter；最后阶段从 `jointslow` 更名为 `highslow`。这主动放弃了
  无法在公开接口下保证语义正确的同步高低层更新，避免用原生命令 reward 训练高层命令下的低层。
- 随机化实际边界：只保留镜像确认平台支持的 friction、base added mass 和显式 observation noise，
  在 0.5h/2h/3.5h 通过完整环境重建生效。push 保持关闭；COM、PD、action gain/delay 均从配置、
  checkpoint contract 和面板删除。`local_abs>3.2m` 仅作诊断，真实 reset 仍由平台 Standard
  termination/timeout 负责。
- 契约：环境 action 恢复为固定 12 维；contract 升级为 `p3_standard_joint_v3`；checkpoint 不再保存
  P3 command sampler，阶段标签固定为 `lowbase/lowmild/lowmedium/lowfull/adaptercalib/highadapt/highslow`。
  模型 ID 仍只用于候选选择和告警，不形成单点硬门禁。
- 本地证据：P3 contract/schedule 定向测试 `20 passed`；真实
  `model.ckpt-safestable-648278.pkl` warm start、一次完整低层 PPO update、save/exact resume 通过，
  输出确认父动作方差精确恢复、Actor/Critic 均有限更新。开发容器的真实 12 维 step、原生命令
  2-8 秒保持、friction/base mass/noise 实际装配、高层阶段低层 digest 不变仍待复核。
- 回滚与最短检查：正式任务不得同步或依赖本地 BaseEnv。启动后依次检查低层阶段 worker 原生命令
  与 policy command 一致、高层阶段 low update counter 不再增长、低层 digest 不变、Adapter records
  使用高层 owned command。若任何一项不成立，停止 P3 而不是恢复 17 维 transport。

#### 2026-07-31 审查补充：阶段 reset 与采集同步开销

- 状态：本地已验证，开发容器待验证。
- 审查发现：早期替代实现只在 DR 变化的 0.5h/2h/3.5h 重建环境，5h/6h/6.5h 仅清理网络
  hidden，导致旧 episode、局部目标和 worker 状态可能跨职责边界延续；`adaptercalib` 在低层冻结时
  仍计算 critic/log-prob/anchor 并写满 80 帧 PPO storage。P3 统计还在每帧执行地形 `any()`，并在
  每个高层 tick 对奖励项 `.item()`，造成大量 CUDA host synchronization。
- 修复：所有阶段边界统一在 checkpoint 后调用公开 `env.reset(config)`，仅 DR 参数是否改变作为
  诊断；校准阶段改为纯低层 actor 推理、environment step 与 Adapter record；高层 Adapter 按每两次
  PPO update 更新一次。命令、地形、奖励与事件统计均在 GPU 累积，rollout 末批量搬到 CPU。

#### 2026-07-31 审查补充：高层 terminal 后命令污染与监控同步

- 状态：代码已修复待开发容器验证。
- 症状与根因：P3 高层一个 5Hz transition 包含 10 个低层 frame。单个环境提前 termination 后，
  `active` mask 虽已清零，但余下 frame 仍可能由 `frame_begin()` 沿用或重新采样非零命令，使自动
  reset 后的新 episode 接收到旧 transition 的命令。另 Standard/joint-success 监控在每个 50Hz
  frame 对两个 CUDA 标量调用 `.item()`，形成不必要的 host synchronization。
- 修复：每个后续 frame 在 `frame_begin()` 前后都按 `active` mask 原地清零 `active_target` 与
  `exec_cmd`；这只操作 agent-owned command，不修改平台 `BaseEnv`。成功计数保持为设备 tensor，
  合并进 rollout accumulator 后才统一搬到 CPU。新增 inactive/live 两环境回归测试。
- 回归边界：terminal-safe aux/exec 快照仍取首次 done 前状态；inactive env 的后续 frame 不进入旧
  transition 的 duration/reward，下一导航 tick由正常 reset 状态重新开始。若容器 smoke 发现平台
  reset 帧无法接受零命令，应停止 P3，而不是恢复 17 维 transport 或修改 `base_env.py`。

#### 2026-07-31 审查补充：空 optimizer update 错误推进低层版本

- 状态：代码已修复待开发容器验证。
- 症状与根因：`AlgorithmVisualPPO.learn()` 在全部 minibatch 因非有限 loss/gradient 被跳过时返回
  `applied_updates=0`，P3 workflow 过去仍无条件增加 `low_updates` 并调用版本切换。这会让
  checkpoint 虚报低层更新、刷新 digest/version，并无必要地清空 Adapter 未完成 future history。
- 修复：新增 `_finalize_low_level_update()`，只有 `applied_updates>0` 才递增低层 rollout 版本并调用
  `note_low_level_update()`；零更新保持模型、版本和 Adapter history 不变。回归测试覆盖零更新与
  多 minibatch 成功更新两条路径。
- 最短检查路径：同时核对 `applied_updates`、`p3_low_updates` 与
  `response_buffer.version_reset_count`；前者为零时后两者不得增加。
- 开发容器证据：同步后 P3/P2 定向测试 `125 passed`；使用容器内真实
  `model.ckpt-safestable-648278.pkl` 完成 P2→P3 warm start、一次低层 PPO update、保存与 exact
  resume，确认 `applied_updates=1`、Actor/Critic 变化和动作方差恢复。随后 1 环境真实 Isaac
  `p3_env_smoke.py` 在 180 秒内未完成启动且未输出 PASS，wrapper 超时后残留的两个父子进程已
  `kill -9` 回收。因此状态仍为“代码已修复待真实环境验证”，不得据此创建正式训练任务。
  2026-07-31 更正：重启容器后以独立日志重跑同一 1-env smoke，约 13 秒完成 reset 和
  3 个 step，明确输出 `status=PASS`。Policy observation 为 `[1,57905]`，critic wire 为
  `[1,385]`，policy/critic/worker aux 有限值比例均为 `1.0`；末帧 reward
  `-0.01940007507801056`，无 termination/truncation，EnvMonitor 最终上报 237 个指标，
  `abnormal=0, timeout=0`。功能验证已通过，但 vGPU workaround 输出调用 `os._exit(0)`
  后父子进程未自动退出，已定向 `SIGKILL` 回收并确认无 `p3_env_smoke.py` 残留。
  该退出异常是 smoke/Isaac 关闭路径问题，不影响上述 reset/step 证据；当前仍未覆盖
  64-env、80 帧低层 update 或 32-tick 高层 PPO/Adapter 联合更新。
  2026-07-31 追加验证：将 `p3_joint_rollout_smoke.py` 扩展为单进程 `integrated`
  场景，修复阶段 reset 返回未定义 `obs/critic_wire` 和 Adapter cadence 校验缺少
  `_high_adapter_update_due` 导入两个仅影响 smoke 工具的错误。开发容器使用真实
  `648278` 父 checkpoint、`64 env`、完整 Standard terrain 在同一 Isaac 进程中完成：
  80 帧低层 rollout 与 PPO update、32-tick/320-frame 高层 rollout 与 PPO update、一次
  Adapter update、P3 save 和 exact resume。最终 `status=PASS`，`low_updates=1`、
  `high_updates=1`，低层实际应用 20 个 minibatch update，高层 Actor/Critic/NavigationEncoder
  与 Adapter 均按职责变化，冻结低层 CNN 和高层阶段的低层模块均未变化。
  checkpoint 恢复模式为 `p3_exact_resume:exact_resume_history_reset`，完成 response records
  为 32。CUDA 峰值 `max_memory_allocated=857867776`、`max_memory_reserved=954204160`，
  pinned depth `235929600` bytes，无 OOM、无 non-finite skip。高层窗口内 2 个 failure 被
  EnvMonitor 记为 `abnormal=2`，是环境 episode outcome，不是训练进程异常。进程仍受
  vGPU 关闭路径影响未自动退出，已定向回收并确认无残留。
  同日进一步以 `128 env` 重跑相同 integrated/full-terrain 规格，再次
  `status=PASS`：80 帧低层 PPO、32-tick/320-frame 高层 PPO、Adapter update 和
  save/exact-resume 均完成，无 OOM、CUDA illegal access 或 non-finite skip。峰值
  `max_memory_allocated=1687526400`、`max_memory_reserved=1725956096`，约占 GPU
  `32.92%`；pinned depth 为 `471859200` bytes。从首次 reset 到 PASS 约 90 秒，
  端到端覆盖 400 个 vector step，约 `4.4 vector steps/s` 或 `568 env-frames/s`；
  高层 update 本身约 13.6 秒。低层收集中出现一次 warning-only
  `action_amplitude=6.5912>6.0000`，环境执行前仍裁剪到 `[-6,6]`；正式训练应继续
  监控 action saturation，但本次未造成更新跳过或有限值错误。
  根据该对照结果，用户选择将 P3 正式 TOML 的 `num_envs` 从 64 调整为 128；
  这是经实测的训练配置变更，不改变 observation、storage、PPO 或 checkpoint 合同。
  `commands.worker_progressive.enabled=false`，避免配置继续暗示存在 worker command override。
- 回归：新增校准推理不访问 critic 的测试、高层 Adapter 两轮一次的 cadence 测试；P3 定向回归
  当前 `50 passed`。容器中的原生命令 readback、阶段 reset、吞吐和真实 64 环境仍待在线验证。

#### 2026-07-31 审查补充：原生命令 epoch、死分支与恢复完整性

- 状态：本地已验证，开发容器待验证。
- 审查发现：P3 低层阶段虽然已改用平台原生命令，但写入 ResponseAdapter buffer 时仍把所有环境、
  所有帧的 `command_epoch` 固定为 0。平台命令按环境异步重采样时，0.2 秒 future label 可能跨越
  target 变化却仍被视为有效。高层 workflow 还保留 `capture_low` 分支，但低层与高层 phase 判定
  互斥，生产路径永远无法进入；这会继续暗示高层命令下仍可能更新低层。资源面板也只统计
  observation tensor，漏掉 critic/action/return/anchor/recurrent hidden 等 storage。容器 smoke
  则在正式 `high_update_interval=2` 下要求首个 high rollout 必须更新 Adapter，测试合同自身矛盾。
- 根因：撤回 BaseEnv command override 后，部分旧 joint-rollout 元数据和测试假设没有同步删除；
  平台原生命令又没有公开 epoch，因此不能直接复用 P2 单一 command controller 的标量 epoch。
- 修复：workflow 根据 observation 中的三轴平台命令为每个环境独立维护 epoch，只有该环境 target
  实际变化时递增；reset/resume/阶段环境 reset 清空 live tracker。`patch_owned_commands()` 同时支持
  标量和逐环境 epoch。高层执行器改为纯冻结低层 inference，删除不可达的低层 metadata、storage、
  return 和 optimizer 分支，并移除每个环境帧无副作用的 `agent.learn(None)` 调用。新增统一
  storage tensor byte 统计；smoke 明确临时使用 interval=1 以在一次昂贵 Isaac rollout 中覆盖
  high-policy→Adapter 边界，生产 TOML 仍为 interval=2。P3 exact resume 现在要求顶层
  `phase_label`、global `compound_schedule_phase` 与 `session_effective_seconds` 推导阶段三者一致。
- 本地验证：P3 contract/schedule 与 P2 邻近定向测试 `123 passed`；Python 编译与
  `git diff --check` 通过，`server/isaac_env/base_env.py` 仍与 HEAD 完全一致。开发容器尚未同步本次
  epoch/dead-branch/smoke 修复，不能升级为平台已验证。
- 防复发：回归固定覆盖逐环境 epoch 独立递增、tensor epoch shape 拒绝、完整 storage bytes 去重、
  Adapter 两轮生产 cadence、exact-resume 阶段/时钟不一致拒绝。正式训练仍不得恢复 17 维 action、
  worker command override 或高层阶段低层 PPO 更新。

## BUG-20260731-002：P3 评估被路由到 lbc_loco 且旧 loader 不搜 highslow 标签

- 日期：2026-07-31；状态：本地已验证，开发容器与平台双评估待验证。
- 影响：P3 `highslow` 等阶段包的 Standard+Camera 与 Track+Camera 平台评估；不影响训练、
  checkpoint 参数或部署契约。
- 首次发现：评估日志
  `/Users/nanbloom001/Downloads/log-599578-18560427.zip`，任务 `599578` / 运行 `18560427`，
  请求模型 ID `852198`，包内 `eval_model_id=884257`；模型包
  `archive/代码存档/p3nav8h-r1_884257.zip`。
- 用户可见症状与原始证据：aisrv 依次打印
  `[eval] Override Config.CURRENT: p3_standard_joint -> lbc_loco (inferred from TOML task_name)`、
  `Stage: lbc_loco, task_type: standard`，随后旧 LBC loader 构建 VisionEncoder/DmEncoder/Teacher
  Actor，`load_model_by_source() Exception [LBC-Loco eval] No same-ID visual checkpoint found ...
  candidate_order=[...model.ckpt-responsecalib-852198.pkl', ...]`，候选不含任何
  `highslow/highadapt/...` P3 标签；`exploit() RuntimeError Exception` 后
  `'NoneType' object is not subscriptable`。模型参数有限且完整（checkpoint SHA256
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`），失败在装配与
  发现，不在权重。
- 根因：`_infer_stage_from_task_name` 在 eval TOML 没有显式 `policy_entry` 时对
  Standard+Camera 无条件返回 `LBCLocoConfig`（第 743 行），P3 血缘在此分支被
  `Config.CURRENT=P3StandardJointConfig` 掩盖；旧 LBC loader 的
  `visual_eval_checkpoint_candidates`/`vision_checkpoint_candidates` 只搜
  `responsecalib/command/anchor/rl/vision` 等视觉标签，从不搜 P3 的
  `lowbase...highslow` 标签。两者叠加导致请求 ID 下无候选、checkpoint 未加载后仍尝试
  `exploit()` 并用 None 模型触发二次异常。
- 排除项：不是 checkpoint 完整性门禁"过严"；失败前的 LBC loader 拒绝未知模型是正确防线。
  也不是模型损坏——真实 `highslow-884257.pkl` 反序列化、模块/spec/shape、有限值全部通过。
  不修改平台托管的 `server/isaac_env/base_env.py`。
- 修复：新增两个显式评估入口 `p3_standard_eval` 与 `p3_track_eval`（`P3StandardEvalConfig`/
  `P3TrackEvalConfig`），在 `_valid_explicit_policy_stage` 注册并在 eval 下把显式
  `p3_standard_joint` 按地形模式重映射到对应入口；无显式入口时按 `Config.CURRENT` 血缘选择
  （Track+Camera→`p3_track_eval`、Standard+Camera→`p3_standard_eval`，历史 P2/Nav/视觉血缘
  行为不变）。两个入口共用 `p3_standard_joint_eval_candidates`（P3 标签优先级
  `highslow>highadapt>adaptercalib>lowfull>lowmedium>lowmild>lowbase`，无同 ID 时唯一
  discovery，多候选明确报歧义）与 `validate_p3_eval_bundle`（format/stage/phase/model_spec/
  模块 class/spec/有限值；请求 ID 与 payload ID 不一致只告警）。Standard 只装低层
  VisionEncoder+Actor77（obs 57901），Track 只装低层+NavigationEncoder+三轴 Actor+
  ResponseAdapter（obs 57905），两者都不创建 Critic/SafetyHead/optimizer/scheduler/训练
  buffer；Track 复用 P2 `AlgorithmP2NavPPO(training=False)` 的 5Hz/50Hz 机器与
  terminal-return bridge，`goal_reached` 继续进入平台 scorer。选中文件缺失/损坏/结构不兼容
  一律硬失败，不落到 P2/LBC/随机权重。`feature/__init__.py` 把 `p3_standard_eval` 映射到
  `LBCObservationProcess`（57901，无 P3 goal provider）、`p3_track_eval` 映射到
  `P2PolicyObservationProcess/P2CriticObservationProcess`（57905 + eval transport）。
- 修改文件：`server/agent_ppo/checkpoint_io.py`、`conf/conf.py`、`agent.py`、
  `algorithm/algorithm_p2_nav_ppo.py`、`feature/__init__.py`、
  `tests/test_nav_stage_and_metrics.py`、`tests/test_p3_eval.py`（新增）、
  `server/CHANGELOG.md`、`shared/interfaces/server-deploy-contract.md` 与本台账。
- 验证：`PY311test`（Python 3.11.13、PyTorch 2.11.0）新增 P3 eval 专项
  `25 passed`，覆盖候选优先级/唯一 discovery/歧义/不落 P2/LBC 标签、validator 六类拒绝、
  路由显式/血缘/重映射、真实 `highslow-884257` Standard 低层抽取与有限 12 维前向、
  Track 完整层级装配且 optimizer/scheduler/rollout/ResponseBuffer 全为 None、12 帧跨
  5Hz tick 边界动作有限、同一 checkpoint 顺序加载进两个独立装配、损坏 checkpoint 硬失败。
  排除两个已知失效旧模块后的训练端回归 `336 passed, 3 subtests passed`。随后同步修正
  `test_p3_contract.py::test_p3_production_config_and_monitor_are_standard_specific` 中遗留的
  `num_envs=64` 期望，使其与已完成 128-env integrated smoke 的正式配置一致；P3、P2 邻近、
  cleanup 与 uploader 定向回归为 `186 passed, 5 skipped, 4 subtests passed`。Python 编译、
  TOML 解析、`git diff --check` 通过；开发容器与平台双评估未执行，状态不得提升为
  "评估已验证"。
- 防复发：任何新阶段标签（含 P3）必须同时补 candidate-order 测试；Camera eval 不得在
  未加载 checkpoint 时推理；评估入口必须显式或按血缘选择，禁止 Standard+Camera 静默回退
  `lbc_loco`。评估验收依次确认 `Stage: p3_standard_eval/p3_track_eval` → selected 为
  highslow → `eval_disposition` → loaded modules → 无 lbc_loco/nav_eval 回退 →
  Standard scorer 正常统计、Track 完成数非恒 0。
- 血缘：分支 `codex/p3-dual-eval`（自 `codex/p3-standard-joint-recovery` 创建）；实现提交
  `6fb20ad`（`feat(server): add P3 standard joint recovery and dual eval`）；父 checkpoint
  `highslow-884257`，SHA256
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`；P3 父模型
  `p2nav2h-r2_648278`。
- 回滚：恢复 eval 路由到旧 `lbc_loco`/`nav_eval` 映射即复现本 Bug；仅需回滚评估装配代码，
  不修改 checkpoint 或训练。再次遇到的最短检查路径：最早 `Stage:` → 是否 `p3_standard_eval`
  或 `p3_track_eval` → 同 ID P3 candidate（highslow...lowbase）→ `eval_disposition`/
  loaded modules → 有限前向；禁止从末尾 `NoneType` 反推模型损坏。
- 2026-07-31 Track 首评失败补充（状态仍为"代码已修复待本地/开发容器复核，平台双评估待验证"）：
  用户上传 v1 修复 ZIP 创建 Track+Camera 评估（任务 `599777` / 运行 `18564415`）。路由已正确
  进入 `Stage: p3_track_eval`、`[P3] Track eval-only assembly initialized`，`save_model id=0`
  正确跳过；但 worker 首次 env reset 失败，aisrv 报 `episode 0, reset failed`。worker
  `base_env_process` 栈顶为
  `agent_ppo/feature/p2_observation_process.py:43 -> p2_response_aux(self.env)` →
  `p2_worker_bridge.py:557 raise RuntimeError("P2 response aux requested while bridge is disabled")`。
  根因：`P2WorkerBridge._resolve_config()` 的启用集合只有
  `{"p2_nav_ppo","p2_nav_eval","p3_standard_joint"}`，不含新 `p3_track_eval`；worker 的
  `P2CriticObservationProcess.process()` 无条件调用 `p2_response_aux`，bridge 被禁用即抛错。
  与 BUG-20260731-001 训练期"P3 critic observation 复用 P2 response transport 被 stage 闸门
  拒绝"同类。修复：`_resolve_config` 把 `p3_track_eval` 纳入启用集合与 eval 模式集合，配置从
  `[p3_standard_joint]` 读取；`track_diagnostics_enabled` 与 `_terminal_safe_root_pose` 保持
  P2 eval 语义（不启用 P3 训练专属的 curriculum probe / terminal-safe root pose）。新增回归：
  `p3_track_eval` eval 下 `_resolve_config` 返回 `enabled=True` 且读 `[p3_standard_joint]`、
  `_worker_stage_type=p3_track_eval`；`p3_standard_eval`/`lbc_loco` 仍返回 disabled。
  本地 P3/P2/Nav 定向回归 `164 passed, 5 skipped`（skipped 为真实 checkpoint 未暂存）。
  修复 ZIP 升级为 `p3nav8h-r1_884257-evalfix-v2.zip`，checkpoint 仍逐字节不变
  （SHA256 `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`）。

### 2026-07-31 补充：Track eval worker bridge stage 闸门遗漏

- 状态：本地已验证，平台 Track 复评待验证。
- 复现：平台 Track 评估任务 `599777` / 运行 `18564415` 已正确进入 `p3_track_eval`，但首次
  reset 在 `P2CriticObservationProcess -> p2_response_aux()` 报
  `RuntimeError: P2 response aux requested while bridge is disabled`，尚未进入 checkpoint 推理。
- 根因：双评估入口已注册到 agent 与 observation 路由，但 `p2_worker_bridge._resolve_config()`
  的启用/eval 白名单仍只包含 `p2_nav_ppo`、`p2_nav_eval` 和 `p3_standard_joint`，遗漏
  `p3_track_eval`；即便只补启用项，旧配置分支也会让它误读 `[p2_nav_ppo]` 而不是
  `[p3_standard_joint]`。
- 修复：仅为 `p3_track_eval` 启用共享 worker response transport，并让
  `p3_standard_joint/p3_track_eval` 共同读取 P3 配置；`p3_standard_eval` 与历史 `lbc_loco`
  继续禁用该桥，避免 Standard 评估无意装配 P2/P3 Track aux。新增回归覆盖以上三个边界。
- 修改文件：`server/agent_ppo/feature/p2_worker_bridge.py`、
  `server/agent_ppo/tests/test_p3_eval.py`、`server/CHANGELOG.md` 与本台账；checkpoint、网络权重、
  eval action contract 和平台托管 `base_env.py` 均未修改。
- 防复发：新增任何复用 P2 Track observation/transport 的 eval stage 时，必须同时验证 agent
  装配、feature 路由、worker bridge stage gate 和配置 section；只验证 `Stage:` 或 checkpoint
  候选不足以证明首次 reset 可用。

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
- warning-only 身份提示：P2 候选按平台请求 ID 优先，但允许配置父包和同类文件发现兜底；
  包内 ID/lineage/digest 属追溯元数据，缺失或不一致只告警。文件不存在、反序列化失败、必需模块缺失、
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

## BUG-20260731-003：P3 局部目标与 Standard 3.9m 完成口径脱节且低层阶段 Adapter 未更新

- 日期：2026-07-31；状态：本地已验证，开发容器与平台待验证。
- 影响：分支 `codex/p3-sim2real-radial-nav`，任务 `p3std2h30-s2r-radial`，父包
  `p3nav8h-r1_884257`。本轮不修改平台覆盖的 `server/isaac_env/base_env.py`。
- 用户可见症状：旧 P3 随机生成 1.5-2.8m 局部目标，完成后继续随机采样，无法保证机器人相对
  真实出生点走到 Standard scorer 的 3.9m 走穿半径；局部完成数、平台完成数和超时无法对齐。
  同时低层阶段只向 ResponseBuffer 写 records，optimizer step 后未调用 Adapter update，面板长期
  显示 `adapter_updates=0`。旧 `p3_frontier_clawback` 还是整个 rollout 的原始求和，纵轴异常大。
- 根因：局部目标状态只跟踪 goal distance，没有 episode origin、径向历史最佳和平台完成公式；
  worker 对 terrain/goal distance 做了 reset 前快照，但 root pose 仍使用 reset 后新 episode 状态。
  Adapter 的 deferred-update 只在高层每两轮路径解除，低层 update/version 边界没有实际 optimizer
  调用。面板把累计 settlement 总和与单 tick reward mean 混在同一尺度。
- 修复：按 `terrain_width/2-0.1m` 统一 M3 proxy；默认 8m 地块使用 3.90m 判定、3.93m 控制目标、
  4.00m 边界。新增 M1 1.3-1.6m、M2 2.5-2.9m；正常晋级保持当前方向，timeout 仅允许相对原
  方向左右 30/60 度重规划。径向 new-best 只奖励历史最大半径新增部分；最后 0.20m/0.04m
  只裁剪向外速度，达到阈值后命令归零。P3 worker aux 的 root pose 在 reset 行改用上一帧旧
  episode pose，wire shape 不变，P2/评估路径保持原行为。低层 PPO 真正完成 optimizer step 后先
  更新 low digest/version 并清 history，再执行 Adapter 1 次，lowmedium 执行 2 次；adaptercalib
  每轮 4 次，高层后段每两轮 1 次。新增 attempts/applied/skipped 面板；frontier settlement 改为
  有效 timeout 的加权均值。
- 排除方向：4.2m 已超过默认 8m 地块的 4.0m 边界，不能用作安全目标；未通过修改 BaseEnv
  强行增加 termination，也未把模型 ID/lineage 设为单点门禁。未验证的 push、action delay/gain、
  PD、COM 和相机随机化没有写成已生效合同。
- 本地验证：P3 contract/schedule/eval 定向测试 `57 passed, 5 skipped`；加入 Adapter 更新与
  P2 邻近回归后 `154 passed, 5 skipped`；除仓库已知失效 J9/ST9 旧测试外的完整测试为
  `339 passed, 5 skipped, 3 subtests passed`。覆盖 3.90/3.93/4.00 半径、径向奖励不可重复、
  向外软制动、reset 前 root pose、M1 到 M2 方向连续、timeout 角度集合、proxy/platform 同帧一致和
  Adapter attempts/applied/skipped。使用真实 `p3nav8h-r1_884257.zip` 中
  `model.ckpt-highslow-884257.pkl` 完成 radial warm start、一次低层 PPO update、P3 保存和 exact
  resume；Actor/Critic 均发生有限更新，动作方差与低层 digest 正确恢复。Python 编译、全部 TOML
  解析与 `git diff --check` 通过；本机缺少平台 `kaiwudrl` 包，monitor 的真实 builder 实例化留待容器。
- 关联：基线提交 `1a5641d`，本轮实现提交 `4230fc8`；PR、新 checkpoint、容器同步、平台任务与
  评估结果均尚未产生。回滚方式是切回 `codex/p3-dual-eval`；再次遇到时最短检查路径为 worker reset 行
  root pose → M1/M2 event → 3.90m proxy → platform reason/completed → Adapter attempts/applied。

## BUG-20260731-004：P3 径向续训缺少步态专项闭环，Adapter/冻结/奖励与双评估合同不一致

- 日期：2026-07-31；状态：本地已验证，开发容器、平台 smoke 与评估待验证。
- 影响：分支 `codex/p3-sim2real-radial-nav`，任务 `p3std2h30-gait-radial`，父包
  `p3nav8h-r1_884257`。本轮保持低层 observation 57901、动作 12 维和部署接口不变，不修改平台
  覆盖的 `server/isaac_env/base_env.py`，模型 ID/标签/lineage 不作为结构正确时的单点门禁。
- 用户可见症状与审查证据：低层评估视频反复出现右后腿低速拖行、右前腿外摆、楼梯硬触地和
  后退交叉落脚；原计划的固定 gait envelope 无法按地形/运动区分这些事件。代码审查同时发现：
  Adapter 真更新后又被 deferred 空调用把面板覆盖为 0；M3 worker runtime terrain size 与 aisrv
  固定 8m 形成双数据源；`adaptercalib` 虽 LR=0 仍执行 Actor Adam step；局部进度仍复用 P2
  frontier；Adapter replay 没有按低层版本分层。`codex/p3-dual-eval` 的 `09889b1` worker bridge
  修复也尚未进入当前分支祖先。
- 根因：P3 低层 rollout 只有 compact PPO/anchor 路径，没有训练专属 contact onset、触地前速度、
  body-frame touchdown-y、continuous stance、joint torque/power 和 runtime terrain size transport；
  gait probe 只能监控，不能形成正常区间为零的事件奖励。旧阶段控制只调整 LR，未冻结 optimizer
  state；Adapter buffer 仍按 P2 的 current/parent 两池采样；高层 reward 面板同时展示包含项和子项，
  造成看似无法闭合的奖励分解。
- 修复：P3 训练 worker wire 扩展为 `critic323 | P2 aux62 | P3 extra35 = 420`，仅
  `p3_standard_joint` 使用；Track/Standard eval 不消费该 training-only 尾部。新增 worker 公开
  `base_velocity` 写入/readback 的 2-8 秒三轴恢复采样器，后退 `[-0.25,-0.05]`、左右转向和横移
  严格对称。前 15 分钟冻结低层 Actor，按正/逆坡、正/逆楼梯 × 低速/前进/后退/转向横移采集
  健康基线；单桶不足依次回退同地形、全局，全局仍不足则对应奖励禁用。之后渐进启用接触
  滑移/冲击、contact-onset 交叉落脚、1.5 秒步态饥饿，旧 air-time/duty/participation shaping
  权重归零。25% 完整 16-step TBPTT sequence 使用镜像一致性，CNN feature 由冻结 CNN 对水平
  翻转 depth 计算；loss 只连接低层 LSTM、RNN 输出层和最终 action head，梯度比例每 PPO epoch
  首 minibatch 标定到 3.5%，硬上限 10%。
- 其他闭环：删除重复 Adapter 空更新；replay 改为最新低层版本 50%、近期 P3 25%、父 records
  25%，记录版本 lag；worker tail 的 runtime terrain size 成为 M3 proxy/目标/软制动共同来源；
  `adaptercalib` 完全跳过 Actor backward、optimizer 和 scheduler step；高层改用正 1.5×、负 0.4×
  的 duration-normalized 局部进度，平台 success terminal 保留本 tick 局部/radial 进展，failure/
  timeout 不领取 radial new-best。奖励面板只展示互不重叠加权项，storage reward 另作总和核对。
  checkpoint 保存未完成 baseline reservoir、冻结后的分桶阈值/digest、mirror RNG/进度、Adapter
  三池和 runtime M3 合同；评估忽略全部 training-only gait 状态。
- 双评估：保留 `p3_standard_eval` 低层-only 与 `p3_track_eval` 完整层级装配；Track worker bridge
  启用 `[p3_standard_joint]` 的反馈/步态 transport，Standard eval 和旧 LBC 不启用。SafetyHead、
  gait baseline/mirror、Critic、optimizer、scheduler 和训练 buffer 均不进入 eval。
- 本地验证：P3 gait/radial/phase/eval、P2 core/nav observation 与 P1.5 邻近回归共
  `215 passed, 5 skipped`；5 个 skip 均为真实 `highslow-884257` checkpoint 未暂存的集成测试。
  覆盖 mirror 双次恢复、显式 joint order、命令范围/符号平衡、分桶基线回退与未完成 reservoir
  resume、事件奖励独立触发、Adapter calibration Actor step 禁用、P3 Track worker bridge 和双评估
  装配。Python 编译、全部 TOML 解析与 `git diff --check` 通过；本地 fake builder 验证 P3
  44 个面板且每个 line 面板最多 20 指标。平台真实 Monitor builder、884257 联合 rollout/update/
  save/resume 和 15-30 分钟 smoke 尚未执行，不能升级为平台已验证。
- 防复发与回滚：新增阶段标签必须同步 checkpoint candidate 和 round-trip 测试；P3 wire 维度只可
  在训练 stage 扩展；足端/ContactSensor/joint mapping 或 tensor shape 异常时关闭镜像及步态奖励并
  告警。回滚可禁用 gait fraction 和 worker sampler、恢复旧 P3 contract，不需要改变 checkpoint
  低层/高层模块；再次遇到时最短路径为 `worker wire shape -> mapping valid -> baseline samples/
  fallback -> mirror gradient ratio -> gait reward terms -> Adapter pool ratios -> Standard/Track eval stage`。
- 关联：父 checkpoint `p3nav8h-r1_884257`；实现 commit `49b0cb5`，双评估 merge commit
  `0b03a8b`（含 `codex/p3-dual-eval` 的 `09889b1`）。PR、新 checkpoint、容器同步与平台任务
  均待产生，产生后追加，不得预填。

- 2026-07-31 审查更正（状态：代码已修复待开发容器验证）：后续独立审查确认首版仍有三处
  训练语义漂移。第一，`JOINT_MAPPING_VALID` 只证明 joint order，未证明 action scale、PD、
  effort limit 左右镜像一致；第二，touchdown-y 只按 yaw 旋转，坡面和楼梯的 roll/pitch 会污染
  body-frame 横向落点；第三，slip 使用 1.5 秒接触帧平均速度，并非计划中的 per-stance budget。
  修复后 preflight 同时验证 joint order、action scale、stiffness、damping、effort limit 与明确的
  `contact_forces` 映射，异常只关闭 mirror/gait 专项并输出检查明细；touchdown-y 使用完整四元数
  逆旋转；slip 改为每个 stance 累计世界 XY 距离，窗口输出 stance 最大值。checkpoint 额外保存
  并 exact-resume 恢复运行时 `terrain_size/platform_complete/m3_target/boundary`，同时明确记录
  worker sampler 因 worker/aisrv 无公开反向状态通道，只能在环境 reset 后 fresh 初始化，不能
  虚构 exact sampler resume。新增定向测试覆盖 roll=90° 的 body-y、stance 断开重启、PD 不对称
  fail-safe 和正负速度条件统计；开发容器与平台 smoke 尚未执行。再次遇到时最短检查路径为
  `P3MirrorPreflight checks -> contact mapping -> touchdown quaternion -> stance slip distance -> M3 runtime contract`。

- 2026-07-31 开发容器验证补充（状态：开发容器 1-env 已验证，真实父包联合更新仍待验证）：
  初次 1-env Isaac smoke 正确暴露平台实际 joint 顺序为 axis-major：`FL/FR/RL/RR hip`、
  `FL/FR/RL/RR thigh`、`FL/FR/RL/RR calf`；首版镜像置换错误地按 leg-major 编排，导致
  `P3MirrorPreflight joint_order=False` 并安全关闭 gait/mirror 训练。修复 `JOINT_SWAP` 为
  `[1,0,3,2,5,4,7,6,9,8,11,10]`，hip 四轴取负、thigh/calf 保持符号，同时修复
  `robot.data.joint_names=None` 时向 `robot.joint_names` 回退。容器定向测试 `11 passed`；复跑
  1-env Camera smoke 得到 `P3MirrorPreflight enabled=True`，全部 joint/PD/effort/action-scale/contact
  检查为真，policy shape `57905`、critic wire `420`、三帧 finite step 与最终 `status=PASS`。
  容器未找到父 checkpoint `p3nav8h-r1_884257`，因此本轮未完成真实父包的低层 PPO、镜像
  backward、Adapter、高层 update 与 save/resume，不能据此升级为平台训练已验证。

- 2026-08-01 真实父包验证补充（状态：开发容器联合 smoke 已验证，平台任务待验证）：上传
  `p3nav8h-r1_884257-evalfix-v2.zip`，容器端 zip SHA256
  `7722cf9f919a08a2e5c7ec06839bc36d2da740f753ad1a947bb16ead452955ab`，只提取真实
  `model.ckpt-highslow-884257.pkl`，未用包内旧源码覆盖当前分支。真实 checkpoint 双评估回归
  `28 passed`；父包 smoke 验证 gaitcalib 阶段 Actor 不变/Critic 更新、lowbase 阶段 Actor/Critic
  均更新、action std 恢复及 exact resume，退出码为 0。4-env Isaac integrated smoke 完成同一
  环境中的 80 帧低层 rollout/PPO、32 tick 高层 PPO、Adapter update 和 save/exact-resume：低层
  阶段仅 low LSTM/Actor/Critic 与 Adapter 变化，高层模块冻结；高层阶段仅 NavigationEncoder、
  high Actor/Critic 与 Adapter 变化，低层全部冻结；生成 32 条 completed response records，
  `status=PASS`。峰值 CUDA allocated/reserved 分别约 325/362 MiB（4-env smoke 口径）。测试过程
  同时修复两个仅影响 smoke 的旧合同：阶段边界由旧 8 小时时钟改为当前
  `5400/6000/7200s`，integrated 低层更新从 Actor 可训练的 `901s` 开始，并允许低层阶段按计划
  更新 Adapter。1-env integrated 因 5 条 recurrent sequence 无法整除 4 minibatch 被明确拒绝，
  改用合法 4-env 规格后通过；这不是正式训练路径故障。

## BUG-20260801-001：P3 低速实机震荡缺少确定性均值约束且命令 anchor 桶语义错位

- 日期：2026-08-01；状态：本地已验证，开发容器、平台 smoke 与实机复测待验证。
- 影响：分支 `codex/p3-sim2real-radial-nav`，任务 `p3std2h30-gait-smooth-radial`，父包
  `p3nav8h-r1_884257`。保持低层 57901 输入、12 维动作和部署接口不变；不修改平台覆盖的
  `server/isaac_env/base_env.py`，本轮 `push_robots=false`。
- 用户可见症状与证据：实机部署出现高频震荡。训练代码只对 sampled environment action 施加
  饱和的 rate/jerk frame cost，而部署主要执行 deterministic policy mean；训练还保存 raw sampled
  action/log-prob，却向环境执行 `clamp[-6,6]`。审查同时确认 P3 bucket 0 是 low-forward、bucket 6
  才是 zero，但继承的 `_sample_hold()` 把 bucket 0 当 zero；worker 已持有准确 bucket/anchor，aisrv
  低层 transition 却无条件写入 `anchor_weights=1`，会把父模型低速震荡过度锚定。
- 根因：P3 sampler 复用了旧六桶的 zero/anchor 语义并使用全局 Torch RNG；worker-to-aisrv wire
  没有传递 committed bucket/anchor。现有 frame reward无法区分策略均值的确定性高频变化和
  Gaussian exploration noise，且超过归一化阈值后很快饱和。
- 修复：P3 sampler 使用独立 seeded generator，bucket 6 使用 zero hold；anchor 映射固定为
  low-forward 0.10、forward 0.50、joint-turn 0.25、lateral 0.05，其余为 0。P3 training-only
  worker extra 增加 bucket/anchor 两列，wire `420->422`，评估和部署不消费。低层 TBPTT replay
  对 deterministic mean 对应的关节目标计算 terminal-safe rate/jerk Huber 超限项，并增加
  `|mean|>5` 软范围约束；辅助前向 detach Actor 中间 MLP，只更新 LSTM、RNN output 和最终 head。
  smooth 目标梯度 3.5%、range 1%，与 mirror 合计硬上限 10%。增加 raw/exec action、clip rate、
  joint-target rate/jerk、15-25Hz power 和命令桶 clip/anchor 面板。
  同时将力矩 P50/P95/max 的 Hip/Thigh/Calf 索引改为平台 axis-major 的
  `[0:4]/[4:8]/[8:12]`，避免旧 leg-major 索引混合三类关节。
- 排除方向：本轮未加入部署端滤波、bounded/tanh Gaussian、command ramp、action/feedback delay、
  PD/电机/传感器偏置或 depth temporal 随机化。外力 push 虽由平台公开事件支持，但当前无法表达
  部分环境和 reset grace，且会污染前 15 分钟健康 gait baseline，因此继续关闭；后续应单独以
  25-30 秒间隔、0.15-0.20m/s 小扰动做抗扰恢复实验。
- 本地验证：P3 gait/radial/contract/eval 与 P2 core 邻近回归 `226 passed, 5 skipped`；排除既有
  J9/ST9 失效测试后的广泛回归为 `360 passed, 5 skipped, 3 subtests passed`。新增测试覆盖固定
  seed、zero bucket hold、七桶 anchor、reset 边界平滑屏蔽、正常范围零损失、极端 mean 软约束、
  20Hz 频谱识别、Actor 中间层梯度隔离、worker bucket/anchor wire 和 axis-major 力矩分组。Python
  编译、全部 TOML 解析和 `git diff --check` 均通过。使用真实父包
  `p3nav8h-r1_884257-evalfix-v2` 完成 warm-start、gaitcalib update、lowbase update、保存和 fresh
  exact-resume；平滑辅助梯度估计比例为 `0.0350000001`，未超过 10% 总上限。5 个 skip 为需要
  特定平台装配的集成场景；上述结果仍不能替代开发容器 Isaac、平台 smoke 或实机验证。
- 回滚与最短检查：将 smooth fraction 置 0、恢复 P3 extra 35/worker wire 420 和全 1 anchor 即可
  回滚训练侧变化，不需要改 checkpoint module 或部署端。再次遇到时依次检查
  `command bucket/anchor -> action_mean 15-25Hz -> raw clip rate -> joint target rate/jerk -> torque/roll`。
  关联 commit/PR、新 checkpoint、平台任务和实机结果均尚未产生。

- 2026-08-01 审查更正（状态：本地已验证，开发容器、平台 smoke 与实机复测待验证）：独立审查
  发现首版仍有四处语义或性能缺口。辅助 multiplier 只在每个 epoch 的首 minibatch 标定，随后
  沿用旧梯度范数；三项 ratio 又以标量相加近似合成范数，因此所谓 10% 上限和面板值都不一定
  对应实际 optimizer step。现改为每个 minibatch 重新取得 mirror/smooth/range 梯度，先按目标
  比例缩放，再对实际合成向量统一裁到 PPO policy gradient 的 10%；无 PPO 或无辅助梯度时该项
  multiplier 为 0。worker command 状态缺失及 aisrv wire 非法值的 anchor 回退从 1 改为 0，并
  输出一次 warning，避免 transport 故障时重新过度锚定。joint/PD/effort/action-scale 属于环境
  装配期静态合同，现只在 `P2WorkerBridge` 初始化（或旧测试对象首次调用）核验一次，环境重建会
  自然创建新 bridge 并重验，不再 50Hz 重复执行 GPU 同步。`target_probability=1` 的 P3 sampler
  跳过无意义的全局 `torch.rand`，全部实际命令随机性继续由专用 generator 提供。新增回归覆盖
  合成梯度实际范数、全局 RNG state、静态 preflight 缓存与 command-state 缺失回退；P2/P3
  定向与邻近回归合计 `186 passed, 5 skipped`，5 项均为缺少指定真实 checkpoint 的既有集成
  skip。Python 编译、9 份 TOML 解析和 `git diff --check` 通过。开发容器和平台验证将在后续
  结果中追加；旧 checkpoint 不受影响，本轮训练必须生成新包后这些修复才进入模型。

- 2026-08-01 第二次审查更正（状态：本地已验证，开发容器、平台 smoke 与实机复测待验证）：
  审查确认首版 preflight 遍历 action terms 后取首个 scale，多 term 环境可能误验无关动作；逐腿
  mirror error 仍按 leg-major 连续三列切片，与容器确认的 axis-major 动作布局冲突；地形/运动
  步态桶把 gait-invalid 帧计入分母，并将稀疏 contact-onset impact 除以全部帧数；父包 smoke
  只比较 Actor 权重，没有发现 LR=0 时 Adam moments 仍可漂移。进一步追踪发现 P3 coordinator
  设置 `gaitcalib` 后，`AlgorithmVisualPPO.learn()` 又按旧 8 小时时钟把 0 分钟映射为 `lowbase`，
  生产路径同样存在策略 optimizer state 漂移，而非仅测试缺口。修复后只接受明确的 12 维
  `JointPositionAction`，并验证解析后的 scale 左右对称且等于训练合同 0.25；逐腿指标使用
  `FL[0,4,8]/FR[1,5,9]/RL[2,6,10]/RR[3,7,11]`；全部 gait 汇总屏蔽无效帧，impact/touchdown
  按各腿 onset 事件计数，分桶 impact 保存独立事件分母。P3 workflow 通过显式 `phase_override`
  把 coordinator 的 2.5 小时阶段传入低层 PPO；真实父包 smoke 同时比较 Actor/LSTM optimizer
  state digest，冻结阶段任何 Adam `step/exp_avg/exp_avg_sq` 变化都会失败。新增定向回归覆盖
  多 action-term 误选、0.25 合同、axis-major 逐腿误差、无效帧屏蔽和 impact onset 分母；P3 与
  P2 core 广泛回归 `182 passed, 5 skipped`，Python 编译、全部 TOML 解析和 `git diff --check`
  通过。真实 `model.ckpt-highslow-884257.pkl` 父包 smoke 进一步确认：gaitcalib 阶段
  Actor 权重不变、策略 optimizer state digest 不变且 Critic 更新；lowbase 阶段 Actor、策略
  optimizer state 与 Critic 均更新，随后保存/exact-resume 通过。平台镜像证据为 Go2
  `ActionsCfg.JointPositionAction scale=0.25`；镜像生成时间
  `2026-07-29T10:15:44.395342+00:00`，当前容器动态装配仍须下一次 Isaac smoke 复核。

- 2026-08-01 开发容器验证补充（状态：开发容器已验证，平台任务与实机复测待验证）：源码 bundle
  同步 15 个 P3 文件后，容器定向回归 `22 passed`。1-env Isaac Camera smoke 读取到真实
  `JointPositionAction` 12 维，`P3MirrorPreflight enabled=True`，joint order、stiffness、damping、
  effort limit、action scale 与 `contact_forces` mapping 全部为真；policy/critic wire shape 分别为
  `57905/422`，三帧 step 有限并输出 `status=PASS`。真实 `884257` 父包 smoke 再次确认 gaitcalib
  阶段 Actor 权重及策略 optimizer state 均不变、Critic 更新，lowbase 阶段三者正常更新并完成
  exact resume。4-env integrated smoke 完成 80 帧低层 PPO、32-tick 高层 PPO、Adapter update 和
  save/exact-resume，低层阶段只更新 low LSTM/Actor/Critic 与 Adapter，高层阶段低层全冻结；
  completed response records 为 32，峰值 CUDA allocated/reserved 约 `325/362 MiB`，最终
  `status=PASS`。Isaac vGPU workaround 遗留的测试子进程和 `/tmp` 日志已清理，父模型保留。

## BUG-20260801-002：WebIDE 上传器默认写入根目录过宽，可能覆盖源码或凭据

- 日期：2026-08-01；状态：本地已验证。
- 影响：未跟踪工具 `shared/tools/tencent_kaiwu_webide_upload.mjs`。用户可见风险是调用方省略
  `remoteRoot` 时，任意项目内路径都可通过校验，误传文件可能覆盖训练源码、配置或
  `conf/.env`；该工具不属于训练运行时。
- 根因：路径规范化只验证目标位于
  `/data/projects/legged_robot_competition_26/` 下，默认边界大于文档约定的
  `agent_ppo/test_artifacts/`，也没有独立的凭据/仓库元数据拒绝规则。
- 修复：默认 root 收窄到 `agent_ppo/test_artifacts/`。调用方即使显式扩大 root，仍拒绝 `.git`、
  `.env*`、credential/secret 文件及 `.key/.pem/.p12/.pfx`。保留显式 root 参数供受控实验使用，
  不把模型 ID、IDE ID 或路径标签作为单点门禁。
- 验证：后续 Node 定向测试扩展为 `10 passed`，`node --check` 通过，覆盖默认白名单、目录穿越、
  根目录目标、NUL 与扩大 root 后的凭据拒绝。已使用 31.46 MiB 真实 ZIP 完成容器上传、SHA256、
  原子改名和自动清理，摘要匹配；该证据不等同于训练平台或模型能力验证。
- 防复发与回滚：上传模型默认先落 `test_artifacts`，由容器内受控命令解压/移动；不得放宽默认
  root。若确需其他临时目录，调用方显式传 root，受保护路径仍不可覆盖。关联 commit/PR 尚未
  产生；该工具和调查文档目前仍为用户未跟踪文件，本任务未擅自暂存。

## BUG-20260801-003：WebIDE 同步服务自举误判鉴权 health 并解析终端回显

- 日期：2026-08-01；状态：本地已验证，开发容器 already-running 分支已验证，冷启动待验证。
- 影响：未跟踪工具 `shared/tools/tencent_kaiwu_webide_upload.mjs` 新增的
  `startKaiwuSyncService()`。首次真实测试中，已有 8765 服务被误判为未启动，工具额外尝试启动
  PID 2652；该进程因端口占用立即以状态 1 退出，原同步服务未停止、Token 与源码未变更。
- 症状与证据：首次返回对象同时出现互斥的 `alreadyRunning=true`、`started=true`、
  `scriptMissing=true`、`healthy=true`；真实 PTY 输出则是 `CODEX_SYNC_STATE=started`、
  `CODEX_SYNC_PID=2652`、`Done(1)` 和 `CODEX_SYNC_HEALTH=failed`。随后本地 `PY311test` 经
  `container_rpc_client.py --cwd . "pwd"` 仍成功返回项目根，证明原服务始终健康。
- 根因：容器 `/health` 需要 Token，未带认证访问返回 HTTP 401；探针使用 `curl -f`，把“服务有
  HTTP 响应”错当成“连接失败”。解析器又用 `output.includes(marker)`，而交互 PTY 会回显整条
  shell 命令，使四个 marker 即使没有作为独立结果输出也全部命中。
- 修复：健康探针移除 `-f`，只把连接/超时失败视为服务不可达；结果解析先去除 ANSI CSI，再仅
  接受独立整行 `CODEX_SYNC_STATE`、`CODEX_SYNC_PID` 和 `CODEX_SYNC_HEALTH`，取最后一个合法
  state/health。解析逻辑提取为 `parseKaiwuSyncBootstrapOutput()`，不再依赖模糊子串。
- 排除方向：不是 `start_tongbu.sh` 缺失或原进程退出。通过当前容器 RPC 读取脚本确认它加载
  `conf/.env` 后 `exec` Python 服务；本轮没有读取或输出 Token 内容，也没有主动停止 8765。
- 验证：Node 定向测试 `10 passed`、`node --check` 通过；新增回归包含命令回显中的伪 marker，
  只接受后续独立结果行。真实容器复测返回 `alreadyRunning=true`、`started=false`、
  `healthy=true`、`scriptMissing=false`、退出码 0；关闭 Chrome/PTY 后再次通过本地
  `PY311test` RPC 执行 `pwd` 成功。因不应破坏当前健康服务，尚未人为关闭 8765 验证冷启动。
- 防复发、回滚与血缘：任何 PTY 机器可读结果都必须使用独立整行解析；带鉴权 HTTP 端点的存活
  检查不得把 401 等业务状态等同于 TCP 不可达。回滚可删除 `startKaiwuSyncService()`，不影响
  上传、终端 SHA256 或现有 RPC。当前分支 `codex/p3-sim2real-radial-nav`；无 commit/PR、训练
  task、模型或 checkpoint 变更，工具与协议文档仍未暂存。
- 2026-08-04 更正：上述工具后来已由提交 `a6927b2` 正式保存，但当前
  `codex/p4-maze8h-attack` 工作树尚未包含该提交；可核验副本位于
  `/Users/nanbloom001/.codex/worktrees/793d/fwwb-Final/shared/tools/tencent_kaiwu_webide_upload.mjs`。
  本次一度按旧的 `/Users/nanbloom001/codespace/fwwb-Final/...` 绝对路径导入失败，并错误判断为
  “模块不存在”。正确最短检查路径是先用 `rg` 搜索全部工作树，再执行
  `git log --all -- shared/tools/tencent_kaiwu_webide_upload.mjs`，区分路径变化、分支未合入和真实
  缺失。使用真实路径后，冷启动分支返回 `started=true`、`healthy=true`，PID 1296；随后 RPC
  `pwd` 与 37 文件 bundle 同步成功，二次 dry-run 为 0 差异。该证据同时补齐此前“冷启动待验证”
  的运行证据，但不等同于训练或模型能力验证。

## BUG-20260801-004：P3 训练不可达负速度且稀疏步态基线跨地形误塑形

- 日期：2026-08-01；状态：本地已验证，开发容器、平台 smoke 和新 checkpoint 待验证。
- 影响：分支 `codex/p3-sim2real-radial-nav`，任务配置
  `p3std2h30-gait-smooth-radial`，计划父包 `p3nav8h-r1_884257-F2`。不修改平台覆盖的
  `server/isaac_env/base_env.py`，不改变 57901 低层输入、12 维动作、高层 Actor85 或部署接口。
- 症状与证据：捕获 `20260801-030946` 覆盖约 25 分钟，低层 mirror error 已下降约
  22%-40%，但 gait baseline
  fallback share 约 0.75；原 sampler 仍把 15% 样本分配给 `vx<0`，而现有高层命令映射只允许
  `vx>=0`，该恢复能力无法被高层调用。面板使用通用 `policy_loss/value_loss`，与高层同名指标
  覆盖；P3 Sim2Real 只显示合计项，无法确认 sustained torque、peak、action rate 和 jerk
  分别是否生效。
- 根因：命令采样方案沿用了尚未进入高层动作合同的后退设想；健康基线按四地形乘四运动分桶，
  15 分钟内 16 个桶大量不足，代码仍允许全局 fallback 直接产生训练惩罚。brake/zero 也会落入
  low-speed 基线，混入静止与切换瞬态。P3 monitor 沿用跨算法通用指标名，且奖励函数内部四项
  Sim2Real 原始分量未进入训练诊断 transport。
- 修复：`vx` 范围统一为 `[0,1.0]`，七桶 wire/checkpoint 布局不变但 reverse 权重固定为 0，
  任何自定义配置重新启用该桶均在启动时报错；有效分布改为低速前进 30%、前进 25%、联合
  转向 15%、横移 10%、brake/restart 15%、zero 5%。gait baseline 升级为 contract v2，运动桶
  改为低速/前进/转向横移，brake/zero 和意外负 `vx` 不采集也不受罚；exact threshold 权重 1，
  same-terrain fallback 权重 0.5，global fallback 和缺失项权重 0。checkpoint contract 同步记录
  新命令域、分布和 fallback 语义，旧状态只能显式 warm start。
- 课程更正：按本轮低层专项目标改为 0-120 分钟低层、120-130 分钟 Critic/Adapter 校准、
  130-140 分钟高层 Actor warm-up、140-150 分钟完整高层收尾。旧 P3 radial warm-start 会通过
  P2 reward loader 重建高层 Critic 和 return statistics，与 30 分钟适配预算冲突；现在加载策略
  和 Adapter 后，必须从父 P3 包恢复高层 Critic、Critic optimizer moments、return statistics、
  Actor/Critic gradient steps。缺少这些训练状态按结构不兼容报错，不依赖模型 ID 或标签阻断。
- 监控修复：低层/高层 PPO 分别输出独立 loss、entropy、KL、clip fraction、更新职责和耗时；
  增加命令桶 share、负 `vx` share、基线 exact/terrain/global/disabled 比例、机械功率 P50/P95/max。
  训练 worker tail 增加四项已有 Sim2Real 原始分量与 validity，extra `37->42`、privileged wire
  `422->427`。字段只供训练诊断，不进入 observation、reward、eval/export/deploy；奖励权重未改。
- 验证：本地 P3 gait/radial/schedule/eval/contract/monitor 与 P2 core 邻近回归
  `215 passed, 5 skipped`；5 项为缺少指定真实 checkpoint 的既有集成场景。Python 编译、13 份
  TOML 解析和 `git diff --check` 通过。新增测试覆盖负 `vx` 禁止、命令合同、全局 fallback
  奖励为零、Sim2Real validity 屏蔽和机械功率分位。开发容器尚未同步，不能标为平台已验证；
  当前进行中的旧任务不会因本地源码修改而改变。
- 真实父包本地 smoke：使用
  `archive/代码存档/p3nav8h-r1_884257-evalfix-v2/ckpt/model.ckpt-highslow-884257.pkl`
  完成 warm start、冻结低层更新、可训练低层更新、保存和 exact resume；逐值确认高层 Critic、
  Critic Adam moments 和 return statistics 均从父包恢复，输出
  `high_critic_restored/high_critic_optimizer_restored/high_return_statistics_restored=true`。
  这证明本地装配合同闭环，但不替代开发容器 Isaac 或平台 smoke。
- 回滚与最短检查：可恢复旧 command weights 与 gait baseline v1，并将 P3 extra/wire 恢复
  `37/422`；无需改变模型模块。再次遇到时依次检查 `target_vx_negative_share -> command bucket
  share -> baseline fallback levels -> gait reward -> sim2real component validity -> low/high PPO panel`。
  commit/PR、新 checkpoint、平台 task/model ID 均待产生；模型 ID 不作为单点硬门禁。

## BUG-20260801-005：P3 stance 滑移被重复扣分且映射失效后平滑辅助仍训练

- 日期：2026-08-01；状态：本地已验证，开发容器、平台 smoke、评估和真机均待验证。
- 影响：分支 `codex/p3-sim2real-radial-nav`，任务
  `p3std2h30-gait-smooth-radial`，父包 `p3nav8h-r1_884257-F2`。只影响训练专用 gait/aux
  transport；57901 低层输入、12 维动作、高层 Actor85、Standard/Track eval 与部署接口不变。
- 用户可见症状：P2 面板的 `*_slip_speed` 实际显示累计 stance distance，单位和历史口径漂移；
  P3 接触质量奖励在 stance 结束前读取 1.5 秒 ring 最大值，同一滑移预算会在最多 75 个 50Hz
  frame 中重复扣分。另有关节顺序、PD、effort、action scale 或 contact mapping preflight 失败时，
  mirror/gait reward 会关闭，但 deterministic-mean smoothing/range auxiliary 仍继续更新低层输出。
- 根因：P3 引入 per-stance slip budget 时直接复用了共享 `GAIT_SLIP_SPEED_SLICE`，并把 ring
  maximum 当成逐帧 reward 输入，没有独立的 stance-completed event。action-smoothing 课程函数
  也没有消费同一个 mapping-valid fail-safe。
- 修复：共享 P2 aux 恢复为接触帧平均足端 XY 速度；P3 私有 tail 增加四腿 completed-slip 和
  completed-event，baseline、reward 及地形/运动分桶只按事件样本累计。训练 extra/wire 从
  `42/427` 扩为 `46/431`；gait baseline 合同升级为 v3，旧错误语义状态只允许重新 warm start，
  不允许 exact resume。`action_smooth_training_fraction()` 新增 mapping-valid 输入，失效时返回 0，
  与 mirror/gait fail-safe 保持一致。模型 ID、请求 ID 和 lineage 仍不作为单点硬门禁。
- 本地验证：`PY311test` 定向执行 P2 core、P3 contract、P3 gait/radial、P3 schedule 和双评估，
  结果 `187 passed, 5 skipped`；5 项为缺少指定真实 checkpoint 的既有集成场景。回归覆盖 P2
  m/s 口径、stance completion 单帧事件、后续帧不重复奖励、P3
  分桶事件分母、wire `431` 维和 mapping-invalid smoothing=0。改动模块 Python 编译、9 份活动
  TOML 解析和 `git diff --check` 均通过。
- 容器/平台/评估/真机：均未执行，不能升级为平台已验证。关联 commit/PR、新 checkpoint、
  平台 task/model ID 与 SHA256：尚未产生。
- 回滚与最短检查：可回滚本条目的 P3 tail 和 baseline v3，但不得恢复“ring maximum 每帧扣分”
  逻辑。再次遇到时依次检查 `GAIT_SLIP_SPEED_SLICE` 单位、completed-event 单帧性、
  `gait_completed_slip_count`、contact reward frame 数和 `mirror_mapping_valid/action_smooth_scale`。

## BUG-20260801-006：P3 近场深度失效、短 TBPTT 与楼梯恢复过度塑形

- 日期：2026-08-01；状态：本地已验证，开发容器、平台 smoke、评估和真机均待验证。
- 影响：分支 `codex/p3-stairmem-dr`，任务 `p3std8h-stairmem-dr`，计划父包
  `p3nav8h-r1_884257-F2`（本地制品目录为 `p3nav8h-r1_884257-evalfix-v2`）。不修改平台覆盖的
  `server/isaac_env/base_env.py`，不改变 57901 低层输入、12 维动作、高层 Actor85、Standard/Track
  双评估或部署接口。模型 ID 和标签只用于选择与 lineage，不作为单点阻断。
- 用户可见症状：实机靠近楼梯时，小于约 0.25m 的深度回波不可靠，中央孔洞率会瞬时接近 80%；
  原 80 帧 rollout 配合 16 帧 TBPTT 只向低层 LSTM 提供约 0.32 秒反传历史，无法针对“远处看见
  楼梯、近处视觉退化后继续动作”建立连续记忆。上一轮同时启用较强动作平滑、接触/交叉/饥饿
  shaping 与 Sim2Real 代价后，评估出现楼梯抬腿、触地和跟进能力回退；真机还记录到前腿
  Hip/Thigh 力矩连续逼近 22Nm 硬线，但不能用硬裁剪牺牲楼梯短时发力。
- 根因：训练观测没有覆盖每环境近裁剪、持续孔洞和黑屏，故障也没有按完整 recurrent sequence
  固定；TBPTT 窗口不足以把故障前视觉与故障期动作建立梯度联系。旧平滑/强步态项又形成奖励
  竞争，可能通过减少快速摆腿和触地事件降低惩罚。初版 stair-memory 实现另有课程 Bug：只要
  `strength>0` 就对全部环境应用 near clip，并把未启用序列记成 `near_only`，实际不是 0->25%
  渐进。高层虽然不 step，初版仍把其 optimizer LR 改为 0，导致保存状态无谓漂移；warm start
  还可能优先读取 legacy `low_level.critic` 而不是 P3 的 `p3_critic`，并误用父包内更早的
  `p3_anchor` 作为 memory 教师而不是选中的 F2 当前低层策略。
- 修复：新增 `p3_depth_memory.py`，按独立 RNG 采样 Beta(1,4) `0.10-0.25m` 近裁剪，并对完整
  128 帧 sequence 持续施加 near-only、3% sparse、1-3 block、60%-85% severe 或 95%-100%
  blackout。启用掩码现在独立于 fault type，课程覆盖率和面板口径一致。clean compact feature
  由冻结 F2 anchor 生成教师动作，fault compact feature 只让低层 LSTM/RNN output/最终 action
  head接受 SmoothL1 memory 梯度；目标 1.5%、上限 3%，mirror 上限 1%，全部 auxiliary 上限 5%。
  rollout/TBPTT 改为 `128/128`，低层 CNN及全部高层模块冻结，高层 rollout/PPO不执行。
- 域随机化与抗扰：前 30 分钟为 friction `[0.60,1.30]`、base mass `+-0.75kg`、restitution
  `[0,0.05]`、noise `0.50`，之后为 `[0.55,1.35]`、`+-0.85kg`、`[0,0.10]`、`0.55`；15 分钟
  后通过平台已有 EventManager 打开 25-35 秒间隔、最大 `0.15m/s` XY push。不宣称 COM、PD、
  action delay 等没有可靠消费链的随机化。
- 力矩与步态边界：Hip/Thigh `17.6/22Nm`、Calf `34.4/43Nm` 采用相对 soft-hard 区间的平方
  sustained/peak 软约束，权重 `0.004/0.001`、frame cap `0.015`，不改仿真 effort limit。
  contact/crossing/starvation、deterministic smoothing 与 action-range auxiliary 保持 shadow-only；
  仅保留 `-0.015` prolonged-air 与 `-0.015` participation 防止单腿长期悬空。
- checkpoint 与恢复：合同升级为 `p3_low_stair_memory_dr_v1`，保存 depth/memory RNG、DR/push
  实际配置、低层版本和冻结高层 digest；exact resume 要求 `high_updates=0`。高层 Actor/Critic
  optimizer 和 scheduler 状态不再通过设置 LR=0 伪冻结，仅 Adapter LR 独立更新。P3 warm start
  优先并严格加载 `low_level.p3_critic`，只为缺少该 leaf 的旧兼容包显式回退 legacy critic；
  加载完成后从 F2 live encoder/actor 建立新的不可变 clean teacher，exact resume 保存并恢复该快照。
  depth fault 恢复只还原 RNG，不再在 `load_state_dict()` 内提前采样 live near-clip；统一由环境
  reset 路径采样一次，避免 resume 比连续训练多消耗一轮随机数。吞吐统计改为读取低层 storage
  的实际 transition 数，128 帧训练不再被错误报告成 80 帧。
- 本地验证：`PY311test` 已完成 P3 合同、调度、深度记忆、步态/径向、双评估、checkpoint、
  lifecycle、P1.5/P2 邻近回归；排除历史失效的 ST9 导入文件与 J9 fixed-LR 文件后结果为
  `378 passed, 5 skipped, 3 subtests passed`。新增 fixture 覆盖 near-clip 分布、故障覆盖率渐进、
  严重孔洞范围、持续故障、memory 梯度隔离、力矩 soft/hard 平方曲线、push/DR阶段及高层
  optimizer LR 不漂移；最终审查补充的 RNG 无额外消费回归与 P3/checkpoint 定向套件为
  `115 passed, 5 skipped`。真实 F2 父包结构含 `p3_critic`、完整高层、四套 optimizer 和双评估 leaf；
  本地 real-bundle smoke 进一步通过 warm start、冻结校准、低层 Actor/LSTM/Critic+memory 更新、
  action smoothing=0、高层两个 optimizer 不变和 save/exact-resume。Python 编译、13 份 TOML、
  `git diff --check`、`base_env.py` 零差异与 P3 monitor 实际构建（70 panels、440 metrics、单线最多
  20）均通过。Isaac camera/push/DR 动态 smoke 尚未执行，不能标为平台已验证。
- 回滚与最短检查：可把 schedule/load mode 回退到上一 P3 合同并移除 training-only
  depth/memory state，无需修改模型 I/O。再次遇到时依次检查 `near_clip histogram -> raw/aug hole
  rate -> fault_enabled_share/type -> memory hidden advantage -> high_updates/digest -> torque margin ->
  prolonged-air/participation -> stairs clean/fault eval`。关联 commit/PR、新 checkpoint、平台 task/
  model ID 与 SHA256：尚未产生。
- 2026-08-01 审查更正：原条目关于“15/30 分钟通过 `env.reset(config)` 启用 push 并增强 DR”的
  结论不成立。平台源码镜像 `kaiwu_runtime/base_env.py:1476-1480` 明确显示 `_create_env()` 在
  `self.env is not None` 时直接返回，阶段 reset 只能开始新 episode，不能重新装配 friction、mass、
  restitution、noise 或 EventManager。另据 Isaac EventManager 镜像 `event_manager.py:123-145`，Go2
  push 是 episode reset 时重采样的非 global interval event；25 秒 episode 配置 25-35 秒 interval
  几乎不会触发。本次改为首次 reset 即装配最终但保守的 `[0.55,1.35]`、`+-0.85kg`、
  `[0,0.10]`、noise `0.55`，并从进程启动启用 `0.15m/s`、`8-15s` push。删除伪环境重建索引、
  墙钟 DR 表和阶段切换日志中的 `environment_reset` 声明；阶段边界 reset 仅用于 episode/live
  recurrent 状态清理。`server/isaac_env/base_env.py` 保持零修改。
- 2026-08-01 深度故障更正：sparse/severe/blackout 原先在每帧重新采样空间掩码，无法形成连续
  缺失，且 severe 在已有约 80% 原始孔洞上继续清零 60%-78% 像素，会把最终无效率推到约
  92%-96%。现为每个 rollout 生成固定空间随机秩图，在故障持续期内复用；severe/blackout 按当前
  有效像素数计算精确 dropout budget，并按固定空间优先级只补足到目标范围，中心/下方优先级也
  计入同一总预算。新增 fixture 验证随机故障跨帧逐像素一致，以及均匀或集中分布的 80% 原始孔洞
  都不会被 severe 叠加到 85% 以上。
- 2026-08-01 更正验证：宿主 `PY311test` 禁用 bytecode/pytest cache 后，P3 contract、schedule、
  stair-memory 定向套件为 `50 passed`，全部 `test_p3*.py` 为 `100 passed, 5 skipped`；Python 编译、
  TOML 解析和 `git diff --check` 通过。尚未同步开发容器，也未取得真实 EventManager push、Isaac
  DR 实配或平台 smoke 证据，状态保持“本地已验证”。回滚时恢复静态启动配置前必须先解决平台
  create-once 边界，不能再次恢复阶段 `env.reset()` 伪重建路径。
- 2026-08-01 开发容器快测补充：通过 `local_sync_client.py` bundle 同步 55 个 server 文件并完成
  远端校验，没有同步或修改平台 `base_env.py`。父包
  `p3nav8h-r1_884257-evalfix-v2.zip` 使用分片上传器写入
  `agent_ppo/test_artifacts/`，远端大小 `32982918` bytes、SHA256
  `7722cf9f919a08a2e5c7ec06839bc36d2da740f753ad1a947bb16ead452955ab`。真实父包 CPU smoke 通过
  `p3_stair_memory_warm_start`、冻结阶段 Critic-only 更新、训练阶段 Actor/LSTM/Critic+memory 更新、
  高层 optimizer 不漂移和 `p3_exact_resume`。随后启动 1-env integrated Isaac smoke，目标覆盖
  camera、128 帧低层 PPO、Adapter 和 save/resume；进程持续运行至 RPC `260s` 超时后被终止，未
  生成结果文件，内核日志无 OOM 记录。该结果只能标记为“Isaac integrated 待验证”，不能宣称
  rollout/update、push 事件或 DR 实际采样已经通过；状态仍为“本地已验证”。
- 2026-08-01 最小监控与生命周期修复：复核发现 `_reset_env()` 在每个训练阶段边界调用
  `depth_fault.reset_environment()`，导致本应绑定环境进程的 near-clip 在八小时内重复采样。现改为
  幂等 `ensure_environment_initialized()`：fresh/warm start 使用构造期唯一采样，exact resume 恢复
  RNG 后只在首次 workflow reset 采样一次，后续 episode/阶段 reset 不再消费 RNG或改变阈值。
  新增 near-clip 十桶楼梯 terminal 完成率，以及 severe/blackout 每 rollout 事件数和实际配置持续
  时间。当前 worker wire 不包含 EventManager push 事件或 PhysX 逐环境 friction/mass/restitution
  readback，因此 Push/DR 面板改名为“配置合同”并显式输出 `runtime_telemetry_available=0`，不再把
  配置范围描述为实际采样；恢复时间仍明确不可用。删除 `short_high_adaptation_preserves`、
  `high_phase_command_owner` 和 `low_level_frozen_while_high_level_owns_command`，合同只保留
  `high_level_training=disabled_full_session` 及冻结模块清单。新增定向测试覆盖 near-clip 初始化幂等、
  楼梯 terminal 与 clip 桶关联和 fault 事件口径；平台 Push/DR 实测仍待平台提供回传链后验证。
- 2026-08-01 监控捕获 `20260801-100322` 更正：采集 866 项、851 项有点、15 项为空且无
  NaN/Inf。该 capture 的 `p3_window_completed_count=0` 与平台 `completed_count>0` 不是完成
  readback 回归：前者只统计当前 rollout 内 worker reset reason，后者来自平台 scorer lifetime
  累计，原面板名称未说明口径。现将其明确命名为 rollout reset 终止结果，并恢复独立的 P3
  M1/M2、standard/platform/joint、径向距离与 M3 边界面板。near-clip 楼梯完成率同步展示逐桶
  attempt 数；四地形乘三运动 gait 指标新增样本占比，避免把无样本桶的 0 当作正常表现。
  `depth_fault_duration_s` 改为 `depth_fault_planned_duration_s`，明确 shadow 阶段非零值只是预采样
  课程参数。P3 低层-only面板移除未接线的 `adapter_confidence`、禁用高层的 update time 和未使用
  CPU-pinned depth 路径的 H2D 指标。该修复只改变监控输出，不改 reward、optimizer、Push/DR、
  checkpoint 或模型接口；当前已启动任务不会热加载新面板，需新任务/新进程后验证。宿主
  `PY311test` 全部 `test_p3*.py` 回归为 `103 passed, 5 skipped`，Python 编译和
  `git diff --check` 通过；使用本地 MonitorConfigBuilder stub 实际构建为 75 panels、486 metrics、
  单 panel 最多 20 项。随后通过 `local_sync_client.py` bundle 定向同步 monitor builder、depth
  memory、P3 workflow 和 3 个对应测试文件，共 6 files；远端校验为 `191.4 KiB`，二次 dry-run
  显示 `files to overwrite: 0`、无删除候选。未同步或修改平台 `base_env.py`，未在容器重跑测试，
  也未创建新平台任务；已运行进程不会热加载本次面板，状态保持“本地已验证/代码已同步待平台验证”。
## BUG-20260801-007：前端监控工具未统一持久化浏览器登录状态

- 日期：2026-08-01；状态：本地已验证。
- 影响范围：`shared/arena_frontend_monitor` 本地人工监控采集工具；不影响 `server/`、`deploy/`、
  训练/部署接口、checkpoint 或平台容器运行时。
- 用户可见症状与证据：每次新开可视化 Chrome 测试窗口后都需要重新输入腾讯竞技平台账号密码。
  原入口虽然设置了 `AGENT_BROWSER_SESSION_NAME=tencent-arena`，但没有统一配置持久化 Chrome
  profile，也没有读取本工具目录的 `.env`；`manual_metric_recorder.sh` 直接执行
  `agent-browser tab new`，各 Python 文件分别拼装浏览器子进程环境。
- 根因与排除方向：采集逻辑只假设“已有登录会话”，没有为所有入口指定同一持久化 profile。
  不采用把 Cookie 字符串或密码写入源码/`.env` 的方案，因为这会产生明文凭据泄露和过期同步
  问题；使用 `agent-browser` 原生 Chrome profile 保存 Cookie、localStorage 和站点状态。
- 核心修复：新增 `browser_auth.py`，统一加载本地 `.env`、保留显式进程环境变量优先级，并默认
  使用 Git 忽略的 `runtime/browser_profile/`；profile 目录按 `0700` 创建。五个 Python 浏览器
  入口和两个 shell 推荐入口均使用同一配置；新增 `.env.example`、局部忽略规则、README 首次
  登录与失效重登说明，以及配置解析/profile 权限回归测试。
- 验证：`py_compile` 覆盖本目录全部 Python 文件，两个 shell 入口通过 `bash -n`，全部 12 个
  单元测试通过，`git diff --check` 通过；真实 `agent-browser 0.26.0 --version` 在统一环境下可执行，
  默认 profile 已按 `0700` 创建且被 Git 忽略。未使用用户账号执行真实登录复用 E2E；平台会话
  是否仍有效取决于腾讯服务端 Cookie 生命周期，服务端主动失效后仍需人工重新登录一次。
- 防复发、回滚和最短检查路径：测试必须确认进程环境覆盖 `.env`、相对 profile 以工具目录解析、
  profile 权限为 `0700`。再次出现重复登录时先检查 `.env` 的 `AGENT_BROWSER_PROFILE` 是否一致，
  再检查平台是否主动使 Cookie 过期。回滚可删除本次工具文件改动；本地 profile 可保留或由用户
  手动删除，禁止将其加入 Git。
- 血缘：当前分支 `codex/p3-stairmem-dr`；本次不涉及模型 ID、checkpoint、制品 SHA256、容器
  同步或真机验证；commit/PR 尚未创建。工作区中既有 P3 训练改动属于用户，未纳入本任务。

## BUG-20260802-008：P3 低速楼梯训练缺少相机时序与可验证的后期 Push，旧奖励会扩大步态回退

- 日期：2026-08-02；状态：本地已验证，待开发容器与平台 smoke。
- 影响范围：分支 `codex/p35-gaitfix2h` 的 `p3_standard_joint` 训练入口、training-only worker
  transport、低层 PPO/Adapter、checkpoint 与监控；不改变 Standard/Track eval、ONNX、部署 I/O，
  不修改平台覆盖的 `server/isaac_env/base_env.py`。
- 用户可见症状与证据：上一轮楼梯恢复同时施加强动作平滑、接触/步态代价和较强 DR，出现低速
  上楼抬腿/触地退化；真实相机以约 30Hz 提供深度而低层以 50Hz 执行，旧训练每控制帧都按新图像
  处理，未覆盖重复帧与 40-150ms 延迟。旧 Push 面板只有配置值，没有真实事件、delta 和恢复指标，
  不能证明 EventManager 链实际生效。用户要求本轮只聚焦低速楼梯非劣化、相机时序和后期小 Push。
- 根因与排除方向：P3 extra 46 列没有 joint acceleration、逐 link contact 持续时间或 Push 实测
  transport，训练侧无法实现 fail-closed deadzone/cap，也无法区分“配置启用”和“真实发生”。
  `BaseEnv` create-once 边界意味着阶段 reset 不能重建 EventTerm，但 Isaac EventManager 的公开
  `get_term_cfg/set_term_cfg/reset` 可以修改既有 interval term；因此不采用 BaseEnv 补丁、自定义
  外力函数、配置值伪装遥测或扩展部署 action/observation 的方案。
- 核心修复：任务改为 `p35gaitfix2h`、7200 秒、128 env、25 秒 episode、课程关闭；阶段为
  `gaitfixcalib/repair/pushwarm/pushfull/stable`。P3 extra/wire 扩为 `108/493`，新增 joint-acc12、
  base/head 聚合与四腿 hip/thigh/calf 共 14 槽的 contact force/onset/over-threshold duration、
  joint/contact mapping valid、Push event/delta/age/active/telemetry。拆分 worker wire 后立即把
  target/exec command 回填本地 aux，修复 gait baseline、reward、Adapter 和面板全部误归低速桶。
- 相机与 Push：冻结 CNN feature32 使用每环境随机 capture phase 的 30Hz capture、50Hz hold、
  10 帧 FP16 FIFO；前 15 分钟主动延迟关闭，随后训练 40-100ms，75 分钟后加入 100-150ms，
  150-250ms 只作 shadow。`push_robot` 启动速度固定为 0；worker preflight 核验 term、interval mode、
  原函数和 public API 后安装透明 wrapper，75/90 分钟分别切到 `+-0.05/+-0.08m/s`、12-18 秒，
  并记录真实 delta。checkpoint 的 session 时间通过 materialized reset `usr_conf` 写入 worker
  `resume_offset_s`，75 分钟后的 exact resume 不重新等待。任一 preflight/shape/finite 失败时
  telemetry 置无效且 Push 条件奖励归零。
- 奖励与冻结：前 15 分钟采集父策略 joint position P99、joint-acc noncontact P95/onset P99、
  posture P95 和 gait frequency P05；不足或映射无效时训练项为 0。之后启用有 deadzone/cap 的真实
  推进、默认姿态、joint acceleration、undesired contact、gait responsibility 与合并姿态项，
  Push 后 0.4 秒只减半 joint-acc/gait/非关键姿态。原生重复 posture/joint-acc/contact 权重置 0；
  torque 继续使用 Hip/Thigh `17.6/22Nm`、Calf `34.4/43Nm` 的低权重平方软约束。只更新低层
  LSTM/RNN output、最终 action head、Critic 与 Adapter；Actor body/std、CNN 和全部高层冻结，
  `high_updates=0`。anchor/memory/mirror 梯度按 3%/1.5%/0.5% 标定，合计封顶 5%。
- 监控与日志：新增监控合同注册/有数据/空项统计、相机 capture/hold/feature age、Push term/mode/
  wrapper/API、事件数/环境覆盖、真实 delta 分位与方向、姿态/速度恢复、地形/难度/命令/速度分桶，
  以及 P3.5 奖励分解和辅助梯度。日志前缀为 `[P35MonitorContract]`、`[P35PushPreflight]`、
  `[P35PushPhase]`、`[P35PushEvent]`、`[P35PushSummary]`、`[P35Baseline]`、`[P35TrainState]`。
- 验证：宿主 `PY311test` 的 P3/P2 worker、双评估、hard-start 与相邻回归为
  `237 passed, 5 skipped, 3 subtests passed`；新增测试覆盖 493 维连续索引、30/50Hz feature FIFO、150ms 主动延迟
  上限、相机 RNG round-trip、正常 envelope 奖励严格为 0、contact cap、Push grace、EventManager
  原函数 wrapper、真实 delta 和公开 term 动态切换。Python 编译与 `git diff --check` 通过；
  `server/isaac_env/base_env.py` 零差异。尚未同步/运行开发容器，尚未取得真实 Isaac camera、
  ContactSensor 14 槽、EventManager delta、平台 15-30 分钟 smoke 或真机证据，状态不得升级。
- 父包与血缘：计划从任务 `235689` 的 `stairrobust/stairfinal` 同 seed A/B 中选择唯一父包；本地
  尚无这两个平台制品，因此最终 model ID、标签、SHA256 和模块 digest 待平台选择后补录。当前
  TOML 保留 `p3nav8h-r1_884257-F2` 作为结构兼容 lineage，不作为模型 ID 硬门禁。commit/PR、
  新 checkpoint 与平台 task ID 尚未产生。
- 遗留风险、回滚与最短检查：正式任务前必须在开发容器 smoke 配置中 5 秒启用 `+-0.03m/s`、
  2-4 秒 interval，确认 `push_telemetry_valid=1`、事件计数递增、delta 非零且不越界；再检查
  `joint/contact mapping -> camera hold/age -> command aux readback -> reward baseline valid ->
  high_updates/digest -> Push recovery`。失败时回滚到选中的父包并关闭 P3.5 training-only shaper、
  camera timing 和动态 Push；网络/部署接口未变，无需修改部署端。

- 2026-08-02 独立审查更正（状态：本地已验证，待开发容器与平台 smoke）：审查确认首版仍有
  七处训练语义偏差。`vx` 三档曾误写为 `0.05-0.25/0.25-0.60/0.60-1.00m/s`，且 restart 绕过
  公共采样器；joint-acc 放宽错误使用碰撞阈值 onset 而不是普通足端 contact onset；姿态包络遗漏
  projected-gravity Y 的 roll-like 分量；15/15/35/35 地形桶仍沿用旧等宽列边界；纯延迟帧未进入
  clean-teacher MemoryAux；最后 15 分钟低层冻结时 Adapter 未更新，且 replay 比例未按阶段切换；
  监控健康度以 `len(metrics)+5` 和固定零年龄伪装真实注册与新鲜度。
- 修复后所有含 `vx` 的运动类型统一使用 `0.10-0.35/0.35-0.70/0.70-1.00m/s`、55/30/15；
  foot onset 按 axis-major `(hip,thigh,calf)` 映射到同一腿；姿态项使用强 roll/roll-rate、次级
  pitch-rate 和弱 pitch；地形列边界统一为 `3/6/13/20`。MemoryAux 选择 mask 改为 pixel fault 与
  实际 delivered feature age 的并集，并新增 fault-only、delay-only、overlap 面板。Adapter 在
  stable 阶段每 rollout 更新一次，replay 依次为 `50/25/25 -> 60/25/15 -> 75/15/10`。监控合同
  按 required key 实际存在、有限值及最后有限数据时间计算，缺项或陈旧项不再显示为健康。
- 父包不再保留未消费的 A/B 元数据，按用户确认固定为上一轮任务 `235689` 的最终模型：平台模型
  ID `1013548`、标签 `stairfinal`、文件 `model.ckpt-stairfinal-1013548.pkl`。平台列表未提供文件
  SHA256；该值和模块 digest 必须在开发容器实际下载/加载后补录，当前不得编造。完整
  `agent_ppo/tests` 回归为 `417 passed, 5 skipped, 3 subtests passed`，覆盖命令分布、普通触地
  放宽、逐腿 onset 基线、姿态轴、地形桶、delay-only MemoryAux、Adapter 阶段 replay/冻结阶段
  更新及监控陈旧度；Python 编译、TOML 解析、`git diff --check` 通过且 `BaseEnv` 零差异。
  容器 Isaac/Push smoke 与平台训练仍待执行。
- 同次核验修复了既有完整测试收集阻断：`test_st9_opt3_d2.py` 仍导入已删除的分段
  `_student_drive_probability`，而当前 LBC 已使用线性 `_ramp_probability`。测试已迁移到当前线性
  ramp、当前无 goal 的 locomotion teacher 和 three-way loss 接口，不恢复废弃生产 helper。
  完整回归还发现通用 `AlgorithmPPO` 的 fixed schedule 守卫被删除，导致 optimizer LR 漂移静默
  通过；现恢复初始化 LR 的前后 update 校验，并使用配置的 adaptive min/max。该通用修复不改变
  P3.5 的显式分组 LR 调度，但防止旧 fixed PPO 入口静默漂移。
- 实现血缘补录：上述 P3.5 训练、审查修复、合同与回归测试已落在 commit `ee5afc8`；PR、新训练
  task、新 checkpoint、父包文件 SHA256、容器 smoke、平台评估和真机结果仍待产生。
- 2026-08-02 开发容器更正（状态：代码已修复待容器复测）：真实
  `model.ckpt-stairfinal-1013548.pkl` 首次 warm-start 暴露配置 `p35_previous_run_final_warm_start`
  未加入 loader 合同迁移允许列表，因而错误抛出 `P3 exact resume contract mismatch`。现将所有
  显式 P3 合同 warm-start mode 收敛为常量并补回归；exact resume 的严格合同校验保持不变。
- 同次容器检查发现 `p3_joint_rollout_smoke.py` 的 integrated 场景仍引用已删除阶段
  `stairrobust`，会在 Isaac 初始化后以 `KeyError` 退出；现改为当前 `repair` 边界并增加静态回归，
  避免浪费开发容器启动时间。
- 开发容器验证补充：模型包通过新的 VSCode Remote 文件协议上传到
  `agent_ppo/test_artifacts/p3stairmem8h_1013548.zip`，大小 `33824100` bytes，上传 staging 与最终
  文件均通过 SHA256 `5ef4021d9ab56cee5ffbc2673248dfc50e616795691a413b38ecdcef332f7e80`；未调用
  RPC 分片上传器。只提取 `model.ckpt-stairfinal-1013548.pkl`，checkpoint SHA256 为
  `574924419ccd58923ccb5b6b2b29a88464ca2744e06bc888b7c7f5021f75250a`，没有用压缩包内旧源码
  覆盖当前分支。容器定向回归 `193 passed`；真实父包 smoke 通过 warm-start、冻结 Actor/Critic
  校准、可训练 LSTM/Actor/Critic+memory 更新、高层 optimizer/digest 冻结及 save/exact-resume。
  1-env、128 帧 integrated Isaac smoke 启动后显存峰值约 `2.78 GiB`、运行期约 `1.09 GiB`，无
  OOM；约 4 分钟时平台先返回 `WEBIDE_RECORD_NOT_FOUND`，随后 VSCode Remote WebSocket 也断开，
  未取得最终结果，因此 Isaac Camera、真实 Push EventManager 与完整 rollout 仍标记待验证。
- 2026-08-02 容器重启后复测更正（状态：代码已修复待容器完整复测）：1-env smoke 实际只有
  1 条 `TBPTT=128` recurrent sequence，但工具仍沿用生产 `num_mini_batches=4`，因而在 Isaac、相机、
  128 帧 rollout 和模型加载均成功后报 `1 recurrent sequences are not divisible by num_mini_batches=4`。
  smoke 工具现根据 env 数选择不超过生产值的最大可整除 minibatch 数，1-env 固定为 1，
  不改正式 128-env 配置。限时复测已进入第 75 帧并观察到单进程运行显存约 `1.47 GiB`；
  旧 smoke 在 IDE 断开后的孤儿 Isaac 子进程与新 smoke 重叠，导致合并显存最高 `3.85 GiB`，
  该值不是单任务峰值，不用作容量结论。受五分钟预算限制未取得最终 PASS；已精确停止所有本次
  smoke 派生进程，删除上传 ZIP、解压 checkpoint、`.uploading`/SHA 临时件、smoke 日志与
  临时 checkpoint，复核 GPU 回到 `0/5000 MiB`。完整 Isaac Camera、Push EventManager 和 save/resume 仍待
  下次使用进程组可控的 smoke runner 验证，不得将本次限时运行记为平台已验证。
- 2026-08-02 监控配置快速更正（状态：本地已验证）：平台拒绝加载包含句点的标题
  `P3.5 低速楼梯与后期Push` 和中文面板名 `P3.5训练侧奖励`，错误为“面板名称非法”。
  根因是平台两类名称合同均不允许 `.`，与 metric key 和数据无关。显示名已改为
  `P35 低速楼梯与后期Push` / `P35训练侧奖励`，未改变面板 key、指标或训练行为。

## BUG-20260802-009：P4 直接复用旧 mapper、Goal/相机链和 Push replay 会破坏导航鲁棒训练合同

- 日期：2026-08-02；状态：本地已验证，开发容器、平台 smoke、评估和真机均待验证。
- 影响范围：分支 `codex/p4-nav8h`、任务 `p4nav8h`、`p4_nav_ppo` 训练、P4 Standard/Track
  双评估与两次四小时 exact resume。父模型要求为本轮 `p35gaitfix2h` 最终 checkpoint；模型 ID、
  文件 SHA256 和模块 digest 尚未产生或未提供，不能用其父模型 `1013548` 冒充。
- 症状与根因：旧 P2 action mapper 不消费 capability15，无法执行三档动态速度上限；旧
  NavGoalChain 无 MAP 传播/jump/dropout 恢复；相机 fault、low feature 与 high input 没有共享
  frame 合同；旧 Adapter future label 不识别随机 Push，且 replay 先混池再采样。首轮实现核验又
  发现 rollout 尾帧会被下一 rollout 重读并重复累计 `push_epoch`、exact resume 时 worker Push
  时钟未恢复、`fault-only` mask 永远为零、Goal 过期仍可横移，以及 P4 exact resume 未重置冻结
  低层基准 digest。
- 核心修复：新增版本化 P4 mapper、GoalBelief v2、共享 30Hz capture/50Hz recurrent 相机状态、
  0.8 秒真实 slew/reversal 积分预测、exec/true yaw cancellation 与同比安全奖励 cap；低层仅复用
  CNN feature32且每 tick 推进 LSTM。Push pulse 按 worker step 去重，resume session 通过
  `env.reset(usr_conf)` 回填 EventManager 阶段；Adapter horizon/pose label 跨 Push 失效，records
  先按八项合同过滤再按目标池比例采样。checkpoint 保存 mapper、Goal/camera/speed RNG、冻结
  digest 与阶段，新增 P4 layered validator 和 Standard/Track loader。
- 监控与防线：新增 Goal innovation/age、normalized/mapped cmd、速度档、yaw cancellation/flip/
  overshoot、安全 raw/applied/scale、共享 frame/fault mask、Push delta/标签拒绝、Adapter兼容池、
  low digest 与 TBPTT 指标，并按真实有限数据维护缺项与最长年龄。新增 `test_p4_nav.py` 覆盖 mapper、
  Goal传播、相机三类 mask、Push 去重、Adapter horizon、冻结隔离、50Hz low LSTM、57905/493
  tick、P4双评估 validator、Track loader和 exact-resume round-trip。
- 最终独立审计更正：合法 segment/goal epoch 原先没有接入 learner/worker GoalBelief，目标切换可能
  被当作 jump 拒绝，且 yaw cancellation 窗口不会清空；现统一接入并新增 MAP、传播、重获时间面板。
  2-6 小时 severe-camera 行禁止长 Goal dropout，组合严重故障只在最终 stress 阶段开放。相机辅助
  loss 的 scalar 指标原先未进入 PPO 汇总，现显式聚合 clean/live action MAE、latent cosine、梯度比
  与三类 mask。另修复 P4 record 采样重建时错误使用 P2 capability15，以及 Track eval 因忽略
  SafetyHead 而使 `safety_cap` 永远为 1 的部署闭环缺口；安全上限现由 delivered depth 与当前 exec
  slew 弧线生成，训练与评估一致。
- 验证：宿主 `PY311test` 完整回归为 `442 passed, 5 skipped, 3 subtests passed`；Python编译、
  TOML解析、`git diff --check` 和 `server/isaac_env/base_env.py` 零差异通过。当前尚未在开发容器验证
  EventManager `get/set/reset`、真实 Push delta、128 env 32-tick PPO/Safety/Adapter、显存与吞吐；
  也未运行15-30分钟平台 smoke，不能升级为平台已验证。
- 回滚与最短检查：若 P4 smoke 失败，继续使用 P3.5 最终包；依次检查父包 ID/SHA/digest、mapper
  version、Goal age、low digest、共享 frame ID、Push telemetry、Adapter reject 与 reward
  conservation。`BaseEnv`、policy/ONNX/deploy shape均未修改，回滚不需要部署端迁移。
- 2026-08-02 快速回归更正（状态：本地已验证，开发容器与平台仍待验证）：固定 fault fixture
  复现 delivered frame ID 从 `4 -> 0`、`15 -> 14` 回退。根因是 delay 在每个 control tick
  重采样，且历史队列暂时找不到目标时间时错误回退到 newest frame；这既破坏“同一故障连续存在”
  的合同，也会让 recurrent policy 看到时间倒流。现将 delay 固定在 fault event 创建时采样，历史
  不足优先 hold 上一 delivered frame，并以显式 monotonic guard 禁止旧 frame replay。相机孔洞与
  结构块增强同时从逐环境 Python 循环改为 128-env batch tensor 操作，减少 rollout 热路径开销。
- 同次修复发现 exact resume 先恢复 generator state、随后调用 `reset()` 会消耗 RNG，导致恢复后
  故障序列不可精确重现；现先清 live buffer、最后恢复保存的 generator state。监控 required contract
  原先仅列 14 个样例指标，无法发现 Goal、Push、Adapter、reward、冻结状态和资源面板的大量缺项；
  现扩展为完整 P4 关键指标集合，并为九类 Adapter 兼容拒绝原因统一输出零值或实际计数。定向
  `test_p4_nav.py` 为 `19 passed`，覆盖 frame 单调性、相机 RNG exact resume 与监控合同；开发
  容器 EventManager、128-env 性能和平台面板数据仍未验证，不得标记为平台已验证。
- 2026-08-03 父包与容器测试更正（状态：开发容器结构验证通过，Isaac smoke 待完成）：操作者
  改为选择 `p3stairmem8h` 最终 `stairfinal` 模型 `1013548`。容器对真实 checkpoint 验证
  `format=kaiwu_train_v1/schema=2/stage_type=p3_standard_joint`，低层、NavigationEncoder、高层
  Actor、SafetyHead 与 Adapter 的 spec/shape/finite 全部兼容；原 loader 唯一阻断是把
  `phase_label=stable` 写成硬条件。现允许结构完整的 `stairfinal/stable`，默认候选改为
  `1013548`，模型 ID仍不替代结构验证。另修复容器中 `agent_ppo -> /workspace/code` 符号链接导致
  BaseEnv 零修改测试误找 `/workspace/code/isaac_env` 的路径假设；测试只读取平台 BaseEnv，未修改它。
- 2026-08-03 首轮 Isaac smoke 更正（状态：代码已修复待容器复测）：1-env Track+Camera 已完成
  P4 装配、父包预加载、Isaac reset/step，并推进至少 75 个低层帧；随后首次框架保存路径
  `model.ckpt-p4warm-1013600.pkl` 被 `validate_probe_filename()` 拒绝。根因不是平台新模型 ID
  `1013600` 与父 ID `1013548` 不同，而是 P4 phase 中的数字 `4` 违反探活正则的纯小写字母段。
  现将 P4 四阶段标签原子改为 `pnavwarm/pnavrobust/pnavfull/pnavstable`，保留平台注入 ID 并新增
  任意新 ID 的探活回归。同期平台监控校验发现
  `adapter_compat_rejected_mismatch_response_capability_profile15` 超过 60 字符，导致整个用户监控
  配置被跳过；展示 metric 已缩短为 `adapter_compat_rejected_mismatch_response_profile15`，底层
  record rejection reason 保持不变，并新增全量 required-metric 长度断言。首轮进程组已精确停止；
  修复后定向测试、checkpoint 保存、32-tick PPO/Adapter 与平台面板仍待复测。另发现旧 smoke
  runner 的停止条件依赖 `iteration/platform_dump_boundary`，而 P4 分支没有发出这两个诊断事件，
  导致训练已更新并保存仍不会自动停止；现补充仅在 `NAV_SMOKE_EVENT_LOG` 配置时落盘的原子事件，
  生产环境未配置时为无 I/O 的 no-op。
- 2026-08-03 容器复测结果（状态：开发容器已验证，平台正式任务待验证）：真实父包
  `p3stairmem8h_1013548` 完成 1-env Track+Camera 全栈 smoke。首轮有效迭代包含 320 个低层帧、
  32 个高层 tick、4 个 PPO epoch/8 次高层 gradient step 和一次 Adapter update；
  `low_digest_drift=0`、`low_optimizer_steps=0`、`push_term_assembly_valid=1`，CUDA allocated/reserved
  峰值约 `91.6/107.0 MiB`（不含 Isaac/相机进程显存）。新文件名成功保存为
  `model.ckpt-pnavwarm-1013600.pkl`，随后平台注入 `1013760/1013868` 也均成功，证明新模型 ID
  不会被父 ID 单点阻断。自定义 monitor 已进入逐 metric 表达式构建，未再出现配置校验失败。
  smoke 事件最终满足 `iteration + platform_dump_boundary`，runner 触发自动停止；随后显式执行
  `stop` 完成进程组收尾并复核无遗留进程；
  SIGTERM 后 model-file-save 子进程短暂出现的 connection-refused 日志属于受控关停顺序，不是训练
  update/save 失败。当前证据不替代 128-env 显存/吞吐或正式平台 15-30 分钟 smoke。

## BUG-20260803-010：P4 首轮相机/Goal 单位污染、warm-up Adam 漂移与卡墙样本长期占用

- 日期：2026-08-03；状态：本地已验证；卡墙 termination 子链已开发容器验证，
  平台 smoke、其余 P4 联合训练、评估和真机均待验证。
- 影响范围：`codex/p4-nav8h`、首轮约 50 分钟 `p4nav8h` 训练及下一任务 `p4nav8h-r2`。
  首轮 checkpoint 只保留诊断，不作为 R2 父包；R2 仍从 `p3stairmem8h_1013548` 干净 warm start。
- 用户可见症状：首轮监控出现 nominal delivered hole 明显高于 raw hole、Goal reject/age 异常且
  卡滞占比很高；大量机器人接触墙后继续占用接近 120 秒 episode。该证据只能证明训练数据链异常，
  不能据此认定原安全奖励权重不足。
- 根因：`P4SharedCameraState` 将 `0.10-0.25m` 直接与 `[0,1]` depth 比较；learner 又从已逐维
  clip 到 `+-10m` 的 Critic goal 编码反推米制目标，远目标传播后必然形成假 innovation。warm-up
  只将 Actor/CNN LR 设为零，Adam `step/exp_avg/exp_avg_sq` 仍会变化。平台已有
  `nav_stuck_timeout`，但 P4 未向其写入经墙面证据确认的 motion-stuck counter。
- 修复：近裁剪比较改为 `near_clip_m/max_depth_m` 并拆分 raw/added/delivered hole；P4-only
  training wire 从 493 扩为 507，附加 raw metric goal2 与 stuck diagnostics12，网络和 eval wire
  不变。GoalBelief 升级为带过程协方差、五次一致重捕获和 stale-MAP 低速的 v3。warm-up 按 optimizer
  group 设置 `requires_grad`，SafetyHead 继续训练而冻结参数及 Adam state 不动。新增 worker-owned
  `MotionWallStuckTracker`，默认 shadow；active 时只经平台现有 termination manager 触发真实 reset，
  reason=4 使用独立 `-6` 且不 bootstrap、不重复结算碰撞和停滞。首次 terminal 帧会冻结旧 episode
  的 command、aux、raw goal 与 stuck diagnostics，自动 reset 后的新 episode 数据不得覆盖 507 维
  transition tail；相机监控使用明确的 `camera_delivered_hole_rate`，旧 `camera_hole_rate` 仅保留为
  兼容别名。
- 验证：宿主 `PY311test` 的完整 `agent_ppo/tests` 回归为
  `454 passed, 5 skipped, 3 subtests passed`；Python 编译、TOML 解析、required monitor 159 项覆盖、
  144 个面板无 line 面板超过 20 指标，以及
  `git diff --check` 通过。新增测试覆盖 0.10m 到 0.02 normalized、15m raw goal transport、冻结组
  Adam moments 位级不变、reason=4 优先级/mask、single terminal penalty 与 fake termination manager
  readback，并直接验证 terminal 后旧 raw goal/stuck tail 不被 reset 后的新值覆盖。
  `server/isaac_env/base_env.py` 未修改。
- 遗留风险与最短检查：本地镜像只证明平台历史源码存在该 term，不能替代当前容器动态对象。
  容器必须打印 active term、time_out、get/set/readback、dt，并先做 1-env 物理 reset；平台依次跑
  shadow、12 秒 active、再决定 10/12 秒。若映射无效、term readback 失败或误杀超过 1%，保持
  shadow/120 秒 timeout。回滚只需关闭 `[p4_nav_ppo.stuck_reset].enabled/mode` 并回到 P3 父包。
- 2026-08-03 容器连接补充：本地 RPC network dry-run 返回 `WEBIDE_RECORD_NOT_FOUND`，已登录的
  Chrome IDE 标签随后返回 Remote Agent WebSocket 断开。此次未发生代码同步、Isaac reset 或
  termination-manager 动态验证，故状态仍为“本地已验证”；重新打开开发容器后应先同步 R2 文件，
  再执行 1-env shadow/active 物理 reset smoke。
- 2026-08-03 审查修正：发现 P4 warm start 与 eval 仍把 `phase_label` 当成结构正确后的第二道
  硬门禁；同时 checkpoint 虽保存运行时 `stuck_reset_contract`，exact resume 却只比较固定的
  shadow/10 秒默认合同且不读取保存值。现将 P3/P4 phase label 降为 warning-only 身份元数据，
  结构、spec、shape 与有限值继续硬校验；stuck-reset 配置统一经版本化规范函数供 worker、reward、
  training contract 和 checkpoint state 使用，active/shadow、10/12 秒或 terminal penalty 不一致时
  exact resume 明确报错，避免静默改变终止与奖励分布。新增未知标签通过结构验证、动态合同写入和
  resume 配置漂移拒绝测试。状态保持“本地已验证”；容器与平台验证层级未因此自动升级。
- 2026-08-03 监控口径更正（状态：本地已验证，容器待验证）：平台镜像的 Track 结果导出明确按
  `_num_cols` 逐列生成 `track_l{col}`；20 列不会自动两两合并成旧 L0-L9。P4 面板此前仍只注册
  10 个旧指标，因此遗漏 L10-L19。现新增 P4-only 20 列结果组，每个 line 面板恰好 20 项，未超过
  平台限制。另修复 shadow tracker 在达到阈值后每帧重复产生 `would_reset/saved_seconds`、P4
  Adapter 三池 batch 被旧 `track_batch_envs` 元数据重复计数导致比例大于 1，以及
  `lifecycle_success` 容易被误读为完成数的问题。配置仍保持 `mode=shadow`：本轮平台指标显示
  `wall_stuck_term_available=0`、`wall_stuck_term_config_valid=0`，直接切 active 不能形成真实 reset。
  定向回归为 `61 passed`；容器仍须先核验 active term、time_out、get/set/readback 与 1-env 物理
  reset，再以 12 秒 active smoke 判断是否启用，不能用配置文本替代运行时证据。
- 2026-08-03 开发容器验证与修正：首次 1-env smoke 暴露 bridge 会在 Isaac
  termination manager 完成装配前做一次预检，并将早期 `term_available=0`
  永久保留；并非平台缺少 `nav_stuck_timeout`。现在 tracker 在首批真实 frame
  内最多重试 8 次终止管理器配置。新增
  `agent_ppo/tools/p4_stuck_runtime_smoke.py`，对当前容器做了 1-env Track+Camera
  实测：active terms 包含 `nav_stuck_timeout`、`time_out=true`、`dt=0.02`，
  临时 `confirmation=0.04s` 成功回读 `max_stuck=2`，episode length 从 1 被物理
  auto-reset 为 0，worker snapshot 保留 `reset=true/reason=4`，smoke 输出
  `P4_STUCK_RUNTIME_SMOKE_PASS`。平台 `BaseEnv` 在 trajectory recorder 未启用时仍会将
  公开 `truncated` 清零，因此 P4 继续以 worker terminal snapshot 恢复 timeout 语义，
  不修改平台覆盖的 `BaseEnv`。容器定向回归为 `167 passed`；下一轮平台
  smoke 配置改为保守的 `mode=active, confirmation_s=12.0`，仍须以误杀率决定
  正式长训保留 12 秒还是改为 10 秒。
- 2026-08-03 开发容器联合 smoke 补充：修正后容器定向回归为
  `168 passed`，生产 TOML 已确认为 `mode=active, confirmation_s=12.0`。使用真实父包
  `p3stairmem8h_1013548` 完成 1-env 全链路 smoke：320 个低层帧保持冻结，
  收集 32 个高层 tick，执行 4 个 PPO epoch（`high_updates=8`）、Adapter update
  和 checkpoint 保存。父 records 兼容迁移 64 条、拒绝 0 条，低层 digest 无漂移；
  CUDA 峰值 `max_memory_allocated=40,605,184` bytes、
  `max_memory_reserved=52,428,800` bytes。smoke checkpoint 已成功生成并校验，
  随后与上传父包、解包副本一并从容器清理。该证据将本条的开发容器
  验证层级提升为已验证；128-env 显存/吞吐和平台 12 秒 active 误杀率
  仍待 smoke，不将 1-env 结果外推为正式长训已验证。

## BUG-20260803-011：P4 安全倍率边界突跳且父 Adapter 长时域记录表面兼容、实际不可采样

- 日期：2026-08-03；状态：本地已验证，开发容器与平台待验证。
- 影响范围：`codex/p4-nav8h`、`p4nav8h-r2` 安全奖励课程、ResponseAdapter 0.6/1.0 秒 velocity、
  pose/stuck 长时域保护、checkpoint exact resume 与监控口径。父包为
  `p3stairmem8h_1013548`；文件 SHA256 与新输出 checkpoint 尚未产生。
- 用户可见症状：上一轮约 39 分钟有效训练中 `reward_multiplier` 仅约 0.052，安全项占比很低；代码
  又会在 7200 秒从接近 0.5 直接跳到 1.0。Adapter 面板同时显示 0.2 秒标签约 99% 有效，但
  0.6/1.0 秒、pose/stuck 基本为零；父包 32 条 completed records 全部计入
  `rejected_missing_contract`。
- 根因：P4 旧倍率在 30-120 分钟只线性到 0.5，进入 2 小时阶段时直接设为 1.0。P3 保存 completed
  records 时尚未写逐记录 P4 contract，P4 过滤器因此全部拒绝；即使简单放行，旧采样器还会把
  32 条连续父记录对半切成 16/16，而一个 Adapter 样本需要 `burn-in8+sequence16=24` 条连续记录，
  两个父池仍永远无法采样。
- 排除项：不是模型 ID `1013548` 不匹配，也不是应直接提高安全最终权重；BUG-20260803-010 已证明
  首轮相机和 Goal 数据污染时不能据此判断基础权重不足。也不允许把缺 contract 的当前 P4 record
  一并放宽。
- 修复：安全倍率改为 0-30 分钟 0、30-60 分钟 0->0.25、60-120 分钟 0.25->1.0，最终权重和
  `-0.05/tick` cap 不变；Goal fault 拆为独立 multiplier，避免奖励修改连带改变观测故障强度；ramp
  版本进入 training digest 并升级 checkpoint contract。P4 warm start
  在加载父 records 前建立当前 Adapter contract，仅对 parent record 验证完整 tensor shape、有限值、
  horizon/sequence、来源 digest 格式与序列内部版本一致性，合格后标记
  `legacy_parent_structural_v1`；历史记录明确作为 off-policy replay，不要求伪装成最终冻结低层 digest。P3 current 与嵌套
  earlier lineage 按真实池保留，缺失层级从第一个可采样的兼容池补齐。新增 migrated/rejected 监控。
- 验证：宿主 `PY311test` 定向回归
  `test_p4_nav.py + test_p2_core.py + test_nav_stage_and_metrics.py` 为 `167 passed`；新增用例确认
  32 条 legacy parent records 能形成 24 条连续窗口并产生有效 0.6/1.0 秒 mask，非法 digest、
  NaN、shape drift 和当前 P4 缺 contract 均继续拒绝。修改 Python 文件编译、全部 TOML 解析和
  `git diff --check` 通过。随后直接从本地真实
  `p3stairmem8h_1013548.zip/model.ckpt-stairfinal-1013548.pkl` 只读加载父 buffer：32 条 P3 current
  加 32 条 nested earlier-lineage 共 64 条全部通过结构迁移并可采样，8-env probe 的
  0.2/0.6/1.0 秒有效 mask 为 `1.000/0.726562/0.625`，无 migration rejection。开发容器、平台
  smoke、评估与真机均未执行，不能升级为平台已验证。
- 防复发：测试必须覆盖 1800/3600/7200 秒 ramp 连续性；32 条 legacy parent record 能形成
  24 条窗口并产生有效 0.6/1.0 秒 mask；非法 digest、NaN、shape drift 和当前 P4 缺 contract
  必须继续拒绝；迁移条件不得引用模型 ID或标签。
- 血缘：分支 `codex/p4-nav8h`；commit/PR、新任务 ID、新 checkpoint 和 SHA256 均待生成。
- 回滚：回滚本条 ramp/legacy migration；不得回滚 BUG-20260803-010 的相机、Goal、冻结或 terminal
  修复。若父记录无法通过结构验证，允许 Track-only Adapter 更新并明确告警，禁止伪造 compatible。
- 再遇检查：`reward_multiplier@7199/7200` -> migrated/rejected parent counts -> parent pool连续长度 ->
  `adapter_valid_06s/10s` -> low-level digest -> Adapter update/pose/stuck loss。

## BUG-20260803-012：P4 Maze 继续训练缺少感知/决策/记忆归因且速度档硬限制干扰 warm start

- 日期：2026-08-03；状态：本地已验证，开发容器与平台待验证。
- 影响范围：分支 `codex/p4-maze2h-attack`、任务 `p4maze2h-attack`、`p4_nav_ppo` Maze-only
  继续训练、P4 监控面板、checkpoint exact resume/warm start 合同。
- 用户可见症状：上一轮 P4 长训后评估仍出现严重卡墙、S 型摆头和迷宫绕圈；同时用户希望用
  `0.6-0.7m/s` 的稳定区间做软性引导，而不是把动作映射硬改成低速上限。旧实现的
  slow/cruise/fast 随机速度档会把同一 Actor 输出映射成不同物理速度，且面板无法区分
  “没看见墙/路口”“看见但 Actor 选错”“局部选对但记忆/路线失败”。
- 根因：P4 旧 mapper 将随机 `user_speed_cap` 纳入 policy target 映射；PPO reward 和诊断只看到
  一个混合后的 target，缺少 policy target、limited target、SafetyHead 风险、teacher scene、
  Head 正确但 Actor 选错、风险后减速以及 zero-hidden shadow 的并列口径。`student_risk_*`
  实际是 SafetyHead 输出，命名会被误读为 predictive sector risk。新任务还需要从旧
  `p4nav8h` 最终包 warm start，不能因合同版本升级误走 exact resume 或静默拒绝。
- 修复：训练合同升级为 `p4_maze_soft_cruise_v1`，任务改为 128 env、75 秒、单段
  `open_entry_maze`、20 静态列、7200 秒。关闭速度档采样，Actor policy target 保持
  `vx=[0,1.0]`，Goal stale 与 safety cap 作为 limited target 保护。新增 soft cruise 负奖励：
  clear/Goal fresh 时轻罚 `vx<0.60` 和 `vx>0.75`，terminal tick 不结算。新增 teacher scene、
  SafetyHead risk、Head-correct/Actor-wrong、风险到 policy/limited 减速、zero-hidden shadow
  指标，并将 `student_risk_*` 更名为 `safety_head_risk_*`。新 checkpoint 使用
  `mazeprobe/mazefull/mazefinal` 标签；同合同配置漂移仍 hard fail，旧 P4 合同结构通过时作为父包
  warm start 并重置本轮 session/live state。
- 验证：宿主 `PY311test` 定向回归
  `agent_ppo/tests/test_p4_nav.py` 为 `39 passed`；
  `agent_ppo/tests/test_p2_core.py + agent_ppo/tests/test_nav_stage_and_metrics.py`
  为 `129 passed`。相关 Python 文件 `py_compile` 通过。由于未连接当前开发容器，尚未执行
  真实父包加载、128-env rollout/backward、平台 10 分钟诊断或 2 小时训练 smoke，不能标为平台已验证。
- 2026-08-03 审查修正：后续审查发现初版仍把 600 秒诊断计入 7200 秒 session，导致正式训练
  少约 10 分钟并跳过 Actor 进攻分支首段；auto 分支只读取最后一帧 scanner/cosine，旧 P4
  warm start 会在诊断前固定为 `visual_recovery`，`actor_attack` exact resume 则可能在恢复已选分支
  前触发 optimizer phase mismatch。另有新 `maze*` checkpoint 排在旧 `pnav*` 后、风险减速在同一
  帧同时要求 `vx>=0.50` 和 `vx<=0.35`、默认父包仍指向 P3 `1013548` 等问题。
  现将合同升级为 `p4_maze_soft_cruise_v2_training_clock`：诊断 wall clock 与正式训练 clock 分离，
  并以诊断完成的 rollout 边界作为 7200 秒正式训练起点；exact resume 在父类 phase 校验前恢复
  saved branch，旧合同 warm start 清空 branch、诊断计数、线性探针和探针 optimizer。auto 选择
  不再读取 SafetyHead 或最后一帧值，而是在 detached `nav_feat32` 上在线训练风险/场景线性探针，
  用 held-out 样本累计 teacher coverage、wall AUROC/漏检率、安全方向 top-1、场景 macro-F1 与
  clean/live cosine，并要求明显优于 `goal4` 探针和随机基线；样本不足保守进入
  `visual_recovery`。探针状态随 exact resume 原样恢复。风险减速锁存风险出现时基准速度，并在
  后续 5 tick 结算。checkpoint 优先级改为
  `mazefinal > mazefull > mazeprobe > mazediag > legacy P4`，ID 未命中时仅允许唯一结构兼容 P4
  discovery；P4 周期保存使用 wall clock，诊断第 5 分钟即可产出可恢复的 `mazediag`，但学习率、
  阶段和 7200 秒结束条件仍只使用 training clock。父包更新为本地已核验的
  `pnavstable-1207698`，checkpoint SHA256
  `781022129ac17e564830c34570213a63f111b44d29b9d1bd247c3481c7b9ea55`。本地回归为
  `198 passed, 5 skipped`，Python 编译、全 TOML 解析、P4 面板逐项 `<=20` 校验与
  `git diff --check` 通过；开发容器、平台
  smoke、正式训练和评估仍待验证，状态保持“本地已验证”。
- 2026-08-03 开发容器更正：代码定向同步后，容器
  `agent_ppo/tests/test_p4_nav.py` 为 `46 passed`，真实父包
  `pnavstable-1207698` 的 CUDA 装配与权重加载成功。随后用 CPU 装配复核发现，旧 P4
  合同 warm start 复用父类 exact loader 时会强制恢复父包 CUDA
  `action_rng_state`，不同 generator 后端的状态长度不兼容，从而在新 session 启动前报
  `Expected CPU Generator state size ... but input RNG state size 16`。修复仅针对旧合同
  warm start：保留当前运行时确定性新种子，并记录 `p4_warm_start_rng`
  迁移报告；同合同 exact resume 仍严格恢复并校验 RNG。回归测试故意注入
  16-byte 不兼容状态，要求 warm start 使用新 session 种子而 exact resume 套件保持不变。
- 2026-08-03 开发容器验证：修复后容器定向回归为 `46 passed`。新增的最小
  `p4_maze_continue_smoke.py` 使用真实 `pnavstable-1207698`、8 env、CUDA 完成旧合同
  warm start、32 tick rollout、4 epoch/16 次 PPO update、1 次 Adapter update、checkpoint 保存与
  同合同 exact resume；低层 digest 全程不变。CUDA
  `max_memory_allocated=231,538,176` bytes，`max_memory_reserved=287,309,824` bytes，
  pinned depth 约 `29,491,200` bytes。按用户要求不再运行 128-env 规模测试；后续默认只跑
  定向单测与 8-env 真实父包联合 smoke，仅当 storage shape、相机分辨率或环境装配改变时
  单独启动规模/Isaac 测试。容器中本任务上传父包、解包目录和 smoke checkpoint 已精确清理。
- 2026-08-03 平台首轮 smoke 更正（状态：代码已修复待平台复测）：任务在首次有效 teacher 样本上
  报 `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`，调用链为
  `finish_tick -> _extra_tick_diagnostics -> _accumulate_maze_diagnostic -> backward`。根因是父类
  `finish_tick()` 使用 `torch.no_grad()`，而新增的训练前线性探针沿用了该上下文。现仅在四个
  training-only probe 的 forward/loss/backward/step 区域使用 `torch.enable_grad()`，输入继续
  `detach()`，因此不会给 NavigationEncoder、Actor 或 rollout 建图。回归测试明确在外层
  `torch.no_grad()` 中调用该函数并验证 probe 权重更新、step 后梯度清空。该修复尚未取得新的平台
  smoke 证据，不能把异常消失预先写成平台已验证。
- 防复发：测试覆盖 Maze 配置、软巡航只在 clear/fresh 条件下生效、速度档关闭后 cap 恒为 1.0、
  新 phase label 可保存/发现、同版本 stuck-reset 合同漂移拒绝、旧 P4 包只能 warm start。
- 血缘：父包为 `p4nav8h-r3 pnavstable-1207698`，checkpoint SHA256 为
  `781022129ac17e564830c34570213a63f111b44d29b9d1bd247c3481c7b9ea55`；新输出 checkpoint、
  commit/PR 待生成。
- 回滚：回滚本条 P4 maze 合同、软巡航和面板改名即可恢复 `p4nav8h-r2`；不要回滚
  BUG-20260803-010/011 中相机、Goal、stuck reset、Adapter replay 的已验证修复。
- 再遇检查：`policy_target_vx vs limited_target_vx` -> `soft_cruise_clear_factor` ->
  `safety_head_risk_* vs teacher_risk_*` -> `head_correct_actor_wrong` -> `risk_no_deceleration` ->
  `zero_hidden_action_mae` -> completion/stuck/collision。

## BUG-20260803-013：P4 Maze 长训仍缺少故障视觉对照，单段 Maze 被误记为坡面且事件率分母错误

- 日期：2026-08-03；状态：本地已验证。
- 影响范围：分支 `codex/p4-maze8h-attack`、任务 `p4maze8h-attack`、P4 Maze-only 八小时训练、
  Adapter/Track 分段面板、安全决策面板、wall-stuck terminal 回报与 P4 exact resume。
- 父模型/制品：配置父包 `p4nav8h-r3 pnavstable-1207698`，历史 SHA256
  `781022129ac17e564830c34570213a63f111b44d29b9d1bd247c3481c7b9ea55`；本任务尚未生成新
  checkpoint、模型 ID 或 SHA256。
- 用户可见症状：两小时 Maze 强化不足以验证是否学会迷宫，且现有面板不能可靠区分 clean depth
  可学、轻故障 depth 退化与 Actor 决策失败。单段 `open_entry_maze` 的 worker physical segment
  恒为 0，但继承的 P2 指标固定解释为 `slope_inv`；Head-correct/Actor-wrong 与风险后减速指标又
  除以所有导航 tick，使真实事件率被严重稀释。卡墙 reset 只显示 terminal impulse，无法判断该
  episode 在前期 progress 后是否仍获得非负总回报。
- 根因：P2 三段训练把 physical segment 0/1/2 与 slope/stairs/maze 绑定，P4 单段配置复用该硬编码；
  诊断链只比较 clean/live，没有独立、不可操纵的轻故障对照。workflow 对所有逐环境诊断统一除以
  `num_envs*ticks`，没有为 eligible safety event、resolved risk event 和 stuck terminal 使用各自
  分母。旧两小时 schedule、checkpoint 标签和测试断言也仍散落在 P4 活动入口。
- 排除项：不是通过修改平台覆盖的 `BaseEnv` 解决；不是把模型 ID、标签或 lineage 改成硬门禁；
  不把故障 shadow 输入 Actor，也不通过增加 recurrent hidden 或改变网络 shape 掩盖感知问题。
- 修复：新建 `p4maze8h-attack` 合同，600 秒只读诊断后完整训练 28800 秒；加入独立 RNG 的 10%
  clean-depth 轻故障 shadow，只计算 latent/action 和线性探针指标。安全奖励使用连续 missed-safe
  event ramp，predictive raw 放大 1.25 倍，安全组下限 -0.06；`frontier_stagnation` 实际 PPO 项
  归零并保留 shadow。指标层从 TOML `sub_terrains` 动态把 physical segment 映射到稳定三类，
  单段 Maze 的 index 0 进入 Maze bucket。workflow 使用 eligible/resolved/terminal 条件分母；
  active wall-stuck impulse 为 -15，并额外记录 terminal episode return 和非负率。checkpoint
  保存 fault RNG/统计/分支，并将 reward 合同纳入 exact resume 核验。
- 2026-08-03 集成更正：初版在每个 50Hz 低层帧重采样 fault mask，但 fault feature 只在 5Hz
  高层 tick 计算，tick 结束时可能出现 mask 与 feature 不属于同一帧，同时产生约十倍无效 shadow
  复制。现已把 shadow 生成移动到 `_update_policy_auxiliary_target()` 的同一 5Hz 教师前向，保证
  clean/fault depth、mask、latent 和 action 诊断一一对应，并降低诊断期开销。
- 2026-08-03 事件归因更正：risk-to-deceleration tracker 现在把本 tick terminal 与下一 episode
  reset 一并视为清理边界，terminal 样本不再计入 policy/limited/no-deceleration 条件分子。
- 2026-08-03 独立审查更正：修复 `p2_contract` 错误局部导入导致真实 P2/P4 Agent 初始化
  `NameError`；fault 鲁棒性统计改为只使用 `env_id % 5 == 0` 的 held-out probe 环境；onset tick
  的 limiter 降速不再冒充后续响应。P4 启动核验 Maze-only segment 与实际 slew，exact resume
  改为比较完整 command contract，而非只比较 mapper 版本。
- 修改文件：`server/agent_ppo/feature/p2_contract.py`、`feature/p4_contract.py`、`feature/p4_camera.py`、
  `algorithm/algorithm_p2_nav_ppo.py`、`algorithm/algorithm_p4_nav_ppo.py`、
  `workflow/p2_nav_ppo_workflow.py`、`agent.py`、`checkpoint_io.py`、P4 TOML、监控配置、测试、
  README/CHANGELOG、训练部署接口合同与本台账。未修改 `server/isaac_env/base_env.py`。
- 当前验证：宿主 `PY311test` 定向回归 `test_p4_nav.py + test_p2_core.py +
  test_nav_stage_and_metrics.py` 为 `185 passed`；修改模块 Python 编译、10 份活动 TOML 解析和
  `git diff --check` 均通过。开发容器、平台 smoke、完整八小时训练、固定种子评估与真机均未验证，
  因此状态只到“本地已验证”。
- 防复发：测试覆盖单段 Maze index 0 动态映射、四阶段 8 小时 schedule、missed-safe 权重边界、
  diagnostic fault 不修改 clean 输入且 RNG 可恢复、条件事件率、visual-recovery exact resume、
  stuck-reset -15 与 fault 统计恢复。监控构建继续检查每个 line panel 不超过 20 指标。
- 回滚：回滚本条八小时合同与 fault shadow即可回到 `codex/p4-maze2h-attack`；不得回滚既有相机
  单位、GoalBelief、terminal snapshot、Adapter replay 和 low-level freeze 修复。若平台 fault 指标
  异常，只关闭 diagnostic shadow，不得把它接入 Actor 或 reward。
- 再遇检查：`maze_sample_share` -> `diagnostic_clean_fault_*` -> `diagnostic_fault_*` ->
  `head_correct_actor_wrong_rate` -> `risk_no_deceleration_rate` -> `reward_frontier_stagnation`/shadow ->
  `stuck_terminal_episode_return_mean/nonnegative_rate` -> checkpoint reward/training contract。

## BUG-20260804-001：P4 Track 精简评估无条件调用缺失的 SafetyHead

- 日期：2026-08-04；状态：本地已验证；影响范围：P4 Track eval-only 装配与后续 P4 训练包内代码。
- 症状：任务 `600835` / 运行 `18612264` 在首次 `exploit()` 中止，首个致命错误为
  `self.safety_head(nav_feat.detach())` 抛 `TypeError: 'NoneType' object is not callable`；外层
  `act_data=None` 与 worker 终止均为连锁错误，未形成有效 episode 或评分。
- 根因：`p4_track_eval` 为减小评估状态明确使用 `critic=absent safety_head=absent optimizers=absent`，
  但 `_update_policy_auxiliary_target()` 复用训练路径，无条件调用 training-only SafetyHead。该异常
  发生在相机、终止、scorer 和模型导航能力可验证之前，不能据此归因编码器或策略。
- 修复：缺少 SafetyHead 时原地清零 `_last_safety_head_risk3` 并继续推理；训练装配存在 Head 时仍
  计算 sigmoid 风险。不创建随机模块，不改变 Actor、NavigationEncoder、ResponseAdapter、网络
  shape、checkpoint 或评估终止语义。旧模型最小修复包为
  `p4nav2h_1256446-trackevalfix.zip`，ZIP SHA256
  `4a8a51f38199de79b3cdd7255bcf9489f584bea95e2d9db488b6f9bccaaf545d`；checkpoint SHA256
  `8aae0892664f2f263949f7e5f9f3b53ecd2936b44e80f3783091579bcf3d3461`，字节未变。
- 修改与验证：当前分支直接修改活动 `algorithm_p4_nav_ppo.py` 并增加真实 eval-only Algorithm
  回归；来源修复提交为 `codex/p4-eval-encoder-trace` 的 `c93dd11`。该旧 ZIP 已完成包内编译及
  定向/邻近 `8 passed`；当前分支 P4/P2/装配邻近回归为 `186 passed`，修改算法与测试通过 Python
  编译，`git diff --check` 通过。开发容器与平台 Track 复评尚未完成，因此不能升级为平台或评估
  已验证。
- 防复发：所有 training-only leaf 在 eval-only assembly 中均允许为 `None`，公共推理路径不得
  无条件调用。模型 ID 只记录 lineage，不作为单点门禁。回滚为恢复原算法分支；再次遇到首帧
  失败时先核对 eval assembly absent 模块与第一条调用栈。

## BUG-20260804-002：P4 10Hz 迁移会重复放大连续奖励且八小时墙钟不足

- 日期：2026-08-04；状态：平台已验证，正式长训进行中。
- 影响范围：分支 `codex/p4-maze8h-attack`、任务 `p4maze8h-10hzroute`、P4 高层 10Hz rollout、
  reward 分解、exact resume、平台任务时长和监控面板。父包为 `p4nav2h_1256446-F`，平台模型
  ID `1256446`，checkpoint SHA256
  `8aae0892664f2f263949f7e5f9f3b53ecd2936b44e80f3783091579bcf3d3461`。
- 用户可见风险：将高层从 5Hz 提升到 10Hz 后，如果仍按每 tick 原权重结算 tracking、crawl、
  predictive、missed-safe、yaw、soft-cruise 和持续卡墙，同一秒会得到约两倍负奖励；同时前 600 秒
  只读诊断不计入 `session_effective_seconds`，平台任务若只设置 8 小时，实际梯度训练会少约
  10 分钟。旧速度档测试和 RNG 又继续引用已经删除的运行时状态，不能证明新合同可恢复。
- 根因：P2 reward 原始合同以 10 个 50Hz frame 为一个 5Hz tick，P4 只缩短 period 未同步区分
  连续时间项、事件 impulse、按米路程项和每决策项；TOML 的 `task_end_hours=8.0` 与
  `diagnostic_seconds=600 + target_effective_seconds=28800` 自相矛盾。P4 构造器在缺少显式 config
  时也可能落回 P2 的 5Hz 默认。后续审查还发现 `configure_app.toml` 与容器 smoke 默认仍指向旧父
  模型 `1207698`，且新增 goal-safe 奖励误用了带噪 GoalBelief，与“奖励只用仿真真值”的合同冲突。
  独立审查继续发现：诊断结束时钟零点按首个 rollout 边界设置，使 29400 秒墙钟无法覆盖完整
  28800 秒梯度训练；worker Push 直接使用进程墙钟，提前把 600 秒诊断计入 2 小时边界；10Hz
  安全组先做 0.5 时间缩放再 cap，使 `-0.06` cap 永远无法触发；2 秒卡滞窗口仍硬编码 11 点，
  在 10Hz deque 长到 21 点后失效；mid-tick checkpoint exact resume 恢复非整 tick `frame_count`，
  会把新 episode 前几帧 reward 配给后创建的 action/log-prob；workflow 二次 live reset 又消耗已恢复的
  camera RNG；风险减速指标还用 policy onset 速度充当 limited onset 基线，制造 limiter 减速假阳性。
  风险事件年龄又在 onset tick 立即加一，使 1.0 秒响应窗口实际约 0.9 秒结束。
- 修复：P4 构造器在父类初始化前注入 `nav_period_frames=5`，P2 默认仍为 10 frame。连续 tick
  奖励统一乘 `duration_frames/10`，正常 10Hz tick 缩放为 0.5；success/failure/timeout/reset 等
  事件 impulse、route-excess 按米项和 command-rate 每决策项不缩放，body collision 明确保留
  10Hz 完整约束。成功 impulse 改为 `+200`，新增持续卡墙、目标一致安全选向和轻量额外路程惩罚。
  旧 slow/cruise/fast 状态与 RNG 全部移除，exact resume 恢复诊断分支后重新应用对应 LR。
  合同新增 `required_platform_wall_seconds=29400`，TOML 元数据改为 8 小时 10 分钟，梯度目标仍为
  28800 秒。为保留干净 rollout 边界，再增加 300 秒 rollout/checkpoint/退出余量，平台最大墙钟改为
  29700 秒，工作流达到有效训练目标后自行结束。worker Push 在诊断未完成时使用剩余诊断时间的负
  offset，诊断完成后使用保存的有效训练时钟。安全组三项先在参考 5Hz 尺度 cap，再统一乘
  `duration_frames/10`；卡滞历史按 deque 实际 `maxlen`；exact resume 将 `frame_count` 向上对齐到
  下一导航 tick，camera live buffer reset 不消费恢复 RNG；policy/limited 分别保存风险 onset 基线。
  风险年龄只累计 onset 后续 tick，严格在第 10 个 10Hz 后续 tick 结算 1.0 秒响应窗口。
  活动预加载和 smoke fallback 统一为 `1256446`；goal-safe 奖励改读 terminal-safe
  `raw_goal_xy_m2`，并用回归测试证明任意修改 GoalBelief estimate 不会改变该奖励。
- 修改文件：`algorithm_p2_nav_ppo.py`、`algorithm_p4_nav_ppo.py`、
  `workflow/p2_nav_ppo_workflow.py`、`feature/p4_contract.py`、P4 TOML、监控、测试、README、
  CHANGELOG、接口合同和本台账。未修改平台覆盖的 `server/isaac_env/base_env.py`。
- 当前验证：宿主 `PY311test` 的 P4 专项为 `65 passed`；P4/P2/监控/双评估合计
  `219 passed, 5 skipped`，另有 checkpoint 邻近回归 `19 passed`；改动 Python 文件通过编译、
  14 份 TOML 通过解析，
  `git diff --check` 通过。新增用例覆盖 P4 默认 10Hz、P2 默认不变、连续奖励 0.5 缩放、成功
  `+200`、卡墙 ramp、安全目标方向、route-excess、GoalBelief 不污染奖励、活动父包一致性、速度档
  删除与 29700 秒墙钟合同。开发容器中定向测试为 `11 passed`；随后使用 checkpoint SHA256
  `8aae0892664f2f263949f7e5f9f3b53ecd2936b44e80f3783091579bcf3d3461`、8 环境完成真实父包
  联合 smoke：`p4_maze_warm_start`、32-tick rollout、16 次 PPO minibatch update、1 次 Adapter
  update、低层 digest 不变、保存后 fresh instance 以 `p4_exact_resume_history_reset` 恢复。峰值
  CUDA allocated/reserved 分别为 `231542784/287309824` bytes，pinned depth 为 `29491200` bytes。
  测试 ZIP、解压父包、smoke checkpoint 和 `/tmp` 日志已经按精确路径清理。2026-08-04 在独立
  审查修复后重新同步 37 个源码/测试文件，二次 dry-run 为 0 差异；容器正确 Isaac Python 路径下
  9 项定向回归通过，覆盖 Push 时钟、10Hz safety cap、camera RNG、卡滞窗口、风险减速/完整 1 秒
  响应窗口、P4/P2 exact resume。随后重新上传本地父包 ZIP，WebIDE 双 SHA256 校验为
  `4a8a51f38199de79b3cdd7255bcf9489f584bea95e2d9db488b6f9bccaaf545d`，解压 checkpoint 校验为
  `8aae0892664f2f263949f7e5f9f3b53ecd2936b44e80f3783091579bcf3d3461`；8 环境联合 smoke 再次
  `PASS`，完成 32-tick rollout、16 次 PPO minibatch update、1 次 Adapter update、保存和
  `p4_exact_resume_history_reset`，低层 digest 不变，峰值 CUDA allocated/reserved 为
  `231543296/287309824` bytes，pinned depth 为 `29491200` bytes。父包和本轮输出暂时保留到正式
  任务首轮更新稳定，随后按精确路径清理。正式平台训练尚待执行，因此不能升级为“平台已验证”。
- 平台 UI 证据：已用父模型 `p4nav2h_1256446-F` 创建并手动释放 1 分钟试验任务
  `p4create-stop-smoke`（任务 ID `236257`），仅证明任务表单、父模型选择和停止流程可用，不证明
  代码、10Hz、checkpoint 或训练行为正确。正式任务第一次提交返回“训练任务创建失败”；容器清理
  dry-run 发现 `10` 个可再生目标、`40149025` bytes，均为 `__pycache__`、`test_artifacts` 和
  `/tmp/IsaacLab`，未包含 `conf/.env` 或凭证。执行 `conf/container_training_cleanup.py --apply`
  后，同一表单成功创建 `p4maze8h-10hzroute`（任务 ID `236263`，平台时长 `8h10min`，父模型
  `p4nav2h_1256446-F`）。任务启动后进入 `mazediag`，低层 digest 未漂移，worker lifecycle 未失败；
  模型列表在 `5min25s` 和 `5min39s` 分别出现训练步数 `1259039/1259166`、大小 `28.49M` 的
  checkpoint，证明 5 分钟首存和手动释放终存链路可用。按用户要求任务在约 6 分钟手动释放，状态为
  `手动释放`，CPU/GPU 资源已归还。由于尚未越过 600 秒只读诊断边界，该证据不能证明训练分支选择、
  首轮 PPO/Adapter 梯度更新或长期漂移；正式任务仍需重新创建并继续验证。
- 防复发：定向测试必须同时断言 P4 period=5 frame、P2 period=10 frame、32 tick/TBPTT16、
  连续奖励每秒尺度、事件 impulse 不缩放、旧 speed RNG 不再写入 checkpoint、exact resume LR
  与诊断分支一致。正式任务页不得用 8 小时覆盖 10 分钟诊断加完整 8 小时训练。
- 回滚：回滚本条 10Hz、三项新奖励和时钟合同即可恢复 5Hz Maze 版本；不得回滚 GoalBelief、
  相机单位、terminal snapshot、active stuck reset 或 eval-only SafetyHead 已验证修复。
- 再遇检查：`nav_period_frames` -> `reward_continuous_time_scale` -> reward conservation ->
  `session_wall_seconds/session_effective_seconds` -> saved training/reward/command digest -> 首个 PPO update。

### 2026-08-04 正式任务阶段性追加

- 正式任务 `p4maze8h10hz-r2` 已创建并运行，平台任务 ID `236264`，UUID
  `7946801e-bf35-41b6-bd7a-63614d1960ee`，父模型为 `p4nav2h_1256446-F`，平台墙钟为
  `8h15min`。任务已越过 600 秒只读诊断并自动选择 `visual_recovery`；前约 30 分钟未出现
  OOM、非有限值、生命周期失败或低层 digest 漂移，PPO/Adapter 持续更新，10 分钟周期保存已成功。
- 平台精确指标显示：诊断 teacher coverage 为 `1.0`，wall AUROC 约 `0.879`、墙面漏检率约
  `0.329`、安全方向 top-1 约 `0.791`、scene macro-F1 约 `0.428`；clean/fault 指标接近，符合
  视觉恢复分支选择。PPO KL 约 `0.004-0.010`、clip fraction 约 `0.04-0.09`，峰值 reserved
  显存约 `1.10GB`，reward conservation error 为 `0`，低层 optimizer steps 为 `0`。
- 发现平台 logger 不兼容 Python logging 的 printf 参数转发，导致自动分支和 exact-resume 对齐日志
  显示未展开的 `%s/%d`。已改为调用前完成字符串格式化；这是可观测性修复，不改变正在运行任务的
  数值行为，当前任务无需为此重启。待下一次任务或 exact resume smoke 验证新日志文本。

### 2026-08-04 `mazeprobe -> mazeattack` 平台验证追加

- 正式任务在 `session_effective_seconds=3702.76` 的干净 rollout 边界将
  `maze_phase_probe/maze_phase_attack` 从 `1/0` 切换为 `0/1`。实际学习率同步从 probe 的
  `navigation=2.25e-5, actor=7.5e-5, critic=3.0e-4, safety=4.5e-4, adapter=5e-6`
  切换为 attack 的 `navigation=1.8e-5, actor=2.55e-4, critic=2.4e-4,
  safety=3.0e-4, adapter=5e-6`，与 `visual_recovery` 课程合同一致。
- 切换后连续观察 6 个独立高层 update rollout：`approx_kl` 在约 `0.015-0.022`
  稳定，`clip_fraction` 在约 `0.16-0.18` 稳定，未继续单调上冲；吞吐约
  `155-158 samples/s`，`max_memory_reserved=1103101952` bytes 保持不变，
  `high_updates` 从 `13888` 持续增长到 `14128`。`low_digest_drift=0` 且
  `reward_conservation_error=0`，证明低层冻结、reward 结算和 10Hz 更新链未因阶段切换回归。
- 行为指标在 rollout 间仍有显著噪声：切换后 `route_efficiency` 约 `0.40-0.52`、
  `goal_progress_m_per_s` 约 `0.096-0.228`、`body_collision_onset` 约
  `0.021-0.027`；`head_correct_actor_wrong_rate` 约 `0.39-0.55`，尚未呈现稳定下降。
  这些证据支持“训练链路正常推进，暂无需中断或重训”，但不证明最终迷宫完成率已提升。
  最终导航能力、提前避障、S-turn 和路径效率仍需在长训后通过固定种子 Track 评估和视频验证，
  不得将平台训练稳定性等同为评估已验证。

## BUG-20260804-003：P4 碰撞后卡死、斜移替代转向与近终点错过缺少训练闭环

- 日期：2026-08-04；状态：平台启动与更新链已验证，行为效果待完整训练和评估。
- 影响范围：分支 `codex/p4-maze3h-recovery`、任务 `p4maze3h-recovery-r2`、P4 Maze 10Hz
  高层 Actor、NavigationEncoder、SafetyHead、training-only StuckHead、wall-stuck reset、命令 slew、
  safety reward、checkpoint continuation 和监控面板。未修改平台覆盖的 `server/isaac_env/base_env.py`。
- 父模型/制品：`p4maze8h10hz_1416926.zip`，ZIP SHA256
  `106908c8830f4fc7989125372f6ed397366f7ca1b4aac071128dccf71add0f6c`；checkpoint
  `model.ckpt-mazefinal-1416926.pkl`，SHA256
  `0bf54e3c18e6450492d7c1d44d69596d79beddbebb57166432220f8cb93dd2e4`。
- 用户可见症状：上一轮评估中多次出现前腿或后腿接触柱体后长期不脱困、目标已在侧前方仍以
  `vy` 斜移而不建立正确 `wz`、target yaw 建立后 exec/true yaw 滞后、绕障后继续撞击，以及进入
  终点附近却离开成功半径。训练面板又显示 Head 正确但 Actor 选错率仍高，说明只改善诊断 Head
  不能直接改变策略动作。
- 根因：旧训练仅把 safe3 用于 SafetyHead 和轻量 reward，没有在 rollout 时间轴上给 Actor mean
  施加容差引导；安全 limiter 主要限制 `vx`，留下 `vy` 斜向碰墙；`wz` slew 建立速度偏慢；卡墙
  reset 为 7 秒但 Actor 没有独立 stuck 表征；近终点 soft-cruise/crawl 与主动减速存在冲突。实现
  期间进一步发现真实父包 Actor optimizer 只有 8 个参数组，而新增 StuckHead 后运行时为 9 组，
  直接 exact-loader 会在 `optimizer.load_state_dict()` 因 group count 不一致失败。合成旧合同测试因
  使用当前代码生成包而错误携带第 9 组，未覆盖该真实迁移边界。
- 修复：新增 rollout-time Actor mean 教师，只在 scanner/mapping/Goal 等高置信条件下处罚超出
  35 度安全方向容差、近墙过快和错误 yaw 符号；教师路径 detach `nav_feat32`，Camera clean/live
  辅助仍允许更新 NavigationEncoder。增加 training-only StuckHead、2x SafetyHead hard-positive
  权重和从 rollout 内任意真实 reset 起点构造的零-hidden完整 TBPTT16 sequence mirror；镜像按
  全部 PPO sequence-update 的 10% 调度，合格 episode 少时有放回抽样。四类辅助梯度采用总 5%、
  teacher 3%、mirror 1% 上限。固定 PPO chunk 的中途非零 recurrent context 不做伪镜像。
  教师 64-step 门槛按完整 rollout 统计，不再错误要求单个 64-frame microbatch 全部有效。
- 命令/奖励修复：slew 改为 `vx 0.30/0.30`、`vy 0.40/0.80`、`wz 1.50/3.00`；部署可得 risk
  同比例缩放完整 `(vx,vy)`，不压缩 `wz`。近终点 `0.65-1.20m` 只做平移软捕获并屏蔽 crawl/低速
  巡航惩罚。predictive、missed-safe、yaw cancellation、yaw-exit 和 goal-safe preference 统一进入
  原 5Hz 等效 `-0.06` 安全预算；reason 4 保持 `-15`，同 terminal tick 不重复 safety/collision。
  wall-stuck 确认合同延长到 10 秒，Push 全程关闭；当前 shadow 配置不生成 reason 4。
- checkpoint 修复：旧 `1416926` 包只能 continuation warm start；按稳定 optimizer group 名和组内
  参数顺序恢复 8 个旧组共 26 个参数的 Adam state，新 `actor_stuck_head` 保持空 state。新合同
  exact resume 保存/恢复 StuckHead leaf、mirror RNG、auxiliary calibration 和 Adapter completed
  records。模型 ID和标签继续只用于选择，不作为结构正确时的单点门禁。
- 平台边界：训练侧 stuck label 能同时读取 policy target、aisrv exec、true velocity 和墙面证据；
  worker 物理 reset 不修改 BaseEnv，也没有公开 P4 exec transport。原生 `base_velocity` 不能替代
  P4 实际 exec command，否则主动停车会被误杀。现将 training wire 升级为 509，并加入显式
  motion-intent hook 可用性与独立 raw wall term；缺少 hook 时 active reset fail-closed，正式 TOML
  保持 `shadow`。进一步定位到 `_termination_reason_codes()` 的原地布尔掩码会修改调用方传入的
  wall term，success 优先时连原始证据也会被清零；现对所有 manager term 输入先 clone，再计算互斥
  reason。raw wall term 因而不受 success/failure reason 优先级覆盖，可核验同帧终止归因冲突。
  世界位移、低 true velocity、墙面证据和候选时长继续用于诊断与训练侧标签，但本轮不声称具备
  10 秒真实物理 reset。
- 监控口径修复：原实现把 `wall_stuck_reset_triggered` 累计成 recovery event，实际会把失败 reset
  误报为成功脱困。现分别统计 candidate entry、带 true XY 运动或正进度证据的 recovery success、
  墙面证据暂失但未确认运动的 unverified exit 边沿事件、候选中 terminal、恢复耗时、early/confirmed sample
  share，以及 60 秒/lifetime 成功恢复计数；reset 永不进入 recovery success 或 completion。
  unverified exit 进入 awaiting-evidence 状态，证据持续缺失时不再逐 tick 重复累计。reset/completion
  交叉检查读取独立 raw wall term；exact resume 恢复 recovery 事件时间戳和三类
  lifetime 计数，避免续训后面板归零。
- 本地验证：审查修复后的 `PY311test` P4/P2/监控邻近回归为 `229 passed`。真实父 checkpoint 已成功以
  `p4_maze_warm_start` 加载：恢复 8 个 optimizer groups、26 个参数 state，新 Head state 为 0，
  Adam step 范围为 `26784-27584`，session 时钟归零，Adapter current/parent completed records
  分别恢复 32/64。最新代码又使用真实父包完成 1-env、32-tick PPO、1 次 Adapter update、保存和
  `p4_exact_resume_history_reset` 联合 smoke，低层 digest 保持不变；同时修复 smoke 工具在 CPU
  设备上无条件调用 CUDA 峰值统计和 `auto` 分支在 P4 诊断 buffer 创建前解析的初始化错误。全部
  修改 Python 编译、14 个活动 TOML 解析、`git diff --check`、BaseEnv 未修改
  和无 `archive/shared` 运行时依赖检查均通过。此处是修复完成时的本地证据快照；后续容器与平台
  启动链复验见本条末尾，三小时完整任务、固定种子评估和真机仍未验证。
  容器首次调用 smoke 时还发现 Isaac 预置 Torch 的 CUDA memory API 不接受 `torch.device` 参数；
  测试工具现仅在显存统计调用中转换为整数 device index，不改变训练设备或算法语义。
- 防复发：回归覆盖真实 8->9 optimizer group 迁移、Camera 辅助视觉梯度、rollout-wide teacher
  门槛、teacher/mirror/stuck TBPTT 对齐、镜像 involution、slew 先归零、vector limiter、近终点捕获、
  safety reward conservation、reason 4 互斥和 eval 不创建 StuckHead。以后 previous-contract 测试必须
  删除新增 leaf/group，不能仅修改 version 字符串冒充旧包。
- 回滚：回滚本条 recovery contract、StuckHead、teacher/mirror、limiter/capture 和新 slew 即可回到
  `p4_maze_attack10hz_v1`；不得回滚已验证的 eval-only SafetyHead、GoalBelief、相机单位、terminal
  snapshot 和 10Hz duration scaling。再遇加载失败先检查 optimizer group name/count，再检查新 leaf
  exact-resume 合同；再遇卡墙误杀先看 mapping、true speed、wall latch 和 reason 4，而不是调整模型 ID。
- 关联 commit/PR：待生成；平台替代任务 ID `236395`，首个输出 checkpoint 和 SHA256 见本条末尾。

### 2026-08-04 平台首轮更正：FP16 mirror auxiliary 未进入 AMP

- 平台任务 `p4maze3h-recovery-r1`（任务 ID `236388`，UUID
  `85bd42f2-7950-402b-9c12-58a980073b93`）正确装配 P4、父包 `1416926`、128 环境和单段 Maze，
  但在第一次真实 PPO update 失败，关键日志为：
  `RuntimeError: Input type (c10::Half) and bias type (float) should be the same`。
- 根因是 mirror auxiliary 从 CPU pinned rollout 读取 FP16 depth 后，直接调用 FP32
  `NavigationEncoder`；主 PPO CNN 路径已使用 autocast，而该新增辅助路径漏掉了同样的 dtype
  边界。本地 CPU smoke 会显式使用 FP32，先前容器联合 smoke 又没有抽中真实 mirror batch，因而
  未覆盖 `CUDA + FP16 depth + mirror auxiliary` 的组合。这与模型 ID、父包加载、reset、显存或容器
  文件数量无关。
- 修复为：CUDA mirror CNN 重算进入 `torch.autocast`，CPU 路径显式转 FP32，输出 feature 再恢复
  FP32 供 recurrent actor 使用；测试强制使用 FP16 mirror depth，并继续断言视觉 CNN 不接收 mirror
  梯度。正式替代任务统一改名为 `p4maze3h-recovery-r2`。
- 验证层级：本地专项及邻近回归 `229 passed`，真实父包 CPU 1-env 联合 smoke 通过。随后同步
  热修复到开发容器，确定性构造 `CUDA + FP16 depth + 16-step mirror batch`，结果
  `fp16_mirror_auxiliary=PASS`；同一进程继续完成真实父包 warm start、32-tick PPO（8 次
  minibatch update）、1 次 Adapter update、保存和 `p4_exact_resume_history_reset`，低层 digest
  未漂移。峰值 CUDA allocated/reserved 为 `87217152/111149056` bytes。测试 ZIP、解压 checkpoint
  和 smoke 输出已按 `agent_ppo/test_artifacts/p4_maze3h` 精确清理。替代平台任务仍待创建和复验；
  `236388` 只作为失败证据，不代表有效训练进度，也不得用于评估能力。

### 2026-08-04 平台替代任务复验

- 替代任务 `p4maze3h-recovery-r2` 已创建并运行：任务 ID `236395`，UUID
  `a95cea3d-52c2-4e3e-8265-286f6a7cc189`，父模型 `p4maze8h10hz_1416926`，128 环境、单段
  `open_entry_maze`、3 小时有效训练合同。模型 ID 仅用于选择父包，实际加载仍通过模块、spec、
  shape、有限值和 optimizer group 迁移校验。
- 截至 `effective_min=15.0`、`iter=28`，任务已连续越过首轮任务的第一次 PPO update 崩溃点，
  `actor/critic/adapter` 均为有限值，`lifecycle_fail=0`，`low_digest_drift=0`，
  `low_optimizer_steps=0`，`high_updates=28032`。GPU 峰值 allocated/reserved 约为
  `820271616/1021313024` bytes，pinned depth storage 为 `471859200` bytes，未出现 OOM 或 dtype
  异常。`mazeprobe` 是 0-20 分钟的低学习率恢复训练阶段，不是只读诊断阶段。
- 5 分钟首次保存已成功生成 `model.ckpt-mazeprobe-1418526.pkl`，SHA256
  `c0fedf70a27afb371fd9d2330911137af5bb073c05438be632d01985cbbce53e`，平台归档 ZIP 同步完成。
  15 分钟周期保存再次成功生成 `model.ckpt-mazeprobe-1421566.pkl`，SHA256
  `ce0180661beb950033f51ecf998215e15268e4c53af8bedd91f8c846dcc4411d`。
  EnvMonitor 持续上报 321 个指标；Adapter 三池目标比例为 `0.50/0.25/0.25`，兼容性拒绝计数为
  0。该证据验证的是平台装配、真实 CUDA PPO/辅助反向、Adapter 更新、低层冻结和保存链，不证明
  三小时后的 Maze 完成率、碰撞恢复或真机能力；这些仍需完整训练 checkpoint 和固定种子评估。

## BUG-20260804-004：开发容器 smoke 反复选错 Isaac Python 入口

- 日期：2026-08-04；状态：容器已验证，流程防线已补充。
- 影响范围：腾讯开悟开发容器中的 Isaac/PyTorch smoke、checkpoint 加载、PPO/Adapter 联合测试；
  不影响平台训练进程自身的 Python 启动栈。
- 用户可见症状：直接调用 Conda Python，或在当前镜像运行
  `/workspace/isaaclab/isaaclab.sh -p` 时，解释器没有注入 Isaac Sim 的
  `omni.isaac.ml_archive/pip_prebundle`，在导入阶段报 `ModuleNotFoundError: No module named
  'torch'`。该症状在 P2、P3 和 P4 容器 smoke 中多次出现，曾被误判为“容器没有 PyTorch”。
- 根因：平台训练栈、Isaac Sim 自带 Python 和普通 Conda Python 是三个不同的依赖环境。
  `isaaclab.sh -p` 的实际选择会随平台镜像变化，不能把历史上可用的 wrapper 当成稳定 ABI；
  直接 Conda 启动也不会自动获得 Isaac Sim 预置的 Torch 路径。
- 当前容器证据：`/workspace/isaaclab/_isaac_sim/python.sh` 可导入 Torch
  `2.7.0+cu128`，`torch.cuda.is_available()` 为 true，并已完成真实父 checkpoint、CUDA FP16
  mirror、32-tick PPO、Adapter update、save/resume smoke。相同代码用当前
  `isaaclab.sh -p` 在导入阶段停止，证明这是解释器入口问题，不是 checkpoint SHA、模型 ID、
  算法代码或 GPU 不可用。
- 修复与防复发：`server/README.md` 统一记录解释器预检和 smoke 命令；活动工具文档不再推荐
  `isaaclab.sh -p`。以后容器测试先执行
  `/workspace/isaaclab/_isaac_sim/python.sh -c 'import torch; print(torch.__version__,
  torch.cuda.is_available())'`，预检成功后全程复用同一入口。若该路径不存在或预检失败，必须先
  在线定位当前镜像入口，禁止静默回退到 Conda Python，也禁止把 import failure 归因于模型或代码。
- 验证边界：本条只验证当前开发容器解释器和 CUDA smoke；未来平台升级镜像后仍须重新执行
  一行预检。回滚只涉及文档说明，不改变训练代码、环境配置或 checkpoint。
- 关联任务：`p4maze3h-recovery-r2`，任务 ID `236395`；commit/PR：待生成。
## BUG-20260805-001：P4 全赛道方案存在出生覆盖假象、五段统计漂移和大角速度 S-turn 逃逸

- 日期：2026-08-05；状态：平台启动与首次保存已验证，正式训练进行中，训练效果待验证。
- 影响范围：`codex/p4-full8h-r2` 的五段 Track 出生、reason 4 卡滞回收、开阔直行奖励、
  P4 training wire、监控归因和长距离 Goal 编码；不修改平台覆盖的 `BaseEnv`。
- 父制品：`p4maze8h10hz_1416926.zip`，ZIP SHA256
  `106908c8830f4fc7989125372f6ed397366f7ca1b4aac071128dccf71add0f6c`；内部 checkpoint
  `model.ckpt-mazefinal-1416926.pkl` SHA256
  `0bf54e3c18e6450492d7c1d44d69596d79beddbebb57166432220f8cb93dd2e4`。
- 用户可见风险：
  1. 计划要求 Maze 内 30% 困难随机位置，但初版即使完成 horizontal clearance raycast，仍显式
     排除 Maze，使全部困难 Maze 样本回落入口；监控可能显示“请求了 hard spawn”，实际未覆盖。
  2. 聚合仍混用旧三段/迷宫命名，出生段、终止所在段和 reset 后新 episode 容易被混为一类，
     正负 `vy/wz` 均值也会互相抵消。
  3. 开阔 S-turn 惩罚被当前 `|wz|` 反向减权，大幅左右交替反而容易逃逸。
  4. 旧 509 wire 无法携带出生段、分位、安全点和 reason4 fallback 事实，面板空值或错误口径无法
     自检。
- 根因：
  - 将“垂直表面 hit 不足以证明 Maze 安全”的旧判断保留到了已经包含八方向 clearance 的新查询后，
    形成过度保守 fallback。
  - 训练从单 Maze/三段历史迭代而来，workflow、monitor 和 worker histogram 没有一次性升级到五段
    单一事实来源。
  - anti-S-turn 公式重复使用当前角速度幅度去保护合法转向，但历史 cancellation 本身已经能区分
    单向转弯和反复换向。
- 修复：
  - 新增 P4-only `p4_spawn.py`，按整轨/五段/四分位/safe-hard quota 调度；困难位置只有表面 hit
    与八方向 0.35m clearance 均有效才写入，Maze 不再被无条件排除。验证不可用仍 fail-closed。
  - reason 4 同桶重试两次后退回同段入口；wire 扩为 519，显式传输出生事实和累计失败计数。
  - workflow 分离出生段与 terminal current segment，分别统计四类 outcome；正负 `vy/wz` 的
    policy/limited/mapped/exec/true 链独立聚合，累计 counter 使用 max/latest 而不是每 tick 重复求和。
  - 五段 current/spawn、Goal clean/fault、跳变径向/切向、frontier、open-straight 和 active stuck
    面板改为实际生产的键；每个 line 面板不超过 20 项。
  - anti-S-turn 直接使用历史 cancellation，移除按当前 `|wz|` 削弱的逃逸口；仍受 open-slope、
    teacher、Goal、边界、路口、接触和恢复门控，总下限 `-0.0125/tick`。
  - Goal 编码 10m 内保持逐值兼容，10m 外保存单位方位；server、deploy 和共享接口合同原子更新。
  - 2026-08-05 独立终审补充发现四项阻断并修复：P4 reset spawn hook 安装失败不再只记录日志；
    active `nav_stuck_timeout` 允许平台装配延迟最多 8 帧，仍不可用则明确失败；物理 root-state 写入
    异常累计后立即抛错。同帧 wall-stuck 现在优先于 success/failure/timeout，确保 reason4 reset
    不获得 `+200`、不进入完成统计。
  - 出生覆盖改为 `transition_done` reset edge 的事件统计：整轨/段内、五段、四分位和 safe/hard
    分母均来自真实新 episode；L0-L9/L10-L19 仅表示静态难度列。worker tail 的废弃 motion-intent
    位改为 `p4_spawn_hook_installed`，并上报物理写入/验证失败。
  - 部署 `UwbGoal` 增加 `ActorGoalEncoding` 版本。现有 Actor80 默认保持 legacy 逐轴 clamp；P4
    方向保持 v2 只能由未来匹配 spec 的可部署制品显式选择，修正了 non-deployable 训练合同会改变
    稳定部署入口的跨端漂移。
- 已排除的错误方向：
  - 未修改 `server/isaac_env/base_env.py`；平台会覆盖该文件。
  - 未用模型 ID、训练分数或 hard-spawn 成功率作为单点加载门禁。
  - 未启用全局最短路惩罚、Maze heading reward、固定速度映射、两次反转确认或运行时动作覆盖。
- 本地验证：
  - `PYTHONPATH=server .../PY311test/bin/python -m pytest -q server/agent_ppo/tests/test_p4_*.py`：
    `116 passed`。
  - 新回归覆盖 3/10/20/40m Goal 方位、五段映射、reason4 同桶/同段退回、困难 Maze spawn
    验证、滑动卡滞窗口、大 `|wz|` S-turn 逃逸和 Maze/边界免罚。
  - 修复终审阻断后的 P4 定向回归：`118 passed`；新增 writer failure、active term bounded retry、
    wall-stuck/success overlap 和 reset-event 面板合同回归。
  - Python 编译、TOML、`git diff --check` 与 `BaseEnv` 未修改检查通过；legacy/v2 两个 C++
    UwbGoal 测试均编译运行通过。
- 开发容器验证（2026-08-05）：
  - 最终同步代码的 P4 定向回归通过：
    `test_p4_nav.py + test_p4_full_track_spawn.py + test_p4_full_track_monitor.py`
    共 `84 passed`。容器解释器为 `env_isaaclab` Python，并保留 Isaac Sim 预置
    `PYTHONPATH`，避免再现“有 pytest 无 torch”的假性环境错误。
  - 父 ZIP SHA256 校验为
    `106908c8830f4fc7989125372f6ed397366f7ca1b4aac071128dccf71add0f6c`，内部
    checkpoint SHA256 校验为
    `0bf54e3c18e6450492d7c1d44d69596d79beddbebb57166432220f8cb93dd2e4`。
  - 8-env CUDA 联合 smoke 通过：`p4_full_track_warm_start`、32-tick rollout、16 次 PPO
    update、Adapter update、FP16 mirror auxiliary、save 和 fresh-process
    `p4_exact_resume_history_reset` 均成功；低层 digest 未漂移。峰值
    `memory_allocated=230938112`、`memory_reserved=295698432`，pinned depth
    `29491200` bytes。
  - 1-env Isaac 工具在补齐 IsaacLab/Unitree 源码 `PYTHONPATH` 后进入 SimulationApp，
    显存上升到约 1.9 GiB；但开发 WebIDE 在进入 reset 断言前两次返回
    `WEBIDE_RECORD_NOT_FOUND` 并回收资源，未产生 exit marker。因此不能声称
    reason4 物理 reset 和 Maze 出生 readback 已通过；改由有界平台 smoke
    检查启动 hard-fail、spawn 事件指标和 reason4 终止链。
- 待验证：
  - 当前平台镜像清单生成于 2026-07-29，不能证明 2026-08-05 容器的 RayCaster mesh、EventManager
    和 termination term。必须在开发容器完成 1-env physical reset 与随机出生 readback。
  - worker EventManager 没有 learner checkpoint transport，fresh-process resume 后 spawn quota/RNG
    会按 seed 重启。文档已明确，未宣称完整位级 exact resume。
  - 8-env PPO/Adapter/save-resume smoke 已完成；平台 reason4 物理 reset、五段面板和
    正式八小时训练结果尚未完成。
- 回滚：关闭 `[p4_nav_ppo.full_track_spawn].enabled` 可恢复平台默认出生；将 stuck
  `mode=shadow` 可关闭物理 reason4；旧父包仍可按结构 warm start。不得回滚 Goal 跨端编码的任一
  单端实现，必须 server/deploy/shared 同步回滚。
- 关联 commit/PR、任务 ID、新 checkpoint：待生成。

### 2026-08-05 平台 smoke 更正：worker bridge P4 标志初始化顺序

- 平台 smoke `p4full-r2-smoke`（任务 ID `236441`，父模型
  `p4maze8h10hz_1416926`）确认活动配置已经是 128 环境、五段固定顺序、20 列和
  `curriculum=false`，但首次环境 reset 在 ObservationManager 装配期停止。关键日志为：
  `AttributeError: 'P2WorkerBridge' object has no attribute '_p4_enabled'`，随后 workflow 只看到
  `RuntimeError: P2 env.reset returned None`。
- 根因是本轮新增的 worker 启动诊断在 `P2WorkerBridge.__init__` 尾部读取 `_p4_enabled`，但该
  成员没有稳定的初始化来源。容器单测和联合算法 smoke 没有构造完整平台 ObservationManager，
  因而没有覆盖这一真实构造路径。这与父 checkpoint、模型 ID、容器磁盘、五段地形或
  `BaseEnv` 无关。
- 修复为只读派生属性：`_p4_enabled` 始终由已经解析的
  `runtime_stage_type == "p4_nav_ppo"` 得出，不再依赖赋值顺序。新增回归覆盖 P4 train、P4 eval
  与 P2 train 的 wire 选择边界。
- 当前状态：代码已修复待平台复验。任务 `236441` 只作为失败证据，不代表训练开始；修复同步后
  必须重新创建有界 smoke，确认环境 reset、spawn/stuck preflight、首轮 PPO 和保存链后才创建
  正式八小时任务。
- 第二次 smoke `p4full-r2-smoke2`（任务 ID `236444`）越过上述属性异常后，在首次 worker
  `step()` 暴露 quota 设备混用：`spawn.last_full` 已迁移到 CUDA，但
  `spawn.last_segment` 仍为 CPU，`torch.where` 报
  `Expected all tensors to be on the same device`。修复为先将全部 CPU-owned spawn quota tensor
  迁移到 worker device，再组合 519 wire；不改变 quota 采样、RNG 或出生写入。该任务同样只作为
  失败证据，正式任务仍等待下一次平台 smoke 通过。

### 2026-08-05 第三次平台 smoke：训练主链通过，监控标题校验失败

- 第三次 smoke `p4full-r2-smoke3`（任务 ID `236445`，UUID
  `a9058628-afa6-48c8-bb85-9f9e70218b07`）越过前两次 worker bridge 构造和设备问题，完成 128
  环境 reset 并运行到 `iter=3`、`effective_min=1.6`。Actor、Critic 和 Adapter 指标均有限，
  `lifecycle_fail=0`、`low_digest_drift=0`、`low_optimizer_steps=0`；五段固定顺序、20 列、
  `curriculum=false` 和 450 MiB pinned depth storage 均按正式合同装配。
- 平台同时报告自定义监控配置校验失败：`出生五段覆盖（Reset事件）` 和
  `整轨与段内位置四分位（Reset事件）` 含平台不接受的中文括号，导致整份
  `monitor_builder.py` 被跳过。该错误不影响 PPO 数据流，但会让正式任务缺失 P4 专用面板。
- 修复为平台兼容的短标题 `出生五段重置覆盖` 和 `出生位置与四分位`，并新增长度 1-20、仅
  中英文/数字/下划线/连字符/空格的静态回归。第三次 smoke 已手动释放；正式任务必须使用同步
  后的监控配置创建，并确认启动日志不再出现 `Error occurred while loading user monitor config`。

### 2026-08-05 正式八小时任务启动验证

- 正式任务 `p4full8h-r2` 已创建：任务 ID `236446`，UUID
  `2db48b40-5028-4ca3-b880-89edd3f8520a`，父模型 `p4maze8h10hz_1416926`，任务页时长
  `8h15min`，代码内有效训练目标 `28800s`。平台日志确认 stage 为 `p4_nav_ppo`、128 环境、
  低层 `inference_only`、Adapter `online_auxiliary`、五段固定顺序、20 列、`track_length=5` 和
  `curriculum=false`。
- 修复后的自定义监控配置已被平台完整加载，显示 `P4全赛道诊断(19)` 及其他 P4 面板，启动阶段
  `暂无错误日志`，不再出现标题校验失败。训练连续运行到 `iter=11`、`effective_min=6.1`；
  Actor/Critic/Adapter 指标有限，`lifecycle_fail=0`、`low_digest_drift=0`、
  `low_optimizer_steps=0`。峰值 CUDA allocated/reserved 约为
  `820380672/1021313024` bytes，pinned depth 为 `471859200` bytes，未出现 OOM。
- 首次保存链已验证：有效训练约 5 分钟时生成
  `/data/ckpt/legged_robot_competition_26_ppo/model.ckpt-fullwarm-1418400.pkl`，SHA256
  `7f20cf566282cb98add08b5556d7f6fef8054fe1eb1fdb6e0fef90a56d7c4839`；随后平台归档生成
  `/data/user_ckpt_dir/legged_robot_competition_26_ppo/model.ckpt-fullwarm-1418526.pkl`，SHA256
  `1acdc8d9e7eae947f214637f522fa95952ad391e980a63af8e3f2d7b712b8153`。
- 当前验证只证明正式任务正常开始、监控与 checkpoint 探活链闭环，不证明八小时后的五段完成率、
  reason 4 误杀率、长距离直行、Maze 能力或真机表现。任务保持运行，后续按固定窗口抓取和最终评估
  更新本条，不得提前把启动证据描述为训练目标已经达成。

### 2026-08-05 正式任务持续启动复核

- Chrome 登录态复核确认任务 `236446` 连续运行至页面时长约 18 分钟，状态仍为“进行中”，错误日志
  区显示“暂无错误日志”。训练日志推进到 `iter=27`、`effective_min=15.1`，Actor `0.0652`、
  Critic `64.5540`、Adapter `-0.1379` 均有限；`lifecycle_fail=0`、`low_digest_drift=0`、
  `low_optimizer_steps=0`，峰值 CUDA allocated/reserved 仍为约
  `820380672/1021313024` bytes，未发生 OOM 或冻结低层误更新。
- EnvMonitor 每个窗口上报 501 个指标，最新窗口出现 `completed=10, abnormal=11, timeout=2`，证明
  环境 reset、scorer、训练监控和终止结果链持续工作；这只是启动健康证据，不作为训练效果结论。
- 五段装配日志再次确认顺序为
  `pyramid_slope -> pyramid_slope_inv -> pyramid_stairs -> pyramid_stairs_inv -> open_entry_maze`。
  第二次周期保存成功生成并归档 `model.ckpt-fullwarm-1421246.pkl`，SHA256
  `9c7f55f4c859f078d499d0807c54b57f34b9a7e4dc22271a189bcf35241706f1`。因此正式任务已经越过
  环境创建、首轮 PPO、首次保存和后续周期保存四个启动边界，当前未观察到计划外 stage、地形、
  低层更新、显存或 checkpoint 偏差。

## BUG-20260805-002：五段混训后 Maze credit 被 terminal clawback 主导且旧时钟污染修复续训

- 日期：2026-08-05；状态：本地已验证，开发容器、平台 smoke 和固定种子评估待验证。
- 影响范围：`codex/p4-maze2h-credit-repair` 的 P4 Maze warm start、奖励归因、平移 limiter、
  Actor/Critic/Adapter 训练职责、checkpoint exact resume 和监控；不修改平台覆盖的
  `server/isaac_env/base_env.py`，不改变 Actor85、Critic 输入、三轴动作或评估 wire。
- 父制品：`p4maze8h10hz_1416926` 的 `mazefinal` checkpoint。模型 ID `1416926` 只用于候选
  选择；实际加载继续校验 bundle、模块 spec、shape、有限值和冻结低层 digest。ZIP/checkpoint
  SHA256 复用 BUG-20260805-001 已记录值，本分支尚未生成新 checkpoint。
- 用户可见症状：五段八小时训练改善了前段通过，却使 Maze 条件成功率明显退化。失败 episode 的
  终止回报被 frontier clawback 主导；风险一出现就把平移压到 35%，会让策略在墙前缺少绕障位移；
  active reason4、额外 Goal/Camera fault、五段出生与跨段奖励同时变化，使 Maze 忘记无法归因。
- 根因：
  1. P2 frontier potential 把已获得进展在 terminal 统一回扣，Maze 中正常绕行和最终 timeout 的
     credit 尺度被终止事件重新改写，策略难以区分“曾正确逼近”与“最终失败”。
  2. 旧 limiter 对任意 risk 连续缩放完整平移，在高风险处最低仅保留 35%，减速机制压过了绕障。
  3. 旧 P4 checkpoint warm start 只改 optimizer phase，却把旧 session seconds 带入继承 P2 loader，
     使新会话在加载期按终局 schedule 校验；新 P4 `train_scope` 也会被继承 loader 的历史字符串拒绝。
  4. 监控仍加载五段出生/frontier/Push/Goal fault 面板，单 Maze 任务产生大量空值并掩盖 credit。
- 修复：
  - 训练配置改为单段 `open_entry_maze`、20 列、课程关闭、75 秒 episode、128 env、7200 秒有效训练；
    Push、额外 Camera/Goal fault、五段出生、segment frontier、route-excess、open-straight 和真实
    stuck reset 全部关闭或置 shadow。
  - 进展奖励改为 episode 内历史最短距离的不可重复 credit：`2.0/m`、累计上限 `+12`，terminal
    不 clawback；success 保持 `+200`，failure/timeout `-25`，10Hz time cost `-0.02/tick`。
  - limiter 仅在 risk `>0.75` 时进入紧急收紧，risk=1 时保留 60% `(vx,vy)`，`wz` 不缩放，
    风险解除每个 10Hz tick 最多恢复 0.20。
  - 冻结低层、NavigationEncoder、SafetyHead 和 Adapter；保留 Actor/LSTM 权重但重置 Actor Adam，
    重建 Critic/return statistics，StuckHead fresh。10 分钟后按 `creditadapt/credittrain/creditfinal`
    训练 Actor，teacher 梯度目标最高 3.5%、硬上限 5%。
  - warm start 的兼容视图归零 session、rollout、gradient 和 optimizer phase 计数，只把旧累计训练秒
    继承为 lifetime；P4 先校验 `maze_actor_critic_stuck_only`，再临时翻译为继承 P2 loader 所需 scope，
    保存包不被改写。新增 `credit*` checkpoint 标签和本地 fresh-instance exact-resume round trip。
  - P4 监控切换为单 Maze 标题，移除五段专用面板，增加非重复 credit、阶段和冻结模块面板。
- 本地验证：
  - `PYTHONPATH=server .../PY311test/bin/python -m pytest -q
    server/agent_ppo/tests/test_p4_*.py`：最终审查修正后 `130 passed`。
  - 覆盖 limiter 阈值/释放、credit 非重复和上限、shadow reason4 语义、四阶段 LR、旧父包 warm start、
    新合同 save/resume、冻结 NavigationEncoder/SafetyHead/Adapter 和监控 route。
  - 新增 creditwarm 无 PPO reference gradient 回归：Actor/CNN 冻结时 BCE 只更新 StuckHead，
    不污染 Actor Adam，也不把 head-only 系数带入 10 分钟后的共享梯度校准。
- 2026-08-05 最终审查修正：原 workflow 将 checkpoint、监控和日志耗时计入
  `session_effective_seconds`，与“7200 秒有效训练”合同不符。现改为只累计每轮 rollout collection
  到 PPO update 完成的活跃区间，墙钟单独保留用于周期保存；同时移除未接线的
  `[p4_nav_ppo.actor_mean_teacher]` 伪配置，Teacher 固定权重和梯度上限只由版本化合同控制。
  共享 `server-deploy-contract.md` 已补齐 credit-repair 的 warm-start、exact-resume、奖励、limiter、
  shadow reset 和 checkpoint 标签边界。
- 证据措辞更正：上述 save/resume 只是在同一 pytest 进程中新建算法实例的 local round-trip，
  不是 fresh-process 或平台 preload 证据。尚未在开发容器使用真实 `1416926` 完成 1-env reset 与
  8-env 32-tick PPO/save-resume；
  尚未创建平台 smoke，也没有固定种子父包对比。因此当前只能称“本地合同已验证”，不能声称 Maze
  完成率、碰撞、卡滞或真机能力改善。
- 回滚：恢复 `p4maze8h10hz_1416926` 原包即可；本轮新 checkpoint 合同只允许同合同 exact resume，
  旧 P4 包始终按结构 warm start。再次遇到 phase mismatch 先检查 high-level session seconds、
  optimizer phase 和 train_scope，再检查模型 ID；不得新增 ID 单点硬门禁。
- 关联 commit/PR、平台任务、新 checkpoint：均待生成。

## BUG-20260805-003：P4 creditwarm 空 stuck 批次对无梯度 loss 反传导致首轮 PPO 停止

- 日期：2026-08-05；状态：平台已验证。
- 影响范围：`codex/p4-maze2h-credit-repair` 的首个 0-10 分钟 `creditwarm` PPO update；
  不改变网络、观测、动作、奖励、checkpoint 或部署接口。
- 父制品与任务：父模型 `p4maze8h10hz_1416926`；失败任务
  `p4maze2h-creditfix`（任务 ID `236575`）。模型 ID 仅记录血缘，不参与修复门禁。
- 用户可见症状：平台在 `algorithm_p2_nav_ppo.py:_run_ppo_epochs()` 的
  `(loss * scale).backward()` 抛出
  `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`，训练在首次
  PPO update 停止。
- 根因：creditwarm 按合同冻结继承 Actor/CNN，只允许 fresh StuckHead 校准；但部分 minibatch
  没有满足 motion-intent/mapping 条件的有效 stuck 标签。此时 PPO、SafetyHead、Teacher、Camera、
  Mirror 和 StuckHead 项全部不连接可训练参数，组合 loss 是有限常量。公共 PPO 循环把
  `_actor_update_enabled()` 误解为“每个 minibatch 必然存在梯度”，无条件 backward；若直接跳过
  backward 但仍 step，还会错误增加 Actor gradient-step 并污染监控口径。
- 修复：公共 PPO 循环按 microbatch 检查 `loss.requires_grad`，只对真实计算图反传；minibatch 至少
  有一次真实反传才执行 Actor optimizer step、log-std clamp 和 gradient-step 递增。无梯度不是
  non-finite，不增加 `skipped_nonfinite`；Critic 始终独立更新。后续阶段 Actor 解冻或任一有效
  辅助监督出现时，原训练语义不变。
- 回归防线：新增完整 32-tick、TBPTT16 的 creditwarm 空 stuck rollout，验证 Actor optimizer
  state/step 不变、Critic gradient-step 前进、更新指标有限且不抛错；原“有效 stuck 标签只更新
  StuckHead”测试继续保留。
- 验证层级：本地 P4 定向回归 `131 passed`，Python 编译和 `git diff --check` 通过。修复文件同步到
  开发容器后，使用 `env_isaaclab` Python 并显式注入 Isaac Sim 的预置 Torch 路径运行：空 stuck
  batch 定向回归 `1 passed`、完整 recovery auxiliary `16 passed`、全部 P4 回归 `131 passed`。
  容器代码测试已覆盖完整 32-tick/TBPTT16 PPO 更新，但尚未证明平台训练任务已经越过首轮真实
  rollout/backward；必须以替代任务的首个 PPO update 为平台复验证据。
- 平台复验：清理容器同步/pytest 缓存后创建替代任务 `p4maze2h-credit-r2`（任务 ID `236585`，
  UUID `31dfe18e-26ec-42cf-95e7-c7bda5e3dce5`），父模型仍为
  `p4maze8h10hz_1416926`，任务时长 `2h15min`。任务已运行到 `iter=4`、
  `effective_min=1.5`、`phase=creditwarm`，越过首轮真实 rollout 和 64 次 high-level update；
  原 `does not require grad` 异常未再出现，`lifecycle_fail=0`、`low_digest_drift=0`、
  `low_optimizer_steps=0`。这证明启动阻断已修复，不代表两小时后的 Maze 完成率或策略效果已经改善。
- 回滚：恢复这两个 PPO loop 条件即可；不需要回滚父模型或训练合同。再次遇到同一异常时先检查
  当前 phase、trainable param group 和各 auxiliary valid mask，不要增加模型 ID 门禁或强行解冻 Actor。
- 关联 commit/PR、新 checkpoint：待生成。

## BUG-20260805-004：P4 full-track checkpoint 合同误标与专用面板未接线

- 日期：2026-08-05；状态：本地已验证，开发容器和新平台任务待验证。
- 影响范围：P4 `full_track` 的 checkpoint exact resume 与自定义监控装配；Maze credit-repair
  训练算法、Actor85、三轴动作、worker wire、部署输入和平台 `BaseEnv` 不变。
- 用户可见风险：五段训练保存包的全局 scope 是 `high_level_and_response_adapter`，但 contracts
  被无条件写成 Maze credit-repair。fresh `load_bundle()` 会先把该包判为 exact-compatible，随后
  因 scope 不是 `maze_actor_critic_stuck_only` 自我拒绝。与此同时 `P4全赛道诊断` 面板虽已定义，
  `_build_p4_monitor()` 从未调用，full-track 任务无法看到五段出生、reason4 回退和分段结果。
- 根因：P4 合同函数在 Maze 归因修复时改成固定单 profile 元数据，保存和加载没有传入算法实例的
  `training_profile`；监控只按共享 `policy_entry=p4_nav_ppo` 路由，没有再读取活动 P4 TOML 的
  profile。原测试只覆盖 Track eval load 和面板常量，没有执行 full-track train save/load 或最终
  dashboard build。
- 修复：command/reward/training/metadata 全部支持显式 profile。五段包写入
  `p4_full_track_command_v2`、`p4_full_track_reward_v2_potential_straight` 和
  `p4_full_track_v2`，exact resume 要求 `high_level_and_response_adapter`；Maze 包继续使用原
  credit 合同和冻结 scope。历史误标 full-track 包因版本不同退化为 warm start。监控构建器读取
  `train_env_conf_track_p4_nav_ppo.toml` 的 `training_profile`，只为 full-track 装配专用组。
- 本地验证：新增 full-track save -> fresh algorithm `load_bundle()` round-trip，验证三套版本、scope、
  28800 秒时钟和 `fullstabilize` 恢复；新增 fake platform builder 验证最终 full/credit dashboard
  的标题和 group 路由。复审补充了历史误标包保留 StuckHead leaf/optimizer group 的真实形状回归：
  同名组按参数名迁移并返回 `p4_full_track_legacy_contract_warm_start`，旧包缺组时才 fresh。
  第二次复审进一步覆盖同时缺少 StuckHead leaf/group 的更旧包：不再强制读取不存在的 leaf，保持
  初始化 head，并与含 head 的路径统一执行 warm-start session/live-state 重置和降级告警。
  历史误标 full-track 包按新 session 重置计数和 optimizer phase，不再把 28800 秒旧时钟带入
  `fullwarm` 校验。最终 P4/P2 核心回归 `266 passed`，Python 编译、TOML、两项 UWB C++ 测试和
  `git diff --check` 通过。
- 回滚：恢复固定 Maze metadata 会重新破坏 full-track exact resume；如需临时绕过，只能把旧包
  明确作为 warm start，不能放宽模型 ID 或 scope 门禁。监控可独立回滚为 Maze 面板，但会失去五段
  验收数据。
- 关联 commit/PR、平台任务、新 checkpoint：待生成。
