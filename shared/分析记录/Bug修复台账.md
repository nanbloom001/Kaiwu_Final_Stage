# Bug 修复台账

> 维护规则：本文件是仓库级 Bug 根因与修复知识库，长期追加，不按训练阶段另起一份。
> `server/CHANGELOG.md` 记录“改了什么”，本文件记录“为什么坏、如何证明修好、如何防复发”。
> 最近更新：2026-08-02。

## BUG-20260801-002：D435i 轻微晃动时画面冻结且设备发生 USB 重枚举

- 状态：本地已验证；物理链路修复待用户验证。
- 影响：`deploy/sim2real_test_p3_standard_strict` 的独立深度调参、断流诊断和后续真机视觉输入稳定性。
- 首次发现：2026-08-01 机器狗本地 D435i 调参；任务名、模型 ID 和 checkpoint 不涉及。
- 症状：用户观察到轻微晃动相机即画面卡顿；内核原文为
  `usb 1-3.4: USB disconnect, device number 19`，后续在 `usb 1-2` 重新枚举并出现
  `reset high-speed USB device number 20 using tegra-xusb`。后续还观察到上游 `usb 1-3` Hub
  整体断开，其下相机、无线网卡和适配器同时掉线；相机也在不同 Hub 端口反复重枚举。
- 根因：已确认设备发生物理 USB 断开/重枚举，并且当前只运行在 `480M / USB 2.x`；线缆、插头、
  转接 Hub、供电或端口机械固定中的具体故障点仍为待验证假设。相机运动导致双目无效像素增加，
  不能解释设备从 USB 总线消失。
- 排除项：不是单纯深度滤波参数或场景运动造成的整机断流。GUI 积压仍可能增加显示延迟，但不能解释
  内核 `USB disconnect`。
- 修复：`tools/realsense_depth_tuner.py` 使用采集线程只发布最新帧，增加设备通知、拓扑变化、超时、
  跳帧、相机/USB 与主机处理停顿分类，以及延迟/CPU/GPU/内存持久日志；GUI 按参数类别分组并支持
  手动输入。GUI 限速读取最新帧且不再放大深度面板，减少显示线程与 30 FPS 采集争用。2026-08-01
  追加修复相机硬件滑块逐格下发：最后一次变化稳定 400 ms 后才发送一次 UVC 控制请求；每个参数
  增加基于 SDK `option_range.default` 的独立复位按钮。再次确认 USB 2 Hub 下 `330` 同样触发
  `Protocol error`、端口 power cycle 和 `unable to enumerate USB device` 后，USB 2.x 操作上限默认
  收紧到 `240`；连续三次超时且设备消失时自动退出并释放 pipeline。
  `scripts/run_depth_tuner.sh` 补齐运行依赖检查，`DEPTH_TUNER.md` 增加最短诊断路径。
- 验证：纯函数单元测试覆盖两类断流判定；Python 编译、shell 语法、无 GUI 采流和桌面 GUI smoke
  在本机执行。物理线缆/端口更换后的晃动复测尚未执行，不能标记真机链路已修复。
- 防复发：每次深度测试保留 `events.jsonl`、`metrics.csv` 与内核 USB 记录；部署前确认 D435i 为
  USB 3.x 枚举并做固定线缆后的运动扰动测试。
- 血缘：当前目录无可用 `.git` 元数据，commit/PR/分支未知；父模型、checkpoint、SHA256 不涉及。
- 回滚：恢复上述三个调参工具文件的旧版本；不会改变正式控制器、训练契约或模型制品。
- 再遇检查：`lsusb -t` 看速率 → `journalctl -k` 查 disconnect/reset → 查事件日志中的设备在位状态
  → 对比 sensor/host gap → 最后看 CPU/GPU 与渲染耗时。

### 2026-08-01 追加：比赛相关相机约束与低延迟参数面板

- 状态：本地已验证，真机新 GUI smoke 待关闭旧进程后执行。用户要求把对单台 D435i 比赛运行有
  直接影响的参数加入调参 GUI；正式控制器配置和训练/部署输入合同未改变。
- 修改：Camera 页按“投射器/预设”“自动曝光约束”“低延迟、时间戳与设备诊断”分组，新增
  `Frames Queue Size`、自动曝光上限开关/数值、自动增益上限开关/数值、`Global Time` 和内部错误
  轮询，每项继续支持手动输入、设备范围、SDK 默认值和独立 Reset。曝光/增益上限标出
  `restart stream`，新增按钮由采集线程安全执行 stop/start，重启前强制提交 debounce 中的参数。
- 监控：实际曝光、增益、激光和 emitter 状态改从每帧 metadata 获取，避免周期性 UVC 查询加重当前
  USB2 链路；Frame queue 与 USB descriptor 同步进入 GUI 和 `metrics.csv`。HDR、多相机同步、外部
  触发、Decimation 和 Advanced Mode 参数未加入，因为会改变无关硬件路径、帧分布或训练几何。
- 验证：`py_compile`、shell 语法、调参工具 10/10 和部署目录全套 Python 26/26 测试通过，覆盖比赛参数清单、restart-required
  标记和无 UVC metadata 回读。真实相机 smoke 首次被用户仍在运行的旧 GUI PID `82518` 以
  `VIDIOC_S_FMT errno=16 Device or resource busy` 阻止；没有终止该用户进程，因此不能把硬件新界面
  标记为已验证。旧进程不会热加载新代码，必须关闭后重新启动。
- 回滚与风险：回滚本次 `realsense_depth_tuner.py`、测试和文档改动即可；不会影响 powered 控制器。
  Queue 设为 1 可能增加丢帧，曝光上限过低或增益上限过低可能增加深度空洞，过高则增加运动模糊或
  噪声，必须逐项比较原始深度、frame gaps 和 metadata。

### 2026-08-02 更正：USB2 Laser 上限不再作为硬门禁

- 用户确认比赛设备暂时只能使用 USB 2.1，USB 版本不得成为启动、采流或参数范围硬门禁。此前为规避
  物理掉线把 USB2 Laser 操作上限收紧到 `240`，属于诊断工具主动改变设备可调范围，现予以撤销。
- 调参器保留 `USB2_LINK policy=warning_only`，不再调用 `cap_maximum()`，删除
  `--usb2-laser-limit`；Laser 继续显示 SDK 声明范围。设置失败仍按 UVC/固件真实错误告警，软件自动
  重连逻辑不变。该更正不宣称 USB2 链路已经稳定，也不修改正式控制器的相机参数。
- 真机 GUI smoke 在 USB `2.1` 下完成 60 帧、0 timeout；启动只打印 warning-only，`session.json`
  回读 Laser `max=device_max=360`，没有 `USB2_LASER_LIMIT_ACTIVE`。

## BUG-20260802-001：D435i 重枚举后采流恢复但参数控制仍绑定旧 video 节点

- 状态：本地已验证，真机重枚举 smoke 待执行。
- 影响：`deploy/sim2real_test_p3_standard_strict/tools/realsense_depth_tuner.py` 的 Laser、曝光、增益
  和其他 Stereo Module 参数控制；正式控制器、训练输入、checkpoint 和 ONNX 不受影响。
- 症状与关键日志：`depth_tuner_20260801_232851_322` 中 Laser `0` 成功应用；请求 `90` 时
  `get_xu(... UVCIOC_CTRL_QUERY) ... Protocol error`，随后同序列号设备先消失再出现。采流从 frame 0
  恢复且 metadata 稳定报告 `Actual laser=90`，但后续 `150/180/240` 和曝光限制全部报
  `map_device_descriptor Cannot open '/dev/video4 ... No such file or directory`。设备已重枚举为新的
  video 节点，因此 Actual laser 不再跟随新请求。
- 根因与排除方向：librealsense pipeline 能在短暂 USB 重枚举后继续交付帧，但脚本保存的
  `depth_sensor`/NumericParameter target 仍引用旧 UVC 节点。旧实现还在 `OPTION_REJECTED` 时把 desired
  写入 applied，导致 GUI 误报并抑制后续重试。不是 frame metadata 刷新延迟；metadata 正确证明
  `90` 已成为设备实际值。物理 USB2 链路掉线是独立根因，本修复只解决软件恢复。
- 修复：设备拓扑回调只记录状态并提交恢复请求，采集线程串行执行 stop/start、按固定 serial 重开
  pipeline、获取新 depth sensor 并重绑 option。重连期间硬件控件禁用，显示
  `DEVICE_LOST/RECONNECT_PENDING/RECONNECTING`；成功后记录 generation 和
  `DEVICE_CONTROL_REBOUND`。只恢复 Queue、Global Time、Error Polling；Laser、preset、emitter、曝光
  和增益同步设备当前值，禁止自动重放旧危险请求。设备缺席时先等待默认 10 秒再退出。
- 参数状态：新增 `Laser requested/applied/frame actual`。设置失败把 applied 置为 unknown，并以 2 秒
  间隔限频重试；若 set 成功但 get 失败，而帧 metadata 与 requested 一致，则记录
  `OPTION_CONFIRMED_BY_FRAME_METADATA` 并停止重试。
- 修改文件：调参脚本、`tests/test_depth_tuner.py`、`DEPTH_TUNER.md`、部署 README 与本台账。
- 验证：`py_compile`、shell 语法、调参专项 14/14 和部署目录全套 Python 30/30 通过；模拟旧节点失效、同序列号新 sensor 重绑时，
  Queue `2` 被恢复，Laser 旧请求 `240` 被丢弃并同步为设备当前 `90`，control state 回到 ONLINE、
  generation 递增。用户当前仍运行旧 GUI 进程，未强制终止，因此真实 USB 重插/掉线恢复待关闭旧
  进程并启动新代码后验证；平台、评估和 powered 真机均不适用。
- 血缘：工作区无 `.git` 元数据，`git status --short --branch` 返回 not a git repository；commit/PR、
  模型 ID、checkpoint 和 SHA256 不适用。
- 遗留风险：软件重绑不能修复 USB2 Hub、线缆、供电或连接器导致的物理掉线。pipeline 重新打开失败
  仍会结束调参会话；不自动恢复 Laser 可避免掉线循环，但操作者需要重新确认当前值后再调节。
- 回滚：恢复调参脚本旧的单一 restart event、设置失败处理和 GUI 状态字段，并删除对应测试/文档；
  不需要回滚正式控制器。再次遇到时最短检查路径为 topology event -> control state ->
  `DEVICE_CONTROL_REBOUND` -> generation -> requested/applied/frame actual -> 新旧 `/dev/video*`。

## BUG-20260802-002：Restart camera stream 在设备掉线后被当成致命异常并关闭 GUI

- 状态：本地已验证，真机恢复 smoke 待相机重新枚举后执行。影响范围仅为
  `deploy/sim2real_test_p3_standard_strict/tools/realsense_depth_tuner.py` 的曝光/增益限制提交、pipeline
  重启和 USB 掉线恢复；正式控制器、训练输入、checkpoint、ONNX 与 powered 安全逻辑不变。
- 症状与关键日志：会话 `depth_tuner_20260802_003921_388` 在 `00:40:37.757` 记录
  `STREAM_RESTART_REQUESTED`，`00:40:37.783` 开始重启；第一次约 1 秒内完成 rebind。内核同一秒出现
  `uvcvideo ... -71`、`usb 1-3.4: USB disconnect`，随后 descriptor read `-110/-71`、端口 power cycle
  和 `unable to enumerate USB device`。设备消失后再次点击，日志在 `00:41:13` 执行 restart，最终
  `CAPTURE_EXCEPTION: No device connected` 并关闭窗口。后续会话 0 帧，`--list-devices` 和 `lsusb -t`
  均看不到 D435i。
- 根因与排除方向：物理相机从 USB 总线消失是硬件/链路事实，软件不能凭 restart 恢复未枚举设备；
  具体是 USB2 Hub、线缆、供电、连接器还是相机固件仍待物理复测。软件另有两个确定缺陷：普通采集
  循环每 2 秒重试固件已用 HWMON `response -9` 拒绝的 restart-required 控件，产生大量 UVC 查询；
  restart 按钮又强制全量提交所有硬件参数，并在单次 `pipeline.start()` 返回 `No device connected`
  后把可恢复缺席升级为 capture fatal，GUI 1.5 秒后自动关闭。
- 修改与理由：正常 `_apply_options()` 跳过 restart-required 控件，只允许按钮显式尝试一次 pending
  曝光/增益限制；不再全量重放 Laser、preset、emitter 等无关参数。若全部限制被拒绝，记录
  `RESTART_REQUIRED_OPTIONS_REJECTED` 和 `STREAM_RESTART_SKIPPED`，不停止正在工作的 pipeline。真正
  reopen 失败时记录 `STREAM_RESTART_DEFERRED`，进入 `WAITING_FOR_DEVICE`；连续采流超时也停止失效
  pipeline 后进入等待，而不是设置 `capture_exception`。同一序列号重新出现后仍复用既有 rebind，
  只恢复 Queue、Global Time、Error Polling。GUI 和后端同时限制按钮：仅 `ONLINE` 且确有 pending
  限制时允许请求，掉线等待期间硬件控件立即禁用，避免重复点击排队第二次 restart。固件拒绝后该值
  不再保持 pending，只有用户再次修改或设备成功 rebind 才可重试；preset 变化也不再把未修改的
  restart-required 参数伪装成 pending。
- 验证：`py_compile`、调参专项 17/17 和部署 Python 全套 33/33 通过。新增回归覆盖“capture loop 永不重试 restart-required
  控件”和“全部 restart-required 写入被拒绝时 pipeline.stop 调用数为 0、状态恢复 ONLINE”；既有
  重枚举测试继续证明 Queue 恢复、Laser 同步设备当前值和 generation 递增。当前 D435i 尚未出现在
  USB 总线，因此无法执行真实 stop/start/replug smoke，状态不得升级为真机已验证。
- 血缘：工作区无 `.git` 元数据，`git status --short --branch` 返回 not a git repository；commit/PR、
  模型 ID、checkpoint 和 SHA256 不适用。
- 遗留风险与回滚：软件等待不会修复 `-71/-110` 或 `unable to enumerate`；设备不恢复时 GUI 会持续
  保持等待，需要操作者重新插拔/断电恢复相机。回滚为恢复 restart-required 的周期 `_apply_options`
  和单次失败 fatal 逻辑，但会重新引入 UVC 请求风暴与假崩溃，不建议。再次遇到时按
  `STREAM_RESTART_*` -> `WAITING_FOR_DEVICE` -> `lsusb -t` -> `journalctl -k` -> 同序列号
  `DEVICE_CONTROL_REBOUND` 的顺序检查。

## BUG-20260802-003：深度调参面板把应用到 GUI 的局部耗时显示成端到端延迟

- 状态：本地已验证并完成真机相机 GUI smoke。影响范围仅为
  `deploy/sim2real_test_p3_standard_strict/tools/realsense_depth_tuner.py` 的诊断时间统计与日志；相机
  profile、滤波算法、正式控制器、训练输入、checkpoint 和 ONNX 不变。
- 症状与证据：用户在 temporal alpha `0.5`、delta `20`、persistence `7`、Frame queue `16` 时体感
  延迟约 `100 ms`，旧面板截图却显示 `To display=32.7 ms`。会话
  `depth_tuner_20260802_012841_211` 实际 `total_display_ms` 平均 `42.40 ms`、范围
  `23.83..64.94 ms`，相对 queue offset 平均 `22.29 ms`、曝光约 `11.55 ms`；旧数值仍未包含相机
  内部绝对排队、USB 基线、GUI 图像转换、X11 compositor 和屏幕扫描。
- 根因：旧实现用 `ready-arrival + GUI tick-ready` 作为 `To display`，起点是主机已经收到帧，而不是
  RealSense 帧时间戳；relative queue 又按本会话最小 offset 归零。GUI 在 PhotoImage 转换前停止
  计时，capture image 与另一次 metrics snapshot 也可能跨帧拼接。
- 修复：在 `global_time/system_time` 域使用主机墙钟减帧时间戳，新增 receive、app-ready、GUI-submit
  三个绝对帧龄；hardware clock、负偏移超过 5 ms 或大于 60 s 的异常值返回 NaN 并标记
  `latency_clock_valid=0`。GUI 原子取得 dashboard/frame/metrics，在 PhotoImage configure 后结束计时，
  并分离 GUI scheduling 与 conversion。面板移除误导性的 `To display`，queue 明确标为 relative。
- 训练诊断输出：新增 `display_latency.csv`，每个 GUI submit 帧绑定 frame number、timestamp domain、
  三段帧龄、曝光、queue 和 temporal 参数；每秒 flush，避免逐帧 fsync 放大延迟。训练域随机化使用
  receive 帧龄分布，GUI-submit 仅解释体感；compositor、显示器扫描和像素响应仍需外部测量。
- 追加无头复现入口：`tools/realsense_latency_test.py` 读取
  `configs/depth_latency_test.yaml`，默认 2 秒预热后逐帧测量 10 秒，终端打印 receive/processed 平均和
  最大延迟，并保存每帧 CSV、P50/P95/P99 summary 及合并后的配置快照。传感器参数只下发 YAML 中
  显式列出的键，requested/applied/rejected 全部进入 summary；USB2 为 warning-only。模板默认
  `strict_sensor_options=true`，任一硬件参数未实际回读确认即失败。
- 无头真机验证：默认 YAML 在 D435i USB2.1 上完成 2 秒预热和 `10.033 s` 正式采样，301 帧、
  `30.000 FPS`、0 丢帧、同步时间戳有效率 100%；global time、Frame queue `2`、自动曝光、emitter 和
  Laser `150` 均请求/回读一致。receive 延迟平均/最大 `28.971/31.016 ms`，P95/P99
  `30.236/30.716 ms`；Temporal alpha `0.5`、delta `20`、persistence `7` 后 processed 平均/最大
  `31.001/32.963 ms`，主机滤波处理平均 `2.030 ms`。
- 2026-08-02 追加更正：用户后续配置将 `filters.hole_fill.mode` 设为 `3`，而当前
  librealsense 独立孔洞填充器的合法范围是 `0..2`，启动只显示 SDK 原始报错
  `out of range value for argument "value"`。模板已改为 `mode=2`，滤波器参数现在下发前校验
  min/max/step，传感器参数失败也会记录完整 YAML 键、请求值和设备范围。该更正
  当前状态为“代码已修复待真机复测”；USB2 仍只告警，未新增总线硬门禁。
- 验证：`py_compile`、调参专项 20/20 和部署 Python 全套 36/36 通过，覆盖 global/system 绝对帧龄、
  hardware clock 拒绝、异常 offset 拒绝及 display CSV 字段。两次真机 GUI smoke 分别完成 90/60 帧、
  0 timeout，设备真实报告 `timestamp_domain.global_time`、`latency_clock_valid=1`。90 帧会话的 39 个
  GUI submit 样本中 receive 平均 `14.88 ms`，GUI submit 平均 `80.42 ms`、范围
  `47.64..115.17 ms`；最终 60 帧会话 25 个样本 receive 平均 `9.13 ms`、GUI submit 平均
  `74.68 ms`、范围 `51.51..137.38 ms`。这与用户约 100 ms 体感一致，并证明旧面板确实低估。
- 血缘与回滚：工作区无 `.git` 元数据，分支、commit、PR 不可用；模型 ID/checkpoint/SHA256 不涉及。
  回滚为恢复旧 `total_display_ms` 面板并删除 `display_latency.csv`，会重新引入低估和跨帧统计。
  再遇检查：timestamp domain -> latency_clock_valid -> receive P50/P95/P99 -> queue/filter 配置 -> GUI
  submit；禁止把 Capture wait、Host gap 或 relative queue 直接相加成端到端延迟。

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

## BUG-20260801-001：P3 Standard 训练有效动作边界与部署安全/诊断口径漂移

- 状态：本地已验证。
- 影响：部署、checkpoint/ONNX 身份、action 处理、相机/状态新鲜度、真机安全和 sim2real 诊断。
- 首次发现：2026-08-01，Go2 P3 Standard strict 重建任务；训练参考
  `/home/unitree/p3_training_reference/p3nav8h-r1_884257-evalfix-v2`，模型/checkpoint 标签
  `highslow-884257`。开始时仓库 Git 元数据在当前受限执行环境不可用，`git status --short --branch`
  返回“not a git repository”，因此当前分支、已有暂存和 commit/PR 无法核实；任务外目录未清理。
- 症状：训练 workflow 在每次 `env.step` 前执行 `torch.clamp(actions,-6,6)`，但历史生成的
  `deploy.yaml` 使用 `[-100,100]`；旧部署没有把 checkpoint、ONNX、精确 ABI、config/deploy
  一起做 SHA 门禁，也没有在同一日志中分离 model raw、clipped、requested、slew、physical、
  applied 和 executed feedback。LowState/深度/推理/目标新鲜度、机械阈值和 powered 前 shadow
  稳定门控也未形成单一 fail-closed 契约。
- 根因：部署口径主要继承 exporter 的静态配置，没有从实际训练 transport、训练参考相机配置和
  Go2 URDF 重新审计有效边界；处理层和诊断层混用 `action/target` 聚合字段，无法区分策略本体
  高频、限幅器修改、跟踪误差和单腿机械响应。该根因是代码/契约缺口；历史 RR 高频是否为机械
  或策略根因仍为待验证假设。
- 排除项：checkpoint 与 strict ONNX 字节身份一致，SHA256 分别为同源 checkpoint
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23` 和 ONNX
  `6ccaf7e033240720a66019885df091097efa2d6d8962a5840966ede8b9de2d87`；100 帧逐帧
  checkpoint→ONNX recurrent rollout 的 joint 最大绝对误差仅 `8.345e-07`，cmd/cmd_raw/
  clearance 为零误差，因此当前没有证据指向 exporter 数值错误。短历史日志不足以证明 RR 根因。
- 修复：新增独立目录 `deploy/sim2real_test_p3_standard_strict`，不修改默认稳定入口。新增
  `artifact_contract.json`、精确 ONNX ABI/哈希预检、训练有效 raw clip `[-6,6]`、完整 action
  分层与 executed-action 回灌、URDF 关节软件边界、不可变 LowState snapshot、D435i 精确 profile、
  新鲜度/推理/机械 watchdog、故障冻结与 FixStand/Passive 接管、sensor-only/shadow 无 LowCmd
  模式，以及 powered 内部 2 秒 + 连续 LowState/深度稳定门控。新增 strict CSV 分层、逐关节
  `5-10/10-15/15-25 Hz`、交替符号、四腿集中度、左右不对称、q/target/dq/tau 和 freshness 分析。
- 验证：Python 12 项安全测试通过，含制品字节和 ONNX ABI 故障注入、非有限 action、深度 golden
  vector、watchdog、并发读取和合成高频/单腿诊断；C++ normal 与 ThreadSanitizer 两项测试通过；
  controller `--preflight` 在 DDS/LowCmd 初始化前通过精确 ABI 和哈希检查；100 帧 parity 通过。
  历史 121 帧 fixed-zero 日志可由新分析器读取，RR 高频集中度在 raw/applied 两层分别为中位腿的
  `1.486x/2.218x`，但仅登记为待复测信号。本机未检测到 RealSense；sensor-only、60 秒 shadow、
  吊起、地面、平台和真机验证均未执行，且本任务未创建 LowCmd、未发送电机命令。
- 防复发：preflight 同时绑定 checkpoint、ONNX、config、deploy SHA，并逐项校验 8 入 8 出名称、
  dtype、rank、shape、opset 和 IR；CMake 注册 normal/TSan 测试；powered 必须 `--arm`、RealSense
  探测、同进程 shadow/stability gate 和物理手柄确认。后续接口变化必须原子更新训练来源说明、
  strict contract、deploy 和 `shared/interfaces/server-deploy-contract.md`。
- 血缘：当前分支/commit/PR 因 Git 元数据不可见而未知；父模型 ID 未作为独立字段提供；checkpoint
  标签 `highslow-884257`，checkpoint/ONNX SHA 如上，contract SHA256
  `9043327ba7341808631f3097ec6566b4bf3dce5d8c71fbca02d7a059b29ba139`。
- 遗留风险：训练相机外参仅作为参考复制，`calibrated=false`，实机安装外参必须测量；本机未发现
  默认稳定 `sim2real_test_loco` 的成功诊断 CSV，无法做匹配 baseline 比较；历史 P3 日志很短；
  LowState tick、D435i 时钟和机械阈值仍需 60 秒无命令数据校核后才能进入吊起测试。
- 回滚：停止使用并移除独立 `deploy/sim2real_test_p3_standard_strict` 目录，同时回滚本条契约文档；
  默认 `deploy/sim2real_test_loco` 未被本修复修改。不得通过放宽 clip、关闭 watchdog 或复用旧二进制回滚。
- 再遇检查：训练 workflow 的 env.step 前处理 → checkpoint/ONNX/config/deploy SHA 与精确 ABI →
  LowState/depth/inference/target freshness → raw/clipped/applied/executed 各层频谱 → 单腿与左右机械响应。

### 2026-08-01 审查修复记录

- 代码审查发现并修复：powered 主循环忽略 SIGINT；策略线程失活时 `run()` 可能使用缓存 snapshot；
  RealSense 缺少精确 profile 或 intrinsics 时静默回退；policy loop 顶层异常未统一转为安全接管。
- 修改：`CtrlFSM::stop()` 先停止 recurrent publisher 再退出当前状态；main 主循环检查 stop signal
  并执行 FSM/Go2 shutdown；`State_VisionLoco::run()` 每次 FSM 周期重新采集并校验当前 LowState，
  失效时清零增益并请求 Passive；RealSense 只接受精确 `424x240@30 Z16` 且 intrinsics 缺失直接
  hard fail；策略线程增加顶层 `std::exception`/unknown exception hard-fault boundary。
- 验证：CMake 构建成功；C++ `ctest` 的 normal 与 TSan 均通过；Python 12 项测试通过；100 帧
  recurrent parity 通过，joint 最大误差 `8.345e-07`；controller `--preflight` 通过且未创建
  DDS/LowCmd。未执行相机、sensor-only、shadow 或真机验证，故状态仍为“本地已验证”。
- 新 controller SHA256：`f8f30c47573c731df536374ff8b5691f7c6abf9cc6fce5f221d562ddded30c0a`。

### 2026-08-01 架高静态测试追加记录

- 发现并修复一个实际启动问题：`param::load_config_file()` 只识别目录名 `build`，在
  `build-strict` 下没有加载 config，导致 LowState 建链后 YAML 节点为空。现在改为从可执行文件
  路径向上搜索实际存在的 `config/config.yaml` 或 `config.yaml`。
- 硬件证据：USB/udev 访问授权后 D435i `338622073957`、firmware `5.13.0.55` 可枚举；
  strict preflight 通过；LowState 报告 connection established；没有 LowCmd。
- 真实阻断：设备没有 `424x240@30 Z16`，仅有 `640x480@30 Z16` 等 profile，strict sensor-only
  按契约 hard fail。独立读取 120 帧 640x480 深度，120 个唯一帧、无重复/倒退，时间间隔均值
  `33.397 ms`，有效像素比例均值 `0.862199`。这证明相机帧通路稳定，但不证明训练几何对齐。
- 验证：C++ normal/TSan 两项通过；未完成 strict 60 秒 sensor-only CSV、shadow 或真机控制。
- 新 controller SHA256：`95bdbbac73bb9cab85980e3c14afb288cc261b5b46e8eb8f0e35c596dd3d734e`。

### 2026-08-01 采集 profile 契约更正

- 更正：上述“`424x240@30 Z16` 为严格训练 profile”的结论不正确。P3 训练环境的
  深度张量和低层 ONNX 输入均为 `320x180x1`；`424x240@30` 只是其他部署路线的原始采集
  期望值。当前 D435I 不支持该深度 profile，其他部署会选择 `480x270@30`，然后用当前
  profile 的运行时内参重投影到训练针孔模型。
- 根因：将“真机原始采集 profile”、“仿真相机几何”和“网络输入 shape”误合并为一个
  exact-profile 门禁；同时旧诊断指标统计原始图有效率，不是网络实际看到的投影张量有效率。
- 修复：strict 路线只允许已审计的 `424x240@30` 和 `480x270@30 Z16`；两者均必须成功
  取得有限、尺寸匹配的运行时内参，再重投影到 `320x180` 训练针孔。不允许任意最近
  profile 或无内参 resize。深度无效率、前方无效率、均值和最小值改为从低层实际消费的
  `320x180` 投影结果统计。
- 范围：仅低层 `modules.low_level.locomotion_encoder + actor`。高层导航 actor、goal 计算、
  nav LSTM 均不参与本轮测试；ONNX 中的 goal/nav 端口仅作 ABI 占位/透传。
- 验证：C++ 重投影回归新增同内参恒等映射和 `0/5m -> 0`、`2.5m -> 0.5` 断言；
  normal 与 TSan 测试二进制直接运行通过。真机 `480x270@30` sensor-only 复测待执行，
  故整体状态仍为“代码已修复待验证”。
- 制品：config SHA256 `4bc83952683bc3502a0684445f60787ae723f1c9bb37a01187419df4247bc9fd`；
  controller SHA256 `cb7f804e6bc014408121641b177123d3eb82f39fd8da9e8928b607315bffa01c`；
  checkpoint/ONNX 未变。回滚时恢复旧 exact-profile 门禁会重现已确认的伪阻断；安全回退是停止使用
  整个实验目录并回到默认稳定路线。

### 2026-08-01 采集 profile 更正后真机 sensor-only 验证

- 首次复测已成功选择 `480x270@30 Z16`，取得内参 `fx=fy=241.9, cx=242.0, cy=135.2`，
  并声明重投影目标 `320x180, fx=fy=168.61, cx=160, cy=90`。随后因后台抓帧线程尚未产生
  第一帧，主循环就将初始空帧判为 `stale/invalid depth`。根因为启动竞态，不是投影或相机断流。
- 修复：sensor-only/shadow 在进入 50 Hz 主循环前，最多等待 3 s 的首个有效投影帧；超时仍
  hard fail。不用常量深度、旧帧或无内参 resize 绕过门禁。
- 真机证据：`logs/sensor-only_1785524014.csv`，`104.920 s`、5247 个 50 Hz 采样；3151 个唯一
  深度帧，实际 `30.023 Hz`，帧号无倒退、无跳帧；深度 age 均值/p95/最大值
  `16.701/31.656/33.589 ms`。投影后整图无效率均值/p95/最大值
  `5.14%/5.49%/10.70%`，前方无效率 `1.63%/1.86%/9.96%`，均低于门禁。LowState tick
  无重复、无倒退，平均步进 20 ms。
- 分析器防线：新增投影后整图/前方无效率、policy/camera 实际频率、相机时间戳间隔、
  depth frame 倒退/跳帧和 LowState tick 单调性统计。正常的 50 Hz 消费者复用 30 Hz 最新帧会得到
  约 40% repeated-policy-frame ratio，不得误判为相机冻结。
- 状态：低层传感器与投影链路“本地已验证”。低层 ONNX shadow、吊起 powered、平地和平台
  仍未验证；高层导航网络未加载、未测试。全过程没有创建 LowCmd，没有发送电机命令。
- 最终 controller SHA256：`34275dcad7c782a8b4978522d32b7d5758d26b587e2f1645edc597ffc4ea2358`；
  contract SHA256：`5c3f409f503a25399fc65c825ffb453345ed34500990a05dcb07f7be5575fa60`。

### 2026-08-01 架高 LT+X 无响应修复

- 状态：本地已验证，真机复测待执行。影响范围仅为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 的 powered VisionLoco 启动生命周期与诊断；
  checkpoint、ONNX、训练观测和动作合同未改变。当前工作区仍无法读取 Git 元数据，
  `git status --short --branch` 返回“not a git repository”，分支、commit 和 PR 未知。
- 症状与证据：架高 `--fixed-zero --arm` 后，LT+X 确实进入 VisionLoco 并创建
  `visloco_diag_1785525352.csv`，但只记录 5 帧（`t_ms=5..84`）便返回 FixStand。
  全部帧 `shadow_gate_ready=0`；LowState age 最大约 `0.133 ms`、深度 age 最大约
  `32.013 ms`、推理最大约 `4.531 ms`，没有传感器或 ONNX deadline 故障。
- 根因：启动门控要求至少 2 秒、连续 100 帧稳定 LowState 和 60 帧稳定深度，但
  `run()` 从进入状态起就对初始化目标启用 `policy_target_stale_ms=100`，门控期间故意不发布
  策略目标却在约 100 ms 被判为 `policy_target_stale`，触发 SoftStop -> FixStand。另有两个
  同源缺陷：门控期间每周期把目标改成当前实测位置，削弱位置保持并使架高关节速度升至约
  `1.279 rad/s`；`shadow_gate_ready` 未锁存，策略开始运动后可能因超过静止速度阈值再次关闭。
- 修复：新增 pre-publication/active/fault-freeze 三态目标决策。门控期间保持进入 VisionLoco
  时冻结的关节位置；首次策略目标发布后才启用 100 ms stale watchdog；安全故障继续跟随当前
  实测位置并请求接管。启动门控改为一次通过后锁存到状态退出。新增
  `--suspended-test`，且只接受 `--fixed-zero --arm`；它只绕过启动稳定门控，其他传感器、推理、
  机械、action/slew、stale target 和接管保护全部保留，并在 CSV 中记录
  `suspended_test_bypass=1`。非法组合在 DDS/LowCmd 初始化前拒绝。
- 排除方向：没有证据表明 LT+X 手柄映射失效；日志文件只在 VisionLoco `enter()` 后创建。
  没有放宽 `stable_max_joint_velocity_rad_s=0.5` 或运行期 `dq_warning/dq_hard_fault`；架高绕过
  模式下静止速度阈值不参与放行，正常/落地模式仍保持原保护。
- 本地验证：全核心并行 CMake 构建成功；C++ normal 与 ThreadSanitizer 两项通过；Python
  12 项安全测试通过；新增回归覆盖门控等待超过 100 ms 不误报 stale、live target 超过
  100 ms 仍触发冻结、gate hold 使用冻结目标、fault freeze 使用实测目标、门控锁存和架高参数
  组合。两个非法 `--suspended-test` 组合均按预期退出。100 帧 checkpoint->ONNX recurrent
  parity 通过，joint 最大绝对误差 `8.345e-07`；strict preflight 通过且未创建 LowCmd。
- 制品：父模型/模型 ID 未新增；checkpoint `highslow-884257` SHA256
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`，ONNX SHA256
  `6ccaf7e033240720a66019885df091097efa2d6d8962a5840966ede8b9de2d87`，controller SHA256
  `07845742fef4e5363367069bf4c9f1361da7263bf8923aeddaab8e8c165cb2b0`，contract SHA256
  `f182f754aa5fef86f60db3d3fdeaf36825598e5c14e0b19f3943cc1763d4529c`。
- 遗留风险：尚未取得修复后的 powered 真机日志，故不得升级为真机已验证。架高绕过模式会让
  策略在首个有效推理帧后接管，四肢可能快速运动；必须确保完全离地、周围无人和障碍物、手指
  远离关节，并保持 LT+B/急停可立即操作。该模式禁止落地和非零速度命令。
- 回滚：移除 `--suspended-test` 参数与三态目标决策并恢复旧二进制；但旧版会确定性重现约
  100 ms 自动退回，不能作为 powered 测试版本。安全回退应停止实验并使用 Passive，而不是
  关闭 watchdog。
- 再遇检查：确认 CSV 是否创建 -> `suspended_test_bypass/shadow_gate_ready` -> 首次策略发布时刻
  -> `policy_target_stale` 是否只发生在 active 阶段 -> LowState/depth/inference freshness ->
  dq/tau/temperature 与 motion rejection。

#### 2026-08-01 第二次架高测试更正

- 修复后的首次 `--suspended-test` 复测证明启动 bypass 生效：
  `visloco_diag_1785527113.csv` 首帧即为 `shadow_gate_ready=1`、
  `suspended_test_bypass=1`。但日志仅 2 帧（`t_ms=7,25`），第二帧因
  `requested_target_step_max=0.129362 rad > max_target_step_rad=0.05 rad`，且
  `max_consecutive_motion_violations=1`，触发 `repeated_unsafe_motion` SoftStop。
- 这不是机械越界：当帧最大关节速度约 `0.911 rad/s`、最大估算力矩约 `1.41 Nm`、
  tracking error `0.050379 rad < 0.45 rad`、推理 `7.253 ms < 20 ms`，LowState/深度均新鲜。
  动作链已经把实际目标变化限制为 `0.05 rad/frame`；旧 motion guard 却用 slew 前的网络请求
  变化再次拒绝，和执行层限速重复，无法观察模型在架高状态下的真实连续输出。
- 更正修复：普通模式继续按 slew 前 requested target 做保守审计；仅
  `--suspended-test --fixed-zero --arm` 改为审计 clip、物理限位和 slew 后真正可能下发的
  applied target step，并保留 `0.45 rad` tracking error 以及全部 freshness、inference、机械、
  stale-target 和接管保护。新增回归确认：普通模式拒绝 `requested=0.129 rad`；架高模式在
  `applied=0.05 rad` 时允许，在 `applied=0.051 rad` 或 tracking error `0.451 rad` 时拒绝。
- 验证：全核心构建成功；C++ normal/TSan、Python 12 项、100 帧 parity 和 strict preflight
  全部通过；非法非零速度架高组合仍在 DDS/LowCmd 前拒绝。真机复测待执行，状态保持
  “本地已验证”。新 controller SHA256
  `160ee7b9898f1f41a9147542a72d4de420367408be3778420c44fe83f36c8264`；artifact contract
  SHA256 `bbd037d9fd0dedf25b2ab6e7cc863a347b3fcc88e6e6c2399f81952db6e49512`。
  checkpoint/ONNX 未变。

#### 2026-08-01 平地零速度跟踪门禁与低门控诊断模式

- 状态：代码已修复待真机验证。影响范围仅为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 新增的显式平地诊断模式；普通模式和架高模式的既有
  门禁不变。当前工作区仍无法读取 Git 元数据，分支、commit 和 PR 未知。
- 症状与证据：机器狗落在平地后使用原架高诊断模式运行，日志
  `visloco_diag_1785528463.csv` 共 10 帧、约 `0.179 s`。第 9 帧策略关节 j9（SDK FR calf）
  `tracking_error_max=0.451216 rad`，刚超过 `0.45 rad` 阈值，触发 motion rejection 和
  SoftStop。目标受 slew 限制以 `0.05 rad/frame` 从约 `-1.49 rad` 向 `-1.09 rad` 移动，实测
  关节在负载下保持约 `-1.54 rad`；估算力矩升至约 `11.617 Nm`。LowState、深度、推理与姿态
  没有同时出现异常，不能把该停止归因为数据链路丢失。
- 根因与边界：这是“位置策略目标与落地负载下实际关节未及时跟随”的诊断门禁，不是 URDF
  物理越界或力矩硬限位。零速度命令不等于零力矩；控制器仍按
  `tau_est = Kp * (q_target - q) - Kd * dq` 产生站立/姿态控制力矩。直接全局放宽普通模式会掩盖
  后续真实部署的跟踪异常，因此只新增需要操作者显式选择的受限诊断入口。
- 修复：新增 `--ground-test`，仅接受 `--fixed-zero --arm --ground-test`，并与
  `--suspended-test` 互斥；非法组合在 preflight/DDS/LowCmd 之前退出。该模式绕过启动稳定门控、
  以 slew 后 applied target 审计动作步长，并关闭 `0.45 rad` tracking-error 软门禁。仍保留
  raw action 非有限检查与 `[-6,6]` clip、`0.05 rad/frame` slew、URDF 物理限位、LowState/深度
  freshness、推理 deadline、policy target stale、关节速度/估算力矩/温度软硬门禁和接管。
  CSV 每帧记录 `ground_test_permissive=1`。
- 排除方向：未修改 checkpoint、ONNX、训练观测/动作口径、Kp/Kd、力矩阈值、物理关节限位或
  非零速度路径；`--ground-test` 不接受 `--fixed-vx`，也不是可长期运行的部署配置。
- 本地验证：使用全部 CPU 核心并行构建成功；C++ normal 与 ThreadSanitizer 2/2、Python
  12/12、100 帧 checkpoint->ONNX recurrent parity 和 strict preflight 均通过，joint 最大
  绝对误差 `8.345e-07`。回归确认普通模式仍拒绝 `requested=0.129 rad`，架高模式仍在
  tracking error `0.451 rad` 拒绝，平地诊断模式允许 tracking error `0.60 rad` 但在 applied
  step `0.051 rad` 拒绝。三种非法 CLI 组合均在 DDS 初始化前退出，未创建 LowCmd。
- 制品：checkpoint `highslow-884257` SHA256
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`；ONNX SHA256
  `6ccaf7e033240720a66019885df091097efa2d6d8962a5840966ede8b9de2d87`；controller SHA256
  `0d49a5812e6ca2a003dc2a61c0c155b41b9b1ec153b31945a2e4540238f79bd5`；artifact contract
  SHA256 `0e3cea64c80d14ce2897dff4365dbf354a1a478f536306bb44763f9a3020833c`。
- 遗留风险：尚未取得该模式的真机日志，故状态不能升级为真机已验证。跟踪误差软门禁被有意
  关闭后，落地姿态偏差可能继续积累到速度或力矩门禁才停止；首次测试应限制在约 1 秒、平整
  防滑且四周净空的地面，操作者持续握持手柄并准备 LT+B/实体急停，禁止人员接触腿部。
- 回滚：停止使用 `--ground-test` 即恢复普通门禁；代码回滚可移除该 CLI、状态参数和对应测试，
  checkpoint/ONNX/config 无需回滚。
- 再遇检查：CSV 的 `ground_test_permissive` -> 首个 rejection 原因 -> 逐关节
  q_target/q/dq/tau_est -> FR calf 是否持续失跟 -> freshness/inference -> 速度、力矩、温度门禁与
  takeover 是否按预期触发。

#### 2026-08-01 固定速度行走误停门禁与终端速度范围

- 状态：代码已修复待真机验证。影响范围仅为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 的 `--fixed-vx` 低层行走测试；训练模型、ONNX、
  Kp/Kd、关节物理限位和机械保护阈值未改变。Git 元数据不可用，分支、commit 和 PR 未知。
- 症状与证据：平地 `--fixed-zero --arm --ground-test` 日志
  `visloco_diag_1785529952.csv` 持续 2030 帧、40.578 s、50.00 Hz，可正常站立且没有 motion
  rejection。最大电机上报 `tau_est=16.263359 Nm`、最大 PD 估算 `16.447447 Nm`，均为 FR calf；
  小腿告警/软停/硬故障为 `27/36/43 Nm`，全程力矩告警帧数为 0。最大跟踪误差
  `0.649902 rad`，说明普通 fixed-vx 若继续使用 `0.45 rad` 跟踪软门禁，会在已验证为可站立的
  输出量级下误停。普通模式对 slew 前网络请求跳变的重复审计也会拒绝实际已被限制到
  `0.05 rad/frame` 的目标。
- 根因与修复：此前 fixed-vx 沿用保守 bring-up 门禁，没有吸收零速平地真机证据。现在 fixed-vx
  保留启动稳定门控，但动作步长只审计 clip、物理限制和 slew 后真正可能下发的 applied target，
  并关闭 tracking-error 软门禁；CSV 新增 `walking_test_permissive=1`。仍保留 raw action 有限性与
  `[-6,6]` clip、`0.05 rad/frame` slew、URDF 限位、LowState/深度 freshness、推理 deadline、
  policy-target stale、关节速度、力矩、温度和接管保护。
- 速度范围：`--vx VALUE` 已存在，部署校验上限由 `0.10` 放宽到 `1.00 m/s`；仅接受有限的
  `[0,1.00]`。该范围与训练端
  `train_env_conf_standard_p3_standard_joint.toml` 和 `deploy.yaml` 的 `lin_vel_x=[0,1]` 对齐。
  下一轮计划值为 `0.3 m/s`；允许 `1.0 m/s` 不等于该速度已经通过真机验证。
- 排除方向：没有提高力矩阈值，因为本轮站立力矩本就没有触发既有告警；没有关闭物理限位、
  过温、传感器/推理失效或硬故障接管；没有放宽侧向速度和角速度，本入口仍固定为纯前向命令。
- 本地验证：全核心并行构建成功；新增单元回归覆盖 fixed-vx 使用 post-slew 审计且忽略
  `0.65 rad` tracking error，并覆盖速度 `0/0.3/1.0` 接受、负值/`1.001`/Inf 拒绝。C++ normal
  与 ThreadSanitizer 2/2、Python 12/12、100 帧 checkpoint->ONNX parity、JSON/脚本语法和 strict
  preflight 全部通过，joint 最大绝对误差 `8.345e-07`；preflight 未创建 LowCmd。非零真机尚未
  运行，状态不得升级。
- 制品：checkpoint `highslow-884257` SHA256
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`；ONNX SHA256
  `6ccaf7e033240720a66019885df091097efa2d6d8962a5840966ede8b9de2d87`；controller SHA256
  `17ba8caf6ddc2bb84f950dc40d2070c80107cdca155a5233b919ed12a8d74449`；artifact contract
  SHA256 `81db4c7333819d38439498c47520b6128cadfc63daf56c99f2099f8f4a931b7f`。
- 遗留风险与测试边界：fixed-vx 的跟踪误差不会再单独触发 SoftStop，姿态或关节失跟将主要由
  物理、速度、力矩、温度、翻倒和操作者接管限制。首次 `0.3 m/s` 应在平整防滑、足够开阔的
  场地做短距离直行，操作者跟随并持续准备 LT+B/实体急停；不得直接跳到 `1.0 m/s`。
- 回滚：恢复 fixed-vx 的 pre-slew/tracking guard 和 `0.10 m/s` 校验；checkpoint、ONNX、配置
  与训练端均无需回滚。
- 再遇检查：确认 `walking_test_permissive=1` 与 `vx=0.3` -> applied step -> q/dq/tau/temp ->
  姿态与翻倒接管 -> LowState/深度/推理 freshness -> LT+B 是否立即返回 Passive。

#### 2026-08-01 行走力矩软停止提升至硬边界

- 状态：本地已验证，真机复测待执行。影响范围仅为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 的电机上报力矩门禁配置；checkpoint、ONNX、动作链、
  Kp/Kd、速度/温度/位置门禁和数据链路 watchdog 未改变。Git 元数据不可用，分支、commit、PR
  未知。
- 症状与证据：两次 `0.8 m/s` 日志 `visloco_diag_1785531316.csv`、
  `visloco_diag_1785531327.csv` 分别在 2.946 s 和 5.063 s 结束。停止前 FL thigh（策略 j4）
  `tau_est` 连续升至 `17.712610 Nm`、`17.267321 Nm`，PD 估算升至 `18.191326 Nm`、
  `17.798235 Nm`，紧邻原髋/大腿 `18 Nm` SoftStop。两次均无 motion rejection、deadline miss、
  推理异常、数据过期、明显倾倒或速度硬越界。由于机械检查在故障帧写 CSV 之前 break，精确终端
  原文未被 CSV 保存；“下一帧触发 `joint_effort_soft_4`”为高置信推断，仍待终端/复测确认。
- 根因：原配置在髋/大腿 `14 Nm` warning 与 `22 Nm` hard fault 之间额外设置 `18 Nm`
  SoftStop，小腿对应 `27/36/43 Nm`。该中间停止线会在模型仍低于既定硬机械边界时返回 FixStand，
  不符合当前实机能力表征阶段“告警但继续观察直到硬边界”的测试要求。
- 修复：保持 warning 为髋/大腿 `14 Nm`、小腿 `27 Nm`；将 `tau_soft_stop` 从 `18/36 Nm`
  提高为与 `tau_hard_fault` 相同的 `22/43 Nm`。运行时先检查 hard fault，再检查 soft stop，因此
  相同阈值使 torque SoftStop 分支不可达：warning 到 hard 之间只打印告警，真正超过
  `22/43 Nm` 仍触发 HardFault 并请求 Passive。没有移除最低硬机械保护。
- 本地验证：控制器读取外部 YAML，本次无需重编译，二进制保持不变。新增 Python 回归解析真实
  YAML，确认 12 个关节 `tau_soft_stop == tau_hard_fault`、硬边界为前 8 关节 `22 Nm`、后 4
  关节 `43 Nm`，且 warning 均低于 hard。C++ normal/TSan 2/2、Python 13/13、100 帧
  checkpoint->ONNX parity、JSON/脚本检查和 strict preflight 全部通过；joint 最大绝对误差
  `8.345e-07`，preflight 未创建 LowCmd。
- 制品：controller config SHA256
  `3dcb2006ba22fa56c662b6ebe6917ae995027e5ba8d56d9e6a28819e5f0ef952`；artifact contract
  SHA256 `ad91cb497f052aadabc4fa291f9ae2944f44438767e654118adc0347b2c752f1`；controller 二进制未变，
  SHA256 `17ba8caf6ddc2bb84f950dc40d2070c80107cdca155a5233b919ed12a8d74449`；checkpoint/ONNX 未变。
- 遗留风险：允许髋/大腿在 `18..22 Nm`、小腿在 `36..43 Nm` 继续运行会增加瞬态冲击和热负荷；
  warning 只每 50 帧打印一次，不能替代操作者观察。真机复测必须保持开阔防滑场地、短距离、
  LT+B/实体急停可立即操作，并重点检查 j4 力矩是否继续快速累积到 `22 Nm`。
- 回滚：把 `tau_soft_stop` 恢复为前 8 关节 `18 Nm`、后 4 关节 `36 Nm`，同步恢复 config 和
  artifact contract 哈希；二进制无需回滚。
- 再遇检查：终端 `joint_effort_*` 原文 -> CSV 最后 10 帧 j4 tau/pd_tau/qerr -> warning 是否出现
  -> 是否到 `22 Nm` hard fault -> 温度、姿态与 LT+B 接管。

#### 2026-08-01 楼梯前 depth_invalid 接管原因与故障前深度录像

- 状态：本地已验证，真机相机选项与楼梯复测待执行。影响范围为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 的 D435i 有效性门禁、故障诊断、FSM 转移日志和深度
  故障录像；网络输入 shape 与 `(0,5 m)/5` 训练归一化不变。Git 元数据不可用，分支、commit、
  PR 未知。
- 症状与证据：楼梯前终端明确打印 `[VisionLoco] hard fault: depth_invalid`，随后自动进入 Go2
  Passive 阻尼模式。最近多次日志在接近楼梯时投影整图/中下前方框无效率快速上升；例如
  `visloco_diag_1785532117.csv` 最后帧为 `dep_inval=0.468160`、诊断前框
  `front_inval=0.559028`，最近有效深度约 `0.218 m`。深度年龄最大约 `33 ms < 150 ms`，因此不是
  `depth_stale`。旧 CSV 的 front box 是中下方 `96x72`，而相机有效性使用中央三分之一全高度，
  故无法从旧 CSV 精确还原触发帧的实际中央无效率。
- 根因判断：`depth_invalid` 只由投影尺寸错误、整图无效率达到阈值或中央三分之一无效率达到阈值
  产生；尺寸在此前传感器长跑中稳定，楼梯前又出现近距离有效点与空洞同步增加，因此当前高置信
  根因为近距离楼梯面、遮挡/双目视差空洞或弱纹理使中央区域超过旧 `0.50` 门限。软件 `5 m`
  远距离边界不是此次故障原因。具体是哪些台阶/像素失效仍需新增图像证据确认。
- 修复与诊断：FSM 每个 registered check 现在携带原因，转移日志直接打印
  `VisionLoco -> Passive trigger=hard_fault:depth_invalid`；手柄、LowState timeout、翻倒和 SoftStop
  也有独立 trigger。深度故障现场额外打印 policy/camera frame、age、profile、投影尺寸、整图和
  中央三分之一无效率、最小/平均深度、序列号与固件。CSV 新增实际相机有效性两列。
- 相机与门限调整：保持训练口径 `(0,5 m)/5`，没有把 `>=5 m` 改成新的模型语义。D435i 请求
  High Density preset、发射器开启、设备允许的最大激光功率和自动曝光，并在启动时回读打印。
  整图/中央三分之一无效率从 `0.50` 开始警告，硬 invalid 边界放宽为 `0.70`。
- 接口核查：主机安装 librealsense `2.54.2`，controller 直接链接 `librealsense2.so.2.54` 并通过
  `rs2::pipeline` 获取 Z16、内参、帧号和时间戳。ROS 2 在线枚举没有 D435i 深度 topic；
  `/frontvideostream` 为 `Go2FrontVideoData` 彩色视频，`/pctoimage_local` 为 `PcToImage` 点云投影。
  实机 D435i 固件 `5.13.0.55`，SDK 工具推荐 `5.15.1`；本任务未执行固件升级，固件差异列为
  后续空洞问题的待验证因素。
- 故障录像：内存只保存最近 90 个唯一的 `320x180` 投影深度帧，约 3 秒；正常运行不持续写盘。
  `depth_invalid/depth_stale` 先触发 fail-closed 接管，再在 `exit()` 把环形缓冲写为 16-bit 毫米
  PGM（0 为无效）、8-bit 有效掩码 PGM（255 有效/0 无效）、无效像素为洋红色的 RGB PPM 预览和
  `manifest.csv`，目录为 `visloco_diag_TIMESTAMP_depth_fault/`，因此写盘不会延迟 Passive 切换。
  新增 `--record-depth`，可在 powered 模式手动退出后把同一环形缓冲写到
  `visloco_diag_TIMESTAMP_depth_capture/`。
- 本地验证：全核心并行构建成功；C++ normal/TSan 2/2、Python 14/14、100 帧 parity、JSON/脚本
  检查和 strict preflight 全部通过，joint 最大绝对误差 `8.345e-07`，preflight 未创建 LowCmd。
  当前隔离环境未检测到物理相机，所以真实设备选项支持/回读与故障录像尚待下一次真机运行验证。
- 制品：config SHA256
  `64d37ed0efa9f6c81f6f1309ed684f90cbb2f8284077fa35c22cd9e08298014b`；controller SHA256
  `dc06652c44f75a2d0d47bc1501d564c30511b1965b96e7a761c940c0b6c456d4`；artifact contract
  SHA256 `492a34f6729d94710e07d080eb6a0b357e6938e9734017c4dd103712499d72ce`；checkpoint/ONNX 未变。
- 遗留风险：把无效率硬线放宽到 `0.70` 会让模型在更多零深度输入下继续运行；楼梯属于高风险
  场景，首次复测应低速、短距离并保持 LT+B/实体急停。High Density 与最大激光功率可能增加
  功耗和对其他主动红外设备的干扰；自动曝光响应仍需图像确认。
- 回滚：恢复无效率 `0.60/0.50`、默认相机选项和旧 controller/config/contract；删除录像环形
  缓冲与具名转移日志不影响 checkpoint/ONNX。
- 再遇检查：启动 camera option 回读 -> `depth validity warning` ->
  `TAKEOVER -> Passive reason=...` -> FSM trigger -> depth fault 目录 manifest 最后帧 -> PGM 中零值
  分布与楼梯边缘 -> 再判断阈值、曝光、激光或安装角度。

#### 2026-08-01 DEPLOY-P3-DEPTH-GATE-002 深度内容与持续出帧门禁降为告警

- 状态：本地已验证并完成真机 sensor-only smoke，powered 真机验证待执行。影响范围仅为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 的 RealSense 启动、sensor-only/shadow smoke 和 powered
  VisionLoco 深度健康处理；训练归一化、投影、checkpoint、ONNX ABI、LowState 与机械安全门禁不变。
  工作区没有 Git 元数据，`git status --short --branch` 返回“not a git repository”，分支、commit、
  PR 均未知。
- 用户可见症状与证据：更换线材后的 `--sensor-only` smoke 中，RealSense 管线成功打开并回读
  `laser_power(max)=360.000`，但 3 秒内没有通过旧无效率门禁的投影帧，程序以
  `strict no-command fault: no valid projected depth frame within 3s` 退出。用户确认模型已经针对深度
  空洞强化训练，要求像素可用性、首帧等待以及运行中持续出帧/150 ms freshness 全部仅告警。
- 根因与排除方向：旧 `DepthFrame.valid` 同时混合了“结构完整帧已到达”和“整图/中央三分之一无效率
  低于 0.70”两种语义，启动、smoke、powered 主循环和可选 shadow gate 又共同把它当硬门禁；因此
  相机管线已经运行也会因为场景内容进入退出或 Passive。此次不是训练/部署张量 shape 或归一化
  不一致；USB 2.1 链路稳定性仍是独立待验证因素，不能通过拆门禁宣称已修复。
- 修改文件与核心修复：`DepthSource.h` 把 `valid` 收窄为真实相机帧结构完整，继续把无效像素编码为
  0 并记录比例；`main.cpp` 删除 3 秒首帧硬等待以及 stale/invalid 退出；
  `State_VisionLoco.cpp` 删除 `depth_invalid/depth_stale` hard fault、Passive 和策略发布阻断，改为
  `DEPTH_WARNING warning_only=true` 的状态切换日志，恢复时打印 recovered，并让可选启动诊断不再依赖
  深度内容或 cadence。配置删除两个 0.70 硬阈值；README、ARTIFACTS、artifact contract 和测试同步。
- 诊断保留：RealSense 管线启动失败仍阻止进入部署；无首帧时模型使用初始化的全零 `320x180`
  张量，断流时继续使用最近一帧。无效率、中央无效率、frame number 和 age 继续写日志；第一次健康
  告警请求在正常退出后写 `visloco_diag_TIMESTAMP_depth_warning/`，不会因录像触发接管。
- 验证：Release 全核心并行构建成功；C++ normal/TSan 2/2、Python 24/24、100 帧 parity 和带相机
  preflight 全部通过，joint 最大绝对误差 `8.345e-07`，preflight 明确
  `motor_command_channel_created=false`。真机 `--sensor-only` 在 360 激光功率下运行到 15 秒测试超时
  后干净退出：初始化后记录 577 个控制样本/11.52 s，相机帧号到 366；即使末帧整图/中央无效率为
  `0.9863/0.9952`，也只有 `DEPTH_WARNING warning_only=true`，没有 `DEPTH_FAULT`、
  `hard_fault:depth_*` 或 Passive，且 LowCmd 从未创建。该 smoke 未创建电机命令，不能替代 powered
  评估；当前用户权限无法读取 `dmesg`，本轮 USB/UVC 内核错误证据未取得。
- 回归防线：Python 合同测试要求配置中不存在无效率硬阈值、合同明确包含 `warning-only`，源码诊断
  marker 必须为 `DEPTH_WARNING`；sensor-only smoke 应在高无效率或帧 stale 时继续运行而非退出。
- 制品：任务名 `DEPLOY-P3-DEPTH-GATE-002`；模型 ID/checkpoint
  `p3nav8h-r1_884257-evalfix-v2` / `model.ckpt-highslow-884257.pkl`；commit/PR 不适用；修改后
  config SHA256 `86f667f8573604a87ddd9c1af96d0e8fd9fe313264b906b250bb9390f911abe2`；
  controller SHA256 `5eeb65aad78226903a30477138ea0ed72c82b9ea356c87cca2d97b2320619d91`；
  artifact contract SHA256 `7f72ebf8b8181ae51acd6023e2d59133e47529c27bea7ace30058a6ccb82b3f7`。
- 2026-08-02 提交格式更正：上述 config 哈希是当时真机工作树 CRLF 字节的历史证据，
  不予改写。新分支按仓库 `.gitattributes` 以 LF 提交，检出可复现的 controller config / policy
  deploy SHA256 分别为 `154d1d47d3c11330c4384eaf6620f41e93cea390e4175f0a4c38898578058ffb` /
  `5aa6ced0023641aed7a5c801f39dc290470a6a81003b62c273dca9a33e66b6a9`；机器可读 contract 与
  `ARTIFACTS.md` 已同步，preflight 应以新哈希校验。配置语义没有变化。
- 遗留风险：这项用户指定的放宽允许全零或长期陈旧深度持续驱动策略，摄像头物理断连后也不会自动
  Passive，风险显著高于原 fail-closed 行为。首次 powered 测试必须低速、短距离、开阔防滑，并让
  LT+B 与实体急停始终可立即触达。回滚方式是恢复 3 秒首帧、0.70 内容门禁、150 ms freshness
  hard fault 及旧 `depth_invalid/depth_stale` 接管分支。
- 再遇最短检查：`RealSense ... initialized` -> `DEPTH_WARNING warning_only=true` -> 确认没有
  `hard_fault:depth_*`/Passive -> frame number 与 age 是否恢复 -> 退出后查看 `_depth_warning/manifest.csv`
  -> 同时核查 `dmesg` 的 USB/UVC 错误，避免把链路断流误判成模型问题。

#### 2026-08-01 DEPLOY-P3-STARTUP-001 recurrent shadow 空跑导致起步前扑

- 状态：代码已修复待真机验证。影响范围仅为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 的 powered VisionLoco 启动时序；checkpoint、ONNX、
  proprio/depth/action 合同和独立深度孔洞填充均未改变。工作区仍无法读取 Git 元数据，
  `git status --short --branch` 返回“not a git repository”，分支、commit 和 PR 未知。
- 用户可见症状与证据：上一轮 23 次 powered 运行中，22 次通过内部启动门控；门控通常约
  `2.2 s` 后放行，此时 `vx` 已达到 `0.5 m/s`。放行后 0.6 s 内最大关节跟踪误差中位数
  `1.001 rad`、22 次中 17 次达到 `0.9 rad`、实测最大力矩中位数 `25.60 Nm`、最高
  `28.59 Nm`。例如 `visloco_diag_1785535126.csv` 中右前小腿目标与实测最大相差
  `1.065 rad`，与“刚开始走就向前栽”一致。
- 根因：旧启动门控在约 2 s 内每帧执行 `runner_->act()` 并把 fixed command 从 0 递增到
  `0.5 m/s`，因此 loco LSTM hidden 持续推进；同时 `shadow_gate_ready=0` 使策略目标不下发，
  `last_action_raw_` 保持 0，真机仍站立。门控打开后直接接入已推进两秒的行走 hidden，跳过了
  训练中的“推理 -> 执行动作 -> 回填 last_action -> 下一帧”闭环，并可能从任意步态相位开始。
- 排除方向：训练参考与部署的 `joint_ids_map`、默认站姿、action scale/offset、25/0.5 PD 增益和
  proprio 顺序一致；日志中 inference deadline miss、motion rejection/violation 均为 0。深度
  `depth_invalid` 是多次 Passive 的另一独立问题，不解释无深度故障运行中的重复起步冲击。
- 修复：新增 `strict_safety.startup_shadow_enabled`，默认 `false`。LT+X 进入 VisionLoco 时仍先
  reset recurrent state、清零 command 与 last_action；首个 LowState/深度有效且推理成功的帧立即
  发布经过 clip/物理限位/slew 的动作，并从该帧开始同步回填 executed last_action。原稳定/时间
  gate 仅在显式设为 `true` 时保留给静止诊断；`--suspended-test`/`--ground-test` 的动作审计和
  其他 freshness、机械、目标 stale 与接管保护不变。
- 修改文件：`State_VisionLoco.cpp/.h`、`StrictSafety.h`、controller `config.yaml`、C++/Python
  安全测试、实验 README/ARTIFACTS、`artifact_contract.json` 与共享接口契约。
- 本地验证：使用 `cmake --build ... --parallel $(nproc)` 全核构建成功；C++ normal/TSan 2/2、
  Python 14/14、100 帧 checkpoint->ONNX recurrent parity、JSON/Python 语法和 strict preflight
  全部通过，joint 最大绝对误差 `8.345e-07`；preflight 明确
  `motor_command_channel_created=false`。物理相机在隔离 preflight 中未检测到，本条尚无修复后
  powered 真机日志，因此状态不得升级为真机已验证。
- 制品：父模型/模型 ID/checkpoint 未变；checkpoint SHA256
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`，ONNX SHA256
  `6ccaf7e033240720a66019885df091097efa2d6d8962a5840966ede8b9de2d87`，config SHA256
  `77ecb3e82a4d5a18ff45b10f456df4d8741fc0a9ad1f6a595fbdf822815e9bb1`，controller SHA256
  `7bbe121de6c6ad0ba2efa6f7f4ed36a832e244c31e797233cecb1bc5277fbef4`，artifact contract SHA256
  `072714385892e7e7fbd316bbdb723d36ba94b75a04389addd6baff3140f6e4e1`；commit/PR 不可用。
- 遗留风险：立即闭环消除了 hidden/action 历史错位，但不能证明模型本身已具备稳定真机步态。
  `--fixed-vx` 仍关闭 tracking-error SoftStop，首次复测必须从架高 `--fixed-zero` 开始，再做平地
  `0.3 m/s` 短测，保持 LT+B/实体急停和防摔保护。不得因为本地测试通过直接尝试楼梯。
- 回滚：把 `startup_shadow_enabled` 显式设为 `true` 可恢复旧等待行为；代码级回滚为恢复初始
  `shadow_gate_ready=false`。旧行为会重新引入 recurrent/action 历史失配，只适合零命令静止诊断。
- 再遇检查：CSV 首帧 `shadow_gate_ready` 应为 1 -> 首帧 `vx` 应从约 `0.014 m/s` 起步 ->
  `last_action` 应从每帧实际 applied target 回填 -> 前 0.6 s tracking error/tau/pgx -> 再区分
  模型 sim2real 能力、足端打滑、相机故障或机械接管。

#### 2026-08-01 DEPLOY-P3-DIAG-001 自动接管原因未持久化且故障上下文不足

- 状态：本地已验证。影响范围仅为实验部署路线
  `deploy/sim2real_test_p3_standard_strict` 的控制器日志与诊断；安全阈值、状态转移、模型输入输出、
  checkpoint、ONNX 和训练/部署数值合同均未改变。工作区根目录仍无法读取 Git 元数据，
  `git status --short --branch` 返回“not a git repository”，当前分支、commit 和 PR 未知。
- 用户可见症状与关键证据：此前多轮自动进入 Passive/FixStand 时，逐帧 CSV 能看到故障前数据，
  但终端中的 `hard fault`、FSM `trigger=` 和退出原因没有写入持久文件；非深度故障只能根据末帧
  `tau/qerr/pgz` 反推，无法确认究竟是 `joint_effort_hard_N`、`bad_orientation`、LowState、推理异常
  或 `policy_target_stale`。历史日志中存在 FL thigh 力矩接近 `22 Nm` 和 tilt 接近 `1.0 rad` 的不同
  候选原因，证明仅靠 CSV 末帧不能给出确定结论。
- 根因：控制器默认 spdlog 仅输出终端，VisionLoco CSV 也不保存 FSM 转移原文；机械、姿态、LowState、
  策略目标 stale 和策略线程异常缺少统一的故障瞬时快照。另有诊断实现风险：策略线程的 `frame_`
  与 `last_exec_cmd_` 若直接由 FSM 线程读取会产生数据竞态，因此不能简单在检查回调中打印原字段。
- 排除方向：没有修改力矩、速度、位置、温度、深度无效率或 stale 时间阈值；没有把警告升级为接管，
  也没有关闭既有保护。此次修复不声称已解释历史每一次失败，只确保下一次能保存精确触发原因。
- 修改文件：controller `main.cpp`、`State_VisionLoco.cpp/.h`、实验 README/ARTIFACTS、
  `artifact_contract.json`、Python 回归测试和本台账。
- 核心修复：每个非 preflight 运行创建毫秒级唯一的
  `logs/loco/logs/strict_controller_TIMESTAMP.log`，同时写终端和文件，warning 及以上立即 flush；新增
  `FAULT_SNAPSHOT` 统一记录 severity/reason、policy frame、command、LowState tick/age、投影重力/倾角、
  最大绝对 q/dq/tau、最高温度、电池、最近 tracking/requested/applied/PD 指标和深度帧/age/无效率。
  `MECHANICAL_LIMIT`、`DEPTH_FAULT`、`LOWSTATE_FAULT`、`POLICY_TARGET_STALE`、`BAD_ORIENTATION` 使用
  独立标签；策略线程未捕获异常也保存最后 snapshot。跨线程帧号和命令改用独立 atomic 诊断镜像，
  姿态/stale 日志按一次故障只打印一次，避免 1 kHz FSM 重复刷屏。
- 本地验证：使用 `cmake --build ... --parallel $(nproc)` 全核构建成功；C++ normal 与 TSan 均通过，
  Python 15/15 通过（新增持久日志/标签契约测试），100 帧 checkpoint->ONNX recurrent parity 通过，
  joint 最大绝对误差 `8.345e-07`；JSON 解析、冲突标记/尾随空白检查和 strict preflight 均通过，
  preflight 明确 `motor_command_channel_created=false`。本轮未启动 DDS/LowCmd，尚无真机生成的新版
  `strict_controller_*.log`，故真机日志可写性和断电场景保留程度待下一轮验证。
- 制品：父模型/模型 ID/checkpoint 未变；checkpoint SHA256
  `8dc9d6028bd8850a3e59cfde2bd2ee1fe5b6768f1af47477a20c28a831edcb23`，ONNX SHA256
  `6ccaf7e033240720a66019885df091097efa2d6d8962a5840966ede8b9de2d87`，config SHA256
  `77ecb3e82a4d5a18ff45b10f456df4d8741fc0a9ad1f6a595fbdf822815e9bb1`，controller SHA256
  `add8cb1271e9640f7867b1b3328f53c84a766cb4fc086c75d52c72e9c2465708`，artifact contract SHA256
  `45c1f5e7a56d41939b71acfb5c7dc764a235bc0ad975b963203559523fb86ee4`；commit/PR 不可用。
- 长期防线：Python 测试验证持久 file sink 与六类结构化标签存在；TSan 覆盖控制器共享安全类型，
  后续真机日志分析应先读取 `strict_controller_*.log` 的 `trigger=` 与 `FAULT_SNAPSHOT`，再把时间点与
  `visloco_diag_*.csv`、深度 manifest 对齐，禁止继续只凭最后一行 CSV 猜测根因。
- 遗留风险：basic file sink 能保证 warning/critical 在进程正常运行时即时 flush，但掉电、存储设备
  故障或内核未落盘仍可能丢失最后数据；info 周期日志在非故障异常掉电时可能未全部刷新。下一轮真机
  应验证文件创建、故障行完整性和 FSM trigger 顺序，之后才能升级为真机/平台已验证。
- 回滚：移除 `setup_runtime_logging()` 和新增结构化日志/atomic 诊断镜像，恢复旧 controller 二进制；
  回滚只降低可观测性，不需要修改 config、checkpoint 或 ONNX。
- 再遇检查：最新 `strict_controller_*.log` -> 首个 `MECHANICAL_LIMIT/DEPTH_FAULT/LOWSTATE_FAULT/`
  `POLICY_TARGET_STALE/BAD_ORIENTATION` -> 紧随其后的 `FAULT_SNAPSHOT` -> FSM `trigger=` ->
  `EXIT due to safety takeover` -> 按 policy frame 对齐 `visloco_diag_*.csv` 和深度 manifest。

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
