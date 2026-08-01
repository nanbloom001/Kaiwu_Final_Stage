# D435i 深度图与滤波参数调试工具

该工具只连接 Intel RealSense 深度相机，不初始化 DDS、LowState 或 LowCmd，适合在机器狗桌面环境中
观察原始深度、部署口径无效像素，以及不同 RealSense 后处理参数的即时影响。

## 启动

从机器狗本地桌面终端启动：

```bash
cd /home/unitree/Kaiwu_Final_Stage-main/deploy/sim2real_test_p3_standard_strict
bash scripts/run_depth_tuner.sh
```

启动脚本会优先使用当前终端的 `DISPLAY/XAUTHORITY`；若为空，会尝试发现当前 `unitree` GNOME
桌面会话。也可以显式指定显示号，例如 `--display :1`。SSH 使用 X11 转发时可以直接运行：

```bash
ssh -X unitree@ROBOT_IP
cd /home/unitree/Kaiwu_Final_Stage-main/deploy/sim2real_test_p3_standard_strict
bash scripts/run_depth_tuner.sh
```

设备枚举与无 GUI 短测：

```bash
bash scripts/run_depth_tuner.sh --list-devices
bash scripts/run_depth_tuner.sh --headless --frames 300
bash scripts/run_depth_tuner.sh --headless --frames 300 --spatial --temporal --hole-fill
```

默认使用当前严格部署审计过的 `480x270@30 Z16`。相机被 `realsense-viewer`、`rs-depth` 或控制器
占用时，工具会收到 `Device or resource busy`；同一台 D435i 不能同时被这些程序独占采流。

## 界面与手动输入

主窗口包含四个区域：

- `RAW DEPTH`：未经软件滤波的深度，部署无效像素显示为洋红色。
- `FILTERED DEPTH`：按当前滑块顺序执行空间、时间和独立孔洞填充后的深度。
- `RAW INVALID MASK`：白色表示无效，黑色表示有效。
- `FILTER EFFECT`：绿色表示被滤波恢复，红色表示滤波后新丢失，白色表示仍然无效。

两条黄色竖线标出部署检查使用的中央三分之一区域。界面同时显示整图与中央区域无效率。
部署口径中，原始值为零、非有限值或距离 `>=5 m` 都算无效。

左侧参数面板按 `Filters`、`Camera`、`Monitor & Logs` 三个页签分组。每个数值参数都有滑块、
手动输入框和设备支持范围；在输入框中键入数值后按 Enter 即可下发，超出范围的值会被夹到设备
实际范围并按设备步长量化。滤波开关与对应参数放在同一组，避免把空间、时间和孔洞填充混在一起。
每一行的 `Reset` 只把该参数恢复到 SDK/滤波器声明的默认值；范围下方同时显示该默认值。
相机硬件参数默认等待最后一次变化稳定 `400 ms` 后才下发，因此拖动滑块不会再逐格发送 UVC 命令。
需要调整时可通过 `--camera-debounce-ms` 改变等待时间。
自动曝光上限和自动增益上限由 D435i 固件声明为“下一次开流生效”；这些行会显示
`restart stream`，修改后点击 Camera 页的 `Apply limits & restart stream`。按钮仅在相机状态为
`ONLINE` 且确有待生效限制时可用；设备掉线或重连期间相机硬件控件和该按钮都会禁用。普通采集循环不会反复写这四个
restart-required 控件；按钮只尝试一次尚未生效的曝光/增益限制，不会重放 Laser、preset、emitter
或其他相机参数。若固件拒绝全部待应用值，工具记录 `STREAM_RESTART_SKIPPED` 并保持当前数据流，
不会为了无效请求 stop/start pipeline。
USB 2.x 是当前受支持运行模式，只产生 `USB2_LINK policy=warning_only` 诊断；不会阻止启动、采流、
自动重连，也不会按 USB 类型收紧 Laser power 的设备声明范围。参数写入仍可能被相机固件或物理链路
拒绝，此时按实际错误告警，但不把“未使用 USB 3.x”本身作为门禁。

键盘操作：

- `q` 或 `Esc`：退出。
- `s`：保存当前原始/滤波深度、掩码、界面截图、NumPy 数组和参数 JSON。
- `p`：在终端打印当前参数及统计。
- `r`：将全部参数恢复到各自声明的默认值，并关闭三个软件滤波器。

输出默认保存到：

```text
logs/depth_tuner/depth_tuner_YYYYMMDD_HHMMSS/
```

每次运行还会持续写入：

- `events.jsonl`：参数变更、设备通知、设备掉线/重连、超时、跳帧和长间隔事件，事件立即刷新到磁盘。
- `metrics.csv`：约每秒一行的延迟、帧间隔、CPU、内存、Jetson GPU/温度和无效像素统计。
- `display_latency.csv`：每个实际提交到 GUI 的帧一行，绑定 frame number、绝对帧龄、GUI 调度/转换、
  曝光、Frame queue 和当帧 temporal 参数。训练延时随机化优先统计 `frame_age_receive_ms`，不要用 GUI
  submit 延迟代替模型输入延迟。
- `session.json`：退出时的最终参数和统计。

## 延迟、算力与断流判断

界面顶部把延迟拆成可验证的时钟边界：

- `Frame age @ receive`：主机收到帧时的绝对帧龄，是估算训练/部署相机输入延迟的主指标；包含相机
  内部排队和 USB 传输，但不包含调参器彩色渲染。
- `Frame age @ app ready`：调参器完成滤波和四宫格渲染时的绝对帧龄。
- `Frame age @ GUI submit`：Tk 图像提交给 X11 时的绝对帧龄；包含 GUI 调度和图像转换，但不包含
  compositor、显示器扫描和像素响应。真实光学端到屏幕延迟仍需高速摄影或光电测量。
- `GUI scheduling` 与 `GUI conversion`：分别为图像准备好后等待 GUI tick、以及 RGB/缩放/PhotoImage
  转换时间。
- `Queue extra (relative)`：只表示相对本会话最小 sensor/host offset 的额外积压，不是绝对延迟。

绝对帧龄只在 RealSense 报告 `global_time` 或 `system_time` 且与主机墙钟偏移合理时计算；独立
`hardware_clock` 或同步异常时显示 `--` 并记录 `latency_clock_valid=0`，不会用相对 offset 冒充准确值。
`Capture wait` 是线程阻塞等待下一帧的时间，`Host/Sensor gap` 是帧周期，都不能直接累加为延迟。
采集保持配置的 30 FPS，GUI 以最高 20 FPS 读取最新一帧，不建立待显示帧队列；显示频率低于采集频率
是有意的 CPU/延迟折中，不表示相机丢帧。

### 无头 10 秒延迟测试

固定参数模板为 `configs/depth_latency_test.yaml`。默认先预热 2 秒，再逐帧测量 10 秒：

```bash
bash scripts/run_depth_latency_test.sh
```

指定其他 YAML 或临时覆盖测量时长：

```bash
bash scripts/run_depth_latency_test.sh --config /absolute/path/test.yaml
bash scripts/run_depth_latency_test.sh --duration 30
```

终端直接打印 receive/processed 的平均与最大延迟。每次运行保存：

- `frames.csv`：10 秒内每一帧的 receive、processed、处理耗时、时间戳域、曝光和帧间隔。
- `summary.json`：平均、最大、P50、P95、P99、有效帧比例、实际滤波/相机参数、丢帧和输出路径。
- `config.yaml`：本轮合并默认值后的完整配置快照。

输出目录默认为 `logs/depth_latency_test/depth_latency_*`。训练域随机化使用
`summary.json.frame_age_receive`；若部署端在送入网络前还有额外预处理，则参考
`frame_age_processed`。默认要求至少 95% 帧具有同步绝对时间戳，否则测试返回失败。USB 2.x 只输出
warning，不改变参数范围或阻止测试。`strict_sensor_options=true` 时任一 YAML 硬件参数未被 SDK 回读
确认都会终止测试，避免使用与配置不一致的数据。

快速查看异常事件：

```bash
latest=$(find logs/depth_tuner -maxdepth 1 -type d -name 'depth_tuner_*' | sort | tail -n 1)
tail -f "$latest/events.jsonl"
rg 'STREAM_TIMEOUT|STREAM_GAP|FRAME_JUMP|DEVICE_' "$latest/events.jsonl"
tail -n 20 "$latest/metrics.csv"
```

事件含义：

- `DEVICE_TOPOLOGY_CHANGED` 且 `selected_device_present=false`：设备已经从 USB 总线消失，优先查线缆、
  插头、Hub、供电和接口机械固定。
- `STREAM_TIMEOUT`：在配置的等待时间内没有新帧；记录会同时查询设备是否仍可枚举。
- `STREAM_ENTERED_RECONNECT_WAIT` / `WAITING_FOR_DEVICE`：连续超时且设备不在 USB 总线时，工具停止
  失效 pipeline，但保留 GUI 并等待同一序列号重新出现；不会再把临时缺席转换成 GUI 崩溃。
  进入等待前的次数可通过 `--disconnect-abort-count` 调整，确认掉线的宽限时间可通过
  `--reconnect-grace-seconds` 调整。
- `STREAM_RESTART_DEFERRED`：按钮或自动恢复打开 pipeline 失败，工具已经转入设备等待状态；先查
  `lsusb -t` 和内核 USB 日志，不要连续点击重启。
- `STREAM_GAP` 且 `suspected_cause=camera_or_usb_stall`：相机时间戳和主机时间同时出现长间隔，更像
  相机或 USB 链路停顿。
- `STREAM_GAP` 且 `suspected_cause=host_processing_or_delivery_stall`：相机时钟仍正常而主机收到帧较晚，
  结合 CPU/GPU/渲染耗时判断主机负载。
- `FRAME_JUMP`：帧号不连续，记录中包含估算丢帧数。

晃动相机时，双目匹配失败和无效像素短时增加是正常现象，但帧号停止、整幅画面冻结或设备从 USB
总线消失不是正常视觉现象。当前机器上的实测链路为 `480M / USB 2.x`，该模式受支持且不会作为
启动或采流硬门禁；已出现的内核级 `USB disconnect` 仍应结合线材、Hub、供电和机械固定诊断。

## 软件滤波参数

处理顺序固定为：

```text
raw Z16 -> spatial -> temporal -> independent hole filling -> visualization
```

### 空间滤波

- `Spatial magnitude`：空间滤波迭代次数。越大通常孔洞和噪声越少，但边缘更容易变钝、CPU 耗时更高。
- `Spatial alpha`：当前像素权重，通常 `0.25..1`。越低平滑越强；接近 `1` 时更保留原始深度和边缘。
- `Spatial delta`：允许参与平滑的相邻深度差阈值。越大越容易跨越深度变化进行平滑，可能模糊台阶边缘；
  越小越保边，但去噪能力下降。
- `Spatial holes`：空间滤波内部的水平孔洞填充半径。SDK 通常使用
  `0=关闭, 1=2px, 2=4px, 3=8px, 4=16px, 5=无限制`。

### 时间滤波

- `Temporal alpha`：当前帧权重。越低越依赖历史帧、画面更稳定但延迟和拖影更明显；越高越跟随当前帧。
- `Temporal delta`：允许使用历史深度的变化阈值。越大越容易把历史值带到当前帧，动态边缘拖影风险更高；
  越小越快响应真实深度变化。
- `Temporal persistence`：无效像素从历史帧保留的规则，范围通常为 `0..8`。数值越激进，静态孔洞
  越可能被历史数据填上，但移动物体和台阶边缘也更容易产生拖影。`0` 关闭；`1..7` 使用不同的
  最近帧有效性条件；`8` 可无限期保留历史值，真机导航通常不建议使用。

### 独立孔洞填充

`Hole fill mode` 不是阈值，而是从哪个邻域值填充：

- `0`：取左侧像素。
- `1`：取邻域中最远深度。
- `2`：取邻域中最近深度。

该滤波会直接制造原始相机没有测得的深度值。楼梯边缘、细杆和落差附近必须同时观察
`FILTER EFFECT`，不能只看无效率是否下降。

## 相机硬件参数

工具按当前 D435i/固件实际支持的范围动态创建滑块：

- `Visual preset`：D400 深度预设。常见映射为 `0=Custom`、`1=Default`、`2=Hand`、
  `3=High Accuracy`、`4=High Density`、`5=Medium Density`。切换预设可能同时重写曝光、镭射和
  深度算法表，因此应一次选择后等待画面稳定。
- `Emitter`：红外投射器模式，通常 `0=关闭`、`1=开启`、`2=自动`。关闭可减少功耗和多相机干扰，
  但无纹理表面的深度有效率通常下降。
- `Laser power`：红外散斑投射强度，单位是设备 SDK 刻度，不应直接解释为瓦特或毫瓦。提高它可改善
  无纹理、弱光表面的匹配，但会增加功耗、近距离饱和、散斑和多相机干扰。当前 USB 2 Hub 链路下
  已确认 `330/360` 会触发协议错误和掉线，因此工具在 USB 2.x 下默认限制到 `240`；物理链路修复前
  建议保持默认 `150`。
- `Auto exposure`：`1` 时由相机自动调节曝光，`0` 时允许手动设置 `Exposure/Gain`。导航动态场景中
  自动曝光通常更稳，但亮暗突变时会有收敛过程。
- `Exposure`：手动曝光时间，D435i 通常以微秒报告。增大可提升暗处有效率，但运动模糊、帧内延迟和
  过曝风险上升；只有关闭自动曝光后才下发。
- `Gain`：传感器模拟增益。增大可提亮弱信号，但同时放大噪声和错误匹配；只有关闭自动曝光后才下发。

### 比赛相关的延迟与自动曝光约束

Camera 页还暴露当前 D435i 实际支持、且会直接影响移动导航输入的参数：

- `Frames queue size`：相机内部允许应用持有的帧数。默认通常为 `16`；增大可减少应用层丢帧，但会
  增加拿到旧帧的风险。比赛建议先测试 `2`，若主机调度稳定再比较 `1`。
- `Exposure limit enabled` / `Exposure limit`：限制自动曝光的最大曝光时间。当前 30 FPS 移动场景
  建议从 `8500 us` 开始，不足时比较 `10000..12000 us`，避免直接放宽到一帧周期。
- `Gain limit enabled` / `Gain limit`：限制自动增益噪声。建议从 `64` 开始，深度不足时依次比较
  `96` 和 `128`，不建议直接使用设备最大 `248`。
- `Global time`：保持相机时间戳与主机全局时间机制开启，用于断流和排队诊断；通常保持 `1`。
- `Internal error polling`：让 SDK轮询并上报相机内部错误；比赛诊断保持 `1`。

主面板从深度帧 metadata 读取 `Actual exposure`、`Actual gain`、`Actual laser`，不会为了刷新监控而
持续发送 UVC `get_option` 请求。面板同时显示当前 Frame queue 和 USB link；这些字段也写入
`metrics.csv`，便于把运动模糊、噪声、排队延迟和 USB2 断流分开分析。

Laser 状态分成三个值：

- `Laser requested`：用户最后请求的值。
- `Laser applied`：UVC set/get 已确认的值；设置失败时显示 `--`，不会伪装成已经应用。
- `Laser frame actual`：深度帧 metadata 报告的该帧实际值。如果 UVC 写入成功但回读因协议错误失败，
  metadata 与 requested 一致时会记录 `OPTION_CONFIRMED_BY_FRAME_METADATA` 并补确认 applied。

相机从 USB 总线消失时，Camera 硬件控件自动禁用，状态依次显示 `DEVICE_LOST`、
`WAITING_FOR_DEVICE`、`RECONNECT_PENDING`、`RECONNECTING`。同一序列号重新出现后，采集线程自动
stop/start pipeline、绑定
新的 video 节点和 depth sensor，并记录 `DEVICE_CONTROL_REBOUND`。重连只恢复 Frame queue、Global
time 和 Error polling；Laser、曝光、增益、preset 和 emitter 使用设备重新枚举后的当前值，不自动
重放可能触发掉线的旧请求。设备持续缺席时 GUI 保持打开并每 2 秒记录一次等待状态；物理设备没有
重新出现在 USB 总线时，软件无法完成恢复。

建议的第一组比赛诊断值为：

```text
Frames queue size = 2
Auto exposure = 1
Exposure limit enabled = 1
Exposure limit = 8500 us
Gain limit enabled = 1
Gain limit = 64
Global time = 1
Internal error polling = 1
```

HDR、HDR sequence、多相机同步、外部触发、Decimation 和 Advanced Mode 立体匹配表没有加入主 GUI。
它们要么针对当前不存在的多相机场景，要么改变帧间分布、投影几何或训练输入语义，不属于比赛现场
应动态调整的参数。

自动曝光开启时，手动 `Exposure/Gain` 不下发。改变 `Visual preset` 可能重置其他硬件参数，工具会
在预设变化后重新同步其余滑块。所有成功或被设备拒绝的参数设置都会打印到终端。

## 显示与诊断参数

- `Display minimum/maximum`：只控制伪彩显示映射，不改变原始深度、滤波结果或部署端固定的 `5 m`
  有效性口径。缩小显示范围可以更清楚地区分近距离深度变化。
- `Gap warning`：主机帧间隔超过该毫秒值时记录 `STREAM_GAP`。它只影响诊断告警，不改变相机采集、
  控制器门禁或模型输入。

## 与正式部署的边界

该工具只用于比较和记录参数，不会自动修改 controller `config.yaml`。当前正式部署仍保持
`spatial=false`、`temporal=false`、独立孔洞填充关闭。选定参数后，应结合原始图、滤波图、动态场景、
楼梯边缘和至少 60 秒 sensor-only 日志审查，再明确地更新正式配置和回归记录。
