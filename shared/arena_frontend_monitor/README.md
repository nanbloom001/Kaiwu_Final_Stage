# 腾讯竞技平台训练监控采集工具

本目录用于从腾讯竞技平台网页端采集训练日志和监控曲线，并生成适合后续分析的 JSON、JSONL 与本地 HTML 快照。

该工具是独立的人工运维工具，不属于 `server/` 或 `deploy/` 的运行时依赖，也不会访问训练容器内部接口。

## 适用范围

- 复用已经登录腾讯竞技平台的 `agent-browser` 会话和本地持久化浏览器 profile。
- 采集“监控总览”中按需加载的指标卡片。
- 采集“训练日志”及重要错误、告警和训练状态记录。
- 导出 `GetTrainMetricRange`、`GetTrainLog` 前端请求。
- 对采集结果做降采样、平滑和摘要处理。
- 在本机启动只监听 `127.0.0.1` 的简易监控页。

## 前置条件

- Python 3.9 或更高版本。
- `agent-browser` 命令已经安装并位于 `PATH`。
- 首次运行时在可视化 `agent-browser` 窗口中完成一次腾讯竞技平台登录。
- 当前标签页已经打开目标训练任务的监控页面，或运行时显式传入监控 URL。

默认会话名为 `tencent-arena-persistent-v2`，也可以通过环境变量覆盖：

```bash
export AGENT_BROWSER_SESSION=tencent-arena-persistent-v2
export AGENT_BROWSER_SESSION_NAME=tencent-arena-persistent-v2
```

### 登录状态持久化

工具默认把独立 Chrome profile 保存在：

```text
shared/arena_frontend_monitor/runtime/browser_profile/
```

该目录会保存 Cookie、localStorage 等登录状态，并已被 Git 忽略。第一次使用
`manual_metric_recorder.sh` 时正常登录一次；以后使用同一 profile 的采集命令会自动复用登录态，
不需要再次输入密码。平台主动让会话过期时，才需要重新登录。

新版使用独立的 `tencent-arena-persistent-v2` 会话，避免复用升级前已经启动、仍绑定临时
Chrome 目录的旧 `tencent-arena` daemon。旧窗口不需要强制关闭；第一次运行新版手动入口时
登录一次即可。确认新版稳定后，可在旧会话没有采集任务时执行
`agent-browser --session tencent-arena close`。

如果本地旧 `.env` 仍把 session 和 session-name 写成 `tencent-arena`，工具会自动迁移到 v2
并在终端打印提示。只有确实需要操作旧 daemon 时才设置
`AGENT_BROWSER_ALLOW_LEGACY_SESSION=1`；普通抓取不要设置该兼容开关。

需要自定义位置时，可创建仅保存在本机的配置：

```bash
cp shared/arena_frontend_monitor/.env.example \
  shared/arena_frontend_monitor/.env
```

然后修改 `.env` 中的 `AGENT_BROWSER_PROFILE`。仓库内路径必须位于已忽略的 `runtime/` 下，
相对路径以本工具目录为基准；也可以使用仓库外的绝对路径。工具会拒绝把 profile 写入其他
仓库目录，避免 Cookie 作为普通未跟踪文件出现。所有 Python 和 shell 入口都会自动读取该文件，已存在的进程环境变量优先于
`.env`。不要在 `.env` 中保存密码或粘贴原始 Cookie；profile 本身已经负责安全地保存浏览器
登录状态。

所有入口默认统一使用 headed 模式，并把 session、session-name 和 profile 作为显式
`agent-browser` 命令行参数传入，
不会只依赖已有 daemon 的启动环境。手动入口启动时还会打印实际 session 和 profile 路径；若
输出不是上述 `runtime/browser_profile/` 或你在 `.env` 中指定的绝对路径，应先停止采集并检查
环境变量覆盖。

先执行不会打开、切换或刷新网页的离线检查：

```bash
python3 shared/arena_frontend_monitor/network_export.py --check
```

该检查只验证 Python、`agent-browser --version`、各配套 CLI 的 `--help`、固定
HAR fixture 解析和输出目录可写性。
它不验证腾讯登录态；首次真实采集仍需人工打开并登录目标监控页面，后续会复用本地 profile。

## 推荐用法

### 自动采集监控总览与训练日志

先在浏览器中切换到目标任务的监控页面，再运行：

```bash
shared/arena_frontend_monitor/collect_monitor.sh
```

只采集指定指标组：

```bash
shared/arena_frontend_monitor/collect_monitor.sh \
  --group "训练进展" \
  --group "Reward指标"
```

默认不会主动猜测或切换到其他监控标签页。确实需要指定页面时，可显式传入：

```bash
MONITOR_URL='<监控页面地址>' \
shared/arena_frontend_monitor/collect_monitor.sh
```

### 手动滚动并持续记录曲线

当自动滚动无法完整触发懒加载卡片时，可以使用可视化浏览器手动展开和滚动：

```bash
shared/arena_frontend_monitor/manual_metric_recorder.sh '<监控页面地址>'
```

macOS、Windows/WSL 或安装了剪贴板工具的 Linux 也可直接读取剪贴板中的地址：

```bash
shared/arena_frontend_monitor/manual_metric_recorder.sh --clipboard
```

脚本打开页面后会等待回车。先在浏览器中展开、滚动或刷新所需卡片，再回到终端按回车开始记录。结束记录使用 `Ctrl-C`，随后会自动执行后处理。

### 导出前端网络数据

```bash
python3 shared/arena_frontend_monitor/network_export.py
```

默认使用自动模式；也可明确选择 HAR 或网络请求采集：

```bash
python3 shared/arena_frontend_monitor/network_export.py --capture-mode har
```

需要把“未抓到任何指标”视为失败时使用：

```bash
python3 shared/arena_frontend_monitor/network_export.py \
  --capture-mode har \
  --fail-on-empty-metrics
```

致命错误或上述空指标条件会返回非零退出码，同时仍保留失败
`summary.json` 和诊断目录，便于自动化发现失败而不丢失证据。

### 生成本地快照页面

单次采集：

```bash
python3 shared/arena_frontend_monitor/frontend_monitor.py collect-once
```

持续采集并启动本地页面：

```bash
python3 shared/arena_frontend_monitor/frontend_monitor.py serve --port 8877
```

访问 `http://127.0.0.1:8877`。服务不会监听公网地址。

### 处理已有采集目录

```bash
python3 shared/arena_frontend_monitor/postprocess_monitor_capture.py \
  shared/arena_frontend_monitor/runtime/manual_metric_recorder/sessions/<时间戳>
```

## 输出目录

默认输出到：

```text
shared/arena_frontend_monitor/runtime/
```

该目录已被 Git 忽略。需要把产物放到仓库外时，设置：

```bash
export ARENA_MONITOR_RUNTIME_DIR="$HOME/arena-monitor-runtime"
```

主要结果包括：

- `summary.json`：采集范围、状态和错误摘要。
- `training_logs.json`：训练日志及重要日志筛选结果。
- `groups/*.json`：各监控组的请求和指标数据。
- `ai_readable_metrics.json`：适合自动分析的指标摘要。
- `all_metric_series_summary.json`：全部曲线统计摘要。
- `all_metric_series_lttb.json`：曲线降采样结果。
- `all_metric_series_smoothed.json`：曲线平滑结果。
- `analysis_report.md`：自动生成的事实性分析报告。

## 文件说明

| 文件 | 用途 |
|---|---|
| `collect_monitor_overview.py` | 自动展开监控组并采集指标与日志 |
| `collect_monitor.sh` | 自动采集的推荐入口 |
| `manual_metric_recorder.py` | 持续记录手动触发的指标请求 |
| `manual_metric_recorder.sh` | 打开页面、等待人工操作并自动后处理 |
| `network_export.py` | 导出监控前端 API 请求与响应 |
| `postprocess_monitor_capture.py` | 将原始采集整理为摘要、降采样和平滑数据 |
| `frontend_monitor.py` | 生成快照和本地 HTML 页面 |
| `metric_probe.py` | 对当前页面做一次轻量调试探测 |

## 安全与使用约束

- 不要把平台 Token、Cookie、HAR、训练日志或采集结果提交到 Git。
- 监控 URL 可能包含任务标识或查询参数，只应通过命令行或环境变量临时传入。
- 工具在本机 Git 忽略目录中保存独立浏览器 profile，不保存明文密码，也不分发登录凭据。
- `runtime/browser_profile/` 和 `.env` 都属于敏感本地状态，不要复制到共享目录或提交到 Git。
- 页面结构变化后，自动点击和滚动选择器可能需要同步调整。
- 采集过程中不要同时让其他程序控制同一个 `agent-browser` 会话。
- 训练已停止时，页面可能需要先启用“每 5 秒自动刷新”才能重新请求历史指标。

## 静态验证

当前状态：**静态与 fixture 验证可执行；真实腾讯登录态 E2E 待人工打开页面后验证。**
远程仓库版本是本工具的唯一代码来源，运行时采集结果不进入 Git。

HAR 是主要采集路径，因为训练指标位于 iframe 并采用懒加载；普通
`agent-browser network requests` 仅作为 HAR 没有获得指标时的备用路径。

```bash
PYTHONPYCACHEPREFIX=/tmp/arena-monitor-pycache \
python3 -m py_compile shared/arena_frontend_monitor/*.py

bash -n shared/arena_frontend_monitor/collect_monitor.sh
bash -n shared/arena_frontend_monitor/manual_metric_recorder.sh

python3 -m unittest discover \
  -s shared/arena_frontend_monitor/tests \
  -p 'test_*.py'
```

完整端到端验收还必须确认：已有登录成功的 `/p/v5/exp/monitor` 标签页、
自动刷新已启用、HAR 中至少存在一个 `GetTrainMetricRange`、日志可解析且
coverage report 与页面指标清单一致。工具不会为了通过检查自动导航到其他页面。
