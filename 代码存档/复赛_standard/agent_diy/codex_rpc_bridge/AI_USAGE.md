# Codex RPC Bridge

这个目录提供一个轻量 HTTP RPC 服务，用于让外部 AI agent 操作腾讯开悟 / code-server 开发容器中的代码。

RPC bridge 放在：

```text
agent_diy/codex_rpc_bridge/
```

运行时 token、日志和备份放在：

```text
agent_diy/codex_rpc_bridge_runtime/
```

运行时目录不要提交到仓库。

## 文档入口

- [SETUP.md](./SETUP.md)：首次安装和部署。包括浏览器控制能力安装、登录、复制/创建文件、容器初始化、测试。
- [DAILY_USAGE.md](./DAILY_USAGE.md)：首次安装后的日常 AI 操作流程。包括如何确认服务、如何通过浏览器上下文调用 RPC、读写文件和执行检查。

## 首次使用前检查

外部 AI 需要能控制一个已登录 Tencent Arena 的浏览器。推荐使用：

```bash
npm i -g agent-browser
agent-browser install
agent-browser --version
```

Windows PowerShell 如果拒绝 `npm.ps1`，改用：

```powershell
npm.cmd i -g agent-browser
```

如果 `agent-browser` 命令不可见，检查并加入 npm 全局目录：

```text
C:\Users\<user>\AppData\Roaming\npm
```

登录态文件 `tencent-arena-auth.json` 只有在用户明确授权后才能保存，且不能提交。

Windows PowerShell 打开开发容器的常用写法：

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
agent-browser --headed --state .\tencent-arena-auth.json open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

## 文件说明

- `codex_file_rpc.py`：RPC 服务脚本。
- `start_rpc.sh`：容器内一键启动脚本。
- `sync_repo_to_container.sh`：本地仓库一键安全同步到容器。
- `test_sync_repo_to_container.sh`：同步脚本本地/远程测试。
- `keepalive_tencent_arena.sh`：底层保活循环。
- `start_keepalive.sh`：本机侧保活启动/状态/停止管理脚本。
- `Mac/`：macOS/Linux shell 入口脚本。
- `win/`：Windows PowerShell 入口脚本。
- `README.md`：快速入口。
- `SETUP.md`：首次安装指南。
- `DAILY_USAGE.md`：日常使用指南。
- `AI_USAGE.md`：本入口文档。

## AI agent 主动调用规则

其他 AI agent 看到本目录时，应按下面规则主动选择工具：

- 严格区分首次初始化验证和日常使用。只有在尚未确认 `IDE_ID`、登录态和 RPC
  代理地址时，才通过浏览器新开标签页访问 `${BASE}/api/health`；必须避免覆盖
  当前 code-server IDE 页面。若已经有开发容器 IDE 标签页（例如
  `/p/common/competition/ide/...`）且 RPC 已通过脚本或 fetch 验证，日常操作不要再
  新开 health 页面；后续 health、read、write、exec 等操作直接走已有
  IDE/session 的 RPC/fetch。

- 用户要求“同步到容器 / 部署到容器 / 对比本地和容器 / 上传本地仓库”时：
  先运行本地 dry-run：

  ```bash
  bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dry-run
  ```

  确认文件列表合理后，再在用户已提供 token 且 RPC 可用时执行：

  ```bash
  CODEX_RPC_TOKEN="<normal token>" \
  CODEX_RPC_ADMIN_TOKEN="<admin token>" \
  bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --apply --py-compile
  ```

  同步目标可以解析到 RPC root
  `/data/projects/legged_robot_competition_26` 或平台代码工作区
  `/workspace/code`。这是为了兼容 `agent_ppo` 等目录的容器映射；不要绕过
  脚本对内部 symlink、token、日志、模型权重和训练产物的保护。

- 用户要求“测试同步脚本 / 检查安全性”时：

  ```bash
  bash agent_diy/codex_rpc_bridge/test_sync_repo_to_container.sh --local
  ```

  只有在用户确认 RPC 已启动并提供 token 后，才运行远程测试：

  ```bash
  CODEX_RPC_TOKEN="<normal token>" \
  CODEX_RPC_ADMIN_TOKEN="<admin token>" \
  bash agent_diy/codex_rpc_bridge/test_sync_repo_to_container.sh --remote
  ```

- 用户要求“保活 / 防止开发容器断开”时：

  ```bash
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh --status
  ```

  默认保活不得切换 IDE tab、不得点击恢复按钮、不得向 code-server 终端输入内容。
  默认路径只允许后台 health 探测、`GetWebIDE` 候选接口探测，以及不聚焦的
  synthetic mousemove 活动事件。只有用户明确要求页面恢复点击时，才加
  `--interactive-recovery`；只有用户明确要求终端输入保活时，才加
  `--terminal-keepalive`。

  如果用户反馈默认后台探测不能保活，优先建议低干扰点击模式：

  ```bash
  bash agent_diy/codex_rpc_bridge/start_keepalive.sh --safe-click-keepalive --interval 480
  ```

  该模式使用 CDP 点击 IDE 页面左上角，不移动系统鼠标、不向终端输入，但可能让
  浏览器页面获得键盘焦点。只有 safe-click 也失败时，再考虑 `--terminal-keepalive`。

- 如果 `--apply` 或远程测试失败并显示 `ECONNREFUSED`，要求用户在容器内运行
  `bash agent_diy/codex_rpc_bridge/start_rpc.sh`。
- 如果返回 `WEBIDE_RECORD_NOT_FOUND`，要求用户重新打开或重启 Tencent Arena
  开发容器，并确认当前 `IDE_ID`。
- token、浏览器登录态、`.codex_rpc/` 和 `agent_diy/codex_rpc_bridge_runtime/`
  都不能提交。

## 最常用命令

容器内启动 RPC：

```bash
bash /data/projects/legged_robot_competition_26/agent_diy/codex_rpc_bridge/start_rpc.sh
```

如果已经在项目根目录：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

成功时会看到：

```json
{
  "ok": true,
  "version": "0.7.1",
  "root": "/data/projects/legged_robot_competition_26",
  "auth": true,
  "admin_auth": true
}
```

外部代理地址格式：

```text
https://tencentarena.com/p5/ide/<IDE_ID>/proxy/8765
```

示例：

```text
https://tencentarena.com/p5/ide/11428/proxy/8765
```

## 开发容器日常恢复

打开开发容器入口：

```text
https://tencentarena.com/p/common/competition/ide/447/11585/11428
```

进入后应看到 code-server 和项目 `legged_robot_competition_26`。如果页面异常：

- 点击 `重 启` 重启开发容器。
- 点击 `立即重新连接` 恢复 code-server 连接。
- 点击 `重新加载窗口` 刷新 IDE 前端。

容器重启后需要重新启动 RPC：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```
