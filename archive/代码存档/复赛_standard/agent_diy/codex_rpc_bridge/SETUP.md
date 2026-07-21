# 首次安装与部署

本文档说明第一次在新的 Codex / 电脑 / 腾讯开悟容器环境中部署 RPC bridge 的步骤。

## 1. 安装浏览器控制能力

腾讯开悟的 `/proxy/<PORT>` 需要浏览器登录态。外部 AI 不能只靠 `curl` 访问代理地址，否则常见返回是：

```json
{"code":1040,"msg":"token 校验失败","reason":"CommonErrors.TOKEN_NOT_VALID"}
```

因此需要让 AI 能控制一个已经登录 Tencent Arena 的浏览器。

### 方案 A：agent-browser 工具

推荐先安装 `agent-browser` 工具。它是一个通用浏览器自动化 CLI，可被 Codex 通过对应 skill 使用，也可以由人或脚本直接调用。

安装前先确认 Node.js 版本。`agent-browser` 和 Chrome DevTools 方案都建议使用较新的 Node：

```bash
node --version
npm --version
```

建议 Node.js 为 `20.19+`。实测 `v22.x` 可用。

安装方式：

```bash
npm i -g agent-browser
agent-browser install
```

安装后确认命令可用：

```bash
agent-browser --version
```

#### Windows / PowerShell 注意事项

Windows PowerShell 可能因为执行策略拒绝运行 `npm.ps1`，典型报错是：

```text
无法加载文件 ...\npm.ps1，因为在此系统上禁止运行脚本
```

这不是 Node 没安装，而是 PowerShell 优先命中了 `.ps1` shim。可以改用：

```powershell
npm.cmd --version
npm.cmd i -g agent-browser
```

如果安装时报 npm cache 或全局目录权限错误，需要在有权限的终端中重新执行安装。

安装后如果 `agent-browser` 命令不可见，检查 npm 全局目录：

```powershell
npm.cmd prefix -g
```

常见目录是：

```text
C:\Users\<user>\AppData\Roaming\npm
```

把该目录加入用户 `PATH` 后，新终端即可直接运行：

```powershell
agent-browser --version
```

在当前终端中也可以临时使用完整路径：

```powershell
& "C:\Users\<user>\AppData\Roaming\npm\agent-browser.cmd" --version
```

然后打开可复用浏览器会话：

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser --headed open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

Windows PowerShell 等价写法：

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
agent-browser --headed open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

如果 `agent-browser` 还没有进入 `PATH`，用完整路径：

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
& "C:\Users\<user>\AppData\Roaming\npm\agent-browser.cmd" --headed open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

如果提示 `--headed ignored`，说明已有无头 daemon 在运行，先关闭再重开：

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser close

AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser --headed open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

登录完成后可保存状态：

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser state save ./tencent-arena-auth.json
```

Windows PowerShell 写法：

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
agent-browser state save .\tencent-arena-auth.json
```

`tencent-arena-auth.json` 是可复用浏览器登录态，等同于敏感认证材料。
只有用户明确授权时才保存。不要提交、截图或发送该文件。仓库 `.gitignore`
应包含：

```text
tencent-arena-auth.json
```

后续可用保存的登录态打开 IDE：

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser --headed --state ./tencent-arena-auth.json open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

Windows PowerShell 写法：

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
agent-browser --headed --state .\tencent-arena-auth.json open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

如果当前 Codex 环境同时安装了 `agent-browser` skill，Codex 应先读取该 skill 的说明，再按上面的命令操作。没有 skill 也不影响工具本身使用，只是需要手动或脚本调用 CLI。

### 方案 B：Chrome DevTools MCP

如果没有安装 `agent-browser` 工具，可以使用 Chrome 官方的 `chrome-devtools-mcp`。

要求：

- Node.js `20.19+`
- npm / npx
- Chrome

Codex MCP 示例配置：

```toml
[mcp_servers.chrome-devtools]
command = "/path/to/node20/bin/npx"
args = [
  "-y",
  "chrome-devtools-mcp@latest",
  "--executablePath",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "--userDataDir",
  "/Users/<user>/.chrome-codex-devtools-mcp",
  "--no-usage-statistics",
  "--no-performance-crux",
]

[mcp_servers.chrome-devtools.env]
PATH = "/path/to/node20/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
```

Windows 可使用同样思路，换成 Windows Chrome 路径和用户数据目录。重点是固定 `userDataDir`，这样登录 cookie 才能复用。

Windows 示例：

```toml
[mcp_servers.chrome-devtools]
command = "C:\\Users\\<user>\\AppData\\Roaming\\nvm\\v20.19.0\\npx.cmd"
args = [
  "-y",
  "chrome-devtools-mcp@latest",
  "--executablePath",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "--userDataDir",
  "C:\\Users\\<user>\\.chrome-codex-devtools-mcp",
  "--no-usage-statistics",
  "--no-performance-crux",
]

[mcp_servers.chrome-devtools.env]
PATH = "C:\\Users\\<user>\\AppData\\Roaming\\nvm\\v20.19.0;C:\\Windows\\System32;C:\\Windows"
```

如果不确定 Node 路径，先在终端运行：

```bash
node --version
npx --version
```

`chrome-devtools-mcp` 要求 Node.js `20.19+`。如果默认 Node 太旧，应把 MCP 配置里的 `command` 和 `PATH` 指向 Node 20+。

## 2. 登录 Tencent Arena

用上一步的浏览器打开 IDE：

```text
https://tencentarena.com/p/common/competition/ide/447/11585/11428
```

用户手动完成登录和验证码。登录成功后，浏览器应能进入 IDE。

进入开发容器后，页面通常是 code-server，资源管理器中应能看到项目：

```text
legged_robot_competition_26
  agent_diy
  agent_ppo
  conf
  log
  kaiwu.json
  train_test.py
```

注意：这里是开发容器 IDE，不是训练监控界面。不要把训练界面路由和开发容器
IDE 混淆。

然后确认代理地址。格式是：

```text
https://tencentarena.com/p5/ide/<IDE_ID>/proxy/8765
```

`IDE_ID` 来自当前 IDE 页面地址，例如：

```text
https://tencentarena.com/p5/ide/11428/?folder=/data/projects/legged_robot_competition_26
```

则 `BASE` 是：

```text
https://tencentarena.com/p5/ide/11428/proxy/8765
```

如果页面显示异常、超时或连接断开，可尝试页面上的按钮：

- `重 启`：重启开发容器，通常需要等待十几秒到数分钟。
- `立即重新连接`：code-server websocket 断开时优先尝试。
- `重新加载窗口`：IDE 前端状态异常时尝试。

重启或重连后，再确认资源管理器里能看到项目代码。

## 3. 在对应位置创建文件

容器重启后，非官方目录可能还原。因此 RPC bridge 应放在官方目录下：

```text
/data/projects/legged_robot_competition_26/agent_diy/codex_rpc_bridge/
```

需要包含：

```text
agent_diy/codex_rpc_bridge/
  AI_USAGE.md
  SETUP.md
  DAILY_USAGE.md
  codex_file_rpc.py
  start_rpc.sh
```

### 自动方式

如果仓库已经包含这些文件，在容器中拉取/复制仓库内容即可。

### 手动方式

如果只能手动创建，至少需要复制：

```text
agent_diy/codex_rpc_bridge/codex_file_rpc.py
agent_diy/codex_rpc_bridge/start_rpc.sh
```

文档文件不是运行必需，但建议一起复制。

完整文件建议包含：

```text
agent_diy/codex_rpc_bridge/
  README.md
  AI_USAGE.md
  SETUP.md
  DAILY_USAGE.md
  codex_file_rpc.py
  start_rpc.sh
```

## 4. 容器内初始化并启动

在容器中运行一行命令即可：

```bash
bash /data/projects/legged_robot_competition_26/agent_diy/codex_rpc_bridge/start_rpc.sh
```

如果已经在项目根目录：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

如果通过 code-server 操作：

1. 进入开发容器 IDE。
2. 打开集成终端：`Ctrl+Shift+\`` 或菜单打开 Terminal。
3. 确认工作区是项目根目录。
4. 执行：

   ```bash
   bash agent_diy/codex_rpc_bridge/start_rpc.sh
   ```

脚本会自动：

- 创建 `agent_diy/codex_rpc_bridge_runtime/`
- 生成或复用 `token`
- 生成或复用 `admin_token`
- 停止旧 RPC 进程
- 后台启动 RPC
- 打印 health 检查

首次启动会在终端打印 normal token 和 admin token。admin token 有命令执行权限，只应发给需要执行检查命令的 AI。

成功输出类似：

```json
{
  "ok": true,
  "version": "0.7.1",
  "root": "/data/projects/legged_robot_competition_26",
  "auth": true,
  "admin_auth": true
}
```

复制命令时，不要复制 shell 提示符，例如 `$`、`bash-5.2#` 或 `root@xxx#`。如果粘贴失败，手打这一行：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

如果用浏览器自动化向 VS Code 终端粘贴命令，优先粘贴单行命令。多行
heredoc 在 xterm 中可能不稳定，容易没有完整提交。

## 5. 测试

### 容器内测试

```bash
curl -s http://127.0.0.1:8765/api/health
```

应返回 `ok: true`。

### 浏览器代理测试

在已登录 Tencent Arena 的浏览器里打开：

```text
https://tencentarena.com/p5/ide/<IDE_ID>/proxy/8765/api/health
```

应返回 `ok: true`。如果返回 `TOKEN_NOT_VALID`，说明浏览器没有登录态或代理地址不对。

如果返回：

```text
connect ECONNREFUSED 0.0.0.0:8765
```

说明 Tencent Arena 登录态和代理路径已经通过，但容器内 RPC 服务没有启动或已被容器重启清掉。
重新执行：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

### AI 侧测试

AI 应通过已登录浏览器上下文访问：

```js
await fetch(`${BASE}/api/token-check`, {
  headers: { "X-Codex-Token": "<normal token>" }
}).then(r => r.json())
```

成功返回：

```json
{"ok": true, "role": "normal"}
```

Admin token 可测试无害命令：

```js
await fetch(`${BASE}/api/exec_b64?cwd=.&timeout=10&cmd=cHdkICYmIHB5dGhvbjMgLS12ZXJzaW9u`, {
  headers: { "X-Codex-Admin-Token": "<admin token>" }
}).then(r => r.json())
```

该命令对应：

```bash
pwd && python3 --version
```

安全提醒：admin token 有执行命令权限，不要写入代码、日志、提交记录或公开消息。

## 6. 已验证的完整流程

一次可复现的完整流程如下：

1. 使用可见 Chrome / agent-browser 打开：

   ```text
   https://tencentarena.com/p/common/competition/ide/447/11585/11428
   ```

2. 如果提示异常或超时，点击页面上的 `重 启`，等待开发容器恢复。
3. 进入 code-server 后，确认资源管理器显示 `legged_robot_competition_26` 项目代码。
4. 打开集成终端并执行：

   ```bash
   bash agent_diy/codex_rpc_bridge/start_rpc.sh
   ```

5. 在已登录浏览器上下文访问：

   ```text
   https://tencentarena.com/p5/ide/11428/proxy/8765/api/health
   ```

6. 成功响应类似：

   ```json
   {
     "ok": true,
     "version": "0.7.1",
     "root": "/data/projects/legged_robot_competition_26",
     "auth": true,
     "admin_auth": true
   }
   ```

## 7. 最终验收清单

另一个 AI 或操作者完成配置后，应能逐项确认：

- `node --version` 可用，且 Node.js 版本满足 `20.19+`。
- `agent-browser --version` 可用。
- `agent-browser install` 已完成，Chrome/Chromium 运行依赖已安装。
- 能用可见 Chrome 打开 Tencent Arena IDE 入口。
- 已由用户手动完成 Tencent Arena 登录和验证码。
- 如需复用登录态，已在用户明确授权后保存 `tencent-arena-auth.json`，且该文件被 `.gitignore` 忽略。
- 浏览器进入的是 code-server 开发容器，资源管理器能看到 `legged_robot_competition_26` 项目。
- 异常/超时时知道使用 `重 启`、`立即重新连接` 或 `重新加载窗口` 恢复。
- 容器终端中已执行：

  ```bash
  bash agent_diy/codex_rpc_bridge/start_rpc.sh
  ```

- 浏览器代理 health 返回 `ok: true`：

  ```text
  https://tencentarena.com/p5/ide/<IDE_ID>/proxy/8765/api/health
  ```

- 未授权访问受保护接口会返回 `401 unauthorized`。
- normal token 和 admin token 没有写入仓库、日志、截图或最终回答。
