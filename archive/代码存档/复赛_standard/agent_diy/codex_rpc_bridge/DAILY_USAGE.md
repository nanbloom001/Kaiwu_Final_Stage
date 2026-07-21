# 日常 AI 操作流程

本文档说明首次安装完成后，AI 每次正常使用 RPC bridge 的流程。

## 0. 代码修改约定

默认工作流是：先在本地仓库修改、检查、记录变更，最后再一次性同步到
腾讯开悟开发容器。

容器主要用于运行和验证，不作为默认编辑源。除非用户明确要求做
container-only 实验，否则不要直接在容器里做探索性改动。需要进容器测试的
补丁，应先落到本地仓库，再通过 RPC bridge 批量同步目标文件，并在写完后读回
hash 或执行针对性检查确认同步成功。

开发容器持久化规则：

- 容器重启后，只有这些顶层目录下的文件预期会持久化：
  `agent_diy`、`agent_ppo`、`conf`、`log`。
- 需要长期保留的本地代码、脚本、说明文档，优先放在这些目录下。
- 其他顶层目录可以用于临时上传、临时测试和一次性输出，但不要依赖它们在容器
  重启后仍然存在。

推荐使用本地侧一键同步脚本：

```bash
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dest /workspace/code --dry-run
CODEX_RPC_TOKEN="<normal token>" \
CODEX_RPC_ADMIN_TOKEN="<admin token>" \
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dest /workspace/code --apply --py-compile
```

该脚本默认只同步 Git 认为应纳入工作区的文件，跳过 `.git`、RPC runtime、
token、日志、模型权重、训练产物、zip/HAR、`node_modules` 等高风险路径。
默认只覆盖/新增，不删除容器端多余文件。

### Symlink 写入失败说明

腾讯开悟容器里常见两套路径：

- RPC root：`/data/projects/legged_robot_competition_26`
- 平台代码工作区：`/workspace/code`

RPC health 返回的 root 通常是 `/data/projects/legged_robot_competition_26`，但该路径下的
顶层业务目录可能是指向 `/workspace/code/...` 的 symlink 或平台映射目录。同步脚本的
容器端 apply 会逐级检查路径，拒绝写穿 symlink 目录或覆盖 symlink 文件。因此如果把
目标设成默认 `.`，可能出现类似错误：

```text
refusing to write through symlink directory: agent_ppo/__init__.py
refusing to write through symlink directory: agent_diy/codex_rpc_bridge/codex_file_rpc.py
```

这不是代码文件损坏，而是安全检查防止归档 apply 通过 symlink 写到未预期位置。日常同步
业务代码必须显式使用：

```bash
--dest /workspace/code
```

不要用默认 `--dest .` 同步业务代码。若只做 RPC runtime 自身维护，另行使用明确的
小范围 RPC 写入或容器内手动更新，不要把 `agent_diy` 放进常规业务代码同步包。

### Runtime 目录清理

RPC 同步会在容器端临时使用：

```text
/data/projects/legged_robot_competition_26/agent_diy/codex_rpc_bridge_runtime/uploads
/data/projects/legged_robot_competition_26/agent_diy/codex_rpc_bridge_runtime/backups
```

这些目录不能进入平台最终代码包。同步脚本的规则是：

- `uploads`：成功 apply 后立即删除上传目录，容器端保留数量为 0。
- `backups/sync`：只保留最近 1 次一键同步备份目录。
- `backups/rpc`：只保留最近 1 次直接 RPC 写文件备份目录。
- `codex_file_rpc.log`：默认限制到 64KB。`start_rpc.sh` 启动前会清空超限旧日志，
  RPC 服务运行时也会周期性截断超限日志。需要调整时设置
  `CODEX_RPC_MAX_LOG_BYTES`。
- 本地打包阶段永远不上传 `agent_diy/codex_rpc_bridge_runtime`。

如果同步被中断或 apply 失败，可能留下半次上传目录。下一次成功 apply 会自动清理；
也可以通过 RPC/容器终端手动删除 runtime 下的 `uploads` 旧目录。

开发容器保活可以用本地侧启动脚本管理：

```bash
bash agent_diy/codex_rpc_bridge/start_keepalive.sh
bash agent_diy/codex_rpc_bridge/start_keepalive.sh --status
bash agent_diy/codex_rpc_bridge/start_keepalive.sh --stop
```

默认保活不再检查/恢复 IDE 页面，不会切换 IDE tab、不会点击、不会向终端输入。
它只在当前浏览器上下文里做后台 health 探测、`GetWebIDE` 候选接口探测，以及
不聚焦的 synthetic mousemove 活动事件。需要旧的页面恢复点击时，手动追加
`--interactive-recovery`；需要旧的终端输入保活时，手动追加 `--terminal-keepalive`。

如果后台探测仍无法避免 10 分钟左右被回收，优先尝试低干扰点击保活：

```bash
bash agent_diy/codex_rpc_bridge/start_keepalive.sh --safe-click-keepalive --interval 480
```

该模式使用 CDP 点击 IDE 页面左上角，不移动系统鼠标、不向终端输入，但可能让浏览器
页面获得键盘焦点。它比 `--terminal-keepalive` 干扰小，建议先测试它。

如果 IDE ID 不是默认的 `11428`，启动时显式传入：

```bash
bash agent_diy/codex_rpc_bridge/start_keepalive.sh --ide-id <IDE_ID>
```

## 1. 打开开发容器

日常使用优先复用已经打开的 Tencent Arena IDE 标签页，例如：

```text
https://tencentarena.com/p/common/competition/ide/447/11585/11428
```

只有在浏览器里没有可用 IDE 标签页、页面已失效，或用户明确要求重新打开时，才用
已保存登录态的 agent-browser 会话打开 Tencent Arena IDE：

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

如果当前 agent-browser daemon 已经在运行，`--state` 或 `--headed` 可能会提示被忽略。
不要因此关闭或覆盖已有可用 IDE 标签页。只有在确实需要重新以可见窗口和保存状态
启动时，才先关闭自动化浏览器 daemon：

```bash
AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser close

AGENT_BROWSER_SESSION=tencent-arena \
AGENT_BROWSER_SESSION_NAME=tencent-arena \
agent-browser --headed --state ./tencent-arena-auth.json open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

Windows PowerShell 写法：

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
agent-browser close
agent-browser --headed --state .\tencent-arena-auth.json open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

进入后应看到 code-server 开发容器。资源管理器里应能看到：

```text
legged_robot_competition_26
  agent_diy
  agent_ppo
  conf
  log
  kaiwu.json
  train_test.py
```

如果页面提示超时、异常或连接断开：

- 看到顶层 `重 启` 时，点击它并等待容器恢复。
- 看到 IDE 内 `立即重新连接` 时，优先尝试重新连接。
- 看到 IDE 内 `重新加载窗口` 时，可重新加载 code-server 前端。

重启开发容器后，RPC 进程通常会消失，需要重新运行启动脚本。

## 2. 启动或恢复 RPC 服务

在开发容器 code-server 中打开集成终端，执行：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

该脚本会复用或生成 token，停止旧 RPC 进程，然后后台启动新进程。

## 3. 确认基础信息

用户或 AI 需要确认：

- `BASE`：Tencent Arena 代理地址，例如：

  ```text
  https://tencentarena.com/p5/ide/11428/proxy/8765
  ```

- normal token：读取和普通工作区写入使用。
- admin token：仅在需要执行命令或全局写入时使用。

不要把 token 写入仓库、日志或截图。

## 4. 确认浏览器登录态和 RPC 状态

严格区分“首次初始化验证”和“日常使用”：

- 首次初始化验证：仅当本机/浏览器会话还没有确认过 `IDE_ID`、Tencent
  Arena 登录态和 RPC 代理地址是否正确时，才可以新开一个浏览器标签页访问
  health。
- 日常使用：如果已经有开发容器 IDE 标签页，例如
  `https://tencentarena.com/p/common/competition/ide/447/11585/11428`，并且本轮
  RPC health 已通过脚本或 fetch 验证，就不要再打开新的 health 标签页。后续
  health、read、write、exec 都应直接走已有 IDE/session 的 RPC/fetch。

首次初始化验证时，新标签页地址为：

```text
${BASE}/api/health
```

必须使用新标签页，避免覆盖当前 code-server IDE 页面。这个页面只用于确认
Tencent Arena 登录态、`IDE_ID` 和 RPC 服务都可用；不是日常操作入口。
同一轮后续操作不要反复新开 `${BASE}/api/health` 页面，应直接在已登录浏览器
上下文里通过 RPC/fetch 获取 health 信息，例如：

```js
await fetch(`${BASE}/api/health`).then(r => r.json())
```

成功返回：

```json
{
  "ok": true,
  "version": "0.7.1",
  "root": "/data/projects/legged_robot_competition_26",
  "auth": true,
  "admin_auth": true
}
```

如果返回：

```json
{"code":1040,"msg":"token 校验失败","reason":"CommonErrors.TOKEN_NOT_VALID"}
```

说明请求被 Tencent Arena 代理层拦截。处理方式：

- 重新用 `agent-browser` 工具、Chrome DevTools MCP、Playwright 或普通 Chrome 登录 Tencent Arena。
- 确认 `IDE_ID` 没变。
- 确认访问的是 `/p5/ide/<IDE_ID>/proxy/8765`。

如果返回：

```text
connect ECONNREFUSED 0.0.0.0:8765
```

说明浏览器登录态已经通过，但容器 RPC 没启动。让用户在容器中运行：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

## 5. 校验 normal token

通过浏览器页面上下文执行：

```js
await fetch(`${BASE}/api/token-check`, {
  headers: { "X-Codex-Token": TOKEN }
}).then(r => r.json())
```

成功返回：

```json
{"ok": true, "role": "normal"}
```

## 6. 读取文件

```js
const file = await fetch(`${BASE}/api/read/agent_ppo/feature/reward_process.py`, {
  headers: { "X-Codex-Token": TOKEN }
}).then(r => r.json());

file.sha256;
file.text;
```

读文件结果会包含：

- `path`
- `size`
- `sha256`
- `text`

`sha256` 必须保存，用于后续写回的并发保护。

## 7. 写文件

写文件必须使用流程：

```text
read -> 保存 sha256 -> write_b64(expected_sha256=旧 sha) -> read back 验证
```

JavaScript 侧 base64url 编码函数：

```js
function b64url(text) {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}
```

写回：

```js
const newText = "new content\n";
const data = b64url(newText);

await fetch(`${BASE}/api/write_b64/path/to/file.py?expected_sha256=${oldSha}&data=${data}`, {
  headers: { "X-Codex-Token": TOKEN }
}).then(r => r.json());
```

写完必须读回：

```js
await fetch(`${BASE}/api/read/path/to/file.py`, {
  headers: { "X-Codex-Token": TOKEN }
}).then(r => r.json());
```

如果返回：

```json
{"ok": false, "error": "file changed since read: ..."}
```

说明文件被别人改过。必须重新读取、合并，再写回。

## 8. 执行检查命令

执行命令需要 admin token。

命令需要 base64url 编码。例如：

```bash
python3 - <<'PY'
import base64
cmd = "python3 -m py_compile train_test.py"
print(base64.urlsafe_b64encode(cmd.encode()).decode().rstrip("="))
PY
```

然后调用：

```js
await fetch(`${BASE}/api/exec_b64?cwd=.&timeout=30&cmd=${CMD_B64}`, {
  headers: { "X-Codex-Admin-Token": ADMIN_TOKEN }
}).then(r => r.json())
```

只在确实需要时使用 admin token。普通文件读写不需要 admin token。

## 8.1 一键同步本地仓库到容器

同步前先 dry-run：

```bash
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dry-run
```

确认文件数量和样例文件合理后再执行：

```bash
CODEX_RPC_TOKEN="<normal token>" \
CODEX_RPC_ADMIN_TOKEN="<admin token>" \
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --apply --py-compile
```

脚本行为：

- 以本地仓库为权威源，打包当前 Git 工作区文件。
- 通过已登录 `agent-browser` 浏览器上下文访问 `${BASE}`。
- 分块上传到容器 `agent_diy/codex_rpc_bridge_runtime/uploads/<sync_id>/`。
- 容器端解包到 staging，校验 archive 和逐文件 sha256。
- 覆盖前自动备份旧文件到 `agent_diy/codex_rpc_bridge_runtime/backups/`。
- 写完后逐文件校验；`--py-compile` 会额外检查核心 PPO Python 文件。

默认安全策略：

- 不同步 ignored 文件。
- 不删除容器端多余文件。
- 长期保留的同步目标应放在 `agent_diy`、`agent_ppo`、`conf`、`log` 之下；
  其他目标目录仅适合作为临时测试目录。
- 容器端目标路径可以解析到 RPC root
  `/data/projects/legged_robot_competition_26` 或代码工作区 `/workspace/code`。
  这是为了兼容平台把 `agent_ppo` 等持久目录映射到 `/workspace/code/...`
  的情况；脚本仍会拒绝写穿内部 symlink 目录或覆盖 symlink 文件。
- 拒绝 `--delete-extra` 镜像删除。
- 跳过 `.git`、`.env`、token、runtime、日志、模型权重、训练产物、zip/HAR、
  `node_modules`、`__pycache__` 等路径。

常用参数：

```bash
--base https://tencentarena.com/p5/ide/11428/proxy/8765
--session tencent-arena
--source .
--dest .
--chunk-size 4096
```

同步脚本测试：

```bash
bash agent_diy/codex_rpc_bridge/test_sync_repo_to_container.sh --local

CODEX_RPC_TOKEN="<normal token>" \
CODEX_RPC_ADMIN_TOKEN="<admin token>" \
bash agent_diy/codex_rpc_bridge/test_sync_repo_to_container.sh --remote
```

`--local` 只做本地 dry-run、安全排除和边界参数测试，不写容器。`--remote`
只写容器临时目录 `tmp_sync_repo_to_container_tests/<timestamp>/`，用于验证上传、
覆盖备份、二进制完整性和远端保护逻辑。若 health 返回
`WEBIDE_RECORD_NOT_FOUND`，需要重新打开/重启开发容器并重新启动 RPC。

## 9. 常用接口

```text
GET /api/health
GET /api/token-check
GET /api/list
GET /api/list/<dir>
GET /api/tree?depth=N
GET /api/tree/<dir>?depth=N
GET /api/stat/<path>
GET /api/read/<file>
GET /api/read/<file>?encoding=base64
GET /api/write_b64/<file>?expected_sha256=<sha>&data=<base64url>
GET /api/append_b64/<file>?expected_sha256=<sha>&data=<base64url>
GET /api/mkdir_get/<dir>
GET /api/touch/<file>
GET /api/delete_get/<file-or-empty-dir>
GET /api/exec_b64?cwd=<dir>&timeout=<seconds>&cmd=<base64url-command>
POST /api/stop
```

说明：

- `exec_b64` 和 `stop` 需要 admin token。
- 删除接口只删除文件或空目录，不递归删除非空目录。
- 覆盖、追加、删除已有文件前会自动备份到：

  ```text
  agent_diy/codex_rpc_bridge_runtime/backups/
  ```

## 10. AI 操作守则

- 先读再写。
- 写入必须带 `expected_sha256`。
- 写完必须读回验证。
- 不并发写同一个文件。
- 普通任务只用 normal token。
- 只有执行命令时才请求 admin token。
- 不把 token 写入仓库、日志、截图或最终回答。
- 遇到代理层 `TOKEN_NOT_VALID`，优先检查浏览器登录态，而不是 RPC token。
- 遇到 `ECONNREFUSED`，优先让用户重启容器内 RPC 服务。
- 不要把开发容器 IDE 和训练监控界面混淆。日常代码操作只需要进入
  code-server 开发容器并启动 RPC。
- 保存浏览器登录态前必须得到用户明确授权；保存的 `tencent-arena-auth.json`
  不要提交或发送。
