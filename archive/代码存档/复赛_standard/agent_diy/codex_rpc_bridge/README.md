# Codex RPC Bridge

轻量 HTTP RPC 服务，用于让外部 AI agent 操作腾讯开悟 / code-server 开发容器中的代码。

## 快速入口

- [SETUP.md](./SETUP.md)：第一次安装、登录、部署到容器、启动和测试。
- [DAILY_USAGE.md](./DAILY_USAGE.md)：安装完成后的日常 AI 操作流程。
- [AI_USAGE.md](./AI_USAGE.md)：给其他 AI agent 的主动调用规则、简短索引和最常用命令。

## 推荐使用方式

外部 AI 访问腾讯开悟 `/proxy/<PORT>` 时，需要复用已登录 Tencent Arena
的浏览器上下文。推荐先安装并使用 `agent-browser`：

```bash
npm i -g agent-browser
agent-browser install
agent-browser --version
```

Windows PowerShell 下如果 `npm` 因执行策略报错，改用 `npm.cmd`：

```powershell
npm.cmd i -g agent-browser
```

如果 `agent-browser` 安装后命令不可见，把 npm 全局命令目录加入用户
`PATH`，常见路径是：

```text
C:\Users\<user>\AppData\Roaming\npm
```

## 容器内启动

```bash
bash /data/projects/legged_robot_competition_26/agent_diy/codex_rpc_bridge/start_rpc.sh
```

项目根目录下也可以运行：

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

本机侧保活脚本：

```bash
bash agent_diy/codex_rpc_bridge/start_keepalive.sh
bash agent_diy/codex_rpc_bridge/start_keepalive.sh --status
bash agent_diy/codex_rpc_bridge/start_keepalive.sh --stop
```

默认保活只在当前浏览器上下文里做后台探测：RPC health 请求、`GetWebIDE`
候选接口探测，以及不点击/不聚焦的 synthetic mousemove 活动事件。默认不会
切换 IDE tab、不会点击恢复按钮、不会向 code-server 终端输入内容。需要旧的
页面恢复点击时才传 `--interactive-recovery`；需要旧的终端输入保活时才传
`--terminal-keepalive`。

如果默认后台探测不足以防止平台回收，可以启用低干扰点击保活：

```bash
bash agent_diy/codex_rpc_bridge/start_keepalive.sh --safe-click-keepalive --interval 480
```

`--safe-click-keepalive` 使用浏览器 CDP 在 IDE 页面左上角发送一次点击，不移动系统
鼠标，也不向终端输入；但它可能让浏览器页面获得键盘焦点。若仍然不够，再考虑
`--terminal-keepalive` 这个强保活兜底。

平台入口脚本：

```text
Mac/start_keepalive.sh
Mac/sync_repo_to_container.sh
win/start_keepalive.ps1
win/sync_repo_to_container.ps1
```

本地仓库一键安全同步到容器：

```bash
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --dry-run
CODEX_RPC_TOKEN="<normal token>" \
CODEX_RPC_ADMIN_TOKEN="<admin token>" \
bash agent_diy/codex_rpc_bridge/sync_repo_to_container.sh --apply --py-compile
```

同步脚本允许容器端目标解析到 RPC root
`/data/projects/legged_robot_competition_26` 或代码工作区 `/workspace/code`，以兼容
平台将 `agent_ppo` 等目录映射到 `/workspace/code/...` 的情况。脚本仍会拒绝
写穿内部 symlink 目录或覆盖 symlink 文件。

运行时文件会写入：

```text
agent_diy/codex_rpc_bridge_runtime/
```

不要提交运行时目录、token、日志、备份或浏览器登录态文件。

## 常见状态

- 浏览器页面显示开发容器 code-server，资源管理器里能看到
  `legged_robot_competition_26`、`agent_diy`、`agent_ppo`，说明已经进入开发容器。
- 页面出现异常或超时弹窗时，优先使用页面上的 `重 启`、`立即重新连接`
  或 `重新加载窗口` 恢复开发容器。
- RPC 健康检查地址示例：

  ```text
  https://tencentarena.com/p5/ide/11428/proxy/8765/api/health
  ```

  这个地址只用于首次初始化验证 `IDE_ID`、登录态和 RPC 代理是否正确。日常已有
  Tencent Arena IDE 标签页时，不要反复打开 health 页面；应复用已有
  IDE/session，通过同步脚本或 RPC/fetch 直接验证和操作。
