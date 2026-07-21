# Open Development Container

Last verified: 2026-05-13

## Goal

Open Tencent Arena's online development container and verify that the session is
really inside the `code-server` workspace.

## Correct entry

Do not rely on the training page's `进入开发` button to discover the URL.

That button is a frontend button and does not expose a stable static link in a
normal DOM snapshot.

Use the verified direct IDE entry URL:

```text
https://tencentarena.com/p/common/competition/ide/447/11585/11428
```

## Correct way to open it

Use the existing logged-in `agent-browser` session:

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
agent-browser open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

## Successful-entry signals

After opening the URL, success is defined by all of the following:

1. Current URL stays:

   ```text
   https://tencentarena.com/p/common/competition/ide/447/11585/11428
   ```

2. The page title becomes:

   ```text
   代码开发 - 腾讯开悟比赛平台
   ```

3. The page shows embedded `code-server`

4. The file explorer shows the project tree, including:

   ```text
   legged_robot_competition_26
     agent_diy
     agent_ppo
     conf
     log
     kaiwu.json
     train_test.py
   ```

## Recovery buttons

If the IDE page is abnormal, timed out, or disconnected, the verified recovery
buttons to look for are:

- `立即重新连接`
- `重新加载窗口`
- `重启`

Preferred order:

1. `立即重新连接`
2. `重新加载窗口`
3. `重启`

If the container has already been reclaimed or the IDE is badly broken, `重启`
is acceptable.

## RPC startup inside the container

Once `code-server` is available, start the RPC service in the integrated
terminal:

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```
