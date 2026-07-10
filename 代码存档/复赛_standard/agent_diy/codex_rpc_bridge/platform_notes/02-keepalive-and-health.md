# Keepalive And Health

Last verified: 2026-05-13

## Goal

Judge whether the online development container keepalive chain is actually
working.

## Script path

Verified keepalive script:

```text
agent_diy/codex_rpc_bridge/keepalive_tencent_arena.ps1
```

## Current verified behavior

The script:

- reuses browser session `tencent-arena`
- ensures the browser is on the IDE URL
- checks the page for recovery buttons
- periodically probes the RPC health endpoint

Current important parameters:

- IDE URL:

  ```text
  https://tencentarena.com/p/common/competition/ide/447/11585/11428
  ```

- health URL:

  ```text
  https://tencentarena.com/p5/ide/11428/proxy/8765/api/health
  ```

- interval:

  ```text
  180 seconds
  ```

## What counts as effective

The keepalive script should only be considered effective when both are true:

1. The keepalive PowerShell process is still running
2. The health endpoint still returns `ok: true`

Merely keeping a browser window open is not enough.

## Verified health endpoint

```text
https://tencentarena.com/p5/ide/11428/proxy/8765/api/health
```

Expected success shape:

```json
{
  "ok": true,
  "version": "0.7.1",
  "root": "/data/projects/legged_robot_competition_26",
  "pid": 360,
  "auth": true,
  "admin_auth": true
}
```

If this succeeds, it proves:

- the development container is alive
- the browser session is authenticated
- the RPC service is alive
