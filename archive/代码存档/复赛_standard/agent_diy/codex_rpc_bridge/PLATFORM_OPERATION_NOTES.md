# Tencent Arena Platform Operation Notes

This file is a temporary, additive note file.

It records the platform operation flow that has been verified in practice.
It should be merged into the main operation manuals later.

Last verified: 2026-05-13

## Scope

These notes cover two parts:

1. How to correctly open and verify the online development container
2. How to correctly inspect the "新增训练任务" flow on the training page

This file does not replace:

- `SETUP.md`
- `AI_USAGE.md`
- `DAILY_USAGE.md`

It is only a temporary verified-operation record.

## 1. Online Development Container

### 1.1 Correct entry

Do not rely on the training page's `进入开发` button to discover the URL.

That button is a frontend button and does not expose a stable static link in the
DOM snapshot.

Use the verified direct IDE entry URL:

```text
https://tencentarena.com/p/common/competition/ide/447/11585/11428
```

### 1.2 Correct way to open it

Use the existing logged-in `agent-browser` session:

```powershell
$env:AGENT_BROWSER_SESSION = "tencent-arena"
$env:AGENT_BROWSER_SESSION_NAME = "tencent-arena"
agent-browser open "https://tencentarena.com/p/common/competition/ide/447/11585/11428"
```

### 1.3 Successful-entry signals

After opening the IDE URL, success is defined by all of the following:

1. The browser URL remains:

   ```text
   https://tencentarena.com/p/common/competition/ide/447/11585/11428
   ```

2. The page title becomes:

   ```text
   代码开发 - 腾讯开悟比赛平台
   ```

3. The embedded `code-server` UI is visible

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

### 1.4 Recovery behavior

If the IDE page is abnormal, timed out, or disconnected, the verified recovery
buttons to look for are:

- `重启`
- `立即重新连接`
- `重新加载窗口`

Preferred order:

1. `立即重新连接`
2. `重新加载窗口`
3. `重启`

If the container has already been reclaimed or is badly broken, `重启` is
acceptable.

### 1.5 RPC service startup inside the container

Once `code-server` is available, start the RPC service in the container terminal:

```bash
bash agent_diy/codex_rpc_bridge/start_rpc.sh
```

### 1.6 RPC health check

Verified health check URL:

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

If this returns successfully, it proves:

- the development container is alive
- the browser session is authenticated
- the RPC service is alive

## 2. Keepalive Script

Verified keepalive script path:

```text
agent_diy/codex_rpc_bridge/keepalive_tencent_arena.ps1
```

### 2.1 Current verified behavior

The script:

- reuses session `tencent-arena`
- ensures the browser is on the IDE URL
- checks for recovery buttons
- periodically probes the RPC health endpoint

Current parameters in the script:

- IDE URL:

  ```text
  https://tencentarena.com/p/common/competition/ide/447/11585/11428
  ```

- health URL:

  ```text
  https://tencentarena.com/p5/ide/11428/proxy/8765/api/health
  ```

- default interval:

  ```text
  180 seconds
  ```

### 2.2 What counts as "effective"

The keepalive script should be considered effective only when both are true:

1. The keepalive PowerShell process is still running
2. The RPC health check still returns `ok: true`

Merely having a browser window open is not enough.

## 3. Training Page

Verified training page URL:

```text
https://tencentarena.com/p/competition/stage/447/11585/11428/training
```

### 3.1 Resource precondition

Before creating a new training task, confirm that the page shows available
resources. A verified example was:

```text
CPU资源: 6核可用
GPU资源: 1卡可用
```

If no GPU is available, the create flow may be blocked or meaningless.

## 4. "新增训练任务" Flow

### 4.1 Important UI behavior

Clicking `新增训练任务` does not open a new page.

It opens a right-side drawer on the same training page.

The drawer is an Ant Design drawer, not a separate navigation target.

### 4.2 Verified visible fields in the drawer

The following fields were observed in the drawer:

- `任务名称`
- `使用预训练模型`
- `选择预训练模型`
- `任务配置`
- `算法`
- `选择算法`
- `训练模式`
- `单机`
- `任务时长 ( 范围 1min - 168h )`
- `同时启动环境数`
- `预估本次消耗`
- `预估团队剩余`
- `描述（选填）`
- `完成新增`
- `取消`

### 4.3 Verified default/visible state

The following defaults or immediately visible values were observed:

- `使用预训练模型`: off by default
- `训练模式`: `单机`
- `任务时长`: hour and minute numeric inputs exist
- `同时启动环境数`: numeric input exists and may be disabled until other
  required fields are set
- cost preview is shown in the drawer

Verified example visible in the drawer:

```text
预估本次消耗
CPU 6.00 核, GPU 1.00 卡

预估团队剩余
CPU 0.00 核, GPU 0.00 卡
```

### 4.4 Important interpretation

The drawer can be present even if a plain page-text scrape does not clearly show
it as a separate modal.

If needed, detect it with DOM inspection such as:

- `.ant-drawer-open`
- `.ant-drawer-content`

Do not assume "click failed" just because the main page text still looks similar.

### 4.5 Verified real task-creation example

One full end-to-end task creation was verified successfully.

Submitted values:

- task name: `codex test`
- pretrained-model mode: enabled
- pretrained model: `nan10_8750`
- duration: `1h 0min`
- description: short free-text test description

Observed result after submission:

- a new task card `codex test` appeared at the top of the list
- resources dropped to `CPU资源: 0核可用 / GPU资源: 0卡可用`
- the task later entered `进行中`
- task-card action buttons became visible:
  - `查看监控`
  - `模型列表`
  - `下载代码`
  - `释放`

For a step-by-step replay, see:

- `platform_notes/07-create-training-task-instance.md`

## 5. Monitor Page Notes

Verified monitor page URL pattern:

```text
https://tencentarena.com/p/v5/exp/monitor?domain_id=447&exp_id=11428&task_uuid=<...>&team_id=11585&task_id=<...>&platform=competition_stage
```

### 5.1 Verified behavior for metrics

The monitor page can expose historical metric curves through:

```text
GetTrainMetricRange
```

### 5.2 Verified behavior for logs

The training log page can expose historical logs through:

```text
GetTrainLog
```

### 5.3 Important UI caveats

- Some monitor groups are collapsible
- Some groups only load or reveal data when expanded
- Some metric cards only issue backend requests after the relevant section is
  in view
- `训练日志` can keep updating through refresh behavior without the same
  viewport dependency
- `监控总览` is stricter: some overview cards must be scrolled into view before
  their data loads or refreshes

This matters for automated scraping.

### 5.4 Verified task-instance monitor URL

For the verified `codex test` example, clicking `查看监控` opened:

```text
https://tencentarena.com/p/v5/exp/monitor?domain_id=447&exp_id=11428&task_uuid=088597df-54ce-46b2-98ba-5c01e6a2dcf2&team_id=11585&task_id=156140&platform=competition_stage
```

This confirms:

- `task_uuid` changes per task instance
- `task_id` also changes per task

The task-specific controls and auto-refresh selector are documented in:

- `platform_notes/08-monitor-page-controls.md`

## 6. Collapsible Monitor Groups

Verified examples:

- `地形-斜坡( 8 )`
- `地形-倒斜坡( 8 )`

These are clickable fold/unfold buttons.

They were observed with:

- `expanded=false` before clicking
- `expanded=true` after clicking

After expansion, the page text revealed the missing metric names.

### 6.1 Verified slope-group metric names

For `地形-斜坡( 8 )`:

- `斜坡-完成数`
- `斜坡-失败数`
- `斜坡-超时数`
- `斜坡-总分`
- `斜坡-时间分数`
- `斜坡-姿态分数`
- `斜坡-能耗分数`
- `斜坡-步数`

For `地形-倒斜坡( 8 )`:

- `倒斜坡-完成数`
- `倒斜坡-失败数`
- `倒斜坡-超时数`
- `倒斜坡-总分`
- `倒斜坡-时间分数`
- `倒斜坡-姿态分数`
- `倒斜坡-能耗分数`
- `倒斜坡-步数`

## 7. Practical Rules

These are the currently verified operation rules:

1. For IDE access, use the direct IDE URL, not the training-page button to
   discover the link
2. For container availability, judge by RPC health, not just by visible page
3. For new-task creation, expect a right-side drawer, not a new page
4. For monitor scraping, expand collapsible groups before concluding that
   metrics are missing
5. For backend data extraction, prefer network responses over screenshots

## 8. Follow-up Merge Target

This file should later be merged into:

- `AI_USAGE.md`
- `DAILY_USAGE.md`
- `SETUP.md`

Suggested merge split:

- IDE direct-open and health-check flow -> `AI_USAGE.md` and `DAILY_USAGE.md`
- keepalive operational criteria -> `SETUP.md` and `AI_USAGE.md`
- training-page drawer behavior -> `DAILY_USAGE.md`
- monitor collapsible-group notes -> `SETUP.md` or a monitor-specific note
