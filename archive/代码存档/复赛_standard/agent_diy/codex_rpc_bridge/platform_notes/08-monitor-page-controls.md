# Monitor Page Controls

Last verified: 2026-05-13

## Goal

Record the task-specific controls on the monitor page after a training task has
started, including the real monitor URL shape and the auto-refresh selector.

## Verified task-specific monitor URL

For the verified example task `codex test`, clicking `查看监控` opened:

```text
https://tencentarena.com/p/v5/exp/monitor?domain_id=447&exp_id=11428&task_uuid=088597df-54ce-46b2-98ba-5c01e6a2dcf2&team_id=11585&task_id=156140&platform=competition_stage
```

## Stable and changing URL parts

The following query parameters were observed:

- `domain_id=447`
- `exp_id=11428`
- `team_id=11585`
- `platform=competition_stage`
- `task_id=156140`
- `task_uuid=088597df-54ce-46b2-98ba-5c01e6a2dcf2`

Practical interpretation:

- `domain_id` is stable for the competition domain
- `exp_id` is stable for the current project / experiment
- `team_id` is stable for the current team
- `platform` is stable
- `task_id` changes for each new task
- `task_uuid` also changes for each new task instance

So the monitor page should be treated as task-instance-specific.

## Verified monitor-page controls

On the running task's monitor page, the following controls were visible:

- `模型列表`
- `释放`
- `监控总览`
- `训练日志`
- a time-range selector
- an auto-refresh selector
- a refresh button

## Auto-refresh selector

The monitor page for `codex test` was first observed with:

```text
禁用自动刷新
```

It was then successfully changed to:

```text
每 5 秒自动刷新
```

## Verified auto-refresh options

The selector was observed to include all of the following options:

- `禁用自动刷新`
- `每 5 秒自动刷新`
- `每 10 秒自动刷新`
- `每 30 秒自动刷新`
- `每 1 分钟自动刷新`
- `每 5 分钟自动刷新`
- `每 15 分钟自动刷新`
- `每 30 分钟自动刷新`
- `每 1 小时自动刷新`

## Practical recommendation

For active debugging or close observation, use:

```text
每 5 秒自动刷新
```

This was explicitly verified as selectable on the real page.

## Important difference between the two tabs

The two monitor tabs should not be treated as having the same refresh behavior.

### `训练日志`

- log content can continue updating through the page's own refresh behavior
- this tab does not require the same viewport sweep as the overview cards

### `监控总览`

- auto-refresh alone is not enough for all cards
- some overview cards only load or refresh after the page has been scrolled to
  the corresponding card region
- a collector should actively scroll through the overview page if it wants a
  more complete metric refresh

## Relationship to training-page controls

Do not confuse the two layers:

1. Training list page task-card controls:
   - `查看监控`
   - `模型列表`
   - `下载代码`
   - `释放`

2. Task-specific monitor page controls:
   - `模型列表`
   - `释放`
   - auto-refresh selector
   - refresh button

The training list page is where a running task is discovered.
The monitor page is where ongoing observation is controlled.
