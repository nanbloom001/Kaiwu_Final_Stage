# Open Monitor Page

Last verified: 2026-05-13

## Goal

Record the monitor-page entry pattern and the backend interfaces that are useful
for data extraction.

## Verified URL pattern

```text
https://tencentarena.com/p/v5/exp/monitor?domain_id=447&exp_id=11428&task_uuid=<...>&team_id=11585&task_id=<...>&platform=competition_stage
```

## URL interpretation

This URL is only partially stable.

Stable parts for the current project were observed to include:

- `domain_id=447`
- `exp_id=11428`
- `team_id=11585`
- `platform=competition_stage`

Task-instance-specific parts:

- `task_id`
- `task_uuid`

Another AI should not hardcode an old `task_uuid` and assume it will keep
working for a newly created task.

The right procedure is:

1. create or locate the target task in the training list
2. click that task's `查看监控`
3. record the resulting actual URL for that task instance

## Verified backend APIs

The monitor page exposes useful historical data through:

- `GetTrainMetricRange`
- `GetTrainLog`

## Practical meaning

- `GetTrainMetricRange` is the right source for historical metric curves
- `GetTrainLog` is the right source for training log history
- screenshots are secondary; backend responses are preferred

## Important UI caveats

- some monitor groups are collapsible
- some groups only reveal their metric names after expansion
- some metric cards only trigger requests after the section is visible
- `训练日志` can keep updating through the page's own refresh behavior
- `监控总览` is different: some overview cards only refresh or load after the
  page has been scrolled to the corresponding card region

For automation, this means:

1. expand collapsed groups first
2. scroll relevant overview sections into view
3. then inspect the request log
