# Platform Task Notes

This directory temporarily splits Tencent Arena platform operation knowledge
into task-oriented files.

These files are intermediate notes only.

They should later be merged back into:

- `AI_USAGE.md`
- `DAILY_USAGE.md`
- `SETUP.md`

Last verified: 2026-05-13

## Files

- `01-open-dev-container.md`
  - How to directly open and verify the online development container
- `02-keepalive-and-health.md`
  - How the keepalive script works and how to judge whether it is effective
- `03-open-training-page.md`
  - Training page URL and preconditions before creating a task
- `04-create-training-task-drawer.md`
  - Verified behavior of the `新增训练任务` drawer and its fields
- `07-create-training-task-instance.md`
  - End-to-end example of creating a real training task from a pretrained model
- `05-open-monitor-page.md`
  - Monitor page entry pattern and backend APIs used by the frontend
- `06-expand-monitor-groups.md`
  - Verified collapsible monitor groups and the slope / inverse-slope metrics
- `08-monitor-page-controls.md`
  - Runtime controls on the task-specific monitor page, including auto-refresh

## Use

Treat each file as one self-contained task note.

When later merging into the main manuals:

- developer-container notes belong mostly in `AI_USAGE.md` and `DAILY_USAGE.md`
- keepalive and health-check notes belong in `SETUP.md` and `AI_USAGE.md`
- training-task and monitor notes belong mostly in `DAILY_USAGE.md`

## Suggested read order for a fresh AI

If the goal is to reproduce the full Tencent Arena workflow, read in this
order:

1. `01-open-dev-container.md`
2. `02-keepalive-and-health.md`
3. `03-open-training-page.md`
4. `04-create-training-task-drawer.md`
5. `07-create-training-task-instance.md`
6. `05-open-monitor-page.md`
7. `08-monitor-page-controls.md`
8. `06-expand-monitor-groups.md`
