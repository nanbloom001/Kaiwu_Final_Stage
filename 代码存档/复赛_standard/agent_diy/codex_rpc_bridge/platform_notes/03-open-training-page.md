# Open Training Page

Last verified: 2026-05-13

## Goal

Reach the Tencent Arena training-task list page and verify whether resources are
available before attempting to create a new task.

## Verified URL

```text
https://tencentarena.com/p/competition/stage/447/11585/11428/training
```

## Resource precondition

Before creating a new training task, check that the page shows available
resources.

Verified example:

```text
CPU资源: 6核可用
GPU资源: 1卡可用
```

If no GPU is available, the create-task flow may be blocked or not useful.

## Observed page actions

From this page, the following actions are visible:

- `新增训练任务`
- `查看监控`
- `模型列表`
- `下载代码`
- `进入开发`

Important note:

- `进入开发` is visible here, but it should not be used as the primary way to
  discover the actual IDE URL
- use the direct IDE URL from `01-open-dev-container.md` instead
