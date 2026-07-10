# Create A Real Training Task

Last verified: 2026-05-13

## Goal

Create one real training task from the Tencent Arena training page, using a
pretrained model, and verify that the task actually enters the task list.

This file records one concrete, successful example that another AI can replay.

## Preconditions

Before starting this flow, verify all of the following:

1. The user is already logged in to Tencent Arena
2. The training page is open:

   ```text
   https://tencentarena.com/p/competition/stage/447/11585/11428/training
   ```

3. The page shows available resources, for example:

   ```text
   CPU资源: 6核可用
   GPU资源: 1卡可用
   ```

If resources are already `0核可用 / 0卡可用`, a new task may not be able to
start immediately.

## Important behavior

Clicking `新增训练任务` opens a right-side drawer on the same page.

It does not navigate to a new page.

## Verified successful example

The following configuration was successfully submitted:

- task name: `codex test`
- use pretrained model: enabled
- pretrained model name: `nan10_8750`
- algorithm: auto-filled as `PPO` after choosing the pretrained model
- training mode: auto-filled as `单机` after choosing the pretrained model
- duration: `1h 0min`
- description:

  ```text
  Codex automated test task created from pretrained model nan10_8750 for 1 hour validation.
  ```

## Step-by-step procedure

1. Open the `新增训练任务` drawer
2. Fill `任务名称` with:

   ```text
   codex test
   ```

3. Turn on `使用预训练模型`
4. Open `选择预训练模型`
5. In the model search box, search:

   ```text
   nan10_8750
   ```

6. Select the result whose visible model name is:

   ```text
   nan10_8750
   ```

7. After model selection, confirm that the drawer now shows:

   - `PPO`
   - `单机`

   These fields were observed as disabled auto-filled fields in the successful
   example.

8. Set task duration to:

   - hour = `1`
   - minute = `0`

9. Fill `描述（选填）` with any short description. The verified example used:

   ```text
   Codex automated test task created from pretrained model nan10_8750 for 1 hour validation.
   ```

10. Click `完成新增`

## Success criteria after submission

A successful submission was observed with the following visible effects on the
training list page:

1. The drawer closed
2. A new task card appeared at the top of the list
3. The new task card showed:

   - task name: `codex test`
   - task id: `#156140`
   - algorithm: `PPO`
   - training mode: `单机`
   - pretrained model: `nan10_8750`
   - simultaneous environments: `1`

4. Available resources dropped to:

   ```text
   CPU资源: 0核可用
   GPU资源: 0卡可用
   ```

This proves the platform accepted the task and allocated the team resources to
it.

## Runtime-state progression

The same task was then observed to transition from the initial waiting state
into an active running state.

Verified running-state signals:

- the task card status became `进行中`
- action buttons became visible on the task card:
  - `查看监控`
  - `模型列表`
  - `下载代码`
  - `释放`

These buttons are important:

- `查看监控`: opens the task-specific monitor page
- `模型列表`: shows saved models for this task
- `释放`: stops the current training run and releases the occupied resources

## Important caution

Do not assume a submission failed just because the task does not immediately
show `进行中`.

The task may first appear in an intermediate waiting or allocation state before
switching into the running state.
