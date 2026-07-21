# Create Training Task Drawer

Last verified: 2026-05-13

## Goal

Inspect how `新增训练任务` actually works on the training page and record the
verified visible configuration fields.

## Important UI behavior

Clicking `新增训练任务` does not open a new page.

It opens a right-side drawer on the same training page.

This drawer is an Ant Design drawer, not a standalone route.

Useful DOM hints:

- `.ant-drawer-open`
- `.ant-drawer-content`

## Verified visible fields

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
- `描述`
- `完成新增`
- `取消`

## Verified control types

The following controls were directly visible in the drawer:

- `任务名称`: required text input
- `使用预训练模型`: switch, off by default
- `选择预训练模型`: dropdown / combobox
- `选择算法`: required dropdown / combobox
- `训练模式`: required dropdown / combobox, currently showing `单机`
- `任务时长`: two numeric inputs, split into hour and minute
- `同时启动环境数`: numeric input, currently disabled in the observed state
- `描述（选填）`: text input
- `完成新增`: submit button
- `取消`: cancel button

## Verified default or visible state

- `使用预训练模型`: off by default
- `训练模式`: `单机`
- `任务时长`: hour and minute numeric inputs exist
- `同时启动环境数`: numeric input exists and may be disabled until other
  required fields are set

Observed input state when the drawer was open:

- `任务名称`: empty, counter shows `0/20`
- `任务时长`: both hour and minute were `0`
- `描述（选填）`: empty

Verified visible cost preview:

```text
预估本次消耗
CPU 6.00 核, GPU 1.00 卡

预估团队剩余
CPU 0.00 核, GPU 0.00 卡
```

## Important interpretation

Do not assume the click failed just because a plain body-text dump still looks
similar to the base page.

The drawer may already be open even when a simple scrape does not clearly show a
separate modal boundary.

## Practical conclusion

The current `新增训练任务` drawer does include the fields the user asked to
verify:

- a task name field
- a pretrain-model toggle
- a pretrain-model selector
- an algorithm selector
- a training duration setting
- a description field
