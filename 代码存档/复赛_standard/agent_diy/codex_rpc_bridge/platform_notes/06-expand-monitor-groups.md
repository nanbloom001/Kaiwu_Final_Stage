# Expand Monitor Groups

Last verified: 2026-05-13

## Goal

Record the verified collapsible-group behavior on the monitor page.

## Verified collapsible groups

The following groups were verified as clickable fold/unfold buttons:

- `地形-斜坡( 8 )`
- `地形-倒斜坡( 8 )`

They were observed with:

- `expanded=false` before clicking
- `expanded=true` after clicking

## Verified result after expansion

After expansion, the previously missing metric names appeared in page text.

### 地形-斜坡( 8 )

- `斜坡-完成数`
- `斜坡-失败数`
- `斜坡-超时数`
- `斜坡-总分`
- `斜坡-时间分数`
- `斜坡-姿态分数`
- `斜坡-能耗分数`
- `斜坡-步数`

### 地形-倒斜坡( 8 )

- `倒斜坡-完成数`
- `倒斜坡-失败数`
- `倒斜坡-超时数`
- `倒斜坡-总分`
- `倒斜坡-时间分数`
- `倒斜坡-姿态分数`
- `倒斜坡-能耗分数`
- `倒斜坡-步数`

## Practical rule

Do not conclude that metrics are missing just because a group title is visible
without child items.

First verify whether the group is collapsed.

If it is collapsed, expand it before collecting the metric inventory.
