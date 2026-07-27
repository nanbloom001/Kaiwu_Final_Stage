# archive/

历史代码快照与旧部署包。**不能被活动训练或部署入口引用**。

## 内容

| 路径 | 说明 |
|---|---|
| `代码存档/` | 各阶段的 Standard、Track、蒸馏与模型下载代码快照；大模型文件不纳入 Git |
| `unitree_isaaclab_deploy/` | （阶段 4 纳入）旧官方包/本地修改快照，标记为非活动部署目录 |

## 说明

- 归档快照仅供溯源，不代表活动代码。活动训练代码在 `../server/`，活动部署在 `../deploy/`。
- `代码存档/` 内含重复内容的代码树（各版本快照），Git 对相同 blob 自动复用对象。
- 归档源码中的历史 Token、Cookie 和 `.env` 不进入 Git；需要保留代码结构时以
  `REDACTED_ARCHIVE_SYNC_TOKEN` 等明确占位符替代。
- 历史版本与 checkpoint 血缘见 `../shared/分析记录/`。
