# archive/

历史代码快照与旧部署包。**不能被活动训练或部署入口引用**。

## 内容

| 路径 | 说明 |
|---|---|
| `代码存档/` | 各阶段的 Standard、Track、蒸馏与模型下载代码快照；大模型文件不纳入 Git |
| `unitree_isaaclab_deploy/` | （阶段 4 纳入）旧官方包/本地修改快照，标记为非活动部署目录 |

## 训练模型包索引

以下条目只用于历史追溯。它们不是活动训练入口，也不是默认部署入口；使用前必须核对
SHA256、阶段代码 tag 和对应说明文档。

| 阶段 | 归档文件 | SHA256 | 对应代码 / 说明 |
|---|---|---|---|
| Standard command 父包 | `代码存档/standard-com_34728/ckpt/model.ckpt-commandfull-34728.pkl` | `0ce3b485053faa5da37c2b0ad792ad414ec6f5a482208c728cee65487d019d0f` | P1.5 父包；见 `shared/分析记录/2026-07-28_P1.5连续指令扩域与响应器八小时实施记录.md` |
| P1.5 ResponseAdapter | `代码存档/p15resp8h-r1_37953-F.zip` | `b251a4eefe56c316cc62463b9c21ad26eefdab80f305a5497812c9d4a28f404e` | ResponseAdapter 校准包 |
| P2 Track Nav PPO | `代码存档/p2nav8h-r2_291713.zip` | `77c2e2d46ac2ba189525c72947e6fdb10292b0ed4e9041cff004290cd819f7c9` | `archived/p2-track-nav-ppo-20260803` |
| P2 Track Nav 修复包 | `代码存档/p2nav2h-r2_648278.zip` | `cd82e6d98e32dc6c09a9341138d84d19cfd190ee1ef0c565a25a7bfd9e5bf784` | P2 safe-direction 后续包 |
| P3 Standard Joint | `代码存档/p3nav8h-r1_884257.zip` | `60d998cca3f0e3f52cbfcdb0c50718ab7b645d0de9888ab69f90b5fee4129f2b` | P3 径向/双评估血缘 |
| P3 Standard evalfix | `代码存档/p3nav8h-r1_884257-evalfix-v2.zip` | `7722cf9f919a08a2e5c7ec06839bc36d2da740f753ad1a947bb16ead452955ab` | 评估修复包，不代表新训练阶段 |
| P3 stair memory | `代码存档/p3stairmem8h_1013548.zip` | `5ef4021d9ab56cee5ffbc2673248dfc50e616795691a413b38ecdcef332f7e80` | `archived/p3-stairmem-dr-20260803` |

## 说明

- 归档快照仅供溯源，不代表活动代码。活动训练代码在 `../server/`，活动部署在 `../deploy/`。
- `代码存档/` 内含重复内容的代码树（各版本快照），Git 对相同 blob 自动复用对象。
- 归档源码中的历史 Token、Cookie 和 `.env` 不进入 Git；需要保留代码结构时以
  `REDACTED_ARCHIVE_SYNC_TOKEN` 等明确占位符替代。
- 历史版本与 checkpoint 血缘见 `../shared/分析记录/`。
- 2026-08-03 起，P2/P3/P3.5 的训练分支谱系统一见
  `../shared/分析记录/2026-08-03_训练版本谱系与分支归档说明.md`。
