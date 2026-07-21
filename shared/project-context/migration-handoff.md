# Kaiwu_Final_Stage 仓库迁移 - 交接状态（供审查）

> 生成：2026-07-21。本文件自包含，供另一 AI / 审查者接手审查与完成剩余工作。
> 源规约：本仓库 `分析记录/版本训练演进与改动规模详解.md` 与本目录 `branch-migration-register.md`、`repository-layout.md`。

## 1. 项目与目标

- **仓库**：https://github.com/nanbloom001/Kaiwu_Final_Stage （腾讯开悟四足机器人自主导航 Sim2Real 决赛；Unitree Go2 + Isaac Lab；track 导航 + standard 运控 + 蒸馏/Sim2Real）
- **目标**：① 仓库重构为四层（`server/` 训练运行时 / `deploy/` 真机部署 / `shared/` 规则+分析+契约 / `archive/` 历史快照）；② 分支治理——保留重大改动节点作 tag 或活动实验分支，删除散落的历史记录分支。
- **关键事实**：原 `main` = `042a1f8`（初始决赛代码，仅 1 提交）；远程主干 = `codex/st7-opt3`(`9b0b3df`) 及其后续。`main` 是迁移分支的祖先（fast-forward 可行，但应走 PR merge commit）。

## 2. 已完成工作（阶段 0-4 + 5a + 静态验证）

迁移分支 `codex/repository-layout-migration`，从 `codex/st7-opt3` 起步，共 11 提交（已推送 origin）：

| 提交 | 内容 |
|---|---|
| `dc24b48` | 四层重构：`git mv` 训练运行时->`server/`、规则+分析->`shared/`、代码存档->`archive/` + 根 README/.gitignore + 目录 README + 迁移登记 |
| `4bcc0f0` | J9 固定学习率接口移植（PPO 构造器接收 schedule/min/max_learning_rate + 运行时守卫 + 启动校验 + 单测） |
| `fda69d5` | Opt4 角速度阈值+权重移植（dynamic_tilt_risk） |
| `97ac193` | merge: J9（--no-ff，可追溯） |
| `904d79d` | oracle 复查修正：强化 `_validate_fixed_lr()`（算法 LR + 全 optimizer 组，learn() entry+exit 双查）；回退 `TrackNavConfig` 到 Opt3 自适应（仅移植接口，未提升固定学习率进基线） |
| `6aed926` | deploy subtree 导入（`git subtree add --prefix=deploy`，第二父 = `b297b3f`，保留部署独立历史） |
| `40f0a11` | 四套 deploy `ARTIFACTS.md` + `deploy/README.md` |
| `3331d23` | archive 纳入 `track蒸馏3_378413`（真机部署 ckpt，SHA256 `37429c1e…5a1288`，10.6MB 真文件） |
| `0694877` | shared 纳入 07-11/07-12 分析报告 4 份 |
| `b507be9` | archive 纳入 `unitree_isaaclab_deploy`（10 个 git-lfs pointer 文件经 `.gitignore` 排除，134 真实文件保留） |

**7 个 tag 已推送 origin**：`stage3i2-nogate-safe-60m`(d777c1f)、`stage3j4-neargoal-rejected`(d372dce)、`st7-opt2a-baseline-rewrite`(09d88e7)、`st7-opt3-server-baseline`(9b0b3df)、`st7-opt5-hard-start-replay`(b81cbf4)、`st7-opt5-debug-hard-start-diagnostics`(9570780)、`st9-opt3-d1-vision-distill`(5ea7ba9)

**PR**：https://github.com/nanbloom001/Kaiwu_Final_Stage/pull/1 （**draft**，未合并）

**静态验证（已执行，全过）**：
- `bash -n` 部署脚本 22/22 通过
- `py_compile` 全量 236 个 .py，0 错误
- TOML 解析 43/43 通过
- markdown 相对链接 17 个，0 断链
- `server/` 无运行时引用 `shared/`/`archive/`；`deploy/` 无运行时引用 `server/`（subtree 独立性确认）
- ⚠️ `git diff --check` 报告的 whitespace 全部是**迁移内容原有**（server 代码、deploy C++ runtime、archive 旧包因 `git mv`/subtree/import 在 diff 中表现为新增行而暴露），**非迁移引入**；迁移作者实际编辑的文件（J9/Opt4 值改动、README、ARTIFACTS）干净。是否清理这些原有 whitespace 为可选（清理会改动被保留内容）。

## 3. 当前状态

- **迁移分支**：`codex/repository-layout-migration` @ `b507be9`，已推送 origin，与 origin 同步
- **结构**：`server/`(108 文件) / `shared/`(41) / `archive/`(444) / `deploy/`(439) + 根 README+.gitignore
- **worktree**：`/Users/nanbloom001/codespace/fwwb-migration`（迁移分支）；主仓 `/Users/nanbloom001/codespace/fwwb-Final` 保持在 `main`，未跟踪文件未动
- **基线**：`server/` = ST7-Opt3 + J9 接口修复（latent，TrackNav 保自适应 1.5e-5）+ Opt4 角速度调整

## 4. 关键决策与注意事项

1. **J9 仅移植通用接口**（非整实验提升）：`TrackNavConfig` 保持 Opt3 自适应 `lr=1.5e-5`（无 schedule/min/max 字段）；接口修复 latent（PPO 构造器不再静默忽略 schedule/bounds）。J9 固定学习率实验以 `server/conf/train_env_conf_track_navj9.toml` 作记录。理由：register 写"TOML 仅作记录"、Plan 写"默认保持原行为"，未经验证的实验不应自动进基线。**如本意是要 TrackNav 用固定 1e-5，请告知**。
2. **deploy 用 merge commit 导入**：第二父 = `b297b3f`（部署独立历史根），已验证。四套目录并列不去重（两份 378413 ckpt Git 自动复用 blob）。
3. **LFS pointer 排除**：`archive/unitree_isaaclab_deploy/` 有 10 个 git-lfs pointer（ONNX Runtime .so、`model.ckpt-lbc-loco-637427.pkl`、mimic CSV、npy、标定 PDF），真实对象不在本地，经 `.gitignore` 排除（Plan 禁止把 pointer 登记为可用 ckpt）。需真实对象则在源仓 `git lfs pull`。
4. **ST7/standard deploy 缺 checkpoint**：`sim2real_test_st7` 缺 `model.ckpt-track-lbc-loco-28608.pkl`；`sim2real_test_standard` 缺 `model.ckpt-hjcnew-20288.pkl`。各 `ARTIFACTS.md` 已标"待提供"。补齐前 deploy 这两套只能源码静态验收。
5. **合并要求**：PR #1 合入 main **必须用 merge commit，禁 squash/rebase**（squash 会使 J8/J9/Opt1A/Opt2B 等 trunk SHA 在分支删除后失去保护；rebase 改写 SHA 并破坏 deploy subtree 拓扑）。建议合入后启用 main 分支保护。

## 5. 剩余工作（gate + 命令）

### 5b. review PR + 合入 main（需人工 review）
1. review https://github.com/nanbloom001/Kaiwu_Final_Stage/pull/1
2. 合入：**merge commit**（PR 页面选 "Create a merge commit"，勿 squash/rebase）
3. 启用 main 分支保护（GitHub Settings → Branches → main：require PR review + 禁 force-push）
4. 验证远程 main 含 `b507be9`、`9b0b3df`、`b297b3f`

### 5c. 验证（需 Isaac Lab 训练环境 + Jetson + Go2，本机无法执行）
1. 在腾讯开悟容器内执行 `server/train_test.py` 短 smoke test；确认 Opt3 goal_noise、Opt2B dynamic_tilt_risk、Opt4 angular-rate 实际注册
2. 跑一个训练周期（30-60min），观察 J9 `_validate_fixed_lr` 日志、学习率恒定、L9 完成率
3. Jetson 上一次 Sim2Real 部署验证（`deploy/sim2real_test_loco/`，378413 ckpt 已入库）：构建 binary -> 部署 ONNX Runtime -> `run_loco_stage_test.sh --check` -> 正式 run

### 6. 删散落分支（须 5c 验证通过后）
按 `shared/project-context/branch-migration-register.md` 的"安全删除判定"：

**先打 archived tag 再删（独立侧支，删前必须 tag 否则 gc 丢失）**：
```bash
git tag -a archived/stage3j5-p14 b589ed1 -m "archived: J5 ablation (superseded)"
git tag -a archived/stage3j6-termination b2891e4 -m "archived: J6 termination (superseded)"
git tag -a archived/stage3j7-rough-energy 1acc731 -m "archived: J7 rough-energy (superseded)"
git tag -a archived/timeout-bootstrap-diag 13a1b19 -m "archived: timeout diagnostic (no model promotion)"
git tag -a archived/st7-opt4-angular-rate 5f8764d -m "archived: Opt4 (ported to server/)"
git push origin --tags
# 然后删分支
git push origin --delete codex/stage3j5-p14 codex/stage3j6-termination codex/stage3j7-rough-energy timeout-bootstrap-diag codex/st7-opt4-angular-rate
```

**tip 在主干上（迁移/main 已保护，直接删零损失）**：
```bash
git push origin --delete cyy codex/stage3j8-hard-level-replay codex/stage3j9-fixed-learning-rate codex/st7-opt1a-speed-wall-stall codex/st7-opt2a-tilt-limit codex/st7-opt2b-dynamic-stability codex/st7-opt3
# deploy 已 subtree 导入，历史作为第二父保留，可删：
git push origin --delete deploy/jetson-sim2real
```

**必须保留分支（活动实验路线，其 tip 不被 main 保护）**：
- `codex/st7-opt5-hard-start-replay`（已 tag，保留）
- `codex/st7-opt5-debug`（已 tag，保留）
- `codex/st9-opt3-d1`（已 tag，保留）

## 6. 给审查 AI 的指引

**重点审查**：
1. J9 移植是否真的仅接口：`server/agent_ppo/conf/conf.py` 的 `TrackNavConfig` 应为 `lr=1.5e-5`、无 `schedule`/`min_learning_rate`/`max_learning_rate`
2. `_validate_fixed_lr()` 在 `learn()` entry+exit 双查（`server/agent_ppo/algorithm/algorithm_ppo.py`）
3. Opt4 值：`reward_process.py` 的 `_reward_dynamic_tilt_risk` 阈值 `0.50/1.60/0.65/1.90`、权重 `0.35/0.30`；`train_env_conf_track_nav.toml` 同步
4. deploy 第二父拓扑：`git cat-file -p <merge-commit> | grep parent` 应含 `b297b3f`
5. LFS pointer 未泄漏：`git ls-files archive/unitree_isaaclab_deploy/ | xargs -I{} sh -c 'test $(wc -c <{}) -eq 133 && echo {}'` 应为空
6. 四套 ARTIFACTS 完整性、st7/standard 缺失 ckpt 是否标注

**关键文件指针**：
- `shared/project-context/branch-migration-register.md`（源 SHA 锚点 + 分支分类 + 安全删除判定）
- `shared/project-context/repository-layout.md`（四层结构 + 重命名显示说明）
- `server/CHANGELOG.md`（J9/Opt4 移植记录 + 接口-only 决策）
- `分析记录/版本训练演进与改动规模详解.md`（各版本训练演进与改动规模）
- `shared/interfaces/server-deploy-contract.md`（stub，待人工填充 server↔deploy 契约）

**回滚**：迁移在独立 worktree/分支完成，未合入 main 前可 `git worktree remove /Users/nanbloom001/codespace/fwwb-migration` 放弃；合入后用 `git revert`（deploy subtree 用 `git revert -m 1`），禁 reset/rebase/force-push。
