# 仓库布局说明

> 本文件说明 kaiwu-final-sim2real 仓库的四层目录结构与迁移背景。

## 四层结构

```text
kaiwu-final-sim2real/
├── README.md
├── .gitignore              # 仓库级规则
├── server/                 # 训练运行时（Opt3 基线 + J9 接口 + Opt4）
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── .gitignore          # server 独立使用时的缓存规则
│   ├── .vscode/
│   ├── agent_diy/
│   ├── agent_ppo/
│   ├── conf/
│   ├── isaac_env/
│   ├── docs/
│   ├── kaiwu.json
│   ├── local_sync_client.py
│   └── train_test.py
├── deploy/                 # Jetson Sim2Real 真机部署树（四套并列，subtree 导入）
│   ├── README.md
│   ├── .gitignore
│   ├── sim2real_test_loco/
│   ├── sim2real_test_st7/
│   ├── sim2real_test_st9/
│   └── sim2real_test_standard/
├── shared/                 # 规则/分析/接口/协作（禁止运行时依赖）
│   ├── README.md
│   ├── 规则说明/
│   ├── 分析记录/
│   ├── interfaces/
│   │   └── server-deploy-contract.md
│   └── project-context/
│       ├── repository-layout.md   ← 本文件
│       └── branch-migration-register.md
└── archive/                # 历史快照（不能被活动入口引用）
    ├── README.md
    ├── 代码存档/
    └── unitree_isaaclab_deploy/   # 阶段 4 纳入
```

## 约束

- `server/` 必须从其自身目录运行；腾讯开悟上传/同步以 `server/` 为项目根。
- `deploy/` 不导入 `server/` 中的 Python 模块、配置或 checkpoint 路径。
- `shared/` 只保存规则、报告、接口契约和协作文档，禁止成为运行时依赖。
- `archive/` 内容不能被活动训练或部署入口引用。
- 四套部署目录暂不去重、不抽公共代码、不改变相互并列结构。

## 迁移背景

- 迁移日期：2026-07-21
- 迁移分支：`codex/repository-layout-migration`（起点 `codex/st7-opt3` = `9b0b3df`）
- 迁移方式：独立 worktree，`git mv` 整体移动，不重写历史、不 force-push；合入后只删除已被 `main`、保留实验分支或 annotated tag 保护的远程分支。
- 回滚：每个阶段单独提交；合并后只用 `git revert`（subtree 合并用 `git revert -m 1`）。

## 路径移动后的适配验证

迁移前调查结论（已验证）：
- 代码/配置中**无**对 `规则说明/`/`代码存档/`/`分析记录/` 的引用（`git grep` 确认）。
- `.vscode/launch.json` 使用 `${workspaceFolder}`（相对工作区），打开 `server/` 即自适应。
- `local_sync_client.py` / `conf/tongbu.py` 以 cwd / `--root` / `IDE_SYNC_ROOT` 为相对根，从 `server/` 运行即生效。
- `kaiwu.json` / `configure_app.toml` 无绝对路径；`preload_model_dir = "{agent_name}/ckpt"` 为模板相对路径。

## 注意：git status 重命名显示

`代码存档/` 内含与根代码内容相同的快照树（如 `决赛_track蒸馏_162404/` 是 `agent_ppo/` 的快照）。`git mv` 整体移动后，`git status`/`git log --stat` 的**重命名检测启发式**会把根文件配对到 archive 路径（显示为 `agent_diy/x -> archive/代码存档/.../agent_diy/x`）。这是**纯显示问题**，实际 tree 与索引正确（`git ls-files` 可验证）。提交后 `git diff --stat` 的重命名配对可能继续如此显示，但不影响内容正确性。
