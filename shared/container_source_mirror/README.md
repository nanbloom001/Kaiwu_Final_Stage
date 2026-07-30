# 腾讯开悟平台托管源码本地镜像

本目录保存镜像刷新工具和读取规则，不保存镜像快照本体。

生成的本地镜像固定位于：

```text
shared/arena_frontend_monitor/runtime/container_source_mirror/
```

该路径由 `shared/arena_frontend_monitor/.gitignore` 的 `runtime/` 规则忽略。镜像仅供
本机查询平台实现，禁止成为 `server/` 或 `deploy/` 的运行时依赖。

## 为什么不提交镜像快照

镜像包含 Isaac Lab、Unitree RL Lab、Unitree ROS 等平台或第三方源码。直接提交完整快照会：

- 重复上游仓库并持续制造大范围差异；
- 增加仓库体积、许可证审查和版本漂移成本；
- 混淆“历史镜像”“当前容器”和“本仓库训练代码”三类证据；
- 诱使运行时代码依赖只应供诊断使用的本地缓存。

因此只提交本工具和规则。需要冻结某次平台实现时，应记录清单时间、来源根、目标文件 SHA256
与必要的短证据片段；除非经过独立的许可证和制品审查，不提交整份镜像。

## 镜像范围

默认读取以下平台托管代码根：

```text
/workspace/isaaclab
/workspace/unitree_rl_lab
/workspace/unitree_ros
/data/projects/legged_robot_competition_26/isaac_env
```

明确排除本项目的 `agent_ppo`、`agent_diy`、`conf` 训练代码，以及 checkpoint、日志、
缓存、模型网格、材质、大型二进制、第三方运行库和敏感凭据。

## 刷新

先在容器终端保持同步服务运行：

```bash
sh /data/projects/legged_robot_competition_26/conf/start_tongbu.sh
```

然后从仓库根目录执行：

```bash
/Users/nanbloom001/miniconda3/envs/PY311test/bin/python \
  shared/container_source_mirror/pull_container_source_mirror.py \
  --no-cookie-prompt \
  --prune-stale
```

工具使用现有 `server/local_sync_client.py` 的 RPC、Token 与 Cookie 选择逻辑；凭据不会写入
镜像或清单。同步是增量式的：本地文件和远端 SHA256 相同则直接复用。`--prune-stale`
只删除镜像清单曾管理、但当前远端清单已不存在的镜像文件。

## AI Agent 读取顺序

涉及平台托管实现时：

1. 先读取 `mirror_manifest.json`。
2. 核对 `format`、`generated_at`、`remote_roots`、`stats` 和目标文件 SHA256。
3. 使用 `rg` 在镜像中定位相关符号，再读取最小必要文件。
4. 明确写出证据来自“本地镜像”还是“当前容器 RPC”。
5. 目标文件缺失、镜像来自旧容器或问题涉及动态对象时，在线复核后再下结论。

示例：

```bash
MIRROR=shared/arena_frontend_monitor/runtime/container_source_mirror

jq '{generated_at,remote_roots,stats}' "$MIRROR/mirror_manifest.json"
rg -n "class ContactSensor|current_air_time" "$MIRROR/isaaclab"
rg -n "TrackTerrainGenerator|terrain_levels_vel" "$MIRROR/unitree_rl_lab"
```

镜像不能单独证明运行时传感器名称、body ID、张量 shape、环境实例配置或当前容器补丁状态；
这些内容必须通过 RPC 在实际环境中核对。
