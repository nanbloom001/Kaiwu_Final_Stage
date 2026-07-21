#!/usr/bin/env bash
# loco 阶段 · 固定速度测试（command_source=fixed）。
set -euo pipefail
exec "$(dirname "$0")/run_loco_stage_mode.sh" fixed "$@"
