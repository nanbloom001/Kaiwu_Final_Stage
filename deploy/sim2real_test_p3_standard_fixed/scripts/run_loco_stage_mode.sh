#!/usr/bin/env bash
# 切换 go2_loco config.yaml 的 FSM.VisionLoco.command_source 后调用 loco 阶段 runner。
# 与 run_visnav_mode.sh 同逻辑，只是目标 config / runner 换成 loco 部署树。
set -euo pipefail

TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu_Final_Stage-main/deploy/sim2real_test_p3_standard_fixed}"
CONFIG="${TEST_ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml"
RUNNER="${TEST_ROOT}/scripts/run_loco_stage_test.sh"
DEPTH_SOURCE_OVERRIDE="${LOCO_DEPTH_SOURCE:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 fixed|uwb|nav [runner arguments]" >&2
    exit 2
fi

MODE="$1"
shift
if [[ "${MODE}" != "fixed" && "${MODE}" != "uwb" && "${MODE}" != "nav" ]]; then
    echo "ERROR: mode must be fixed, uwb or nav." >&2
    exit 2
fi
if [[ -n "${DEPTH_SOURCE_OVERRIDE}" &&
      "${DEPTH_SOURCE_OVERRIDE}" != "realsense" &&
      "${DEPTH_SOURCE_OVERRIDE}" != "constant" ]]; then
    echo "ERROR: LOCO_DEPTH_SOURCE must be realsense or constant." >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" || ! -x "${RUNNER}" ]]; then
    echo "ERROR: loco config or runner is missing." >&2
    exit 1
fi

backup="$(mktemp /tmp/loco_config.XXXXXX)"
cp "${CONFIG}" "${backup}"
restore_config()
{
    if [[ -f "${backup}" ]]; then
        cp "${backup}" "${CONFIG}"
        rm -f "${backup}"
    fi
}
trap restore_config EXIT INT TERM

python3 - "${CONFIG}" "${MODE}" "${DEPTH_SOURCE_OVERRIDE}" <<'PY'
import sys
import yaml

path, mode, depth_source = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

vision = config["FSM"]["VisionLoco"]
vision["command_source"] = mode
if depth_source:
    vision.setdefault("depth", {})["source"] = depth_source

with open(path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
PY

"${RUNNER}" "$@"
