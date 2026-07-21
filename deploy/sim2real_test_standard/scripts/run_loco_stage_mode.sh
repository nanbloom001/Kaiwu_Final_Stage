#!/usr/bin/env bash
# 切换 go2_loco config.yaml 的 FSM.VisionLoco.command_source 后调用 loco 阶段 runner。
# 与 run_visnav_mode.sh 同逻辑，只是目标 config / runner 换成 loco 部署树。
set -euo pipefail

TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_standard}"
CONFIG="${TEST_ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml"
RUNNER="${TEST_ROOT}/scripts/run_loco_stage_test.sh"

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
if [[ ! -f "${CONFIG}" || ! -x "${RUNNER}" ]]; then
    echo "ERROR: loco config or runner is missing." >&2
    exit 1
fi

backup="$(mktemp /tmp/loco_config.XXXXXX)"
cp "${CONFIG}" "${backup}"
restore_config()
{
    cp "${backup}" "${CONFIG}"
    rm -f "${backup}"
}
trap restore_config EXIT INT TERM

python3 - "${CONFIG}" "${MODE}" <<'PY'
import re
import sys

path, mode = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    text = handle.read()

updated, count = re.subn(
    r"(?m)^(\s*command_source:\s*)(fixed|uwb|nav)(\s*(?:#.*)?)$",
    rf"\g<1>{mode}\g<3>",
    text,
    count=1,
)
if count != 1:
    raise SystemExit("failed to locate exactly one VisionLoco command_source")

with open(path, "w", encoding="utf-8") as handle:
    handle.write(updated)
PY

"${RUNNER}" "$@"
