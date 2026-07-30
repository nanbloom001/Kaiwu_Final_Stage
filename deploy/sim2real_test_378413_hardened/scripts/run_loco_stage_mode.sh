#!/usr/bin/env bash
# 切换 go2_loco config.yaml 的 FSM.VisionLoco.command_source 后调用 loco 阶段 runner。
# 与 run_visnav_mode.sh 同逻辑，只是目标 config / runner 换成 loco 部署树。
set -euo pipefail

TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_378413_hardened}"
CONFIG="${TEST_ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml"
RUNNER="${TEST_ROOT}/scripts/run_loco_stage_test.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 fixed|uwb|nav|keyboard [--once] [runner arguments]" >&2
    exit 2
fi

MODE="$1"
shift
if [[ "${MODE}" != "fixed" && "${MODE}" != "uwb" && "${MODE}" != "nav" && "${MODE}" != "keyboard" ]]; then
    echo "ERROR: mode must be fixed, uwb, nav or keyboard." >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" || ! -x "${RUNNER}" ]]; then
    echo "ERROR: loco config or runner is missing." >&2
    exit 1
fi

ONCE=false
runner_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)
            ONCE=true
            shift
            ;;
        *)
            runner_args+=("$1")
            shift
            ;;
    esac
done
if ${ONCE} && [[ "${MODE}" != "uwb" ]]; then
    echo "ERROR: --once is only valid with uwb mode." >&2
    exit 2
fi

backup="$(mktemp /tmp/loco_config.XXXXXX)"
cp "${CONFIG}" "${backup}"
restore_config()
{
    cp "${backup}" "${CONFIG}"
    rm -f "${backup}"
    if [[ -t 0 ]]; then stty sane 2>/dev/null || true; fi
}
trap restore_config EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - "${CONFIG}" "${MODE}" "${ONCE}" <<'PY'
import re
import sys

path, mode, once = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    text = handle.read()

updated, count = re.subn(
    r"(?m)^(\s*command_source:\s*)(fixed|uwb|nav|keyboard)(\s*(?:#.*)?)$",
    rf"\g<1>{mode}\g<3>",
    text,
    count=1,
)
if count != 1:
    raise SystemExit("failed to locate exactly one VisionLoco command_source")

updated, count = re.subn(
    r"(?m)^(\s*stop_once:\s*)(true|false)(\s*(?:#.*)?)$",
    rf"\g<1>{once.lower()}\g<3>",
    updated,
    count=1,
)
if count != 1:
    raise SystemExit("failed to locate exactly one VisionLoco uwb.stop_once")

with open(path, "w", encoding="utf-8") as handle:
    handle.write(updated)
PY

"${RUNNER}" "${runner_args[@]}"
