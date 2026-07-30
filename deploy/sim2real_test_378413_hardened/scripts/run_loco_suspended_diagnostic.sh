#!/usr/bin/env bash
# Compare constant depth with unfiltered RealSense input while the robot is suspended.
set -euo pipefail

TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_378413_hardened}"
CONFIG="${TEST_ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml"
RUNNER="${TEST_ROOT}/scripts/run_loco_stage_test.sh"

usage() {
    echo "Usage: $0 constant|realsense [zero|pulse] [runner arguments]" >&2
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

depth_source="$1"
shift
if [[ "${depth_source}" != "constant" && "${depth_source}" != "realsense" ]]; then
    usage
    exit 2
fi

diagnostic_mode="zero"
if [[ $# -gt 0 && ( "$1" == "zero" || "$1" == "pulse" ) ]]; then
    diagnostic_mode="$1"
    shift
fi
if [[ ! -f "${CONFIG}" || ! -x "${RUNNER}" ]]; then
    echo "ERROR: deployment config or runner is missing." >&2
    exit 1
fi

backup="$(mktemp /tmp/loco_378413_suspended_diag.XXXXXX)"
cp "${CONFIG}" "${backup}"
restore_config() {
    cp "${backup}" "${CONFIG}"
    rm -f "${backup}"
}
trap restore_config EXIT INT TERM

python3 - "${CONFIG}" "${depth_source}" "${diagnostic_mode}" <<'PY'
import re
import sys

path, source, mode = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    text = handle.read()

def replace_one(pattern, replacement, label):
    global text
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"failed to update {label}")

replace_one(
    r"^(\s*command_source:\s*)(fixed|uwb|nav|keyboard)(\s*(?:#.*)?)$",
    r"\1keyboard\3",
    "command_source",
)
replace_one(
    r"^(\s{6}source:\s*)(constant|realsense)(\s*(?:#.*)?)$",
    rf"\g<1>{source}\g<3>",
    "depth.source",
)
replace_one(
    r"^(\s{8}mode:\s*)(none|light_spatial|light_spatial_weak_temporal)(\s*(?:#.*)?)$",
    r"\1none\3",
    "depth.filters.mode",
)
replace_one(r"^(\s{6}max_vx:\s*)[^\s#]+(.*)$", r"\g<1>0.15\2", "keyboard.max_vx")
replace_one(r"^(\s{6}max_wz:\s*)[^\s#]+(.*)$", r"\g<1>0.1\2", "keyboard.max_wz")
if mode == "pulse":
    replace_one(r"^(\s{6}step_vx:\s*)[^\s#]+(.*)$", r"\g<1>0.15\2", "keyboard.step_vx")
    replace_one(
        r"^(\s{6}max_nonzero_s:\s*)[^\s#]+(.*)$",
        r"\g<1>2.0\2",
        "keyboard.max_nonzero_s",
    )

with open(path, "w", encoding="utf-8") as handle:
    handle.write(text)
PY

if [[ "${diagnostic_mode}" == "pulse" ]]; then
    echo "Suspended timeout diagnostic: depth=${depth_source}, filter=none, one W=0.15 m/s, hard pulse=2.0s."
    echo "Keep the robot firmly suspended with every foot clear. Enter VisionLoco and wait for ready stand complete."
    echo "Press W exactly once. Do not press Space or repeat W; verify the command is forced to zero at 2 seconds."
    echo "After the zero command remains stable, use LT+B. This tests the timeout only, not gait quality."
    export LOCO_SAFETY_CONTEXT="Robot is firmly suspended with every foot clear; handler and LT+B are ready."
    export LOCO_SAFETY_RUN_MESSAGE="Safety: suspended hard-timeout test, one W only; automatic zero is required at 2 seconds."
else
    echo "Suspended diagnostic: depth=${depth_source}, filter=none, keyboard caps=[0.15, 0.1]."
    echo "Keep the robot suspended. Enter VisionLoco at zero command, observe for 15 seconds,"
    echo "then use LT+B. Do not press W/A/S/D during the zero-command comparison."
fi
"${RUNNER}" "$@"
