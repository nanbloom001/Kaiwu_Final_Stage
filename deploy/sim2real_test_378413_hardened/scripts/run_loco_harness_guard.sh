#!/usr/bin/env bash
# Contact-only candidate guard. Requires a fall-arresting harness.
set -euo pipefail

TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_378413_hardened}"
CONFIG="${TEST_ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml"
RUNNER="${TEST_ROOT}/scripts/run_loco_stage_test.sh"

if [[ ! -f "${CONFIG}" || ! -x "${RUNNER}" ]]; then
    echo "ERROR: deployment config or runner is missing." >&2
    exit 1
fi

backup="$(mktemp /tmp/loco_378413_harness_guard.XXXXXX)"
cp "${CONFIG}" "${backup}"
restore_config() {
    cp "${backup}" "${CONFIG}"
    rm -f "${backup}"
}
trap restore_config EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - "${CONFIG}" <<'PY'
import re
import sys

path = sys.argv[1]
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
    r"\1realsense\3",
    "depth.source",
)
replace_one(
    r"^(\s{8}mode:\s*)(none|light_spatial|light_spatial_weak_temporal)(\s*(?:#.*)?)$",
    r"\1none\3",
    "depth.filters.mode",
)
replace_one(r"^(\s{6}step_vx:\s*)[^\s#]+(.*)$", r"\g<1>0.15\2", "keyboard.step_vx")
replace_one(r"^(\s{6}max_vx:\s*)[^\s#]+(.*)$", r"\g<1>0.15\2", "keyboard.max_vx")
replace_one(r"^(\s{6}max_wz:\s*)[^\s#]+(.*)$", r"\g<1>0.1\2", "keyboard.max_wz")
replace_one(
    r"^(\s{6}max_nonzero_s:\s*)[^\s#]+(.*)$",
    r"\g<1>2.0\2",
    "keyboard.max_nonzero_s",
)

with open(path, "w", encoding="utf-8") as handle:
    handle.write(text)
PY

echo "CONTACT HARNESS candidate: RealSense, filter=none, one W=0.15 m/s, hard pulse=2.0s."
echo "The harness must carry the robot if all four legs lose support; ordinary hand support is not sufficient."
echo "Keep the handler beside LT+B. Enter VisionLoco, wait for ready stand, then press W exactly once."
echo "A 12 Nm/3-frame joint effort event, output-envelope violation, or inference fault transitions to Passive."
echo "Do not repeat W. Do not use this wrapper without the fall-arresting harness."

export LOCO_HARNESS_GUARD_WRAPPER=1
export LOCO_SAFETY_ACK_WORD="HARNESS_CLEAR"
export LOCO_SAFETY_CONTEXT="Robot is secured by a load-bearing fall-arrest harness; feet contact a flat high-friction surface; handler and LT+B are ready."
export LOCO_SAFETY_RUN_MESSAGE="Safety: harnessed contact candidate, one W only; automatic zero is required at 2 seconds."
"${RUNNER}" --harness-guard "$@"
