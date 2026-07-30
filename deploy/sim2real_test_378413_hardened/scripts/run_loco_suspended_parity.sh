#!/usr/bin/env bash
# Suspended-only diagnostic for measuring the historical 378413 motion envelope.
set -euo pipefail

TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_378413_hardened}"
CONFIG="${TEST_ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml"
RUNNER="${TEST_ROOT}/scripts/run_loco_stage_test.sh"

usage() {
    echo "Usage: $0 constant|realsense [--check] [--network IFACE]" >&2
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
if [[ ! -f "${CONFIG}" || ! -x "${RUNNER}" ]]; then
    echo "ERROR: deployment config or runner is missing." >&2
    exit 1
fi

backup="$(mktemp /tmp/loco_378413_suspended_parity.XXXXXX)"
cp "${CONFIG}" "${backup}"
restore_config() {
    cp "${backup}" "${CONFIG}"
    rm -f "${backup}"
}
trap restore_config EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - "${CONFIG}" "${depth_source}" <<'PY'
import re
import sys

path, source = sys.argv[1:]
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

echo "SUSPENDED-ONLY historical parity diagnostic: depth=${depth_source}, filter=none."
echo "Motion-only envelope: raw action <= 8.0, raw step <= 8.0, target step <= 1.0 rad/frame, no entry blend."
echo "Zero command and the forced-zero return retain the normal 6.0/1.0/0.03 envelope."
echo "Keep every foot clear. After ready stand completes, press W exactly once; do not press any other motion key."
echo "The command must return to zero after 2.0 seconds. Then use LT+B and Ctrl+C. This does not authorize ground use."

export LOCO_SUSPENDED_PARITY_WRAPPER=1
export LOCO_SAFETY_ACK_WORD="SUSPENDED"
export LOCO_SAFETY_CONTEXT="Robot is firmly suspended with every foot clear; handler and LT+B are ready."
export LOCO_SAFETY_RUN_MESSAGE="Safety: suspended parity pulse, one W only; automatic zero is required at 2 seconds."
"${RUNNER}" --suspended-parity "$@"
