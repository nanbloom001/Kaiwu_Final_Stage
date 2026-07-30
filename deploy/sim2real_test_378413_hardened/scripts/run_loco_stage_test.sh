#!/usr/bin/env bash
# 378413 hardened 对照部署核心 runner。
# 但目标切换到 go2_loco 并行部署树：
#   binary : deploy/robots/go2_loco/build/go2_loco_ctrl
#   config : deploy/robots/go2_loco/config/config.yaml  （FSM.VisionLoco 段）
#   model  : logs/loco/exported/policy.onnx             （export_loco_onnx.py 产出）
#   log dir: logs/loco/logs/                            （visloco_diag_*.csv）
set -uo pipefail

restore_terminal()
{
    if [[ -t 0 ]]; then stty sane 2>/dev/null || true; fi
}
trap restore_terminal EXIT

# loco 阶段自包含部署包（deploy_loco）在 Jetson 上的落点，与 vision-nav 的
# sim2real_test 平级、互不干扰。落点不同就改这一行（或用 LOCO_TEST_ROOT 覆盖）。
TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_378413_hardened}"
LAB_ROOT="${TEST_ROOT}/runtime/unitree_rl_lab_test"
BUILD_DIR="${LAB_ROOT}/deploy/robots/go2_loco/build"
CONFIG="${LAB_ROOT}/deploy/robots/go2_loco/config/config.yaml"
MODEL="${LAB_ROOT}/logs/loco/exported/policy.onnx"
SOURCE_CHECKPOINT="${TEST_ROOT}/models/model.ckpt-vision-378413.pkl"
POLICY_LOG_DIR="${LAB_ROOT}/logs/loco/logs"
ANALYZER="${TEST_ROOT}/analyze_sim2real_logs.py"
SAFETY_CHECK="${LAB_ROOT}/deploy/robots/go2_loco/tests/check_safety_config.py"
ORT_LIB_DIR="${LAB_ROOT}/deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/lib"
EXPECTED_IPV4="${VISNAV_IPV4:-192.168.123.18/24}"
EXPECTED_MODEL_SHA256="823614aa579b38a57d3210bfdf32a15e92002768694cb0c9882983eba6bd8e72"
EXPECTED_CHECKPOINT_SHA256="37429c1e2c1d263844a74ecc2fdb97201ae296663f5ad8890ce173d5525a1288"

CHECK_ONLY=false
SUSPENDED_PARITY=false
HARNESS_GUARD=false
NETWORK=""
SAFETY_ACK_WORD="${LOCO_SAFETY_ACK_WORD:-SUSPENDED}"
SAFETY_CONTEXT="${LOCO_SAFETY_CONTEXT:-Suspend or firmly support the robot, clear all legs, and hold LT+B ready.}"
SAFETY_RUN_MESSAGE="${LOCO_SAFETY_RUN_MESSAGE:-Safety: suspend the robot, clear all legs, keep LT+B ready.}"

usage()
{
    cat <<'EOF'
Usage:
  ./scripts/run_loco_stage_test.sh [--check] [--network IFACE]

The suspended historical-parity mode is not a public runner option. Use
./scripts/run_loco_suspended_parity.sh while every foot is clear.
The contact-harness guard is not a public runner option. Use
./scripts/run_loco_harness_guard.sh only with a fall-arresting harness.

loco 阶段部署（go2_loco / VisionLoco）。命令来源、限幅、深度源均直接读 config.yaml。
不传 --network 时自动选择拥有 192.168.123.18/24 的接口。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --network)
            [[ $# -ge 2 ]] || { echo "ERROR: --network requires an interface." >&2; exit 2; }
            NETWORK="$2"
            shift 2
            ;;
        --suspended-parity)
            if [[ "${LOCO_SUSPENDED_PARITY_WRAPPER:-}" != "1" ]]; then
                echo "ERROR: --suspended-parity is only accepted from run_loco_suspended_parity.sh." >&2
                exit 2
            fi
            SUSPENDED_PARITY=true
            shift
            ;;
        --harness-guard)
            if [[ "${LOCO_HARNESS_GUARD_WRAPPER:-}" != "1" ]]; then
                echo "ERROR: --harness-guard is only accepted from run_loco_harness_guard.sh." >&2
                exit 2
            fi
            HARNESS_GUARD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            # Keep compatibility with the old positional interface argument.
            [[ -z "${NETWORK}" ]] || { echo "ERROR: unexpected argument: $1" >&2; exit 2; }
            NETWORK="$1"
            shift
            ;;
    esac
done

if ${SUSPENDED_PARITY} && ${HARNESS_GUARD}; then
    echo "ERROR: suspended parity and harness guard modes are mutually exclusive." >&2
    exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config is missing: ${CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${SAFETY_CHECK}" ]]; then
    echo "ERROR: safety configuration check is missing: ${SAFETY_CHECK}" >&2
    exit 1
fi

safety_check_args=("${CONFIG}" "${LAB_ROOT}/logs/loco/params/deploy.yaml")
if ${SUSPENDED_PARITY}; then
    safety_check_args+=(--suspended-parity)
elif ${HARNESS_GUARD}; then
    safety_check_args+=(--harness-guard)
fi
if ! python3 "${SAFETY_CHECK}" "${safety_check_args[@]}"; then
    echo "ERROR: safety configuration validation failed; controller was not started." >&2
    exit 1
fi

mapfile -t config_values < <(
    python3 - "${CONFIG}" <<'PY'
import sys
import math
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

vision = cfg["FSM"]["VisionLoco"]
source = vision.get("command_source", "fixed")
cmd = vision.get("fixed_cmd", [])
if len(cmd) != 3:
    raise SystemExit("fixed_cmd must contain [vx, vy, wz]")
if source not in ("fixed", "uwb", "nav", "keyboard"):
    raise SystemExit("command_source must be fixed, uwb, nav or keyboard")
max_vx = float(vision.get("uwb", {}).get("max_vx", 0.3))
if source != "keyboard" and not 0.0 <= max_vx <= 0.3:
    raise SystemExit("uwb.max_vx must be in [0, 0.3]")
uwb = vision.get("uwb", {})
is_uwb = (source == "uwb")
keyboard = vision.get("keyboard", {})
key_step_vx = float(keyboard.get("step_vx", 0.1))
key_step_wz = float(keyboard.get("step_wz", 0.1))
key_max_vx = float(keyboard.get("max_vx", 1.0))
key_max_wz = float(keyboard.get("max_wz", 0.8))
key_target_hz = float(keyboard.get("target_hz", 5.0))
key_idle_timeout = float(keyboard.get("idle_timeout_s", 10.0))
key_max_nonzero = float(keyboard.get("max_nonzero_s", 0.0))
if not (math.isfinite(key_step_vx) and key_step_vx > 0.0):
    raise SystemExit("keyboard.step_vx must be finite and > 0")
if not (math.isfinite(key_step_wz) and key_step_wz > 0.0):
    raise SystemExit("keyboard.step_wz must be finite and > 0")
if not (0.0 < key_max_vx <= 0.2):
    raise SystemExit("keyboard.max_vx must be in (0, 0.2] for ground testing")
if not (0.0 < key_max_wz <= 0.2):
    raise SystemExit("keyboard.max_wz must be in (0, 0.2] for ground testing")
if not (0.0 < key_target_hz <= 5.0):
    raise SystemExit("keyboard.target_hz must be in (0, 5]")
if not (math.isfinite(key_idle_timeout) and key_idle_timeout >= 1.0):
    raise SystemExit("keyboard.idle_timeout_s must be finite and >= 1")
if not (math.isfinite(key_max_nonzero) and 0.0 <= key_max_nonzero <= 10.0):
    raise SystemExit("keyboard.max_nonzero_s must be finite and in [0, 10]")
stop_distance = float(uwb.get("stop_distance", 0.35)) if is_uwb else 0.35
slow_distance = float(uwb.get("slow_distance", 1.0))
stop_hysteresis = float(uwb.get("stop_hysteresis", 0.15))
stop_once = bool(uwb.get("stop_once", False))
goal_driven = uwb.get("goal_driven", True)
if not isinstance(goal_driven, bool):
    raise SystemExit("uwb.goal_driven must be true or false")
cruise_cmd = uwb.get("cruise_cmd", [0.3, 0.0, 0.0])
if not isinstance(cruise_cmd, list) or len(cruise_cmd) != 3:
    raise SystemExit("uwb.cruise_cmd must contain [vx, vy, wz]")
cruise_cmd = [float(value) for value in cruise_cmd]
if not all(math.isfinite(value) for value in cruise_cmd):
    raise SystemExit("uwb.cruise_cmd must contain finite values")
if goal_driven and not (0.0 <= cruise_cmd[0] <= 0.8 and
                        abs(cruise_cmd[1]) <= 1e-6 and abs(cruise_cmd[2]) <= 1e-6):
    raise SystemExit("goal-driven uwb.cruise_cmd requires vx in [0,0.8] and vy=wz=0")
stale_timeout = float(uwb.get("stale_timeout", 0.5))
hold_timeout = float(uwb.get("hold_timeout", 1.5))
buffer_seconds = float(uwb.get("buffer_seconds", 6.0))
median_window = float(uwb.get("median_window", 0.45))
filter_tau = float(uwb.get("filter_tau", 0.25))
if stop_distance < 0.0:
    raise SystemExit("uwb.stop_distance must be >= 0")
if slow_distance <= stop_distance:
    raise SystemExit("uwb.slow_distance must be greater than uwb.stop_distance")
if stop_hysteresis < 0.0:
    raise SystemExit("uwb.stop_hysteresis must be >= 0")
if stale_timeout <= 0.0:
    raise SystemExit("uwb.stale_timeout must be > 0")
if hold_timeout <= stale_timeout:
    raise SystemExit("uwb.hold_timeout must be greater than uwb.stale_timeout")
for name, value in (("buffer_seconds", buffer_seconds),
                    ("median_window", median_window), ("filter_tau", filter_tau)):
    if value <= 0.0:
        raise SystemExit(f"uwb.{name} must be > 0")

print(source)
for value in cmd:
    print(float(value))
print(vision.get("depth", {}).get("source", "constant"))
print(max_vx)
print(str(stop_once).lower())
print(str(goal_driven).lower())
for value in cruise_cmd:
    print(value)
print(key_step_vx)
print(key_step_wz)
print(key_max_vx)
print(key_max_wz)
print(key_target_hz)
print(key_idle_timeout)
print(key_max_nonzero)
PY
)

if [[ ${#config_values[@]} -ne 18 ]]; then
    echo "ERROR: failed to read VisionLoco parameters from ${CONFIG}." >&2
    exit 1
fi

COMMAND_SOURCE="${config_values[0]}"
VX="${config_values[1]}"
VY="${config_values[2]}"
WZ="${config_values[3]}"
DEPTH_SOURCE="${config_values[4]}"
MAX_VX="${config_values[5]}"
STOP_ONCE="${config_values[6]}"
GOAL_DRIVEN="${config_values[7]}"
CRUISE_VX="${config_values[8]}"
CRUISE_VY="${config_values[9]}"
CRUISE_WZ="${config_values[10]}"
KEY_STEP_VX="${config_values[11]}"
KEY_STEP_WZ="${config_values[12]}"
KEY_MAX_VX="${config_values[13]}"
KEY_MAX_WZ="${config_values[14]}"
KEY_TARGET_HZ="${config_values[15]}"
KEY_IDLE_TIMEOUT="${config_values[16]}"
KEY_MAX_NONZERO="${config_values[17]}"

if ${SUSPENDED_PARITY}; then
    if [[ "${SAFETY_ACK_WORD}" != "SUSPENDED" ]]; then
        echo "ERROR: suspended parity requires the literal SUSPENDED acknowledgement." >&2
        exit 1
    fi
    echo "Suspended parity gate (motion only): raw action <= 8.0, target step <= 1.0 rad/frame, entry blend bypassed."
    echo "All other guards remain active; the non-zero command is still forced to zero at 2.0 seconds."
elif ${HARNESS_GUARD}; then
    if [[ "${SAFETY_ACK_WORD}" != "HARNESS_CLEAR" ]]; then
        echo "ERROR: harness guard requires the literal HARNESS_CLEAR acknowledgement." >&2
        exit 1
    fi
    echo "Harness guard: historical per-joint output bounds, effort <= 12.0 Nm/3 frames."
    echo "Motion targets are transparent for at most 2.0 seconds; forced-zero return keeps the normal limiter."
fi

if [[ -z "${NETWORK}" ]]; then
    NETWORK="$(
        ip -4 -o address show |
            awk -v expected="${EXPECTED_IPV4}" '$4 == expected {print $2; exit}'
    )"
fi
if [[ -z "${NETWORK}" ]]; then
    echo "ERROR: no interface owns ${EXPECTED_IPV4}." >&2
    echo "Check NetworkManager static-IP profile before running the controller." >&2
    exit 1
fi
if [[ ! -d "/sys/class/net/${NETWORK}" ]]; then
    echo "ERROR: network interface ${NETWORK} does not exist." >&2
    exit 1
fi
if [[ "$(cat "/sys/class/net/${NETWORK}/operstate")" != "up" ]]; then
    echo "ERROR: network interface ${NETWORK} is not UP." >&2
    exit 1
fi
if [[ -r "/sys/class/net/${NETWORK}/carrier" ]] &&
   [[ "$(cat "/sys/class/net/${NETWORK}/carrier")" != "1" ]]; then
    echo "ERROR: network interface ${NETWORK} has no physical carrier." >&2
    exit 1
fi
if ! ip -4 -o address show dev "${NETWORK}" | grep -Fq "inet ${EXPECTED_IPV4}"; then
    echo "ERROR: ${NETWORK} does not own ${EXPECTED_IPV4}." >&2
    exit 1
fi

if pgrep -x go2_loco_ctrl >/dev/null; then
    echo "ERROR: go2_loco_ctrl is already running. Stop the existing controller first." >&2
    exit 1
fi
if [[ ! -x "${BUILD_DIR}/go2_loco_ctrl" || ! -f "${MODEL}" ]]; then
    echo "ERROR: controller binary or ONNX model is missing." >&2
    exit 1
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source checkpoint is missing." >&2
    exit 1
fi
ACTUAL_MODEL_SHA256="$(sha256sum "${MODEL}" | awk '{print $1}')"
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
if [[ "${ACTUAL_MODEL_SHA256}" != "${EXPECTED_MODEL_SHA256}" ]]; then
    echo "ERROR: ONNX SHA256 mismatch: ${ACTUAL_MODEL_SHA256}" >&2
    exit 1
fi
if [[ "${ACTUAL_CHECKPOINT_SHA256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
    echo "ERROR: checkpoint SHA256 mismatch: ${ACTUAL_CHECKPOINT_SHA256}" >&2
    exit 1
fi
python3 - "${LAB_ROOT}/logs/loco/params/deploy.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    deploy = yaml.safe_load(handle)
meta = deploy.get("meta", {})
if meta.get("source_ckpt") != "model.ckpt-vision-378413.pkl":
    raise SystemExit("deploy.yaml source checkpoint mismatch")
if meta.get("source_ckpt_format") != "lbc_loco":
    raise SystemExit("deploy.yaml checkpoint format mismatch")
if meta.get("paired_onnx") != "policy.onnx":
    raise SystemExit("deploy.yaml paired ONNX mismatch")
PY

speed_tag="$(
    python3 - "${VX}" "${VY}" "${WZ}" <<'PY'
import sys

def tag(value):
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text

print("_".join(("vx" + tag(sys.argv[1]), "vy" + tag(sys.argv[2]), "wz" + tag(sys.argv[3]))))
PY
)"
stamp="$(date +%Y%m%d_%H%M%S)"
if [[ "${COMMAND_SOURCE}" == "fixed" ]]; then
    run_tag="loco_fixed_${speed_tag}"
elif [[ "${COMMAND_SOURCE}" == "uwb" ]]; then
    run_tag="loco_uwb_maxvx$(printf '%s' "${MAX_VX}" | tr '.-' 'pm')"
elif [[ "${COMMAND_SOURCE}" == "keyboard" ]]; then
    if ${HARNESS_GUARD}; then
        run_tag="loco_harness_guard_378413"
    elif ${SUSPENDED_PARITY}; then
        run_tag="loco_suspended_parity_378413"
    else
        run_tag="loco_keyboard_378413_hardened"
    fi
else
    run_tag="loco_nav_zero"
fi
run_dir="${TEST_ROOT}/logs/${stamp}_${run_tag}"

if [[ "${DEPTH_SOURCE}" == "realsense" ]]; then
    if ! command -v rs-enumerate-devices >/dev/null; then
        echo "ERROR: rs-enumerate-devices is unavailable." >&2
        exit 1
    fi
    camera_summary="$(rs-enumerate-devices -s 2>&1)"
    camera_status=$?
    if [[ ${camera_status} -ne 0 ]] ||
       ! grep -qi "Intel RealSense" <<<"${camera_summary}"; then
        echo "ERROR: no Intel RealSense depth camera was detected." >&2
        printf '%s\n' "${camera_summary}" >&2
        exit 1
    fi
    if ! ${CHECK_ONLY}; then
        mkdir -p "${run_dir}"
        printf '%s\n' "${camera_summary}" >"${run_dir}/realsense.txt"
    fi
fi

echo "Preflight: network=${NETWORK}, source=${COMMAND_SOURCE}, depth=${DEPTH_SOURCE}"
if [[ "${COMMAND_SOURCE}" == "fixed" ]]; then
    echo "Fixed command: [${VX}, ${VY}, ${WZ}]"
elif [[ "${COMMAND_SOURCE}" == "uwb" ]]; then
    if [[ "${GOAL_DRIVEN}" == "true" ]]; then
        echo "UWB autonomous navigation: goal-driven, model cmd=[${CRUISE_VX}, ${CRUISE_VY}, ${CRUISE_WZ}], stop_once=${STOP_ONCE}."
    else
        echo "UWB geometric control: vx hard limit=${MAX_VX} m/s, stop_once=${STOP_ONCE}."
    fi
elif [[ "${COMMAND_SOURCE}" == "keyboard" ]]; then
    echo "Keyboard setpoint: W/S=vx, A/D=wz, Space=zero; max=[${KEY_MAX_VX}, ${KEY_MAX_WZ}], idle=${KEY_IDLE_TIMEOUT}s, hard_nonzero=${KEY_MAX_NONZERO}s."
else
    echo "loco 阶段无 nav actor：command_source=nav 等价零速度（请用 fixed / uwb 做正式测试）。"
fi
if ${CHECK_ONLY}; then
    echo "Preflight passed. No controller was started."
    exit 0
fi

if [[ ! -t 0 ]]; then
    echo "ERROR: physical controller startup requires an interactive terminal (use ssh -t)." >&2
    exit 1
fi

echo "WARNING: starting the controller releases Unitree high-level control."
echo "${SAFETY_CONTEXT}"
read -r -p "Type ${SAFETY_ACK_WORD} to permit low-level takeover: " safety_ack
if [[ "${safety_ack}" != "${SAFETY_ACK_WORD}" ]]; then
    echo "Startup cancelled; low-level controller was not started."
    exit 1
fi

mkdir -p "${run_dir}" "${POLICY_LOG_DIR}"
cp "${CONFIG}" "${run_dir}/config.yaml"
cp "${LAB_ROOT}/logs/loco/params/deploy.yaml" "${run_dir}/deploy.yaml"
{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "network=${NETWORK}"
    echo "command_source=${COMMAND_SOURCE}"
    echo "fixed_cmd=${VX},${VY},${WZ}"
    echo "uwb_max_vx=${MAX_VX}"
    echo "uwb_stop_once=${STOP_ONCE}"
    echo "uwb_goal_driven=${GOAL_DRIVEN}"
    echo "uwb_cruise_cmd=${CRUISE_VX},${CRUISE_VY},${CRUISE_WZ}"
    echo "depth_source=${DEPTH_SOURCE}"
    echo "suspended_parity=${SUSPENDED_PARITY}"
    echo "harness_guard=${HARNESS_GUARD}"
    echo "hostname=$(hostname)"
    uname -a
    sha256sum "${BUILD_DIR}/go2_loco_ctrl" "${MODEL}" "${CONFIG}"
} >"${run_dir}/manifest.txt"

before_csv="$(find "${POLICY_LOG_DIR}" -maxdepth 1 -name 'visloco_diag_*.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"

echo "Run directory: ${run_dir}"
if [[ "${COMMAND_SOURCE}" == "fixed" ]]; then
    echo "Fixed command: vx=${VX} m/s, vy=${VY} m/s, wz=${WZ} rad/s"
elif [[ "${COMMAND_SOURCE}" == "uwb" ]]; then
    if [[ "${GOAL_DRIVEN}" == "true" ]]; then
        echo "UWB autonomous model: goal from UWB; model cmd=[${CRUISE_VX},${CRUISE_VY},${CRUISE_WZ}]; approach scaling only; stop_once=${STOP_ONCE}."
    else
        echo "UWB geometric controller: vx<=${MAX_VX} m/s; stop_once=${STOP_ONCE}."
    fi
elif [[ "${COMMAND_SOURCE}" == "keyboard" ]]; then
    echo "Keyboard: W/S changes vx by ${KEY_STEP_VX}; A/D changes wz by ${KEY_STEP_WZ}; Space requests zero."
    echo "Limits: vx=[0,${KEY_MAX_VX}], |wz|<=${KEY_MAX_WZ}, target_hz=${KEY_TARGET_HZ}, idle=${KEY_IDLE_TIMEOUT}s, hard_nonzero=${KEY_MAX_NONZERO}s."
else
    echo "loco nav mode: zero cmd (no learned nav actor at loco stage)."
fi
echo "${SAFETY_RUN_MESSAGE}"
echo "Controller: LT+A -> FixStand, LT+X -> VisionLoco, LT+B -> Passive."

cd "${BUILD_DIR}"
# Unitree SDK 的 CycloneDDS 共享库 (libddsc.so.0 / libddscxx.so.0) 装在
# /usr/local/lib,但这台 Jetson 的 /etc/ld.so.conf.d/ 没有把 /usr/local/lib
# 加入 ldconfig cache,直接启动会报 libddsc.so.0 not found。这里用 export
# 局部修复,只影响 go2_loco_ctrl 这个子进程:
#   - 不改 /etc/ld.so.conf.d/(避免 sudo + 全局污染)
#   - 不改 CMakeLists.txt RPATH(避免动活动 C++ 源码、避免重编)
#   - export 在这层 shell 退出后即失效,不泄漏到登录 session
export LD_LIBRARY_PATH="${ORT_LIB_DIR}:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ${HARNESS_GUARD}; then
    export LOCO_HARNESS_GUARD_TOKEN="HARNESS_GUARD_V1"
    unset LOCO_SUSPENDED_PARITY_TOKEN
elif ${SUSPENDED_PARITY}; then
    export LOCO_SUSPENDED_PARITY_TOKEN="SUSPENDED_PARITY_V1"
    unset LOCO_HARNESS_GUARD_TOKEN
else
    unset LOCO_SUSPENDED_PARITY_TOKEN
    unset LOCO_HARNESS_GUARD_TOKEN
fi
set +e
stdbuf -oL -eL ./go2_loco_ctrl --network "${NETWORK}" 2>&1 | tee "${run_dir}/console.log"
controller_status=${PIPESTATUS[0]}
set -e

after_csv="$(find "${POLICY_LOG_DIR}" -maxdepth 1 -name 'visloco_diag_*.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -n "${after_csv}" && "${after_csv}" != "${before_csv}" ]]; then
    cp "${after_csv}" "${run_dir}/"
    if [[ -f "${ANALYZER}" ]]; then
        python3 "${ANALYZER}" "${after_csv}" --compact \
            --json-out "${run_dir}/summary.json" | tee "${run_dir}/summary.txt"
    else
        echo "WARNING: analyzer is missing: ${ANALYZER}" | tee "${run_dir}/analysis_warning.txt"
    fi
else
    echo "No new VisionLoco CSV was created." | tee "${run_dir}/analysis_warning.txt"
fi

echo "controller_exit_status=${controller_status}" >>"${run_dir}/manifest.txt"
echo "finished_at=$(date --iso-8601=seconds)" >>"${run_dir}/manifest.txt"
exit "${controller_status}"
