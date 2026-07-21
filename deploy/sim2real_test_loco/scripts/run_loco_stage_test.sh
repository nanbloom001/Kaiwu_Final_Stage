#!/usr/bin/env bash
# loco 阶段（lbc_loco）部署核心 runner —— 与 run_visnav_fixed_test.sh 同骨架，
# 但目标切换到 go2_loco 并行部署树：
#   binary : deploy/robots/go2_loco/build/go2_loco_ctrl
#   config : deploy/robots/go2_loco/config/config.yaml  （FSM.VisionLoco 段）
#   model  : logs/loco/exported/policy.onnx             （export_loco_onnx.py 产出）
#   log dir: logs/loco/logs/                            （visloco_diag_*.csv）
set -uo pipefail

# loco 阶段自包含部署包（deploy_loco）在 Jetson 上的落点，与 vision-nav 的
# sim2real_test 平级、互不干扰。落点不同就改这一行（或用 LOCO_TEST_ROOT 覆盖）。
TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_loco}"
LAB_ROOT="${TEST_ROOT}/runtime/unitree_rl_lab_test"
BUILD_DIR="${LAB_ROOT}/deploy/robots/go2_loco/build"
CONFIG="${LAB_ROOT}/deploy/robots/go2_loco/config/config.yaml"
MODEL="${LAB_ROOT}/logs/loco/exported/policy.onnx"
POLICY_LOG_DIR="${LAB_ROOT}/logs/loco/logs"
ANALYZER="${TEST_ROOT}/analyze_sim2real_logs.py"
ORT_LIB_DIR="${LAB_ROOT}/deploy/thirdparty/onnxruntime-linux-aarch64-1.19.2/lib"
EXPECTED_IPV4="${VISNAV_IPV4:-192.168.123.18/24}"

CHECK_ONLY=false
NETWORK=""

usage()
{
    cat <<'EOF'
Usage:
  ./scripts/run_loco_stage_test.sh [--check] [--network IFACE]

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

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config is missing: ${CONFIG}" >&2
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
if source not in ("fixed", "uwb", "nav"):
    raise SystemExit("command_source must be fixed, uwb or nav")
max_vx = float(vision.get("uwb", {}).get("max_vx", 0.3))
if not 0.0 <= max_vx <= 0.3:
    raise SystemExit("uwb.max_vx must be in [0, 0.3]")
uwb = vision.get("uwb", {})
stop_distance = float(uwb.get("stop_distance", 0.35))
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
PY
)

if [[ ${#config_values[@]} -ne 11 ]]; then
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
else
    echo "loco 阶段无 nav actor：command_source=nav 等价零速度（请用 fixed / uwb 做正式测试）。"
fi
if ${CHECK_ONLY}; then
    echo "Preflight passed. No controller was started."
    exit 0
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
else
    echo "loco nav mode: zero cmd (no learned nav actor at loco stage)."
fi
echo "Safety: suspend the robot, clear the path, keep LT+B ready."
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
