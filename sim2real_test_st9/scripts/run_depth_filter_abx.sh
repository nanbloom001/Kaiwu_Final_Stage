#!/usr/bin/env bash
# =============================================================================
# run_depth_filter_abx.sh —— 学长建议的深度后处理 A/B/C 对照测试
#
# 目的: 在固定路径、固定障碍、固定光照、固定速度下，对 RealSense SDK 后处理
#       滤波做 A/B/C 三组对照，判断哪种配置避障最好。
#
# 三组对照:
#   A (none)                        完全不加后处理（基线，训练分布最接近）
#   B (light_spatial)               仅轻量 spatial（学长认为最有希望）
#   C (light_spatial_weak_temporal) B + 很弱 temporal
#
# 每组都会:
#   1. 备份当前 config.yaml（trap EXIT 自动还原，绝不污染原配置）
#   2. 改 depth.filters.mode 为目标模式
#   3. 调用 run_loco_stage_test.sh 跑一次（自动归档到 logs/<ts>_<tag>/）
#   4. 记录 manifest（mode、bin sha256、时间戳）
#
# 用法:
#   cd ~/Kaiwu-test/sim2real_test_st9
#   ./scripts/run_depth_filter_abx.sh A           # 只跑配置 A
#   ./scripts/run_depth_filter_abx.sh B --network eth0
#   ./scripts/run_depth_filter_abx.sh C
#   ./scripts/run_depth_filter_abx.sh all         # 顺序跑 A→B→C（间隔人工重置场地）
#
# 注意:
#   - 每跑完一组请把狗放回起点、障碍复原、光照不变，再继续下一组
#   - 同步录像（手机侧面拍），便于事后比对
#   - 必须先 suspend 狗，LT+A 进 FixStand，再 LT+X 进 VisionLoco
# =============================================================================
set -uo pipefail

TEST_ROOT="${LOCO_TEST_ROOT:-/home/unitree/Kaiwu-test/sim2real_test_st9}"
CONFIG="${TEST_ROOT}/runtime/unitree_rl_lab_test/deploy/robots/go2_loco/config/config.yaml"
RUNNER="${TEST_ROOT}/scripts/run_loco_stage_test.sh"

if [[ ! -f "${CONFIG}" || ! -x "${RUNNER}" ]]; then
    echo "ERROR: config 或 runner 缺失" >&2
    echo "  CONFIG : ${CONFIG}" >&2
    echo "  RUNNER : ${RUNNER}" >&2
    exit 1
fi

MODE_TAG="$1"
shift || true
if [[ -z "${MODE_TAG}" || ! "${MODE_TAG}" =~ ^(A|B|C|all)$ ]]; then
    echo "Usage: $0 A|B|C|all [runner args...]" >&2
    exit 2
fi

declare -A MODE_VALUE=(
    [A]="none"
    [B]="light_spatial"
    [C]="light_spatial_weak_temporal"
)

# ── 改 config.yaml 的 depth.filters.mode ──
set_filter_mode() {
    local target="$1"
    python3 - "${CONFIG}" "${target}" <<'PY'
import re, sys
path, target = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    text = f.read()
# 匹配 depth.filters.mode（容许任意缩进、行尾注释）
pat = re.compile(r'(?m)^(\s*mode:\s*)(none|light_spatial|light_spatial_weak_temporal)(\s*(?:#.*)?)$')
new, n = pat.subn(lambda m: f"{m.group(1)}{target}{m.group(3)}", text)
if n == 0:
    raise SystemExit("ERROR: config.yaml 里没找到 depth.filters.mode 行")
with open(path, "w", encoding="utf-8") as f:
    f.write(new)
print(f"[filter] depth.filters.mode → {target}")
PY
}

# ── 跑一组 ──
run_one() {
    local tag="$1"
    shift   # 去掉 tag，剩余 $@ 才是给 runner 的参数
    local value="${MODE_VALUE[$tag]}"
    echo ""
    echo "=================================================================="
    echo "  对照 ${tag}: depth.filters.mode=${value}"
    echo "=================================================================="

    # 备份原 config，trap 退出时还原（即使 Ctrl+C 也还原）
    local backup
    backup="$(mktemp /tmp/loco_filter_cfg.XXXXXX)"
    cp "${CONFIG}" "${backup}"
    trap 'cp "${backup}" "${CONFIG}"; rm -f "${backup}"; echo "[filter] config 已还原"' EXIT INT TERM

    set_filter_mode "${value}"
    # 同步记录到运行目录的 manifest（run_loco_stage_test.sh 自己会归档）
    "${RUNNER}" "$@" || echo "[filter] runner 退出码非 0（继续）"

    # 还原 + 清 trap，下一组重新备份
    cp "${backup}" "${CONFIG}"
    rm -f "${backup}"
    trap - EXIT INT TERM
    echo "[filter] config 已还原，可切换下一组"
}

# ── 主流程 ──
if [[ "${MODE_TAG}" == "all" ]]; then
    echo "=================================================================="
    echo "  A/B/C 全量对照：每组跑完请人工把狗放回起点、障碍复原"
    echo "  按 Enter 开始 A（Ctrl+C 取消整个流程）"
    echo "=================================================================="
    read -r
    run_one "A" "$@"
    echo ""
    echo ">>> A 完成。请把狗放回起点 / 障碍复原 / 光照不变，按 Enter 跑 B"
    read -r
    run_one "B" "$@"
    echo ""
    echo ">>> B 完成。请把狗放回起点 / 障碍复原 / 光照不变，按 Enter 跑 C"
    read -r
    run_one "C" "$@"
    echo ""
    echo "=================================================================="
    echo "  A/B/C 全部完成。logs/ 下三份 run_dir，按时间戳排序对照"
    echo "=================================================================="
else
    run_one "${MODE_TAG}" "$@"
fi
