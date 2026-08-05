// Convert Go2 UWB polar estimates to the normalized local goal expected by Actor80.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace vision_nav
{

inline float planar_distance_from_uwb(float pitch, float distance_m)
{
    if (!std::isfinite(pitch) || !std::isfinite(distance_m))
        return 0.0f;

    const float spatial_distance = std::max(0.0f, distance_m);
    return std::max(0.0f, spatial_distance * std::cos(pitch));
}

// 角度差(考虑环绕, 结果在 [-180, 180])
inline float angle_diff_deg(float a_deg, float b_deg)
{
    float d = a_deg - b_deg;
    while (d > 180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

// 跳变检测: 判断当前帧是否相对上一帧发生突变
// 抓包数据显示: 正常帧间跳变 <5°, 遮挡时跳变 30-46°
// 阈值设 60°: 超过就判定为跳变, 拒绝这一帧
struct JumpRejectState {
    bool   initialized = false;       // 是否有上一帧可比较
    float  last_beta_deg = 0.0f;      // 上一帧的方位角(度)
    float  last_planar_m = 0.0f;      // 上一帧的平面距离
    int    consecutive_rejects = 0;   // 连续拒绝次数
};

// 返回 true = 这帧可用; false = 跳变, 应丢弃
inline bool uwb_jump_reject(
    JumpRejectState& st,
    float beta_rad, float planar_distance_m,
    float max_angle_jump_deg = 60.0f,
    float max_distance_jump_m = 1.5f,
    int max_consecutive_rejects = 10)
{
    if (!std::isfinite(beta_rad) || !std::isfinite(planar_distance_m))
        return false;

    const float beta_deg = beta_rad * 57.29578f;

    if (!st.initialized) {
        st.initialized = true;
        st.last_beta_deg = beta_deg;
        st.last_planar_m = planar_distance_m;
        st.consecutive_rejects = 0;
        return true;
    }

    // 角度跳变
    const float ang_jump = std::fabs(angle_diff_deg(beta_deg, st.last_beta_deg));
    // 距离跳变(相对值, 防止远距离正常变化被误判)
    const float dist_jump = std::fabs(planar_distance_m - st.last_planar_m);

    const bool angle_bad = ang_jump > max_angle_jump_deg;
    const bool dist_bad = dist_jump > max_distance_jump_m;

    if (angle_bad || dist_bad) {
        st.consecutive_rejects++;
        // 连续拒绝太多次, 强制接受(防止UWB永久卡死)
        if (st.consecutive_rejects >= max_consecutive_rejects) {
            st.last_beta_deg = beta_deg;
            st.last_planar_m = planar_distance_m;
            st.consecutive_rejects = 0;
            return true;
        }
        return false;  // 拒绝这帧
    }

    // 正常帧, 更新状态
    st.last_beta_deg = beta_deg;
    st.last_planar_m = planar_distance_m;
    st.consecutive_rejects = 0;
    return true;
}

inline bool uwb_arrived_with_hysteresis(
    bool arrived, float planar_distance_m, float stop_distance_m, float hysteresis_m)
{
    if (!std::isfinite(planar_distance_m))
        return arrived;

    const float stop = std::max(0.0f, stop_distance_m);
    const float resume = stop + std::max(0.0f, hysteresis_m);
    return arrived ? planar_distance_m < resume : planar_distance_m <= stop;
}

inline float approach_speed_scale(
    float planar_distance_m, float stop_distance_m, float slow_distance_m)
{
    if (!std::isfinite(planar_distance_m) || !std::isfinite(stop_distance_m) ||
        !std::isfinite(slow_distance_m) || slow_distance_m <= stop_distance_m)
        return 0.0f;
    return std::clamp(
        (planar_distance_m - stop_distance_m) /
            (slow_distance_m - stop_distance_m),
        0.0f, 1.0f);
}

enum class ActorGoalEncoding
{
    LegacyActor80,
    DirectionPreservingV2,
};

inline std::array<float, 4> actor_goal_from_planar_xy(
    float local_x, float local_y,
    ActorGoalEncoding encoding = ActorGoalEncoding::LegacyActor80)
{
    if (!std::isfinite(local_x) || !std::isfinite(local_y))
        return {0.0f, 0.0f, 0.0f, 0.0f};

    const float planar_distance = std::hypot(local_x, local_y);
    const float distance_scale = std::clamp(planar_distance / 20.0f, 0.0f, 1.0f);
    if (encoding == ActorGoalEncoding::LegacyActor80 || planar_distance <= 10.0f) {
        return {
            std::clamp(local_x / 10.0f, -1.0f, 1.0f),
            std::clamp(local_y / 10.0f, -1.0f, 1.0f),
            distance_scale,
            0.0f,
        };
    }

    // Preserve bearing for distant goals instead of independently saturating X/Y.
    return {
        local_x / planar_distance,
        local_y / planar_distance,
        distance_scale,
        0.0f,
    };
}

inline std::array<float, 4> actor_goal_from_uwb(
    float beta, float pitch, float distance_m,
    ActorGoalEncoding encoding = ActorGoalEncoding::LegacyActor80)
{
    if (!std::isfinite(beta) || !std::isfinite(pitch) || !std::isfinite(distance_m))
        return {0.0f, 0.0f, 0.0f, 0.0f};

    const float planar_distance = planar_distance_from_uwb(pitch, distance_m);
    const float local_x = planar_distance * std::cos(beta);
    const float local_y = planar_distance * std::sin(beta);
    return actor_goal_from_planar_xy(local_x, local_y, encoding);
}

inline float time_filter_alpha(float dt_s, float tau_s)
{
    if (!std::isfinite(dt_s) || !std::isfinite(tau_s) || dt_s <= 0.0f)
        return 0.0f;
    if (tau_s <= 0.0f)
        return 1.0f;
    return std::clamp(1.0f - std::exp(-dt_s / tau_s), 0.0f, 1.0f);
}

inline float uwb_freshness_scale(float age_s, float stale_timeout_s, float hold_timeout_s)
{
    if (!std::isfinite(age_s) || !std::isfinite(stale_timeout_s) ||
        !std::isfinite(hold_timeout_s) || age_s < 0.0f ||
        stale_timeout_s < 0.0f || hold_timeout_s <= stale_timeout_s)
        return 0.0f;
    if (age_s <= stale_timeout_s)
        return 1.0f;
    if (age_s >= hold_timeout_s)
        return 0.0f;
    return (hold_timeout_s - age_s) / (hold_timeout_s - stale_timeout_s);
}

} // namespace vision_nav
