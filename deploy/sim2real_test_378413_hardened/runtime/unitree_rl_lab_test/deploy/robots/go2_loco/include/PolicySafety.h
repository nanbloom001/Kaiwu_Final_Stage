#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace loco_safety
{

struct BoundedValueViolation
{
    bool found = false;
    size_t index = 0;
    float value = 0.0f;
    bool non_finite = false;
};

struct PerJointViolation
{
    bool found = false;
    size_t index = 0;
    float value = 0.0f;
    float limit = 0.0f;
    bool non_finite = false;
};

struct RuntimeActionEnvelope
{
    float max_raw_action_abs = 0.0f;
    float max_raw_action_step = 0.0f;
    float max_target_step_rad = 0.0f;
    bool bypass_entry_blend = false;
};

inline RuntimeActionEnvelope runtime_action_envelope(
    bool transparent_motion_mode, bool motion_active, float configured_raw_abs,
    float configured_raw_step, float configured_target_step)
{
    if (!transparent_motion_mode || !motion_active) {
        return {configured_raw_abs, configured_raw_step,
                configured_target_step, false};
    }
    // Covers all recorded successful 378413 motion frames while remaining
    // bounded. Stand hold and the forced-zero return retain the normal envelope.
    return {8.0f, 8.0f, 1.0f, true};
}

inline BoundedValueViolation first_bounded_value_violation(
    const std::vector<float>& values, float abs_limit)
{
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]))
            return {true, i, values[i], true};
        if (std::fabs(values[i]) > abs_limit)
            return {true, i, values[i], false};
    }
    return {};
}

inline PerJointViolation first_per_joint_abs_violation(
    const std::vector<float>& values, const std::vector<float>& limits)
{
    if (values.size() != limits.size())
        return {true, 0, std::numeric_limits<float>::infinity(), 0.0f, true};
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || !std::isfinite(limits[i]) ||
            limits[i] <= 0.0f)
            return {true, i, values[i], limits[i], true};
        if (std::fabs(values[i]) > limits[i])
            return {true, i, values[i], limits[i], false};
    }
    return {};
}

inline PerJointViolation first_per_joint_delta_violation(
    const std::vector<float>& values, const std::vector<float>& previous,
    const std::vector<float>& limits)
{
    if (values.size() != previous.size() || values.size() != limits.size())
        return {true, 0, std::numeric_limits<float>::infinity(), 0.0f, true};
    for (size_t i = 0; i < values.size(); ++i) {
        const float delta = values[i] - previous[i];
        if (!std::isfinite(delta) || !std::isfinite(limits[i]) ||
            limits[i] <= 0.0f)
            return {true, i, delta, limits[i], true};
        if (std::fabs(delta) > limits[i])
            return {true, i, delta, limits[i], false};
    }
    return {};
}

inline PerJointViolation first_per_joint_range_violation(
    const std::vector<float>& values, const std::vector<float>& lower,
    const std::vector<float>& upper)
{
    if (values.size() != lower.size() || values.size() != upper.size())
        return {true, 0, std::numeric_limits<float>::infinity(), 0.0f, true};
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || !std::isfinite(lower[i]) ||
            !std::isfinite(upper[i]) || lower[i] >= upper[i])
            return {true, i, values[i], 0.0f, true};
        if (values[i] < lower[i])
            return {true, i, values[i], lower[i], false};
        if (values[i] > upper[i])
            return {true, i, values[i], upper[i], false};
    }
    return {};
}

inline PerJointViolation update_effort_violation_counts(
    const std::vector<float>& effort, float abs_limit, int trip_frames,
    std::vector<int>& counts)
{
    if (effort.size() != counts.size() || !std::isfinite(abs_limit) ||
        abs_limit <= 0.0f || trip_frames < 1)
        return {true, 0, std::numeric_limits<float>::infinity(), abs_limit, true};
    for (size_t i = 0; i < effort.size(); ++i) {
        if (!std::isfinite(effort[i]))
            return {true, i, effort[i], abs_limit, true};
        counts[i] = std::fabs(effort[i]) > abs_limit ? counts[i] + 1 : 0;
        if (counts[i] >= trip_frames)
            return {true, i, effort[i], abs_limit, false};
    }
    return {};
}

inline bool reset_last_action_on_mode_transition(
    bool was_motion_active, bool motion_active,
    const std::vector<float>& stand_hold_action,
    std::vector<float>& last_action)
{
    if (was_motion_active == motion_active) return false;
    if (motion_active)
        last_action.assign(stand_hold_action.size(), 0.0f);
    else
        last_action = stand_hold_action;
    return true;
}

inline float torque_preserving_target(
    float position, float velocity, float previous_target,
    float previous_kp, float previous_kd,
    float next_kp, float next_kd, float max_offset)
{
    if (!std::isfinite(position) || !std::isfinite(velocity) ||
        !std::isfinite(previous_target) || !std::isfinite(previous_kp) ||
        !std::isfinite(previous_kd) || !std::isfinite(next_kp) ||
        !std::isfinite(next_kd) || !std::isfinite(max_offset) ||
        previous_kp < 0.0f || previous_kd < 0.0f || next_kp <= 0.0f ||
        next_kd < 0.0f || max_offset < 0.0f) {
        return position;
    }
    const float previous_torque =
        previous_kp * (previous_target - position) - previous_kd * velocity;
    const float candidate =
        position + (previous_torque + next_kd * velocity) / next_kp;
    return position + std::clamp(candidate - position, -max_offset, max_offset);
}

inline float max_abs_delta(
    const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    if (lhs.size() != rhs.size()) return std::numeric_limits<float>::infinity();
    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i]))
            return std::numeric_limits<float>::infinity();
        result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    }
    return result;
}

inline int update_violation_count(float observed, float limit, int previous_count)
{
    if (!std::isfinite(observed) || observed > limit) return previous_count + 1;
    return 0;
}

inline bool should_trip(int violation_count, int trip_frames)
{
    return trip_frames > 0 && violation_count >= trip_frames;
}

inline float limit_target_step(float previous, float candidate, float max_step)
{
    return previous + std::clamp(candidate - previous, -max_step, max_step);
}

inline bool command_is_zero(
    const std::array<float, 3>& command, float threshold)
{
    if (!std::isfinite(threshold) || threshold < 0.0f) return false;
    for (float value : command) {
        if (!std::isfinite(value) || std::fabs(value) > threshold) return false;
    }
    return true;
}

inline bool command_duration_expired(float elapsed_s, float limit_s)
{
    return std::isfinite(elapsed_s) && std::isfinite(limit_s) &&
           limit_s > 0.0f && elapsed_s >= limit_s;
}

} // namespace loco_safety
