#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace strict_loco
{

using Clock = std::chrono::steady_clock;

struct SensorSnapshot
{
    uint64_t sequence = 0;
    uint32_t lowstate_tick = 0;
    Clock::time_point received_at{};
    float lowstate_age_ms = 1.0e9f;
    std::array<float, 4> quaternion_wxyz{};
    std::array<float, 3> projected_gravity{};
    std::array<float, 3> angular_velocity{};
    std::array<float, 12> q{};
    std::array<float, 12> dq{};
    std::array<float, 12> tau_est{};
    std::array<float, 12> motor_temperature_c{};
    float battery_voltage_v = 0.0f;
    float battery_soc = 0.0f;
    bool finite = false;
    bool quaternion_valid = false;
    bool valid = false;
    std::string invalid_reason = "uninitialized";
};

inline bool finite_array(const float* values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

inline bool validate_quaternion_wxyz(const std::array<float, 4>& q)
{
    if (!finite_array(q.data(), q.size())) return false;
    const float norm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    return norm2 >= 0.81f && norm2 <= 1.21f;
}

class SnapshotStore
{
public:
    template <typename LowStatePtr>
    SensorSnapshot capture(
        const LowStatePtr& lowstate,
        const std::vector<int>& joint_ids_map,
        Clock::time_point now = Clock::now())
    {
        SensorSnapshot next;
        {
            std::lock_guard<std::mutex> lowstate_lock(lowstate->mutex_);
            const auto& msg = lowstate->msg_;
            next.lowstate_tick = msg.tick();
            const auto& imu = msg.imu_state();
            std::copy(imu.quaternion().begin(), imu.quaternion().end(), next.quaternion_wxyz.begin());
            std::copy(imu.gyroscope().begin(), imu.gyroscope().end(), next.angular_velocity.begin());
            for (size_t i = 0; i < next.q.size(); ++i) {
                const auto& motor = msg.motor_state().at(joint_ids_map.at(i));
                next.q[i] = motor.q();
                next.dq[i] = motor.dq();
                next.tau_est[i] = motor.tau_est();
                next.motor_temperature_c[i] = static_cast<float>(motor.temperature());
            }
            next.battery_soc = static_cast<float>(msg.bms_state().soc());
            for (uint16_t millivolts : msg.bms_state().cell_vol())
                next.battery_voltage_v += static_cast<float>(millivolts) * 0.001f;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!have_tick_ || next.lowstate_tick != last_tick_) {
            last_tick_ = next.lowstate_tick;
            last_tick_time_ = now;
            have_tick_ = true;
        }
        next.sequence = ++sequence_;
        next.received_at = last_tick_time_;
        next.lowstate_age_ms = have_tick_
            ? std::chrono::duration<float, std::milli>(now - last_tick_time_).count()
            : 1.0e9f;
        next.quaternion_valid = validate_quaternion_wxyz(next.quaternion_wxyz);
        if (next.quaternion_valid) {
            const Eigen::Quaternionf quat(
                next.quaternion_wxyz[0], next.quaternion_wxyz[1],
                next.quaternion_wxyz[2], next.quaternion_wxyz[3]);
            const Eigen::Vector3f gravity = quat.normalized().conjugate() * Eigen::Vector3f(0.f, 0.f, -1.f);
            for (int i = 0; i < 3; ++i) next.projected_gravity[i] = gravity[i];
        }
        next.finite = finite_array(next.angular_velocity.data(), 3) &&
                      finite_array(next.q.data(), 12) && finite_array(next.dq.data(), 12) &&
                      finite_array(next.tau_est.data(), 12) &&
                      finite_array(next.projected_gravity.data(), 3) &&
                      std::isfinite(next.battery_voltage_v) && std::isfinite(next.battery_soc);
        if (!next.quaternion_valid) next.invalid_reason = "invalid_quaternion_wxyz";
        else if (!next.finite) next.invalid_reason = "nonfinite_lowstate";
        else if (next.lowstate_age_ms > 50.0f) next.invalid_reason = "lowstate_stale";
        else {
            next.invalid_reason = "none";
            next.valid = true;
        }
        latest_ = next;
        return next;
    }

    SensorSnapshot latest() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_;
    }

private:
    mutable std::mutex mutex_;
    SensorSnapshot latest_{};
    uint64_t sequence_ = 0;
    uint32_t last_tick_ = 0;
    bool have_tick_ = false;
    Clock::time_point last_tick_time_{};
};

enum class FaultSeverity { None, Warning, SoftStop, HardFault };

enum class TargetCommandMode { GateHold, PolicyTarget, FaultFreeze };

struct TargetCommandDecision
{
    TargetCommandMode mode = TargetCommandMode::FaultFreeze;
    bool stale_fault = false;
};

inline bool update_shadow_gate(bool ready,
                               bool suspended_test_bypass,
                               int stable_lowstate_count,
                               int stable_depth_count,
                               int shadow_frame_count,
                               int stable_lowstate_required,
                               int stable_depth_required,
                               int shadow_frames_required)
{
    return ready || suspended_test_bypass ||
           (stable_lowstate_count >= stable_lowstate_required &&
            stable_depth_count >= stable_depth_required &&
            shadow_frame_count >= shadow_frames_required);
}

inline bool initial_shadow_gate_ready(bool startup_shadow_enabled)
{
    return !startup_shadow_enabled;
}

inline bool valid_suspended_test_request(bool suspended_test,
                                         bool fixed_zero,
                                         bool armed)
{
    return !suspended_test || (fixed_zero && armed);
}

inline bool valid_ground_test_request(bool ground_test,
                                      bool suspended_test,
                                      bool fixed_zero,
                                      bool armed)
{
    return !ground_test || (!suspended_test && fixed_zero && armed);
}

inline bool valid_fixed_vx(float vx)
{
    return std::isfinite(vx) && vx >= 0.0f && vx <= 1.0f;
}

inline bool audit_post_slew_motion(bool suspended_test,
                                   bool ground_test,
                                   bool walking_test)
{
    return suspended_test || ground_test || walking_test;
}

inline bool tracking_guard_enabled(bool ground_test, bool walking_test)
{
    return !ground_test && !walking_test;
}

inline bool reject_motion_request(bool audit_post_slew_step,
                                  bool tracking_guard_enabled,
                                  bool have_previous_request,
                                  float requested_target_step_rad,
                                  float applied_target_step_rad,
                                  float tracking_error_rad,
                                  float max_target_step_rad,
                                  float max_tracking_error_rad)
{
    constexpr float kStepToleranceRad = 1.0e-5f;
    const bool step_violation = audit_post_slew_step
        ? applied_target_step_rad > max_target_step_rad + kStepToleranceRad
        : (have_previous_request && requested_target_step_rad > max_target_step_rad);
    const bool tracking_violation = tracking_guard_enabled &&
                                    tracking_error_rad > max_tracking_error_rad;
    return step_violation || tracking_violation;
}

inline TargetCommandDecision decide_target_command(bool policy_target_active,
                                                   bool allow_target_publish,
                                                   bool freeze_policy_target,
                                                   bool have_target,
                                                   float target_age_ms,
                                                   float stale_limit_ms)
{
    if (freeze_policy_target)
        return {TargetCommandMode::FaultFreeze, false};
    if (!policy_target_active)
        return {have_target ? TargetCommandMode::GateHold
                            : TargetCommandMode::FaultFreeze,
                false};
    if (!allow_target_publish || !have_target)
        return {TargetCommandMode::FaultFreeze, false};
    if (!std::isfinite(target_age_ms) || target_age_ms > stale_limit_ms)
        return {TargetCommandMode::FaultFreeze, true};
    return {TargetCommandMode::PolicyTarget, false};
}

inline float select_joint_target(TargetCommandMode mode,
                                 float held_or_policy_target,
                                 float measured_position)
{
    return mode == TargetCommandMode::FaultFreeze
               ? measured_position
               : held_or_policy_target;
}

struct SafetyState
{
    std::atomic<bool> freeze_policy_target{false};
    std::atomic<bool> takeover_requested{false};
    std::atomic<FaultSeverity> severity{FaultSeverity::None};
    mutable std::mutex mutex;
    std::string reason = "none";

    void fault(FaultSeverity level, const std::string& why)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (severity.load() == FaultSeverity::HardFault) return;
        severity = level;
        reason = why;
        if (level == FaultSeverity::SoftStop || level == FaultSeverity::HardFault) {
            freeze_policy_target = true;
            takeover_requested = true;
        }
    }

    std::string fault_reason() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return reason;
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex);
        freeze_policy_target = false;
        takeover_requested = false;
        severity = FaultSeverity::None;
        reason = "none";
    }
};

struct ActionLayers
{
    std::array<float, 12> model_raw_action{};
    std::array<float, 12> clipped_raw_action{};
    std::array<float, 12> requested_joint_target{};
    std::array<float, 12> slew_limited_joint_target{};
    std::array<float, 12> physical_limit_joint_target{};
    std::array<float, 12> applied_joint_target{};
    std::array<float, 12> executed_raw_action{};
    int modified_count = 0;
};

inline ActionLayers execute_action_chain(
    const std::vector<float>& raw,
    const std::vector<float>& previous_applied,
    const std::vector<float>& offset,
    float scale,
    float raw_clip_lo,
    float raw_clip_hi,
    float max_step,
    const std::vector<float>& lower,
    const std::vector<float>& upper)
{
    if (raw.size() != 12 || previous_applied.size() != 12 || offset.size() != 12 ||
        lower.size() != 12 || upper.size() != 12 || !std::isfinite(scale) || std::fabs(scale) < 1e-6f)
        throw std::runtime_error("invalid action-chain dimensions or scale");
    ActionLayers result;
    for (size_t i = 0; i < 12; ++i) {
        if (!std::isfinite(raw[i]) || !std::isfinite(previous_applied[i]))
            throw std::runtime_error("non-finite model_raw_action or previous target");
        result.model_raw_action[i] = raw[i];
        result.clipped_raw_action[i] = std::clamp(raw[i], raw_clip_lo, raw_clip_hi);
        result.requested_joint_target[i] = offset[i] + scale * result.clipped_raw_action[i];
        result.slew_limited_joint_target[i] = previous_applied[i] + std::clamp(
            result.requested_joint_target[i] - previous_applied[i], -max_step, max_step);
        result.physical_limit_joint_target[i] = std::clamp(
            result.slew_limited_joint_target[i], lower[i], upper[i]);
        result.applied_joint_target[i] = result.physical_limit_joint_target[i];
        result.executed_raw_action[i] = (result.applied_joint_target[i] - offset[i]) / scale;
        if (std::fabs(result.applied_joint_target[i] - result.requested_joint_target[i]) > 1e-6f)
            ++result.modified_count;
    }
    return result;
}

} // namespace strict_loco
