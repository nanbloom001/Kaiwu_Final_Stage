#include "StrictSafety.h"
#include "DepthSource.h"

#include <atomic>
#include <cassert>
#include <cmath>
#include <thread>
#include <vector>

int main()
{
    using namespace strict_loco;
    assert(validate_quaternion_wxyz({1.0f, 0.0f, 0.0f, 0.0f}));
    assert(!validate_quaternion_wxyz({0.0f, 0.0f, 0.0f, 0.0f}));
    assert(!validate_quaternion_wxyz({NAN, 0.0f, 0.0f, 0.0f}));

    const std::vector<float> raw{10.f, -10.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const std::vector<float> offset{0.1f,-0.1f,0.1f,-0.1f,0.8f,0.8f,1.f,1.f,-1.5f,-1.5f,-1.5f,-1.5f};
    const std::vector<float> lower{-0.9972f,-0.9972f,-0.9972f,-0.9972f,-1.5208f,-1.5208f,-0.4736f,-0.4736f,-2.6727f,-2.6727f,-2.6727f,-2.6727f};
    const std::vector<float> upper{0.9972f,0.9972f,0.9972f,0.9972f,3.4407f,3.4407f,4.4879f,4.4879f,-0.88776f,-0.88776f,-0.88776f,-0.88776f};
    const auto result = execute_action_chain(raw, offset, offset, 0.25f, -6.f, 6.f, 0.05f, lower, upper);
    assert(result.clipped_raw_action[0] == 6.f);
    assert(result.clipped_raw_action[1] == -6.f);
    assert(std::fabs(result.applied_joint_target[0] - offset[0]) <= 0.050001f);
    assert(std::fabs(result.executed_raw_action[0] -
                     (result.applied_joint_target[0] - offset[0]) / 0.25f) < 1e-6f);

    // The startup gate may legitimately take longer than the live target
    // watchdog. It must hold the entry pose without reporting a stale target.
    const auto gate_hold = decide_target_command(
        false, false, false, true, 2500.0f, 100.0f);
    assert(gate_hold.mode == TargetCommandMode::GateHold);
    assert(!gate_hold.stale_fault);
    assert(select_joint_target(gate_hold.mode, 0.8f, 0.6f) == 0.8f);

    const auto live_target = decide_target_command(
        true, true, false, true, 20.0f, 100.0f);
    assert(live_target.mode == TargetCommandMode::PolicyTarget);
    assert(!live_target.stale_fault);

    const auto stale_target = decide_target_command(
        true, true, false, true, 101.0f, 100.0f);
    assert(stale_target.mode == TargetCommandMode::FaultFreeze);
    assert(stale_target.stale_fault);

    const auto fault_freeze = decide_target_command(
        true, false, true, true, 20.0f, 100.0f);
    assert(fault_freeze.mode == TargetCommandMode::FaultFreeze);
    assert(!fault_freeze.stale_fault);
    assert(select_joint_target(fault_freeze.mode, 0.8f, 0.6f) == 0.6f);

    assert(!update_shadow_gate(false, false, 99, 60, 100, 100, 60, 100));
    assert(update_shadow_gate(false, false, 100, 60, 100, 100, 60, 100));
    assert(update_shadow_gate(false, true, 0, 0, 1, 100, 60, 100));
    assert(initial_shadow_gate_ready(false));
    assert(!initial_shadow_gate_ready(true));
    // Once locomotion starts, velocity is expected to exceed the stationary
    // threshold. A passed startup gate must therefore stay latched.
    assert(update_shadow_gate(true, false, 0, 0, 101, 100, 60, 100));

    assert(valid_suspended_test_request(false, false, false));
    assert(valid_suspended_test_request(true, true, true));
    assert(!valid_suspended_test_request(true, false, true));
    assert(!valid_suspended_test_request(true, true, false));
    assert(valid_ground_test_request(false, false, false, false));
    assert(valid_ground_test_request(true, false, true, true));
    assert(!valid_ground_test_request(true, true, true, true));
    assert(!valid_ground_test_request(true, false, false, true));
    assert(!valid_ground_test_request(true, false, true, false));

    assert(valid_fixed_vx(0.0f));
    assert(valid_fixed_vx(0.3f));
    assert(valid_fixed_vx(1.0f));
    assert(!valid_fixed_vx(-0.001f));
    assert(!valid_fixed_vx(1.001f));
    assert(!valid_fixed_vx(std::numeric_limits<float>::infinity()));

    assert(!audit_post_slew_motion(false, false, false));
    assert(audit_post_slew_motion(true, false, false));
    assert(audit_post_slew_motion(false, true, false));
    assert(audit_post_slew_motion(false, false, true));
    assert(tracking_guard_enabled(false, false));
    assert(!tracking_guard_enabled(true, false));
    assert(!tracking_guard_enabled(false, true));

    // Normal mode retains the conservative pre-slew request audit.
    assert(reject_motion_request(
        false, true, true, 0.129f, 0.05f, 0.05f, 0.05f, 0.45f));
    // Suspended mode audits what can actually reach the motors after slew.
    assert(!reject_motion_request(
        true, true, true, 0.129f, 0.05f, 0.05f, 0.05f, 0.45f));
    assert(reject_motion_request(
        true, true, true, 0.129f, 0.051f, 0.05f, 0.05f, 0.45f));
    assert(reject_motion_request(
        true, true, true, 0.01f, 0.01f, 0.451f, 0.05f, 0.45f));
    // Ground characterization keeps post-slew step protection but intentionally
    // disables only the tracking-error soft gate.
    assert(!reject_motion_request(
        true, false, true, 0.129f, 0.05f, 0.60f, 0.05f, 0.45f));
    assert(reject_motion_request(
        true, false, true, 0.129f, 0.051f, 0.60f, 0.05f, 0.45f));
    // Fixed-vx walking uses the same minimum motion guard as the validated
    // ground characterization, while retaining the startup stability gate.
    assert(!reject_motion_request(
        audit_post_slew_motion(false, false, true),
        tracking_guard_enabled(false, true),
        true, 0.50f, 0.05f, 0.65f, 0.05f, 0.45f));

    // With identical source/target intrinsics, reprojection must preserve every
    // pixel and apply exactly the training normalization/invalid convention.
    vision_nav::CamIntrin intrinsics{
        vision_nav::SIM_FX, vision_nav::SIM_FY,
        vision_nav::SIM_CX, vision_nav::SIM_CY,
        vision_nav::DEPTH_W, vision_nav::DEPTH_H};
    std::vector<float> source(vision_nav::DEPTH_W * vision_nav::DEPTH_H, 2.5f);
    source[0] = 0.0f;
    source[1] = 5.0f;
    std::vector<float> projected;
    vision_nav::reproject_normalize_into(source.data(), intrinsics, projected);
    assert(projected.size() == source.size());
    assert(projected[0] == 0.0f);
    assert(projected[1] == 0.0f);
    assert(std::fabs(projected[2] - 0.5f) < 1e-6f);

    SafetyState state;
    std::atomic<int> completed{0};
    std::vector<std::thread> workers;
    for (int thread = 0; thread < 8; ++thread) {
        workers.emplace_back([&] {
            for (int i = 0; i < 10000; ++i) {
                const auto local = execute_action_chain(raw, offset, offset, 0.25f, -6.f, 6.f,
                                                        0.05f, lower, upper);
                assert(std::isfinite(local.executed_raw_action[0]));
                if ((i % 997) == 0) state.fault(FaultSeverity::Warning, "stress_warning");
                (void)state.fault_reason();
            }
            ++completed;
        });
    }
    for (auto& worker : workers) worker.join();
    assert(completed == 8);
    return 0;
}
