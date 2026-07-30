#include "PolicySafety.h"
#include "EntryBlend.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
}

int main()
{
    try {
        require(std::fabs(loco_safety::max_abs_delta(
                             {0.0f, 1.0f}, {0.2f, -0.5f}) - 1.5f) < 1e-6f,
                "max_abs_delta returned the wrong value");
        require(std::isinf(loco_safety::max_abs_delta({0.0f}, {0.0f, 1.0f})),
                "size mismatch must be unsafe");
        require(std::isinf(loco_safety::max_abs_delta(
                             {std::numeric_limits<float>::quiet_NaN()}, {0.0f})),
                "non-finite input must be unsafe");

        int count = 0;
        count = loco_safety::update_violation_count(1.1f, 1.0f, count);
        require(count == 1 && !loco_safety::should_trip(count, 3),
                "first violation must not trip a three-frame guard");
        count = loco_safety::update_violation_count(1.2f, 1.0f, count);
        count = loco_safety::update_violation_count(1.3f, 1.0f, count);
        require(loco_safety::should_trip(count, 3),
                "three consecutive violations must trip");
        count = loco_safety::update_violation_count(0.9f, 1.0f, count);
        require(count == 0, "a safe sample must reset the counter");

        require(std::fabs(loco_safety::limit_target_step(0.0f, 1.0f, 0.03f) -
                          0.03f) < 1e-6f,
                "positive target step must be limited");
        require(std::fabs(loco_safety::limit_target_step(0.0f, -1.0f, 0.03f) +
                          0.03f) < 1e-6f,
                "negative target step must be limited");
        require(std::fabs(loco_safety::limit_target_step(0.0f, 0.02f, 0.03f) -
                          0.02f) < 1e-6f,
                "safe target step must remain unchanged");

        const auto normal_envelope = loco_safety::runtime_action_envelope(
            false, true, 6.0f, 1.0f, 0.03f);
        require(normal_envelope.max_raw_action_abs == 6.0f &&
                    normal_envelope.max_raw_action_step == 1.0f &&
                    normal_envelope.max_target_step_rad == 0.03f &&
                    !normal_envelope.bypass_entry_blend,
                "normal runtime must preserve the configured action envelope");
        const auto parity_envelope = loco_safety::runtime_action_envelope(
            true, true, 6.0f, 1.0f, 0.03f);
        require(parity_envelope.max_raw_action_abs == 8.0f &&
                    parity_envelope.max_raw_action_step == 8.0f &&
                    parity_envelope.max_target_step_rad == 1.0f &&
                    parity_envelope.bypass_entry_blend,
                "suspended parity must use the bounded historical envelope");
        const auto parity_stand_envelope = loco_safety::runtime_action_envelope(
            true, false, 6.0f, 1.0f, 0.03f);
        require(parity_stand_envelope.max_raw_action_abs == 6.0f &&
                    parity_stand_envelope.max_raw_action_step == 1.0f &&
                    parity_stand_envelope.max_target_step_rad == 0.03f &&
                    !parity_stand_envelope.bypass_entry_blend,
                "suspended parity stand and forced-zero return must retain the normal envelope");
        require(std::fabs(loco_safety::policy_transition_target(
                             0.0f, 1.0f, 0.1f, 1.0f, false) - 0.028f) < 1e-6f,
                "normal transition must retain the smooth entry blend");
        require(loco_safety::policy_transition_target(
                    0.0f, 1.0f, 0.1f, 1.0f, true) == 1.0f,
                "suspended parity must bypass the entry blend");

        require(loco_safety::command_is_zero({0.02f, -0.02f, 0.0f}, 0.02f),
                "commands on the hold threshold must be treated as zero");
        require(!loco_safety::command_is_zero({0.021f, 0.0f, 0.0f}, 0.02f),
                "a command above the hold threshold must activate the policy");
        require(!loco_safety::command_is_zero(
                    {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f}, 0.02f),
                "a non-finite command must not enter stand hold");

        require(!loco_safety::command_duration_expired(1.99f, 2.0f),
                "command duration must remain active before the hard limit");
        require(loco_safety::command_duration_expired(2.0f, 2.0f),
                "command duration must expire on the hard limit");
        require(!loco_safety::command_duration_expired(100.0f, 0.0f),
                "a zero hard limit must disable the timeout");

        const auto bounded = loco_safety::first_bounded_value_violation(
            {0.0f, -6.1f, 1.0f}, 6.0f);
        require(bounded.found && bounded.index == 1 && !bounded.non_finite &&
                    std::fabs(bounded.value + 6.1f) < 1e-6f,
                "bounded output diagnostics must identify the first over-limit value");
        const auto non_finite = loco_safety::first_bounded_value_violation(
            {0.0f, std::numeric_limits<float>::infinity()}, 6.0f);
        require(non_finite.found && non_finite.index == 1 && non_finite.non_finite,
                "bounded output diagnostics must identify non-finite values");

        const std::vector<float> per_joint_limits{1.0f, 2.0f};
        require(!loco_safety::first_per_joint_abs_violation(
                     {0.5f, -2.0f}, per_joint_limits).found,
                "values on a per-joint limit must pass");
        const auto per_joint_abs = loco_safety::first_per_joint_abs_violation(
            {0.5f, -2.1f}, per_joint_limits);
        require(per_joint_abs.found && per_joint_abs.index == 1 &&
                    per_joint_abs.value == -2.1f && per_joint_abs.limit == 2.0f,
                "per-joint absolute guard must identify the exact joint");
        const auto per_joint_delta = loco_safety::first_per_joint_delta_violation(
            {0.6f, -1.0f}, {0.0f, 0.5f}, {0.5f, 2.0f});
        require(per_joint_delta.found && per_joint_delta.index == 0 &&
                    std::fabs(per_joint_delta.value - 0.6f) < 1e-6f,
                "per-joint delta guard must compare with the supplied previous action");
        const auto per_joint_range = loco_safety::first_per_joint_range_violation(
            {0.0f, 2.1f}, {-1.0f, -2.0f}, {1.0f, 2.0f});
        require(per_joint_range.found && per_joint_range.index == 1 &&
                    per_joint_range.limit == 2.0f,
                "per-joint range guard must identify an upper-bound violation");

        std::vector<int> effort_counts(2, 0);
        require(!loco_safety::update_effort_violation_counts(
                     {12.1f, 0.0f}, 12.0f, 3, effort_counts).found,
                "the first effort violation must not trip a three-frame guard");
        require(!loco_safety::update_effort_violation_counts(
                     {12.2f, 0.0f}, 12.0f, 3, effort_counts).found,
                "the second effort violation must not trip a three-frame guard");
        const auto effort_trip = loco_safety::update_effort_violation_counts(
            {12.3f, 0.0f}, 12.0f, 3, effort_counts);
        require(effort_trip.found && effort_trip.index == 0 &&
                    effort_trip.value == 12.3f,
                "the third consecutive effort violation must trip");
        require(!loco_safety::update_effort_violation_counts(
                     {0.0f, 0.0f}, 12.0f, 3, effort_counts).found &&
                    effort_counts[0] == 0,
                "a safe effort frame must reset its joint counter");

        const std::vector<float> stand_action{0.4f, -0.2f};
        std::vector<float> last_action = stand_action;
        require(loco_safety::reset_last_action_on_mode_transition(
                    false, true, stand_action, last_action),
                "entering motion must report a mode transition");
        require(last_action == std::vector<float>({0.0f, 0.0f}),
                "the first motion observation must receive zero last_action");
        require(loco_safety::reset_last_action_on_mode_transition(
                    true, false, stand_action, last_action),
                "returning to stand must report a mode transition");
        require(last_action == stand_action,
                "stand hold must restore the stand-derived last_action");

        const float support_target = loco_safety::torque_preserving_target(
            1.0f, 0.2f, 1.05f, 80.0f, 3.0f, 25.0f, 0.5f, 0.20f);
        const float previous_torque = 80.0f * (1.05f - 1.0f) - 3.0f * 0.2f;
        const float next_torque = 25.0f * (support_target - 1.0f) - 0.5f * 0.2f;
        require(std::fabs(previous_torque - next_torque) < 1e-5f,
                "ready target must preserve the estimated PD support torque");
        require(std::fabs(loco_safety::torque_preserving_target(
                             0.0f, 0.0f, 1.0f, 80.0f, 0.0f,
                             25.0f, 0.5f, 0.20f) - 0.20f) < 1e-6f,
                "ready target compensation must respect its offset limit");
        require(loco_safety::torque_preserving_target(
                    0.4f, 0.0f, std::numeric_limits<float>::quiet_NaN(),
                    80.0f, 3.0f, 25.0f, 0.5f, 0.20f) == 0.4f,
                "invalid entry PD state must fall back to measured position");
    } catch (const std::exception& error) {
        std::cerr << "policy safety test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Policy safety guards passed.\n";
    return 0;
}
