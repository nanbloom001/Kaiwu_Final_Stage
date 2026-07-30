#include "isaaclab/algorithms/loco_runner.h"
#include "EntryBlend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_finite(const std::vector<float>& values, const std::string& label)
{
    for (float value : values)
        require(std::isfinite(value), label + " contains a non-finite value");
}
} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "usage: test_loco_runner <policy.onnx>");
        isaaclab::LocoRunner runner(argv[1]);
        runner.set_cmd_override(true, 0.3f, 0.0f, 0.2f);

        std::vector<float> depth(isaaclab::LocoRunner::DEPTH_N, 0.5f);
        std::vector<float> proprio(isaaclab::LocoRunner::PROPRIO_DIM, 0.0f);
        std::vector<float> goal(isaaclab::LocoRunner::GOAL_DIM, 0.0f);

        for (int frame = 0; frame < 2; ++frame) {
            const auto output = runner.act(depth, proprio, goal);
            require(output.joint.size() == isaaclab::LocoRunner::NUM_ACTIONS,
                    "joint output size mismatch");
            require(output.cmd.size() == isaaclab::LocoRunner::NUM_CMD,
                    "cmd output size mismatch");
            require(output.clearance.size() == isaaclab::LocoRunner::NUM_CLR,
                    "clearance output size mismatch");
            require_finite(output.joint, "joint");
            require_finite(output.cmd, "cmd");
            require(std::fabs(output.cmd[0] - 0.3f) < 1.0e-6f,
                    "vx command echo mismatch");
            require(std::fabs(output.cmd[1]) < 1.0e-6f,
                    "vy command echo mismatch");
            require(std::fabs(output.cmd[2] - 0.2f) < 1.0e-6f,
                    "wz command echo mismatch");
        }

        isaaclab::LocoRunner standing_runner(argv[1]);
        standing_runner.set_cmd_override(true, 0.0f, 0.0f, 0.0f);
        std::fill(depth.begin(), depth.end(), 0.6f);
        std::fill(proprio.begin(), proprio.end(), 0.0f);
        proprio[5] = -1.0f;

        constexpr float action_scale = 0.25f;
        float max_initial_target_delta = 0.0f;
        float max_target_step = 0.0f;
        float max_blended_target_step = 0.0f;
        std::vector<float> previous_action(isaaclab::LocoRunner::NUM_ACTIONS, 0.0f);
        std::vector<float> previous_blended_target(
            isaaclab::LocoRunner::NUM_ACTIONS, 0.0f);
        for (int frame = 0; frame < 64; ++frame) {
            const auto output = standing_runner.act(depth, proprio, goal);
            require_finite(output.joint, "standing joint");
            for (int joint = 0; joint < isaaclab::LocoRunner::NUM_ACTIONS; ++joint) {
                const float initial_delta = action_scale * std::fabs(output.joint[joint]);
                const float target_step = action_scale *
                    std::fabs(output.joint[joint] - previous_action[joint]);
                if (frame == 0)
                    max_initial_target_delta =
                        std::max(max_initial_target_delta, initial_delta);
                else
                    max_target_step = std::max(max_target_step, target_step);
                const float policy_target = action_scale * output.joint[joint];
                const float blended_target = loco_safety::entry_blend_target(
                    0.0f, policy_target, (frame + 1) * 0.02f, 1.0f);
                max_blended_target_step = std::max(
                    max_blended_target_step,
                    std::fabs(blended_target - previous_blended_target[joint]));
                previous_blended_target[joint] = blended_target;
                proprio[33 + joint] = output.joint[joint];
            }
            previous_action = output.joint;
        }
        std::cout << "standing max initial target delta rad="
                  << max_initial_target_delta
                  << ", max recurrent target step rad=" << max_target_step
                  << ", max blended target step rad=" << max_blended_target_step
                  << '\n';
        require(max_target_step <= 0.35f,
                "standing recurrent joint target step exceeds 0.35 rad");
        require(max_blended_target_step <= 0.05f,
                "standing blended joint target step exceeds 0.05 rad");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
