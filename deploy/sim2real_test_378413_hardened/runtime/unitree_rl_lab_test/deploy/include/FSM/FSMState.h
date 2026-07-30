#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

#include <algorithm>
#include <cmath>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        // Run the orientation guard before ordinary controller transitions so
        // a fall cannot be routed through a high-stiffness state.
        auto global_safety = param::config["FSM"]["safety"];
        if (global_safety && global_safety["max_tilt_deg"])
        {
            const float max_tilt_deg = global_safety["max_tilt_deg"].as<float>();
            if (!std::isfinite(max_tilt_deg) || max_tilt_deg < 10.0f ||
                max_tilt_deg > 25.0f)
            {
                throw std::runtime_error(
                    "FSM.safety.max_tilt_deg must be in [10, 25]");
            }
            const float cos_limit = std::cos(max_tilt_deg * 3.1415926535f / 180.0f);
            registered_checks.emplace_back(std::make_pair(
                [state_string, cos_limit, reported = false]() mutable -> bool {
                    const auto& q = lowstate->msg_.imu_state().quaternion();
                    const float w = q[0];
                    const float x = q[1];
                    const float y = q[2];
                    const float z = q[3];
                    const float norm_sq = w * w + x * x + y * y + z * z;
                    const bool finite = std::isfinite(norm_sq) && norm_sq > 1.0e-6f;
                    const float upright_z = finite
                        ? std::clamp(1.0f - 2.0f * (x * x + y * y) / norm_sq,
                                     -1.0f, 1.0f)
                        : -1.0f;
                    const bool unsafe = !finite || upright_z < cos_limit;
                    if (unsafe && state_string != "Passive" && !reported)
                    {
                        spdlog::critical(
                            "FSM global tilt guard tripped in {}; transitioning to Passive",
                            state_string);
                        reported = true;
                    }
                    if (!unsafe) reported = false;
                    return unsafe;
                },
                FSMStringMap.right.at("Passive")
            ));
        }

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};
