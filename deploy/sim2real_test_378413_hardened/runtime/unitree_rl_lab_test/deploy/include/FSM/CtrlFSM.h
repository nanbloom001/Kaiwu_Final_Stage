// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }
    }

    void start(const std::string& initial_state_name)
    {
        if (started_) throw std::runtime_error("FSM: already started");
        currentState.reset();
        for (const auto& state : states)
        {
            if (state->getStateString() == initial_state_name)
            {
                currentState = state;
                break;
            }
        }
        if (!currentState)
        {
            throw std::runtime_error(
                "FSM: configured initial state is not enabled: " + initial_state_name);
        }
        currentState->enter();

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        started_ = true;
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void start()
    {
        if (states.empty()) throw std::runtime_error("FSM: no states are enabled");
        start(states.front()->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }
    
    void stop()
    {
        if (!started_) return;

        // Join the recurrent thread before touching currentState. In
        // particular, VisionLoco::exit() joins its policy thread and destroys
        // Keyboard, which restores the terminal settings.
        fsm_thread_.reset();

        if (currentState && currentState->getStateString() != "Passive")
        {
            spdlog::warn("FSM: Graceful stop from {} to Passive",
                         currentState->getStateString());
            currentState->exit();
            for (const auto& state : states)
            {
                if (state->getStateString() == "Passive")
                {
                    currentState = state;
                    break;
                }
            }
        }

        if (currentState && currentState->getStateString() == "Passive")
        {
            currentState->enter();
            // Publish several damping-only frames before the process exits.
            for (int i = 0; i < 20; ++i)
            {
                currentState->pre_run();
                currentState->run();
                currentState->post_run();
                usleep(1000);
            }
        }
        started_ = false;
    }

    ~CtrlFSM()
    {
        stop();
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();
        
        // Check if need to change state
        int nextStateMode = 0;
        for(int i(0); i<currentState->registered_checks.size(); i++)
        {
            if(currentState->registered_checks[i].first())
            {
                nextStateMode = currentState->registered_checks[i].second;
                break;
            }
        }

        if(nextStateMode != 0 && !currentState->isState(nextStateMode))
        {
            for(auto & state : states)
            {
                if(state->isState(nextStateMode))
                {
                    spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
                    currentState->exit();
                    currentState = state;
                    currentState->enter();
                    break;
                }
            }
        }
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
    bool started_ = false;
};
