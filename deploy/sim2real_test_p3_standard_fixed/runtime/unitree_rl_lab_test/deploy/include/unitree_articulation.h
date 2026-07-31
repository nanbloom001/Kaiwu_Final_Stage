// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // joint positions, velocities, and estimated efforts
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            const auto& motor = lowstate->msg_.motor_state()[data.joint_ids_map[i]];
            data.joint_pos[i] = motor.q();
            data.joint_vel[i] = motor.dq();
            data.joint_effort[i] = motor.tau_est();
        }
        for(int i(0); i<4; i++) {
            data.foot_force[i] = lowstate->msg_.foot_force()[i];
            data.foot_force_est[i] = lowstate->msg_.foot_force_est()[i];
        }
    }

    LowStatePtr lowstate;
};

}
