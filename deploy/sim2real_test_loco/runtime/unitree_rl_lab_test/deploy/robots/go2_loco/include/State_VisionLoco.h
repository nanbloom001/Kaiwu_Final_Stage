// Copyright (c) 2026.
// State_VisionLoco — Go2 lbc_loco 阶段部署 FSM 策略态（视觉运控 loco）。
//
// 与 State_VisionNav 结构完全一致，仅把推理核换成 LocoRunner（loco 阶段模型无 nav
// 分支）。obs 装配（proprio 45）、深度预处理、UWB 订阅与 command_for_frame 的 uwb
// 分支、诊断 CSV **全部保留**，与 vision-nav 部署链路一致，便于同一套现场流程复用。
//
// 不走官方 ManagerBasedRLEnv（其 obs/action 管理器假设单输出、标准 obs 项），
// 而是自建 proprio(45) + depth + goal(4)，调 LocoRunner（8 进 8 出 + LSTM 状态；
// nav 端口占位不参与计算），把输出 joint(12) 经 action scale/offset/clip 处理为
// 关节目标，按 joint_ids_map 下发 PD。
//
// 复用官方设施：BaseArticulation（IMU/关节读取，已按策略序映射）、FSMState（lowcmd/lowstate/
// 手柄 DSL 转移）、bad_orientation 安全检查（翻倒→Passive）。
//
// 部署目录约定（与 State_VisionNav 一致）：
//   <policy_dir>/exported/policy.onnx   ← export_loco_onnx.py 产出
//   <policy_dir>/params/deploy.yaml

#pragma once

#include "FSM/FSMState.h"
#include "unitree_articulation.h"
#include "isaaclab/algorithms/loco_runner.h"
#include "DepthSource.h"
#include "UwbGoal.h"

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/UwbState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/utrack/utrack_client.hpp>

#include <atomic>
#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class State_VisionLoco : public FSMState
{
public:
    State_VisionLoco(int state_mode, std::string state_string);
    ~State_VisionLoco();

    void enter() override;
    void run() override;
    void exit() override;

private:
    void policy_loop();
    std::vector<float> build_proprio();   // 45, 策略序
    void on_uwb(const void* message);
    void on_sport_state(const void* message);
    std::array<float, 3> apply_stuck_recovery(
        const std::array<float, 3>& requested,
        const std::chrono::steady_clock::time_point& now,
        float measured_wz);
    std::array<float, 3> command_for_frame(
        const std::chrono::steady_clock::time_point& now,
        std::vector<float>& goal,
        std::array<float, 3>& theory_cmd,
        bool& uwb_valid,
        float& uwb_age_s,
        float& closing_speed,
        int& uwb_error,
        int& uwb_enabled,
        int& uwb_channel);

    // ---- 来自 deploy.yaml ----
    std::vector<int>   joint_ids_map_;     // 策略序 i → SDK 电机下标
    std::vector<float> default_joint_pos_; // 策略序
    std::vector<float> stiffness_, damping_;
    std::vector<float> act_offset_;        // 动作 offset（= default_joint_pos）
    float act_scale_   = 0.25f;
    float act_clip_lo_ = -100.f, act_clip_hi_ = 100.f;
    float proprio_clip_ = 100.f;
    float sc_ang_vel_ = 0.25f, sc_proj_g_ = 1.0f, sc_velcmd_ = 1.0f,
          sc_jpos_ = 1.0f, sc_jvel_ = 0.05f, sc_lastact_ = 1.0f;
    float step_dt_ = 0.02f;

    // ---- 来自 config.yaml ----
    std::vector<float> goal_;              // 4 (orientation,pitch,distance,yaw)
    std::string command_source_ = "fixed"; // fixed / uwb / nav
    std::array<float, 3> fixed_cmd_{0.15f, 0.0f, 0.0f};
    std::array<float, 3> last_exec_cmd_{0.0f, 0.0f, 0.0f};
    float max_vx_ = 0.30f, max_vy_ = 0.0f, max_wz_ = 0.40f;
    float yaw_kp_ = 0.8f;
    float stop_distance_ = 0.35f, slow_distance_ = 1.0f, stop_hysteresis_ = 0.15f;
    bool uwb_stop_once_ = false;
    bool uwb_goal_driven_ = true;
    std::array<float, 3> uwb_cruise_cmd_{0.30f, 0.0f, 0.0f};
    float turn_in_place_angle_ = 1.0472f;
    float uwb_stale_timeout_s_ = 0.5f;
    float uwb_hold_timeout_s_ = 1.5f;
    float uwb_buffer_seconds_ = 6.0f;
    float uwb_median_window_s_ = 0.45f;
    float uwb_filter_tau_s_ = 0.25f;
    std::array<float, 3> cmd_slew_rate_{0.30f, 0.30f, 1.0f};
    std::string uwb_topic_ = "rt/uwbstate";
    std::string sport_topic_ = "rt/sportmodestate";
    bool uwb_diagnostic_feedback_ = true;
    float uwb_velocity_alpha_ = 0.20f;
    bool stuck_recovery_enabled_ = true;

    // ---- 逐帧诊断日志（CSV），进 VisionLoco 态自动开 ----
    bool logging_enabled_ = true;
    int log_flush_every_ = 50;
    int max_consecutive_errors_ = 5;
    float max_raw_action_abs_ = 20.0f;
    std::string log_dir_;
    std::ofstream diag_;
    std::string   diag_path_;

    // ---- 运行期 ----
    std::shared_ptr<unitree::BaseArticulation<LowState_t::SharedPtr>> robot_;
    std::unique_ptr<isaaclab::LocoRunner>       runner_;
    std::unique_ptr<vision_nav::DepthSource>    depth_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::UwbState_> uwb_sub_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> sport_sub_;
    std::unique_ptr<unitree::robot::go2::UtrackClient> utrack_;

    struct UwbSample
    {
        // Raw SDK order: beta/orientation, pitch, spatial distance (m), yaw.
        std::array<float, 4> raw{0.0f, 0.0f, 0.0f, 0.0f};
        int error_state = 255;
        int enabled_from_app = 0;
        int channel = -1;
        bool received = false;
        std::chrono::steady_clock::time_point received_at{};
        bool filtered_received = false;
        std::chrono::steady_clock::time_point filtered_received_at{};
        float filtered_x = 0.0f;
        float filtered_y = 0.0f;
        float filtered_planar_distance = 0.0f;
        float filtered_beta = 0.0f;
        float closing_speed = 0.0f;
        float base_yaw = 0.0f;
        std::array<float, 2> body_velocity{0.0f, 0.0f};
        bool velocity_valid = false;
        std::chrono::steady_clock::time_point velocity_received_at{};
    };
    struct UwbFilterFrame
    {
        std::chrono::steady_clock::time_point received_at{};
        float x = 0.0f;
        float y = 0.0f;
    };
    struct SportSample
    {
        std::array<float, 3> velocity{0.0f, 0.0f, 0.0f};
        float yaw_speed = 0.0f;
        bool received = false;
        std::chrono::steady_clock::time_point received_at{};
    };
    std::mutex sensor_mtx_;
    UwbSample uwb_;
    std::deque<UwbFilterFrame> uwb_filter_queue_;
    vision_nav::JumpRejectState uwb_jump_state_;   // UWB跳变拒绝状态
    SportSample sport_;
    float last_uwb_planar_distance_ = 0.0f;
    float last_uwb_filtered_x_ = 0.0f;
    float last_uwb_filtered_y_ = 0.0f;
    float last_uwb_filtered_beta_ = 0.0f;
    float last_uwb_filter_age_s_ = -1.0f;
    float last_uwb_freshness_scale_ = 0.0f;
    bool last_uwb_hold_ = false;
    bool uwb_arrived_ = false;

    std::mutex tgt_mtx_;
    std::vector<float> joint_target_;      // 12，已处理（offset+scale*raw，clip），策略序
    std::vector<float> last_action_raw_;   // 12，原始 joint，回喂 proprio[33:45]
    bool have_target_ = false;

    std::thread policy_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> policy_fault_{false};
    int consecutive_errors_ = 0;
    long deadline_misses_ = 0;
    long frame_ = 0;

    // Runtime-only safety override. It never changes the ONNX contract.
    enum class RecoveryPhase { idle, braking, lateral, settling };
    RecoveryPhase recovery_phase_ = RecoveryPhase::idle;
    int recovery_direction_ = 1;
    int last_recovery_direction_ = -1;
    float recovery_progress_m_ = 0.0f;
    float recovery_effective_s_ = 0.0f;
    float recovery_low_speed_s_ = 0.0f;
    float recovery_total_s_ = 0.0f;
    float stuck_window_s_ = 0.0f;
    float stuck_path_m_ = 0.0f;
    float stuck_low_speed_s_ = 0.0f;
    float recent_turn_integral_rad_ = 0.0f;
    float recent_turn_window_s_ = 0.0f;
    std::chrono::steady_clock::time_point watchdog_last_tick_{};
    std::chrono::steady_clock::time_point recovery_phase_started_{};
};

REGISTER_FSM(State_VisionLoco)
