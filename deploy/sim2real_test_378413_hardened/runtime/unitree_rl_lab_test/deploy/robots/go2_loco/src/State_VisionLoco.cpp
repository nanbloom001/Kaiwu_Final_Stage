// Copyright (c) 2026.
// State_VisionLoco 实现，见同名头文件说明。
//
// 与 State_VisionNav.cpp 逐行一致，仅把推理核换成 LocoRunner / LocoOutput，
// 日志标签改为 [VisionLoco]。obs 装配、UWB、command_for_frame、诊断 CSV 全保留。

#include "State_VisionLoco.h"
#include "EntryBlend.h"
#include "PolicySafety.h"
#include "UwbGoal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace
{
// 安全检查：翻倒（与 isaaclab::mdp::bad_orientation 同式，基于 body 系投影重力 z 分量）
inline bool bad_orientation(const Eigen::Vector3f& proj_g, float limit_angle)
{
    const float z = std::clamp(-proj_g[2], -1.0f, 1.0f);
    return std::fabs(std::acos(z)) > limit_angle;
}

template <typename T>
T yaml_get(const YAML::Node& n, const char* key, T fallback)
{
    try { if (n[key]) return n[key].as<T>(); } catch (...) {}
    return fallback;
}

template <typename Container>
void write_values(std::ofstream& out, const Container& values)
{
    for (const auto& value : values) out << ',' << value;
}

float median(std::vector<float> values)
{
    if (values.empty()) return 0.0f;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const float upper = values[middle];
    if ((values.size() % 2) != 0) return upper;
    std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
    return 0.5f * (values[middle - 1] + upper);
}

// 深度统计（归一化 [0,1]，0=无效/远；front box=中下方中央 96x72）
struct DepthStats { float invalid_frac, mean_valid, front_invalid, front_min, front_mean; };
inline DepthStats depth_stats(const std::vector<float>& d)
{
    const int H = 180, W = 320;
    long n_inval = 0, n_val = 0; double sum = 0;
    for (float v : d) { if (v <= 0.f) ++n_inval; else { sum += v; ++n_val; } }
    long fn = 0, finval = 0, fval = 0; double fsum = 0; float fmin = 1e9f;
    for (int y = 72; y < 144; ++y)
        for (int x = 112; x < 208; ++x) {
            float v = d[(size_t)y * W + x]; ++fn;
            if (v <= 0.f) ++finval; else { fsum += v; ++fval; if (v < fmin) fmin = v; }
        }
    (void)H;
    DepthStats r;
    r.invalid_frac  = d.empty() ? 0.f : (float)n_inval / d.size();
    r.mean_valid    = n_val ? (float)(sum / n_val) : 0.f;
    r.front_invalid = fn ? (float)finval / fn : 0.f;
    r.front_min     = fval ? fmin : 0.f;
    r.front_mean    = fval ? (float)(fsum / fval) : 0.f;
    return r;
}
} // namespace

State_VisionLoco::State_VisionLoco(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // ---------- 读 deploy.yaml ----------
    auto dy = YAML::LoadFile((policy_dir / "params" / "deploy.yaml").string());

    joint_ids_map_     = dy["joint_ids_map"].as<std::vector<int>>();
    default_joint_pos_ = dy["default_joint_pos"].as<std::vector<float>>();
    stiffness_         = dy["stiffness"].as<std::vector<float>>();
    damping_           = dy["damping"].as<std::vector<float>>();
    step_dt_           = dy["step_dt"].as<float>();
    if (joint_ids_map_.size() != 12 || default_joint_pos_.size() != 12 ||
        stiffness_.size() != 12 || damping_.size() != 12) {
        throw std::runtime_error("deploy.yaml 的 Go2 关节数组必须全部为 12 维");
    }
    std::vector<int> ids = joint_ids_map_;
    std::sort(ids.begin(), ids.end());
    for (int i = 0; i < 12; ++i)
        if (ids[i] != i) throw std::runtime_error("joint_ids_map 必须是 0..11 的排列");

    // 动作 term
    auto ja = dy["actions"]["JointPositionAction"];
    try { act_scale_ = ja["scale"].as<float>(); }
    catch (...) { try { act_scale_ = ja["scale"][0].as<float>(); } catch (...) {} }
    try { act_offset_ = ja["offset"].as<std::vector<float>>(); }
    catch (...) { act_offset_ = default_joint_pos_; }
    if (act_offset_.size() != 12) act_offset_ = default_joint_pos_;
    try {
        auto c = ja["clip"].as<std::vector<float>>();
        if (c.size() >= 2) { act_clip_lo_ = c[0]; act_clip_hi_ = c[1]; }
    } catch (...) {}
    if (!std::isfinite(act_scale_) || std::fabs(act_scale_) < 1.0e-6f)
        throw std::runtime_error("deploy.yaml action scale must be finite and non-zero");

    // proprio 缩放
    auto ps = dy["proprio_scales"];
    if (ps) {
        sc_ang_vel_ = yaml_get(ps, "ang_vel", sc_ang_vel_);
        sc_proj_g_  = yaml_get(ps, "projected_gravity", sc_proj_g_);
        sc_velcmd_  = yaml_get(ps, "vel_cmd", sc_velcmd_);
        sc_jpos_    = yaml_get(ps, "joint_pos_rel", sc_jpos_);
        sc_jvel_    = yaml_get(ps, "joint_vel_rel", sc_jvel_);
        sc_lastact_ = yaml_get(ps, "last_action", sc_lastact_);
    }
    try {
        auto pc = dy["proprio_clip"].as<std::vector<float>>();
        if (pc.size() >= 2) proprio_clip_ = std::max(std::fabs(pc[0]), std::fabs(pc[1]));
    } catch (...) {}

    // ---------- goal / 外部命令源（config.yaml）----------
    goal_ = {0.0f, 0.0f, 2.0f, 0.0f};
    try { if (cfg["goal"]) goal_ = cfg["goal"].as<std::vector<float>>(); } catch (...) {}
    if (goal_.size() != 4) goal_ = {0.0f, 0.0f, 2.0f, 0.0f};
    entry_blend_s_ = yaml_get<float>(cfg, "entry_blend_s", entry_blend_s_);
    if (!std::isfinite(entry_blend_s_) ||
        entry_blend_s_ < 0.5f || entry_blend_s_ > 3.0f) {
        throw std::runtime_error("VisionLoco entry_blend_s must be in [0.5, 3.0]");
    }
    if (cfg["ready_stand"]) {
        auto ready = cfg["ready_stand"];
        ready_gain_blend_s_ = yaml_get<float>(
            ready, "gain_blend_s", ready_gain_blend_s_);
        ready_min_stable_s_ = yaml_get<float>(
            ready, "min_stable_s", ready_min_stable_s_);
        ready_max_wait_s_ = yaml_get<float>(
            ready, "max_wait_s", ready_max_wait_s_);
        ready_max_joint_velocity_ = yaml_get<float>(
            ready, "max_joint_velocity", ready_max_joint_velocity_);
        ready_max_tracking_error_ = yaml_get<float>(
            ready, "max_tracking_error_rad", ready_max_tracking_error_);
        ready_max_support_target_offset_ = yaml_get<float>(
            ready, "max_support_target_offset_rad",
            ready_max_support_target_offset_);
    }
    if (!std::isfinite(ready_gain_blend_s_) || ready_gain_blend_s_ < 0.5f ||
        ready_gain_blend_s_ > 3.0f ||
        !std::isfinite(ready_min_stable_s_) || ready_min_stable_s_ < 0.2f ||
        ready_min_stable_s_ > 2.0f ||
        !std::isfinite(ready_max_wait_s_) ||
        ready_max_wait_s_ < ready_gain_blend_s_ + ready_min_stable_s_ ||
        ready_max_wait_s_ > 10.0f ||
        !std::isfinite(ready_max_joint_velocity_) ||
        ready_max_joint_velocity_ <= 0.0f || ready_max_joint_velocity_ > 1.0f ||
        !std::isfinite(ready_max_tracking_error_) ||
        ready_max_tracking_error_ <= 0.0f || ready_max_tracking_error_ > 0.5f ||
        !std::isfinite(ready_max_support_target_offset_) ||
        ready_max_support_target_offset_ <= 0.0f ||
        ready_max_support_target_offset_ > 0.30f) {
        throw std::runtime_error("VisionLoco ready_stand config is outside the safety envelope");
    }
    command_source_ = yaml_get<std::string>(cfg, "command_source", "fixed");
    if (command_source_ != "fixed" && command_source_ != "uwb" &&
        command_source_ != "nav" && command_source_ != "keyboard")
        throw std::runtime_error("command_source 必须是 fixed、uwb、nav 或 keyboard");
    try {
        auto c = cfg["fixed_cmd"].as<std::vector<float>>();
        if (c.size() == 3) std::copy(c.begin(), c.end(), fixed_cmd_.begin());
    } catch (...) {}
    // Keyboard command source. Keep it inside the main vx/wz branch of the
    // checkpoint's piecewise_union_v1 command envelope.
    key_step_vx_ = 0.1f; key_step_wz_ = 0.1f;
    key_max_vx_ = 0.2f; key_max_wz_ = 0.2f;
    key_target_period_s_ = 0.2f; key_idle_timeout_s_ = 10.0f;
    key_max_nonzero_s_ = 0.0f;
    if (cfg["keyboard"]) {
        auto k = cfg["keyboard"];
        key_step_vx_ = yaml_get<float>(k, "step_vx", key_step_vx_);
        key_step_wz_ = yaml_get<float>(k, "step_wz", key_step_wz_);
        key_max_vx_ = yaml_get<float>(k, "max_vx", key_max_vx_);
        key_max_wz_ = yaml_get<float>(k, "max_wz", key_max_wz_);
        key_target_period_s_ = 1.0f / yaml_get<float>(k, "target_hz", 5.0f);
        key_idle_timeout_s_ = yaml_get<float>(k, "idle_timeout_s", key_idle_timeout_s_);
        key_max_nonzero_s_ = yaml_get<float>(k, "max_nonzero_s", key_max_nonzero_s_);
    }
    if (!std::isfinite(key_step_vx_) || key_step_vx_ <= 0.0f ||
        !std::isfinite(key_step_wz_) || key_step_wz_ <= 0.0f ||
        !std::isfinite(key_max_vx_) || key_max_vx_ <= 0.0f || key_max_vx_ > 0.2f ||
        !std::isfinite(key_max_wz_) || key_max_wz_ <= 0.0f || key_max_wz_ > 0.2f ||
        !std::isfinite(key_target_period_s_) || key_target_period_s_ < 0.1f ||
        !std::isfinite(key_idle_timeout_s_) || key_idle_timeout_s_ < 1.0f ||
        !std::isfinite(key_max_nonzero_s_) || key_max_nonzero_s_ < 0.0f ||
        key_max_nonzero_s_ > 10.0f) {
        throw std::runtime_error(
            "keyboard ground-test config exceeds the 0.2 m/s, 0.2 rad/s safety envelope");
    }
    key_target_vx_ = 0.0f; key_target_wz_ = 0.0f;  // keyboard current target speed

    const char* parity_token = std::getenv("LOCO_SUSPENDED_PARITY_TOKEN");
    if (parity_token != nullptr &&
        std::string(parity_token) != "SUSPENDED_PARITY_V1") {
        throw std::runtime_error("invalid suspended parity authorization token");
    }
    suspended_parity_mode_ = parity_token != nullptr;
    const char* harness_token = std::getenv("LOCO_HARNESS_GUARD_TOKEN");
    if (harness_token != nullptr &&
        std::string(harness_token) != "HARNESS_GUARD_V1") {
        throw std::runtime_error("invalid harness guard authorization token");
    }
    harness_guard_mode_ = harness_token != nullptr;
    if (suspended_parity_mode_ && harness_guard_mode_) {
        throw std::runtime_error(
            "suspended parity and harness guard modes are mutually exclusive");
    }
    if ((suspended_parity_mode_ || harness_guard_mode_) &&
        (command_source_ != "keyboard" ||
         std::fabs(key_step_vx_ - 0.15f) > 1.0e-6f ||
         key_max_vx_ > 0.15f || key_max_wz_ > 0.10f ||
         std::fabs(key_max_nonzero_s_ - 2.0f) > 1.0e-6f)) {
        throw std::runtime_error(
            "transparent diagnostic modes require keyboard W=0.15, caps [0.15,0.10], "
            "and a 2.0 second hard timeout");
    }

    if (cfg["uwb"]) {
        auto u = cfg["uwb"];
        uwb_topic_ = yaml_get<std::string>(u, "topic", uwb_topic_);
        sport_topic_ = yaml_get<std::string>(u, "sport_state_topic", sport_topic_);
        max_vx_ = std::clamp(yaml_get<float>(u, "max_vx", max_vx_), 0.0f, 0.30f);
        max_vy_ = std::clamp(yaml_get<float>(u, "max_vy", max_vy_), 0.0f, 0.15f);
        max_wz_ = std::clamp(yaml_get<float>(u, "max_wz", max_wz_), 0.0f, 1.0f);
        yaw_kp_ = std::max(0.0f, yaml_get<float>(u, "yaw_kp", yaw_kp_));
        stop_distance_ = std::max(0.0f, yaml_get<float>(u, "stop_distance", stop_distance_));
        slow_distance_ = std::max(
            stop_distance_ + 0.05f, yaml_get<float>(u, "slow_distance", slow_distance_));
        stop_hysteresis_ = std::max(
            0.0f, yaml_get<float>(u, "stop_hysteresis", stop_hysteresis_));
        uwb_stop_once_ = yaml_get<bool>(u, "stop_once", uwb_stop_once_);
        uwb_goal_driven_ = yaml_get<bool>(u, "goal_driven", uwb_goal_driven_);
        try {
            auto cruise = u["cruise_cmd"].as<std::vector<float>>();
            if (cruise.size() != 3)
                throw std::runtime_error("uwb.cruise_cmd 必须是 [vx, vy, wz]");
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(cruise[i]))
                    throw std::runtime_error("uwb.cruise_cmd 必须是有限值");
                uwb_cruise_cmd_[i] = cruise[i];
            }
        } catch (const YAML::Exception&) {
            throw std::runtime_error("uwb.cruise_cmd 必须是三个数值");
        }
        if (uwb_goal_driven_ &&
            (uwb_cruise_cmd_[0] < 0.0f || uwb_cruise_cmd_[0] > 0.8f ||
             std::fabs(uwb_cruise_cmd_[1]) > 1e-6f ||
             std::fabs(uwb_cruise_cmd_[2]) > 1e-6f)) {
            throw std::runtime_error(
                "goal_driven 模式要求 cruise_cmd vx 在 [0,0.8] 且 vy=wz=0");
        }
        turn_in_place_angle_ = std::clamp(
            yaml_get<float>(u, "turn_in_place_angle", turn_in_place_angle_),
            0.1f, 3.1415926f);
        // cmd_vx_mode: A/B/C 对照测试开关 (见头文件注释)
        const std::string mode = yaml_get<std::string>(u, "cmd_vx_mode", "c_wz");
        if (mode != "c_wz" && mode != "goal_03" && mode != "goal_08") {
            throw std::runtime_error(
                "uwb.cmd_vx_mode 必须是 c_wz / goal_03 / goal_08 之一");
        }
        cmd_vx_mode_ = mode;
        uwb_stale_timeout_s_ = yaml_get<float>(u, "stale_timeout", uwb_stale_timeout_s_);
        uwb_hold_timeout_s_ = yaml_get<float>(u, "hold_timeout", uwb_hold_timeout_s_);
        uwb_buffer_seconds_ = yaml_get<float>(u, "buffer_seconds", uwb_buffer_seconds_);
        uwb_median_window_s_ = yaml_get<float>(u, "median_window", uwb_median_window_s_);
        uwb_filter_tau_s_ = yaml_get<float>(u, "filter_tau", uwb_filter_tau_s_);
        if (uwb_stale_timeout_s_ <= 0.0f ||
            uwb_hold_timeout_s_ <= uwb_stale_timeout_s_ ||
            uwb_buffer_seconds_ <= 0.0f || uwb_median_window_s_ <= 0.0f ||
            uwb_filter_tau_s_ <= 0.0f) {
            throw std::runtime_error(
                "UWB timeout/filter 参数无效：要求 stale>0、hold>stale、buffer/median/tau>0");
        }
        uwb_diagnostic_feedback_ =
            yaml_get<bool>(u, "diagnostic_feedback", uwb_diagnostic_feedback_);
        uwb_velocity_alpha_ = std::clamp(
            yaml_get<float>(u, "velocity_filter_alpha", uwb_velocity_alpha_),
            0.01f, 1.0f);
        try {
            auto rates = u["cmd_slew_rate"].as<std::vector<float>>();
            if (rates.size() == 3)
                for (int i = 0; i < 3; ++i) cmd_slew_rate_[i] = std::max(0.01f, rates[i]);
        } catch (...) {}
    }

    // ---------- articulation（复用官方读取，已按策略序映射）----------
    robot_ = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
    robot_->data.joint_ids_map.assign(joint_ids_map_.begin(), joint_ids_map_.end()); // vector<float>
    robot_->data.joint_pos.resize(joint_ids_map_.size());
    robot_->data.joint_vel.resize(joint_ids_map_.size());
    robot_->data.joint_effort.resize(joint_ids_map_.size());
    robot_->data.default_joint_pos =
        Eigen::VectorXf::Map(default_joint_pos_.data(), default_joint_pos_.size());
    robot_->data.joint_stiffness = stiffness_;
    robot_->data.joint_damping   = damping_;

    // ---------- ONNX runner ----------
    runner_ = std::make_unique<isaaclab::LocoRunner>(
        (policy_dir / "exported" / "policy.onnx").string());

    // LSTM 状态诊断开关 (config.yaml: FSM.VisionLoco.lstm_debug_mode)
    //   "stateful" (默认): 连续保留 h/c, 正常部署行为
    //   "reset_each_frame": 每帧清零 h/c, 退化测试用
    const std::string lstm_mode = yaml_get<std::string>(cfg, "lstm_debug_mode", "stateful");
    runner_->set_lstm_debug_mode(lstm_mode);

    // ---------- 深度源 ----------
    std::string dsrc = "constant";
    float dval = 1.0f;
    vision_nav::FilterConfig filters;  // 默认 mode="none"（基线 A）
    if (cfg["depth"]) {
        dsrc = yaml_get<std::string>(cfg["depth"], "source", "constant");
        dval = yaml_get<float>(cfg["depth"], "constant_value", 1.0f);
        if (cfg["depth"]["filters"]) {
            auto f = cfg["depth"]["filters"];
            filters.mode = yaml_get<std::string>(f, "mode", filters.mode);
            filters.spatial_smooth_alpha = yaml_get<float>(f, "spatial_smooth_alpha", filters.spatial_smooth_alpha);
            filters.spatial_magnitude     = yaml_get<int>  (f, "spatial_magnitude",    filters.spatial_magnitude);
            filters.spatial_smooth_delta  = yaml_get<int>  (f, "spatial_smooth_delta", filters.spatial_smooth_delta);
            filters.temporal_alpha        = yaml_get<float>(f, "temporal_alpha",       filters.temporal_alpha);
            filters.temporal_delta        = yaml_get<float>(f, "temporal_delta",       filters.temporal_delta);
            // 安全白名单: 只允许学长建议的三种模式, 误填回退基线
            if (filters.mode != "none" && filters.mode != "light_spatial" &&
                filters.mode != "light_spatial_weak_temporal") {
                spdlog::warn("[VisionLoco] 未知 depth.filters.mode='{}', 回退 'none'(基线 A)", filters.mode);
                filters.mode = "none";
            }
            // 安全范围: spatial_magnitude 必须在 [1,5]
            filters.spatial_magnitude = std::clamp(filters.spatial_magnitude, 1, 5);
        }
    }
    if (dsrc == "realsense") {
#ifdef USE_REALSENSE
        depth_ = std::make_unique<vision_nav::RealSenseDepth>(424, 240, 30, filters);
#else
        spdlog::error("[VisionLoco] 编译时未开 USE_REALSENSE，退回 ConstantDepth。"
                      "请在 CMake 打开 -DUSE_REALSENSE=ON 并装 librealsense2-dev。");
        depth_ = std::make_unique<vision_nav::ConstantDepth>(dval);
#endif
    } else {
        depth_ = std::make_unique<vision_nav::ConstantDepth>(dval);
    }

    // SportModeState 只作为实际运动反馈；控制安全不依赖它是否发布。
    sport_sub_ = std::make_shared<
        unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(sport_topic_);
    sport_sub_->InitChannel([this](const void* msg) { on_sport_state(msg); }, 1);

    if (command_source_ == "uwb" || uwb_diagnostic_feedback_) {
        uwb_sub_ = std::make_shared<
            unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::UwbState_>>(uwb_topic_);
        uwb_sub_->InitChannel([this](const void* msg) { on_uwb(msg); }, 1);
        utrack_ = std::make_unique<unitree::robot::go2::UtrackClient>();
        utrack_->SetTimeout(3.0f);
        utrack_->Init();
    }

    log_dir_ = (policy_dir / "logs").string();
    if (cfg["logging"]) {
        logging_enabled_ = yaml_get<bool>(cfg["logging"], "enabled", true);
        log_flush_every_ = std::max(1, yaml_get<int>(cfg["logging"], "flush_every", 50));
        max_consecutive_errors_ =
            std::max(1, yaml_get<int>(cfg["logging"], "max_consecutive_errors", 1));
    }
    if (cfg["safety"]) {
        auto safety = cfg["safety"];
        const float max_tilt_deg = yaml_get<float>(safety, "max_tilt_deg", 25.0f);
        max_tilt_rad_ = max_tilt_deg * 3.1415926f / 180.0f;
        safety_grace_s_ = yaml_get<float>(safety, "grace_s", safety_grace_s_);
        max_raw_action_abs_ =
            yaml_get<float>(safety, "max_raw_action_abs", max_raw_action_abs_);
        max_raw_action_step_ =
            yaml_get<float>(safety, "max_raw_action_step", max_raw_action_step_);
        max_target_step_rad_ =
            yaml_get<float>(safety, "max_target_step_rad", max_target_step_rad_);
        zero_command_hold_threshold_ = yaml_get<float>(
            safety, "zero_command_hold_threshold", zero_command_hold_threshold_);
        action_step_trip_frames_ =
            yaml_get<int>(safety, "action_step_trip_frames", action_step_trip_frames_);
        max_tracking_error_rad_ =
            yaml_get<float>(safety, "max_tracking_error_rad", max_tracking_error_rad_);
        max_motion_tracking_error_rad_ = yaml_get<float>(
            safety, "max_motion_tracking_error_rad", max_motion_tracking_error_rad_);
        tracking_error_trip_frames_ =
            yaml_get<int>(safety, "tracking_error_trip_frames", tracking_error_trip_frames_);
        motion_tracking_error_trip_frames_ = yaml_get<int>(
            safety, "motion_tracking_error_trip_frames",
            motion_tracking_error_trip_frames_);
    }
    if (harness_guard_mode_) {
        if (!cfg["harness_guard"])
            throw std::runtime_error("harness guard configuration is missing");
        const auto guard = cfg["harness_guard"];
        if (yaml_get<std::string>(guard, "profile", "") !=
            "transparent_guard_candidate_v1") {
            throw std::runtime_error("unexpected harness guard profile");
        }
        try {
            harness_raw_abs_limits_ =
                guard["raw_action_abs_max"].as<std::vector<float>>();
            harness_raw_step_limits_ =
                guard["raw_action_step_max"].as<std::vector<float>>();
            harness_target_min_ =
                guard["target_min_rad"].as<std::vector<float>>();
            harness_target_max_ =
                guard["target_max_rad"].as<std::vector<float>>();
        } catch (const YAML::Exception&) {
            throw std::runtime_error("invalid harness guard joint arrays");
        }
        harness_effort_abs_nm_ =
            yaml_get<float>(guard, "joint_effort_abs_nm", 0.0f);
        harness_effort_trip_frames_ =
            yaml_get<int>(guard, "effort_trip_frames", 0);
        const auto valid_positive_limits = [](const std::vector<float>& values) {
            return values.size() == 12 &&
                std::all_of(values.begin(), values.end(), [](float value) {
                    return std::isfinite(value) && value > 0.0f;
                });
        };
        if (!valid_positive_limits(harness_raw_abs_limits_) ||
            !valid_positive_limits(harness_raw_step_limits_) ||
            harness_target_min_.size() != 12 ||
            harness_target_max_.size() != 12 ||
            !std::isfinite(harness_effort_abs_nm_) ||
            harness_effort_abs_nm_ != 12.0f ||
            harness_effort_trip_frames_ != 3) {
            throw std::runtime_error("harness guard limits are invalid");
        }
        for (size_t i = 0; i < 12; ++i) {
            if (!std::isfinite(harness_target_min_[i]) ||
                !std::isfinite(harness_target_max_[i]) ||
                harness_target_min_[i] >= harness_target_max_[i] ||
                harness_raw_abs_limits_[i] > 8.0f ||
                harness_raw_step_limits_[i] > 4.0f) {
                throw std::runtime_error("harness guard joint limit is invalid");
            }
        }
    }
    if (!std::isfinite(max_tilt_rad_) || max_tilt_rad_ < 0.17f || max_tilt_rad_ > 0.79f ||
        !std::isfinite(safety_grace_s_) || safety_grace_s_ < entry_blend_s_ ||
        safety_grace_s_ > 5.0f ||
        !std::isfinite(max_raw_action_abs_) || max_raw_action_abs_ <= 0.0f ||
        max_raw_action_abs_ > 6.0f ||
        !std::isfinite(max_raw_action_step_) || max_raw_action_step_ <= 0.0f ||
        max_raw_action_step_ > 1.0f || action_step_trip_frames_ < 1 ||
        !std::isfinite(max_target_step_rad_) || max_target_step_rad_ <= 0.0f ||
        max_target_step_rad_ > 0.05f ||
        !std::isfinite(zero_command_hold_threshold_) ||
        zero_command_hold_threshold_ < 0.0f || zero_command_hold_threshold_ > 0.05f ||
        !std::isfinite(max_tracking_error_rad_) || max_tracking_error_rad_ <= 0.0f ||
        max_tracking_error_rad_ > 0.50f || tracking_error_trip_frames_ < 1 ||
        tracking_error_trip_frames_ > 25 ||
        !std::isfinite(max_motion_tracking_error_rad_) ||
        max_motion_tracking_error_rad_ < max_tracking_error_rad_ ||
        max_motion_tracking_error_rad_ > 1.00f ||
        motion_tracking_error_trip_frames_ < tracking_error_trip_frames_ ||
        motion_tracking_error_trip_frames_ > 50) {
        throw std::runtime_error(
            "VisionLoco safety limits are missing or outside the active runtime envelope");
    }

    // ---------- 安全转移：翻倒 → Passive ----------
    this->registered_checks.emplace_back(std::make_pair(
        [this]() -> bool {
            return bad_orientation(robot_->data.projected_gravity_b, max_tilt_rad_);
        },
        FSMStringMap.right.at("Passive")));
    this->registered_checks.emplace_back(std::make_pair(
        [this]() -> bool { return policy_fault_.load(); },
        FSMStringMap.right.at("Passive")));

    spdlog::info("[VisionLoco] 就绪：step_dt={:.3f}s scale={:.3f} source={} "
                 "goal=[{:.2f},{:.2f},{:.2f},{:.2f}] depth={} filters={} logging={}",
                 step_dt_, act_scale_, command_source_,
                 goal_[0], goal_[1], goal_[2], goal_[3],
                 dsrc, filters.mode, logging_enabled_);
    spdlog::info(
        "[VisionLoco][safety] mode={} tilt={:.1f}deg normal_raw_abs={:.2f} "
        "normal_raw_step={:.2f}/{} normal_target_step={:.3f}rad "
        "tracking_stand={:.2f}rad/{} "
        "tracking_motion={:.2f}rad/{} grace={:.1f}s zero_hold<={:.3f}",
        harness_guard_mode_ ? "harness_guard" :
            (suspended_parity_mode_ ? "suspended_parity" : "normal"),
        max_tilt_rad_ * 180.0f / 3.1415926f, max_raw_action_abs_,
        max_raw_action_step_, action_step_trip_frames_, max_target_step_rad_,
        max_tracking_error_rad_, tracking_error_trip_frames_,
        max_motion_tracking_error_rad_, motion_tracking_error_trip_frames_,
        safety_grace_s_, zero_command_hold_threshold_);
    if (suspended_parity_mode_ || harness_guard_mode_) {
        spdlog::warn(
            "[VisionLoco][safety] transparent motion-only envelope: "
            "raw_abs=8.00 raw_step=8.00 target_step=1.000rad entry_blend=bypass; "
            "stand and forced-zero return retain the normal envelope");
    }
    if (harness_guard_mode_) {
        spdlog::warn(
            "[VisionLoco][safety] harness-only guard active: per-joint historical "
            "output bounds and effort_abs={:.1f}Nm/{} frames",
            harness_effort_abs_nm_, harness_effort_trip_frames_);
    }
}

State_VisionLoco::~State_VisionLoco()
{
    running_ = false;
    if (policy_thread_.joinable()) policy_thread_.join();
    if (uwb_sub_) uwb_sub_->CloseChannel();
    if (sport_sub_) sport_sub_->CloseChannel();
}

void State_VisionLoco::on_uwb(const void* message)
{
    const auto* msg = static_cast<const unitree_go::msg::dds_::UwbState_*>(message);
    const auto now = std::chrono::steady_clock::now();
    const std::array<float, 4> next{
        msg->orientation_est(), msg->pitch_est(), msg->distance_est(), msg->yaw_est()};

    std::lock_guard<std::mutex> lk(sensor_mtx_);
    float closing = uwb_.closing_speed;
    bool velocity_valid = false;
    std::array<float, 2> body_velocity = uwb_.body_velocity;
    const bool current_valid =
        msg->error_state() == 0 && msg->enabled_from_app() == 1 &&
        std::all_of(next.begin(), next.end(), [](float value) {
            return std::isfinite(value);
        }) && next[2] >= 0.0f &&
        std::isfinite(msg->base_yaw());
    const bool previous_valid =
        uwb_.received && uwb_.error_state == 0 && uwb_.enabled_from_app == 1 &&
        std::all_of(uwb_.raw.begin(), uwb_.raw.end(), [](float value) {
            return std::isfinite(value);
        }) &&
        std::isfinite(uwb_.base_yaw);
    if (current_valid && previous_valid) {
        const float dt = std::chrono::duration<float>(now - uwb_.received_at).count();
        if (dt >= 0.03f && dt <= 1.0f && std::isfinite(next[2]) && std::isfinite(uwb_.raw[2])) {
            const float instantaneous = std::clamp((uwb_.raw[2] - next[2]) / dt, -2.0f, 2.0f);
            closing = 0.8f * closing + 0.2f * instantaneous;

            // 将“机器人到静止 UWB tag”的相对向量转到公共航向系，
            // 对其取负差分得到机器人平面速度，再转回当前机体系。
            const float previous_range = uwb_.raw[2] * std::cos(uwb_.raw[1]);
            const float current_range = next[2] * std::cos(next[1]);
            const float previous_heading = uwb_.base_yaw + uwb_.raw[0];
            const float current_heading = msg->base_yaw() + next[0];
            const float previous_x = -previous_range * std::cos(previous_heading);
            const float previous_y = -previous_range * std::sin(previous_heading);
            const float current_x = -current_range * std::cos(current_heading);
            const float current_y = -current_range * std::sin(current_heading);
            const float world_vx = (current_x - previous_x) / dt;
            const float world_vy = (current_y - previous_y) / dt;
            const float cy = std::cos(msg->base_yaw());
            const float sy = std::sin(msg->base_yaw());
            const float measured_vx = cy * world_vx + sy * world_vy;
            const float measured_vy = -sy * world_vx + cy * world_vy;
            if (std::isfinite(measured_vx) && std::isfinite(measured_vy) &&
                std::hypot(measured_vx, measured_vy) <= 2.0f) {
                const float alpha = uwb_velocity_alpha_;
                body_velocity[0] =
                    (1.0f - alpha) * body_velocity[0] + alpha * measured_vx;
                body_velocity[1] =
                    (1.0f - alpha) * body_velocity[1] + alpha * measured_vy;
                velocity_valid = true;
            }
        }
    }
    uwb_.raw = next;
    uwb_.error_state = msg->error_state();
    uwb_.enabled_from_app = msg->enabled_from_app();
    uwb_.channel = msg->channel();
    uwb_.received = true;
    uwb_.received_at = now;
    uwb_.closing_speed = closing;
    uwb_.base_yaw = msg->base_yaw();
    uwb_.body_velocity = body_velocity;
    uwb_.velocity_valid = velocity_valid;
    if (velocity_valid) uwb_.velocity_received_at = now;

    if (!current_valid) return;

    const float planar_distance =
        vision_nav::planar_distance_from_uwb(next[1], next[2]);

    // 跳变拒绝: 数据分析显示 97% 帧间跳变 <5°, 只需滤掉 >60° 的尖峰噪声。
    // 跳变帧直接丢弃, 不更新 filtered 值 (保持上一帧)。
    if (!vision_nav::uwb_jump_reject(
            uwb_jump_state_, next[0], planar_distance)) {
        return;
    }

    // 不再用 median + EMA: 这两者叠加造成 2.66s 滞后, 比噪声本身更破坏跟随。
    // 通过跳变拒绝的帧直接作为 filtered 值, 让模型看到接近真实的 goal。
    //
    // beta 死区: UWB 方位角实测噪声 std≈0.064 rad (3.7°), 是训练 bearing_noise
    // (0.035 rad/2°) 的 1.8 倍。目标在正前方时 beta 持续抖动 → goal_y 反复正负
    // 跳变 → C++ wz=yaw_kp*beta 放大摆动 → 模型输出 s 型动作。
    // 死区把 |beta|<threshold 的帧强制 goal_y=0 且 beta=0 (使 wz=0)。
    // 阈值 0.15 rad (8.6°): 实测 s 段从 1008 减到 66, 是最佳平衡点。
    // (0.30 实测过激, 大角度转向反应不及时, 效果反而更差。)
    constexpr float beta_deadband_rad = 0.15f;
    float effective_beta = next[0];
    if (std::fabs(effective_beta) < beta_deadband_rad) {
        effective_beta = 0.0f;
    }
    uwb_.filtered_x = planar_distance * std::cos(effective_beta);
    uwb_.filtered_y = planar_distance * std::sin(effective_beta);
    uwb_.filtered_planar_distance = planar_distance;
    uwb_.filtered_beta = effective_beta;
    uwb_.filtered_received = true;
    uwb_.filtered_received_at = now;
}

void State_VisionLoco::on_sport_state(const void* message)
{
    const auto* msg = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
    std::lock_guard<std::mutex> lk(sensor_mtx_);
    sport_.velocity = msg->velocity();
    sport_.yaw_speed = msg->yaw_speed();
    sport_.received = true;
    sport_.received_at = std::chrono::steady_clock::now();
}

std::array<float, 3> State_VisionLoco::command_for_frame(
    const std::chrono::steady_clock::time_point& now,
    std::vector<float>& goal,
    std::array<float, 3>& theory_cmd,
    bool& uwb_valid,
    float& uwb_age_s,
    float& closing_speed,
    int& uwb_error,
    int& uwb_enabled,
    int& uwb_channel)
{
    uwb_valid = false;
    uwb_age_s = -1.0f;
    closing_speed = 0.0f;
    uwb_error = 255;
    uwb_enabled = 0;
    uwb_channel = -1;
    theory_cmd = fixed_cmd_;

    bool force_keyboard_zero = false;
    if (command_source_ == "fixed") {
        theory_cmd[0] = std::clamp(theory_cmd[0], -0.20f, 1.30f);
        theory_cmd[1] = std::clamp(theory_cmd[1], -0.15f, 0.15f);
        theory_cmd[2] = std::clamp(theory_cmd[2], -1.00f, 1.00f);
    } else if (command_source_ == "keyboard") {
        std::string key;
        if (FSMState::keyboard && FSMState::keyboard->pop_latest_key(key)) {
            key_last_input_at_ = now;
            key_seen_input_ = true;
            key_timeout_reported_ = false;
            const bool zero_request = key == " ";
            const bool adjustment_due =
                now - key_last_adjust_at_ >= std::chrono::duration<float>(key_target_period_s_);
            bool target_changed = false;
            if (zero_request) {
                key_target_vx_ = 0.0f;
                key_target_wz_ = 0.0f;
                key_last_adjust_at_ = now;
                target_changed = true;
            } else if (adjustment_due && !key_motion_timeout_latched_) {
                if (key == "w" || key == "W") {
                    key_target_vx_ = std::min(key_target_vx_ + key_step_vx_, key_max_vx_);
                    target_changed = true;
                } else if (key == "s" || key == "S") {
                    key_target_vx_ = std::max(key_target_vx_ - key_step_vx_, 0.0f);
                    target_changed = true;
                } else if (key == "a" || key == "A") {
                    key_target_wz_ = std::min(key_target_wz_ + key_step_wz_, key_max_wz_);
                    target_changed = true;
                } else if (key == "d" || key == "D") {
                    key_target_wz_ = std::max(key_target_wz_ - key_step_wz_, -key_max_wz_);
                    target_changed = true;
                }
                if (target_changed) key_last_adjust_at_ = now;
            }
            if (target_changed) {
                spdlog::info("[VisionLoco][keyboard] target=[{:.2f},0.00,{:.2f}] key={}",
                             key_target_vx_, key_target_wz_, zero_request ? "Space" : key);
            }
        }
        if (FSMState::keyboard && !FSMState::keyboard->healthy()) {
            key_target_vx_ = 0.0f;
            key_target_wz_ = 0.0f;
            if (!key_health_reported_) {
                spdlog::error("[VisionLoco][keyboard] terminal input failed; target forced to zero");
                key_health_reported_ = true;
            }
        }
        if (key_seen_input_ &&
            now - key_last_input_at_ > std::chrono::duration<float>(key_idle_timeout_s_)) {
            key_target_vx_ = 0.0f;
            key_target_wz_ = 0.0f;
            if (!key_timeout_reported_) {
                spdlog::warn("[VisionLoco][keyboard] {:.1f}s idle timeout; target forced to zero",
                             key_idle_timeout_s_);
                key_timeout_reported_ = true;
            }
        }
        const bool target_nonzero =
            std::fabs(key_target_vx_) > 0.02f || std::fabs(key_target_wz_) > 0.02f;
        if (target_nonzero && !key_motion_active_) {
            key_motion_active_ = true;
            key_motion_started_at_ = now;
        } else if (!target_nonzero) {
            key_motion_active_ = false;
        }
        if (key_motion_active_ && loco_safety::command_duration_expired(
                std::chrono::duration<float>(now - key_motion_started_at_).count(),
                key_max_nonzero_s_)) {
            key_target_vx_ = 0.0f;
            key_target_wz_ = 0.0f;
            key_motion_active_ = false;
            key_motion_timeout_latched_ = true;
            force_keyboard_zero = true;
            spdlog::critical(
                "[VisionLoco][keyboard] hard non-zero timeout {:.1f}s; command forced to zero "
                "and motion keys latched until VisionLoco is re-entered",
                key_max_nonzero_s_);
        }
        theory_cmd[0] = key_target_vx_;
        theory_cmd[1] = 0.0f;
        theory_cmd[2] = key_target_wz_;
    } else if (command_source_ == "uwb") {
        UwbSample sample;
        {
            std::lock_guard<std::mutex> lk(sensor_mtx_);
            sample = uwb_;
        }
        if (sample.filtered_received)
            uwb_age_s = std::chrono::duration<float>(
                now - sample.filtered_received_at).count();
        uwb_error = sample.error_state;
        uwb_enabled = sample.enabled_from_app;
        uwb_channel = sample.channel;
        const bool filtered_finite =
            std::isfinite(sample.filtered_x) && std::isfinite(sample.filtered_y) &&
            std::isfinite(sample.filtered_planar_distance) &&
            std::isfinite(sample.filtered_beta);
        // 错误/禁用/非法帧不会进入滤波器，也不造成瞬时命令跳变；它们与丢帧
        // 一样消耗最后有效坐标的有限 hold 时间，随后平滑归零。
        uwb_valid = sample.filtered_received && filtered_finite &&
                    uwb_age_s >= 0.0f && uwb_age_s <= uwb_hold_timeout_s_;
        closing_speed = sample.closing_speed;
        theory_cmd = {0.0f, 0.0f, 0.0f};
        last_uwb_hold_ = false;
        last_uwb_freshness_scale_ = 0.0f;
        last_uwb_filter_age_s_ = uwb_age_s;
        if (uwb_stop_once_ && uwb_arrived_)
            goal.assign(4, 0.0f);
        if (uwb_valid) {
            const float beta = sample.filtered_beta;
            const float planar_distance = sample.filtered_planar_distance;
            const float freshness_scale = vision_nav::uwb_freshness_scale(
                uwb_age_s, uwb_stale_timeout_s_, uwb_hold_timeout_s_);
            last_uwb_filtered_x_ = sample.filtered_x;
            last_uwb_filtered_y_ = sample.filtered_y;
            last_uwb_filtered_beta_ = beta;
            last_uwb_planar_distance_ = planar_distance;
            last_uwb_hold_ = uwb_age_s > uwb_stale_timeout_s_;
            last_uwb_freshness_scale_ = freshness_scale;
            if (uwb_stop_once_) {
                if (!uwb_arrived_ && planar_distance <= stop_distance_)
                    uwb_arrived_ = true;
            } else {
                uwb_arrived_ = vision_nav::uwb_arrived_with_hysteresis(
                    uwb_arrived_, planar_distance, stop_distance_, stop_hysteresis_);
            }
            if (uwb_stop_once_ && uwb_arrived_) {
                goal.assign(4, 0.0f);
            } else {
                const auto actor_goal = vision_nav::actor_goal_from_planar_xy(
                    sample.filtered_x, sample.filtered_y);
                goal.assign(actor_goal.begin(), actor_goal.end());
            }
            if (!uwb_arrived_) {
                const float distance_scale = vision_nav::approach_speed_scale(
                    planar_distance, stop_distance_, slow_distance_);
                if (cmd_vx_mode_ == "goal_03" || cmd_vx_mode_ == "goal_08") {
                    // 训练分布模式: cmd_wz=0 (训练时恒零), cmd_vx 固定 (匹配训练 commands.ranges)
                    // goal 仍由 UWB 正常生成 (上面 actor_goal_from_planar_xy),
                    // 模型从 goal 自主决定转向方向。
                    // 仅在接近终点时用 distance_scale 减速, 防止撞 UWB tag。
                    const float cruise_vx =
                        (cmd_vx_mode_ == "goal_08") ? 0.80f : 0.30f;
                    theory_cmd[0] = cruise_vx * distance_scale;
                    theory_cmd[1] = 0.0f;
                    theory_cmd[2] = 0.0f;  // 关键: cmd_wz=0, 匹配训练分布
                } else if (uwb_goal_driven_) {
                    for (int i = 0; i < 3; ++i)
                        theory_cmd[i] = uwb_cruise_cmd_[i] * distance_scale;
                } else {
                    const float heading_scale = std::max(0.0f, std::cos(beta));
                    theory_cmd[0] = std::fabs(beta) >= turn_in_place_angle_
                                        ? 0.0f
                                        : max_vx_ * distance_scale * heading_scale;
                    theory_cmd[1] = std::clamp(
                        max_vy_ * std::sin(beta), -max_vy_, max_vy_);
                    theory_cmd[2] = std::clamp(yaw_kp_ * beta, -max_wz_, max_wz_);
                }
                // freshness 只影响前进速度, 不影响转向
                // 这样信号stale时狗减速但不停止转向, 给机会重新收到信号
                theory_cmd[0] *= freshness_scale;
                theory_cmd[1] *= freshness_scale;
                // theory_cmd[2](wz) 不乘freshness, 保持转向力度
            }
        }
    } else {
        // loco 阶段无 learned nav actor（模型无 nav 分支）：nav 命令源等价零速度，
        // 仅保留分支以复用 State。goal 使用 config.yaml 固定值，不启动 UWB。
        theory_cmd = {0.0f, 0.0f, 0.0f};
    }

    std::array<float, 3> exec{};
    // 所有模式都经过slew rate平滑, 防止速度突变(包括goal_driven模式)
    {
        for (int i = 0; i < 3; ++i) {
            const float max_step = cmd_slew_rate_[i] * step_dt_;
            exec[i] = last_exec_cmd_[i] +
                      std::clamp(theory_cmd[i] - last_exec_cmd_[i], -max_step, max_step);
        }
    }
    if (force_keyboard_zero) exec = {0.0f, 0.0f, 0.0f};
    if (command_source_ == "uwb" && !uwb_valid) {
        // UWB信号丢失: 不前进(vx/vy归零), 但保持最后的转向(wz)
        // 给狗机会转过身来重新收到UWB信号(目标在后方时信号会被机身遮挡)
        exec[0] = 0.0f;  // vx归零
        exec[1] = 0.0f;  // vy归零
        // exec[2] 保持slew rate算出的值(继续转向)
    }
    last_exec_cmd_ = exec;
    return exec;
}

std::vector<float> State_VisionLoco::build_proprio()
{
    // 策略序：ang_vel(3) proj_g(3) vel_cmd(3,留0待runner填上一帧cmd)
    //         joint_pos_rel(12) joint_vel_rel(12) last_action(12)
    std::vector<float> p(45, 0.0f);
    const auto& av  = robot_->data.root_ang_vel_b;
    const auto& pg  = robot_->data.projected_gravity_b;
    const auto& q   = robot_->data.joint_pos;
    const auto& dq  = robot_->data.joint_vel;
    const auto& dft = robot_->data.default_joint_pos;

    for (int i = 0; i < 3; ++i) p[0 + i] = av[i] * sc_ang_vel_;
    for (int i = 0; i < 3; ++i) p[3 + i] = pg[i] * sc_proj_g_;
    // p[6:9] 由 runner 写为上一帧 cmd
    for (int i = 0; i < 12; ++i) p[9  + i] = (q[i] - dft[i]) * sc_jpos_;
    for (int i = 0; i < 12; ++i) p[21 + i] = dq[i] * sc_jvel_;
    {
        std::lock_guard<std::mutex> lk(tgt_mtx_);
        for (int i = 0; i < 12; ++i) p[33 + i] = last_action_raw_[i] * sc_lastact_;
    }
    // clip（与 deploy.yaml proprio_clip 一致；量级很小，基本不触发）
    for (auto& v : p) v = std::clamp(v, -proprio_clip_, proprio_clip_);
    return p;
}

void State_VisionLoco::enter()
{
    // Initialize keyboard for keyboard command mode
    if (command_source_ == "keyboard") {
        FSMState::keyboard = std::make_shared<Keyboard>();
        const auto now = std::chrono::steady_clock::now();
        key_target_vx_ = 0.0f;
        key_target_wz_ = 0.0f;
        key_seen_input_ = false;
        key_timeout_reported_ = false;
        key_health_reported_ = false;
        key_motion_active_ = false;
        key_motion_timeout_latched_ = false;
        key_last_input_at_ = now;
        key_last_adjust_at_ = now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                        std::chrono::duration<float>(key_target_period_s_));
        spdlog::info("[VisionLoco] Keyboard initialized. WASD=setpoint, Space=zero");
        std::cout << "\nKeyboard Control:\n"
                  << "  W/S : forward speed +/-" << key_step_vx_
                  << " m/s  (range 0.." << key_max_vx_ << ", no reverse)\n"
                  << "  A/D : turn left/right  +/-" << key_step_wz_ << " rad/s  (max " << key_max_wz_ << ")\n"
                  << "  Space : request zero command\n"
                  << "  Idle timeout : " << key_idle_timeout_s_ << " s\n"
                  << "  Hard non-zero timeout : " << key_max_nonzero_s_ << " s (0=disabled)\n"
                  << "  L2+B : emergency stop\n\n";
    }

    robot_->update();

    entry_joint_pos_.resize(12);
    entry_command_target_.resize(12);
    entry_stiffness_.resize(12);
    entry_damping_.resize(12);
    stand_hold_action_raw_.resize(12);
    float max_support_target_offset = 0.0f;
    for (int i = 0; i < 12; ++i) {
        const float q = robot_->data.joint_pos[i];
        const float dq = robot_->data.joint_vel[i];
        if (!std::isfinite(q))
            throw std::runtime_error("VisionLoco entry joint position is not finite");
        const int motor_id = joint_ids_map_[i];
        auto& motor = lowcmd->msg_.motor_cmd()[motor_id];
        const float kp = lowcmd->msg_.motor_cmd()[motor_id].kp();
        const float kd = lowcmd->msg_.motor_cmd()[motor_id].kd();
        const float previous_target = motor.q();
        if (!std::isfinite(dq) || !std::isfinite(kp) || !std::isfinite(kd) ||
            !std::isfinite(previous_target))
            throw std::runtime_error("VisionLoco entry PD state is not finite");
        const float bounded_previous_target = q + std::clamp(
            previous_target - q,
            -ready_max_support_target_offset_, ready_max_support_target_offset_);
        entry_command_target_[i] = bounded_previous_target;
        entry_joint_pos_[i] = loco_safety::torque_preserving_target(
            q, dq, bounded_previous_target, kp, kd,
            stiffness_[i], damping_[i], ready_max_support_target_offset_);
        max_support_target_offset = std::max(
            max_support_target_offset, std::fabs(entry_joint_pos_[i] - q));
        entry_stiffness_[i] = kp;
        entry_damping_[i] = kd;
        stand_hold_action_raw_[i] =
            std::clamp((entry_joint_pos_[i] - act_offset_[i]) / act_scale_,
                       act_clip_lo_, act_clip_hi_);
        motor.dq() = 0;
        motor.tau() = 0;
    }
    {
        std::lock_guard<std::mutex> lk(tgt_mtx_);
        joint_target_    = entry_command_target_;
        last_action_raw_ = stand_hold_action_raw_;
        have_target_     = true;
    }
    spdlog::info(
        "[VisionLoco] support-preserving ready target: max offset={:.3f}rad "
        "(limit {:.3f}rad)",
        max_support_target_offset, ready_max_support_target_offset_);
    runner_->reset();
    frame_ = 0;
    deadline_misses_ = 0;
    consecutive_errors_ = 0;
    policy_fault_ = false;
    policy_ready_ = false;
    ready_stable_frames_ = 0;
    action_step_violation_frames_ = 0;
    tracking_error_violation_frames_ = 0;
    last_exec_cmd_ = {0.0f, 0.0f, 0.0f};
    last_uwb_planar_distance_ = 0.0f;
    last_uwb_filtered_x_ = 0.0f;
    last_uwb_filtered_y_ = 0.0f;
    last_uwb_filtered_beta_ = 0.0f;
    last_uwb_filter_age_s_ = -1.0f;
    last_uwb_freshness_scale_ = 0.0f;
    last_uwb_hold_ = false;
    uwb_arrived_ = false;
    {
        std::lock_guard<std::mutex> lk(sensor_mtx_);
        uwb_filter_queue_.clear();
        uwb_.filtered_received = false;
        uwb_.filtered_x = 0.0f;
        uwb_.filtered_y = 0.0f;
        uwb_.filtered_planar_distance = 0.0f;
        uwb_.filtered_beta = 0.0f;
    }

    // fixed/uwb 始终外部覆盖；nav 使用 config.yaml 固定 goal（loco 阶段等价零速度）。
    runner_->set_cmd_override(true, 0.0f, 0.0f, 0.0f);
    if (command_source_ == "uwb" || uwb_diagnostic_feedback_) {
        try {
            const int ret = utrack_->SwitchSet(true);
            bool enabled = false, tracking = false;
            utrack_->SwitchGet(enabled);
            utrack_->IsTracking(tracking);
            if (command_source_ == "uwb") {
                spdlog::warn("[VisionLoco] UWB 控制：SwitchSet ret={} enabled={} tracking={}，"
                             "cmd_vx_mode={} goal_driven={} cruise_cmd=[{:.2f},{:.2f},{:.2f}] stop_once={}。",
                             ret, enabled, tracking, cmd_vx_mode_, uwb_goal_driven_,
                             uwb_cruise_cmd_[0], uwb_cruise_cmd_[1], uwb_cruise_cmd_[2],
                             uwb_stop_once_);
            } else {
                spdlog::info("[VisionLoco] UWB 仅作运动反馈：enabled={} tracking={}，"
                             "不参与 {} 命令生成。",
                             enabled, tracking, command_source_);
            }
        } catch (const std::exception& e) {
            if (command_source_ == "uwb")
                spdlog::error("[VisionLoco] UWB 使能失败，将保持零速度: {}", e.what());
            else
                spdlog::warn("[VisionLoco] UWB 诊断反馈不可用: {}", e.what());
        }
    }
    if (command_source_ == "nav") {
        spdlog::warn("[VisionLoco] loco 阶段无 nav actor：command_source=nav 等价零速度，"
                     "正式 loco 测试请用 fixed / uwb。");
    } else if (command_source_ == "fixed") {
        spdlog::warn("[VisionLoco] 固定外部控制 cmd=[{:.3f},{:.3f},{:.3f}]。",
                     fixed_cmd_[0], fixed_cmd_[1], fixed_cmd_[2]);
    }

    // 打开逐帧诊断 CSV。
    if (logging_enabled_) {
        std::filesystem::create_directories(log_dir_);
        char name[64];
        std::snprintf(name, sizeof(name), "visloco_diag_%ld.csv", (long)std::time(nullptr));
        diag_path_ = (std::filesystem::path(log_dir_) / name).string();
        diag_.open(diag_path_, std::ios::out | std::ios::trunc);
        if (diag_) {
            diag_ << "frame,t_ms,loop_ms,inference_ms,deadline_misses,consecutive_errors,"
                     "vx,vy,wz,vx_raw,vy_raw,wz_raw,"
                     "theory_vx,theory_vy,theory_wz,cmd_source,"
                     "uwb_beta,uwb_pitch,uwb_distance,uwb_yaw,"
                     "uwb_planar_distance,uwb_arrived,uwb_stop_once,uwb_goal_driven,"
                     "uwb_filtered_x,uwb_filtered_y,uwb_filtered_beta,"
                     "uwb_filter_age_s,uwb_hold,uwb_freshness_scale,"
                     "actor_goal_x,actor_goal_y,actor_goal_distance,actor_goal_reserved,"
                     "uwb_valid,uwb_age_s,uwb_error,uwb_enabled,uwb_channel,"
                     "uwb_closing,expected_closing,closing_error,"
                     "sport_valid,sport_age_s,sport_vx,sport_vy,sport_vz,sport_wz,"
                     "sport_err_vx,sport_err_vy,sport_err_wz,"
                     "feedback_source,feedback_valid,feedback_age_s,"
                     "feedback_vx,feedback_vy,feedback_vz,feedback_wz,"
                     "feedback_err_vx,feedback_err_vy,feedback_err_wz,"
                     "clr_L,clr_F,clr_R,"
                     "dep_inval,dep_meanv,front_inval,front_min,front_mean,"
                     "avx,avy,avz,pgx,pgy,pgz";
            for (int i = 0; i < 12; ++i) diag_ << ",q" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",dq" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",tau" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",action" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",target" << i;
            diag_ << '\n';
            diag_ << std::fixed << std::setprecision(6);
            spdlog::info("[VisionLoco] 诊断日志: {}", diag_path_);
        } else {
            spdlog::warn("[VisionLoco] 诊断日志打开失败: {}", diag_path_);
        }
    }

    entry_started_at_ = std::chrono::steady_clock::now();
    spdlog::info(
        "[VisionLoco] ready stand: gain blend={:.2f}s stable={:.2f}s max_wait={:.2f}s; "
        "policy takeover blend={:.2f}s",
        ready_gain_blend_s_, ready_min_stable_s_, ready_max_wait_s_, entry_blend_s_);
    running_ = true;
    policy_thread_ = std::thread([this] { policy_loop(); });
}

void State_VisionLoco::policy_loop()
{
    using clock = std::chrono::steady_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(step_dt_));
    auto next_tick = clock::now() + dt;
    const auto t_start = clock::now();
    bool motion_policy_active = false;
    auto motion_started_at = t_start;
    std::vector<float> motion_entry_target = entry_joint_pos_;

    while (running_) {
        const auto loop_start = clock::now();
        robot_->update();                          // 刷新 IMU/关节（策略序）
        const float ready_elapsed_s =
            std::chrono::duration<float>(loop_start - entry_started_at_).count();
        if (!policy_ready_) {
            float max_joint_velocity = 0.0f;
            float max_tracking_error = 0.0f;
            for (int i = 0; i < 12; ++i) {
                const float q = robot_->data.joint_pos[i];
                const float dq = robot_->data.joint_vel[i];
                if (!std::isfinite(q) || !std::isfinite(dq)) {
                    max_joint_velocity = std::numeric_limits<float>::infinity();
                    max_tracking_error = std::numeric_limits<float>::infinity();
                    break;
                }
                max_joint_velocity = std::max(max_joint_velocity, std::fabs(dq));
                max_tracking_error = std::max(
                    max_tracking_error, std::fabs(q - entry_joint_pos_[i]));
            }
            const bool posture_stable = ready_elapsed_s >= ready_gain_blend_s_ &&
                max_joint_velocity <= ready_max_joint_velocity_ &&
                max_tracking_error <= ready_max_tracking_error_;
            ready_stable_frames_ = posture_stable ? ready_stable_frames_ + 1 : 0;
            const int required_frames = std::max(
                1, static_cast<int>(std::ceil(ready_min_stable_s_ / step_dt_)));
            std::string discarded_key;
            if (FSMState::keyboard)
                FSMState::keyboard->pop_latest_key(discarded_key);
            runner_->set_cmd_override(true, 0.0f, 0.0f, 0.0f);
            if (ready_stable_frames_ >= required_frames) {
                {
                    std::lock_guard<std::mutex> lk(tgt_mtx_);
                    joint_target_ = entry_joint_pos_;
                    last_action_raw_ = stand_hold_action_raw_;
                }
                policy_ready_ = true;
                runner_->reset();
                key_target_vx_ = 0.0f;
                key_target_wz_ = 0.0f;
                key_last_input_at_ = loop_start;
                key_last_adjust_at_ = loop_start -
                    std::chrono::duration_cast<clock::duration>(
                        std::chrono::duration<float>(key_target_period_s_));
                spdlog::info(
                    "[VisionLoco] ready stand complete: max_dq={:.3f}rad/s "
                    "tracking={:.3f}rad; policy input enabled",
                    max_joint_velocity, max_tracking_error);
            } else if (ready_elapsed_s >= ready_max_wait_s_) {
                spdlog::critical(
                    "[VisionLoco][safety] ready stand did not stabilize within {:.1f}s "
                    "(max_dq={:.3f}rad/s tracking={:.3f}rad); transitioning to Passive",
                    ready_max_wait_s_, max_joint_velocity, max_tracking_error);
                policy_fault_ = true;
                running_ = false;
                break;
            }
            ++frame_;
            std::this_thread::sleep_until(next_tick);
            next_tick += dt;
            continue;
        }
        auto depth   = depth_->get();              // 57600（归一化）
        auto goal = goal_;
        std::array<float, 3> theory_cmd{};
        bool uwb_valid = false;
        float uwb_age_s = 0.0f, closing_speed = 0.0f;
        int uwb_error = 255, uwb_enabled = 0, uwb_channel = -1;
        auto exec_cmd = command_for_frame(
            loop_start, goal, theory_cmd, uwb_valid, uwb_age_s, closing_speed,
            uwb_error, uwb_enabled, uwb_channel);
        const float entry_elapsed_s =
            std::chrono::duration<float>(loop_start - entry_started_at_).count();
        std::vector<float> previous_action;
        std::vector<float> previous_target;
        {
            std::lock_guard<std::mutex> lk(tgt_mtx_);
            previous_action = last_action_raw_;
            previous_target = joint_target_;
        }
        const bool hold_stand = loco_safety::command_is_zero(
            exec_cmd, zero_command_hold_threshold_);
        const bool was_motion_policy_active = motion_policy_active;
        if (!hold_stand && !motion_policy_active) {
            runner_->reset();
            motion_policy_active = true;
            motion_started_at = loop_start;
            motion_entry_target = previous_target;
            action_step_violation_frames_ = 0;
            tracking_error_violation_frames_ = 0;
            spdlog::info(
                "[VisionLoco] non-zero command: activating policy from stand hold");
        } else if (hold_stand && motion_policy_active) {
            runner_->reset();
            motion_policy_active = false;
            action_step_violation_frames_ = 0;
            tracking_error_violation_frames_ = 0;
            spdlog::info(
                "[VisionLoco] zero command: resetting policy and returning to stand hold");
        }
        {
            std::lock_guard<std::mutex> lk(tgt_mtx_);
            loco_safety::reset_last_action_on_mode_transition(
                was_motion_policy_active, motion_policy_active,
                stand_hold_action_raw_, last_action_raw_);
        }
        if (harness_guard_mode_ && !was_motion_policy_active &&
            motion_policy_active) {
            previous_action.assign(12, 0.0f);
        }
        if (harness_guard_mode_ && motion_policy_active) {
            std::vector<float> effort(12);
            if (robot_->data.joint_effort.size() != 12) {
                spdlog::critical(
                    "[VisionLoco][harness] joint effort vector has invalid size {}; "
                    "transitioning to Passive",
                    robot_->data.joint_effort.size());
                policy_fault_ = true;
                running_ = false;
                break;
            }
            for (int i = 0; i < 12; ++i)
                effort[i] = robot_->data.joint_effort[i];
            const auto effort_violation =
                loco_safety::update_effort_violation_counts(
                    effort, harness_effort_abs_nm_,
                    harness_effort_trip_frames_,
                    harness_effort_violation_frames_);
            if (effort_violation.found) {
                spdlog::critical(
                    "[VisionLoco][harness] joint effort[{}]={:.3f}Nm exceeded "
                    "{:.3f}Nm for {} frames; transitioning to Passive",
                    effort_violation.index, effort_violation.value,
                    effort_violation.limit, harness_effort_trip_frames_);
                policy_fault_ = true;
                running_ = false;
                break;
            }
        } else {
            std::fill(
                harness_effort_violation_frames_.begin(),
                harness_effort_violation_frames_.end(), 0);
        }
        // The historical runtime started motion with zero last_action. Build the
        // observation only after applying the zero-to-motion transition reset.
        auto proprio = build_proprio();            // 45
        const float motion_elapsed_s = motion_policy_active
            ? std::chrono::duration<float>(loop_start - motion_started_at).count()
            : 0.0f;
        const bool policy_checks_active =
            motion_policy_active && motion_elapsed_s >= safety_grace_s_;
        const bool tracking_check_active = motion_policy_active
            ? policy_checks_active
            : entry_elapsed_s >= safety_grace_s_;
        if (tracking_check_active) {
            const float tracking_error_limit = motion_policy_active
                ? max_motion_tracking_error_rad_
                : max_tracking_error_rad_;
            const int tracking_error_trip_frames = motion_policy_active
                ? motion_tracking_error_trip_frames_
                : tracking_error_trip_frames_;
            float tracking_error = std::numeric_limits<float>::infinity();
            if (previous_target.size() == static_cast<size_t>(robot_->data.joint_pos.size())) {
                tracking_error = 0.0f;
                for (size_t i = 0; i < previous_target.size(); ++i) {
                    const float joint_pos =
                        robot_->data.joint_pos[static_cast<int>(i)];
                    if (!std::isfinite(joint_pos)) {
                        tracking_error = std::numeric_limits<float>::infinity();
                        break;
                    }
                    tracking_error = std::max(
                        tracking_error,
                        std::fabs(joint_pos - previous_target[i]));
                }
            }
            tracking_error_violation_frames_ = loco_safety::update_violation_count(
                tracking_error, tracking_error_limit, tracking_error_violation_frames_);
            if (loco_safety::should_trip(
                    tracking_error_violation_frames_, tracking_error_trip_frames)) {
                spdlog::critical(
                    "[VisionLoco][safety] {} joint tracking error {:.3f}rad exceeded "
                    "{:.3f}rad for {} frames; transitioning to Passive",
                    motion_policy_active ? "motion" : "stand",
                    tracking_error, tracking_error_limit,
                    tracking_error_violation_frames_);
                policy_fault_ = true;
                running_ = false;
                break;
            }
        }
        // loco 阶段速度命令始终由外部（fixed/uwb/零）注入 proprio[6:9]。
        const bool learned_nav_active = command_source_ == "nav";
        runner_->set_cmd_override(
            !learned_nav_active, exec_cmd[0], exec_cmd[1], exec_cmd[2]);
        const auto action_envelope = loco_safety::runtime_action_envelope(
            suspended_parity_mode_ || harness_guard_mode_, motion_policy_active,
            max_raw_action_abs_, max_raw_action_step_, max_target_step_rad_);
        isaaclab::LocoOutput out;
        try {
            out = runner_->act(depth, proprio, goal);  // joint(12,raw) + cmd(回显) + clearance(0)
            const auto validate_output = [](
                const char* group, const std::vector<float>& values, float limit) {
                const auto violation =
                    loco_safety::first_bounded_value_violation(values, limit);
                if (!violation.found) return;
                if (violation.non_finite) {
                    spdlog::critical(
                        "[VisionLoco][safety] ONNX output {}[{}]={} is non-finite",
                        group, violation.index, violation.value);
                } else {
                    spdlog::critical(
                        "[VisionLoco][safety] ONNX output {}[{}]={:.6f} exceeded "
                        "absolute limit {:.6f}",
                        group, violation.index, violation.value, limit);
                }
                throw std::runtime_error(
                    std::string("ONNX output safety violation in ") + group);
            };
            validate_output("joint", out.joint, action_envelope.max_raw_action_abs);
            validate_output("cmd", out.cmd, 10.0f);
            validate_output("clearance", out.clearance, 2.0f);
            if (harness_guard_mode_ && motion_policy_active) {
                const auto abs_violation =
                    loco_safety::first_per_joint_abs_violation(
                        out.joint, harness_raw_abs_limits_);
                if (abs_violation.found) {
                    spdlog::critical(
                        "[VisionLoco][harness] raw action[{}]={:.6f} exceeded "
                        "per-joint limit {:.6f}",
                        abs_violation.index, abs_violation.value,
                        abs_violation.limit);
                    throw std::runtime_error(
                        "harness per-joint raw action violation");
                }
                const auto delta_violation =
                    loco_safety::first_per_joint_delta_violation(
                        out.joint, previous_action, harness_raw_step_limits_);
                if (delta_violation.found) {
                    spdlog::critical(
                        "[VisionLoco][harness] raw action delta[{}]={:.6f} exceeded "
                        "per-joint limit {:.6f}",
                        delta_violation.index, delta_violation.value,
                        delta_violation.limit);
                    throw std::runtime_error(
                        "harness per-joint raw action delta violation");
                }
            }
            if (!learned_nav_active) {
                for (int i = 0; i < 3; ++i)
                    if (std::fabs(out.cmd[i] - exec_cmd[i]) > 1e-4f)
                        throw std::runtime_error("cmd override 隔离校验失败");
            } else {
                // loco 阶段 nav 分支恒零；cmd 回显为 0，与 exec_cmd 一致。
                std::copy(out.cmd.begin(), out.cmd.end(), exec_cmd.begin());
                theory_cmd = exec_cmd;
                last_exec_cmd_ = exec_cmd;
            }
            consecutive_errors_ = 0;
        } catch (const std::exception& e) {
            ++consecutive_errors_;
            spdlog::error("[VisionLoco] 推理异常 ({}/{}): {}",
                          consecutive_errors_, max_consecutive_errors_, e.what());
            if (consecutive_errors_ >= max_consecutive_errors_) {
                spdlog::critical("[VisionLoco] inference fault; transitioning to Passive");
                policy_fault_ = true;
                running_ = false;
                break;
            }
            std::this_thread::sleep_until(next_tick);
            next_tick += dt;
            continue;
        }

        if (policy_checks_active) {
            const float raw_action_step =
                loco_safety::max_abs_delta(out.joint, previous_action);
            action_step_violation_frames_ = loco_safety::update_violation_count(
                raw_action_step, action_envelope.max_raw_action_step,
                action_step_violation_frames_);
            if (loco_safety::should_trip(
                    action_step_violation_frames_, action_step_trip_frames_)) {
                spdlog::critical(
                    "[VisionLoco][safety] raw action step {:.3f} exceeded {:.3f} "
                    "for {} frames; transitioning to Passive",
                    raw_action_step, action_envelope.max_raw_action_step,
                    action_step_violation_frames_);
                policy_fault_ = true;
                running_ = false;
                break;
            }
        }
        std::vector<float> tgt(12);
        std::vector<float> unmodified_target(12);
        for (int i = 0; i < 12; ++i) {
            float blended_target = entry_joint_pos_[i];
            unmodified_target[i] = entry_joint_pos_[i];
            if (motion_policy_active) {
                const float policy_target = std::clamp(
                    act_offset_[i] + act_scale_ * out.joint[i],
                    act_clip_lo_, act_clip_hi_);
                unmodified_target[i] = policy_target;
                blended_target = loco_safety::policy_transition_target(
                    motion_entry_target[i], policy_target,
                    motion_elapsed_s, entry_blend_s_,
                    action_envelope.bypass_entry_blend);
            }
            tgt[i] = loco_safety::limit_target_step(
                previous_target[i], blended_target,
                action_envelope.max_target_step_rad);
        }
        if (harness_guard_mode_ && motion_policy_active) {
            const auto target_violation =
                loco_safety::first_per_joint_range_violation(
                    unmodified_target, harness_target_min_, harness_target_max_);
            if (target_violation.found) {
                spdlog::critical(
                    "[VisionLoco][harness] policy target[{}]={:.6f} crossed "
                    "historical range boundary {:.6f}; transitioning to Passive",
                    target_violation.index, target_violation.value,
                    target_violation.limit);
                policy_fault_ = true;
                running_ = false;
                break;
            }
            const float target_distortion =
                loco_safety::max_abs_delta(tgt, unmodified_target);
            if (!std::isfinite(target_distortion) ||
                target_distortion > 1.0e-5f) {
                spdlog::critical(
                    "[VisionLoco][harness] target transparency error {:.6f}rad; "
                    "transitioning to Passive",
                    target_distortion);
                policy_fault_ = true;
                running_ = false;
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lk(tgt_mtx_);
            joint_target_    = tgt;
            last_action_raw_ = out.joint;          // 回喂用原始动作
            have_target_     = true;
        }
        if (!motion_policy_active) {
            std::lock_guard<std::mutex> lk(tgt_mtx_);
            last_action_raw_ = stand_hold_action_raw_;
        }

        const auto work_end = clock::now();
        const float loop_ms =
            std::chrono::duration<float, std::milli>(work_end - loop_start).count();
        if (work_end > next_tick) ++deadline_misses_;

        if ((frame_ % 50) == 0) {
            spdlog::info("[VisionLoco] mode={} source={} theory=[{:.3f},{:.3f},{:.3f}] "
                         "exec=[{:.3f},{:.3f},{:.3f}] uwb_ok={} "
                         "uwb_planar={:.3f} arrived={} hold={} fresh={:.2f} "
                         "goal=[{:.2f},{:.2f},{:.2f},{:.2f}] "
                         "clr=[{:.2f},{:.2f},{:.2f}] infer={:.2f}ms loop={:.2f}ms miss={}",
                         motion_policy_active ? "policy" : "stand_hold",
                         command_source_, theory_cmd[0], theory_cmd[1], theory_cmd[2],
                         out.cmd[0], out.cmd[1], out.cmd[2],
                         uwb_valid, last_uwb_planar_distance_, uwb_arrived_,
                         last_uwb_hold_, last_uwb_freshness_scale_,
                         goal[0], goal[1], goal[2], goal[3],
                         out.clearance[0], out.clearance[1], out.clearance[2],
                         out.inference_ms, loop_ms, deadline_misses_);
        }

        // 逐帧 CSV：时延 / cmd / clearance / depth / IMU / q,dq / action,target。
        if (diag_) {
            DepthStats ds = depth_stats(depth);
            SportSample sport;
            UwbSample uwb;
            {
                std::lock_guard<std::mutex> lk(sensor_mtx_);
                sport = sport_;
                uwb = uwb_;
            }
            const float sport_age_s = sport.received
                ? std::chrono::duration<float>(clock::now() - sport.received_at).count()
                : -1.0f;
            const bool sport_valid = sport.received && sport_age_s <= 0.5f;
            const float closing_beta =
                command_source_ == "uwb" && uwb_valid ? uwb.filtered_beta : 0.0f;
            const float expected_closing =
                exec_cmd[0] * std::cos(closing_beta) +
                exec_cmd[1] * std::sin(closing_beta);
            const float closing_error = uwb_valid ? closing_speed - expected_closing : 0.0f;
            const auto& av = robot_->data.root_ang_vel_b;
            const auto& pg = robot_->data.projected_gravity_b;
            const float uwb_velocity_age_s = uwb.velocity_valid
                ? std::chrono::duration<float>(
                      clock::now() - uwb.velocity_received_at).count()
                : -1.0f;
            const bool uwb_velocity_valid =
                uwb.velocity_valid && uwb_velocity_age_s <= uwb_stale_timeout_s_;
            const int feedback_source = sport_valid ? 1 : (uwb_velocity_valid ? 2 : 0);
            const bool feedback_valid = feedback_source != 0;
            const float feedback_age_s =
                sport_valid ? sport_age_s :
                (uwb_velocity_valid ? uwb_velocity_age_s : -1.0f);
            const float feedback_vx =
                sport_valid ? sport.velocity[0] :
                (uwb_velocity_valid ? uwb.body_velocity[0] : 0.0f);
            const float feedback_vy =
                sport_valid ? sport.velocity[1] :
                (uwb_velocity_valid ? uwb.body_velocity[1] : 0.0f);
            const float feedback_vz = sport_valid ? sport.velocity[2] : 0.0f;
            const float feedback_wz = sport_valid ? sport.yaw_speed : av[2];
            const auto& q = robot_->data.joint_pos;
            const auto& dq = robot_->data.joint_vel;
            const auto& tau = robot_->data.joint_effort;
            long t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now() - t_start).count();
            diag_ << frame_ << ',' << t_ms << ',' << loop_ms << ',' << out.inference_ms
                  << ',' << deadline_misses_ << ',' << consecutive_errors_;
            write_values(diag_, out.cmd);
            write_values(diag_, out.cmd_raw);
            write_values(diag_, theory_cmd);
            const int source_code =
                command_source_ == "fixed" ? 0 : (command_source_ == "uwb" ? 1 : (command_source_ == "keyboard" ? 3 : 2));
            diag_ << ',' << source_code;
            write_values(diag_, uwb.raw);
            diag_ << ',' << last_uwb_planar_distance_
                  << ',' << (uwb_arrived_ ? 1 : 0)
                  << ',' << (uwb_stop_once_ ? 1 : 0)
                  << ',' << (uwb_goal_driven_ ? 1 : 0)
                  << ',' << last_uwb_filtered_x_ << ',' << last_uwb_filtered_y_
                  << ',' << last_uwb_filtered_beta_ << ',' << last_uwb_filter_age_s_
                  << ',' << (last_uwb_hold_ ? 1 : 0)
                  << ',' << last_uwb_freshness_scale_;
            write_values(diag_, goal);
            diag_ << ',' << (uwb_valid ? 1 : 0) << ',' << uwb_age_s
                  << ',' << uwb_error << ',' << uwb_enabled << ',' << uwb_channel
                  << ',' << closing_speed << ',' << expected_closing << ',' << closing_error
                  << ',' << (sport_valid ? 1 : 0) << ',' << sport_age_s;
            write_values(diag_, sport.velocity);
            diag_ << ',' << sport.yaw_speed
                  << ',' << (sport_valid ? sport.velocity[0] - exec_cmd[0] : 0.0f)
                  << ',' << (sport_valid ? sport.velocity[1] - exec_cmd[1] : 0.0f)
                  << ',' << (sport_valid ? sport.yaw_speed - exec_cmd[2] : 0.0f)
                  << ',' << feedback_source << ',' << (feedback_valid ? 1 : 0)
                  << ',' << feedback_age_s
                  << ',' << feedback_vx << ',' << feedback_vy << ',' << feedback_vz
                  << ',' << feedback_wz
                  << ',' << (feedback_valid ? feedback_vx - exec_cmd[0] : 0.0f)
                  << ',' << (feedback_valid ? feedback_vy - exec_cmd[1] : 0.0f)
                  << ',' << (feedback_valid ? feedback_wz - exec_cmd[2] : 0.0f);
            write_values(diag_, out.clearance);
            diag_ << ',' << ds.invalid_frac << ',' << ds.mean_valid
                  << ',' << ds.front_invalid << ',' << ds.front_min << ',' << ds.front_mean
                  << ',' << av[0] << ',' << av[1] << ',' << av[2]
                  << ',' << pg[0] << ',' << pg[1] << ',' << pg[2];
            for (int i = 0; i < q.size(); ++i) diag_ << ',' << q[i];
            for (int i = 0; i < dq.size(); ++i) diag_ << ',' << dq[i];
            for (int i = 0; i < tau.size(); ++i) diag_ << ',' << tau[i];
            write_values(diag_, out.joint);
            write_values(diag_, tgt);
            diag_ << '\n';
            if ((frame_ % log_flush_every_) == 0) diag_.flush();
        }

        ++frame_;
        std::this_thread::sleep_until(next_tick);
        next_tick += dt;
        if (clock::now() > next_tick + dt) next_tick = clock::now() + dt;
    }
}

void State_VisionLoco::run()
{
    std::lock_guard<std::mutex> lk(tgt_mtx_);
    if (!have_target_) return;
    const float elapsed_s = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - entry_started_at_).count();
    const float gain_alpha = std::clamp(
        elapsed_s / ready_gain_blend_s_, 0.0f, 1.0f);
    for (int i = 0; i < (int)joint_ids_map_.size(); ++i) {
        auto& motor = lowcmd->msg_.motor_cmd()[joint_ids_map_[i]];
        motor.kp() = entry_stiffness_[i] +
            gain_alpha * (stiffness_[i] - entry_stiffness_[i]);
        motor.kd() = entry_damping_[i] +
            gain_alpha * (damping_[i] - entry_damping_[i]);
        motor.q() = policy_ready_
            ? joint_target_[i]
            : entry_command_target_[i] + gain_alpha *
                (entry_joint_pos_[i] - entry_command_target_[i]);
    }
}

void State_VisionLoco::exit()
{
    running_ = false;
    if (policy_thread_.joinable()) policy_thread_.join();
    // The policy thread is the keyboard event consumer; release input only
    // after it has stopped to avoid a use-after-free during state transition.
    if (FSMState::keyboard) {
        FSMState::keyboard.reset();
        spdlog::info("[VisionLoco] Keyboard released");
    }
    if (diag_) { diag_.flush(); diag_.close(); spdlog::info("[VisionLoco] 诊断日志已保存: {}", diag_path_); }
}
