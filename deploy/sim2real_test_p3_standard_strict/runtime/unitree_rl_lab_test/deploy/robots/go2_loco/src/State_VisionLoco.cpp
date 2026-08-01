// Copyright (c) 2026.
// State_VisionLoco 实现，见同名头文件说明。
//
// 与 State_VisionNav.cpp 逐行一致，仅把推理核换成 LocoRunner / LocoOutput，
// 日志标签改为 [VisionLoco]。obs 装配、UWB、command_for_frame、诊断 CSV 全保留。

#include "State_VisionLoco.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace
{
// 安全检查：翻倒（与 isaaclab::mdp::bad_orientation 同式，基于 body 系投影重力 z 分量）
inline bool bad_orientation(const Eigen::Vector3f& proj_g, float limit_angle = 1.0f)
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

inline bool finite_and_bounded(const std::vector<float>& values, float abs_limit)
{
    for (float value : values)
        if (!std::isfinite(value) || std::fabs(value) > abs_limit) return false;
    return true;
}

template <typename Container>
void write_values(std::ofstream& out, const Container& values)
{
    for (const auto& value : values) out << ',' << value;
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
    command_source_ = yaml_get<std::string>(cfg, "command_source", "fixed");
    if (command_source_ != "fixed" && command_source_ != "uwb" &&
        command_source_ != "nav")
        throw std::runtime_error("command_source 必须是 fixed、uwb 或 nav");
    try {
        auto c = cfg["fixed_cmd"].as<std::vector<float>>();
        if (c.size() == 3) std::copy(c.begin(), c.end(), fixed_cmd_.begin());
    } catch (...) {}
    if (cfg["uwb"]) {
        auto u = cfg["uwb"];
        uwb_topic_ = yaml_get<std::string>(u, "topic", uwb_topic_);
        sport_topic_ = yaml_get<std::string>(u, "sport_state_topic", sport_topic_);
        max_vx_ = std::clamp(yaml_get<float>(u, "max_vx", max_vx_), 0.0f, 1.5f);
        max_vy_ = std::clamp(yaml_get<float>(u, "max_vy", max_vy_), 0.0f, 0.15f);
        max_wz_ = std::clamp(yaml_get<float>(u, "max_wz", max_wz_), 0.0f, 1.0f);
        yaw_kp_ = std::max(0.0f, yaml_get<float>(u, "yaw_kp", yaw_kp_));
        stop_distance_ = std::max(0.0f, yaml_get<float>(u, "stop_distance", stop_distance_));
        slow_distance_ = std::max(
            stop_distance_ + 0.05f, yaml_get<float>(u, "slow_distance", slow_distance_));
        turn_in_place_angle_ = std::clamp(
            yaml_get<float>(u, "turn_in_place_angle", turn_in_place_angle_),
            0.1f, 3.1415926f);
        uwb_stale_timeout_s_ = std::max(
            0.1f, yaml_get<float>(u, "stale_timeout", uwb_stale_timeout_s_));
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

    // ---------- 深度源 ----------
    std::string dsrc = "constant";
    float dval = 1.0f;
    if (cfg["depth"]) {
        dsrc = yaml_get<std::string>(cfg["depth"], "source", "constant");
        dval = yaml_get<float>(cfg["depth"], "constant_value", 1.0f);
    }
    if (dsrc == "realsense") {
#ifdef USE_REALSENSE
        vision_nav::RealSenseConfig camera;
        if (cfg["depth"]["camera_options"]) {
            const auto options = cfg["depth"]["camera_options"];
            const auto preset = yaml_get<std::string>(options, "visual_preset", "high_density");
            if (preset != "high_density" && preset != "default")
                throw std::runtime_error("camera visual_preset must be high_density or default");
            camera.high_density_preset = preset == "high_density";
            camera.emitter_enabled = yaml_get<bool>(options, "emitter_enabled", true);
            const auto laser = yaml_get<std::string>(options, "laser_power", "max");
            if (laser != "max" && laser != "default")
                throw std::runtime_error("camera laser_power must be max or default");
            camera.max_laser_power = laser == "max";
            camera.auto_exposure = yaml_get<bool>(options, "auto_exposure", true);
        }
        if (cfg["depth"]["filters"]) {
            const auto f = cfg["depth"]["filters"];
            camera.filters.mode = yaml_get<std::string>(f, "mode", camera.filters.mode);
            camera.filters.spatial_magnitude = std::clamp(
                yaml_get<int>(f, "spatial_magnitude", camera.filters.spatial_magnitude), 1, 5);
            camera.filters.spatial_smooth_alpha = std::clamp(
                yaml_get<float>(f, "spatial_smooth_alpha", camera.filters.spatial_smooth_alpha),
                0.25f, 1.0f);
            camera.filters.spatial_smooth_delta = std::clamp(
                yaml_get<int>(f, "spatial_smooth_delta", camera.filters.spatial_smooth_delta), 1, 50);
            camera.filters.temporal_alpha = std::clamp(
                yaml_get<float>(f, "temporal_alpha", camera.filters.temporal_alpha), 0.0f, 1.0f);
            camera.filters.temporal_delta = std::clamp(
                yaml_get<float>(f, "temporal_delta", camera.filters.temporal_delta), 1.0f, 100.0f);
        }
        if (camera.filters.mode != "none" && camera.filters.mode != "light_spatial" &&
            camera.filters.mode != "light_spatial_weak_temporal") {
            throw std::runtime_error("unknown RealSense filter mode: " + camera.filters.mode);
        }
        depth_ = std::make_unique<vision_nav::RealSenseDepth>(424, 240, 30, camera);
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
            std::max(1, yaml_get<int>(cfg["logging"], "max_consecutive_errors", 5));
        max_raw_action_abs_ =
            std::max(1.0f, yaml_get<float>(cfg["logging"], "max_raw_action_abs", 20.0f));
        max_target_step_rad_ = std::max(
            0.05f, yaml_get<float>(cfg["logging"], "max_target_step_rad", 0.35f));
        target_slew_rate_rad_s_ = std::max(
            0.1f, yaml_get<float>(cfg["logging"], "target_slew_rate_rad_s", 3.0f));
        max_tracking_error_rad_ = std::max(
            0.10f, yaml_get<float>(cfg["logging"], "max_tracking_error_rad", 0.45f));
        max_consecutive_motion_violations_ = std::max(
            1, yaml_get<int>(cfg["logging"], "max_consecutive_motion_violations", 2));
    }

    auto strict = cfg["strict_safety"];
    if (!strict) throw std::runtime_error("strict_safety configuration is required");
    lowstate_stale_ms_ = yaml_get<float>(strict, "lowstate_stale_ms", 50.0f);
    depth_stale_ms_ = yaml_get<float>(strict, "depth_stale_ms", 150.0f);
    depth_fault_capture_enabled_ = yaml_get<bool>(
        strict, "depth_fault_capture_enabled", true);
    depth_capture_dump_on_exit_ = yaml_get<bool>(
        strict, "depth_capture_dump_on_exit", false);
    depth_fault_capture_frames_ = static_cast<size_t>(std::max(
        1, yaml_get<int>(strict, "depth_fault_capture_frames", 90)));
    policy_target_stale_ms_ = yaml_get<float>(strict, "policy_target_stale_ms", 100.0f);
    inference_deadline_ms_ = yaml_get<float>(strict, "inference_deadline_ms", 20.0f);
    stable_lowstate_frames_required_ = std::max(
        1, yaml_get<int>(strict, "stable_lowstate_frames", 100));
    stable_depth_frames_required_ = std::max(
        1, yaml_get<int>(strict, "stable_depth_frames", 60));
    shadow_gate_seconds_ = std::max(
        0.0f, yaml_get<float>(strict, "shadow_gate_seconds", 2.0f));
    startup_shadow_enabled_ = yaml_get<bool>(strict, "startup_shadow_enabled", false);
    suspended_test_bypass_ = yaml_get<bool>(strict, "suspended_test_bypass", false);
    ground_test_permissive_ = yaml_get<bool>(strict, "ground_test_permissive", false);
    walking_test_permissive_ = yaml_get<bool>(strict, "walking_test_permissive", false);
    stable_max_joint_velocity_rad_s_ = std::max(
        0.0f, yaml_get<float>(strict, "stable_max_joint_velocity_rad_s", 0.5f));
    stable_max_tilt_rad_ = std::max(
        0.0f, yaml_get<float>(strict, "stable_max_tilt_rad", 0.35f));
    joint_lower_ = strict["joint_lower"].as<std::vector<float>>();
    joint_upper_ = strict["joint_upper"].as<std::vector<float>>();
    dq_warning_ = strict["dq_warning"].as<std::vector<float>>();
    dq_hard_fault_ = strict["dq_hard_fault"].as<std::vector<float>>();
    tau_warning_ = strict["tau_warning"].as<std::vector<float>>();
    tau_soft_stop_ = strict["tau_soft_stop"].as<std::vector<float>>();
    tau_hard_fault_ = strict["tau_hard_fault"].as<std::vector<float>>();
    motor_temperature_warning_c_ = yaml_get<float>(strict, "motor_temperature_warning_c", 65.0f);
    motor_temperature_hard_c_ = yaml_get<float>(strict, "motor_temperature_hard_c", 75.0f);
    for (const auto* values : {&joint_lower_, &joint_upper_, &dq_warning_, &dq_hard_fault_,
                               &tau_warning_, &tau_soft_stop_, &tau_hard_fault_})
        if (values->size() != 12) throw std::runtime_error("strict_safety joint arrays must have 12 values");

    // ---------- 安全转移：翻倒 → Passive ----------
    this->registered_checks.push_back({
        [this]() -> bool {
            const auto snapshot = snapshot_store_.latest();
            const Eigen::Vector3f gravity(
                snapshot.projected_gravity[0], snapshot.projected_gravity[1],
                snapshot.projected_gravity[2]);
            const bool triggered = snapshot.valid && bad_orientation(gravity, 1.0f);
            if (triggered && !bad_orientation_logged_.exchange(true)) {
                const float tilt = std::acos(std::clamp(-gravity[2], -1.0f, 1.0f));
                spdlog::critical(
                    "[VisionLoco][BAD_ORIENTATION] frame={} tilt_rad={:.4f} limit_rad=1.0000 "
                    "projected_gravity=[{:.4f},{:.4f},{:.4f}] lowstate_age_ms={:.3f}",
                    last_policy_frame_.load(), tilt, gravity[0], gravity[1], gravity[2],
                    snapshot.lowstate_age_ms);
                log_fault_snapshot("HardFault", "bad_orientation", snapshot);
            }
            return triggered;
        },
        FSMStringMap.right.at("Passive"),
        []()->std::string { return "bad_orientation"; }
    });
    this->registered_checks.push_back({
        [this]() -> bool {
            return policy_fault_.load() ||
                   (safety_state_.takeover_requested.load() &&
                    safety_state_.severity.load() == strict_loco::FaultSeverity::HardFault);
        },
        FSMStringMap.right.at("Passive"),
        [this]()->std::string { return "hard_fault:" + safety_state_.fault_reason(); }
    });
    this->registered_checks.push_back({
        [this]() -> bool {
            return motion_fault_.load() ||
                   (safety_state_.takeover_requested.load() &&
                    safety_state_.severity.load() == strict_loco::FaultSeverity::SoftStop);
        },
        FSMStringMap.right.at("FixStand"),
        [this]()->std::string { return "soft_stop:" + safety_state_.fault_reason(); }
    });

    spdlog::info("[VisionLoco] 就绪：step_dt={:.3f}s scale={:.3f} source={} "
                 "goal=[{:.2f},{:.2f},{:.2f},{:.2f}] depth={} logging={}",
                 step_dt_, act_scale_, command_source_,
                 goal_[0], goal_[1], goal_[2], goal_[3],
                 dsrc, logging_enabled_);
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
        }) &&
        std::isfinite(msg->base_yaw());
    const bool previous_valid =
        uwb_.received && uwb_.error_state == 0 && uwb_.enabled_from_app == 1 &&
        std::all_of(uwb_.goal.begin(), uwb_.goal.end(), [](float value) {
            return std::isfinite(value);
        }) &&
        std::isfinite(uwb_.base_yaw);
    if (current_valid && previous_valid) {
        const float dt = std::chrono::duration<float>(now - uwb_.received_at).count();
        if (dt >= 0.03f && dt <= 1.0f && std::isfinite(next[2]) && std::isfinite(uwb_.goal[2])) {
            const float instantaneous = std::clamp((uwb_.goal[2] - next[2]) / dt, -2.0f, 2.0f);
            closing = 0.8f * closing + 0.2f * instantaneous;

            // 将“机器人到静止 UWB tag”的相对向量转到公共航向系，
            // 对其取负差分得到机器人平面速度，再转回当前机体系。
            const float previous_range = uwb_.goal[2] * std::cos(uwb_.goal[1]);
            const float current_range = next[2] * std::cos(next[1]);
            const float previous_heading = uwb_.base_yaw + uwb_.goal[0];
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
    uwb_.goal = next;
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

    if (command_source_ == "fixed") {
        theory_cmd[0] = std::clamp(theory_cmd[0], -0.50f, 1.00f);
        theory_cmd[1] = std::clamp(theory_cmd[1], -0.15f, 0.15f);
        theory_cmd[2] = std::clamp(theory_cmd[2], -1.00f, 1.00f);
    } else if (command_source_ == "uwb") {
        UwbSample sample;
        {
            std::lock_guard<std::mutex> lk(sensor_mtx_);
            sample = uwb_;
        }
        if (sample.received)
            uwb_age_s = std::chrono::duration<float>(now - sample.received_at).count();
        uwb_error = sample.error_state;
        uwb_enabled = sample.enabled_from_app;
        uwb_channel = sample.channel;
        const bool finite = std::all_of(
            sample.goal.begin(), sample.goal.end(), [](float v) { return std::isfinite(v); });
        uwb_valid = sample.received && finite && sample.error_state == 0 &&
                    sample.enabled_from_app == 1 && sample.goal[2] >= 0.0f &&
                    uwb_age_s <= uwb_stale_timeout_s_;
        closing_speed = sample.closing_speed;
        theory_cmd = {0.0f, 0.0f, 0.0f};
        if (uwb_valid) {
            goal.assign(sample.goal.begin(), sample.goal.end());
            const float beta = sample.goal[0];
            const float distance = sample.goal[2];
            if (distance > stop_distance_) {
                const float distance_scale = std::clamp(
                    (distance - stop_distance_) / (slow_distance_ - stop_distance_),
                    0.0f, 1.0f);
                const float heading_scale = std::max(0.0f, std::cos(beta));
                theory_cmd[0] = std::fabs(beta) >= turn_in_place_angle_
                                    ? 0.0f
                                    : max_vx_ * distance_scale * heading_scale;
                theory_cmd[1] = std::clamp(max_vy_ * std::sin(beta), -max_vy_, max_vy_);
                theory_cmd[2] = std::clamp(yaw_kp_ * beta, -max_wz_, max_wz_);
            }
        }
    } else {
        // loco 阶段无 learned nav actor（模型无 nav 分支）：nav 命令源等价零速度，
        // 仅保留分支以复用 State。goal 使用 config.yaml 固定值，不启动 UWB。
        theory_cmd = {0.0f, 0.0f, 0.0f};
    }

    std::array<float, 3> exec{};
    for (int i = 0; i < 3; ++i) {
        const float max_step = cmd_slew_rate_[i] * step_dt_;
        exec[i] = last_exec_cmd_[i] +
                  std::clamp(theory_cmd[i] - last_exec_cmd_[i], -max_step, max_step);
    }
    if (command_source_ == "uwb" && !uwb_valid)
        exec = {0.0f, 0.0f, 0.0f};
    last_exec_cmd_ = exec;
    last_cmd_vx_ = exec[0];
    last_cmd_vy_ = exec[1];
    last_cmd_wz_ = exec[2];
    return exec;
}

std::vector<float> State_VisionLoco::build_proprio(
    const strict_loco::SensorSnapshot& snapshot)
{
    // 策略序：ang_vel(3) proj_g(3) vel_cmd(3,留0待runner填上一帧cmd)
    //         joint_pos_rel(12) joint_vel_rel(12) last_action(12)
    std::vector<float> p(45, 0.0f);
    for (int i = 0; i < 3; ++i) p[0 + i] = snapshot.angular_velocity[i] * sc_ang_vel_;
    for (int i = 0; i < 3; ++i) p[3 + i] = snapshot.projected_gravity[i] * sc_proj_g_;
    // p[6:9] 由 runner 写为上一帧 cmd
    for (int i = 0; i < 12; ++i)
        p[9 + i] = (snapshot.q[i] - default_joint_pos_[i]) * sc_jpos_;
    for (int i = 0; i < 12; ++i) p[21 + i] = snapshot.dq[i] * sc_jvel_;
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
    const auto initial_snapshot = snapshot_store_.capture(FSMState::lowstate, joint_ids_map_);
    if (!initial_snapshot.valid)
        throw std::runtime_error("cannot enter VisionLoco with invalid LowState: " +
                                 initial_snapshot.invalid_reason);

    // 设 PD 增益（全 12 关节）
    for (int i = 0; i < (int)stiffness_.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = stiffness_[i];
        lowcmd->msg_.motor_cmd()[i].kd() = damping_[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    // 初始目标 = 当前实测关节位置；shadow 门控通过前不发布策略目标。
    {
        std::lock_guard<std::mutex> lk(tgt_mtx_);
        joint_target_.assign(initial_snapshot.q.begin(), initial_snapshot.q.end());
        last_action_raw_ = std::vector<float>(12, 0.0f);
        have_target_     = true;
        policy_target_at_ = std::chrono::steady_clock::now();
    }
    runner_->reset();
    frame_ = 0;
    deadline_misses_ = 0;
    consecutive_errors_ = 0;
    policy_fault_ = false;
    motion_fault_ = false;
    safety_state_.reset();
    depth_capture_ring_.clear();
    last_recorded_depth_frame_ = 0;
    depth_capture_requested_ = false;
    depth_warning_active_ = false;
    last_tracking_error_max_ = 0.0f;
    last_requested_target_step_max_ = 0.0f;
    last_target_step_max_ = 0.0f;
    last_pd_tau_abs_max_ = 0.0f;
    last_depth_invalid_fraction_ = 1.0f;
    last_depth_front_invalid_fraction_ = 1.0f;
    last_depth_age_ms_ = 0.0f;
    last_depth_frame_number_ = 0;
    last_policy_frame_ = 0;
    last_cmd_vx_ = 0.0f;
    last_cmd_vy_ = 0.0f;
    last_cmd_wz_ = 0.0f;
    bad_orientation_logged_ = false;
    target_stale_logged_ = false;
    allow_target_publish_ = false;
    policy_target_active_ = false;
    last_exec_cmd_ = {0.0f, 0.0f, 0.0f};

    // fixed/uwb 始终外部覆盖；nav 使用 config.yaml 固定 goal（loco 阶段等价零速度）。
    runner_->set_cmd_override(true, 0.0f, 0.0f, 0.0f);
    if (command_source_ == "uwb" || uwb_diagnostic_feedback_) {
        try {
            const int ret = utrack_->SwitchSet(true);
            bool enabled = false, tracking = false;
            utrack_->SwitchGet(enabled);
            utrack_->IsTracking(tracking);
            if (command_source_ == "uwb") {
                spdlog::warn("[VisionLoco] UWB 外部控制：SwitchSet ret={} enabled={} tracking={}，"
                             "vx 硬上限 {:.2f}m/s。",
                             ret, enabled, tracking, max_vx_);
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
    if (suspended_test_bypass_) {
        spdlog::critical(
            "[VisionLoco] SUSPENDED TEST: optional startup stability gate bypassed; "
            "fixed-zero command and all runtime watchdogs remain active");
    }
    if (ground_test_permissive_) {
        spdlog::critical(
            "[VisionLoco] GROUND CHARACTERIZATION: optional startup and tracking soft gates disabled; "
            "fixed-zero, slew, physical limits, freshness, torque and hard watchdogs remain active");
    }
    if (walking_test_permissive_) {
        spdlog::critical(
            "[VisionLoco] WALK CHARACTERIZATION: first valid post-reset inference publishes; "
            "post-slew motion audit is used and the tracking soft gate is disabled; "
            "physical, freshness, velocity, torque, temperature and takeover guards remain active");
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
                     "uwb_valid,uwb_age_s,uwb_error,uwb_enabled,uwb_channel,"
                     "uwb_closing,expected_closing,closing_error,"
                     "sport_valid,sport_age_s,sport_vx,sport_vy,sport_vz,sport_wz,"
                     "sport_err_vx,sport_err_vy,sport_err_wz,"
                     "feedback_source,feedback_valid,feedback_age_s,"
                     "feedback_vx,feedback_vy,feedback_vz,feedback_wz,"
                     "feedback_err_vx,feedback_err_vy,feedback_err_wz,"
                     "clr_L,clr_F,clr_R,"
                     "dep_inval,dep_meanv,front_inval,front_min,front_mean,"
                     "camera_invalid_fraction,camera_front_invalid_fraction,"
                     "action_step_max,requested_target_step_max,target_step_max,tracking_error_max,"
                     "motion_rejected,motion_violations,safety_modified_count,safety_modified_rate,"
                     "shadow_gate_ready,suspended_test_bypass,ground_test_permissive,walking_test_permissive,"
                     "stable_lowstate_count,stable_depth_count,"
                     "sensor_sequence,lowstate_tick,lowstate_age_ms,depth_age_ms,depth_frame_number,depth_sensor_timestamp_ms,"
                     "avx,avy,avz,pgx,pgy,pgz,"
                     "tau_abs_max,pd_tau_abs_max,mechanical_power_abs_sum";
            for (int i = 0; i < 12; ++i) diag_ << ",q" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",dq" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",tau" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",model_raw_action" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",clipped_raw_action" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",requested_target" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",slew_target" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",physical_target" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",applied_target" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",executed_raw_action" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",qerr" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",pd_tau" << i;
            diag_ << '\n';
            diag_ << std::fixed << std::setprecision(6);
            spdlog::info("[VisionLoco] 诊断日志: {}", diag_path_);
        } else {
            spdlog::warn("[VisionLoco] 诊断日志打开失败: {}", diag_path_);
        }
    }

    running_ = true;
    policy_thread_ = std::thread([this] {
        try {
            policy_loop();
        } catch (const std::exception& e) {
            const std::string reason = std::string("policy_thread_exception: ") + e.what();
            safety_state_.fault(strict_loco::FaultSeverity::HardFault, reason);
            policy_fault_ = true;
            allow_target_publish_ = false;
            running_ = false;
            spdlog::critical("[VisionLoco] uncaught policy-thread exception: {}", e.what());
            log_fault_snapshot("HardFault", reason, snapshot_store_.latest());
        } catch (...) {
            safety_state_.fault(strict_loco::FaultSeverity::HardFault,
                                "policy_thread_exception: unknown");
            policy_fault_ = true;
            allow_target_publish_ = false;
            running_ = false;
            spdlog::critical("[VisionLoco] uncaught policy-thread exception: unknown");
            log_fault_snapshot(
                "HardFault", "policy_thread_exception: unknown", snapshot_store_.latest());
        }
    });
}

void State_VisionLoco::log_fault_snapshot(
    const char* severity, const std::string& reason,
    const strict_loco::SensorSnapshot& snapshot) const
{
    int q_joint = 0;
    int dq_joint = 0;
    int tau_joint = 0;
    int temp_joint = 0;
    for (int i = 1; i < 12; ++i) {
        if (std::fabs(snapshot.q[i]) > std::fabs(snapshot.q[q_joint])) q_joint = i;
        if (std::fabs(snapshot.dq[i]) > std::fabs(snapshot.dq[dq_joint])) dq_joint = i;
        if (std::fabs(snapshot.tau_est[i]) > std::fabs(snapshot.tau_est[tau_joint])) tau_joint = i;
        if (snapshot.motor_temperature_c[i] > snapshot.motor_temperature_c[temp_joint]) temp_joint = i;
    }
    const float tilt_rad = std::acos(std::clamp(
        -snapshot.projected_gravity[2], -1.0f, 1.0f));
    spdlog::critical(
        "[VisionLoco][FAULT_SNAPSHOT] severity={} reason={} policy_frame={} "
        "sensor_valid={} sensor_reason={} sensor_sequence={} lowstate_tick={} "
        "lowstate_age_ms={:.3f} cmd=[{:.3f},{:.3f},{:.3f}] "
        "projected_gravity=[{:.4f},{:.4f},{:.4f}] tilt_rad={:.4f} "
        "max_abs_q_joint={} max_abs_q={:.5f} max_abs_dq_joint={} max_abs_dq={:.5f} "
        "max_abs_tau_joint={} max_abs_tau={:.3f} max_temp_joint={} max_temp_c={:.1f} "
        "battery_v={:.2f} battery_soc={:.1f} tracking_error_max={:.5f} "
        "requested_step_max={:.5f} applied_step_max={:.5f} pd_tau_abs_max={:.3f} "
        "depth_frame={} depth_age_ms={:.3f} depth_invalid={:.4f} depth_front_invalid={:.4f}",
        severity, reason, last_policy_frame_.load(), snapshot.valid,
        snapshot.invalid_reason, snapshot.sequence, snapshot.lowstate_tick,
        snapshot.lowstate_age_ms, last_cmd_vx_.load(), last_cmd_vy_.load(),
        last_cmd_wz_.load(), snapshot.projected_gravity[0],
        snapshot.projected_gravity[1], snapshot.projected_gravity[2], tilt_rad,
        q_joint, std::fabs(snapshot.q[q_joint]), dq_joint,
        std::fabs(snapshot.dq[dq_joint]), tau_joint,
        std::fabs(snapshot.tau_est[tau_joint]), temp_joint,
        snapshot.motor_temperature_c[temp_joint], snapshot.battery_voltage_v,
        snapshot.battery_soc, last_tracking_error_max_.load(),
        last_requested_target_step_max_.load(), last_target_step_max_.load(),
        last_pd_tau_abs_max_.load(), last_depth_frame_number_.load(),
        last_depth_age_ms_.load(), last_depth_invalid_fraction_.load(),
        last_depth_front_invalid_fraction_.load());
}

void State_VisionLoco::policy_loop()
{
    using clock = std::chrono::steady_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(step_dt_));
    auto next_tick = clock::now() + dt;
    const auto t_start = clock::now();
    std::vector<float> accepted_target = default_joint_pos_;
    std::vector<float> requested_action(12, 0.0f);
    std::vector<float> requested_target = default_joint_pos_;
    bool have_requested_motion = false;
    int motion_violations = 0;
    int stable_lowstate_count = 0;
    int stable_depth_count = 0;
    int shadow_frame_count = 0;
    const int shadow_frames_required = std::max(
        1, static_cast<int>(std::ceil(shadow_gate_seconds_ / step_dt_)));
    bool shadow_gate_ready = strict_loco::initial_shadow_gate_ready(startup_shadow_enabled_);
    bool initialized_from_snapshot = false;
    if (shadow_gate_ready) {
        spdlog::info(
            "[VisionLoco] startup shadow disabled; first valid inference frame publishes immediately");
    }

    while (running_) {
        last_policy_frame_ = frame_;
        const auto loop_start = clock::now();
        const auto snapshot = snapshot_store_.capture(FSMState::lowstate, joint_ids_map_, loop_start);
        auto hard_fault = [this, &snapshot](const std::string& reason) {
            safety_state_.fault(strict_loco::FaultSeverity::HardFault, reason);
            policy_fault_ = true;
            allow_target_publish_ = false;
            spdlog::critical("[VisionLoco] hard fault: {}", reason);
            log_fault_snapshot("HardFault", reason, snapshot);
        };
        auto soft_stop = [this, &snapshot](const std::string& reason) {
            safety_state_.fault(strict_loco::FaultSeverity::SoftStop, reason);
            motion_fault_ = true;
            allow_target_publish_ = false;
            spdlog::error("[VisionLoco] soft stop: {}", reason);
            log_fault_snapshot("SoftStop", reason, snapshot);
        };
        if (!snapshot.valid || snapshot.lowstate_age_ms > lowstate_stale_ms_) {
            hard_fault(snapshot.invalid_reason == "none" ? "lowstate_stale" : snapshot.invalid_reason);
            running_ = false;
            break;
        }
        bool mechanical_fault = false;
        for (int i = 0; i < 12; ++i) {
            if (snapshot.q[i] < joint_lower_[i] || snapshot.q[i] > joint_upper_[i]) {
                spdlog::critical(
                    "[VisionLoco][MECHANICAL_LIMIT] type=position joint={} value={:.5f} "
                    "lower={:.5f} upper={:.5f} dq={:.5f} tau={:.3f} temp_c={:.1f}",
                    i, snapshot.q[i], joint_lower_[i], joint_upper_[i], snapshot.dq[i],
                    snapshot.tau_est[i], snapshot.motor_temperature_c[i]);
                hard_fault("joint_position_limit_" + std::to_string(i));
                mechanical_fault = true; break;
            }
            if (std::fabs(snapshot.dq[i]) > dq_hard_fault_[i]) {
                spdlog::critical(
                    "[VisionLoco][MECHANICAL_LIMIT] type=velocity joint={} value={:.5f} "
                    "abs_limit={:.5f} q={:.5f} tau={:.3f} temp_c={:.1f}",
                    i, snapshot.dq[i], dq_hard_fault_[i], snapshot.q[i],
                    snapshot.tau_est[i], snapshot.motor_temperature_c[i]);
                hard_fault("joint_velocity_limit_" + std::to_string(i));
                mechanical_fault = true; break;
            }
            if (std::fabs(snapshot.tau_est[i]) > tau_hard_fault_[i]) {
                spdlog::critical(
                    "[VisionLoco][MECHANICAL_LIMIT] type=effort joint={} value={:.5f} "
                    "abs_limit={:.5f} q={:.5f} dq={:.5f} temp_c={:.1f}",
                    i, snapshot.tau_est[i], tau_hard_fault_[i], snapshot.q[i],
                    snapshot.dq[i], snapshot.motor_temperature_c[i]);
                hard_fault("joint_effort_hard_" + std::to_string(i));
                mechanical_fault = true; break;
            }
            if (snapshot.motor_temperature_c[i] >= motor_temperature_hard_c_) {
                spdlog::critical(
                    "[VisionLoco][MECHANICAL_LIMIT] type=temperature joint={} value_c={:.1f} "
                    "limit_c={:.1f} q={:.5f} dq={:.5f} tau={:.3f}",
                    i, snapshot.motor_temperature_c[i], motor_temperature_hard_c_,
                    snapshot.q[i], snapshot.dq[i], snapshot.tau_est[i]);
                hard_fault("motor_temperature_hard_" + std::to_string(i));
                mechanical_fault = true; break;
            }
            if (std::fabs(snapshot.tau_est[i]) > tau_soft_stop_[i]) {
                soft_stop("joint_effort_soft_" + std::to_string(i));
                mechanical_fault = true; break;
            }
            if ((std::fabs(snapshot.dq[i]) > dq_warning_[i] ||
                 std::fabs(snapshot.tau_est[i]) > tau_warning_[i] ||
                 snapshot.motor_temperature_c[i] >= motor_temperature_warning_c_) &&
                frame_ % 50 == 0)
                spdlog::warn("[VisionLoco] joint {} warning q={:.3f} dq={:.3f} tau={:.2f} temp={:.0f}",
                             i, snapshot.q[i], snapshot.dq[i], snapshot.tau_est[i],
                             snapshot.motor_temperature_c[i]);
        }
        if (mechanical_fault) { running_ = false; break; }

        auto proprio = build_proprio(snapshot);
        const auto depth_frame = depth_->get();
        const float depth_age_ms = std::chrono::duration<float, std::milli>(
            loop_start - depth_frame.received_at).count();
        last_depth_invalid_fraction_ = depth_frame.invalid_fraction;
        last_depth_front_invalid_fraction_ = depth_frame.front_invalid_fraction;
        last_depth_age_ms_ = depth_age_ms;
        last_depth_frame_number_ = depth_frame.frame_number;
        remember_depth_frame(depth_frame, frame_, depth_age_ms);
        const bool depth_warning = !depth_frame.valid || !std::isfinite(depth_age_ms) ||
                                   depth_age_ms > depth_stale_ms_ ||
                                   depth_frame.invalid_fraction >= 0.50f ||
                                   depth_frame.front_invalid_fraction >= 0.50f;
        if (depth_warning && !depth_warning_active_) {
            spdlog::warn(
                "[VisionLoco][DEPTH_WARNING] warning_only=true policy_frame={} depth_valid={} "
                "age_ms={:.3f} stale_limit_ms={:.3f} camera_frame={} profile={}x{}@{} "
                "normalized_size={} invalid_fraction={:.4f} central_third_invalid={:.4f} "
                "min_depth_m={:.3f} mean_depth_m={:.3f} serial={} firmware={}",
                frame_, depth_frame.valid, depth_age_ms, depth_stale_ms_,
                depth_frame.frame_number, depth_frame.width, depth_frame.height, depth_frame.fps,
                depth_frame.normalized.size(), depth_frame.invalid_fraction,
                depth_frame.front_invalid_fraction, depth_frame.min_depth_m,
                depth_frame.mean_depth_m, depth_frame.serial, depth_frame.firmware);
            depth_capture_requested_ = depth_fault_capture_enabled_;
        } else if (!depth_warning && depth_warning_active_) {
            spdlog::info(
                "[VisionLoco][DEPTH_WARNING] recovered policy_frame={} camera_frame={} age_ms={:.3f}",
                frame_, depth_frame.frame_number, depth_age_ms);
        }
        depth_warning_active_ = depth_warning;
        if (!initialized_from_snapshot) {
            accepted_target.assign(snapshot.q.begin(), snapshot.q.end());
            initialized_from_snapshot = true;
        }
        float max_abs_dq = 0.0f;
        for (float value : snapshot.dq) max_abs_dq = std::max(max_abs_dq, std::fabs(value));
        const float upright_z = std::clamp(-snapshot.projected_gravity[2], -1.0f, 1.0f);
        const float tilt_rad = std::fabs(std::acos(upright_z));
        const bool lowstate_stable = max_abs_dq <= stable_max_joint_velocity_rad_s_ &&
                                     tilt_rad <= stable_max_tilt_rad_;
        stable_lowstate_count = lowstate_stable ? stable_lowstate_count + 1 : 0;
        // The optional startup shadow gate no longer depends on camera content or cadence.
        ++stable_depth_count;
        ++shadow_frame_count;
        const bool shadow_gate_was_ready = shadow_gate_ready;
        shadow_gate_ready = strict_loco::update_shadow_gate(
            shadow_gate_ready, suspended_test_bypass_ || ground_test_permissive_,
            stable_lowstate_count, stable_depth_count,
            shadow_frame_count, stable_lowstate_frames_required_,
            stable_depth_frames_required_, shadow_frames_required);
        if (!shadow_gate_was_ready && shadow_gate_ready) {
            spdlog::info(
                "[VisionLoco] shadow gate passed after {} frames; policy target publication enabled",
                shadow_frame_count);
        }
        const auto& depth = depth_frame.normalized;
        auto goal = goal_;
        std::array<float, 3> theory_cmd{};
        bool uwb_valid = false;
        float uwb_age_s = 0.0f, closing_speed = 0.0f;
        int uwb_error = 255, uwb_enabled = 0, uwb_channel = -1;
        auto exec_cmd = command_for_frame(
            loop_start, goal, theory_cmd, uwb_valid, uwb_age_s, closing_speed,
            uwb_error, uwb_enabled, uwb_channel);
        // loco 阶段速度命令始终由外部（fixed/uwb/零）注入 proprio[6:9]。
        const bool learned_nav_active = command_source_ == "nav";
        runner_->set_cmd_override(
            !learned_nav_active, exec_cmd[0], exec_cmd[1], exec_cmd[2]);
        isaaclab::LocoOutput out;
        try {
            out = runner_->act(depth, proprio, goal);  // joint(12,raw) + cmd(回显) + clearance(0)
            if (!finite_and_bounded(out.joint, max_raw_action_abs_) ||
                !finite_and_bounded(out.cmd, 10.0f) ||
                !finite_and_bounded(out.clearance, 2.0f)) {
                throw std::runtime_error("ONNX 输出包含非有限值或越过安全量级");
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
                last_cmd_vx_ = exec_cmd[0];
                last_cmd_vy_ = exec_cmd[1];
                last_cmd_wz_ = exec_cmd[2];
            }
            consecutive_errors_ = 0;
        } catch (const std::exception& e) {
            ++consecutive_errors_;
            spdlog::error("[VisionLoco] 推理异常 ({}/{}): {}",
                          consecutive_errors_, max_consecutive_errors_, e.what());
            if (consecutive_errors_ >= max_consecutive_errors_) {
                spdlog::critical("[VisionLoco] 连续推理异常，触发 Passive");
                hard_fault("inference_exception_limit");
                running_ = false;
                break;
            }
            std::this_thread::sleep_until(next_tick);
            next_tick += dt;
            continue;
        }

        if (out.inference_ms > inference_deadline_ms_) {
            ++deadline_misses_;
            soft_stop("inference_deadline_miss");
            running_ = false;
            break;
        }

        const auto layers = strict_loco::execute_action_chain(
            out.joint, accepted_target, act_offset_, act_scale_, act_clip_lo_, act_clip_hi_,
            target_slew_rate_rad_s_ * step_dt_, joint_lower_, joint_upper_);
        const std::vector<float> clipped_action(
            layers.clipped_raw_action.begin(), layers.clipped_raw_action.end());
        const std::vector<float> tgt(
            layers.requested_joint_target.begin(), layers.requested_joint_target.end());
        const std::vector<float> applied_target(
            layers.applied_joint_target.begin(), layers.applied_joint_target.end());
        const std::vector<float> applied_action(
            layers.executed_raw_action.begin(), layers.executed_raw_action.end());
        float action_step_max = 0.0f;
        float requested_target_step_max = 0.0f;
        float target_step_max = 0.0f;
        float tracking_error_max = 0.0f;
        for (int i = 0; i < 12; ++i) {
            if (have_requested_motion) {
                action_step_max = std::max(
                    action_step_max, std::fabs(clipped_action[i] - requested_action[i]));
                requested_target_step_max = std::max(
                    requested_target_step_max,
                    std::fabs(tgt[i] - requested_target[i]));
            }
            tracking_error_max = std::max(
                tracking_error_max, std::fabs(snapshot.q[i] - accepted_target[i]));
            target_step_max = std::max(
                target_step_max, std::fabs(applied_target[i] - accepted_target[i]));
        }
        const bool audit_post_slew = strict_loco::audit_post_slew_motion(
            suspended_test_bypass_, ground_test_permissive_, walking_test_permissive_);
        const bool tracking_guard = strict_loco::tracking_guard_enabled(
            ground_test_permissive_, walking_test_permissive_);
        const bool motion_rejected = shadow_gate_ready && strict_loco::reject_motion_request(
            audit_post_slew, tracking_guard,
            have_requested_motion,
            requested_target_step_max, target_step_max, tracking_error_max,
            max_target_step_rad_, max_tracking_error_rad_);
        last_tracking_error_max_ = tracking_error_max;
        last_requested_target_step_max_ = requested_target_step_max;
        last_target_step_max_ = target_step_max;
        // Compare safety against adjacent policy requests, even when the current
        // request is held back. A stable request after one spike must not be
        // repeatedly compared with an older accepted target.
        if (shadow_gate_ready) {
            requested_action = clipped_action;
            requested_target = tgt;
            have_requested_motion = true;
        }
        if (motion_rejected) {
            ++motion_violations;
            spdlog::warn(
                "[VisionLoco] rejected motion frame: requested_step={:.3f}rad "
                "applied_step={:.3f}rad tracking_error={:.3f}rad permissive={} ({}/{})",
                requested_target_step_max, target_step_max, tracking_error_max,
                audit_post_slew && !tracking_guard, motion_violations,
                max_consecutive_motion_violations_);
            if (motion_violations >= max_consecutive_motion_violations_) {
                spdlog::critical("[VisionLoco] repeated unsafe motion; requesting FixStand");
                soft_stop("repeated_unsafe_motion");
                running_ = false;
            }
        } else if (shadow_gate_ready) {
            motion_violations = 0;
            accepted_target = applied_target;
            std::lock_guard<std::mutex> lk(tgt_mtx_);
            joint_target_    = applied_target;
            last_action_raw_ = applied_action;
            have_target_     = true;
            policy_target_at_ = clock::now();
            policy_target_active_ = true;
            allow_target_publish_ = true;
        } else {
            accepted_target.assign(snapshot.q.begin(), snapshot.q.end());
            allow_target_publish_ = false;
        }

        std::vector<float> joint_error(12);
        std::vector<float> pd_torque(12);
        float tau_abs_max = 0.0f;
        float pd_tau_abs_max = 0.0f;
        float mechanical_power_abs_sum = 0.0f;
        for (int i = 0; i < 12; ++i) {
            joint_error[i] = accepted_target[i] - snapshot.q[i];
            pd_torque[i] = stiffness_[i] * joint_error[i] - damping_[i] * snapshot.dq[i];
            tau_abs_max = std::max(tau_abs_max, std::fabs(snapshot.tau_est[i]));
            pd_tau_abs_max = std::max(pd_tau_abs_max, std::fabs(pd_torque[i]));
            mechanical_power_abs_sum += std::fabs(snapshot.tau_est[i] * snapshot.dq[i]);
        }
        last_pd_tau_abs_max_ = pd_tau_abs_max;

        const auto work_end = clock::now();
        const float loop_ms =
            std::chrono::duration<float, std::milli>(work_end - loop_start).count();
        if (work_end > next_tick) ++deadline_misses_;

        if ((frame_ % 50) == 0) {
            spdlog::info("[VisionLoco] source={} theory=[{:.3f},{:.3f},{:.3f}] "
                         "exec=[{:.3f},{:.3f},{:.3f}] uwb_ok={} goal=[{:.2f},{:.2f},{:.2f},{:.2f}] "
                         "clr=[{:.2f},{:.2f},{:.2f}] step(a/r/t/e)=[{:.2f},{:.2f},{:.2f},{:.2f}] "
                         "reject={} effort(meas/pd/pwr)=[{:.1f},{:.1f},{:.1f}] "
                         "infer={:.2f}ms loop={:.2f}ms miss={}",
                         command_source_, theory_cmd[0], theory_cmd[1], theory_cmd[2],
                         out.cmd[0], out.cmd[1], out.cmd[2],
                         uwb_valid, goal[0], goal[1], goal[2], goal[3],
                         out.clearance[0], out.clearance[1], out.clearance[2],
                         action_step_max, requested_target_step_max, target_step_max,
                         tracking_error_max,
                         motion_rejected, tau_abs_max, pd_tau_abs_max,
                         mechanical_power_abs_sum,
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
            const float expected_closing =
                exec_cmd[0] * std::cos(goal[0]) + exec_cmd[1] * std::sin(goal[0]);
            const float closing_error = uwb_valid ? closing_speed - expected_closing : 0.0f;
            const auto& av = snapshot.angular_velocity;
            const auto& pg = snapshot.projected_gravity;
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
            const auto& q = snapshot.q;
            const auto& dq = snapshot.dq;
            const auto& tau = snapshot.tau_est;
            long t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now() - t_start).count();
            diag_ << frame_ << ',' << t_ms << ',' << loop_ms << ',' << out.inference_ms
                  << ',' << deadline_misses_ << ',' << consecutive_errors_;
            write_values(diag_, out.cmd);
            write_values(diag_, out.cmd_raw);
            write_values(diag_, theory_cmd);
            const int source_code =
                command_source_ == "fixed" ? 0 : (command_source_ == "uwb" ? 1 : 2);
            diag_ << ',' << source_code;
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
                  << ',' << depth_frame.invalid_fraction
                  << ',' << depth_frame.front_invalid_fraction
                  << ',' << action_step_max << ',' << requested_target_step_max
                  << ',' << target_step_max
                  << ',' << tracking_error_max << ',' << (motion_rejected ? 1 : 0)
                  << ',' << motion_violations
                  << ',' << layers.modified_count << ',' << (layers.modified_count / 12.0f)
                  << ',' << (shadow_gate_ready ? 1 : 0)
                  << ',' << (suspended_test_bypass_ ? 1 : 0)
                  << ',' << (ground_test_permissive_ ? 1 : 0)
                  << ',' << (walking_test_permissive_ ? 1 : 0)
                  << ',' << stable_lowstate_count << ',' << stable_depth_count
                  << ',' << snapshot.sequence << ',' << snapshot.lowstate_tick
                  << ',' << snapshot.lowstate_age_ms << ',' << depth_age_ms
                  << ',' << depth_frame.frame_number << ',' << depth_frame.sensor_timestamp_ms
                  << ',' << av[0] << ',' << av[1] << ',' << av[2]
                  << ',' << pg[0] << ',' << pg[1] << ',' << pg[2]
                  << ',' << tau_abs_max << ',' << pd_tau_abs_max
                  << ',' << mechanical_power_abs_sum;
            for (int i = 0; i < q.size(); ++i) diag_ << ',' << q[i];
            for (int i = 0; i < dq.size(); ++i) diag_ << ',' << dq[i];
            for (int i = 0; i < tau.size(); ++i) diag_ << ',' << tau[i];
            write_values(diag_, layers.model_raw_action);
            write_values(diag_, layers.clipped_raw_action);
            write_values(diag_, layers.requested_joint_target);
            write_values(diag_, layers.slew_limited_joint_target);
            write_values(diag_, layers.physical_limit_joint_target);
            write_values(diag_, layers.applied_joint_target);
            write_values(diag_, layers.executed_raw_action);
            write_values(diag_, joint_error);
            write_values(diag_, pd_torque);
            diag_ << '\n';
            if ((frame_ % log_flush_every_) == 0) diag_.flush();
        }

        ++frame_;
        last_policy_frame_ = frame_;
        if (!running_) break;
        std::this_thread::sleep_until(next_tick);
        next_tick += dt;
        if (clock::now() > next_tick + dt) next_tick = clock::now() + dt;
    }
}

void State_VisionLoco::run()
{
    std::lock_guard<std::mutex> lk(tgt_mtx_);
    const auto now = std::chrono::steady_clock::now();
    strict_loco::SensorSnapshot current_snapshot;
    try {
        current_snapshot = snapshot_store_.capture(FSMState::lowstate, joint_ids_map_, now);
    } catch (const std::exception& e) {
        const std::string reason = std::string("run_lowstate_exception: ") + e.what();
        safety_state_.fault(strict_loco::FaultSeverity::HardFault, reason);
        policy_fault_ = true;
        allow_target_publish_ = false;
        spdlog::critical("[VisionLoco][LOWSTATE_FAULT] reason={}", reason);
        log_fault_snapshot("HardFault", reason, snapshot_store_.latest());
        for (auto& motor : lowcmd->msg_.motor_cmd()) {
            motor.kp() = 0.0f;
            motor.kd() = 0.0f;
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
            motor.q() = 0.0f;
        }
        return;
    }
    if (!current_snapshot.valid || current_snapshot.lowstate_age_ms > lowstate_stale_ms_) {
        const std::string reason = current_snapshot.invalid_reason == "none"
            ? "run_lowstate_stale" : current_snapshot.invalid_reason;
        safety_state_.fault(strict_loco::FaultSeverity::HardFault, reason);
        policy_fault_ = true;
        allow_target_publish_ = false;
        spdlog::critical(
            "[VisionLoco][LOWSTATE_FAULT] reason={} age_ms={:.3f} limit_ms={:.3f}",
            reason, current_snapshot.lowstate_age_ms, lowstate_stale_ms_);
        log_fault_snapshot("HardFault", reason, current_snapshot);
        for (auto& motor : lowcmd->msg_.motor_cmd()) {
            motor.kp() = 0.0f;
            motor.kd() = 0.0f;
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
            motor.q() = 0.0f;
        }
        return;
    }
    const float target_age_ms = have_target_
        ? std::chrono::duration<float, std::milli>(now - policy_target_at_).count()
        : std::numeric_limits<float>::infinity();
    const auto target_decision = strict_loco::decide_target_command(
        policy_target_active_.load(), allow_target_publish_.load(),
        safety_state_.freeze_policy_target.load(), have_target_, target_age_ms,
        policy_target_stale_ms_);
    if (target_decision.mode != strict_loco::TargetCommandMode::PolicyTarget) {
        if (target_decision.stale_fault) {
            safety_state_.fault(strict_loco::FaultSeverity::SoftStop, "policy_target_stale");
            if (!target_stale_logged_.exchange(true)) {
                spdlog::critical(
                    "[VisionLoco][POLICY_TARGET_STALE] age_ms={:.3f} limit_ms={:.3f} "
                    "have_target={} policy_target_active={} allow_publish={} freeze_target={}",
                    target_age_ms, policy_target_stale_ms_, have_target_,
                    policy_target_active_.load(), allow_target_publish_.load(),
                    safety_state_.freeze_policy_target.load());
                log_fault_snapshot("SoftStop", "policy_target_stale", current_snapshot);
            }
        }
        for (int i = 0; i < (int)joint_ids_map_.size(); ++i) {
            const float target = strict_loco::select_joint_target(
                target_decision.mode, joint_target_[i], current_snapshot.q[i]);
            lowcmd->msg_.motor_cmd()[joint_ids_map_[i]].q() = target;
        }
        return;
    }
    for (int i = 0; i < (int)joint_ids_map_.size(); ++i) {
        lowcmd->msg_.motor_cmd()[joint_ids_map_[i]].q() = joint_target_[i];
    }
}

void State_VisionLoco::exit()
{
    const auto severity = safety_state_.severity.load();
    const std::string reason = safety_state_.fault_reason();
    if (safety_state_.takeover_requested.load()) {
        const char* target = severity == strict_loco::FaultSeverity::HardFault
            ? "Passive(damping)" : "FixStand";
        spdlog::critical(
            "[VisionLoco] EXIT due to safety takeover: target={} reason={} last_policy_frame={} diag={}",
            target, reason, last_policy_frame_.load(), diag_path_);
    } else {
        spdlog::info(
            "[VisionLoco] EXIT without safety fault: operator/configured transition; "
            "last_policy_frame={} diag={}", last_policy_frame_.load(), diag_path_);
    }
    safety_state_.freeze_policy_target = true;
    running_ = false;
    if (policy_thread_.joinable()) policy_thread_.join();
    if (depth_capture_requested_ || depth_capture_dump_on_exit_) dump_depth_capture();
    allow_target_publish_ = false;
    policy_target_active_ = false;
    runner_->reset();
    runner_->set_cmd_override(true, 0.0f, 0.0f, 0.0f);
    last_exec_cmd_ = {0.0f, 0.0f, 0.0f};
    last_cmd_vx_ = 0.0f;
    last_cmd_vy_ = 0.0f;
    last_cmd_wz_ = 0.0f;
    {
        std::lock_guard<std::mutex> lk(tgt_mtx_);
        have_target_ = false;
        joint_target_ = default_joint_pos_;
        last_action_raw_.assign(12, 0.0f);
    }
    if (diag_) { diag_.flush(); diag_.close(); spdlog::info("[VisionLoco] 诊断日志已保存: {}", diag_path_); }
}

void State_VisionLoco::remember_depth_frame(
    const vision_nav::DepthFrame& frame, long policy_frame, float age_ms)
{
    if (!depth_fault_capture_enabled_ || frame.frame_number == 0 ||
        frame.frame_number == last_recorded_depth_frame_)
        return;
    last_recorded_depth_frame_ = frame.frame_number;
    depth_capture_ring_.push_back({frame, policy_frame, age_ms});
    while (depth_capture_ring_.size() > depth_fault_capture_frames_)
        depth_capture_ring_.pop_front();
}

void State_VisionLoco::dump_depth_capture()
{
    if (depth_capture_ring_.empty()) {
        spdlog::error("[VisionLoco] depth capture requested but ring buffer is empty");
        return;
    }
    try {
        const std::filesystem::path diag_path(diag_path_);
        const std::filesystem::path capture_dir = diag_path.parent_path() /
            (diag_path.stem().string() +
             (depth_capture_requested_ ? "_depth_warning" : "_depth_capture"));
        std::filesystem::create_directories(capture_dir);
        std::ofstream manifest(capture_dir / "manifest.csv", std::ios::out | std::ios::trunc);
        manifest << "index,policy_frame,camera_frame,sensor_timestamp_ms,age_ms,"
                    "width,height,fps,valid,invalid_fraction,front_invalid_fraction,"
                    "min_depth_m,mean_depth_m,raw_depth_file,valid_mask_file,preview_file\n";
        size_t index = 0;
        for (const auto& recorded : depth_capture_ring_) {
            const auto& frame = recorded.frame;
            char raw_name[64], mask_name[64], preview_name[64];
            const auto camera_frame = static_cast<unsigned long long>(frame.frame_number);
            std::snprintf(raw_name, sizeof(raw_name), "depth_%03zu_cam_%llu.pgm", index, camera_frame);
            std::snprintf(mask_name, sizeof(mask_name), "valid_%03zu_cam_%llu.pgm", index, camera_frame);
            std::snprintf(preview_name, sizeof(preview_name), "preview_%03zu_cam_%llu.ppm", index, camera_frame);
            std::ofstream raw_image(capture_dir / raw_name, std::ios::binary | std::ios::trunc);
            std::ofstream mask_image(capture_dir / mask_name, std::ios::binary | std::ios::trunc);
            std::ofstream preview(capture_dir / preview_name, std::ios::binary | std::ios::trunc);
            raw_image << "P5\n" << vision_nav::DEPTH_W << ' ' << vision_nav::DEPTH_H
                      << "\n5000\n";
            mask_image << "P5\n" << vision_nav::DEPTH_W << ' ' << vision_nav::DEPTH_H
                       << "\n255\n";
            preview << "P6\n" << vision_nav::DEPTH_W << ' ' << vision_nav::DEPTH_H
                    << "\n255\n";
            const size_t pixel_count =
                static_cast<size_t>(vision_nav::DEPTH_W) * vision_nav::DEPTH_H;
            for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
                const float normalized = pixel < frame.normalized.size()
                    ? frame.normalized[pixel] : 0.0f;
                const bool valid = std::isfinite(normalized) && normalized > 0.0f;
                const float meters = valid
                    ? std::clamp(normalized * vision_nav::MAX_DEPTH_M, 0.0f,
                                 vision_nav::MAX_DEPTH_M)
                    : 0.0f;
                const uint16_t millimeters =
                    static_cast<uint16_t>(std::lround(meters * 1000.0f));
                const unsigned char bytes[2] = {
                    static_cast<unsigned char>((millimeters >> 8) & 0xff),
                    static_cast<unsigned char>(millimeters & 0xff)};
                raw_image.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
                const unsigned char mask = valid ? 255 : 0;
                mask_image.write(reinterpret_cast<const char*>(&mask), sizeof(mask));
                unsigned char rgb[3];
                if (!valid) {
                    rgb[0] = 255; rgb[1] = 0; rgb[2] = 255;
                } else {
                    const float t = std::clamp(normalized, 0.0f, 1.0f);
                    rgb[0] = static_cast<unsigned char>(255.0f * (1.0f - t));
                    rgb[1] = static_cast<unsigned char>(255.0f * (1.0f - std::fabs(2.0f * t - 1.0f)));
                    rgb[2] = static_cast<unsigned char>(255.0f * t);
                }
                preview.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
            }
            manifest << index << ',' << recorded.policy_frame << ',' << frame.frame_number
                     << ',' << frame.sensor_timestamp_ms << ',' << recorded.age_ms
                     << ',' << frame.width << ',' << frame.height << ',' << frame.fps
                     << ',' << (frame.valid ? 1 : 0) << ',' << frame.invalid_fraction
                     << ',' << frame.front_invalid_fraction << ',' << frame.min_depth_m
                     << ',' << frame.mean_depth_m << ',' << raw_name << ',' << mask_name
                     << ',' << preview_name << '\n';
            ++index;
        }
        manifest.flush();
        spdlog::critical(
            "[VisionLoco] depth capture saved: dir={} unique_frames={} duration_est={:.2f}s "
            "formats=raw_mm_pgm+valid_mask_pgm+color_preview_ppm",
            capture_dir.string(), depth_capture_ring_.size(),
            depth_capture_ring_.size() / 30.0f);
    } catch (const std::exception& error) {
        spdlog::error("[VisionLoco] failed to save depth fault capture: {}", error.what());
    }
}
