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
        vision_nav::FilterConfig filters;
        if (cfg["depth"]["filters"]) {
            const auto f = cfg["depth"]["filters"];
            filters.mode = yaml_get<std::string>(f, "mode", filters.mode);
            filters.spatial_magnitude = std::clamp(
                yaml_get<int>(f, "spatial_magnitude", filters.spatial_magnitude), 1, 5);
            filters.spatial_smooth_alpha = std::clamp(
                yaml_get<float>(f, "spatial_smooth_alpha", filters.spatial_smooth_alpha),
                0.25f, 1.0f);
            filters.spatial_smooth_delta = std::clamp(
                yaml_get<int>(f, "spatial_smooth_delta", filters.spatial_smooth_delta), 1, 50);
            filters.temporal_alpha = std::clamp(
                yaml_get<float>(f, "temporal_alpha", filters.temporal_alpha), 0.0f, 1.0f);
            filters.temporal_delta = std::clamp(
                yaml_get<float>(f, "temporal_delta", filters.temporal_delta), 1.0f, 100.0f);
        }
        if (filters.mode != "none" && filters.mode != "light_spatial" &&
            filters.mode != "light_spatial_weak_temporal") {
            throw std::runtime_error("unknown RealSense filter mode: " + filters.mode);
        }
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

    // ---------- 安全转移：翻倒 → Passive ----------
    this->registered_checks.emplace_back(std::make_pair(
        [this]() -> bool {
            return bad_orientation(robot_->data.projected_gravity_b, 1.0f);
        },
        FSMStringMap.right.at("Passive")));
    this->registered_checks.emplace_back(std::make_pair(
        [this]() -> bool { return policy_fault_.load(); },
        FSMStringMap.right.at("Passive")));
    this->registered_checks.emplace_back(std::make_pair(
        [this]() -> bool { return motion_fault_.load(); },
        FSMStringMap.right.at("FixStand")));

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
    // 设 PD 增益（全 12 关节）
    for (int i = 0; i < (int)stiffness_.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = stiffness_[i];
        lowcmd->msg_.motor_cmd()[i].kd() = damping_[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    robot_->update();

    // 初始目标 = 默认站姿；last_action=0；runner 清状态。
    {
        std::lock_guard<std::mutex> lk(tgt_mtx_);
        joint_target_    = default_joint_pos_;
        last_action_raw_ = std::vector<float>(12, 0.0f);
        have_target_     = true;
    }
    runner_->reset();
    frame_ = 0;
    deadline_misses_ = 0;
    consecutive_errors_ = 0;
    policy_fault_ = false;
    motion_fault_ = false;
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
                     "action_step_max,requested_target_step_max,target_step_max,tracking_error_max,"
                     "motion_rejected,motion_violations,"
                     "avx,avy,avz,pgx,pgy,pgz";
            for (int i = 0; i < 12; ++i) diag_ << ",q" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",dq" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",action" << i;
            for (int i = 0; i < 12; ++i) diag_ << ",target" << i;
            diag_ << '\n';
            diag_ << std::fixed << std::setprecision(6);
            spdlog::info("[VisionLoco] 诊断日志: {}", diag_path_);
        } else {
            spdlog::warn("[VisionLoco] 诊断日志打开失败: {}", diag_path_);
        }
    }

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
    std::vector<float> accepted_target = default_joint_pos_;
    std::vector<float> requested_action(12, 0.0f);
    std::vector<float> requested_target = default_joint_pos_;
    bool have_requested_motion = false;
    int motion_violations = 0;

    while (running_) {
        const auto loop_start = clock::now();
        robot_->update();                          // 刷新 IMU/关节（策略序）
        auto proprio = build_proprio();            // 45
        auto depth   = depth_->get();              // 57600（归一化）
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
            }
            consecutive_errors_ = 0;
        } catch (const std::exception& e) {
            ++consecutive_errors_;
            spdlog::error("[VisionLoco] 推理异常 ({}/{}): {}",
                          consecutive_errors_, max_consecutive_errors_, e.what());
            if (consecutive_errors_ >= max_consecutive_errors_) {
                spdlog::critical("[VisionLoco] 连续推理异常，触发 Passive");
                policy_fault_ = true;
                running_ = false;
                break;
            }
            std::this_thread::sleep_until(next_tick);
            next_tick += dt;
            continue;
        }

        // 处理动作：target = clamp(offset + scale*raw, clip)
        std::vector<float> tgt(12);
        for (int i = 0; i < 12; ++i) {
            float v = act_offset_[i] + act_scale_ * out.joint[i];
            tgt[i] = std::clamp(v, act_clip_lo_, act_clip_hi_);
        }
        float action_step_max = 0.0f;
        float requested_target_step_max = 0.0f;
        float target_step_max = 0.0f;
        float tracking_error_max = 0.0f;
        const auto& policy_q = robot_->data.joint_pos;
        for (int i = 0; i < 12; ++i) {
            if (have_requested_motion) {
                action_step_max = std::max(
                    action_step_max, std::fabs(out.joint[i] - requested_action[i]));
                requested_target_step_max = std::max(
                    requested_target_step_max,
                    std::fabs(tgt[i] - requested_target[i]));
            }
            tracking_error_max = std::max(
                tracking_error_max, std::fabs(policy_q[i] - accepted_target[i]));
        }
        const bool motion_rejected =
                                     (have_requested_motion &&
                                      requested_target_step_max > max_target_step_rad_) ||
                                     tracking_error_max > max_tracking_error_rad_;
        // Compare safety against adjacent policy requests, even when the current
        // request is held back. A stable request after one spike must not be
        // repeatedly compared with an older accepted target.
        requested_action = out.joint;
        requested_target = tgt;
        have_requested_motion = true;
        if (motion_rejected) {
            ++motion_violations;
            spdlog::warn(
                "[VisionLoco] rejected motion frame: requested_step={:.3f}rad "
                "tracking_error={:.3f}rad ({}/{})",
                requested_target_step_max, tracking_error_max, motion_violations,
                max_consecutive_motion_violations_);
            if (motion_violations >= max_consecutive_motion_violations_) {
                spdlog::critical("[VisionLoco] repeated unsafe motion; requesting FixStand");
                motion_fault_ = true;
                running_ = false;
            }
        } else {
            motion_violations = 0;
            const float max_applied_step = target_slew_rate_rad_s_ * step_dt_;
            std::vector<float> applied_target(12);
            std::vector<float> applied_action(12);
            for (int i = 0; i < 12; ++i) {
                applied_target[i] = accepted_target[i] + std::clamp(
                    tgt[i] - accepted_target[i], -max_applied_step, max_applied_step);
                applied_action[i] = std::fabs(act_scale_) > 1e-6f
                    ? (applied_target[i] - act_offset_[i]) / act_scale_
                    : 0.0f;
                target_step_max = std::max(
                    target_step_max, std::fabs(applied_target[i] - accepted_target[i]));
            }
            accepted_target = applied_target;
            std::lock_guard<std::mutex> lk(tgt_mtx_);
            joint_target_    = applied_target;
            last_action_raw_ = applied_action;
            have_target_     = true;
        }

        const auto work_end = clock::now();
        const float loop_ms =
            std::chrono::duration<float, std::milli>(work_end - loop_start).count();
        if (work_end > next_tick) ++deadline_misses_;

        if ((frame_ % 50) == 0) {
            spdlog::info("[VisionLoco] source={} theory=[{:.3f},{:.3f},{:.3f}] "
                         "exec=[{:.3f},{:.3f},{:.3f}] uwb_ok={} goal=[{:.2f},{:.2f},{:.2f},{:.2f}] "
                         "clr=[{:.2f},{:.2f},{:.2f}] step(a/r/t/e)=[{:.2f},{:.2f},{:.2f},{:.2f}] "
                         "reject={} infer={:.2f}ms loop={:.2f}ms miss={}",
                         command_source_, theory_cmd[0], theory_cmd[1], theory_cmd[2],
                         out.cmd[0], out.cmd[1], out.cmd[2],
                         uwb_valid, goal[0], goal[1], goal[2], goal[3],
                         out.clearance[0], out.clearance[1], out.clearance[2],
                         action_step_max, requested_target_step_max, target_step_max,
                         tracking_error_max,
                         motion_rejected, out.inference_ms, loop_ms, deadline_misses_);
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
                  << ',' << action_step_max << ',' << requested_target_step_max
                  << ',' << target_step_max
                  << ',' << tracking_error_max << ',' << (motion_rejected ? 1 : 0)
                  << ',' << motion_violations
                  << ',' << av[0] << ',' << av[1] << ',' << av[2]
                  << ',' << pg[0] << ',' << pg[1] << ',' << pg[2];
            for (int i = 0; i < q.size(); ++i) diag_ << ',' << q[i];
            for (int i = 0; i < dq.size(); ++i) diag_ << ',' << dq[i];
            write_values(diag_, out.joint);
            write_values(diag_, tgt);
            diag_ << '\n';
            if ((frame_ % log_flush_every_) == 0) diag_.flush();
        }

        ++frame_;
        if (!running_) break;
        std::this_thread::sleep_until(next_tick);
        next_tick += dt;
        if (clock::now() > next_tick + dt) next_tick = clock::now() + dt;
    }
}

void State_VisionLoco::run()
{
    std::lock_guard<std::mutex> lk(tgt_mtx_);
    if (!have_target_) return;
    for (int i = 0; i < (int)joint_ids_map_.size(); ++i) {
        lowcmd->msg_.motor_cmd()[joint_ids_map_[i]].q() = joint_target_[i];
    }
}

void State_VisionLoco::exit()
{
    running_ = false;
    if (policy_thread_.joinable()) policy_thread_.join();
    if (diag_) { diag_.flush(); diag_.close(); spdlog::info("[VisionLoco] 诊断日志已保存: {}", diag_path_); }
}
