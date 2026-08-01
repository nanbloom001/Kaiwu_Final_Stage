#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "State_VisionLoco.h"
#include "StrictSafety.h"
#include "DepthSource.h"
#include "isaaclab/algorithms/loco_runner.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <sys/wait.h>
#include <thread>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace
{
std::atomic<bool> keep_running{true};

void stop_handler(int) { keep_running = false; }

std::filesystem::path package_root()
{
    auto path = param::proj_dir;
    for (int i = 0; i < 8; ++i) {
        if (std::filesystem::exists(path / "artifact_contract.json")) return path;
        path = path.parent_path();
    }
    throw std::runtime_error("cannot locate strict package root from executable path");
}

std::filesystem::path setup_runtime_logging()
{
    const auto cfg = param::config["FSM"]["VisionLoco"];
    const auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    const auto log_dir = policy_dir / "logs";
    std::filesystem::create_directories(log_dir);
    const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto log_path = log_dir /
        ("strict_controller_" + std::to_string(timestamp_ms) + ".log");

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        log_path.string(), true);
    auto logger = std::make_shared<spdlog::logger>(
        "strict_controller", spdlog::sinks_init_list{console_sink, file_sink});
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [thread %t] [%^%l%$] %v");
    spdlog::set_default_logger(logger);
    spdlog::info("[strict] persistent controller log={}", log_path.string());
    return log_path;
}

int run_preflight(bool require_camera = false)
{
    const auto tool = package_root() / "tools" / "preflight.py";
    const std::string command = "python3 \"" + tool.string() +
                                "\" --json" + (require_camera ? " --require-camera" : "");
    const int status = std::system(command.c_str());
    if (status == -1) return 1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

std::unique_ptr<vision_nav::DepthSource> create_strict_depth(const YAML::Node& cfg)
{
    const std::string source = cfg["depth"]["source"].as<std::string>();
    if (source == "constant")
        return std::make_unique<vision_nav::ConstantDepth>(
            cfg["depth"]["constant_value"].as<float>());
    if (source != "realsense") throw std::runtime_error("unknown strict depth source");
#ifdef USE_REALSENSE
    vision_nav::RealSenseConfig camera;
    camera.filters.mode = cfg["depth"]["filters"]["mode"].as<std::string>();
    const auto options = cfg["depth"]["camera_options"];
    camera.high_density_preset =
        options["visual_preset"].as<std::string>() == "high_density";
    camera.emitter_enabled = options["emitter_enabled"].as<bool>();
    camera.max_laser_power = options["laser_power"].as<std::string>() == "max";
    camera.auto_exposure = options["auto_exposure"].as<bool>();
    return std::make_unique<vision_nav::RealSenseDepth>(424, 240, 30, camera);
#else
    throw std::runtime_error("strict sensor/shadow mode requires a RealSense-enabled build");
#endif
}

std::vector<float> build_shadow_proprio(
    const strict_loco::SensorSnapshot& snapshot,
    const std::vector<float>& defaults,
    const std::vector<float>& last_action)
{
    std::vector<float> p(45, 0.0f);
    for (int i = 0; i < 3; ++i) p[i] = snapshot.angular_velocity[i] * 0.25f;
    for (int i = 0; i < 3; ++i) p[3 + i] = snapshot.projected_gravity[i];
    for (int i = 0; i < 12; ++i) p[9 + i] = snapshot.q[i] - defaults[i];
    for (int i = 0; i < 12; ++i) p[21 + i] = snapshot.dq[i] * 0.05f;
    for (int i = 0; i < 12; ++i) p[33 + i] = last_action[i];
    return p;
}

int run_no_command_mode(const std::string& mode)
{
    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
    auto lowstate = std::make_shared<LowState_t>();
    lowstate->set_timeout_ms(50);
    spdlog::info("[strict] {} waiting for LowState; LowCmd will not be created", mode);
    lowstate->wait_for_connection();
    spdlog::info("[strict] {} LowState connection established", mode);

    const auto cfg = param::config["FSM"]["VisionLoco"];
    spdlog::info("[strict] {} loaded VisionLoco config", mode);
    const auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    spdlog::info("[strict] {} policy_dir={}", mode, policy_dir.string());
    const auto deploy = YAML::LoadFile((policy_dir / "params" / "deploy.yaml").string());
    spdlog::info("[strict] {} loaded deploy.yaml", mode);
    const auto joint_map = deploy["joint_ids_map"].as<std::vector<int>>();
    const auto defaults = deploy["default_joint_pos"].as<std::vector<float>>();
    auto depth = create_strict_depth(cfg);
    spdlog::info("[strict] {} depth source initialized", mode);
    spdlog::warn(
        "[strict] {} depth frame freshness and pixel validity are warning-only; "
        "the RealSense pipeline opening successfully is the only camera startup gate",
        mode);
    strict_loco::SnapshotStore snapshots;

    std::filesystem::create_directories(package_root() / "logs");
    const auto stamp = std::to_string(std::time(nullptr));
    const auto log_path = package_root() / "logs" / (mode + "_" + stamp + ".csv");
    std::ofstream log(log_path);
    log << "frame,t_ms,sensor_sequence,lowstate_tick,lowstate_age_ms,depth_frame_number,"
           "depth_sensor_timestamp_ms,depth_age_ms,depth_invalid_fraction,front_invalid_fraction,"
           "battery_voltage_v,battery_soc";
    for (int i = 0; i < 12; ++i) log << ",q" << i;
    for (int i = 0; i < 12; ++i) log << ",dq" << i;
    for (int i = 0; i < 12; ++i) log << ",tau" << i;
    for (int i = 0; i < 12; ++i) log << ",temp" << i;
    if (mode == "shadow") {
        log << ",inference_ms,safety_modified_count";
        for (int i = 0; i < 12; ++i) log << ",model_raw_action" << i;
        for (int i = 0; i < 12; ++i) log << ",clipped_raw_action" << i;
        for (int i = 0; i < 12; ++i) log << ",requested_target" << i;
        for (int i = 0; i < 12; ++i) log << ",applied_target" << i;
        for (int i = 0; i < 12; ++i) log << ",executed_raw_action" << i;
    }
    log << '\n' << std::fixed << std::setprecision(6);

    std::unique_ptr<isaaclab::LocoRunner> runner;
    std::vector<float> last_action(12, 0.0f), previous_target = defaults;
    std::vector<float> lower, upper;
    if (mode == "shadow") {
        runner = std::make_unique<isaaclab::LocoRunner>(
            (policy_dir / "exported" / "policy.onnx").string());
        runner->set_cmd_override(true, 0.0f, 0.0f, 0.0f);
        lower = cfg["strict_safety"]["joint_lower"].as<std::vector<float>>();
        upper = cfg["strict_safety"]["joint_upper"].as<std::vector<float>>();
    }

    const auto start = strict_loco::Clock::now();
    uint64_t frame = 0;
    bool depth_warning_active = false;
    auto next = start;
    while (keep_running) {
        const auto now = strict_loco::Clock::now();
        const auto snapshot = snapshots.capture(lowstate, joint_map, now);
        const auto depth_frame = depth->get();
        const float depth_age = std::chrono::duration<float, std::milli>(
            now - depth_frame.received_at).count();
        if (!snapshot.valid || snapshot.lowstate_age_ms > 50.0f)
            throw std::runtime_error("strict no-command fault: " + snapshot.invalid_reason);
        const bool depth_warning = !depth_frame.valid || !std::isfinite(depth_age) ||
                                   depth_age > 150.0f ||
                                   depth_frame.invalid_fraction >= 0.50f ||
                                   depth_frame.front_invalid_fraction >= 0.50f;
        if (depth_warning && !depth_warning_active) {
            spdlog::warn(
                "[strict][DEPTH_WARNING] warning_only=true valid={} age_ms={:.3f} "
                "frame={} invalid_fraction={:.4f} front_invalid_fraction={:.4f}",
                depth_frame.valid, depth_age, depth_frame.frame_number,
                depth_frame.invalid_fraction, depth_frame.front_invalid_fraction);
        } else if (!depth_warning && depth_warning_active) {
            spdlog::info("[strict][DEPTH_WARNING] recovered frame={} age_ms={:.3f}",
                         depth_frame.frame_number, depth_age);
        }
        depth_warning_active = depth_warning;

        log << frame << ','
            << std::chrono::duration<float, std::milli>(now - start).count() << ','
            << snapshot.sequence << ',' << snapshot.lowstate_tick << ',' << snapshot.lowstate_age_ms
            << ',' << depth_frame.frame_number << ',' << depth_frame.sensor_timestamp_ms << ','
            << depth_age << ',' << depth_frame.invalid_fraction << ','
            << depth_frame.front_invalid_fraction << ',' << snapshot.battery_voltage_v << ','
            << snapshot.battery_soc;
        for (float value : snapshot.q) log << ',' << value;
        for (float value : snapshot.dq) log << ',' << value;
        for (float value : snapshot.tau_est) log << ',' << value;
        for (float value : snapshot.motor_temperature_c) log << ',' << value;

        if (mode == "shadow") {
            auto proprio = build_shadow_proprio(snapshot, defaults, last_action);
            const std::vector<float> ignored_goal(4, 0.0f);
            const auto output = runner->act(depth_frame.normalized, proprio, ignored_goal);
            const auto layers = strict_loco::execute_action_chain(
                output.joint, previous_target, defaults, 0.25f, -6.0f, 6.0f, 0.05f,
                lower, upper);
            previous_target.assign(layers.applied_joint_target.begin(), layers.applied_joint_target.end());
            last_action.assign(layers.executed_raw_action.begin(), layers.executed_raw_action.end());
            log << ',' << output.inference_ms << ',' << layers.modified_count;
            for (float value : layers.model_raw_action) log << ',' << value;
            for (float value : layers.clipped_raw_action) log << ',' << value;
            for (float value : layers.requested_joint_target) log << ',' << value;
            for (float value : layers.applied_joint_target) log << ',' << value;
            for (float value : layers.executed_raw_action) log << ',' << value;
            if (output.inference_ms > 20.0f)
                throw std::runtime_error("strict shadow inference deadline miss");
        }
        log << '\n';
        if ((frame++ % 50) == 0) log.flush();
        next += std::chrono::milliseconds(20);
        std::this_thread::sleep_until(next);
    }
    log.flush();
    spdlog::info("[strict] {} stopped cleanly; log={}; LowCmd was never created",
                 mode, log_path.string());
    return 0;
}

void init_armed_fsm_state()
{
    unitree::robot::go2::shutdown();
    usleep(0.2 * 1e6);
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout())
        throw std::runtime_error("another process is using the LowCmd channel");
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    FSMState::lowstate->set_timeout_ms(50);
    FSMState::lowstate->wait_for_connection();
}
} // namespace

int main(int argc, char** argv)
{
    try {
        auto vm = param::helper(argc, argv);
        const int mode_count = vm.count("preflight") + vm.count("sensor-only") + vm.count("shadow") +
                               vm.count("fixed-zero") + vm.count("fixed-vx");
        if (mode_count > 1) throw std::runtime_error("select exactly one strict run mode");
        const bool no_mode = mode_count == 0;
        if (no_mode || vm.count("preflight")) return run_preflight();
        const auto controller_log_path = setup_runtime_logging();

        const bool powered_mode = vm.count("fixed-zero") || vm.count("fixed-vx");
        const bool record_depth = vm.count("record-depth");
        if (record_depth && !powered_mode)
            throw std::runtime_error("--record-depth requires --fixed-zero or --fixed-vx");
        const bool suspended_test = vm.count("suspended-test");
        const bool ground_test = vm.count("ground-test");
        if (!strict_loco::valid_suspended_test_request(
                suspended_test, vm.count("fixed-zero"), vm.count("arm"))) {
            throw std::runtime_error(
                "--suspended-test requires the exact combination --fixed-zero --arm");
        }
        if (!strict_loco::valid_ground_test_request(
                ground_test, suspended_test, vm.count("fixed-zero"), vm.count("arm"))) {
            throw std::runtime_error(
                "--ground-test requires --fixed-zero --arm and cannot be combined with --suspended-test");
        }
        if (run_preflight(powered_mode) != 0)
            throw std::runtime_error("preflight failed before DDS/LowCmd initialization");

        unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
        if (vm.count("sensor-only")) return run_no_command_mode("sensor-only");
        if (vm.count("shadow")) return run_no_command_mode("shadow");

        if (!vm.count("arm"))
            throw std::runtime_error("powered mode requires explicit --arm");
        auto vision = param::config["FSM"]["VisionLoco"];
        vision["command_source"] = "fixed";
        vision["strict_safety"]["suspended_test_bypass"] = suspended_test;
        vision["strict_safety"]["ground_test_permissive"] = ground_test;
        vision["strict_safety"]["walking_test_permissive"] = vm.count("fixed-vx") > 0;
        vision["strict_safety"]["depth_capture_dump_on_exit"] = record_depth;
        if (vm.count("fixed-zero")) {
            vision["fixed_cmd"] = std::vector<float>{0.0f, 0.0f, 0.0f};
        } else {
            const float vx = vm["vx"].as<float>();
            if (!strict_loco::valid_fixed_vx(vx))
                throw std::runtime_error("fixed-vx must be finite and within [0,1.00] m/s");
            vision["fixed_cmd"] = std::vector<float>{vx, 0.0f, 0.0f};
        }

        init_armed_fsm_state();
        auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
        fsm->start();
        spdlog::warn(
            "[strict] ARMED GATE: LT+A -> FixStand, then LT+X -> VisionLoco; "
            "LT+B -> Passive; persistent_log={}", controller_log_path.string());
        if (suspended_test)
            spdlog::critical(
                "[strict] SUSPENDED TEST ACTIVE: optional startup gate bypassed; "
                "command is fixed zero");
        if (ground_test)
            spdlog::critical(
                "[strict] GROUND TEST ACTIVE: tracking soft gate is disabled; "
                "minimum hard guards remain");
        if (vm.count("fixed-vx"))
            spdlog::critical(
                "[strict] WALK TEST ACTIVE: first valid post-reset inference publishes; "
                "post-slew motion audit is used and tracking soft stop is disabled");
        if (record_depth)
            spdlog::warn(
                "[strict] DEPTH RECORDING ACTIVE: recent raw/mask/preview frames "
                "will be saved on exit");
        while (keep_running.load()) sleep(1);
        spdlog::warn("[strict] stop signal received; stopping FSM and LowCmd publication");
        fsm->stop();
        unitree::robot::go2::shutdown();
        spdlog::info("[strict] controller stopped cleanly; persistent_log={}",
                     controller_log_path.string());
        spdlog::default_logger()->flush();
        return 0;
    } catch (const std::exception& error) {
        spdlog::critical("strict controller stopped: {}", error.what());
        return 1;
    }
}
