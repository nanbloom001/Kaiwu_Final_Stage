// Copyright (c) 2026.
// 深度源（可插拔）：给 LocoRunner 喂归一化深度 [1,180,320,1]（NHWC, row-major）。
//
//   归一化与训练/Python 验证脚本严格一致（vision_nav_infer_test.py::normalize_depth）：
//       d(米) → clamp(d,0,5)/5；无效像素(<=0 / >=5 / 非有限) → 0.0
//
//   ★ 内参/FOV 对齐（本版相对旧版的改动）：
//     旧版用"独立 x/y 最近邻缩放"把真机分辨率塞到 320x180，隐含假设真机与 sim 的
//     FOV 完全相同。实测不成立：sim(PinholeCameraCfg) HFOV=87.0° VFOV=56.18°
//     (fx=fy=168.61, cx=160, cy=90)；D435i 深度规格 87°x58°，VFOV 偏宽 1.82°、
//     等效 fy 偏小 3.7%，导致同一物体竖直方向落在的行号偏移最多 ±3 行（边缘）。
//     本版改为"内参感知重投影"：对每个 sim 输出像素按 sim 针孔模型算射线，用真机
//     运行时 rs2_intrinsics（实测 per-unit）投影回真机像素采样，一次性纠正
//     HFOV/VFOV/主点/各向异性。两边 "depth" 都是到像平面的 Z 垂直距离，故按射线
//     重投影后 Z 值可直接复用（前提：相机指向一致，即外参对齐——属另一议题）。
//
//   原始采集 profile 不是网络输入契约。当 424x240@30 不可用时，仅允许回退到
//   已审计的 480x270@30；两者都必须用当前 profile 的运行时内参重投影为 320x180。
//   - RealSenseDepth: librealsense2 直连 D435i，后台线程持续抓帧；线程安全 get()。
//   - ConstantDepth : 固定值（默认 1.0 = 远处无障碍），相机未接时的安全 bring-up 兜底。

#pragma once

#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <spdlog/spdlog.h>

namespace vision_nav
{

constexpr int DEPTH_H = 180, DEPTH_W = 320;
constexpr float MAX_DEPTH_M = 5.0f;

// ── sim 相机内参（训练固定，PinholeCameraCfg focal=1.88 h_aperture=3.568 @320x180）──
//    HFOV=87.00° VFOV=56.18°，方形像素。重投影的目标坐标系即此。
constexpr float SIM_FX = 168.61f, SIM_FY = 168.61f;
constexpr float SIM_CX = 160.0f,  SIM_CY = 90.0f;

// 真机相机内参（运行时由 rs2_intrinsics 填）。W/H = 真机抓帧分辨率。
struct CamIntrin { float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f; int W = 0, H = 0; };

struct DepthFrame
{
    std::vector<float> normalized;
    uint64_t frame_number = 0;
    double sensor_timestamp_ms = 0.0;
    std::chrono::steady_clock::time_point received_at{};
    int width = 0, height = 0, fps = 0;
    CamIntrin intrinsics{};
    float invalid_fraction = 1.0f;
    float front_invalid_fraction = 1.0f;
    float min_depth_m = 0.0f;
    float mean_depth_m = 0.0f;
    bool valid = false;
    std::string serial;
    std::string firmware;
};

// 旧法：任意 HxW 米制深度，最近邻 resize 到 (DEPTH_H,DEPTH_W) 并归一化。保留作兜底。
inline void normalize_depth_into(const float* d_m, int H, int W, std::vector<float>& out)
{
    out.resize(DEPTH_H * DEPTH_W);
    for (int y = 0; y < DEPTH_H; ++y) {
        int sy = (H == DEPTH_H) ? y : (int)((long)y * H / DEPTH_H);
        if (sy >= H) sy = H - 1;
        for (int x = 0; x < DEPTH_W; ++x) {
            int sx = (W == DEPTH_W) ? x : (int)((long)x * W / DEPTH_W);
            if (sx >= W) sx = W - 1;
            float v = d_m[sy * W + sx];
            bool invalid = !std::isfinite(v) || v <= 0.0f || v >= MAX_DEPTH_M;
            out[y * DEPTH_W + x] = invalid ? 0.0f : (v / MAX_DEPTH_M);
        }
    }
}

// 新法：内参感知重投影。把真机米制深度 d_m（尺寸 rin.W×rin.H、内参 rin）重采样到
// sim 针孔模型（320×180, SIM_FX/FY/CX/CY），并归一化写入 out(57600)。
//   对每个 sim 像素 (u,v)：
//     归一化像平面坐标 xn=(u-CX)/FX, yn=(v-CY)/FY  →  真机像素 ur=fx*xn+cx, vr=fy*yn+cy
//   视野外 → 0.0（当无效/远）。最近邻采样（深度图不宜双线性，会在边沿插出伪值）。
inline void reproject_normalize_into(const float* d_m, const CamIntrin& rin, std::vector<float>& out)
{
    out.resize(DEPTH_H * DEPTH_W);
    for (int v = 0; v < DEPTH_H; ++v) {
        float yn = ((float)v - SIM_CY) / SIM_FY;
        for (int u = 0; u < DEPTH_W; ++u) {
            float xn = ((float)u - SIM_CX) / SIM_FX;
            int ur = (int)std::lround(rin.fx * xn + rin.cx);
            int vr = (int)std::lround(rin.fy * yn + rin.cy);
            float val = 0.0f;  // 默认：视野外 = 无效
            if (ur >= 0 && ur < rin.W && vr >= 0 && vr < rin.H) {
                float dm = d_m[(size_t)vr * rin.W + ur];
                bool invalid = !std::isfinite(dm) || dm <= 0.0f || dm >= MAX_DEPTH_M;
                val = invalid ? 0.0f : (dm / MAX_DEPTH_M);
            }
            out[(size_t)v * DEPTH_W + u] = val;
        }
    }
}

class DepthSource
{
public:
    virtual ~DepthSource() {}
    // 返回最近一帧归一化深度（长度 57600）。线程安全。
    virtual DepthFrame get() = 0;
};

class ConstantDepth : public DepthSource
{
public:
    explicit ConstantDepth(float norm_value = 1.0f)
    : frame_{}
    {
        frame_.normalized.assign(DEPTH_H * DEPTH_W, norm_value);
        frame_.width = DEPTH_W;
        frame_.height = DEPTH_H;
        frame_.received_at = std::chrono::steady_clock::now();
        frame_.valid = true;
        spdlog::warn("[depth] 使用 ConstantDepth({:.2f})：无真实深度，仅用于无相机 bring-up。", norm_value);
    }
    DepthFrame get() override
    {
        frame_.received_at = std::chrono::steady_clock::now();
        ++frame_.frame_number;
        return frame_;
    }
private:
    DepthFrame frame_;
};

// Optional RealSense post-processing. Hole filling and decimation stay disabled
// because they can erase thin obstacle boundaries seen by the policy.
struct FilterConfig
{
    std::string mode = "none";
    float spatial_smooth_alpha = 0.5f;
    int spatial_magnitude = 2;
    int spatial_smooth_delta = 20;
    float temporal_alpha = 0.25f;
    float temporal_delta = 20.0f;
};

struct RealSenseConfig
{
    FilterConfig filters{};
    bool high_density_preset = true;
    bool emitter_enabled = true;
    bool max_laser_power = true;
    bool auto_exposure = true;
};

#ifdef USE_REALSENSE
} // namespace vision_nav
#include <librealsense2/rs.hpp>
namespace vision_nav
{

class RealSenseDepth : public DepthSource
{
public:
    RealSenseDepth(int want_w = 424, int want_h = 240, int want_fps = 30,
                   const RealSenseConfig& settings = {})
    : filt_(settings.filters)
    {
        int cw, ch, cfps;
        _pick_profile(want_w, want_h, want_fps, cw, ch, cfps);
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_DEPTH, cw, ch, RS2_FORMAT_Z16, cfps);
        auto profile = pipe_.start(cfg);
        auto device = profile.get_device();
        auto depth_sensor = device.first<rs2::depth_sensor>();
        units_ = depth_sensor.get_depth_scale();  // raw→米
        serial_ = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        firmware_ = device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION);
        actual_w_ = cw; actual_h_ = ch; actual_fps_ = cfps;
        if (!(std::isfinite(units_) && units_ > 0.0f))
            throw std::runtime_error("strict RealSense depth scale is invalid");

        auto set_option = [&depth_sensor](rs2_option option, float value, const char* name) {
            if (!depth_sensor.supports(option)) {
                spdlog::warn("[depth] option {} is not supported by this device", name);
                return;
            }
            depth_sensor.set_option(option, value);
            spdlog::info("[depth] option {}={:.3f}", name, depth_sensor.get_option(option));
        };
        if (settings.high_density_preset)
            set_option(RS2_OPTION_VISUAL_PRESET,
                       static_cast<float>(RS2_RS400_VISUAL_PRESET_HIGH_DENSITY),
                       "visual_preset(high_density)");
        if (settings.emitter_enabled)
            set_option(RS2_OPTION_EMITTER_ENABLED, 1.0f, "emitter_enabled");
        if (settings.max_laser_power && depth_sensor.supports(RS2_OPTION_LASER_POWER)) {
            const auto range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
            set_option(RS2_OPTION_LASER_POWER, range.max, "laser_power(max)");
        }
        if (settings.auto_exposure)
            set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.0f, "auto_exposure");

        // ── 读真机深度内参（实测 per-unit），重投影到 sim 针孔模型 ──
        try {
            auto dvsp = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            rs2_intrinsics in = dvsp.get_intrinsics();
            rin_.fx = in.fx; rin_.fy = in.fy; rin_.cx = in.ppx; rin_.cy = in.ppy;
            rin_.W = in.width; rin_.H = in.height;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("strict RealSense intrinsics unavailable: ") + e.what());
        }

        if (!(std::isfinite(rin_.fx) && std::isfinite(rin_.fy) &&
              std::isfinite(rin_.cx) && std::isfinite(rin_.cy) &&
              rin_.fx > 0.0f && rin_.fy > 0.0f &&
              rin_.cx >= 0.0f && rin_.cx < cw && rin_.cy >= 0.0f && rin_.cy < ch &&
              rin_.W == cw && rin_.H == ch))
            throw std::runtime_error("strict RealSense intrinsics are invalid or profile-sized differently");

        if (rin_.fx > 0.0f) {
            float hfov = 2.0f * std::atan(rin_.W / (2.0f * rin_.fx)) * 57.29578f;
            float vfov = 2.0f * std::atan(rin_.H / (2.0f * rin_.fy)) * 57.29578f;
            spdlog::info("[depth] RealSense {}x{}@{}fps scale={:.5f}", cw, ch, cfps, units_);
            spdlog::info("[depth] 真机内参 fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}  FOV={:.1f}x{:.1f}°",
                         rin_.fx, rin_.fy, rin_.cx, rin_.cy, hfov, vfov);
            spdlog::info("[depth] 重投影→sim 针孔(fx={:.1f} cx={:.0f} cy={:.0f}  FOV=87.0x56.2°) 320x180",
                         SIM_FX, SIM_CX, SIM_CY);
        }

        use_spatial_ = filt_.mode == "light_spatial" ||
                       filt_.mode == "light_spatial_weak_temporal";
        use_temporal_ = filt_.mode == "light_spatial_weak_temporal";
        if (use_spatial_) {
            spatial_.set_option(RS2_OPTION_FILTER_MAGNITUDE,
                                static_cast<float>(filt_.spatial_magnitude));
            spatial_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA,
                                filt_.spatial_smooth_alpha);
            spatial_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA,
                                static_cast<float>(filt_.spatial_smooth_delta));
            spatial_.set_option(RS2_OPTION_HOLES_FILL, 0.0f);
        }
        if (use_temporal_) {
            temporal_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, filt_.temporal_alpha);
            temporal_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, filt_.temporal_delta);
            temporal_.set_option(RS2_OPTION_HOLES_FILL, 0.0f);
        }
        spdlog::info("[depth] filter={} spatial={} temporal={} hole_filling=OFF",
                     filt_.mode, use_spatial_, use_temporal_);
        spdlog::warn(
            "[depth] pixel validity is diagnostic-only; no invalid-fraction or frame-stale "
            "condition requests takeover; training normalization remains (0,5m)");

        frame_.normalized.assign(DEPTH_H * DEPTH_W, 0.0f);
        frame_.width = actual_w_; frame_.height = actual_h_; frame_.fps = actual_fps_;
        frame_.intrinsics = rin_; frame_.serial = serial_; frame_.firmware = firmware_;
        running_ = true;
        grab_ = std::thread([this]{ _loop(); });
    }

    ~RealSenseDepth() override
    {
        running_ = false;
        if (grab_.joinable()) grab_.join();
        try { pipe_.stop(); } catch (...) {}
    }

    DepthFrame get() override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return frame_;
    }

private:
    void _pick_profile(int ww, int wh, int wf, int& cw, int& ch, int& cf)
    {
        rs2::context ctx;
        auto devs = ctx.query_devices();
        if (devs.size() == 0) throw std::runtime_error("没找到 RealSense 设备（确认 D435i USB 直连 Jetson）。");
        struct P { int w, h, f; };
        std::vector<P> profs;
        for (auto&& s : devs[0].query_sensors())
            for (auto&& p : s.get_stream_profiles())
                if (p.stream_type() == RS2_STREAM_DEPTH && p.format() == RS2_FORMAT_Z16) {
                    auto vp = p.as<rs2::video_stream_profile>();
                    profs.push_back({vp.width(), vp.height(), p.fps()});
                }
        if (profs.empty()) throw std::runtime_error("该 D435i 没有可用的深度 z16 模式。");
        const std::array<P, 2> allowed{{P{ww, wh, wf}, P{480, 270, 30}}};
        for (size_t candidate = 0; candidate < allowed.size(); ++candidate) {
            for (const auto& p : profs) {
                if (p.w == allowed[candidate].w && p.h == allowed[candidate].h &&
                    p.f == allowed[candidate].f) {
                    cw = p.w; ch = p.h; cf = p.f;
                    if (candidate != 0) {
                        spdlog::warn(
                            "[depth] requested {}x{}@{} unavailable; using audited acquisition profile {}x{}@{}",
                            ww, wh, wf, cw, ch, cf);
                    }
                    return;
                }
            }
        }
        throw std::runtime_error(
            "D435i lacks an audited strict Z16 profile (424x240@30 or 480x270@30)");
    }

    void _loop()
    {
        std::vector<float> meters;
        while (running_) {
            rs2::frameset frames;
            try {
                if (!pipe_.try_wait_for_frames(&frames, 1000)) continue;
            } catch (const std::exception& e) {
                spdlog::warn("[depth] wait_for_frames: {}", e.what());
                continue;
            }
            auto depth = frames.get_depth_frame();
            if (!depth) continue;
            if (use_spatial_) depth = spatial_.process(depth);
            if (use_temporal_) depth = temporal_.process(depth);
            int W = depth.get_width(), H = depth.get_height();
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(depth.get_data());
            meters.resize((size_t)W * H);
            for (size_t i = 0; i < (size_t)W * H; ++i) meters[i] = raw[i] * units_;
            std::vector<float> norm;
            // Strict mode never silently falls back to a different camera geometry.
            if (!(rin_.fx > 0.0f && rin_.fy > 0.0f && rin_.W == W && rin_.H == H))
                continue;
            reproject_normalize_into(meters.data(), rin_, norm);
            size_t invalid = 0, front_invalid = 0, valid_count = 0;
            double sum = 0.0;
            float min_depth = MAX_DEPTH_M;
            const int x0 = DEPTH_W / 3, x1 = 2 * DEPTH_W / 3;
            for (int y = 0; y < DEPTH_H; ++y) for (int x = 0; x < DEPTH_W; ++x) {
                const float normalized = norm[(size_t)y * DEPTH_W + x];
                const bool bad = !std::isfinite(normalized) || normalized <= 0.0f;
                if (bad) {
                    ++invalid;
                    if (x >= x0 && x < x1) ++front_invalid;
                } else {
                    const float value_m = normalized * MAX_DEPTH_M;
                    ++valid_count; sum += value_m; min_depth = std::min(min_depth, value_m);
                }
            }
            DepthFrame next;
            next.normalized.swap(norm);
            next.frame_number = depth.get_frame_number();
            next.sensor_timestamp_ms = depth.get_timestamp();
            next.received_at = std::chrono::steady_clock::now();
            next.width = W; next.height = H; next.fps = actual_fps_;
            next.intrinsics = rin_; next.serial = serial_; next.firmware = firmware_;
            next.invalid_fraction = static_cast<float>(invalid) /
                static_cast<float>((size_t)DEPTH_W * DEPTH_H);
            next.front_invalid_fraction = static_cast<float>(front_invalid) /
                static_cast<float>((size_t)(x1 - x0) * DEPTH_H);
            next.min_depth_m = valid_count ? min_depth : 0.0f;
            next.mean_depth_m = valid_count ? static_cast<float>(sum / valid_count) : 0.0f;
            // Pixel holes remain encoded as zero exactly as in training. Frame validity only
            // describes whether a structurally complete camera frame reached the consumer.
            next.valid = next.normalized.size() == DEPTH_H * DEPTH_W;
            std::lock_guard<std::mutex> lk(mtx_);
            frame_ = std::move(next);
        }
    }

    rs2::pipeline pipe_;
    float units_ = 0.001f;
    CamIntrin rin_{};
    DepthFrame frame_;
    std::mutex mtx_;
    std::thread grab_;
    std::atomic<bool> running_{false};
    FilterConfig filt_{};
    bool use_spatial_ = false;
    bool use_temporal_ = false;
    int actual_w_ = 0, actual_h_ = 0, actual_fps_ = 0;
    std::string serial_, firmware_;
    rs2::spatial_filter spatial_;
    rs2::temporal_filter temporal_;
};
#endif // USE_REALSENSE

} // namespace vision_nav
