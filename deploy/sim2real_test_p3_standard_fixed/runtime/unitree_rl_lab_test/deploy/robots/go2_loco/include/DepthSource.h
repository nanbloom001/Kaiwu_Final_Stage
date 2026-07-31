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
//   - RealSenseDepth: librealsense2 直连 D435i，后台线程持续抓帧；线程安全 get()。
//   - ConstantDepth : 固定值（默认 1.0 = 远处无障碍），相机未接时的安全 bring-up 兜底。

#pragma once

#include <atomic>
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
    virtual std::vector<float> get() = 0;
};

class ConstantDepth : public DepthSource
{
public:
    explicit ConstantDepth(float norm_value = 1.0f)
    : buf_(DEPTH_H * DEPTH_W, norm_value)
    {
        spdlog::warn("[depth] 使用 ConstantDepth({:.2f})：无真实深度，仅用于无相机 bring-up。", norm_value);
    }
    std::vector<float> get() override { return buf_; }
private:
    std::vector<float> buf_;
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

#ifdef USE_REALSENSE
} // namespace vision_nav
#include <librealsense2/rs.hpp>
namespace vision_nav
{

class RealSenseDepth : public DepthSource
{
public:
    RealSenseDepth(int want_w = 424, int want_h = 240, int want_fps = 30,
                   const FilterConfig& filters = {})
    : filt_(filters)
    {
        int cw, ch, cfps;
        _pick_profile(want_w, want_h, want_fps, cw, ch, cfps);
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_DEPTH, cw, ch, RS2_FORMAT_Z16, cfps);
        auto profile = pipe_.start(cfg);
        units_ = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();  // raw→米

        // ── 读真机深度内参（实测 per-unit），重投影到 sim 针孔模型 ──
        try {
            auto dvsp = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            rs2_intrinsics in = dvsp.get_intrinsics();
            rin_.fx = in.fx; rin_.fy = in.fy; rin_.cx = in.ppx; rin_.cy = in.ppy;
            rin_.W = in.width; rin_.H = in.height;
        } catch (const std::exception& e) {
            spdlog::warn("[depth] 取 rs2_intrinsics 失败({})，回退最近邻缩放。", e.what());
            rin_ = CamIntrin{};
        }

        if (rin_.fx > 0.0f) {
            float hfov = 2.0f * std::atan(rin_.W / (2.0f * rin_.fx)) * 57.29578f;
            float vfov = 2.0f * std::atan(rin_.H / (2.0f * rin_.fy)) * 57.29578f;
            spdlog::info("[depth] RealSense {}x{}@{}fps scale={:.5f}", cw, ch, cfps, units_);
            spdlog::info("[depth] 真机内参 fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}  FOV={:.1f}x{:.1f}°",
                         rin_.fx, rin_.fy, rin_.cx, rin_.cy, hfov, vfov);
            spdlog::info("[depth] 重投影→sim 针孔(fx={:.1f} cx={:.0f} cy={:.0f}  FOV=87.0x56.2°) 320x180",
                         SIM_FX, SIM_CX, SIM_CY);
        } else {
            spdlog::warn("[depth] RealSense {}x{}@{}fps scale={:.5f}（无内参，最近邻缩放→320x180）",
                         cw, ch, cfps, units_);
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

        buf_.assign(DEPTH_H * DEPTH_W, 0.0f);
        running_ = true;
        grab_ = std::thread([this]{ _loop(); });
    }

    ~RealSenseDepth() override
    {
        running_ = false;
        if (grab_.joinable()) grab_.join();
        try { pipe_.stop(); } catch (...) {}
    }

    std::vector<float> get() override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return buf_;
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
        // 优先精确命中；否则挑 <=848 宽、30fps、最接近目标的
        int best = -1; long best_score = (1L << 60);
        for (int i = 0; i < (int)profs.size(); ++i) {
            auto& p = profs[i];
            if (p.w == ww && p.h == wh && p.f == wf) { best = i; break; }
            long fps_rank = (p.f == 30) ? 0 : (p.f == 15 ? 1 : (p.f == 6 ? 2 : 3));
            long score = ((p.w <= 848) ? 0 : 1) * 1000000L + fps_rank * 100000L
                       + std::abs(p.w - ww) + std::abs(p.h - wh);
            if (score < best_score) { best_score = score; best = i; }
        }
        cw = profs[best].w; ch = profs[best].h; cf = profs[best].f;
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
            // 内参就绪 → 重投影对齐 sim FOV；否则回退旧的最近邻缩放
            if (rin_.fx > 0.0f && rin_.W == W && rin_.H == H)
                reproject_normalize_into(meters.data(), rin_, norm);
            else
                normalize_depth_into(meters.data(), H, W, norm);
            std::lock_guard<std::mutex> lk(mtx_);
            buf_.swap(norm);
        }
    }

    rs2::pipeline pipe_;
    float units_ = 0.001f;
    CamIntrin rin_{};
    std::vector<float> buf_;
    std::mutex mtx_;
    std::thread grab_;
    std::atomic<bool> running_{false};
    FilterConfig filt_{};
    bool use_spatial_ = false;
    bool use_temporal_ = false;
    rs2::spatial_filter spatial_;
    rs2::temporal_filter temporal_;
};
#endif // USE_REALSENSE

} // namespace vision_nav
