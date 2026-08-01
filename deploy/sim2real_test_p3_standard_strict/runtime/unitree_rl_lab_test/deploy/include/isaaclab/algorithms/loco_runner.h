// Copyright (c) 2026.
// LocoRunner — 多输入/多输出 ONNX 推理节点（Go2 lbc_loco 阶段部署）。
//
// 与 export_loco_onnx.py 导出的图严格对应。为了让 loco 阶段复用与 vision-nav
// **完全相同**的 obs 装配 / 部署链路，本图保留 8 入 8 出端口；loco 用不到的 nav
// 端口不参与计算：
//   输入(8): depth[1,180,320,1] proprio[1,45] goal[1,4]        ← goal 接收但忽略
//            loco_h[2,1,64] loco_c[2,1,64]                      ← loco LSTM 真实状态
//            nav_h[2,1,64] nav_c[2,1,64]                        ← 接收并原样透传（不计算）
//            cmd_override[1,4] = [vx,vy,wz,gate]                ← 速度命令经 proprio[6:9] 进入，
//                                                                此端口仅用于 cmd 回显
//   输出(8): cmd[1,3]       = cmd_override[:, :3]（回显输入速度命令）
//            cmd_raw[1,3]   = cmd
//            clearance[1,3] = 0（loco 无 clearance）
//            joint[1,12]    = teacher_actor 原始动作（未 scale/offset）
//            loco_h_out loco_c_out                              ← loco LSTM 真实回喂
//            nav_h_out nav_c_out                                ← = nav_h / nav_c 透传
//
// 关键契约（与 vision_nav_runner.h 一致，仅 nav 分支语义为透传/占位）：
//   1) proprio[6:9] = **上一帧** 的 clamp 后 cmd（速度指令槽）。本类内部持有 prev_cmd_，
//      每帧在推理前写入传入 proprio 的 [6:9]；首帧为 0。
//   2) loco LSTM 状态由本类内部持有并逐帧回喂；reset() 清零。nav 状态同样持有并回喂，
//      但图内为恒等透传（loco 阶段无 nav 分支），保留仅为端口兼容。
//   3) cmd 是输入速度命令的回显（图内 cmd = cmd_override[:, :3]），这里不再 clamp。
//   4) 输出 joint(12) 是 teacher_actor 的**原始动作**（未乘 scale / 未加 offset）。
//      关节目标 = offset + scale*joint，由 State 负责；
//      回喂进 proprio[33:45] 的 last_action 也用这个**原始** joint（与训练一致）。

#pragma once

#include "onnxruntime_cxx_api.h"
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

struct LocoOutput
{
    std::vector<float> joint;      // 12  原始动作（未 scale/offset）
    std::vector<float> cmd;        // 3   (vx,vy,wz) 输入速度命令的回显
    std::vector<float> cmd_raw;    // 3   同 cmd
    std::vector<float> clearance;  // 3   loco 阶段恒 0
    float inference_ms = 0.0f;
};

class LocoRunner
{
public:
    // 维度常量（与导出图固定一致，batch=1）
    static constexpr int DEPTH_H = 180, DEPTH_W = 320, DEPTH_C = 1;
    static constexpr int DEPTH_N = DEPTH_H * DEPTH_W * DEPTH_C;  // 57600
    static constexpr int PROPRIO_DIM = 45;
    static constexpr int GOAL_DIM = 4;
    static constexpr int LSTM_LAYERS = 2, LSTM_HIDDEN = 64;
    static constexpr int STATE_N = LSTM_LAYERS * 1 * LSTM_HIDDEN;  // 128
    static constexpr int NUM_ACTIONS = 12, NUM_CMD = 3, NUM_CLR = 3;

    explicit LocoRunner(const std::string& model_path)
    {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "loco");
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_options_.SetIntraOpNumThreads(2);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        // 缓存输入/输出名（按图声明顺序）。先存进 owned 持久字符串，再取稳定指针。
        for (size_t i = 0; i < session_->GetInputCount(); ++i)
            in_names_owned_.emplace_back(session_->GetInputNameAllocated(i, alloc_).get());
        for (size_t i = 0; i < session_->GetOutputCount(); ++i)
            out_names_owned_.emplace_back(session_->GetOutputNameAllocated(i, alloc_).get());
        for (auto& s : in_names_owned_)  in_names_.push_back(s.c_str());
        for (auto& s : out_names_owned_) out_names_.push_back(s.c_str());

        _check_io();
        reset();
    }

    void reset()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        loco_h_.assign(STATE_N, 0.0f); loco_c_.assign(STATE_N, 0.0f);
        nav_h_.assign(STATE_N, 0.0f);  nav_c_.assign(STATE_N, 0.0f);
        prev_cmd_.assign(NUM_CMD, 0.0f);
        last_out_ = LocoOutput{std::vector<float>(NUM_ACTIONS, 0.0f),
                               std::vector<float>(NUM_CMD, 0.0f),
                               std::vector<float>(NUM_CMD, 0.0f),
                               std::vector<float>(NUM_CLR, 0.0f),
                               0.0f};
    }

    // depth: 长度 57600（NHWC, 已归一化 [0,1]）
    // proprio: 长度 45（本函数会把 [6:9] 覆写为上一帧 cmd）
    // goal: 长度 4（loco 阶段图内忽略，仍需传入以填满端口）
    LocoOutput act(const std::vector<float>& depth,
                   std::vector<float> proprio,
                   const std::vector<float>& goal)
    {
        if ((int)depth.size() != DEPTH_N)   throw std::runtime_error("depth size != 57600");
        if ((int)proprio.size() != PROPRIO_DIM) throw std::runtime_error("proprio size != 45");
        if ((int)goal.size() != GOAL_DIM)   throw std::runtime_error("goal size != 4");

        std::lock_guard<std::mutex> lk(mtx_);

        // 契约①：proprio[6:9] = 上一帧 cmd（override 开启时改为固定注入值）
        const std::vector<float>& cmd_in = cmd_override_on_ ? cmd_override_ : prev_cmd_;
        proprio[6] = cmd_in[0];
        proprio[7] = cmd_in[1];
        proprio[8] = cmd_in[2];

        auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // name → (data ptr, element count, shape)
        std::unordered_map<std::string, std::pair<float*, std::vector<int64_t>>> bufs;
        bufs["depth"]   = {const_cast<float*>(depth.data()),   {1, DEPTH_H, DEPTH_W, DEPTH_C}};
        bufs["proprio"] = {proprio.data(),                     {1, PROPRIO_DIM}};
        bufs["goal"]    = {const_cast<float*>(goal.data()),    {1, GOAL_DIM}};
        bufs["loco_h"]  = {loco_h_.data(), {LSTM_LAYERS, 1, LSTM_HIDDEN}};
        bufs["loco_c"]  = {loco_c_.data(), {LSTM_LAYERS, 1, LSTM_HIDDEN}};
        bufs["nav_h"]   = {nav_h_.data(),  {LSTM_LAYERS, 1, LSTM_HIDDEN}};
        bufs["nav_c"]   = {nav_c_.data(),  {LSTM_LAYERS, 1, LSTM_HIDDEN}};
        bufs["cmd_override"] = {cmd_override_.data(), {1, 4}};

        std::vector<Ort::Value> in_tensors;
        in_tensors.reserve(in_names_.size());
        for (const char* nm : in_names_) {
            auto it = bufs.find(nm);
            if (it == bufs.end()) throw std::runtime_error(std::string("unexpected input: ") + nm);
            float* ptr = it->second.first;
            auto& shp = it->second.second;
            size_t cnt = 1; for (auto d : shp) cnt *= (size_t)d;
            in_tensors.push_back(Ort::Value::CreateTensor<float>(
                mem, ptr, cnt, shp.data(), shp.size()));
        }

        const auto infer_start = std::chrono::steady_clock::now();
        auto outs = session_->Run(Ort::RunOptions{nullptr},
                                  in_names_.data(), in_tensors.data(), in_tensors.size(),
                                  out_names_.data(), out_names_.size());
        const auto infer_end = std::chrono::steady_clock::now();

        auto get = [&](const char* name) -> const float* {
            for (size_t i = 0; i < out_names_.size(); ++i)
                if (std::strcmp(out_names_[i], name) == 0)
                    return outs[i].GetTensorMutableData<float>();
            throw std::runtime_error(std::string("missing output: ") + name);
        };

        LocoOutput r;
        r.cmd.assign(get("cmd"), get("cmd") + NUM_CMD);
        r.cmd_raw.assign(get("cmd_raw"), get("cmd_raw") + NUM_CMD);
        r.clearance.assign(get("clearance"), get("clearance") + NUM_CLR);
        r.joint.assign(get("joint"), get("joint") + NUM_ACTIONS);
        r.inference_ms = std::chrono::duration<float, std::milli>(
            infer_end - infer_start).count();

        // 契约②：回喂 LSTM 状态（loco 真实更新；nav 为透传，回喂等于原样写回）
        std::memcpy(loco_h_.data(), get("loco_h_out"), STATE_N * sizeof(float));
        std::memcpy(loco_c_.data(), get("loco_c_out"), STATE_N * sizeof(float));
        std::memcpy(nav_h_.data(),  get("nav_h_out"),  STATE_N * sizeof(float));
        std::memcpy(nav_c_.data(),  get("nav_c_out"),  STATE_N * sizeof(float));

        // 契约①：保存本帧 cmd 供下一帧 proprio[6:9]（override 模式下保持固定值）
        prev_cmd_ = r.cmd;

        last_out_ = r;
        return r;
    }

    LocoOutput last() { std::lock_guard<std::mutex> lk(mtx_); return last_out_; }

    // 速度命令注入：把喂给 teacher_actor 的速度指令（proprio[6:9]）强制成固定值。
    // loco 阶段 cmd 始终来自外部（fixed/uwb），因此常态 on=true。
    // on=false 时回落到上一帧 cmd（loco 阶段无 nav actor 产出，等价于 0）。
    void set_cmd_override(bool on, float vx = 0.f, float vy = 0.f, float wz = 0.f)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        cmd_override_on_ = on;
        cmd_override_    = {vx, vy, wz, on ? 1.0f : 0.0f};
    }

private:
    void _check_io()
    {
        struct Port { const char* name; std::vector<int64_t> shape; };
        const std::vector<Port> expected_inputs{
            {"depth", {1,180,320,1}}, {"proprio", {1,45}}, {"goal", {1,4}},
            {"loco_h", {2,1,64}}, {"loco_c", {2,1,64}}, {"nav_h", {2,1,64}},
            {"nav_c", {2,1,64}}, {"cmd_override", {1,4}}};
        const std::vector<Port> expected_outputs{
            {"cmd", {1,3}}, {"cmd_raw", {1,3}}, {"clearance", {1,3}}, {"joint", {1,12}},
            {"loco_h_out", {2,1,64}}, {"loco_c_out", {2,1,64}},
            {"nav_h_out", {2,1,64}}, {"nav_c_out", {2,1,64}}};
        if (session_->GetInputCount() != expected_inputs.size() ||
            session_->GetOutputCount() != expected_outputs.size())
            throw std::runtime_error("ONNX ABI requires exactly 8 inputs and 8 outputs");

        auto check = [&](bool input, const std::vector<Port>& expected) {
            for (const auto& port : expected) {
                size_t index = expected.size();
                const auto& names = input ? in_names_owned_ : out_names_owned_;
                for (size_t i = 0; i < names.size(); ++i)
                    if (names[i] == port.name) { index = i; break; }
                if (index == expected.size())
                    throw std::runtime_error(std::string("ONNX missing ") +
                        (input ? "input: " : "output: ") + port.name);
                Ort::TypeInfo type = input ? session_->GetInputTypeInfo(index)
                                           : session_->GetOutputTypeInfo(index);
                auto tensor = type.GetTensorTypeAndShapeInfo();
                if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    throw std::runtime_error(std::string("ONNX port is not float32: ") + port.name);
                const auto shape = tensor.GetShape();
                if (shape != port.shape) {
                    std::ostringstream message;
                    message << "ONNX shape mismatch for " << port.name;
                    throw std::runtime_error(message.str());
                }
            }
        };
        check(true, expected_inputs);
        check(false, expected_outputs);

        const std::string ort_version = OrtGetApiBase()->GetVersionString();
        if (ort_version.rfind("1.19.", 0) != 0 && ort_version.rfind("1.20.", 0) != 0 &&
            ort_version.rfind("1.21.", 0) != 0)
            throw std::runtime_error("unapproved ONNX Runtime version: " + ort_version);
    }

    Ort::Env env_{nullptr};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions alloc_;

    std::vector<std::string> in_names_owned_, out_names_owned_;
    std::vector<const char*> in_names_, out_names_;

    // 跨帧状态
    std::vector<float> loco_h_, loco_c_, nav_h_, nav_c_;
    std::vector<float> prev_cmd_;
    bool cmd_override_on_ = false;
    std::vector<float> cmd_override_{0.f, 0.f, 0.f, 0.f};
    LocoOutput last_out_;
    std::mutex mtx_;
};

} // namespace isaaclab
