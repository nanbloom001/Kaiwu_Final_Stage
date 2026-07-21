#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_loco_onnx.py — 把 Go2 lbc_loco 学生策略导出为单图 ONNX（真机部署用）。

本脚本是 sim2real **loco 阶段** 的「卡点 A」产物。它把 lbc_loco 训练 ckpt 里的
  vision_encoder(VisionEncoder: cnn + rnn + rnn_output_layer) + teacher_actor
两段权重，包成 **一个无状态 wrapper module**，forward 逐行复刻 eval/真机推理路径
`AlgorithmLBC.act_student`（agent_ppo/algorithm/algorithm_lbc.py:282-297），
并把 loco LSTM 的 (h,c) 作为**显式输入+输出**导成 ONNX。

=============================================================================
为什么仍保留 8 入 8 出（与 export_vision_nav_onnx.py 同契约）
=============================================================================
loco 阶段模型本身只有一条链路：
    depth → cnn → cat(cnn_feat, proprio) → loco LSTM → loco head → loco_latent
    teacher_actor(cat[proprio, loco_latent]) → joint(12)
没有 nav 分支、没有 goal、没有 clearance、没有 cmd_override 计算。

但为了让 **同一份 C++ runner / State / 部署链路** 不必为 loco 阶段单独改 obs 装配，
本图保留与 vision-nav 图**完全相同的 8 入 8 出端口**，loco 用不到的端口不参与计算：

  输入(8): depth[1,180,320,1] proprio[1,45] goal[1,4]          ← goal 接收但忽略
           loco_h[2,1,64] loco_c[2,1,64]                        ← loco LSTM 真实状态
           nav_h[2,1,64] nav_c[2,1,64]                          ← 接收并原样透传（不计算）
           cmd_override[1,4] = [vx,vy,wz,gate]                  ← 速度命令经 proprio[6:9] 进入，
                                                                  此端口仅用于 cmd 回显
  输出(8): cmd[1,3]       = cmd_override[:, :3]   （回显输入速度命令）
           cmd_raw[1,3]   = cmd                   （同值）
           clearance[1,3] = zeros                 （loco 无 clearance，恒 0）
           joint[1,12]    = teacher_actor 原始动作（未 scale/offset）
           loco_h_out / loco_c_out                （loco LSTM 下一帧回喂，真实）
           nav_h_out / nav_c_out                  （= nav_h / nav_c，原样透传）

这样 State 的 cmd-override 隔离校验（cmd==exec_cmd）与诊断日志（cmd_raw / clr_*）
都不报错，UWB / command_for_frame / proprio 装配全部无需改动。

图内烘进的逻辑（C++ 侧不再做）：
  - cnn + loco LSTM + loco head 全部权重
  - loco_latent 的 L2 normalize（F.normalize p=2）
  - teacher_actor 前向 → joint
  - clearance 恒 0、cmd/cmd_raw = cmd_override[:, :3]、nav 状态透传

图外由 C++ runner 负责（**不在本图内**）：
  - 推理前把 proprio[6:9] 覆写为「上一帧 cmd / 固定注入 cmd」（契约①）
  - joint 的 scale/offset/clip（输出是 teacher_actor 的**原始**动作）
  - loco LSTM 状态的逐帧回喂、episode 起始重置

用法:
  python export_loco_onnx.py \
      --ckpt model.ckpt-vision-XXXX.pth \
      --out  logs/loco/exported/policy.onnx
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 维度常量（与导出图 / loco_runner.h 固定一致，batch=1）
# ============================================================================
DEPTH_H, DEPTH_W, DEPTH_C = 180, 320, 1
PROPRIO_DIM = 45
GOAL_DIM = 4
CLEAR_DIM = 3
LATENT_DIM = 32
CNN_OUT = 32
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
NUM_ACTIONS = 12
NUM_CMD = 3

# loco LSTM 吃 cat[cnn_feat(32), proprio(45)] = 77
LOCO_LSTM_IN = CNN_OUT + PROPRIO_DIM   # 77
# teacher_actor 入口 = cat[proprio(45), loco_latent(32)] = 77
# 注意: 本 ckpt(vision-162404) 训练时把 goal(3) 也拼进了 teacher 输入, 故 = 80
TEACHER_ACTOR_IN = PROPRIO_DIM + LATENT_DIM + 3   # 77 + goal(3) = 80

# teacher MLP 隐层（agent_ppo/conf/conf.py: teacher_actor_hidden_dims = [512,256,128]，
# teacher_actor_activation = "elu"）
TEACHER_HIDDEN = (512, 256, 128)


# ============================================================================
# SimpleCNN —— 与 agent_ppo/model/simple_cnn.py 逐层一致（键名必须对得上 ckpt）
#   state_dict 键: conv_layers.{0,1,2}.0.{weight,bias}, fc.{weight,bias}
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_shape=(DEPTH_H, DEPTH_W, DEPTH_C), output_dim=CNN_OUT,
                 hidden_channels=(32, 64, 128)):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        h, w, in_c = input_shape

        self.conv_layers = nn.ModuleList()
        # Conv1: 1→32, k5 s2, +MaxPool2
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(in_c, hidden_channels[0], kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ))
        # Conv2: 32→64, k3 s2, +MaxPool2
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ))
        # Conv3: 64→128, k3 s1 (no pool)
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        ))

        self._conv_out = self._compute_conv_output_size(h, w)
        self.fc = nn.Linear(self._conv_out, output_dim)

    def _compute_conv_output_size(self, h, w):
        h = (h - 5) // 2 + 1; w = (w - 5) // 2 + 1
        h = h // 2; w = w // 2
        h = (h - 3) // 2 + 1; w = (w - 3) // 2 + 1
        h = h // 2; w = w // 2
        h = (h - 3) // 1 + 1; w = (w - 3) // 1 + 1
        return 128 * h * w   # 128*8*17 = 17408

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        # NHWC → NCHW（导出图固定 batch=1，下面的条件在 trace 期为常量）
        if depth_image.dim() == 4 and depth_image.shape[-1] == self.input_shape[2]:
            depth_image = depth_image.permute(0, 3, 1, 2)
        elif depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(1)
        x = depth_image
        for conv in self.conv_layers:
            x = conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def build_teacher_mlp(in_dim: int, out_dim: int,
                      hidden=TEACHER_HIDDEN, activation=nn.ELU) -> nn.Sequential:
    """复刻 agent.py 里 teacher_actor 的 Sequential 结构。
    层索引: Linear@0 Act@1 Linear@2 Act@3 Linear@4 Act@5 Linear@6 —— 键 0/2/4/6 与 ckpt 对齐。
    activation 默认 ELU（conf.py: teacher_actor_activation='elu'）。
    """
    layers = []
    prev = in_dim
    for hdim in hidden:
        layers += [nn.Linear(prev, hdim), activation()]
        prev = hdim
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


# ============================================================================
# 无状态导出 wrapper —— forward 逐行复刻 act_student，loco LSTM 状态显式 I/O，
# nav 端口保留但不参与计算。
# ============================================================================
class LocoExportWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- vision_encoder 子模块（键名与 lbc_loco 的 VisionEncoder 对齐）----
        #   cnn               ← vision_encoder_state_dict.cnn.*
        #   rnn  (LSTM)       ← vision_encoder_state_dict.rnn.*
        #   rnn_output_layer  ← vision_encoder_state_dict.rnn_output_layer.*
        self.cnn = SimpleCNN()
        self.rnn = nn.LSTM(LOCO_LSTM_IN, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True)
        self.rnn_output_layer = nn.Linear(LSTM_HIDDEN, LATENT_DIM)

        # ---- teacher actor（loco_latent → joint）----
        self.teacher_actor = build_teacher_mlp(TEACHER_ACTOR_IN, NUM_ACTIONS)

    def forward(self, depth, proprio, goal,
                loco_h, loco_c, nav_h, nav_c, cmd_override):
        # ---- loco pipeline：cnn → cat(cnn_feat, proprio) → rnn → head → L2 ----
        # 与 VisionEncoder.forward / act_student 严格一致：LSTM 吃 cat(cnn_feat, proprio)。
        cnn_feat = self.cnn(depth)                                  # [B,32]
        rnn_in = torch.cat([cnn_feat, proprio], dim=-1).unsqueeze(1)  # [B,1,77]
        rnn_out, (loco_h_out, loco_c_out) = self.rnn(rnn_in, (loco_h, loco_c))
        hidden_t = rnn_out.squeeze(1)                               # [B,64]
        loco_latent = F.normalize(self.rnn_output_layer(hidden_t), p=2.0, dim=-1)  # [B,32]

        # ---- teacher_actor：cat[proprio, loco_latent] → joint ----
        # 注意：proprio 直接使用传入值（[6:9] 已由 C++ runner 覆写为速度命令），
        # loco 阶段不做 nav wrapper 里的 proprio[6:9]=cmd 再注入。
        actor_in = torch.cat([proprio, loco_latent, goal[:, :3]], dim=-1)  # [B,80]
        joint = self.teacher_actor(actor_in)                       # [B,12] 原始动作

        # ---- 保留端口（不参与 loco 计算）----
        # cmd / cmd_raw 回显输入的速度命令（cmd_override[:, :3]），使部署侧 State 的
        # cmd-override 隔离校验（期望 cmd==exec_cmd）通过；clearance 恒 0；
        # nav LSTM 状态原样透传（C++ runner 持有缓冲，但本图不更新）。
        cmd = cmd_override[:, 0:3]                                  # [B,3]
        cmd_raw = cmd
        clearance = torch.zeros(depth.shape[0], CLEAR_DIM,
                                dtype=joint.dtype, device=joint.device)  # [B,3]
        nav_h_out = nav_h
        nav_c_out = nav_c

        # ---- 防裁剪：让 loco 用不到的输入（goal）也参与到图里 ----
        # ONNX 导出时会做常量折叠 / 死代码消除，未被任何输出依赖的输入会被裁掉，
        # 导致 onnxruntime 报 "Invalid input name: goal"。这里用乘 0 的方式把
        # goal 接到 cmd/cmd_raw 上，数值恒为 0，不改变任何真实计算，仅保留输入端口。
        # （nav_h/nav_c 已作为输出被引用，端口天然会保留。）
        goal_zero = (goal.sum(dim=-1, keepdim=True) * 0.0)          # [B,1]
        cmd = cmd + goal_zero                                       # [B,3] + [B,1] 广播
        cmd_raw = cmd_raw + goal_zero

        return (cmd, cmd_raw, clearance, joint,
                loco_h_out, loco_c_out, nav_h_out, nav_c_out)


# ============================================================================
# 权重加载
# ============================================================================
def _sub_state_dict(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_weights(model: LocoExportWrapper, ckpt: dict) -> None:
    def need(key):
        if key not in ckpt:
            print(f"[ERR] ckpt 缺少键 '{key}'，现有顶层键: {list(ckpt.keys())}", file=sys.stderr)
            raise KeyError(key)
        return ckpt[key]

    ve = need("vision_encoder_state_dict")
    ta = need("teacher_actor_state_dict")

    # vision_encoder：按前缀拆进各子模块（前缀与 lbc_loco 的 VisionEncoder 属性一致）
    #   cnn. / rnn. / rnn_output_layer.
    cnn_state = _sub_state_dict(ve, "cnn.")
    rnn_state = _sub_state_dict(ve, "rnn.")
    head_state = _sub_state_dict(ve, "rnn_output_layer.")
    if not cnn_state or not rnn_state or not head_state:
        print("[ERR] vision_encoder_state_dict 缺少 cnn.*/rnn.*/rnn_output_layer.* 前缀键。\n"
              "      这看起来不是 lbc_loco 的 VisionEncoder ckpt（lbc_nav 的 DualPipeline\n"
              "      用 loco_cnn./loco_lstm./loco_head. 前缀，不能用本脚本导出）。",
              file=sys.stderr)
        raise KeyError("cnn./rnn./rnn_output_layer.")

    model.cnn.load_state_dict(cnn_state)
    model.rnn.load_state_dict(rnn_state)
    model.rnn_output_layer.load_state_dict(head_state)

    # teacher actor（键为 0/2/4/6）
    model.teacher_actor.load_state_dict(ta)

    print("[OK] 权重加载完成（loco CNN+LSTM+head、teacher_actor）。")


# ============================================================================
# 导出
# ============================================================================
def make_dummy(batch=1):
    g = torch.Generator().manual_seed(0)
    depth = torch.rand(batch, DEPTH_H, DEPTH_W, DEPTH_C, generator=g)          # 已归一化 [0,1]
    proprio = torch.randn(batch, PROPRIO_DIM, generator=g) * 0.1
    goal = torch.tensor([[0.0, 0.0, 2.0, 0.0]]).repeat(batch, 1)
    z = lambda: torch.zeros(LSTM_LAYERS, batch, LSTM_HIDDEN)
    return depth, proprio, goal, z(), z(), z(), z()


def export(model, out_path, opset=17):
    model.eval()
    depth, proprio, goal, lh, lc, nh, nc = make_dummy()

    input_names = ["depth", "proprio", "goal", "loco_h", "loco_c", "nav_h", "nav_c",
                   "cmd_override"]
    cmd_override = torch.zeros(1, 4)   # [vx,vy,wz,gate]
    args = (depth, proprio, goal, lh, lc, nh, nc, cmd_override)

    output_names = ["cmd", "cmd_raw", "clearance", "joint",
                    "loco_h_out", "loco_c_out", "nav_h_out", "nav_c_out"]

    torch.onnx.export(
        model, args, out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,            # 固定 batch=1，与 loco_runner.h 一致
    )
    print(f"[OK] 已导出 ONNX → {out_path}")
    print(f"     输入({len(input_names)}): {input_names}")
    print(f"     输出({len(output_names)}): {output_names}")


# ============================================================================
# Python ↔ ONNX 多帧数值对齐（带 loco LSTM 状态回喂 + proprio[6:9]=上一帧 cmd）
# ============================================================================
@torch.no_grad()
def verify(model, onnx_path, frames=8, tol=1e-4):
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"[ERR] 无法做数值对齐（缺 {e.name}）。", file=sys.stderr)
        return False

    model.eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # 两侧各自持有的 loco LSTM 状态 + prev_cmd，逐帧回喂（模拟 runner）。
    # nav 状态也维护一份（透传校验），但 loco 阶段恒等映射。
    def zeros(): return np.zeros((LSTM_LAYERS, 1, LSTM_HIDDEN), dtype=np.float32)
    pt = dict(lh=torch.zeros(LSTM_LAYERS, 1, LSTM_HIDDEN), lc=torch.zeros(LSTM_LAYERS, 1, LSTM_HIDDEN),
              nh=torch.zeros(LSTM_LAYERS, 1, LSTM_HIDDEN), nc=torch.zeros(LSTM_LAYERS, 1, LSTM_HIDDEN),
              prev_cmd=torch.zeros(1, NUM_CMD))
    ox = dict(lh=zeros(), lc=zeros(), nh=zeros(), nc=zeros(), prev_cmd=np.zeros((1, NUM_CMD), np.float32))

    rng = np.random.default_rng(42)
    names = ["cmd", "cmd_raw", "clearance", "joint"]
    max_diff = {k: 0.0 for k in names}
    max_rel = {k: 0.0 for k in names}

    for t in range(frames):
        depth_np = rng.random((1, DEPTH_H, DEPTH_W, DEPTH_C), dtype=np.float32)
        proprio_np = (rng.standard_normal((1, PROPRIO_DIM)) * 0.1).astype(np.float32)
        goal_np = np.array([[0.0, 0.0, 2.0, 0.0]], np.float32)

        # 契约①：proprio[6:9] = 上一帧 cmd（两侧用各自上一帧 cmd，理论上应一致）
        proprio_pt = proprio_np.copy(); proprio_pt[0, 6:9] = pt["prev_cmd"].numpy()[0]
        proprio_ox = proprio_np.copy(); proprio_ox[0, 6:9] = ox["prev_cmd"][0]

        # 后半段帧打开 override（gate=1），同时验证 cmd 回显路径。
        override_np = np.zeros((1, 4), np.float32)
        if t >= frames // 2:
            override_np[0] = [0.12, -0.03, 0.25, 1.0]

        # --- PyTorch ---
        out_pt = model(
            torch.from_numpy(depth_np), torch.from_numpy(proprio_pt), torch.from_numpy(goal_np),
            pt["lh"], pt["lc"], pt["nh"], pt["nc"],
            torch.from_numpy(override_np),
        )
        cmd_pt, cmd_raw_pt, clr_pt, joint_pt = (o.numpy() for o in out_pt[:4])
        pt["lh"], pt["lc"], pt["nh"], pt["nc"] = out_pt[4], out_pt[5], out_pt[6], out_pt[7]
        pt["prev_cmd"] = out_pt[0]

        # --- ONNX ---
        feeds = {"depth": depth_np, "proprio": proprio_ox, "goal": goal_np,
                 "loco_h": ox["lh"], "loco_c": ox["lc"], "nav_h": ox["nh"], "nav_c": ox["nc"],
                 "cmd_override": override_np}
        res = sess.run(None, feeds)
        cmd_ox, cmd_raw_ox, clr_ox, joint_ox = res[0], res[1], res[2], res[3]
        ox["lh"], ox["lc"], ox["nh"], ox["nc"] = res[4], res[5], res[6], res[7]
        ox["prev_cmd"] = res[0]

        for name, a, b in [("cmd", cmd_pt, cmd_ox), ("cmd_raw", cmd_raw_pt, cmd_raw_ox),
                           ("clearance", clr_pt, clr_ox), ("joint", joint_pt, joint_ox)]:
            abs_err = np.abs(a - b)
            rel_err = abs_err / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
            max_diff[name] = max(max_diff[name], float(abs_err.max()))
            max_rel[name] = max(max_rel[name], float(rel_err.max()))

    print("\n[VERIFY] 逐帧 rollout（含 loco LSTM 回喂 + proprio[6:9]=prev_cmd），各输出最大绝对误差:")
    ok = True
    for k, v in max_diff.items():
        passed = v < tol
        flag = "OK " if passed else "FAIL"
        if not passed:
            ok = False
        print(f"    [{flag}] {k:10s} max|Δ| = {v:.3e}  "
              f"max rel = {max_rel[k]:.3e}  (abs tol {tol:.0e})")
    print("[VERIFY] 通过 ✅" if ok else "[VERIFY] 不通过 ❌ —— 不要上机！")
    return ok


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="导出 Go2 lbc_loco 学生策略为 ONNX")
    ap.add_argument("--ckpt", required=True, help="lbc_loco 训练 ckpt（.pth），含 vision_encoder/teacher_actor state_dict，format='lbc_loco'")
    ap.add_argument("--out", required=True, help="输出 ONNX 路径，如 logs/loco/exported/policy.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-verify", action="store_true", help="跳过 Python↔ONNX 数值对齐")
    ap.add_argument("--frames", type=int, default=8, help="数值对齐 rollout 帧数")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[*] 加载 ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"ckpt 顶层不是 dict，而是 {type(ckpt)}")
    fmt = ckpt.get("format", "?")
    print(f"[*] ckpt format = {fmt}")
    if fmt != "lbc_loco":
        print(f"[ERR] 本脚本只导出 lbc_loco 阶段模型，但 ckpt format='{fmt}'。\n"
              f"      vision-nav 部署请用 export_vision_nav_onnx.py。",
              file=sys.stderr)
        sys.exit(1)

    model = LocoExportWrapper()
    load_weights(model, ckpt)

    export(model, args.out, opset=args.opset)

    if not args.no_verify:
        ok = verify(model, args.out, frames=args.frames)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
