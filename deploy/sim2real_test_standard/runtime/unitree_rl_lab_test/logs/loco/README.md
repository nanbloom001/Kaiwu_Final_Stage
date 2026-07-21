# loco 阶段部署落点

本目录是 lbc_loco 阶段部署树（go2_loco）的策略目录，对应
`deploy/robots/go2_loco/config/config.yaml` 里 `FSM.VisionLoco.policy_dir: ../../../logs/loco`。

约定结构：

```
logs/loco/
├── exported/policy.onnx   ← export_loco_onnx.py 产出（8 入 8 出，nav 端口占位）
├── params/deploy.yaml     ← 与 lbc_loco ckpt 同源的部署契约
└── logs/                  ← 运行期逐帧诊断 CSV（visloco_diag_*.csv）
```

导出命令：

```bash
python export_loco_onnx.py \
  --ckpt <lbc_loco ckpt .pth，format='lbc_loco'> \
  --out  runtime/unitree_rl_lab_test/logs/loco/exported/policy.onnx
```

注意：`deploy.yaml` 必须与 lbc_loco ckpt 同源（joint_ids_map / default_joint_pos /
KP/KD / proprio_scales / action.scale），不可与 vision-nav 的 deploy.yaml 混用。
