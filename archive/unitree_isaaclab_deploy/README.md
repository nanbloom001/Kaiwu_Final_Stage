# unitree_isaaclab_deploy (archived)

旧官方 Unitree Isaac Lab 部署包 / 本地修改快照。**非活动部署目录**，不能被活动训练/部署入口引用。活动部署在 `../../deploy/`。

## LFS pointer 说明

本目录从主仓未跟踪副本纳入时，检测到若干 git-lfs **pointer 文件**（133 字节占位符，真实 LFS 对象不在本地副本）。按迁移 Plan "133 字节旧 LFS pointer 不得登记成可用 checkpoint"，这些 pointer 文件已通过 `.gitignore` 排除，不提交。排除的包括：ONNX Runtime `.so` 库、`model.ckpt-lbc-loco-637427.pkl` checkpoint、mimic `.bvh_60hz.csv` 动作数据、`camera_calibration/*.npy` 与标定 PDF。

如需真实对象，在源仓执行 `git lfs pull` 后重新纳入。完整排除清单见 `.gitignore`。

## 保留内容
真实源码、配置、标定脚本、ROS 节点等（非 pointer 文件）原样保留，作旧部署包参考。
