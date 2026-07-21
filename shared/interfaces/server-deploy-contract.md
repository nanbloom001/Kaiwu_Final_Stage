# server ↔ deploy 接口契约（stub）

> ⚠️ 本文件为 stub，待人工填充。固化 `server/` 训练侧与 `deploy/` 部署侧的接口契约，确保两侧不漂移。

## 待固化条目

按迁移 Plan 第三节，本契约需记录：

- [ ] **checkpoint 来源**：训练分支、提交 SHA、`id_list`、`kaiwu.json` project_code
- [ ] **checkpoint SHA256**：如 `model.ckpt-vision-378413.pkl` = `37429c1e2c1d263844a74ecc2fdb97201ae296663f5ad8890ce173d5525a1288`
- [ ] **Actor 输入维度与 goal 编码**：proprio(45) + latent(32) + goal(3) = 80；goal 三维编码与 clipping
- [ ] **depth 尺寸/归一化/相机内外参**：D435i，180×320×1，内外参与标定来源
- [ ] **字段顺序**：proprio / command / goal 的索引与语义
- [ ] **ONNX I/O**：输入输出名称、shape、LSTM 状态初始化与 reset 语义
- [ ] **`deploy.yaml` 与训练配置的对应关系**：字段映射

## 当前已知

- 378413 checkpoint 在 `deploy/sim2real_test_loco/models/` 与 `deploy/sim2real_test_st9/models/` 各一份，SHA256 完全一致（Git 自动复用 blob）。
- ST7、standard 部署树**缺失 checkpoint/ONNX**，需人工提供实际 Jetson 路径或制品哈希。
- 本轮采用"源码自洽 + 外部制品清单"标准：不把 Jetson 二进制、ONNX Runtime、全部 ONNX 提交进 Git（待人工确认后执行）。
