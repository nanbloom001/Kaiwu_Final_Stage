# P3 fixed policy directory

`exported/policy.onnx` contains the P3 standard low-level policy. It excludes
the checkpoint's high-level track navigation modules. `params/deploy.yaml`
defines the Go2 joint mapping, gains, observation scales, and action transform.

Regenerate the model from the package root with `export_p3_standard_onnx.py`.
The output contract is consumed by `deploy/include/isaaclab/algorithms/loco_runner.h`.
