# Baseline audit

The source baseline is `deploy/sim2real_test_loco`. It remains the repository's
stable deployment entry and was not modified by this work. This strict package
was copied into a new directory so experimental P3 behavior cannot silently
change the baseline route.

Material differences in the strict package:

- binds checkpoint, ONNX, ABI, controller config, and deploy config by SHA256;
- extracts the P3 Standard low-level modules from the named training reference;
- uses the training-effective raw action clip `[-6,6]`;
- separates every action/target safety layer and feeds back executed action;
- enforces immutable LowState snapshots and tick/timestamp freshness;
- rejects RealSense profile fallback and records device/profile/intrinsics data;
- adds no-command sensor-only and shadow modes;
- gates powered policy publication on stable sensors plus two seconds of shadow;
- adds per-joint/per-leg frequency and asymmetry diagnostics;
- adds fault-injection, concurrency, parity, and TSan regression tests.

No successful `sim2real_test_loco` diagnostic CSV was present in the baseline
tree at audit time, so numerical baseline-vs-candidate motion comparison remains
pending. The analyzer supports a future baseline CSV explicitly; absence of that
evidence is not treated as success.

A short historical P3 fixed-zero capture was analyzed only to verify legacy
field support; see `HISTORICAL_LOG_FINDINGS.md` for its scoped observations.
