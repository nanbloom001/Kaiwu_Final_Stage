# Historical P3 fixed log findings

Input: `deploy/sim2real_test_p3_standard_fixed/logs/20260801_010805_loco_fixed_vx0_vy0_wz0/visloco_diag_1785517828.csv`.

This is a 121-frame, roughly 2.4-second historical fixed-zero capture. It is
useful for validating legacy CSV compatibility, but is too short and predates
the strict action/freshness layers, so it is not a successful baseline or a
hardware acceptance result.

The strict analyzer reported:

- inference mean `4.875 ms`, p95 `6.730 ms`, max `10.005 ms`;
- maximum measured tracking error `0.2635 rad`;
- measured joint velocity peaks up to `5.5706 rad/s` and effort peaks up to
  `6.4484 Nm`;
- raw-action high-frequency concentration was worst on RR, at `1.486x` the
  median leg energy;
- applied-target high-frequency concentration was also worst on RR, at
  `2.218x` the median leg energy, with left/right asymmetry `0.318`;
- applied-target steps repeatedly reached `0.06 rad`, the historical limiter.

These numbers justify keeping raw action and applied target separate and adding
per-leg diagnostics. They do not establish that RR is the root cause: the log
duration is short, contains no strict sensor/depth freshness fields, and has no
matched stable-route comparison. Recheck the same metrics on at least 60 seconds
of no-command strict shadow data before any powered interpretation.
