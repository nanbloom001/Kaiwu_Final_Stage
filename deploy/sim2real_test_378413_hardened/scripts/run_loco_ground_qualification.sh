#!/usr/bin/env bash
# Ground qualification is intentionally disabled after the rejected 2026-07-30 run.
set -euo pipefail

echo "ERROR: 378413 hardened is blocked from further ground qualification." >&2
echo "The 20260730_153540 run pitched forward and fell after the policy output exceeded its safety bound." >&2
echo "Do not bypass this block, increase speed/KP, or relax any protection limit." >&2
exit 1
