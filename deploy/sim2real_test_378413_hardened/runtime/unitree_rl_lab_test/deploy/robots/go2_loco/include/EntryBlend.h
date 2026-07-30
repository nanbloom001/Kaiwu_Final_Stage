#pragma once

#include <algorithm>

namespace loco_safety
{
inline float entry_blend_alpha(float elapsed_s, float duration_s)
{
    if (duration_s <= 0.0f) return 1.0f;
    const float x = std::clamp(elapsed_s / duration_s, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

inline float entry_blend_target(
    float initial_target, float policy_target, float elapsed_s, float duration_s)
{
    const float alpha = entry_blend_alpha(elapsed_s, duration_s);
    return initial_target + alpha * (policy_target - initial_target);
}

inline float policy_transition_target(
    float entry_target, float policy_target, float elapsed_s,
    float duration_s, bool bypass_entry_blend)
{
    return bypass_entry_blend
        ? policy_target
        : entry_blend_target(entry_target, policy_target, elapsed_s, duration_s);
}

} // namespace loco_safety
