#include "UwbGoal.h"

#include <array>
#include <cassert>
#include <cmath>

namespace
{
void expect_near(const std::array<float, 4>& actual, const std::array<float, 4>& expected)
{
    for (size_t i = 0; i < actual.size(); ++i)
        assert(std::fabs(actual[i] - expected[i]) < 1e-5f);
}
} // namespace

int main()
{
    // The threshold is exactly legacy XY scaling.
    expect_near(vision_nav::actor_goal_from_planar_xy(
                    6.0f, 8.0f, vision_nav::ActorGoalEncoding::DirectionPreservingV2),
                {0.6f, 0.8f, 0.5f, 0.0f});

    // Distant planar and polar paths retain their 3:4 bearing, not [1, 1].
    expect_near(vision_nav::actor_goal_from_planar_xy(
                    12.0f, 16.0f, vision_nav::ActorGoalEncoding::DirectionPreservingV2),
                {0.6f, 0.8f, 1.0f, 0.0f});
    expect_near(vision_nav::actor_goal_from_uwb(
                    std::atan2(4.0f, 3.0f), 0.0f, 20.0f,
                    vision_nav::ActorGoalEncoding::DirectionPreservingV2),
                {0.6f, 0.8f, 1.0f, 0.0f});

    // A distant goal below the distance-channel saturation still keeps its range.
    expect_near(vision_nav::actor_goal_from_planar_xy(
                    9.0f, 12.0f, vision_nav::ActorGoalEncoding::DirectionPreservingV2),
                {0.6f, 0.8f, 0.75f, 0.0f});
    return 0;
}
