#include "UwbGoal.h"

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace
{
constexpr float PI = 3.14159265358979323846f;

void expect_near(const std::array<float, 4>& actual, const std::array<float, 4>& expected)
{
    for (size_t i = 0; i < actual.size(); ++i)
        assert(std::fabs(actual[i] - expected[i]) < 1e-5f);
}
} // namespace

int main()
{
    assert(std::fabs(vision_nav::planar_distance_from_uwb(PI / 3.0f, 2.0f) - 1.0f) < 1e-5f);
    assert(std::fabs(vision_nav::planar_distance_from_uwb(PI / 2.0f, 2.0f)) < 1e-5f);
    assert(vision_nav::planar_distance_from_uwb(0.0f, -1.0f) == 0.0f);
    assert(vision_nav::planar_distance_from_uwb(
               std::numeric_limits<float>::quiet_NaN(), 1.0f) == 0.0f);

    bool arrived = false;
    arrived = vision_nav::uwb_arrived_with_hysteresis(arrived, 0.34f, 0.35f, 0.15f);
    assert(arrived);
    arrived = vision_nav::uwb_arrived_with_hysteresis(arrived, 0.45f, 0.35f, 0.15f);
    assert(arrived);
    arrived = vision_nav::uwb_arrived_with_hysteresis(arrived, 0.50f, 0.35f, 0.15f);
    assert(!arrived);

    assert(vision_nav::approach_speed_scale(2.0f, 0.35f, 1.0f) == 1.0f);
    assert(vision_nav::approach_speed_scale(1.0f, 0.35f, 1.0f) == 1.0f);
    assert(std::fabs(vision_nav::approach_speed_scale(0.675f, 0.35f, 1.0f) - 0.5f) < 1e-5f);
    assert(vision_nav::approach_speed_scale(0.35f, 0.35f, 1.0f) == 0.0f);
    assert(vision_nav::approach_speed_scale(0.2f, 0.35f, 1.0f) == 0.0f);

    expect_near(vision_nav::actor_goal_from_uwb(0.0f, 0.0f, 1.0f),
                {0.1f, 0.0f, 0.05f, 0.0f});
    expect_near(vision_nav::actor_goal_from_uwb(PI / 2.0f, 0.0f, 2.0f),
                {0.0f, 0.2f, 0.1f, 0.0f});
    expect_near(vision_nav::actor_goal_from_uwb(0.0f, PI / 3.0f, 2.0f),
                {0.1f, 0.0f, 0.05f, 0.0f});
    expect_near(vision_nav::actor_goal_from_uwb(0.0f, 0.0f, 50.0f),
                {1.0f, 0.0f, 1.0f, 0.0f});
    expect_near(vision_nav::actor_goal_from_uwb(0.0f, 0.0f, -1.0f),
                {0.0f, 0.0f, 0.0f, 0.0f});
    expect_near(vision_nav::actor_goal_from_uwb(
                    std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f),
                {0.0f, 0.0f, 0.0f, 0.0f});

    expect_near(vision_nav::actor_goal_from_planar_xy(1.0f, -2.0f),
                {0.1f, -0.2f, std::sqrt(5.0f) / 20.0f, 0.0f});
    assert(std::fabs(vision_nav::time_filter_alpha(0.25f, 0.25f) -
                     (1.0f - std::exp(-1.0f))) < 1e-5f);
    assert(vision_nav::time_filter_alpha(0.0f, 0.25f) == 0.0f);
    assert(vision_nav::time_filter_alpha(0.25f, 0.0f) == 1.0f);

    assert(vision_nav::uwb_freshness_scale(0.0f, 0.5f, 1.5f) == 1.0f);
    assert(vision_nav::uwb_freshness_scale(0.5f, 0.5f, 1.5f) == 1.0f);
    assert(std::fabs(vision_nav::uwb_freshness_scale(1.0f, 0.5f, 1.5f) - 0.5f) < 1e-5f);
    assert(vision_nav::uwb_freshness_scale(1.5f, 0.5f, 1.5f) == 0.0f);
    assert(vision_nav::uwb_freshness_scale(2.0f, 0.5f, 1.5f) == 0.0f);
    return 0;
}
