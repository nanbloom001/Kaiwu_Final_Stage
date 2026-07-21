#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
// 注：不引入 State_RLBase.h（其构造函数定义在 go2/src/State_RLBase.cpp）。
// loco 部署树只用 Passive / FixStand / VisionLoco 三态，config.yaml 已禁用 Velocity(RLBase)。
#include "State_VisionLoco.h"

#include <cstdlib>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    // Go2 boots into its default high-level motion mode, which publishes lowcmd.
    // Release it before checking for any remaining external low-level controller.
    unitree::robot::go2::shutdown();
    usleep(0.2 * 1e6);

    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        std::exit(EXIT_FAILURE);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Loco Controller \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "And then press [Start] to start controlling the robot.\n";

    while (true)
        sleep(1);

    return 0;
}
