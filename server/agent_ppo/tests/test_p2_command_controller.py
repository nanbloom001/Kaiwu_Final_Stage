import torch

from agent_ppo.feature.p2_command_controller import P2CommandController


def test_instant_hold_syncs_target_and_exec_and_ignores_steps():
    command = P2CommandController(
        2,
        "cpu",
        command_transition_mode="instant_hold_10hz",
    )
    target = torch.tensor(((0.8, -0.2, 0.6), (-0.5, 0.3, -1.0)))

    command.set_target(target)

    assert torch.equal(command.active_target, target)
    assert torch.equal(command.exec_cmd, target)
    for _ in range(3):
        command.step()
    assert torch.equal(command.exec_cmd, target)
    assert command.state_dict()["command_transition_mode"] == "instant_hold_10hz"
    assert command.state_dict()["hold_frames"] == 5


def test_instant_hold_reversal_is_immediate_and_reset_zeros_selected_envs():
    command = P2CommandController(
        2,
        "cpu",
        command_transition_mode="instant_hold_10hz",
    )
    command.set_target(torch.tensor(((0.4, 0.2, 0.8), (0.1, -0.3, -0.6))))
    command.set_target(torch.tensor(((-0.4, -0.2, -0.8), (0.1, -0.3, -0.6))))

    assert torch.equal(
        command.exec_cmd,
        torch.tensor(((-0.4, -0.2, -0.8), (0.1, -0.3, -0.6))),
    )
    command.reset(torch.tensor([0]))
    assert torch.equal(command.active_target[0], torch.zeros(3))
    assert torch.equal(command.exec_cmd[0], torch.zeros(3))
    assert torch.equal(command.exec_cmd[1], torch.tensor((0.1, -0.3, -0.6)))


def test_slew_mode_remains_the_default_transition():
    command = P2CommandController(1, "cpu", slew_rate=(0.30, 0.30, 1.00))
    command.set_target(torch.tensor(((1.0, 0.0, -1.0),)))

    assert torch.equal(command.exec_cmd, torch.zeros(1, 3))
    command.step()
    assert torch.allclose(
        command.exec_cmd,
        torch.tensor(((0.006, 0.0, -0.02),)),
        atol=1.0e-7,
    )


def test_rejects_unknown_command_transition_mode():
    import pytest

    with pytest.raises(ValueError, match="Unsupported P2 command transition mode"):
        P2CommandController(1, "cpu", command_transition_mode="delayed")
