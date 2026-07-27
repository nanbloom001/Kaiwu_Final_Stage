#!/usr/bin/env python3
"""Nav workflow platform-lifecycle publication regressions."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import agent_ppo.tests._nav_test_stubs  # noqa: F401

from agent_ppo.feature import nav_contract
from agent_ppo.checkpoint_io import CheckpointSaveError
from agent_ppo.workflow import nav_dagger_workflow as workflow_module


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class _Monitor:
    def __init__(self):
        self.rows = []

    def put_data(self, row):
        self.rows.append(row)


class _FakeEnv:
    def __init__(self, num_envs=2, fail_step=None):
        self.num_envs = num_envs
        self.fail_step = fail_step
        self.step_calls = 0

    def _obs(self):
        return torch.zeros(self.num_envs, nav_contract.POLICY_OBS_DIM)

    def _critic_obs(self):
        return torch.zeros(self.num_envs, nav_contract.CRITIC_OBS_DIM)

    def reset(self, _usr_conf):
        return self._obs(), self._critic_obs()

    def step(self, _actions):
        self.step_calls += 1
        if self.step_calls == self.fail_step:
            raise RuntimeError("synthetic env.step failure")
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        infos = {"time_outs": torch.zeros_like(done)}
        return (
            self.step_calls,
            self._obs(),
            torch.zeros(self.num_envs),
            done,
            done,
            (infos, self._critic_obs()),
        )


class _FakeAlgorithm:
    def __init__(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.SGD([parameter], lr=3e-4)
        self.low_level_state_digest = "digest"
        self.resume_loaded = False
        self.lr_scheduler_state = {}
        self.ramp_start_h = 0.5
        self.ramp_end_h = 4.0
        self.ramp_clock_h = 0.0
        self.ramp_probability = 0.0
        self.soft_stay_frozen = False
        self.soft_stay_reason = ""
        self.training_status = "running"
        self.current_iteration = 0
        self.nonfinite_fallback_count = 0
        self.total_nav_ticks = 0
        self.frame_begin_calls = 0
        self.frame_end_calls = 0
        self.finish_update_calls = 0
        self.tick_count = 0

    def train_mode(self):
        return None

    def frame_begin(self, obs, _critic_obs):
        self.frame_begin_calls += 1
        is_tick = self.frame_begin_calls % nav_contract.NAV_PERIOD_FRAMES == 0
        metrics = {}
        if is_tick:
            self.tick_count += 1
            self.total_nav_ticks += obs.shape[0]
            metrics = {
                "buffer_full": self.tick_count % nav_contract.TBPTT_T == 0,
                "ce_loss": 1.0,
                "top1_accuracy": 0.5,
                "disagreement_rate": 0.5,
                "token_entropy": 1.0,
                "switch_rate": 0.0,
                "oracle_valid_count": float(obs.shape[0]),
                "oracle_sample_count": float(obs.shape[0]),
                "goal4_fresh_count": float(obs.shape[0]),
                "goal4_sample_count": float(obs.shape[0]),
            }
        return {
            "actions": torch.zeros(obs.shape[0], 12),
            "is_tick": is_tick,
            "tick_metrics": metrics,
        }

    def frame_end(self, _dones):
        self.frame_end_calls += 1

    def finish_nav_sequence_update(self):
        self.finish_update_calls += 1
        self.optimizer.step()
        return {
            "valid_ticks": float(nav_contract.TBPTT_T),
            "update_skipped_no_valid": 0.0,
            "grad_norm": 0.5,
        }

    def assert_high_level_parameters_finite(self):
        return None


class _FakeAgent:
    def __init__(
        self,
        max_iterations=1,
        fail_learn_attempt=None,
        fail_checkpoint_attempt=None,
        fail_save=False,
    ):
        self.stage = SimpleNamespace(
            name="nav_dagger",
            max_iterations=max_iterations,
            log_interval=1,
            num_steps_per_env=(
                nav_contract.TBPTT_T * nav_contract.NAV_PERIOD_FRAMES
            ),
            tbptt_sequence_length=nav_contract.TBPTT_T,
            lr=3e-4,
            lr_min=1e-5,
            lr_scheduler_iterations=100,
        )
        self.algorithm = _FakeAlgorithm()
        self.device = "cpu"
        self.is_nav_dagger = True
        self._process_role = "test"
        self._nav_training_started = False
        self.fail_learn_attempt = fail_learn_attempt
        self.fail_checkpoint_attempt = fail_checkpoint_attempt
        self.learn_attempts = 0
        self.learn_successes = 0
        self.save_calls = 0
        self.fail_save = fail_save

    def learn(self, list_sample_data=None):
        self.learn_attempts += 1
        if self.learn_attempts == self.fail_checkpoint_attempt:
            raise CheckpointSaveError("synthetic checkpoint write failure")
        if self.learn_attempts == self.fail_learn_attempt:
            raise RuntimeError("synthetic lifecycle failure")
        if list_sample_data is not None:
            raise AssertionError("Nav lifecycle callback must receive None")
        self.learn_successes += 1

    def save_model(self):
        if self.fail_save:
            raise RuntimeError("synthetic save failure")
        self.save_calls += 1


def _nav_conf(max_iterations):
    return {
        "nav_dagger": {
            "max_iterations": max_iterations,
            "log_interval": 1,
            "num_steps_per_env": (
                nav_contract.TBPTT_T * nav_contract.NAV_PERIOD_FRAMES
            ),
            "tbptt_sequence_length": nav_contract.TBPTT_T,
            "nav_period_frames": nav_contract.NAV_PERIOD_FRAMES,
            "min_dwell_ticks": nav_contract.MIN_DWELL_TICKS,
            "learning_rate": 3e-4,
            "lr_min": 1e-5,
            "lr_scheduler_iterations": 100,
            "ramp_start_h": 0.5,
            "ramp_end_h": 4.0,
        }
    }


class TestNavLifecyclePublication(unittest.TestCase):
    def _run(
        self,
        *,
        max_iterations=1,
        fail_step=None,
        fail_learn_attempt=None,
        fail_checkpoint_attempt=None,
        dump_model_freq=3600,
    ):
        agent = _FakeAgent(
            max_iterations,
            fail_learn_attempt,
            fail_checkpoint_attempt,
        )
        env = _FakeEnv(fail_step=fail_step)
        logger = _Logger()
        monitor = _Monitor()
        with mock.patch.object(
            workflow_module.Config,
            "load_conf",
            return_value=(
                _nav_conf(max_iterations),
                "fake.toml",
                False,
                agent.stage,
            ),
        ), mock.patch.object(
            workflow_module,
            "_platform_dump_model_freq",
            return_value=dump_model_freq,
        ):
            workflow_module._workflow_impl(
                [env], [agent], logger=logger, monitor=monitor
            )
        return agent, env, logger, monitor

    def test_every_successful_frame_advances_lifecycle_without_outer_extra(self):
        agent, env, _logger, monitor = self._run(max_iterations=2)
        frames = 2 * nav_contract.TBPTT_T * nav_contract.NAV_PERIOD_FRAMES
        self.assertEqual(env.step_calls, frames)
        self.assertEqual(agent.algorithm.frame_begin_calls, frames)
        self.assertEqual(agent.algorithm.frame_end_calls, frames)
        self.assertEqual(agent.learn_attempts, frames)
        self.assertEqual(agent.learn_successes, frames)
        self.assertEqual(agent.algorithm.finish_update_calls, 2)
        self.assertEqual(agent.algorithm.current_iteration, 2)
        self.assertEqual(agent.save_calls, 1)
        self.assertTrue(agent._nav_final_save_done)
        self.assertEqual(agent._nav_final_save_reason, "normal_completion")
        self.assertEqual(len(monitor.rows), 2)
        final_metrics = next(iter(monitor.rows[-1].values()))
        self.assertEqual(final_metrics["platform_lifecycle_callbacks"], frames)
        self.assertEqual(final_metrics["total_low_level_steps"], frames)
        self.assertEqual(final_metrics["total_env_frames"], frames * env.num_envs)

    def test_failed_env_step_does_not_advance_failed_frame(self):
        agent = _FakeAgent()
        env = _FakeEnv(fail_step=50)
        with mock.patch.object(
            workflow_module.Config,
            "load_conf",
            return_value=(_nav_conf(1), "fake.toml", False, agent.stage),
        ), mock.patch.object(
            workflow_module, "_platform_dump_model_freq", return_value=3600
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic env.step failure"):
                workflow_module._workflow_impl(
                    [env], [agent], logger=_Logger(), monitor=_Monitor()
                )
        self.assertEqual(env.step_calls, 50)
        self.assertEqual(agent.algorithm.frame_begin_calls, 50)
        self.assertEqual(agent.algorithm.frame_end_calls, 49)
        self.assertEqual(agent.learn_attempts, 49)
        self.assertEqual(agent.algorithm.finish_update_calls, 0)
        self.assertEqual(agent.algorithm.current_iteration, 0)
        self.assertEqual(agent.save_calls, 0)

    def test_lifecycle_failure_is_counted_and_training_continues(self):
        agent, env, logger, monitor = self._run(fail_learn_attempt=2)
        frames = nav_contract.TBPTT_T * nav_contract.NAV_PERIOD_FRAMES
        self.assertEqual(env.step_calls, frames)
        self.assertEqual(agent.learn_attempts, frames)
        self.assertEqual(agent.learn_successes, frames - 1)
        self.assertEqual(agent.algorithm.finish_update_calls, 1)
        metrics = next(iter(monitor.rows[-1].values()))
        self.assertEqual(metrics["platform_lifecycle_callbacks"], frames - 1)
        self.assertEqual(metrics["platform_lifecycle_failures"], 1)
        self.assertTrue(
            any(
                level == "error" and "platform lifecycle callback failed" in message
                for level, message in logger.messages
            )
        )

    def test_checkpoint_failure_inside_lifecycle_stops_training(self):
        agent = _FakeAgent(fail_checkpoint_attempt=2)
        env = _FakeEnv()
        logger = _Logger()
        with mock.patch.object(
            workflow_module.Config,
            "load_conf",
            return_value=(_nav_conf(1), "fake.toml", False, agent.stage),
        ), mock.patch.object(
            workflow_module, "_platform_dump_model_freq", return_value=3600
        ):
            with self.assertRaisesRegex(
                CheckpointSaveError, "synthetic checkpoint write failure"
            ):
                workflow_module._workflow_impl(
                    [env], [agent], logger=logger, monitor=_Monitor()
                )

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(agent.learn_attempts, 2)
        self.assertEqual(agent.learn_successes, 1)
        self.assertEqual(agent._nav_lifecycle_failure_callbacks, 0)
        self.assertTrue(
            any(
                level == "error" and "stopping training" in message
                for level, message in logger.messages
            )
        )

    def test_dump_boundary_uses_successful_callback_count(self):
        agent, _env, logger, monitor = self._run(dump_model_freq=160)
        self.assertEqual(agent.learn_successes, 160)
        metrics = next(iter(monitor.rows[-1].values()))
        self.assertEqual(metrics["callbacks_until_next_dump"], 0)
        self.assertTrue(
            any(
                level == "info"
                and "platform lifecycle dump boundary reached" in message
                for level, message in logger.messages
            )
        )

    def test_callbacks_until_next_dump_uses_3600_boundaries(self):
        self.assertEqual(workflow_module._callbacks_until_next_dump(3599, 3600), 1)
        self.assertEqual(workflow_module._callbacks_until_next_dump(3600, 3600), 0)
        self.assertEqual(workflow_module._callbacks_until_next_dump(3601, 3600), 3599)
        self.assertEqual(workflow_module._callbacks_until_next_dump(7200, 3600), 0)

    def test_final_checkpoint_is_idempotent_and_records_reason(self):
        agent = _FakeAgent()
        logger = _Logger()
        agent._nav_training_started = True
        agent._nav_final_save_done = False
        agent._nav_final_save_reason = None
        self.assertTrue(
            workflow_module._save_final_checkpoint(
                agent, logger, reason="graceful_platform_exit"
            )
        )
        self.assertFalse(
            workflow_module._save_final_checkpoint(
                agent, logger, reason="duplicate_exit"
            )
        )
        self.assertEqual(agent.save_calls, 1)
        self.assertTrue(agent._nav_final_save_done)
        self.assertEqual(agent._nav_final_save_reason, "graceful_platform_exit")

    def test_final_checkpoint_skips_before_training_started(self):
        agent = _FakeAgent()
        logger = _Logger()
        agent._nav_training_started = False
        agent._nav_final_save_done = False
        self.assertFalse(
            workflow_module._save_final_checkpoint(
                agent, logger, reason="graceful_platform_exit"
            )
        )
        self.assertEqual(agent.save_calls, 0)

    def test_final_checkpoint_failure_is_logged_without_marking_done(self):
        agent = _FakeAgent(fail_save=True)
        logger = _Logger()
        agent._nav_training_started = True
        agent._nav_final_save_done = False
        self.assertFalse(
            workflow_module._save_final_checkpoint(
                agent, logger, reason="graceful_platform_exit"
            )
        )
        self.assertFalse(agent._nav_final_save_done)
        self.assertTrue(
            any(
                level == "error" and "final checkpoint save failed" in message
                for level, message in logger.messages
            )
        )

    def test_wrapper_system_exit_triggers_one_final_save(self):
        agent = _FakeAgent()
        env = _FakeEnv()
        logger = _Logger()

        def _raise_system_exit(_envs, agents, logger=None, monitor=None):
            agents[0]._nav_training_started = True
            raise SystemExit("synthetic stop")

        with mock.patch.object(
            workflow_module, "_workflow_impl", side_effect=_raise_system_exit
        ), mock.patch.object(
            workflow_module, "_install_sigterm_checkpoint_handler", return_value=None
        ):
            with self.assertRaisesRegex(SystemExit, "synthetic stop"):
                workflow_module.workflow([env], [agent], logger=logger)

        self.assertEqual(agent.save_calls, 1)
        self.assertTrue(agent._nav_final_save_done)
        self.assertEqual(agent._nav_final_save_reason, "graceful_platform_exit")


if __name__ == "__main__":
    unittest.main()
