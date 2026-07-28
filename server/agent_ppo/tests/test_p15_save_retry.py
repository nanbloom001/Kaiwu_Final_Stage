#!/usr/bin/env python3

import sys
import types
import signal

import pytest


tools_utils = types.ModuleType("tools.utils")
tools_utils.load_reward_keys_from_monitor_config = lambda: []
sys.modules.setdefault("tools.utils", tools_utils)

from agent_ppo.workflow.visual_ppo_workflow import (
    _attempt_scheduled_checkpoint,
    _install_sigterm_checkpoint_handler,
    _restore_sigterm_handler,
    _save_final_checkpoint,
    _save_iteration_cap_checkpoint,
)


class _Logger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.infos = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)


class _Agent:
    def __init__(self, error=None, errors=None):
        self.error = error
        self.errors = list(errors or [])
        self.calls = 0
        self._visual_ppo_final_save_done = False
        self._visual_ppo_final_save_reason = None
        self._visual_ppo_training_started = True
        self.is_visual_ppo = True
        self.is_p15_response = True

    def save_model(self):
        self.calls += 1
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        if self.error is not None:
            raise self.error


def test_p15_save_failure_is_reported_for_retry_without_stopping():
    logger = _Logger()
    agent = _Agent(OSError("disk busy"))
    assert not _attempt_scheduled_checkpoint(agent, logger, p15_schedule=True)
    assert agent.calls == 1
    assert "retry is scheduled in 60s" in logger.errors[0]


def test_legacy_save_failure_remains_strict():
    with pytest.raises(OSError, match="disk busy"):
        _attempt_scheduled_checkpoint(
            _Agent(OSError("disk busy")), _Logger(), p15_schedule=False
        )


def test_p15_iteration_cap_final_save_retries_once_and_marks_success():
    logger = _Logger()
    agent = _Agent(errors=[OSError("disk busy"), None])
    sleeps = []
    assert _save_iteration_cap_checkpoint(
        agent,
        logger,
        p15_schedule=True,
        sleep_fn=sleeps.append,
    )
    assert agent.calls == 2
    assert sleeps == [60.0]
    assert agent._visual_ppo_final_save_done is True
    assert agent._visual_ppo_final_save_reason == "iteration_cap"


def test_p15_iteration_cap_final_save_keeps_outcome_after_retry_failure():
    logger = _Logger()
    agent = _Agent(errors=[OSError("disk busy"), OSError("still busy")])
    sleeps = []
    assert not _save_iteration_cap_checkpoint(
        agent,
        logger,
        p15_schedule=True,
        sleep_fn=sleeps.append,
    )
    assert agent.calls == 2
    assert sleeps == [60.0]
    assert agent._visual_ppo_final_save_done is False
    assert "failed after the 60s retry" in logger.errors[-1]


def test_p15_graceful_final_save_retries_once_without_hiding_exit():
    logger = _Logger()
    agent = _Agent(errors=[OSError("disk busy"), None])
    sleeps = []
    assert _save_final_checkpoint(
        agent,
        logger,
        reason="test_sigterm",
        sleep_fn=sleeps.append,
    )
    assert agent.calls == 2
    assert sleeps == [5.0]
    assert agent._visual_ppo_final_save_done is True
    assert agent._visual_ppo_final_save_reason == "test_sigterm"


def test_sigterm_wrapper_chains_existing_handler_then_enters_graceful_path():
    logger = _Logger()
    called = []

    def previous_handler(signum, _frame):
        called.append(signum)

    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, previous_handler)
    previous = None
    try:
        previous = _install_sigterm_checkpoint_handler(logger)
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit, match="SIGTERM"):
            handler(signal.SIGTERM, None)
        assert called == [signal.SIGTERM]
        assert previous is previous_handler
    finally:
        if previous is not None:
            _restore_sigterm_handler(previous)
        signal.signal(signal.SIGTERM, original)
