#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""平台专属模块的本地测试 stub（tools / toml / common_python）。

平台镜像里存在这些模块；本地开发环境没有。仅在真实模块缺失时安装最小
stub——平台上运行不受影响。unittest 与 pytest 都可用：测试文件把
``import agent_ppo.tests._nav_test_stubs`` 放在其他 agent_ppo import 之前；
conftest.py（pytest）亦引用本模块。
"""

import sys
import types


def _install_tools_stub():
    try:
        import tools.base_env.observation_process  # noqa: F401
        return
    except ImportError:
        pass

    import torch

    class ObservationProcess:  # 最小 stub：Nav process 测试注入 default obs
        target_group = "policy"

        def __init__(self, env=None):
            self.env = env

        def default_observation(self):
            return self._default_obs

        def concatenate_terms(self, *terms):
            return torch.cat(terms, dim=-1)

    class RewardProcessBase:
        def __init__(self, *args, **kwargs):
            pass

    tools_mod = types.ModuleType("tools")
    base_env_mod = types.ModuleType("tools.base_env")
    obs_mod = types.ModuleType("tools.base_env.observation_process")
    reward_mod = types.ModuleType("tools.base_env.base_reward")
    obs_mod.ObservationProcess = ObservationProcess
    reward_mod.RewardProcessBase = RewardProcessBase
    base_env_mod.observation_process = obs_mod
    base_env_mod.base_reward = reward_mod
    tools_mod.base_env = base_env_mod
    sys.modules.setdefault("tools", tools_mod)
    sys.modules.setdefault("tools.base_env", base_env_mod)
    sys.modules.setdefault("tools.base_env.observation_process", obs_mod)
    sys.modules.setdefault("tools.base_env.base_reward", reward_mod)


def _install_toml_stub():
    if "toml" in sys.modules:
        return
    try:
        import toml  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import tomllib
    except ImportError:
        return

    toml_stub = types.ModuleType("toml")

    def _load(path):
        with open(path, "rb") as stream:
            return tomllib.load(stream)

    toml_stub.load = _load
    sys.modules["toml"] = toml_stub


def _install_common_python_stub():
    try:
        from common_python.utils.common_func import create_cls  # noqa: F401
        return
    except ImportError:
        pass

    def create_cls(cls_name, **fields):
        def __init__(self, **kwargs):
            for key, default in fields.items():
                setattr(self, key, kwargs.get(key, default))

        return type(cls_name, (), {"__init__": __init__})

    class Frame:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    common_mod = types.ModuleType("common_python")
    utils_mod = types.ModuleType("common_python.utils")
    func_mod = types.ModuleType("common_python.utils.common_func")
    func_mod.create_cls = create_cls
    func_mod.Frame = Frame
    utils_mod.common_func = func_mod
    common_mod.utils = utils_mod
    sys.modules.setdefault("common_python", common_mod)
    sys.modules.setdefault("common_python.utils", utils_mod)
    sys.modules.setdefault("common_python.utils.common_func", func_mod)


_install_tools_stub()
_install_toml_stub()
_install_common_python_stub()
