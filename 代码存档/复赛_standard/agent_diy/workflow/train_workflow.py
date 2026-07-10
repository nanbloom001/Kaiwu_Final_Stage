#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import Frame
import random
import os
from agent_diy.feature.definition import (
    sample_process,
)
from agent_diy.conf.conf import Config
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):

    # Number of agents, in legged_robot_competition_26 the value is 1
    # 智能体数量，在legged_robot_competition_26中值为1
    agent = agents[0]
    # legged_robot_competition_26 environment
    # legged_robot_competition_26环境
    env = envs[0]

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check {usr_conf_file}")
        raise Exception(f"usr_conf is None, please check {usr_conf_file}")

    # Please implement your DIY algorithm flow
    # 请实现你DIY的算法流程
    # ......

    # Model saving
    # 保存模型
    agent.save_model()

    env.close()

    return
