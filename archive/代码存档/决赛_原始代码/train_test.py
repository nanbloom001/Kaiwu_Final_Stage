#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.utils.train_test_utils import run_train_test

# To run the train_test, you must modify the algorithm name here. It must be one of ppo, diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是ppo、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = ["ppo", "diy"]
algorithm_name = "ppo"


if __name__ == "__main__":
    run_train_test(
        algorithm_name=algorithm_name,
        algorithm_name_list=algorithm_name_list,
        env_vars={
            "replay_buffer_capacity": "10",
            "preload_ratio": "10",
            "train_batch_size": "2",
            "dump_model_freq": "1",
            "max_frame_no": "1000",
            # train_test 时显存不够，将 num_envs 降为 1
            # reduce num_envs to 1 for train_test due to GPU memory
            "KAIWU_TRAIN_TEST": "1",
        },
        shell="bash",
        skip_aisrv_alive_check=True,
        skip_error_scan=True,
        check_model_method="glob_stage_pkl",
    )
