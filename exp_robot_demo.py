import os
from datetime import datetime
import logging
import yaml

from lang2ltl import SHARED_DPATH, lang2ltl


if __name__ == "__main__":
    result_dpath = os.path.join(SHARED_DPATH, "results", "lang2ltl_api")
    os.makedirs(result_dpath, exist_ok=True)
    result_fpath = os.path.join(result_dpath, f"log_robot-demo_{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(result_fpath, mode='w'),
                            logging.StreamHandler()
                        ]
    )

    env_fpath = "data/robot_demo_configs/robot_demo_envs.yaml"
    env_fpath = env_fpath if os.path.isfile(env_fpath) else os.path.join(SHARED_DPATH, "data/robot_demo_configs/robot_demo_envs.yaml")

    exp_name = "robot-demo-house1"
    with open(env_fpath, "r") as file:
        robot_env = yaml.safe_load(file)[exp_name]

    for utt in robot_env["utts"]:
        out_ltl = lang2ltl(utt, robot_env["obj2sem"], keep_keys=robot_env["keep_keys"], exp_name=exp_name)
