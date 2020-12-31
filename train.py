import os
import json
import pickle
import argparse

import torch
import gym

from models.nets import Expert
from models.gail import GAIL


def main(env_name, gpu_num):
    if not os.path.isdir(".ckpts"):
        os.mkdir(".ckpts")

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts/%s/" % env_name

    print(expert_ckpt_path)
    with open(expert_ckpt_path + "model_config.json") as f:
        expert_config = json.load(f)

    ckpt_path = ".ckpts/%s/" % env_name
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(ckpt_path + "model_config.json", "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    env.reset()

    print(env.action_space)

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    expert = Expert(state_dim, action_dim, discrete, **expert_config)
    expert.pi.load_state_dict(torch.load(expert_ckpt_path + "policy.ckpt"))

    model = GAIL(state_dim, action_dim, discrete, config)

    if torch.cuda.is_available():
        with torch.cuda.device(gpu_num):
            results = model.train(env, expert)
    else:
        results = model.train(env, expert)

    env.close()

    with open(ckpt_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), ckpt_path + "policy.ckpt")
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), ckpt_path + "value.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        default=0,
        help="Type the number of the GPU of your GPU mahine \
            you want to use if possible"
    )
    args = parser.parse_args()

    main(**vars(args))
