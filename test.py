import time
import os
import gym
import numpy as np
# import torch as torch # dont need in this file
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
# from networks import *
from td3_torch import *
# from buffer import *

if __name__ == '__main__':

    if not os.path.exists("tmp/td3_proj"):
        os.makedirs("tmp/td3_proj")

    env_name = "Door"

    env = suite.make(
        env_name,
        robots = ["Panda"],
        controller_configs = suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        render_camera="frontview",
        has_offscreen_renderer=True,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
        warmup=0,
        noise=0.0,
    )

    agent.load_module()

    n_games = 10
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0.0

        while not done:
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = next_observation

        print(f"Episode:{i} Score: {score}")

    env.close()
