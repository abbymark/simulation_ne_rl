import gymnasium as gym
import time
import argparse
import numpy as np
import torch

from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME, help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory for video recording")
    parser.add_argument("--no-vis", default=True, dest="vis", action="store_false", help="Disable visualization of the game play")
    args = parser.parse_args()

    env = wrappers.make_env(args.env, render_mode=("human"))
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda storage, loc: storage)
    net.load_state_dict(state)

    state = env.reset()[0]
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals_v = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals_v)
        c[action] += 1

        state, reward, done, *_ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.vis:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()
