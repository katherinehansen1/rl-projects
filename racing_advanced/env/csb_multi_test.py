import gymnasium as gym
from csb_multi_env import *
import time
import numpy as np


env = CodersStrikeBackMulti()
fps =  env.metadata.get('video.frames_per_second')

# Set pseudorandom seed for the getting the same game
env.seed(1234)

for n in range(10):
    state, _ = env.reset()

    done = False
    while not done:
        # display the game
        render = env.render()
        if not render:
            break
        # Take action (simple policy)
        actions = {}
        for racer in env.racers:
            aid = racer.aid
            targetX, targetY = state[aid][6:8]
            thrust = 100
            action = np.array([targetX, targetY, thrust], dtype=np.float32)
            actions[aid] = action

        # Slow down the cycle for a realistic simulation
        time.sleep(1.0/fps)

        # Do a game step
        state, reward, dones, trunc, _ = env.step(actions)
        done = dones["__all__"]
env.close()

