import gymnasium as gym
from csb_multi_env import *
import time
import numpy as np


frames_per_action = 4
env = CodersStrikeBackMulti(dt=1/frames_per_action)
fps =  env.metadata.get('video.frames_per_second')

# Set pseudorandom seed for the getting the same game
env.seed(1234)

for n in range(10):
    state, _ = env.reset()

    done = False
    i=0
    while not done:
        # display the game
        render = env.render()
        if not render:
            break
        # Take action (simple policy)
        if i % frames_per_action == 0:
            actions = {}
            for racer in env.racers:
                aid = racer.aid
                targetX, targetY = state[aid][6:8]
                thrust = 100
                action = np.array([targetX, targetY, thrust], dtype=np.float32)
                actions[aid] = action

        # Slow down the cycle for a realistic simulation
        time.sleep(1.0/(fps*frames_per_action))

        # Do a game step
        state, reward, dones, trunc, _ = env.step(actions)
        done = dones["__all__"]
env.close()

