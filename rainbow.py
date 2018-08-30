import random
import time

from agent import agent
from enviroment import enviroment


# Constants

GAME = 'AtlantisNoFrameskip-v4' # the gym enviroment used. The right choice is "GameNoFrameskip-v4". For Space Invaders, change also the frameskip parameter in enviroment.py
TOTAL_FRAMES = 20000000 # length of the training run

SAVE_FREQ = 50 # episodes between saving the network
SAVE_PATH = "ai.data"

MIN_NO_OPS = 4 # lower limit of the no-ops inserted at the start of each episode
MAX_NO_OPS = 30 # upper limit of the no-ops inserted at the start of each episode


"""
The main training loop.
A random number of no-ops is inserted at the start of each episode.
The DQN is saved periodically.
"""

def training_loop(env, ai, total_frames):
    while(env.frame_count < total_frames):
        t0 = time.time()
        no_ops = random.randrange(MIN_NO_OPS, MAX_NO_OPS + 1)
        reward = env.run_episode(ai, no_ops)
        print(env.episode, env.frame_count, reward, time.time()-t0)
        
        if (env.episode % SAVE_FREQ) == SAVE_FREQ - 1:
            ai.save(SAVE_PATH)


# Running the training

env = enviroment(GAME)
ai = agent(env.n_actions)
training_loop(env, ai, TOTAL_FRAMES)
