import math
import random
import numpy as np
import torch

from dqn import device


# Constants

FRAMES_CONCAT = 4 # number of frames concatenated to make an input state for the DQN

LEARNING_START = 100000 # number of frames before we start learning (to fill the replay buffer somewhat). 
BETA_START = 0.4 # prioritization weights exponent at the start of the learning
BETA_FINAL = 1.0 # prioritization weights exponent at the end of the learning
BETA_DURATION = 20000000 # number of frames for which the prioritization weights exponent changes


# experience processing machinery
"""
An experience consists of the frame before an action was taken, the action, the received reward,
a binary flag signaling whether the next state is terminal
and a flag that signals whether the state was not one of the initial ones for which we don't have enough ancestors to build up the input to the DQN.
"""

class experience:
    def __init__(self, frame, action, reward, terminal = False, learn_from = True):
        self.frame = frame
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.learn_from = learn_from

"""
The replay buffer stores the experiences, its .idx variable is the position-to-be of the next input.
There are also methods for constructing states (=concatenated frames) out of frames saved in the buffer.
The sampling of transitions is done here as well as the construction of all the data necessary for learning.
"""

class replay_buffer:
    def __init__(self, size, n_steps):
        self.memories = []
        self.size = size
        self.n_steps = n_steps
        self.idx = 0
        self.sum_tree = sum_tree(self.size)

    def __len__(self):
        return len(self.memories)

    def add_exp(self, exp):
        if len(self) == self.size:
            self.memories[self.idx] = exp
        else:
            self.memories.append(exp)

        priority = self.sum_tree.max_priority if exp.learn_from else 0.0
        self.sum_tree.update_leaf(self.idx, priority)
        self.idx = self.shift_idx(self.idx, 1)

    def shift_idx(self, idx, shift):
        return (idx + shift) % self.size

    # constructs a state from frames in the buffer
    def construct_state(self, idx):
        frames = tuple(self.memories[self.shift_idx(idx, -i)].frame for i in range(FRAMES_CONCAT))
        state = concat_frames(frames)
        return state

    # constructs the current state
    def current_state(self, frame):
        idx = self.shift_idx(self.idx, -1)
        frames = (frame, ) + tuple(self.memories[self.shift_idx(idx, -i)].frame for i in range(FRAMES_CONCAT - 1))
        state = concat_frames(frames)
        return state

    # the transitions are sampled using the sum tree
    def sample_exps(self, batch_size, frame_count):
        sampled_indices = []
        priorities = []

        # due to Multistep Learning and our way of constructing the input states, not all the indices can be used
        last_n = [self.shift_idx(self.idx - 1, -i) for i in range(self.n_steps)]
        first_few = [self.shift_idx(self.idx, i) for i in range(FRAMES_CONCAT - 1)]
        forbidden = last_n + first_few

        i = 0
        while(i < batch_size):
            r = random.uniform(0.0, self.sum_tree.sum)
            idx, priority = self.sum_tree.find_idx(r)
            if idx not in sampled_indices and idx not in forbidden and self.memories[idx].learn_from:
                sampled_indices.append(idx)
                priorities.append(priority)
                i += 1

        # the weights that fix the prioritization bias
        weights = torch.tensor([pow(1.0/len(self) * self.sum_tree.sum/priorities[i].item(), beta(frame_count)) for i in range(batch_size)])
        weights = weights/(weights.max().item())
        return (sampled_indices, weights.to(device))

    # states before and after transitions, rewards, the actions taken and the terminal flags are read from the buffer
    def make_learning_data(self, sampled_indices, batch_size, gamma):
        states_before = torch.cat(tuple(self.construct_state(sampled_indices[i]) for i in range(batch_size)))
        actions = [self.memories[sampled_indices[i]].action for i in range(batch_size)]

        rewards = torch.zeros(batch_size)
        terminal = [False for i in range(batch_size)]

        shifted_indices = sampled_indices.copy()

        for i in range(self.n_steps):
            rewards += pow(gamma, i)*torch.tensor([self.memories[shifted_indices[j]].reward if not terminal[j] else 0.0 for j in range(batch_size)])
            terminal = [terminal[j] or self.memories[shifted_indices[j]].terminal for j in range(batch_size)]
            shifted_indices = [self.shift_idx(shifted_indices[j], 1) for j in range(batch_size)]

        states_after = torch.cat(tuple(self.construct_state(shifted_indices[i]) for i in range(batch_size)))

        return states_before, states_after, actions, rewards.unsqueeze(1).to(device), terminal


"""
Priority sampling is proportional to the TD delta^OMEGA and is provided by a sum tree in which every node is a sum of its two children.
In sampling, the leaf nodes are effectively put side by side and the sampled node is chosen as the location where the random number falls.
"""

class sum_tree:
    def __init__(self, size):
        self.depth = math.ceil(math.log2(size)) + 1
        self.size = pow(2, self.depth - 1)
        self.levels = []
        self.sum = 0.0
        self.max_priority = 1.0

        # the bottom level is level 0
        s = self.size
        for _ in range(self.depth):
            self.levels.append(np.zeros(s))
            s = s//2

    def update_leaf(self, idx, priority):
        self.levels[0][idx] = priority
        for i in range(1, self.depth):
            idx = idx // 2
            self.levels[i][idx] = self.levels[i-1][2*idx] + self.levels[i-1][2*idx + 1]
        self.sum = self.levels[self.depth - 1][0]
        if priority > self.max_priority:
            self.max_priority = priority

    def find_idx(self, value):
        idx = 0
        for i in range(self.depth-1, 0, -1):
            going_right = value > self.levels[i-1][2*idx]
            value = value - int(going_right)*self.levels[i-1][2*idx]
            idx = 2*idx + int(going_right)
        return (idx, self.levels[0][idx])


# To make an input state, FRAMES_CONCAT frames are concatenated.

def concat_frames(frames):
    state = torch.cat(frames, 0)
    state = torch.unsqueeze(state, 0)
    return state


# The prioritization weights exponent beta runs throughout the training.

def beta(frame):
    beta = BETA_FINAL if frame > (BETA_DURATION + LEARNING_START) else BETA_FINAL + (BETA_START - BETA_FINAL)*(1.0-(frame-LEARNING_START)/BETA_DURATION)
    return beta
