import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import gym

from replay_buffer import experience
from dqn import device
from replay_buffer import LEARNING_START


# Constants

FRAMESKIP = 4 # frames skipped between actions chosen by the network
REPLAY_FREQ = 4 # actions chosen between each learning phase
TARGET_UPDATE_FREQ = 1000 # learning phases between each update of the target network
RESIDUAL_EPS = 0.02 # the actions are sometimes chosen randomly. Some games can get stuck otherwise.


# The main enviroment class. Frame preprocessing is also done here.

class enviroment:
    def __init__(self, game):
        self.env = gym.make(game)
        self.frame_count = 0
        self.episode = 1
        self.target_update_counter = 0
        self.n_actions = self.env.action_space.n
    
    # a container for the enviroment step method. Frameskip and preprocessing is taken care of here.
    def step(self, action):
        reward = 0.0
        done = False
        info = None
        raw = []
        for i in range(FRAMESKIP):
            raw_frame, reward_now, done_now, info = self.env.step(action)
            reward += reward_now
            done = done or done_now
            if (i > FRAMESKIP - 3):
                raw.append(tf_small(raw_frame))

        frame = torch.max(raw[0], raw[1])
        return frame, reward, done, info

    def run_episode(self, ai, no_ops, train = True):
        # first no_ops frames, do nothing (but save the experiences so that we're able to stack 4 frames to the DQN input)
        raw_frame = self.env.reset()
        info = None

        for i in range(0, no_ops):
            frame = preprocess(raw_frame)
            exp = experience(frame, 0, 0.0, terminal = False, learn_from = False)
            ai.buf.add_exp(exp)
            raw_frame, _, _, info = self.step(0)
    
        self.frame_count += no_ops*FRAMESKIP  

        # the episode variables
        lives = info['ale.lives']
        life_lost = False
        done = False  
        replay_counter = 0
        total_reward = 0

        # the main loop
        while(not done):

            # constructing the input state
            frame = preprocess(raw_frame)
            state = ai.buf.current_state(frame)

            # choosing an action
            if random.random() > RESIDUAL_EPS:
                with torch.no_grad():
                    Q = torch.mm(ai.dqn(state)[0], torch.unsqueeze(ai.atoms, 1))
                action = torch.argmax(Q).item()
            else:
                action = random.randrange(self.n_actions)

            # interacting with the enviroment
            raw_frame, reward, done, info = self.step(action)
            self.env.render()
     
            # dealing with the experience
            life_lost = info['ale.lives'] < lives or done
            lives = info['ale.lives']
        
            total_reward += reward
            # The reward is clipped to [-1.0, 1.0]
            reward = np.clip(reward, -1.0, 1.0)
            self.frame_count += FRAMESKIP
            replay_counter += 1

            exp = experience(frame, action, reward, terminal = life_lost)
            ai.buf.add_exp(exp)

            # learning from saved experiences
            if replay_counter % REPLAY_FREQ == 0:
                ai.dqn.reset_noise()
                if train and self.frame_count > LEARNING_START:
                    self.target_update_counter += 1
                    ai.learn(self.frame_count)
                    if (self.target_update_counter == TARGET_UPDATE_FREQ):
                        self.target_update_counter = 0
                        ai.dqn_target.load_state_dict(ai.dqn.state_dict())
        self.episode +=1
        return total_reward


"""
Transforming raw input into the dqn input plus some debug helpers.
The raw frame is transformed to grayscale and resized to 84x84. 
"""

tf = transforms.Compose([transforms.ToPILImage(),
                         transforms.Grayscale(),
                         transforms.Resize((84,84)),
                         transforms.ToTensor()])

tf_small = transforms.Compose([transforms.ToPILImage(),
                         transforms.ToTensor()])

def preprocess(raw):
    frame = tf(raw)
    frame = frame.to(device)
    return frame

def view_preprocessed(frame):
    state = torch.unsqueeze(frame, 0)
    im_show(torchvision.utils.make_grid(state))

def im_show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block = False)
    input('')
    plt.clf()
