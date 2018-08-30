import torch
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import replay_buffer
from dqn import DQN
from dqn import device


# Constants

N_STEPS = 3 # number of steps for Multistep Learning

N_ATOMS = 51 # number of distributional atoms
V_MIN = -10.0 # lower limit of allowed Q-values
V_MAX = 10.0 # upper limit of allowed Q-values

REPLAY_BUFFER_SIZE = 250000
OMEGA = 0.5 # prioritization exponent

GAMMA = 0.99 # discount rate

LEARNING_RATE = 0.0000625
ADAM_EPS = 0.00015
BATCH_SIZE = 32


# The AI class
"""
The agent has two networks, the online one on which the learning is done 
and the target network which lags behind as to decorrelate the updates.
The learning algrorithm is the Distributional Prioritized Multistep Double DQN.
"""

class agent():
    def __init__(self, n_actions, path = None):
        self.dqn = DQN(n_actions, N_ATOMS)
        self.dqn.to(device)

        # loading the network
        if path is not None:
            self.dqn.load_state_dict(torch.load(path))

        self.dqn_target = DQN(n_actions, N_ATOMS)
        self.dqn_target.to(device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        self.optimizer = optim.Adam(self.dqn.parameters(), lr = LEARNING_RATE, eps = ADAM_EPS)
        self.buf = replay_buffer(REPLAY_BUFFER_SIZE, N_STEPS)

        # the distribution for terminal states
        self.dist_zero = torch.zeros(N_ATOMS, device = device)
        self.dist_zero[N_ATOMS // 2] = 1.0

        self.atoms = torch.linspace(V_MIN, V_MAX, steps = N_ATOMS, device = device)
        self.delta_z = (V_MAX - V_MIN)/(N_ATOMS - 1)

    def learn(self, frame_count):
        # setting up the batch
        sampled_indices, weights = self.buf.sample_exps(BATCH_SIZE, frame_count)
        states_before, states_after, actions, rewards, terminal = self.buf.make_learning_data(sampled_indices, BATCH_SIZE, GAMMA)

        # running the ai
        self.optimizer.zero_grad()

        dist_before = self.dqn(states_before)
        dist_actions_taken = torch.stack(tuple(dist_before[i][actions[i]] for i in range(BATCH_SIZE)))
        
        # figuring out the distribution for the states after transitions
        self.dqn_target.reset_noise()
        with torch.no_grad():
            dist_after_target = self.dqn_target(states_after)
            dist_after_online = self.dqn(states_after)

        actions_after = [torch.mm(dist_after_online[i], torch.unsqueeze(self.atoms, 1)).argmax().item() for i in range(BATCH_SIZE)]

        dist_after = torch.stack(tuple((self.dist_zero if terminal[i] else dist_after_target[i][actions_after[i]]) for i in range(BATCH_SIZE)))
        atoms_shifted = rewards + pow(GAMMA, N_STEPS)*self.atoms.expand(BATCH_SIZE, N_ATOMS)
        dist_projected = self.project_dist(atoms_shifted, dist_after)

        # computing the loss (K-L divergence)
        loss = F.kl_div(dist_actions_taken.log(), dist_projected, reduce = False)
        loss = loss.sum(1)

        # updating the sum tree
        for i in range(BATCH_SIZE):
            priority = pow(abs(loss[i]), OMEGA)
            self.buf.sum_tree.update_leaf(sampled_indices[i], priority)
        
        loss = loss * weights
        loss.sum().backward()
        self.optimizer.step()

    # projects a distribution onto the atoms of the network
    def project_dist(self, shifted_atoms, probs):
        shifted_atoms.clamp_(V_MIN, V_MAX)
        shifted_atoms = shifted_atoms.to("cpu")*0.99999
        probs = probs.to("cpu")
        projection = torch.zeros(BATCH_SIZE, N_ATOMS)
        
        indices_float = (shifted_atoms - V_MIN)/self.delta_z
        indices_lower = indices_float.floor().long()
        indices_upper = indices_float.ceil().long()
        prob_lower = (1.0 - indices_float + indices_lower.float())*probs
        prob_upper = (indices_float - indices_lower.float())*probs

        for i in range(BATCH_SIZE):
            projection[i].index_add_(0, indices_lower[i], prob_lower[i])
            projection[i].index_add_(0, indices_upper[i], prob_upper[i])

        return projection.to(device)

    def save(self, path):
        torch.save(self.dqn.state_dict(), 'ai.data')

