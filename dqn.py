import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants

SIGMA_0 = 0.5 # scales the initial noise
SIZE = 32 # size of the DQN, the standard choice is 32, but on weaker hardware 16 could be preferable

# device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# deep q network
"""
The DQN takes in the last FRAMES_CONCAT = 4 consecutive preprocessed frames (= a state).
It has three convolutional layers, that decrease the size of each channel to 7x7.
Two fully connected layers follow for each of the advantage and value streams. 
The streams are then combined into a distribution over the atoms.
""" 

class DQN(nn.Module):
    def __init__(self, n_actions, n_atoms):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.conv1 = nn.Conv2d(4, SIZE, 8, stride = 4)
        self.conv2 = nn.Conv2d(SIZE, 2*SIZE, 4, stride = 2)
        self.conv3 = nn.Conv2d(2*SIZE, 2*SIZE, 3, stride = 1)

        self.hidden_adv = noisy_linear(49*2*SIZE, 16*SIZE, SIGMA_0)
        self.hidden_val = noisy_linear(49*2*SIZE, 16*SIZE, SIGMA_0)
        self.out_adv = noisy_linear(16*SIZE, self.n_actions * self.n_atoms, SIGMA_0)
        self.out_val = noisy_linear(16*SIZE, self.n_atoms, SIGMA_0)

        self.softmax = nn.Softmax(dim = 2)

        # the noise parameters
        self.hidden_adv_weight_eps = torch.empty([16*SIZE, 49*2*SIZE])
        self.hidden_adv_bias_eps = torch.empty(16*SIZE)
        self.hidden_val_weight_eps = torch.empty([16*SIZE, 49*2*SIZE])
        self.hidden_val_bias_eps = torch.empty(16*SIZE)

        self.out_adv_weight_eps = torch.empty([self.n_actions * self.n_atoms, 16*SIZE])
        self.out_adv_bias_eps = torch.empty(self.n_actions * self.n_atoms)
        self.out_val_weight_eps = torch.empty([self.n_atoms, 16*SIZE])
        self.out_val_bias_eps = torch.empty(self.n_atoms)

        self.reset_noise()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 49*2*SIZE)

        a = F.relu(self.hidden_adv(x, self.hidden_adv_weight_eps, self.hidden_adv_bias_eps))
        v = F.relu(self.hidden_val(x, self.hidden_val_weight_eps, self.hidden_val_bias_eps))

        a = self.out_adv(a, self.out_adv_weight_eps, self.out_adv_bias_eps)
        a = a.view(-1, self.n_actions, self.n_atoms)
        v = self.out_val(v, self.out_val_weight_eps, self.out_val_bias_eps)
        v = v.view(-1, 1, self.n_atoms)

        # combining the value and advantage streams
        d = v + a - a.mean(dim = 1).view(-1, 1, self.n_atoms)

        # outputting a distribution
        p = self.softmax(d)
        return p

    # factorized Gaussian noise is used to save time
    def reset_noise(self):
        self.hidden_adv_weight_eps, self.hidden_adv_bias_eps = self.generate_eps(49*2*SIZE, 16*SIZE)
        self.hidden_val_weight_eps, self.hidden_val_bias_eps = self.generate_eps(49*2*SIZE, 16*SIZE)

        self.out_adv_weight_eps, self.out_adv_bias_eps = self.generate_eps(16*SIZE, self.n_actions*self.n_atoms)
        self.out_val_weight_eps, self.out_val_bias_eps = self.generate_eps(16*SIZE, self.n_atoms)

    def generate_eps(self, in_channels, out_channels):
        in_eps = torch.randn(in_channels, device = device)
        out_eps = torch.randn(out_channels, device = device)
        f_in_eps = in_eps.abs().sqrt()*in_eps.sign()
        f_out_eps = out_eps.abs().sqrt()*out_eps.sign()
        weight_eps = torch.ger(f_out_eps, f_in_eps)
        return (weight_eps, f_out_eps)


"""
A noisy version of a fully connected linear layer. 
This is only a modification of the default nn.Linear PyTorch class.
"""

class noisy_linear(nn.Module):
    def __init__(self, in_features, out_features, sigma_0):
        super(noisy_linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters(sigma_0)

    def reset_parameters(self, sigma_0):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_sigma.data.fill_(sigma_0*stdv)

        self.bias.data.uniform_(-stdv, stdv)
        self.bias_sigma.data.fill_(sigma_0*stdv)

    def forward(self, input, weight_eps, bias_eps):
        return F.linear(input, self.weight + self.weight_sigma*weight_eps, self.bias + self.bias_sigma*bias_eps)
