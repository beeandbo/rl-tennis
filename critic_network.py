import torch
import torch.nn as nn
import torch.nn.functional as functional
import network

HIDDEN_LAYER_1 = 64
HIDDEL_LAYER_2 = 64

class CriticNetwork(nn.Module):
    """
    Fully connected network for critic.  Takes in states and actions and
    produces a Q-value.  Actions are only input into network in second layer.
    """
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYER_1)
        self.fc2 = nn.Linear(HIDDEN_LAYER_1 + action_dim, HIDDEL_LAYER_2)
        self.output_layer = nn.Linear(HIDDEL_LAYER_2, 1)

        network.hidden_layer_init(self.fc1)
        network.hidden_layer_init(self.fc2)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        x = functional.relu(self.fc1(states))
        x = functional.relu(self.fc2(torch.cat((x, actions), dim=1)))
        return self.output_layer(x)
