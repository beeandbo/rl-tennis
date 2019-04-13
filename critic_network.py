import torch
import torch.nn as nn
import torch.nn.functional as functional
import network

class CriticNetwork(nn.Module):
    """
    Fully connected network for critic.  Takes in states and actions and
    produces a Q-value.  Actions are only input into network in second layer.
    """
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.output_layer = nn.Linear(300, 1)

        network.hidden_layer_init(self.fc1)
        network.hidden_layer_init(self.fc2)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        x = functional.relu(self.fc1(states))
        x = functional.relu(self.fc2(torch.cat((x, actions), dim=1)))
        return self.output_layer(x)
