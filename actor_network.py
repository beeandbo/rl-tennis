import torch.nn as nn
import torch.nn.functional as functional
import network

HIDDEN_LAYER_1 = 64
HIDDEL_LAYER_2 = 64

class ActorNetwork(nn.Module):
    """
    The Actor network used by the MADDPG agent.  It takes states as input
    and produces an action vector as output.
    """
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.input_norm = nn.BatchNorm1d(state_dim, affine=False)
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYER_1)
        self.fc1_norm = nn.BatchNorm1d(HIDDEN_LAYER_1, affine=False)
        self.fc2 = nn.Linear(HIDDEN_LAYER_1, HIDDEL_LAYER_2)
        self.fc2_norm = nn.BatchNorm1d(HIDDEL_LAYER_2, affine=False)
        self.output_layer = nn.Linear(HIDDEL_LAYER_2, action_dim)

        network.hidden_layer_init(self.fc1)
        network.hidden_layer_init(self.fc2)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        x = states
        x = self.input_norm(x)
        x = functional.relu(self.fc1(x))
        x = self.fc1_norm(x)
        x = functional.relu(self.fc2(x))
        x = self.fc2_norm(x)
        return functional.tanh(self.output_layer(x))
