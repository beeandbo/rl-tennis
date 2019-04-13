import torch.nn as nn
import torch.nn.functional as functional
import network

class ActorNetwork(nn.Module):
    """
    The Actor network used by the DDPG agent.  It takes states as input
    and produces an action vector as output.
    """
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.input_norm = nn.BatchNorm1d(state_dim, affine=False)
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc1_norm = nn.BatchNorm1d(400, affine=False)
        self.fc2 = nn.Linear(400, 300)
        self.fc2_norm = nn.BatchNorm1d(300, affine=False)
        self.output_layer = nn.Linear(300, action_dim)

        network.hidden_layer_init(self.fc1)
        network.hidden_layer_init(self.fc2)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        x = self.input_norm(states)
        x = functional.relu(self.fc1(x))
        x = self.fc1_norm(x)
        x = functional.relu(self.fc2(x))
        x = self.fc2_norm(x)
        return functional.tanh(self.output_layer(x))
