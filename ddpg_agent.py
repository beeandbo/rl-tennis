import numpy as np
import random
from critic_network import CriticNetwork
from actor_network import ActorNetwork
import torch
from replay_buffer import ReplayBuffer
import torch.optim as optim
import torch.nn.functional as F
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess

# Size of the replay buffer storing past experiences for training
REPLAY_BUFFER_SIZE = 10**6

# Number of experiences to use per training minibatch
BATCH_SIZE = 256

# Number of steps taken between each round of training.  Each agent
# action is considered a step (so 20 simultaneous agents acting mean 20 steps)
STEPS_BETWEEN_TRAINING = 2 * 4 # 2 agents for 4 steps

# Perform training this many times per rouund
ITERATIONS_PER_TRAINING = 1

# Reward decay
GAMMA = 0.99

# Learning rate for the actor network
ACTOR_LEARNING_RATE = 1e-4

# Learning rate for the critic network
CRITIC_LEARNING_RATE = 1e-3

# Rate at which target networks are updated
TAU = 0.001

# Weight decay term used for training the critic network
CRITIC_WEIGHT_DECAY = 0.0001

# Random process parameters
RANDOM_THETA = 0.15
RANDOM_SIGMA = 0.2


class DDPGAgent():
    """
    Deep deterministic policy gradient agent as described in
    https://arxiv.org/abs/1509.02971.

    This agent is meant to operate on low dimensional inputs, not raw pixels.

    To use the agent, you can get action predictions using act(), and to teach
    the agent, feed the results to learn.
    """
    def __init__(self, state_size, action_size, num_agents):
        """ Initialize agent.

        Params
        ======
        state_size (integer): Size of input state vector
        action_size (integer): Size of action vector
        num_agents (integer): Number of simultaneous agents in the environment
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # Actor
        self.local_actor_network = ActorNetwork(state_size, action_size)
        self.target_actor_network = ActorNetwork(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=ACTOR_LEARNING_RATE)

        # Critic
        self.local_critic_network = CriticNetwork(state_size, action_size)
        self.target_critic_network = CriticNetwork(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY)

        self.replay_buffer = ReplayBuffer(action_size, REPLAY_BUFFER_SIZE, None)
        self.steps = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.random_process = OrnsteinUhlenbeckProcess((num_agents, action_size), sigma=RANDOM_SIGMA, theta=RANDOM_THETA)


    def act(self, states, noise = True):
        """
        Returns an action vector based on the current game state.

        Params
        ======
        states (array_like): A matrix of game states (each row represents the
            state of an agent)
        noise (boolean): Add random noise to the predicted action.  Aids
            exploration of the environment during training.
        """
        #print("states")
        #print(states)

        self.local_actor_network.eval()
        with torch.no_grad():
            actions = self.local_actor_network(torch.tensor(states, dtype=torch.float32)).detach().numpy()
        self.local_actor_network.train()
        if noise:
            actions = actions + self.random_process.sample()
        actions = np.clip(actions, -1, 1)
        #print("actions")
        #print(actions)
        return actions

    def vectorize_experiences(self, experiences):
        """Vectorizes experience objects for use by pytorch

        Params
        ======
            experiences (array_like of Experience objects): Experiences to
                vectorize
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def normalize(self, to_normalize):
        """
        Normalize the each row of the input along the 0 dimension using the
        formula (value - mean)/std

        Params
        ======
        to_normalize (array_like): Values to normalize
        """

        std = to_normalize.std(0)
        mean = to_normalize.mean(0)
        return (to_normalize - mean)/(std + 1e-5)

    def soft_update(self, target_parameters, local_parameters):
        """
        Updates the given target network parameters with the local parameters
        using a soft update strategy: tau * local + (1-tau) * target
        """

        for target, local in zip(target_parameters, local_parameters):
            target.data.copy_(TAU*local.data + (1.0-TAU)*target.data)

    def train(self, experiences):
        """
        Trains the actor and critic networks using a minibatch of experiences

        Params
        ======
        experiences (array_like of Experience): Minibatch of experiences
        """
        states, actions, rewards, next_states, dones = self.vectorize_experiences(experiences)
        #states = self.normalize(states)
        #next_states = self.normalize(next_states)
        rewards = self.normalize(rewards)

        # Use the target critic network to calculate a target q value
        next_actions = self.target_actor_network(next_states)
        q_target = rewards + GAMMA * self.target_critic_network(next_states, next_actions) * (1-dones)

        # Calculate the predicted q value
        q_predicted = self.local_critic_network(states, actions)

        # Update critic network
        critic_loss = F.mse_loss(q_predicted, q_target)
        #print(critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic_network.parameters(), 1)
        self.critic_optimizer.step()

        # Update predicted action using policy gradient
        actions_predicted = self.local_actor_network(states)
        #print(self.local_critic_network(states, actions_predicted).mean())
        policy_loss = -self.local_critic_network(states, actions_predicted).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        #print(policy_loss)
        self.actor_optimizer.step()

        self.soft_update(self.target_actor_network.parameters(), self.local_actor_network.parameters())
        self.soft_update(self.target_critic_network.parameters(), self.local_critic_network.parameters())

    def learn(self, experience):
        """
        Tells the agent to learn from an experience.  This may not immediately
        result in training since this agent uses a replay buffer.

        Params
        ======
        experience (Experience): An experience used to teach the agent.
        """
        self.replay_buffer.add(experience)
        self.steps += 1
        if self.steps % STEPS_BETWEEN_TRAINING == 0 and len(self.replay_buffer) >= BATCH_SIZE:
            for i in range(ITERATIONS_PER_TRAINING):
                self.train(self.replay_buffer.sample(BATCH_SIZE))

    def save(self, filename):
        """Saves learned params of underlying networks to a checkpoint file.

        Params
        ======
            filename (string): Target file.  agent- and critic- are prepended
                for the agent and critic network, respectively
        """
        torch.save(self.local_actor_network.state_dict(), "actor-" + filename)
        torch.save(self.local_critic_network.state_dict(), "critic-" + filename)

    def load(self, filename):
        """Loads learned params generated by save() into underlying networks.

            filename (string): Path to file.  There should be an agent- and
            critic- version of this file.
        """
        self.local_actor_network.load_state_dict(torch.load("actor-" + filename))
        self.target_actor_network.load_state_dict(torch.load("actor-" + filename))

        self.local_critic_network.load_state_dict(torch.load("critic-" + filename))
        self.target_critic_network.load_state_dict(torch.load("critic-" + filename))


    def end_episode(self):
        """
        Tell the agent that an episode is complete.
        """
        self.random_process.reset()
        self.steps = 0
