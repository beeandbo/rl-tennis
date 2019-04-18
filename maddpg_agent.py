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
BATCH_SIZE = 1024

# Number of steps taken between each round of training.  Each agent
# action is considered a step (so 20 simultaneous agents acting mean 20 steps)
STEPS_BETWEEN_TRAINING = 20

# Reward decay
GAMMA = 0.95

# Learning rate for the actor network
ACTOR_LEARNING_RATE = 1e-2

# Learning rate for the critic network
CRITIC_LEARNING_RATE = 1e-2

# Rate at which target networks are updated
TAU = 1e-2

# Weight decay term used for training the critic network
CRITIC_WEIGHT_DECAY = 0.0000

# Random process parameters
RANDOM_THETA = 0.15
RANDOM_SIGMA = 0.2


class MADDPGAgent():
    """
    Multi Aagent Deep deterministic policy gradient agent as described in
    https://arxiv.org/pdf/1706.02275.pdf.

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
        self.actors = []
        self.actor_targets = []
        self.actor_optimizers = []

        self.critics = []
        self.critic_targets = []
        self.critic_optimizers = []
        for i in range(num_agents):

            # Actor
            actor = ActorNetwork(state_size, action_size)
            self.actors.append(actor)
            self.actor_targets.append(ActorNetwork(state_size, action_size))
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE))

            # Critic
            # Note: we use action_size * num_agents since we'll pass in the actions of all agents concatenated
            critic = CriticNetwork(state_size, action_size * num_agents)
            self.critics.append(critic)
            self.critic_targets.append(CriticNetwork(state_size, action_size * num_agents))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY))

        self.replay_buffer = ReplayBuffer(action_size, REPLAY_BUFFER_SIZE, None)
        self.steps = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.random_process = OrnsteinUhlenbeckProcess((1, action_size), sigma=RANDOM_SIGMA, theta=RANDOM_THETA)


    def act(self, all_states, noise = True):
        """
        Returns an action vector based on the current game state.

        Params
        ======
        all_states (array_like): A matrix of game states (each row represents the
            state of an agent)
        noise (boolean): Add random noise to the predicted action.  Aids
            exploration of the environment during training.
        """
        #print("states")
        #print(states)
        all_actions = []

        # Generate actions for each 'agent'
        for state, actor in zip(all_states, self.actors):
            actor.eval()
            with torch.no_grad():
                actions = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().numpy()
            actor.train()
            if noise:
                actions = actions + self.random_process.sample()
            actions = np.clip(actions, -1, 1)
            all_actions.append(actions)
        print(all_actions)
        return np.vstack(all_actions)

    def predict_and_vectorize_actions(self, experiences, agent_index):
        vectors = []
        for experience in experiences:
            actions = []
            for i in range(self.num_agents):
                if i == agent_index:
                    actor = self.actors[agent_index]
                    state = torch.tensor(experience.states[agent_index], dtype=torch.float32)
                    actions.append(actor(state).unsqueeze(0))
                else:
                    actions.append(torch.from_numpy(experience.actions[i]).float().unsqueeze(0))
            vectors.append(torch.cat(actions, dim=1))
        return torch.cat(vectors, dim=0)

    def predict_and_vectorize_next_actions(self, experiences):
        vectors = []
        for exprience in experiences:
            actions = []
            for state, actor in zip(exprience.next_states, self.actor_targets):
                    actions.append(actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach())
            vectors.append(torch.cat(actions, dim=1))
        return torch.cat(vectors)

    def vectorize_actions(self, experiences):
        return torch.from_numpy(np.vstack([np.concatenate(e.actions) for e in experiences if e is not None])).float().to(self.device)

    def vectorize_per_agent_data(self, experiences, agent_index):
        states = torch.from_numpy(np.vstack([e.states[agent_index] for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.rewards[agent_index]  for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_states[agent_index]  for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.dones[agent_index]  for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, rewards, next_states, dones)

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

        # Transform agent indendent data into vectorized tensors
        next_actions = self.predict_and_vectorize_next_actions(experiences)
        actions = self.vectorize_actions(experiences)

        # Iterate through each agent
        for i in range(self.num_agents):

            # Transform agent dependent data into vectorized tensors
            states, rewards, next_states, dones = self.vectorize_per_agent_data(experiences, i)
            rewards = self.normalize(rewards)

            # Grab networks this agent offset
            critic = self.critics[i]
            critic_target = self.critic_targets[i]
            critic_optimizer = self.critic_optimizers[i]
            actor = self.actors[i]
            actor_target = self.actor_targets[i]
            actor_optimizer = self.actor_optimizers[i]

            # Use the target critic network to calculate a target q value\
            q_target = rewards + GAMMA * critic_target(next_states, next_actions) * (1-dones)

            # Calculate the predicted q value
            q_predicted = critic(states, actions)

            # Update critic network
            critic_loss = F.mse_loss(q_predicted, q_target)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
            critic_optimizer.step()

            # Update predicted action using policy gradient
            actions_predicted = self.predict_and_vectorize_actions(experiences, i)
            policy_loss = -critic(states, actions_predicted).mean()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

            # Soft update target networks
            self.soft_update(actor_target.parameters(), actor.parameters())
            self.soft_update(critic_target.parameters(), critic.parameters())

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
            self.train(self.replay_buffer.sample(BATCH_SIZE))

    def save(self, filename):
        """Saves learned params of underlying networks to a checkpoint file.

        Params
        ======
            filename (string): Target file.  agent- and critic- are prepended
                for the agent and critic network, respectively
        """
        # torch.save(self.local_actor_network.state_dict(), "actor-" + filename)
        # torch.save(self.local_critic_network.state_dict(), "critic-" + filename)

    def load(self, filename):
        """Loads learned params generated by save() into underlying networks.

            filename (string): Path to file.  There should be an agent- and
            critic- version of this file.
        """
        # self.local_actor_network.load_state_dict(torch.load("actor-" + filename))
        # self.target_actor_network.load_state_dict(torch.load("actor-" + filename))
        #
        # self.local_critic_network.load_state_dict(torch.load("critic-" + filename))
        # self.target_critic_network.load_state_dict(torch.load("critic-" + filename))


    def end_episode(self):
        """
        Tell the agent that an episode is complete.
        """
        self.random_process.reset()
        self.steps = 0
