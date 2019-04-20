import numpy as np
from experience import Experience

class Coach():
    """
    Class used to train and eval an agent
    """
    def __init__(self, agent, env):
        """
        Init the coach

        Params
        ======
        agent (DDPGAgent): Agent under training
        env (Unity environment): Environment agent is acting within
        """

        self.agent = agent
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

    def run_episode(self, max_steps, train = True):
        """
        Executes a single episode.

        Params
        ======
        max_steps (integer): The maximum time steps to run in a single episode.
        """

        env_info = self.env.reset(train_mode=train)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(len(states))
        for i in range(max_steps):
            actions = self.agent.act(states, noise = train)
            env_info = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            self.agent.learn(Experience(states, actions, rewards, next_states, dones))
            if dones[0]:
                break
            states = next_states
        self.agent.end_episode()
        return scores.max()

    def diagnostic(self, episode, scores, average_scores_over):
        """
        Prints diagnostic info about the performance of an agent.

        Params
        ======
        episode (ineger): The current episode
        scores (vector of floats): Scores earned by the agent
        average_scores_over (ineger): Calcualte average over the last X scores.
        """

        score_window = scores[-average_scores_over:]
        mean_score = np.mean(score_window)
        max_score = np.max(score_window)
        if (episode + 1) % average_scores_over == 0:
            end = "\r\n"
        else:
            end = "\r"
        print("Episode: {}, Mean: {}, Max: {}, Last: {}".format(episode, mean_score, max_score, scores[-1]), end=end)


    def run_episodes(self, num_episodes, max_steps, train = True):
        """
        Run a series of episodes.

        Params
        ======
        num_episodes (integer): Number of episodes to run
        max_steps (integer): Max steps per episode
        """

        scores = []
        print(num_episodes)
        for i in range(num_episodes):
            scores.append(self.run_episode(max_steps, train))
            self.diagnostic(i, scores, 100)
        return scores
