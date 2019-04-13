import numpy as np
import random
import copy

# Adapted from Udacity reference implementation



class OrnsteinUhlenbeckProcess():
    """
    Class for producing random values used to add noise to the agent.
    """
    def __init__(self, size, theta=0.15, sigma=0.2, mu=0.):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        self.size = size

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.size)
        self.state = x + dx
        return self.state
