from collections import deque
import random

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.seed = random.seed(seed)

    def add(self, experience):
        """Add a new experience to memory."""
        self.memory.append(experience)

    def sample(self, sample_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=sample_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
