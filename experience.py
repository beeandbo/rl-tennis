from collections import namedtuple

Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones'])
