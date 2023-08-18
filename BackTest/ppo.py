import numpy as np
import torch


class SimpleBandit:
    def __init__(self, n_levers):
        self.n_levers = n_levers
        self.rewards = np.random.randint(20, size=n_levers)

    def context():
        return 0

    def pull(self, lever):
        return self.rewards[lever]


def bandit_run(bandit, agent):
    return bandit
