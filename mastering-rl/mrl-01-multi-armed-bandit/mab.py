import numpy as np

class Bandit():
    """
    class for single slot machine where rewards are gaussian
    """
    def __init__(self, mean=0, stdev=1):
        self.mean = mean
        self.stdev = stdev

    def pull_lever(self):
        reward = np.random.normal(self.mean, self.stdev)
        return np,round(reward, 1)


if __name__ == "__main__":
    g1 = Bandit(5, 2)
    g2 = Bandit(6, 2)
    g3 = Bandit(1, 5)

    g1.pull_lever()