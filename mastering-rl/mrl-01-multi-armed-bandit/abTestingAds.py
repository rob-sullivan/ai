import numpy as np

class Advert():
    """
    class for ad where rewards come from a bernoulli distribution
    """
    def __init__(self, p):
        self.p = p
    
    def display_ad(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward


if __name__ == "__main__":
    adA = Advert(0.004)
    adB = Advert(0.016)
    adC = Advert(0.02)
    adD = Advert(0.028)
    adE = Advert(0.031)

    ads = [adA, adB, adC, adD, adE]