import numpy as np
import pandas as pd
import cufflinks as cf
import plotly.offline

class Advert():
    """
    class for ad where rewards come from a bernoulli distribution
    """
    def __init__(self, p):
        self.p = p
    
    def displayAd(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward


if __name__ == "__main__":
    adA = Advert(0.004)
    adB = Advert(0.016)
    adC = Advert(0.02)
    adD = Advert(0.028)
    adE = Advert(0.031)

    ads = [adA, adB, adC, adD, adE]

    #epsilon greedy
    eps = 0.1 #start with a high ε value and gradually reduce it
    nProd = 100000
    nAds = len(ads)
    Q = np.zeros(nAds) # action values
    N = np.zeros(nAds) # total impressions
    totalReward = 0
    avgRewards = [] #save average reward over time

    adChosen = np.random.randint(nAds)
    for i in range(nProd):
        R = ads[adChosen].display_ad()
        N[adChosen] += 1
        Q[adChosen] += (1 / N[adChosen]) * (R - Q[adChosen])
        totalReward += R
        avgReward = totalReward / (i + 1)
        avgRewards.append(avgReward)

    # Select the next ad to display
    if np.random.uniform() <= eps:
        adChosen = np.random.randint(nAds)
    else:
        adChosen = np.argmax(Q)
    dfRewardComparision['e-greedy: {}'.format(eps)] = avgRewards