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

    #A/B Testing
    nTest = 10000
    nProd = 90000
    nAds = len(ads)
    Q = np.zeros(nAds) # action values
    N = np.zeros(nAds) # total impressions

    totalReward = 0
    avgRewards = [] #save average reward over time


    #A/B/n test
    for i in range(nTest):
        adChosen = np.random.randint(nAds)
        R = ads[adChosen].displayAd() # observe reward
        N[adChosen] += 1
        Q[adChosen] += (1/N[adChosen]) * (R-Q[adChosen])
        totalReward += R
        avgReward = totalReward / (i+1)
        avgRewards.append(avgReward)

    bestAd = np.argmax(Q) #find best action
    print("Best Performing Ad: " + chr(ord('A') + bestAd))

    adChosen = bestAd
    for i in range(nProd):
        R = ads[adChosen].displayAd()
        totalReward += R
        avgReward = totalReward / (i+1)
        avgRewards.append(avgReward)

    dfRewardComparision = pd.DataFrame(avgReward, columns=['A/B/n'])

    cf.go_offline()
    cf.set_config_file(world_readable=True, theme='white')

    dfRewardComparision['A/B/n'].iplot(title="A/B/n Test Avg. Reward: {:.4f}".format(avgReward),xTitle='Impressions', yTitle='Avg. Reward')