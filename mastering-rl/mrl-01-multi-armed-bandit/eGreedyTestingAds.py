import numpy as np
import pandas as pd
import cufflinks as cf
import plotly.offline
cf.go_offline() # required to use plotly offline (no account required).
cf.set_config_file(world_readable=True, theme="white")
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"

class Advert():
    """
    class for ad where rewards come from a bernoulli distribution
    """
    def __init__(self, p):
        self.p = p
    
    def displayAd(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward

class epsilonGreedy():
    def __init__(self, a):
        self.ads = a
        self.dfRewardComparision = pd.DataFrame([])

        #run tests, start with a high ε value and gradually reduce it
        self.choose(0.2)
        self.choose(0.1)
        self.choose(0.05)
        self.choose(0.01)

        greedy_list = ['e-greedy: 0.01', 'e-greedy: 0.05','e-greedy: 0.1', 'e-greedy: 0.2']
        print(self.dfRewardComparision.head())
        self.dfRewardComparision[greedy_list].iplot(title="ε-Greedy Actions", dash=['solid', 'dash', 'dashdot', 'dot'], xTitle='Impressions', yTitle='Avg. Reward').show(renderer="png")

    #epsilon greedy
    def choose(self, eps):
        nProd = 100000
        nAds = len(self.ads)
        Q = np.zeros(nAds) # action values
        N = np.zeros(nAds) # total impressions
        totalReward = 0
        avgRewards = [] #save average reward over time

        adChosen = np.random.randint(nAds)
        for i in range(nProd):
            R = self.ads[adChosen].displayAd()
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

        self.dfRewardComparision['e-greedy: {}'.format(eps)] = avgRewards

if __name__ == "__main__":
    adA = Advert(0.004)
    adB = Advert(0.016)
    adC = Advert(0.02)
    adD = Advert(0.028)
    adE = Advert(0.031)

    ads = [adA, adB, adC, adD, adE]

    epsilonGreedy(ads)