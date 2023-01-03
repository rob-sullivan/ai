import numpy as np
import pandas as pd
import cufflinks as cf
import plotly.offline

class Advert():
    """
    class for simulating online adverts, where rewards come from a bernoulli distribution
    """
    def __init__(self, p):
        self.p = p
    
    def displayAd(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward

class ABNTesting():
    def __init__(self, ads):
        #A/B Testing
        nTest = 10000
        nProd = 90000
        nAds = len(ads)
        Q = np.zeros(nAds) # action values
        N = np.zeros(nAds) # total impressions

        totalReward = 0
        avgRewards = [] #save average reward over time

        #A/B/n test
        self.choose(ads, nTest,nAds, N, Q, avgRewards, nProd, totalReward)

    
    def choose(self, ads, nTest, nAds, N, Q, avgRewards, nProd, totalReward):
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


class EpsilonGreedy():
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
    ABNTesting(ads)


    adA = Advert(0.004)
    adB = Advert(0.016)
    adC = Advert(0.02)
    adD = Advert(0.028)
    adE = Advert(0.031)
    ads = [adA, adB, adC, adD, adE]
    EpsilonGreedy(ads)

