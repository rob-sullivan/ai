import numpy as np
import pandas as pd
#import cufflinks as cf
#from plotly.offline import iplot
#import plotly
#import plotly.graph_objs as go
import matplotlib.pyplot as plt

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
        self.nTest = 10000
        self.nProd = 90000
        self.nAds = len(ads)
        self.Q = np.zeros(self.nAds) # action values
        self.N = np.zeros(self.nAds) # total impressions

        self.totalReward = 0
        self.avgRewards = [] #save average reward over time

        #A/B/n test
        self.choose(ads)

    
    def choose(self, ads):
        for i in range(self.nTest):
            adChosen = np.random.randint(self.nAds)
            R = ads[adChosen].displayAd() # observe reward
            self.N[adChosen] += 1
            self.Q[adChosen] += (1/self.N[adChosen]) * (R-self.Q[adChosen])
            self.totalReward += R
            avgReward = self.totalReward / (i+1)
            self.avgRewards.append(avgReward)

        bestAd = np.argmax(self.Q) #find best action
        print("Best Performing Ad: " + chr(ord('A') + bestAd))

        adChosen = bestAd
        for i in range(self.nProd):
            R = ads[adChosen].displayAd()
            self.totalReward += R
            avgReward = self.totalReward / (i+1)
            self.avgRewards.append(avgReward)

        dfRewardComparision = pd.DataFrame(self.avgRewards, columns=['A/B/n'])

        #cf.go_offline() #will make cufflinks offline to avoid Authentication credentials not provided error
        #cf.set_config_file(offline=False, world_readable=True, theme='white')

        #print(dfRewardComparision['A/B/n'])

        self.plotGraph(dfRewardComparision, avgReward) #will open plot in a webbrowser

    def plotGraph(self, data, avgReward):
        fig = data['A/B/n'].plot(
            title="A/B/n Test Avg. Reward: {:.4f}".format(avgReward),
            xlabel='Impressions', 
            ylabel='Avg. Reward', 
            color='r')
        #fig.set_ylim([0,0.04])
        fig.set_xticks(np.arange(0, 100000, step=10000))
        fig
        plt.show()

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
    #EpsilonGreedy(ads)

