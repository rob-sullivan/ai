import numpy as np

class Bandit():
    """
    class for single slot machine where rewards are gaussian
    """
    def __init__(self, mean=0, stdev=1):
        self.mean = mean
        self.stdev = stdev

    def pullLever(self):
        reward = np.random.normal(self.mean, self.stdev)
        return np.round(reward, 1)


class Casino():
    def __init__(self, bandits):
        self.bandits = bandits
        np.random.shuffle(self.bandits)
        self.resetGame()

    def resetGame(self):
        self.rewards = []
        self.totalReward = 0
        self.nPlayed = 0

    def play(self, choice):
        reward = self.bandits[choice - 1].pullLever()
        self.rewards.append(reward)
        self.totalReward += reward
        self.nPlayed += 1
        return reward

    def gamblerPlay(self):
        self.resetGame()
        print("Game Started. Enter 0 to quit")

        while True:
            print("-- Round: " + str(self.nPlayed))
            choice = int(input("Choose a machine from 1 to " + str(len(self.bandits)) + ": "))

            if choice in range(1, len(self.bandits) + 1):
                reward = self.play(choice)
                print("Machine " + str(choice) + " gave a reward of: " + str(reward))
                avgReward = self.totalReward/self.nPlayed
                print("Average Reward is " + str(avgReward))
            else:
                break
        print("\nGame has ended")
        if self.nPlayed > 0:
            print("Total reward is " + str(self.totalReward) + "after " + str(self.nPlayed) + " round(s).")
            avgReward = self.totalReward/self.nPlayed
            print("Average reward is " + str(avgReward) + ".")



if __name__ == "__main__":
    #g1 = Bandit(5, 2)
    #g2 = Bandit(6, 2)
    #g3 = Bandit(1, 5)

    #g1.pull_lever()

    slotA = Bandit(5, 3)
    slotB = Bandit(6, 2)
    slotC = Bandit(1, 5)

    game = Casino([slotA, slotB, slotC])

    game.gamblerPlay()