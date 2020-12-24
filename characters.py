from constants import *

class Character:
    def __init__(self, victoryPoints, cost):
        self.vp = victoryPoints
        self.cost = cost

    def __str__(self):
        return "vp: {}, cost: {}".format(self.vp, [str(self.cost[i]) + strColor(i) for i in range(0,5)])
