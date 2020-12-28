from constants import *

class Character:
    def __init__(self, victoryPoints, cost):
        self.vp = victoryPoints
        self.cost = cost

    def __str__(self):
        return "vp: {}, cost: {}".format(self.vp, [str(self.cost[i]) + strColor(i) for i in range(0,5)])

    def __repr__(self):
        return "vp: {}, cost: {}".format(self.vp, [str(self.cost[i]) + strColor(i) for i in range(0,5)])
    
    def getShow(self):
        return f"{self.vp} [{bcolors.WHITE}{self.cost[WHITE]} {bcolors.BLUE}{self.cost[BLUE]} {bcolors.GREEN}{self.cost[GREEN]} {bcolors.RED}{self.cost[RED]} {bcolors.BLACK}{self.cost[BLACK]}{bcolors.RESET}]"

    def __eq__(self, other):
        return other != None and self.vp == other.vp and self.cost == other.cost

    def __ne__(self, other):
        return not self.__eq__(other)
