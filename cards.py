from constants import *

class Card:
    def __init__(self, victoryPoints, bonus, cost, lvl):
        self.vp = victoryPoints
        self.bonus = bonus
        self.cost = cost
        self.lvl = lvl

    def show(self):
        print(f"{self.vp} {strColor(self.bonus)} [{bcolors.WHITE}{self.cost[WHITE]} {bcolors.BLUE}{self.cost[BLUE]} {bcolors.GREEN}{self.cost[GREEN]} {bcolors.RED}{self.cost[RED]} {bcolors.BLACK}{self.cost[BLACK]}{bcolors.RESET}]")

    def getShow(self):
        return f"{self.vp} {strColor(self.bonus)} [{bcolors.WHITE}{self.cost[WHITE]} {bcolors.BLUE}{self.cost[BLUE]} {bcolors.GREEN}{self.cost[GREEN]} {bcolors.RED}{self.cost[RED]} {bcolors.BLACK}{self.cost[BLACK]}{bcolors.RESET}]"
        
        
    def __str__(self):
        return "vp: {}, color: {}, cost: {}, lvl{}".format(self.vp, strColor(self.bonus), [str(self.cost[i]) + strColor(i) for i in range(0,5)], self.lvl)

   
