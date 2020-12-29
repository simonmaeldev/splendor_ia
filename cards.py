from constants import *

class Card:
    def __init__(self, victoryPoints, bonus, cost, lvl):
        self.vp = victoryPoints
        self.bonus = bonus
        self.cost = cost
        self.lvl = lvl
        self.visible = False

    def isVisible(self):
        return self.visible

    def setVisible(self):
        self.visible = True

    def show(self):
        print(f"{self.vp} {strColor(self.bonus)} [{bcolors.WHITE}{self.cost[WHITE]} {bcolors.BLUE}{self.cost[BLUE]} {bcolors.GREEN}{self.cost[GREEN]} {bcolors.RED}{self.cost[RED]} {bcolors.BLACK}{self.cost[BLACK]}{bcolors.RESET}]")

    def getShow(self):
        return f"{self.vp} {strColor(self.bonus)} [{bcolors.WHITE}{self.cost[WHITE]} {bcolors.BLUE}{self.cost[BLUE]} {bcolors.GREEN}{self.cost[GREEN]} {bcolors.RED}{self.cost[RED]} {bcolors.BLACK}{self.cost[BLACK]}{bcolors.RESET}]"
        
        
    def __str__(self):
        return self.getShow()

    def __repr__(self):
        return "|" + self.getShow() + "|"

    def __eq__(self, other):
        return type(self) == type(other) and self.vp == other.vp and self.bonus == other.bonus and self.cost == other.cost and self.lvl == other.lvl and self.visible == other.visible

    def __ne__(self, other):
        return not self.__eq__(other)
