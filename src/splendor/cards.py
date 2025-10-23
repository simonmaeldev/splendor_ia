from typing import List
from .constants import *

class Card:
    def __init__(self, victoryPoints: int, bonus: int, cost: List[int], lvl: int) -> None:
        self.vp: int = victoryPoints
        self.bonus: int = bonus
        self.cost: List[int] = cost
        self.lvl: int = lvl
        self.visible: bool = False

    def isVisible(self) -> bool:
        return self.visible

    def setVisible(self) -> None:
        self.visible = True

    def show(self) -> None:
        print(f"{self.vp} {strColor(self.bonus)} [{bcolors.WHITE}{self.cost[WHITE]} {bcolors.BLUE}{self.cost[BLUE]} {bcolors.GREEN}{self.cost[GREEN]} {bcolors.RED}{self.cost[RED]} {bcolors.BLACK}{self.cost[BLACK]}{bcolors.RESET}]")

    def getShow(self) -> str:
        return f"{self.vp} {strColor(self.bonus)} [{bcolors.WHITE}{self.cost[WHITE]} {bcolors.BLUE}{self.cost[BLUE]} {bcolors.GREEN}{self.cost[GREEN]} {bcolors.RED}{self.cost[RED]} {bcolors.BLACK}{self.cost[BLACK]}{bcolors.RESET}]"


    def __str__(self) -> str:
        return self.getShow()

    def __repr__(self) -> str:
        return "|" + self.getShow() + "|"

    def __eq__(self, other: object) -> bool:
        return type(self) == type(other) and self.vp == other.vp and self.bonus == other.bonus and self.cost == other.cost and self.lvl == other.lvl

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
