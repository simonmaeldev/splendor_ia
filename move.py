class Move:
    """ Define what a move is
    """

    def __init__(self, actionType, action, tokensToRemove, characterSelected):
        self.actionType = actionType # can be BUILD, RESERVE, TAKE2, TAKE3
        self.action = action # if build or reserve : [[lvl][pos]], if take tokens : list of tokens
        self.tokensToRemove = tokensToRemove # tokens the player will remove at the end of his turn if he have too many of them
        self.character = characterSelected # the noble the player will take if he can take one
