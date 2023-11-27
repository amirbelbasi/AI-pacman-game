# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        res = 1
        currentPos = currentGameState.getPacmanPosition()
        
        disToGhosts = []
        for gs in newGhostStates:
            manDisGhost = manhattanDistance(gs.getPosition(), newPos)
            disToGhosts.append(manDisGhost)
        minDisGhost = min(disToGhosts)
        if minDisGhost < 2: return 0

        disToFoods = []
        for f in currentGameState.getFood().asList():
            manDisFood = manhattanDistance(f, currentPos)
            disToFoods.append(manDisFood)
        minDisFood = min(disToFoods)

        newDisToFoods = []
        for f in newFood.asList():
            manDisFood = manhattanDistance(f, newPos)
            newDisToFoods.append(manDisFood)
        if len(newDisToFoods) == 0: minNewDisToFoods = 0
        else: minNewDisToFoods = min(newDisToFoods)

        if successorGameState.getScore() > currentGameState.getScore(): res = 1000
        elif minNewDisToFoods < minDisFood: res = 100
        elif sum(newScaredTimes) > 0: res = 10
        return res

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimizer(agentIndex, state, depth):
            v = float('inf')
            if state.isLose() or state.isWin(): return self.evaluationFunction(state)
            for a in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, a)
                if state.getNumAgents()-1 == agentIndex: v = min(v, maximizer(successorState, depth))
                else: v = min(v, minimizer(agentIndex + 1, successorState, depth))
            return v
        
        def maximizer(state, depth):
            v = float('-inf')
            if depth+1 == self.depth or state.isLose() or state.isWin(): return self.evaluationFunction(state)
            for a in state.getLegalActions(0):
                successorState = state.generateSuccessor(0, a)
                v = max(v, minimizer(1, successorState, depth+1))
            return v
        
        currentScore = float('-inf')
        for a in gameState.getLegalActions(0):
            newGameState = gameState.generateSuccessor(0, a)
            bestScore = minimizer(1, newGameState, 0)
            if bestScore > currentScore:
                res = a
                currentScore = bestScore
        return res

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimizer(agentIndex, state, depth, alpha, beta):
            v = float('inf')
            if state.isLose() or state.isWin(): return self.evaluationFunction(state)
            betaHolder = beta
            for a in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, a)
                if state.getNumAgents()-1 == agentIndex:
                    v = min(v, maximizer(successorState, depth, alpha, betaHolder))
                    if v < alpha: return v
                    betaHolder = min(betaHolder, v)
                else:
                    v = min(v, minimizer(agentIndex + 1, successorState, depth, alpha, betaHolder))
                    if v < alpha: return v
                    betaHolder = min(betaHolder, v)
            return v
        
        def maximizer(state, depth, alpha, beta):
            v = float('-inf')
            if depth+1 == self.depth or state.isLose() or state.isWin(): return self.evaluationFunction(state)
            alphaHolder = alpha
            for a in state.getLegalActions(0):
                successorState = state.generateSuccessor(0, a)
                v = max(v, minimizer(1, successorState, depth+1, alphaHolder, beta))
                if v > beta: return v
                alphaHolder = max(alphaHolder, v)
            return v
        
        alpha = float('-inf')
        beta = float('inf')
        currentScore = float('-inf')
        for a in gameState.getLegalActions(0):
            newGameState = gameState.generateSuccessor(0, a)
            bestScore = minimizer(1, newGameState, 0, alpha, beta)
            if bestScore > currentScore:
                res = a
                currentScore = bestScore
            if bestScore > beta: return res
            alpha = max(alpha, bestScore)
        return res

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expecti(agentIndex, state, depth):
            sumV = 0
            if state.isLose() or state.isWin(): return self.evaluationFunction(state)
            for a in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, a)
                if state.getNumAgents()-1 == agentIndex: v = maximizer(successorState, depth)
                else: v = expecti(agentIndex + 1, successorState, depth)
                sumV += v
            if not len(state.getLegalActions(agentIndex)): return  0
            res = sumV / len(state.getLegalActions(agentIndex))
            return res
        
        def maximizer(state, depth):
            v = float('-inf')
            if depth+1 == self.depth or state.isLose() or state.isWin(): return self.evaluationFunction(state)
            for a in state.getLegalActions(0):
                successorState = state.generateSuccessor(0, a)
                v = max(v, expecti(1, successorState, depth+1))
            return v
        
        currentScore = float('-inf')
        for a in gameState.getLegalActions(0):
            newGameState = gameState.generateSuccessor(0, a)
            bestScore = expecti(1, newGameState, 0)
            if bestScore > currentScore:
                res = a
                currentScore = bestScore
        return res

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    res = 0

    disToFoods = []
    for f in foods.asList():
        manDisFood = manhattanDistance(f, pacmanPosition)
        disToFoods.append(manDisFood)
    if not disToFoods: minDisFood = 1
    else: minDisFood = min(disToFoods)
    res += 1 / float(minDisFood)

    disToGhosts = []
    for g in ghostPositions:
        manDisGhost = manhattanDistance(g, pacmanPosition)
        disToGhosts.append(manDisGhost)
    if not disToGhosts: minDisGhost = 0
    else: minDisGhost = min(disToGhosts)
    res += float(minDisGhost) / 10

    res += sum(scaredTimers)
    res += currentGameState.getScore()
    return res

# Abbreviation
better = betterEvaluationFunction