# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
from math import *
from copy import * 

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # 1/Dist to the nearest pellet + dist to ghost.
    distToNearestPellet = [util.manhattanDistance(newPos, item) for item in oldFood.asList()]
    smallestDist = min(distToNearestPellet)
    distToGhost = util.manhattanDistance(newPos, newGhostStates[0].configuration.pos)
    numPelletsRem = successorGameState.getFood().asList()
    #print len(numPelletsRem)
    if(distToGhost < 2):
        return -100000000000
    metric = (1./(smallestDist+1)**2) - (1./((distToGhost+1)**2)) + (successorGameState.getScore()**2)
    return metric

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

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # Objective is to return the the best possible action to pacman
        # considering the depth of our minmax tree. A single ply is one
        # move for pacman, and one for each ghost, so we have to multiply
        # both to get the total. 
        def minimizer(gameState,depth,agentID):
            v = float('inf')
            for action in gameState.getLegalActions(agentID):
                successorState = gameState.generateSuccessor(agentID,action)
                v = min(v,SolveMinimax(successorState,depth+1))
            return v

        def maximizer(gameState,depth,agentID):
            v = -float('inf')
            for action in gameState.getLegalActions(agentID):
                successorState = gameState.generateSuccessor(agentID,action)
                v = max(v,SolveMinimax(successorState,depth+1))
            return v

        def SolveMinimax(gameState,depth):
            pacmanOrGhost = (depth % numAgents)
            if(gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            if(depth == singlePly):
                return self.evaluationFunction(gameState)
            if(pacmanOrGhost == 0):
                # PACMAN
                return maximizer(gameState,depth,pacmanOrGhost)
            else:
                # GHOST
                return minimizer(gameState,depth,pacmanOrGhost)
        
        # We'll start with depth=1, so pacman always moves first.
        numAgents = gameState.getNumAgents()
        StopDepth = self.depth
        singlePly = numAgents * StopDepth

        # Return the max for each action
        v = -float('inf')
        # No point evaluating further - just return utility. 
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        # Solve minimax for each action
        todo = None
        for action in gameState.getLegalActions():
            result = SolveMinimax(gameState.generateSuccessor(0,action),1)
            if(result > v):
                v = result
                todo = action
        return todo

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """ 
        "*** YOUR CODE HERE ***"
        def minimizer(gameState,depth,agentID,alphaBeta):
            v = float('inf')
            local_alphaBeta = copy(alphaBeta)
            for action in gameState.getLegalActions(agentID):
                successorState = gameState.generateSuccessor(agentID,action)
                v = min(v,SolveMinimax(successorState,depth+1,local_alphaBeta))
                # Alpha already has a higher value available from parent to root. 
                # return now. 
                if(v < local_alphaBeta[0]):
                    return v
                local_alphaBeta[agentID] = min(local_alphaBeta[agentID],v)
            return v

        def maximizer(gameState,depth,agentID,alphaBeta):
            v = -float('inf')
            local_alphaBeta = copy(alphaBeta)
            for action in gameState.getLegalActions(agentID):
                successorState = gameState.generateSuccessor(agentID,action)
                v = max(v,SolveMinimax(successorState,depth+1,local_alphaBeta))
                # Largest value available to the maximizer is more than the 
                # best option to root from minimizer
                if(v > min(local_alphaBeta[1:])):
                    return v
                local_alphaBeta[agentID] = max(local_alphaBeta[agentID],v)
            return v

        def SolveMinimax(gameState,depth,alphaBeta):
            pacmanOrGhost = (depth % numAgents)
            if(gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            if(depth == singlePly):
                return self.evaluationFunction(gameState)
            if(pacmanOrGhost == 0):
                # PACMAN
                return maximizer(gameState,depth,pacmanOrGhost,alphaBeta)
            else:
                # GHOST
                return minimizer(gameState,depth,pacmanOrGhost,alphaBeta)
        
        # We'll start with depth=1, so pacman always moves first.
        numAgents = gameState.getNumAgents()
        StopDepth = self.depth
        singlePly = numAgents * StopDepth
        # We'll need some sort of a list/array like structure
        # to maintain both alpha & beta
        startAlphaBeta = [-float('inf')] + (numAgents-1)*[float('inf')]

        # Check for each action
        v = -float('inf')
        # No point evaluating further - just return utility. 
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        # Solve minimax for each action
        todo = None
        for action in gameState.getLegalActions():
            result = SolveMinimax(gameState.generateSuccessor(0,action),1,startAlphaBeta)
            if(result > v):
                v = result
                todo = action
            # MUST UPDATE THE ROOT NODE. 
            startAlphaBeta[0] = max(startAlphaBeta[0],v)
        return todo
    #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def minimizer(gameState,depth,agentID):
            v = 0.0
            for action in gameState.getLegalActions(agentID):
                successorState = gameState.generateSuccessor(agentID,action)
                v += SolveMinimax(successorState,depth+1)
            return (v/len(gameState.getLegalActions(agentID)))

        def maximizer(gameState,depth,agentID):
            v = -float('inf')
            for action in gameState.getLegalActions(agentID):
                successorState = gameState.generateSuccessor(agentID,action)
                v = max(v,SolveMinimax(successorState,depth+1))
            return v

        def SolveMinimax(gameState,depth):
            pacmanOrGhost = (depth % numAgents)
            if(gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            if(depth == singlePly):
                return self.evaluationFunction(gameState)
            if(pacmanOrGhost == 0):
                # PACMAN
                return maximizer(gameState,depth,pacmanOrGhost)
            else:
                # GHOST
                return minimizer(gameState,depth,pacmanOrGhost)
        
        # We'll start with depth=1, so pacman always moves first.
        numAgents = gameState.getNumAgents()
        StopDepth = self.depth
        singlePly = numAgents * StopDepth

        # Return the max for each action
        v = -float('inf')
        # No point evaluating further - just return utility. 
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        # Solve minimax for each action
        todo = None
        for action in gameState.getLegalActions():
            result = SolveMinimax(gameState.generateSuccessor(0,action),1)
            if(result > v):
                v = result
                todo = action
        return todo

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

