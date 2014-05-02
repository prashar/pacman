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
from game import Actions
from operator import add
from collections import Counter
from sets import Set

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
    smallestDistToPellet = min(distToNearestPellet)
    distToGhost = util.manhattanDistance(newPos, newGhostStates[0].configuration.pos)
    numPelletsRem = successorGameState.getFood().asList()
    #print len(numPelletsRem)
    if(distToGhost < 2):
        return -10000000

    pellet_dist_metric = (1./(smallestDistToPellet+1)**2)
    ghost_dist_metric = - (1./((distToGhost+1)**2))
    score_metric = (successorGameState.getScore()**2)

    #print "SM, ",score_metric
    #print "PDM, ",pellet_dist_metric
    #print "GDM, ",ghost_dist_metric
    #print "newScaredTimes, ",newScaredTimes

    metric = pellet_dist_metric + ghost_dist_metric + score_metric
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
            # Need this to maintain the action associated with the highest value
            if(result > v):
                v = result
                todo = action
        #print v
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
        # to maintain both alpha & beta. Here creating a list with
        # it's first value set to -inf, and the remaining values set to +inf
        # The remaining values are euqal to the number of num of ghosts.
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
        #print v
        return todo
    #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        #newGhostStates = gameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print newScaredTimes

        #print "---xxxx------"
        #PrintState(gameState)


        def minimizer(gameState,depth,agentID):
            v = 0.0
            all_legal_actions = gameState.getLegalActions(agentID)
            for action in all_legal_actions:
                successorState = gameState.generateSuccessor(agentID,action)
                v += SolveMinimax(successorState,depth+1)
            # This is expectimax - just add and divide by the # of actions.
            return (v/len(all_legal_actions))

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
        #print "----"
        for action in gameState.getLegalActions():
            result = SolveMinimax(gameState.generateSuccessor(0,action),1)
            #DEBUG
            #print result,action
            if(result > v):
                v = result
                todo = action
        return todo

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    ['__doc__', '__eq__', '__hash__', '__init__', '__module__', '__str__',
    'data', 'deepCopy', 'generatePacmanSuccessor', 'generateSuccessor',
    'getCapsules', 'getFood', 'getGhostPosition', 'getGhostPositions',
    'getGhostState', 'getGhostStates', 'getLegalActions',
    'getLegalPacmanActions', 'getNumAgents', 'getNumFood', 'getPacmanPosition',
    'getPacmanState', 'getScore', 'getWalls', 'hasFood', 'hasWall', 'initialize',
    'isLose', 'isWin']

    DESCRIPTION:

    Basically, my evaluation function is a linear combination of
    the following:


    1. Food Total Distance Calculating the total distance to all food pellets.
    2. Game Score - Cubing the game score, since it's very important we gravitate towards a state
    which warrants highest score.
    3. Power Pellets - Get to at least one power pellet ( to increase the score, so avg > 1000 )
    4. Distance to nearest pellet - find out which pellet is closest to you, and run a
    breadth first search from project 1 to find out the length of the optimal path to it.

    Exceptions:
    1. If the total distance to the ghost happens to be less than 2, return a very high
    negative utility, so you don't pick that state.
    2. If you get no food elements in the state, return high utility.

    """

    # Gather the new position we're in
    newPos = currentGameState.getPacmanPosition()
    # Gather all the food
    Food = currentGameState.getFood()
    # Save all the ghost states
    newGhostStates = currentGameState.getGhostStates()
    # Get the location of all power pellets
    capsules = currentGameState.getCapsules()
    # This is when the scared counter will be hit.
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # the location of all the ghosts
    ghostLocations = [ghostState.configuration.pos for ghostState in newGhostStates]

    # Distance to all food pellets.
    totalDistToAllPellets = 0

    # Question Specific Constants
    DEATHDISTANCE = 2
    COUNTERTHRESHOLD = 25

    # This condition returns the highest utility if we're nearing a power pellet.
    ScaredTimerCheck = lambda x,y: len(x) and (max(y) > COUNTERTHRESHOLD)
    AnyGhostsNearby = lambda x: x <= DEATHDISTANCE

    # Calculate the following:
    # a ) Distance to each food pellet.
    # b ) Smallest dist to a pellet.
    # c )  total dist to all pellets.
    # d ) BFS Length of Distance to the nearest pellet.
    if(len(Food.asList())):
        cnt = Counter()
        for location in Food.asList():
            cnt[location] = util.manhattanDistance(location, newPos)
        closest_food_pellet = min(cnt.iterkeys(),key=lambda key: cnt[key])
        #distToNearestPellet = cnt[closest_food_pellet]
        distToNearestPellet = mazeDistance(newPos, closest_food_pellet,currentGameState)
        totalDistToAllPellets = sum(cnt.itervalues())

    # distance to each ghost - call sum() for total distance.
    distToGhosts = map(lambda x: util.manhattanDistance(newPos, x), ghostLocations)

    # Use BFS to figure out the closest point, or else disregard all contributions
    # from capsule dist.
    if(ScaredTimerCheck(capsules,newScaredTimes)):
        if(AnyGhostsNearby(sum(distToGhosts))):
            return -10e11

        #capsuleDist = map(lambda x: util.manhattanDistance(newPos, x), capsules)
        capsuleDist = map(lambda x: mazeDistance(newPos, x,currentGameState), capsules)
        minCapDist = min(capsuleDist)
    else:
        minCapDist = 10e11

    # Take care of trivial cases like nearby ghosts, no food instances, etc

    # DO NOT DIE - if distance to any ghost is  <=2, don't make the move.
    if(AnyGhostsNearby(sum(distToGhosts))):
        return -10e11

    if(totalDistToAllPellets == 0):
        return 10e8 * currentGameState.getScore()

    # Total food distance
    food_dist_metric = (totalDistToAllPellets)

    # Nearest food pellet distance ( Manhattan )
    pellet_dist_metric = (10.*(distToNearestPellet+1)**2)

    # Total Ghost distance
    # Penalize for being close to ghosts.
    ghost_dist_metric = - (10./((sum(distToGhosts)+1)**2))

    # This matters a lot, because we need an avg score of a 1000.
    score_metric = (currentGameState.getScore()**3)

    # We need to eat that power pellet, so this must activate
    # it's influence in two steps from the depth.
    capdist_metric = 10e7 /(minCapDist+1)

    metric = - food_dist_metric - pellet_dist_metric - ghost_dist_metric + score_metric + capdist_metric
    return metric

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
        Your agent for the mini-contest
    """
    def getAction(self, gameState):
        # Using the same evaluation function & expectimax with depth = 3.

        def minimizer(gameState,depth,agentID):
            v = 0.0
            all_legal_actions = gameState.getLegalActions(agentID)
            for action in all_legal_actions:
                successorState = gameState.generateSuccessor(agentID,action)
                v += SolveMinimax(successorState,depth+1)
            # This is expectimax - just add and divide by the # of actions.
            return (v/len(all_legal_actions))

        def maximizer(gameState,depth,agentID):
            v = -float('inf')
            for action in gameState.getLegalActions(agentID):
                successorState = gameState.generateSuccessor(agentID,action)
                v = max(v,SolveMinimax(successorState,depth+1))
            return v

        def SolveMinimax(gameState,depth):
            pacmanOrGhost = (depth % numAgents)
            if(gameState.isWin() or gameState.isLose()):
                return betterEvaluationFunction(gameState)
            if(depth == singlePly):
                return betterEvaluationFunction(gameState)
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
            return betterEvaluationFunction(gameState)
        # Solve minimax for each action
        todo = None
        #print "----"
        for action in gameState.getLegalActions():
            result = SolveMinimax(gameState.generateSuccessor(0,action),1)
            #DEBUG
            #print result,action
            if(result > v):
                v = result
                todo = action
        return todo

class SearchProblem:

    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        util.raiseNotDefined()

class PositionSearchProblem(SearchProblem):

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):

        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = False
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
          print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
         isGoal = state == self.goal

         # For display purposes only
         if isGoal:
           self._visitedlist.append(state)
           import __main__
           if '_display' in dir(__main__):
             if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
               __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

         return isGoal

    def getSuccessors(self, state):

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
          x,y = state
          dx, dy = Actions.directionToVector(action)
          nextx, nexty = int(x + dx), int(y + dy)
          if not self.walls[nextx][nexty]:
            nextState = (nextx, nexty)
            cost = self.costFn(nextState)
            successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
          self._visited[state] = True
          self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
          # Check figure out the next state and see whether its' legal
          dx, dy = Actions.directionToVector(action)
          x, y = int(x + dx), int(y + dy)
          if self.walls[x][y]: return 999999
          cost += self.costFn((x,y))
        return cost

'''
----------------------
----------------------
Code from Project # 1 ( Used for my last question )
----------------------
----------------------
'''

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return len(graphSearch(prob,'bfs'))

class GraphNode:
    def __init__(self,problem,strategy,state,cost,path,heuresticFn,hrValue):
        self.heuristic = heuresticFn
        self.problem = problem
        self.strategy = strategy
        self.pathToNode = path
        self.state = state
        self.cost = cost
        self.hrValue = hrValue

    def GetChildNodes(self):
        '''
            Returns a list of GraphNodes that are children
            of this node.
        '''
        listOfChildren = self.problem.getSuccessors(self.state)
        childrenOfThisNode = []
        for child in listOfChildren:
            state,direction,cost = child
            # Since we're using Priority Queue, we must adjust
            # the cost based on the strategy.
            hrValue = 0
            if(self.strategy == 'bfs' or self.strategy == 'ucs'):
                cost += self.cost
            elif(self.strategy == 'dfs'):
                # Priority Queue uses min heap - means I need to add elements
                # with lower priority.
                cost = self.cost - cost
            elif(self.strategy == 'astar'):
                cost += self.cost
                hrValue = self.heuristic(state,self.problem)
            # We'll keep a list of directions to get to this child
            # graph node in here.
            directionList = list(self.pathToNode)
            directionList.append(direction)
            childNode = GraphNode(self.problem,self.strategy,state,cost,directionList,self.heuristic,hrValue)
            childrenOfThisNode.append(childNode)
        return childrenOfThisNode

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

# Common routine which adjusts the general strategy based on the type of search
# we are executing ..
def graphSearch(problem, strategy='dfs',heuristic=nullHeuristic):

    # Algorithm taken from page 77 of the textbook

    # Initialize both the fringe & ExploredSet
    Fringe = util.PriorityQueue()
    ExploredSet = Set()

    # Addin the root node, and assigning prio 0 ( base prio )
    startState = problem.getStartState()
    graphRoot = GraphNode(problem,strategy,startState,0,[],heuristic,0)
    if(strategy == 'astar'):
        Fringe.push(graphRoot,graphRoot.cost+graphRoot.hrValue)
    else:
        Fringe.push(graphRoot,graphRoot.cost)

    while(not Fringe.isEmpty()):
        # Get the highest(in this case lowest) item from PQueue, and view the last state in path
        graphNode = Fringe.pop()
        # If we're already in the goal state, we're done.
        if(problem.isGoalState(graphNode.state)):
            finalDirections = graphNode.pathToNode
            #print "-----Found a Solution------"
            #print "Solution Length: ", len(finalDirections)
            #print finalDirections
            return finalDirections

        # State add to the ExploredSet, so we won't be ever expanding any path with this state in it
        if(graphNode.state not in ExploredSet):
            ExploredSet.add(graphNode.state)

            # Get the rest of the successors of this node
            # and create paths to explore with those successors
            # as last node.
            successorsToState = graphNode.GetChildNodes()
            for successor in successorsToState:
                Fringe.push(successor,successor.cost+successor.hrValue)