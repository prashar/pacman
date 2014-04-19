# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from sets import Set
import copy


class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).

  You do not need to change anything in this class, ever.
  """

  def getStartState(self):
     """
     Returns the start state for the search problem
     """
     util.raiseNotDefined()

  def isGoalState(self, state):
     """
       state: Search state

     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state

     For a given state, this should return a list of triples,
     (successor, action, stepCost), where 'successor' is a
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take

     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    return graphSearch(problem)

def breadthFirstSearch(problem):
    return graphSearch(problem,'bfs')

def uniformCostSearch(problem):
    return graphSearch(problem,'ucs')

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    return graphSearch(problem,'astar',heuristic)

'''
Each Node in the search problem gets represented as a GraphNode.
which maintains everything I need about the node.
'''
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
            print "-----Found a Solution------"
            print "Solution Length: ", len(finalDirections)
            print finalDirections
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

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch