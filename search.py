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

class Node():
    """
    A container storing the current state of a node, the list
    of  directions that need to be followed from the start state to
    get to the current state and the specific problem in which the
    node will be used.
    """
    def __init__(self, state, path, cost=0, heuristic=0, problem=None):
        self.state = state
        self.path = path
        self.cost = cost
        self.heuristic = heuristic
        self.problem = problem

    def __str__(self):
        string = "Current State: "
        string += __str__(self.state)
        string += "\n"
        string == "Path: " + self.path + "\n"
        return string

    def getSuccessors(self, heuristicFunction=None):
        children = []
        for successor in self.problem.getSuccessors(self.state):
            state = successor[0]
            path = list(self.path)
            path.append(successor[1])
            cost = self.cost + successor[2]
            if heuristicFunction:
                heuristic = heuristicFunction(state, self.problem)
            else:
                heuristic = 0
            node = Node(state, path, cost, heuristic, self.problem)
            children.append(node)
        return children


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

'''
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    closed = set()
    fringe = util.Queue()

    startNode = Node(problem.getStartState(), [], 0, 0, problem)
    fringe.push(startNode)

    while True:
      if fringe.isEmpty():
        return False
      node = fringe.pop()
      print node.state
      if problem.isGoalState(node.state):
        print len(closed)
        return node.path
      if node.state not in closed:
        closed.add(node.state)
        for childNode in node.getSuccessors():
            fringe.push(childNode)
'''

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

'''
class GraphNode:
    def __init__(self,problem,strategy,state,cost,path):
        self.problem = problem
        self.strategy = strategy
        self.pathToNode = path
        self.state = state
        self.cost = cost

    def GetChildNodes(self):
        '''
            Returns a list of GraphNodes that are children
            of this node.
        '''
        childrenOfThisNode = []
        listOfChildren = self.problem.getSuccessors(self.state)
        for child in listOfChildren:
            state,direction,cost = child
            # Since we're using Priority Queue, we must adjust
            # the cost based on the strategy.
            if(self.strategy == 'bfs'):
                cost += self.cost
            elif(self.strategy == 'dfs'):
                cost = self.cost - cost
            # We'll keep a list of directions to get to this child
            # graph node in here.
            directionList = list(self.pathToNode)
            directionList.append(direction)
            childNode = GraphNode(self.problem,self.strategy,state,cost,directionList)
            childrenOfThisNode.append(childNode)
        return childrenOfThisNode



def graphSearch(problem, strategy='dfs',heuristic=nullHeuristic):

    # Algorithm taken from page 77 of the textbook

    # Initialize both the fringe & ExploredSet
    Fringe = util.PriorityQueue()
    ExploredSet = Set()

    # Addin the root node, and assigning prio 0 ( base prio )
    startState = problem.getStartState()
    graphRoot = GraphNode(problem,strategy,startState,0,[])
    Fringe.push(graphRoot,0)

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
                Fringe.push(successor,successor.cost)

# Utility function to parse out which states to go to
def GetDirectionsFromFinalState(sourceToDestinationPath):
  from game import Directions
  #directionDictionary = {'South':Directions.SOUTH,'North':Directions.NORTH,'East':Directions.EAST,'West':Directions.WEST}
  return sourceToDestinationPath[1]

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
