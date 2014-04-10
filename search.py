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
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

def graphSearch(problem, strategy='dfs'):
    '''
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    print "Len of Children:", len(problem.getSuccessors(problem.getStartState()))
    print dir(problem)
    print problem.goal
    print problem.getStartState()
    print problem.walls
    '''
    # Initialize both the fringe & ExploredSet
    Fringe = util.PriorityQueue()
    ExploredSet = Set()

    # Add the start state to the explored set, and add the successors to the Fringe.
    init_successors = problem.getSuccessors(problem.getStartState())
    if(len(init_successors) == 0):
      print "Invalid Start State - no nodes to explore"
      return
    ExploredSet.add(problem.getStartState())

    # Add the successors of the root node to the fringe(as a list) with priority 0
    # For DFS, priorities would keep decreasing ( as we want ) to execute the lowest 
    # layer of the tree first. 
    for state in init_successors:
        Fringe.push(([state], 0),0)
    
    while(not Fringe.isEmpty()):
        # Get the highest(in this case lowest) item from PQueue, and view the last state in path
        pathToExplore = Fringe.pop()
        (state, direction, cost) = (pathToExplore[0])[-1]
        priority = pathToExplore[1]

        # If we're already in the goal state, we're done. 
        if(problem.isGoalState(state)):
            finalDirections = list(map(GetDirectionsFromFinalState,pathToExplore[0]))
            print "-------------Found a Solution---------------"
            print finalDirections
            print "Solution Length: ", len(finalDirections)
            return finalDirections

        # State add to the ExploredSet, so we won't be ever expanding any path with this state in it
        ExploredSet.add(state)

        # Get the rest of the successors of this node
        # and create paths to explore with those successors
        # as last node.  
        successorsToState = problem.getSuccessors(state)
        for successor in successorsToState:
            successorState = successor[0]
            if(successorState not in ExploredSet):
                NewPathsToExplore = copy.deepcopy(pathToExplore[0])
                NewPathsToExplore.append(successor)
                if(strategy == 'dfs'):
                    Fringe.push((NewPathsToExplore,priority-1),priority-1)
                elif(strategy == 'bfs'):
                    Fringe.push((NewPathsToExplore,priority+1),priority+1)
                

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
