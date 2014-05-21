# bustersAgents.py
# ----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import random 
import inference

class BustersAgent:
  "An agent that tracks and displays its beliefs about ghost positions."
  
  def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None ):
    inferenceType = util.lookup(inference, globals())
    self.inferenceModules = [inferenceType(a) for a in ghostAgents]
    
  def registerInitialState(self, gameState):
    "Initializes beliefs and inference modules"
    import __main__
    self.display = __main__._display
    for inference in self.inferenceModules: inference.initialize(gameState)
    self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
    self.firstMove = True
    
  def observationFunction(self, gameState):
    "Removes the ghost states from the gameState"
    agents = gameState.data.agentStates
    gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
    return gameState

  def getAction(self, gameState):
    "Updates beliefs, then chooses an action based on updated beliefs."
    for index, inf in enumerate(self.inferenceModules):
      if not self.firstMove: inf.elapseTime(gameState)
      self.firstMove = False
      inf.observeState(gameState)
      self.ghostBeliefs[index] = inf.getBeliefDistribution()
    self.display.updateDistributions(self.ghostBeliefs)
    return self.chooseAction(gameState)

  def chooseAction(self, gameState):
    "By default, a BustersAgent just stops.  This should be overridden."
    return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
  "An agent controlled by the keyboard that displays beliefs about ghost positions."
  
  def __init__(self, index = 0, inference = "ExactInference", ghostAgents = None):
    KeyboardAgent.__init__(self, index)
    BustersAgent.__init__(self, index, inference, ghostAgents)
    
  def getAction(self, gameState):
    return BustersAgent.getAction(self, gameState)
    
  def chooseAction(self, gameState):
    return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions

class GreedyBustersAgent(BustersAgent):
  "An agent that charges the closest ghost."
  
  def registerInitialState(self, gameState):
    "Pre-computes the distance between every two points."
    BustersAgent.registerInitialState(self, gameState)
    self.distancer = Distancer(gameState.data.layout, False)
    
  def chooseAction(self, gameState):
    """
    First computes the most likely position of each ghost that 
    has not yet been captured, then chooses an action that brings 
    Pacman closer to the closest ghost (in maze distance!).
    
    To find the maze distance between any two positions, use:
    self.distancer.getDistance(pos1, pos2)
    
    To find the successor position of a position after an action:
    successorPosition = Actions.getSuccessor(position, action)
    
    livingGhostPositionDistributions, defined below, is a list of
    util.Counter objects equal to the position belief distributions
    for each of the ghosts that are still alive.  It is defined based
    on (these are implementation details about which you need not be
    concerned):

      1) gameState.getLivingGhosts(), a list of booleans, one for each
         agent, indicating whether or not the agent is alive.  Note
         that pacman is always agent 0, so the ghosts are agents 1,
         onwards (just as before).

      2) self.ghostBeliefs, the list of belief distributions for each
         of the ghosts (including ghosts that are not alive).  The
         indices into this list should be 1 less than indices into the
         gameState.getLivingGhosts() list.
     
    You may remove Directions.STOP from the list of available actions.
    """
    pacmanPosition = gameState.getPacmanPosition()
    legal = [a for a in gameState.getLegalPacmanActions() if a != Directions.STOP]
    livingGhosts = gameState.getLivingGhosts()
    livingGhostPositionDistributions = [beliefs for i,beliefs
                                        in enumerate(self.ghostBeliefs)
                                        if livingGhosts[i+1]]
    
    # Location == State 
    # First iterate over the livingghostdistributions and try to find the one
    # with highest prob, highest state
    possibleGhostPositions = [] 
    numGhostDists = len(livingGhostPositionDistributions) 
    # approximating the location of the cell, as well as the distance to it for
    # each ghost
    for idx in range(numGhostDists):
        maxProb = 0 
        maxProbState = None 
        # Find the max prob state for this ghost id
        for curProbState,curProb in livingGhostPositionDistributions[idx].items():
            if(curProb > maxProb):
                maxProb,maxProbState = curProb,curProbState
        # Store the (location,dist) for this ghostID. 
        dist_to_state = self.distancer.getDistance(maxProbState,pacmanPosition)
        possibleGhostPositions.append((maxProbState,dist_to_state))

    # possibleGhostPositions has k different maxprob,maxprobstates
    # Take the min to all of these ghosts and then go after it .. 
    minDist = float('inf') 
    minState = None
    for curState,curDist in possibleGhostPositions:
        if(curDist < minDist):
            minDist = curDist 
            minState = curState

    # Now you have the best dist/best state
    # Choose an action which minizes the distance to the location chosen
    # previously
    minAction = None 
    minDist = float('inf') 
    candidates = []
    for successorAction in legal:
        nextPos = Actions.getSuccessor(pacmanPosition, successorAction)
        pacmanDistToNextPos = self.distancer.getDistance(minState,nextPos)
        if(pacmanDistToNextPos < minDist):
            minDist = pacmanDistToNextPos
            candidates = [successorAction]
        elif(pacmanDistToNextPos == minDist):
            candidates.append(successorAction)
    return random.choice(candidates)
