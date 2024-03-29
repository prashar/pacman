# inference.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import random
import busters
import game

class InferenceModule:
  """
  An inference module tracks a belief distribution over a ghost's location.
  This is an abstract class, which you should not modify.
  """

  ############################################
  # Useful methods for all inference modules #
  ############################################

  def __init__(self, ghostAgent):
    "Sets the ghost agent for later access"
    self.ghostAgent = ghostAgent
    self.index = ghostAgent.index

  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.

    You must first place the ghost in the gameState, using setGhostPosition below.
    """
    ghostPosition = gameState.getGhostPosition(self.index) # The position you set
    actionDist = self.ghostAgent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist

  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index] = game.AgentState(conf, False)
    return gameState

  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getNoisyGhostDistances()
    if len(distances) >= self.index: # Check for missing observations
      obs = distances[self.index - 1]
      self.observe(obs, gameState)

  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.initializeUniformly(gameState)

  ######################################
  # Methods that need to be overridden #
  ######################################

  def initializeUniformly(self, gameState):
    "Sets the belief state to a uniform prior belief over all positions."
    pass

  def observe(self, observation, gameState):
    "Updates beliefs based on the given distance observation and gameState."
    pass

  def elapseTime(self, gameState):
    "Updates beliefs for a time step elapsing from a gameState."
    pass

  def getBeliefDistribution(self):
    """
    Returns the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence so far.
    """
    pass

class ExactInference(InferenceModule):
  """
  The exact dynamic inference module should use forward-algorithm
  updates to compute the exact belief function at each time step.
  """

  def initializeUniformly(self, gameState):
    "Begin with a uniform distribution over ghost positions."
    self.beliefs = util.Counter()
    for p in self.legalPositions: self.beliefs[p] = 1.0
    self.beliefs.normalize()

  def observe(self, observation, gameState):
    """
    Updates beliefs based on the distance observation and Pacman's position.

    The noisyDistance is the estimated manhattan distance to the ghost you are tracking.

    The emissionModel below stores the probability of the noisyDistance for any true
    distance you supply.  That is, it stores P(noisyDistance | TrueDistance).

    self.legalPositions is a list of the possible ghost positions (you
    should only consider positions that are in self.legalPositions).
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getPacmanPosition()

    # Replace this code with a correct observation update
    updatedBeliefs = util.Counter()
    if(noisyDistance != 999):
      for p in self.legalPositions:
        trueDistance = util.manhattanDistance(p, pacmanPosition)
        if emissionModel[trueDistance] > 0:
          updatedBeliefs[p] = emissionModel[trueDistance] * self.beliefs[p]
    else:
      # Must take care of this base case - as autograder seems to ask for it.
      # Setting the probability of the ghost jail cell to be 1 and rest to be 0
      ghostJailPos = (2 * self.ghostAgent.index - 1, 1)
      updatedBeliefs[ghostJailPos] = 1.0

    # MUST normalize !
    updatedBeliefs.normalize()

    self.beliefs = updatedBeliefs

  def elapseTime(self, gameState):
    """
    Update self.beliefs in response to a time step passing from the current state.

    The transition model is not entirely stationary: it may depend on Pacman's
    current position (e.g., for DirectionalGhost).  However, this is not a problem,
    as Pacman's current position is known.

    In order to obtain the distribution over new positions for the
    ghost, given its previous position (oldPos) as well as Pacman's
    current position, use this line of code:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    Note that you may need to replace "oldPos" with the correct name
    of the variable that you have used to refer to the previous ghost
    position for which you are computing this distribution.

    newPosDist is a util.Counter object, where for each position p in self.legalPositions,

    newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

    (and also given Pacman's current position).  You may also find it useful to loop over key, value pairs
    in newPosDist, like:

      for newPos, prob in newPosDist.items():
			...

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper methods provided in InferenceModule above:

      1) self.setGhostPosition(gameState, ghostPosition)
          This method alters the gameState by placing the ghost we're tracking
          in a particular position.  This altered gameState can be used to query
          what the ghost would do in this position.

      2) self.getPositionDistribution(gameState)
          This method uses the ghost agent to determine what positions the ghost
          will move to from the provided gameState.  The ghost must be placed
          in the gameState with a call to self.setGhostPosition above.

    """
    updatedPacmanBeliefs = util.Counter()
    for prevPos in self.legalPositions:
        # Using the line provided above
        newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, prevPos))
        for nextPos,probNextPos in newPosDist.items():
            updatedPacmanBeliefs[nextPos] += probNextPos * self.beliefs[prevPos]
    self.beliefs = updatedPacmanBeliefs


  def getBeliefDistribution(self):
    return self.beliefs

class ParticleFilter(InferenceModule):
  """
  A particle filter for approximately tracking a single ghost.

  Useful helper functions will include random.choice, which chooses
  an element from a list uniformly at random, and util.sample, which
  samples a key from a Counter by treating its values as probabilities.
  """

  def initializeUniformly(self, gameState, numParticles=300):
    "Initializes a list of particles."
    self.numParticles = numParticles
    self.particles = []
    allValidPositions = self.legalPositions
    for id in range(self.numParticles):
        # if we want too many particles ..
        self.particles.append(allValidPositions[id % len(allValidPositions)])

  def observe(self, observation, gameState):
    "Update beliefs based on the given distance observation."
    emissionModel = busters.getObservationDistribution(observation)
    pacmanPosition = gameState.getPacmanPosition()

    if(observation == 999):
        for particleId in range(self.numParticles):
            self.particles[particleId] = (2*self.ghostAgent.index - 1, 1)
    else:
        currentBeliefs = self.getBeliefDistribution()
        newBeliefs = util.Counter()
        for position in self.legalPositions:
            truthDist = util.manhattanDistance(position, pacmanPosition)
            if emissionModel[truthDist] > 0:
                newBeliefs[position] = emissionModel[truthDist] * currentBeliefs[position]
        if(newBeliefs.totalCount() == 0):
            self.initializeUniformly(gameState)
        else:
            for particleId in range(self.numParticles):
                self.particles[particleId] = util.sample(newBeliefs)

  def elapseTime(self, gameState):
    """
    Update beliefs for a time step elapsing.

    As in the elapseTime method of ExactInference, you should use:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    to obtain the distribution over new positions for the ghost, given
    its previous position (oldPos) as well as Pacman's current
    position.
    """
    newLocDist = util.Counter()
    for particleId in range(self.numParticles):
        location = self.particles[particleId]
        if(newLocDist[location] == 0):
            newLocDist[location] = self.getPositionDistribution(self.setGhostPosition(gameState,location))
        self.particles[particleId] = util.sample(newLocDist[location])

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    curBeliefs = util.Counter()
    # Notice this is not iterating over self.NumParticles ..
    for particle in self.particles:
        # Notice the += , this is because the same particle could be referred to
        # twice or more ..
        curBeliefs[particle] += (1.0 / self.numParticles)
    return curBeliefs

class MarginalInference(InferenceModule):
  "A wrapper around the JointInference module that returns marginal beliefs about ghosts."

  def initializeUniformly(self, gameState):
    "Set the belief state to an initial, prior value."
    if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
    jointInference.addGhostAgent(self.ghostAgent)

  def observeState(self, gameState):
    "Update beliefs based on the given distance observation and gameState."
    if self.index == 1: jointInference.observeState(gameState)

  def elapseTime(self, gameState):
    "Update beliefs for a time step elapsing from a gameState."
    if self.index == 1: jointInference.elapseTime(gameState)

  def getBeliefDistribution(self):
    "Returns the marginal belief over a particular ghost by summing out the others."
    jointDistribution = jointInference.getBeliefDistribution()
    dist = util.Counter()
    for t, prob in jointDistribution.items():
      dist[t[self.index - 1]] += prob
    return dist

class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."

  def initialize(self, gameState, legalPositions, numParticles = 600):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.numParticles = numParticles
    self.ghostAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()

  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of ghost positions."
    self.particles = []
    for i in range(self.numParticles):
      self.particles.append(tuple([random.choice(self.legalPositions) for j in range(self.numGhosts)]))

  def addGhostAgent(self, agent):
    "Each ghost agent is registered separately and stored (in case they are different)."
    self.ghostAgents.append(agent)

  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.

    To loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    Then, assuming that "i" refers to the (0-based) index of the
    ghost, to obtain the distributions over new positions for that
    single ghost, given the list (prevGhostPositions) of previous
    positions of ALL of the ghosts, use this line of code:

      newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, prevGhostPositions),
                                                   i + 1, self.ghostAgents[i])

    Note that you may need to replace "prevGhostPositions" with the
    correct name of the variable that you have used to refer to the
    list of the previous positions of all of the ghosts, and you may
    need to replace "i" with the variable you have used to refer to
    the index of the ghost for which you are computing the new
    position distribution.

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper functions defined below in this file:

      1) setGhostPositions(gameState, ghostPositions)
          This method alters the gameState by placing the ghosts in the supplied positions.

      2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
          This method uses the supplied ghost agent to determine what positions
          a ghost (ghostIndex) controlled by a particular agent (ghostAgent)
          will move to in the supplied gameState.  All ghosts
          must first be placed in the gameState using setGhostPositions above.
          Remember: ghosts start at index 1 (Pacman is agent 0).

          The ghost agent you are meant to supply is self.ghostAgents[ghostIndex-1],
          but in this project all ghost agents are always the same.
    """
    newParticles = []
    newPositionDistributions = {}
    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of ghost positions
      # Each particle is basically just a list of possible combinations of
      # location that the ghosts could be in ..
      # particle = [(3,3),(2,2),(1,2),(0,0))]
      # curParticleDist = [(ghost1dist),(ghost2dist),(ghost3dist) ...]
      # newPositionDistributions = {(particle[id]),curParticleDist}
      # particle == key: (GhostIDX, GhostDistributionOverNextStates)
      # distribution over all successor states ..
      if oldParticle not in newPositionDistributions.keys():
        curParticleDist = []
        for ghostIdx in range(self.numGhosts):
          curParticleDist.append(getPositionDistributionForGhost(setGhostPositions(gameState, oldParticle), ghostIdx+1, self.ghostAgents[ghostIdx]))
        newPositionDistributions[oldParticle] = curParticleDist
      # Now update the newParticle for each ghost
      for ghostIdx in range(self.numGhosts):
        # This will return you one of the samples based on the distribution ..
        newParticle[ghostIdx] = util.sample(newPositionDistributions[oldParticle][ghostIdx])
      # Put it back as a tuple ..
      newParticles.append(tuple(newParticle))
    self.particles = newParticles

  def getJailPosition(self, i):
    return (2 * i + 1, 1);

  def getParticleWithGhostInJail(self, particle, ghostIndex):
    particle = list(particle)
    particle[ghostIndex] = self.getJailPosition(ghostIndex)
    return tuple(particle)

  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy observations.

    As in elapseTime, to loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated so
         that the ghost appears in its prison cell, position (2 * i + 1, 1),
         where "i" is the 0-based index of the ghost.

         You can check if a ghost has been captured by Pacman by
         checking if it has a noisyDistance of 999 (a noisy distance
         of 999 will be returned if, and only if, the ghost is
         captured).

      2) When all particles receive 0 weight, they should be recreated from the
          prior distribution by calling initializeParticles.
    """
    pacmanPosition = gameState.getPacmanPosition()
    noisyDistances = gameState.getNoisyGhostDistances()
    if len(noisyDistances) < self.numGhosts: return
    emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]
    curBeliefDist = self.getBeliefDistribution()

    # It's a DBN, so we need to track every single ghost distribution.
    # Sample is a list of distributions for each fhost.
    for ghostIdx in range(self.numGhosts):
      # If no noisy distance, meaning we caught it ..
      if(noisyDistances[ghostIdx] == 999):
        for particleId in range(self.numParticles):
          self.particles[particleId] = self.getParticleWithGhostInJail(self.particles[particleId], ghostIdx)
        curBeliefDist = self.getBeliefDistribution()
      # else the normal case where we multiply the
      # current distribution of the particle with the distribution for this
      # particular ghost.
    for ghostIdx in range(self.numGhosts):
      # If no noisy distance, meaning we caught it ..
      if(noisyDistances[ghostIdx] != 999):
        for particle in curBeliefDist:
          truthDistance = util.manhattanDistance(particle[ghostIdx],pacmanPosition)
          curBeliefDist[particle] *= emissionModels[ghostIdx][truthDistance]
    # Jail Case
    if curBeliefDist.totalCount() == 0:
      self.initializeParticles()
      for ghostIdx in range(self.numGhosts):
        if noisyDistances[ghostIdx] == 999:
          for id in range(self.numParticles):
            self.particles[id] = self.getParticleWithGhostInJail(self.particles[id], ghostIdx)
    # Normal case
    else:
      curBeliefDist.normalize()
      for particleId in range(self.numParticles):
        self.particles[particleId] = util.sample(curBeliefDist)

  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
  """
  Returns the distribution over positions for a ghost, using the supplied gameState.
  """
  ghostPosition = gameState.getGhostPosition(ghostIndex)
  actionDist = agent.getDistribution(gameState)
  dist = util.Counter()
  for action, prob in actionDist.items():
    successorPosition = game.Actions.getSuccessor(ghostPosition, action)
    dist[successorPosition] = prob
  return dist

def setGhostPositions(gameState, ghostPositions):
  "Sets the position of all ghosts to the values in ghostPositionTuple."
  for index, pos in enumerate(ghostPositions):
    conf = game.Configuration(pos, game.Directions.STOP)
    gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
  return gameState

