# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    # Current state of the board
    # based on whatever the agent has learnt so far. 
    self.qlearntVals = util.Counter()

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    return self.qlearntVals[(state,action)]
  
  # This routine takes in the state and returns
  # the maximum utility - or qValue. It calculates that for each action ..   
  def getValue(self,state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    max_q_val = -float('inf')
    for action in self.getLegalActions(state):
      temp_qval = self.getQValue(state,action)
      max_q_val = max(max_q_val,temp_qval)
    # Corner case - we need to make sure we return a 0.0, in case there are no legal actions. 
    if(max_q_val == -float('inf')):
        return 0.0
    return max_q_val ; 
  
  def getPolicy(self,state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    action_list = []
    max_q_val = -float('inf') 
    for action in self.getLegalActions(state):
      temp_qval = self.getQValue(state,action) 
      # Keep adding to list on ties .. 
      if(temp_qval == max_q_val):
          action_list.append(action)
      elif(temp_qval > max_q_val):
          max_q_val = temp_qval
          # overwrite list if > max_q_val .. 
          action_list = [action] 
    # If no legal actions, we must return none .. 
    if not action_list:
        return None
    # in case there are multiple options, we choose randomly
    # as discussed in the assignment.
    return random.choice(action_list)

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick random action or next action based on epsilon .. 
    if(util.flipCoin(self.epsilon)):
        return random.choice(self.getLegalActions(state))
    else:
        return self.getPolicy(state)

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    # from class slides .. .
    # using exponential average.
    new_qval = (1-self.alpha)*self.qlearntVals[(state,action)] + (self.alpha)*(reward + self.discount*self.getValue(nextState))
    self.qlearntVals[(state,action)] = new_qval

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)
    # Adding an weights vector as per the requirement. 
    self.weights = util.Counter() 
  
  def getWeights(self):
    return self.weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    # This is really meant to be a dot product. 
    return self.weights * self.featExtractor.getFeatures(state,action)

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    # Basically I need to change each weight by alpha times the increase times the old value. 
    # and then add that to the previous value of weights to get the new value.  
    features_newstate = util.Counter()
    features_prevstate = self.featExtractor.getFeatures(state,action)
    correction = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action)
    for key in features_prevstate.keys():
        features_newstate[key] = features_prevstate[key] * self.alpha *  correction
    # Notice this is a += , since we need to add the change's effect to our running total.
    self.weights += features_newstate 

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass