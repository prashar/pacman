# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0

    # Precompute the policy beforehand ..
    # For each iteration
    #   For each state
    #       for each action
    #   Update values
    for iteration in range(self.iterations):
      # Store a dictionary to maintain the newest states .. 
      newState_values = util.Counter() ; 
      for state in self.mdp.getStates():
        # If it is a terminal state, then just set have it's value set to 0.      
        if(self.mdp.isTerminal(state)):
            newState_values[state] = 0
            continue 
        cur_val = -float('inf')
        for action in self.mdp.getPossibleActions(state):
            cur_val = max(cur_val,self.getQValue(state,action))
        newState_values[state] = cur_val
      # Assign to instance variable .. 
      self.values = newState_values  

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  # This routine computes the sum of all q values that could result if we did a specific 
  # (state,action) sequence. Basically, it's the sum of all Q's -> R(s,a,s') + gamma * v(s')
  # for each AVAIL action(based on prob). 
  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    total_q_val = 0 
    # For terminal state, there is only one action - exit. 
    for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state,action):
        total_q_val += prob * (self.mdp.getReward(state, action, nextState) + self.discount*self.getValue(nextState))
    return total_q_val

  # This routine just returns the MAX Q-Value corresponding to a given state
  # it iterates over all actions avaialble in state, and returns the one with max utility. 
  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    utility = -float('inf')
    max_value_action = None 
    for action in self.mdp.getPossibleActions(state):
        total_q_val_for_state = self.getQValue(state,action) 
        if( total_q_val_for_state > utility):
            max_value_action = action 
            utility = total_q_val_for_state
    return max_value_action  

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
