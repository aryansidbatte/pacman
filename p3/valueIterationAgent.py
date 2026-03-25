from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        for state in self.mdp.getStates():
            self.values[state] = 0.0

        # Perform val iter
        for _ in range(self.iters):
            newValues = {}  # Store new values for this iteration
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0.0
                else:
                    # Compute the maximum q-value for this state
                    actions = self.mdp.getPossibleActions(state)
                    maxQValue = max(self.getQValue(state, action) for action in actions)
                    newValues[state] = maxQValue
            self.values = newValues

        # raise NotImplementedError()

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)
    
    def getPolicy(self, state):
        if self.mdp.isTerminal(state):
            return None
        
        # Find the action with the highest Q-value
        actions = self.mdp.getPossibleActions(state)
        bestAction = max(actions, key=lambda action: self.getQValue(state, action))
        return bestAction
    
    def getQValue(self, state, action):
        # Compute teh Q-value for the (state, action) pair
        qValue = 0.0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discountRate * self.getValue(nextState))
        return qValue

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
