import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # init score from successor state's score
        score = successorGameState.getScore()

        # 1 - distance to closest food
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min([distance.manhattan(newPos, food) for food in foodList])
            score += 2.0 / minFoodDistance

        # 2 - distance to ghosts
        for ghostIndex, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDistance = distance.manhattan(newPos, ghostPos)

            if newScaredTimes[ghostIndex] > 0:
                # apporach ghost if scared for potential points
                score += 1.0 / (ghostDistance + 1)

            else:
                # avoid ghost if not scared
                if ghostDistance < 2:
                    score -= 10 / (ghostDistance + 1)

        # 3 - Remaining food coount to encourage eating and finishing the board
        score -= len(foodList)

        # return successorGameState.getScore()
        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using
        self.getTreeDepth() and self.getEvaluationFunction().
        """
        bestScore, bestAction = self.minimax(gameState, 0, 0)
        return bestAction
    
    def minimax(self, state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(state), None

        if agentIndex == 0:
            return self.maximize(state, depth, agentIndex)
        else:
            return self.minimize(state, depth, agentIndex)
        
    def maximize(self, state, depth, agentIndex):
        bestScore = float("-inf")
        bestAction = None

        legalActions = [action for action in state.getLegalActions(0) if action != Directions.STOP]
        
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(successor, depth, agentIndex + 1)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction

    def minimize(self, state, depth, agentIndex):
        bestScore = float("inf")
        bestAction = None

        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            score, _ = self.minimax(successor, nextDepth, nextAgent)

            if score < bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using alpha-beta pruning.
        """
        bestScore, bestAction = self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))
        return bestAction
    
    def alphaBeta(self, state, depth, agentIndex, alpha, beta):
        # Check if we've reached teh depth limit / game over
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state), None

        # If pacman turn (agent 0), maximize the score
        if agentIndex == 0:
            return self.maximize(state, depth, agentIndex, alpha, beta)
        # If ghost turn, (agent >= 1), minimize the score
        else:
            return self.minimize(state, depth, agentIndex, alpha, beta)
        
    def maximize(self, state, depth, agentIndex, alpha, beta):
        bestScore = float("-inf")
        bestAction = None

        legalActions = [action for action in state.getLegalActions(0) if action != Directions.STOP]

        # Go through all possible actions for Pacman.
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            score, _ = self.alphaBeta(successor, depth, agentIndex + 1, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            # Update alpha and check for pruning.
            alpha = max(alpha, bestScore)
            if alpha >= beta:
                break  # Prune the branch.

        return bestScore, bestAction

    def minimize(self, state, depth, agentIndex, alpha, beta):
        bestScore = float("inf")
        bestAction = None

        # Go through all possible actions for the ghost.
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            # Determine the next agent (or Pacman if this is the last ghost).
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            score, _ = self.alphaBeta(successor, nextDepth, nextAgent, alpha, beta)

            if score < bestScore:
                bestScore = score
                bestAction = action

            # Update beta and check for pruning.
            beta = min(beta, bestScore)
            if beta <= alpha:
                break  # Prune the branch.

        return bestScore, bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using the eval func.
        """
        bestScore, bestAction = self.expectimax(gameState, 0, 0)
        return bestAction
    
    def expectimax(self, state, depth, agentIndex):
        # Check if we've reached teh depth limit / game over
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state), None
        
        # If pacman turn (agent 0), maximize the score
        if agentIndex == 0:
            return self.maximize(state, depth, agentIndex)
        # If ghost turn, (agent >= 1), minimize the score
        else:
            return self.expect(state, depth, agentIndex)
        
    def maximize(self, state, depth, agentIndex):
        bestScore = float("-inf")
        bestAction = None

        # Go through all possible actions for Pacman.
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            score, _ = self.expectimax(successor, depth, agentIndex + 1)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction

    def expect(self, state, depth, agentIndex):
        totalScore = 0
        legalActions = state.getLegalActions(agentIndex)

        # No actions mean we reached a terminal state; return evaluation.
        if not legalActions:
            return self.getEvaluationFunction()(state), None

        # Calculate the expected value for each action for the ghost.
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            # Determine the next agent (or Pacman if this is the last ghost).
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            score, _ = self.expectimax(successor, nextDepth, nextAgent)

            # Accumulate the scores for averaging.
            totalScore += score

        # Return the average score as the expected value.
        expectedScore = totalScore / len(legalActions)
        return expectedScore, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    # Basic stuff of current state
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    foodScore = 0
    ghostScore = 0
    capsuleScore = 0

    # 1 - distance to closest food
    if foodList:
        closestFoodDist = min(distance.manhattan(pacmanPos, foodPos) for foodPos in foodList)
        foodScore = 1.0 / closestFoodDist

    # 2 - ghost dist and scared times
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        scaredTimer = ghostState.getScaredTimer()
        distanceToGhost = distance.manhattan(pacmanPos, ghostPos)

        # Avoiding active ghosts
        if scaredTimer == 0:
            ghostScore -= 10 / max(distanceToGhost, 1)  # Closer active ghosts have a high penalty

        # Chasing scared ghosts
        elif scaredTimer > 0:
            ghostScore += 10 / max(distanceToGhost, 1)  # Closer scared ghosts have a high reward

    # 3 - distance to capsules (closer = better)
    if capsules:
        closestCapsuleDist = min(distance.manhattan(pacmanPos, capsule) for capsule in capsules)
        capsuleScore = 2.0 / closestCapsuleDist

    evaluation = score + (10 * foodScore) + ghostScore + (15 * capsuleScore)

    return evaluation

    # return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
