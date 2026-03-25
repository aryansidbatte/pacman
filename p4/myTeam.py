from pacai.agents.capture.defense import DefensiveReflexAgent
from pacai.agents.capture.offense import OffensiveReflexAgent
from pacai.core.directions import Directions

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexes.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        ImprovedOffensiveAgent(firstIndex),
        ImprovedDefensiveAgent(secondIndex)
    ]

class ImprovedOffensiveAgent(OffensiveReflexAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Distance to capsule for making ghosts scared.
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            capsuleDistances = [self.getMazeDistance(myPos, c) for c in capsules]
            features['distanceToCapsule'] = min(capsuleDistances)
        else:
            features['distanceToCapsule'] = 0

        # Visible enemy ghosts.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        visibleGhosts = [e for e in enemies if not e.isPacman() and e.getPosition() is not None]
        if len(visibleGhosts) > 0:
            ghostDistances = [self.getMazeDistance(myPos, g.getPosition()) for g in visibleGhosts]
            minGhostDist = min(ghostDistances)
            scaredTimes = [g.getScaredTimer() for g in visibleGhosts]

            # If any ghost is scared, don't be afraid/ignore.
            if any(t > 0 for t in scaredTimes):
                features['ghostDistance'] = 0  # No fear if they're scared.
            else:
                features['ghostDistance'] = minGhostDist
        else:
            # No ghosts in large distance, treat as safe.
            features['ghostDistance'] = 10

        # no stopping
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        # no looping
        currentDirection = gameState.getAgentState(self.index).getDirection()
        rev = Directions.REVERSE[currentDirection]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        # Stay on enemy territory.
        layout = gameState.getInitialLayout()
        midX = layout.getWidth() // 2
        if self.red:
            boundaryPositions = [(midX - 1, y) for y in range(layout.getHeight())
                                 if not gameState.getWalls()[midX - 1][y]]
        else:
            boundaryPositions = [(midX, y) for y in range(layout.getHeight())
                                 if not gameState.getWalls()[midX][y]]

        if len(boundaryPositions) > 0:
            boundaryDist = min([self.getMazeDistance(myPos, b) for b in boundaryPositions])
        else:
            boundaryDist = 0
        features['distanceToBoundary'] = boundaryDist

        # Check if on own side.
        if myState.isPacman():
            features['onOwnSide'] = 0
        else:
            features['onOwnSide'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,      # Increase score by eating food.
            'distanceToFood': -3,       # Strongly encourage going towards food.
            'distanceToCapsule': -2,    # Encourage getting capsules.
            'ghostDistance': 1,         # Avoid non-scared ghosts. No penalty if scared.
            'stop': -100,               # Don't stop.
            'reverse': -20,             # Strongly discourage going back and forth.
            'distanceToBoundary': -0.5,  # Encourage crossing into enemy territory.
            'onOwnSide': -1             # Don't linger on your own side too long.
        }

class ImprovedDefensiveAgent(DefensiveReflexAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Patrol between team's central region.
        layout = gameState.getInitialLayout()
        midX = layout.getWidth() // 2
        if self.red:
            patrolPositions = [(midX - 1, y) for y in range(layout.getHeight())
                               if not gameState.getWalls()[midX - 1][y]]
        else:
            patrolPositions = [(midX, y) for y in range(layout.getHeight())
                               if not gameState.getWalls()[midX][y]]

        if len(patrolPositions) > 0:
            patrolDist = min([self.getMazeDistance(myPos, p) for p in patrolPositions])
        else:
            patrolDist = 0
        features['patrolDistance'] = patrolDist

        # If no invaders are visible, encourage patrolling.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        if len(invaders) == 0:
            features['noInvaders'] = 1
        else:
            features['noInvaders'] = 0

        # Add reverse feature like in reflex agent:
        currentDirection = gameState.getAgentState(self.index).getDirection()
        rev = Directions.REVERSE[currentDirection]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def getWeights(self, gameState, action):
        weights = super().getWeights(gameState, action)
        # Adjust and add new weights for defense.
        weights.update({
            'patrolDistance': -1,   # Encourage being close to patrol line.
            'noInvaders': 100,      # Reward being prepared when no invaders.
            'invaderDistance': -20,  # Increased emphasis on catching invaders quickly.
            'stop': -100,           # Don't stop.
            'reverse': -20          # Strongly discourage going back and forth.
        })
        return weights
