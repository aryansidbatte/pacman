"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from collections import deque
from pacai.util.priorityQueue import PriorityQueue
import heapq

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***

    # Stack to store the nodes, starting with the initial state
    stack = [(problem.startingState(), [])]
    # Set to store the visited nodes
    visited = set()

    # While there are nodes to visit
    while stack:
        # Pop the node and the path to it
        node, path = stack.pop()
        # If the node is the goal, return the path
        if problem.isGoal(node):
            return path
        # If the node has not been visited
        if node not in visited:
            # Add the node to the visited set
            visited.add(node)
            # For each successor of the node
            for successor, action, cost in problem.successorStates(node):
                # Add the successor and the path to it to the stack
                stack.append((successor, path + [action]))
    # If the goal is not found, return an empty path
    return []
    # raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***

    # Queue to store the nodes, starting with the initial state
    queue = deque([(problem.startingState(), [])])
    # Set to store the visited nodes
    visited = set()

    # While there are nodes to visit
    while queue:
        # Pop the node and the path to it
        node, path = queue.popleft()
        # If the node is the goal, return the path
        if problem.isGoal(node):
            return path
        # If the node has not been visited
        if node not in visited:
            # Add the node to the visited set
            visited.add(node)
            # For each successor of the node
            for successor, action, cost in problem.successorStates(node):
                # Add the successor and the path to it to the queue
                queue.append((successor, path + [action]))
    # If the goal is not found, return an empty path
    return []

    # raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***

    # Priority queue to store the nodes, starting with the initial state
    queue = [(0, (problem.startingState(), []))]
    # Set to store the visited nodes
    visited = set()
    heapq.heapify(queue)

    while queue:
        # Pop the node and the path to it
        cost, (node, path) = heapq.heappop(queue)
        # If the node is the goal, return the path
        if problem.isGoal(node):
            return path
        # If the node has not been visited
        if node not in visited:
            # Add the node to the visited set
            visited.add(node)
            # For each successor of the node
            for successor, action, successorCost in problem.successorStates(node):
                # Add the successor and the path to it to the queue
                heapq.heappush(queue, (cost + successorCost, (successor, path + [action])))
    # If the goal is not found, return an empty path
    return []

    # raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    # Initialize a priority queue with the starting state
    pq = PriorityQueue()
    start_state = problem.startingState()
    pq.push((start_state, []), 0)  # (state, path), priority
    visited = set()
    costs = {start_state: 0}

    while not pq.isEmpty():
        state, path = pq.pop()

        # If the current state is the goal, return the path to reach it
        if problem.isGoal(state):
            return path

        if state not in visited:
            visited.add(state)

            # Add successors to the priority queue
            for successor, action, step_cost in problem.successorStates(state):
                new_cost = costs[state] + step_cost
                if successor not in visited or new_cost < costs.get(successor, float('inf')):
                    costs[successor] = new_cost
                    priority = new_cost + heuristic(successor, problem)
                    pq.push((successor, path + [action]), priority)

    # If the goal is not found, return an empty path
    return []

    # raise NotImplementedError()
