# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    currentState = problem.getStartState() # initialize currentState to startState
    
     # initialize frontier using stack of tuples of states and actions
    #successorStates = problem.getSuccessors(currentState)
    statesToVisit = util.Stack()
    statesToVisit.push((currentState, []))
    
    #initialize empty list for explored set and actionSet
    exploredSet = []
    actionSet = []
    
    while statesToVisit:
        currentState, actionSet = statesToVisit.pop() #set currentState and actionSet
        #actionSet.append(currentState[1]) #add action to solution set
        if currentState not in exploredSet: #check if node seen before
            exploredSet.append(currentState) #add the current state(NODE) to the exploredSet
            
            if problem.isGoalState(currentState): #check if the current state is the goal state
                return actionSet #if true return current actionSet

            successorStates = problem.getSuccessors(currentState) #expand the node we just added to the exploredSet
            for state in successorStates:
                if state[0] not in exploredSet:
                    statesToVisit.push((state[0] ,actionSet + [state[1]]))
                    #actionSet.append(state)
        
"""         for i in range(len(successorStates)):#check for next state in explored set
            frontierCopy = statesToVisit
            while not frontierCopy.isEmpty():
                frontierNode = frontierCopy.pop()
                print("Next node",frontierNode)
                if (frontierNode is not successorStates[i]) and (successorStates[i] not in exploredSet):
                    #if successorStates[i] not in exploredSet: #if not found push next state onto stack
                    print("here")
                    statesToVisit.push(successorStates[i])
    print("exit while") """

                    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    startState = problem.getStartState()
    if problem.isGoalState(startState):
        return []
    
    frontier = util.Queue()
    frontier.push((startState, []))
    
    exploredSet = []
    
    while frontier:
        node, actionSet = frontier.pop()
        if node not in exploredSet:
            exploredSet.append(node)
        
            if problem.isGoalState(node):
                return actionSet
            
            successors = problem.getSuccessors(node)
            for child in successors:
                if child[0] not in exploredSet:
                # if problem.isGoalState(child[0]):
                #     return actionSet
                    frontier.push((child[0], actionSet + [child[1]]))
                    #actionSet.append(child[1])

    
"""     startNode = problem.getStartState()
    actionSet = []
    
    if problem.isGoalState(startNode):
        return actionSet
    
    exploredSet = []  
    frontier = util.Queue()
    frontier.push((startNode, []))
    
    while not frontier.isEmpty():
        node, actionSet = frontier.pop()
        exploredSet.append(node)
        print(node)
        successors = problem.getSuccessors(node)
        for newNode in successors:
            #frontier.push((newNode, actionSet + [newAction]))
            if newNode not in exploredSet:
                if problem.isGoalState(node):
                    return actionSet
                frontier.push((newNode, actionSet + [newAction])) """
    
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    #implement using priority queue
    #give higher priority to lower cost paths
    
    startState = problem.getStartState()
    
    frontier = util.PriorityQueue()
    frontier.push((startState, [], 0), 0) #use a priority queue of triples of (state, action, cost)
    #initial priority is 0
    
    exploredSet = [] #initialize exploredSet as the empty set
    #totalCost = 0 #initialize cost to 0
    
    while frontier:
        state, actionSet, cost = frontier.pop()
        if state not in exploredSet:
            exploredSet.append(state)

            if problem.isGoalState(state):
                return actionSet
            successors = problem.getSuccessors(state)
            for child in successors:
                frontier.push((child[0], actionSet + [child[1]], cost + child[2]), cost + child[2])
    
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    startState = problem.getStartState()
    
    frontier = util.PriorityQueue()
    frontier.push((startState, [], 0), 0)
    
    exploredSet = []
    
    while frontier:
        state, actionSet, cost = frontier.pop()
        if state not in exploredSet:
            exploredSet.append(state)

            if problem.isGoalState(state):
                return actionSet
            successors = problem.getSuccessors(state)
            for child in successors:
                realCost = cost + child[2]
                hCost = realCost + heuristic(child[0], problem)
                frontier.push((child[0], actionSet + [child[1]], realCost),hCost)
        
    
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
