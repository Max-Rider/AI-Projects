# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore() #init score == successor state score
        
        foodList = newFood.asList() #get iterable list of food posistions
        
        #find the manhattanDist of each food pellet to pacman
        #and add the reciprocal of that value to the score (recommendation from assignment)
        for food in foodList:
            fDist = util.manhattanDistance(newPos, food)
            score += (1/fDist)
        
        #similar to above but I have to account for more than one ghost
        #as well as making sure that ghost is at least one space away
        #(i.e. not sharing the same space as pacman)
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            gDSist = util.manhattanDistance(newPos, ghostPos)
            if gDSist > 1:
                score += (1/gDSist)
        
        return score
        
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agent):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        #numAgents = gameState.getNumAgents()
        #legalActions = gameState.getLegalActions()
        
        return self.maxval(gameState, 0, 0)[0]

    def minimax(self, gameState, agent, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.maxval(gameState, agent, depth)[1]
        else:
            return self.minval(gameState, agent, depth)[1]

    def maxval(self, gameState, agent, depth):
        maxEval = ("Stop",float("-inf")) #init maxEval to lowest num possible
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            nextGameState = gameState.generateSuccessor(agent,action) #find the successors states from the given action
            nextAgent = (depth + 1)%gameState.getNumAgents() # calculate the numAgents; x%x==0 so it rolls over to pacman after all ghosts move
            nextAction = (action,self.minimax(nextGameState, nextAgent ,depth+1)) #call minimax with new values
            maxEval = max(maxEval,nextAction,key=lambda val:val[1]) #take the max val; needed to use key function to compare the proper values
        return maxEval

    #This is similar to the function above but evaluates for the min val instead of max
    def minval(self, gameState, agent, depth):
        minEval = ("Stop",float("inf"))
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            nextGameState = gameState.generateSuccessor(agent,action)
            nextAgent = (depth + 1)%gameState.getNumAgents()
            nextAction = (action,self.minimax(nextGameState, nextAgent ,depth+1))
            minEval = min(minEval,nextAction,key=lambda val:val[1])
        return minEval
           
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        return self.maxval(gameState, 0, 0, float("-inf"), float("inf"))[0]

    def ABprune(self, gameState, agent, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.maxval(gameState, agent, depth, alpha, beta)[1]
        else:
            return self.minval(gameState, agent, depth, alpha, beta)[1]

    def maxval(self, gameState, agent, depth, alpha, beta):
        maxEval = ("Stop",float("-inf")) #init maxEval to a tuple of the lowest num possible and the action "Stop"
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            nextGameState = gameState.generateSuccessor(agent,action) #find the successors states from the given action
            nextAgent = (depth + 1)%gameState.getNumAgents() # calculate the numAgents; x%x==0 so it rolls over to pacman after all ghosts move
            nextAction = (action, self.ABprune(nextGameState, nextAgent ,depth+1, alpha, beta)) #call minimax with new values
            maxEval = max(maxEval,nextAction,key=lambda val:val[1]) #take the max val; needed to use key function to compare the proper values
            if maxEval[1] > beta: #prune leaves from tree
                return maxEval
            else:
                alpha = max(alpha, maxEval[1])
        return maxEval

    #This is similar to the function above but evaluates for the min val instead of max
    def minval(self, gameState, agent, depth, alpha, beta):
        minEval = ("Stop",float("inf"))
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            nextGameState = gameState.generateSuccessor(agent,action)
            nextAgent = (depth + 1)%gameState.getNumAgents()
            nextAction = (action, self.ABprune(nextGameState, nextAgent ,depth+1, alpha, beta))
            minEval = min(minEval,nextAction,key=lambda val:val[1])
            if minEval[1] < alpha:
                return minEval
            else:
                beta = min(beta, minEval[1])
        return minEval
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        return self.maxVal(gameState,0,0)[0]
        
        #util.raiseNotDefined()
        
    #probability stuff here
    
    def expectimax(self, gameState, agent, depth):
      if depth == self.depth* gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)
      if (agent == 0):
        return self.maxVal(gameState, agent, depth)[1]  
      else:
        return self.expectedVal(gameState, agent, depth)
    
    #basically a copy of the previous minimax and ABPruning maxVal functions see those for a better description of each component
    def maxVal(self, gameState, agent, depth):
        maxEval = ("Stop",float("-inf"))
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            nextState = gameState.generateSuccessor(agent, action)
            nextAgent = (depth + 1) % gameState.getNumAgents()
            nextAction = (action, self.expectimax(nextState, nextAgent, depth + 1))
            maxEval = max([maxEval, nextAction], key=lambda val:val[1])
        return maxEval

    # This function is different than minval in that it calculates the uniform probability of each ghost action 
    # instead of just taking the minimum or pruning selections.
    # It functions the same basic way though
    def expectedVal(self, gameState, agent, depth):
        probEval = 0
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            nextState = gameState.generateSuccessor(agent, action)
            nextAgent = (depth + 1) % gameState.getNumAgents()
            nextAction = self.expectimax(nextState, nextAgent, depth + 1)
            probEval += nextAction
        return probEval / len(legalActions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    I basically took my eval function from question 1 and added the capsule locations to it.
    I then added this to my score variable and it works perfectly.
    
    Initially I thought it was going to be much more in depth but after seeing that this worked perfectly
    I realized that I was overthinking it's difficulty.
    
    
    """
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCaps = currentGameState.getCapsules()
    
        
    foodList = newFood.asList()
    #capList = newCap.asList()
    score = currentGameState.getScore()
    
    #print("NEW POS",newPos)
    
    for food in foodList:
        fDist = util.manhattanDistance(newPos, food)
        score += (1/fDist)
        
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        gDSist = util.manhattanDistance(newPos, ghostPos)
        if gDSist > 1:
            score += (1/gDSist)
    
    # Added capsules as a part of the score. The more capsules eaten the more points         
    for capsule in newCaps:
        cDist = util.manhattanDistance(newPos, capsule)
        score += (1/cDist)
   
    return score
    
    
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
