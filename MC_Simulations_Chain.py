"""
MC Simulations with Chain environment
"""

import logging
import math
import numpy as np
from tqdm import tqdm

EPS = 1e-8

log = logging.getLogger(__name__)


class MCChain():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, max_moves = 10000):
        self.env = env
        self.reject_rate = []
        self.actions = []
        self.score = []
        self.max_moves = max_moves
        
    def sim(self, state, temp = 1, ctr = 0):
        """
        This function performs one Monte Carlo Simulation using Metropolis moves.
        It is recursively called until the STOP action is chosen. 
        
        More specifically, given state x(t), randomly select one of the moves from A(x(t)), 
        and execute it to get x(t+1). Now, let c(x(t)) = number of bonds formed in state x at time t,
        and let delta = c(x(t+1)) - c(x(t)). Accept the move with probability p < min(1, exp(delta c / T)),
        where T is the temperature (assuming T = 1 in this example). 

        Parameters
        ----------
        state : numpy.ndarray
            starting state given by the environment
        temp : integer
            Temperature T in the formula above
            
        Returns
        -------
        int
            Score / Reward at the end of the simulation
        dict
            Additional information regarding the simulation.
        
        """
        ctr = 0
        state = state
        for ctr in tqdm(range(self.max_moves)):
            if ctr > 100:
                temp = 0.5
            if ctr > 200:
                temp = 0.4
            if ctr > 300:
                temp = 0.3
            if ctr > 450:
                temp = 0.2
            if ctr > 750:
                temp = 0.1
            if ctr > 1000:
                temp = 0.05
            
            curr_score = self.env.calc_score(state)
            valids = np.array(np.nonzero(self.env.valid_moves(state)))[0][:-1]
            # Find suitable action
            action = 0
            cnt = 0
            while True:
                action = np.random.choice(valids, 1)[0]
                temp_state = np.copy(state)
                temp_state = self.env.next_state(temp_state, action)
                new_score = self.env.calc_score(temp_state)
                delta_c = new_score - curr_score
                threshhold = min(1, math.exp(delta_c / temp))
                if np.random.random_sample() < threshhold: #accept
                    break
                else:
                    cnt += 1
            reject_rate = ((100 * cnt / (cnt + 1)) // 1) / 100
            state = self.env.next_state(state, action)
            
            self.reject_rate.append(reject_rate)
            self.actions.append(action)
            self.score.append(self.env.calc_score(state))
            
            
            ctr += 1

        info = {
            "Actions"     : self.actions,
            "Reject_rate" : self.reject_rate,
            "Rewards"     : self.score
        }
        reward = self.env.calc_score(state)
        return (reward, info)