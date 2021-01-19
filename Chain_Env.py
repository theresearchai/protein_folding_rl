"""
Implements the Chain Environment with Pull Moves
"""

import sys
from math import floor
from collections import OrderedDict
from sklearn.metrics.pairwise import euclidean_distances
import itertools

import gym
from gym import (spaces, utils, logger)
import numpy as np
from six import StringIO

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

POLY_TO_INT = {
    'H' : 1, 'P' : -1
}

DIR_TO_ARRAY = {
    0: np.array([-1, 1]),
    1: np.array([-1, -1]),
    2: np.array([1, -1]),
    3: np.array([1, 1])
}

ACTION_TO_DIR = {
    0: "upper right",
    1: "upper left",
    2: "lower left",
    3: "lower right"
}

class ChainEnv(gym.Env):
    """A 2-dimensional chain environment with pull moves, inspired by
    N. Lesh, M. Mitzenmacher, and S. Whitesides.

    The environment will place the entire sequence horizontally in the
    center of the grid. Then, for each step, pull one of the polymers in
    one of four directions (NE, NW, SW, SE), or pull one of the end polymers
    two positions in one of the four cardinal directions (N, E,, S, W). The
    rest of the chain will shift accordingly in response to the pull.
    
    An episode either when specified with the STOP action, or when the agent 
    has made n + 2 moves, where n is the length of the sequence. This is 
    because the maximum number of pulls required to reach the optimal 
    structure is n, so we limit the total number of moves possible by a 
    little more than the length of the sequence. We then compute the reward
    using the energy minimization rule. Unlike the lattice environment, we
    do not allow the agent to make illegal moves, and the agent cannot be
    trapped, so we do not have a trapped_penalty or a collision_penalty.

    Attributes
    ----------
    seq : str
        Polymer sequence describing a particular protein.
    state : OrderedDict
        Dictionary of the current polymer chain with coordinates and
        polymer type (H or P).
    actions : list
        List of actions performed by the model.
    reward : int
        Current reward for the actions performed by the model.
    grid_len : int
        Length of one side of the grid.
    max_len : int
        Maximum length for a sequence.
    midpoint : tuple
        Coordinate containing the midpoint of the grid.
    grid : numpy.ndarray
        Actual grid containing the polymer chain.
        
    """
    def __init__(self, seq, max_len = 100, grid_len = 201):
        try:
            if not set(seq.upper()) <= set('HP'):
                raise ValueError("%r (%s) is an invalid sequence" % (seq, type(seq)))
            self.seq = seq.upper()
        except AttributeError:
            logger.error("%r (%s) must be of type 'str'" % (seq, type(seq)))
            raise

        try:
            if len(seq) > 100:
                raise ValueError("%r (%s) must have length <= 100" % (seq, type(seq)))
        except AttributeError:
            logger.error("%r (%s) must be of type 'str'" % (seq, type(seq)))
            raise
            
        self.seq = seq
        self.actions = []
        self.reward = 0
        self.max_len = max_len
        self.grid_len = grid_len
        self.midpoint = (grid_len // 2, grid_len // 2)
        self.grid = np.zeros(shape = (self.grid_len, self.grid_len), dtype = int)
        
        self.state = np.full((2, len(seq)), grid_len // 2) #state[0] = x-coords, state[1] = y-coords, state[:,i] = coords for mol i
        for i in range(len(seq)):
            self.state[1][i] = grid_len // 2 - len(seq) // 2 + i
            
        self.prev_state = np.copy(self.state)
        
        for i in range(len(seq)):
            self.grid[self.state[0, i]][self.state[1, i]] = POLY_TO_INT[self.seq[i]]

        self.action_space = spaces.Discrete(4 * max_len + 9)
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.grid_len, self.grid_len,), dtype = int)
        
    def step(self, action):
        """Updates the current chain with the specified action.

        The action supplied by the agent should be an integer from 0
        to 4 * max_len + 8. The action 4 * i + j pulls the polymer in 
        position i in the direction j, where j = 0, 1, 2, 3 corresponds to 
        NE, NW, SW, SE, respectively. The action 4 * max_len + j pulls the
        first polymer two positions in direction j, where j = 0, 1, 2, 3
        corresponds to N, W, S, E, respectively. The action 4 * (max_len + 1) + j
        pulls the last polymer two positions in direction j, where j = 0, 1, 2, 3
        corresponds to N, W, S, E, respectively. The action 4 * max_len + 9
        is the STOP action.

        This method returns a set of values similar to the OpenAI gym, that
        is, a tuple :code:`(observations, reward, done, info)`.

        The observations are arranged as a :code:`numpy.ndarray` matrix, more
        suitable for agents built using convolutional neural networks. The
        'H' is represented as :code:`1`s whereas the 'P's as :code:`-1`s.
        However, for the actual chain, that is, an :code:`OrderedDict` and
        not its grid-like representation, can be accessed from
        :code:`info['state_chain]`.

        The reward is calculated at the end of every episode, that is, when
        the length of the chain is equal to the length of the input sequence.

        Parameters
        ----------
        action : int, {0, 1, 2, ..., 4 * max_len + 8)
            Specifies the polymer and the direction that will be pulled.
            
        Returns
        -------
        numpy.ndarray
            Current state of the lattice.
        int or None
            Reward for the current episode.
        bool
            Control signal when the episode ends.
        dict
            Additional information regarding the environment.

        Raises
        ------
        AssertionError
            When the specified action is invalid.
        IndexError
            When :code:`step()` is still called even if done signal
            is already :code:`True`.
        """
        if not self.valid(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))
        
        self.actions.append(action)
        np.copyto(self.prev_state, self.state)
        done = False
        if action == 4 * self.max_len + 8: #Stop
            done = True
            return (self.grid, self.reward, done, self.actions)
        elif action >= 4 * self.max_len + 4: #For the last mol
            if action == 4 * self.max_len + 7:
                self.state[1][-1] += 2
                self.state[0][-2] = self.state[0][-1]
                self.state[1][-2] = self.state[1][-1] - 1
            elif action == 4 * self.max_len + 6:
                self.state[0][-1] += 2
                self.state[0][-2] = self.state[0][-1] - 1
                self.state[1][-2] = self.state[1][-1]
            elif action == 4 * self.max_len + 5:
                self.state[1][-1] -= 2
                self.state[0][-2] = self.state[0][-1]
                self.state[1][-2] = self.state[1][-1] + 1
            else:
                self.state[0][-1] -= 2
                self.state[0][-2] = self.state[0][-1] + 1
                self.state[1][-2] = self.state[1][-1]
                
            i = len(self.seq) - 3
            self.state[:, i] = self.prev_state[:, i + 2]
            while not self.is_adj(self.state[:, i], self.state[:, i - 1]) and i >= 1:
                i -= 1
                self.state[:,  i] = self.prev_state[:, i + 2]
                
        elif action >= 4 * self.max_len: #For the first mol
            if action == 4 * self.max_len + 3:
                self.state[1][0] += 2
                self.state[0][1] = self.state[0][0]
                self.state[1][1] = self.state[1][0] - 1
            elif action == 4 * self.max_len + 2:
                self.state[0][0] += 2
                self.state[0][1] = self.state[0][0] - 1
                self.state[1][1] = self.state[1][0]
            elif action == 4 * self.max_len + 1:
                self.state[1][0] -= 2
                self.state[0][1] = self.state[0][0]
                self.state[1][1] = self.state[1][0] + 1
            else:
                self.state[0][0] -= 2
                self.state[0][1] = self.state[0][0] + 1
                self.state[1][1] = self.state[1][0]
                
            i = 2
            self.state[:, i] = self.prev_state[:, i - 2]
            while i <= len(self.seq) - 2 and not self.is_adj(self.state[:, i], self.state[:, i + 1]):
                i += 1
                self.state[:, i] = self.prev_state[:, i - 2]
                
        else: #For all other mols    
            i = action // 4
            dir = action % 4
            loc = DIR_TO_ARRAY[dir] + self.prev_state[:, i]
            self.state[:, i] = loc
            pre = True if i == len(self.seq) - 1 else False
            post = True if i == 0 else False
            if i != 0 and i != len(self.seq) - 1:
                pre = True if self.is_adj(loc, self.state[:, i - 1]) else False # i - 1 adjacent to i's new location
                post = True if self.is_adj(loc, self.state[:, i + 1]) else False # i + 1 adjacent to i's new location
            if pre and post:
                self.grid[self.prev_state[0, i], self.prev_state[1, i]] = 0
                self.grid[self.state[0, i], self.state[1, i]] = POLY_TO_INT[self.seq[i]]
                return (self.grid, self.reward, done, self.actions)
            elif pre:
                C = loc + self.prev_state[:, i] - self.prev_state[:, i - 1]
                if i != len(self.seq) - 1:
                    self.state[:, i + 1] = C
                i += 2
                if i <= len(self.seq) - 1:
                    self.state[:, i] = self.prev_state[:, i - 2]
                    while i <= len(self.seq) - 2 and not self.is_adj(self.state[:, i], self.state[:, i + 1]):
                        i += 1
                        self.state[:, i] = self.prev_state[:, i - 2]

            elif post:
                C = loc + self.prev_state[:, i] - self.prev_state[:, i + 1]
                if i != 0:
                    self.state[:, i - 1] = C
                i -= 2
                if i >= 0:
                    self.state[:, i] = self.prev_state[:, i + 2]
                    while not self.is_adj(self.state[:, i], self.state[:, i - 1]) and i >= 1:
                        i -= 1
                        self.state[:,  i] = self.prev_state[:, i + 2]
            else:
                print("Illegal Move")
                return (False, self.grid, self.reward, done, self.actions)
        
        done = True if (len(self.actions) == len(self.seq) + 1) else False
            
        self.grid = self.update_grid()
        self.reward = self.compute_reward()
        return (True, self.grid, self.reward, done, self.actions)
        
    def compute_reward(self):
        """Computes the reward for a given time step

        For every timestep, we compute the reward using the 
        Gibbs free energy given the chain's state.

        Returns
        -------
        int
            Reward function
        """
        state = []
        for i in range(len(self.seq)):
            if self.seq[i] == 'H':
                state.append((self.state[0][i], self.state[1][i]))
            else:
                state.append((-1000, -1000))
        distances = euclidean_distances(state, state)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        return len(bond_idx[0])
    
    def is_adj(self, x, y):
        """Helper function that checks if x and y are lattice-adjacent
        
        Parameters
        ----------
        x : np.array
            First coordinate
        y : np.array
            Second coordinate
            
        Returns
        -------
        bool
            True if x and y are adjacent
        """
        return True if abs(np.sum(x[0] - y[0])) + abs(np.sum(x[1] - y[1])) == 1 else False
    
    def fourth_vertex(self, x, y, z):
        """Helper function that returns last vertex of a square given 3
        
        Parameters
        ----------
        x : np.array
            First coordinate
        y : np.array
            Second coordinate
        z : np.array
            Third coordinate
            
        Returns
        -------
        np.array
            Fourth coordinate
        """
        return np.array([x[0] ^ y[0] ^ z[0], x[1] ^ y[1] ^ z[1]])
    
    def update_grid(self):
        """Helper function that fills the grid with the current state"""
        self.grid = np.zeros(shape = (self.grid_len, self.grid_len), dtype = int)
        for i in range(len(self.seq)):
            self.grid[self.state[0, i]][self.state[1, i]] = POLY_TO_INT[self.seq[i]]
        return self.grid        
    
    def valid(self, action):
        """Checks if action is valid for the current state"""
        if action > 4 * self.max_len + 8:
            return False
        elif action == 4 * self.max_len + 8: 
            return True
        elif action == 4 * self.max_len + 7:
            return self.grid[self.state[0, -1]][self.state[1, -1] + 2] == 0 and self.grid[self.state[0, -1]][self.state[1, -1] + 1] == 0
        elif action == 4 * self.max_len + 6:
            return self.grid[self.state[0, -1] + 2][self.state[1, -1]] == 0 and self.grid[self.state[0, -1] + 1][self.state[1, -1]] == 0
        elif action == 4 * self.max_len + 5:
            return self.grid[self.state[0, -1]][self.state[1, -1] - 2] == 0 and self.grid[self.state[0, -1]][self.state[1, -1] - 1] == 0
        elif action == 4 * self.max_len + 4:
            return self.grid[self.state[0, -1] - 2][self.state[1, -1]] == 0 and self.grid[self.state[0, -1] - 1][self.state[1, -1]] == 0
        elif action == 4 * self.max_len + 3:
            return self.grid[self.state[0, 0]][self.state[1, 0] + 2] == 0 and self.grid[self.state[0, 0]][self.state[1, 0] + 1] == 0
        elif action == 4 * self.max_len + 2:
            return self.grid[self.state[0, 0] + 2][self.state[1, 0]] == 0 and self.grid[self.state[0, 0] + 1][self.state[1, 0]] == 0
        elif action == 4 * self.max_len + 1:
            return self.grid[self.state[0, 0]][self.state[1, 0] - 2] == 0 and self.grid[self.state[0, 0]][self.state[1, 0] - 1] == 0
        elif action == 4 * self.max_len:
            return self.grid[self.state[0, 0] - 2][self.state[1, 0]] == 0 and self.grid[self.state[0, 0] - 1][self.state[1, 0]] == 0
        elif action < 4 * self.max_len and action >= 4 * len(self.seq):
            return False
        else:
            i = action // 4
            dir = action % 4
            loc = DIR_TO_ARRAY[dir] + self.state[:, i]
            if self.grid[loc[0]][loc[1]] != 0:
                return False
            if i == 0:
                return self.is_adj(loc, self.state[:, 1])
            elif i == len(self.seq) - 1:
                return self.is_adj(loc, self.state[:, -2])
            elif self.is_adj(loc, self.state[:, i - 1]):
                if self.is_adj(loc, self.state[:, i + 1]):
                    return True
                loc2 = self.fourth_vertex(loc, self.state[:, i - 1], self.state[:, i])
                return self.grid[loc2[0]][loc2[1]] == 0
            elif self.is_adj(loc, self.state[:, i + 1]):
                loc2 = self.fourth_vertex(loc, self.state[:, i + 1], self.state[:, i])
                return self.grid[loc2[0]][loc2[1]] == 0
            else:
                return False
            
    def valid_moves(self):
        """Returns an np.array of valid moves
        
        val[a] = 1 if action a is valid
        """
        val = np.zeros(4 * self.max_len + 9)
        for i in range(4 * self.max_len + 9):
            if self.valid(i) == True:
                val[i] = 1
        return val
            
    def reset(self):
        """Resets the environment"""
        self.actions = []
        self.reward = 0
        self.state = np.full((2, len(self.seq)), self.grid_len // 2) #state[0] = x-coords, state[1] = y-coords, state[:,i] = coords for mol i
        for i in range(len(self.seq)):
            self.state[1][i] = self.grid_len // 2 - len(self.seq) // 2 + i
            
        self.prev_state = np.copy(self.state)
        return self.grid
    
    def render(self):
        ''' Renders the environment '''
        # Set up plot
        state_dict = OrderedDict()
        for i in range(len(self.seq)):
            state_dict.update({ (int(self.state[1][i]) - self.grid_len // 2, self.grid_len // 2 - int(self.state[0][i])) : self.seq[i] })
        fig, ax = plt.subplots()
        plt.axis('scaled')
        if len(self.actions) != 0:
            action = self.actions[-1]
            if action == 4 * self.max_len + 8:
                plt.title("{}: STOP".format(len(self.actions)))
            elif action == 4 * self.max_len + 7:
                plt.title("{}: Last position right two".format(len(self.actions)))
            elif action == 4 * self.max_len + 6:
                plt.title("{}: Last position down two".format(len(self.actions)))
            elif action == 4 * self.max_len + 5:
                plt.title("{}: Last position left two".format(len(self.actions)))
            elif action == 4 * self.max_len + 4:
                plt.title("{}: Last position up two".format(len(self.actions)))
            elif action == 4 * self.max_len + 3:
                plt.title("{}: First position right two".format(len(self.actions)))
            elif action == 4 * self.max_len + 2:
                plt.title("{}: First position down two".format(len(self.actions)))
            elif action == 4 * self.max_len + 1:
                plt.title("{}: First position left two".format(len(self.actions)))
            elif action == 4 * self.max_len:
                plt.title("{}: First position up two".format(len(self.actions)))
            else:
                i = action // 4
                dir = action % 4
                plt.title("{}: Pull position {} {}".format(len(self.actions), i + 1, ACTION_TO_DIR[dir]))
        else: 
            plt.title("0: Starting Position")
        xmid = (min(self.state[1]) + max(self.state[1])) / 2 - self.grid_len // 2
        ymid = -(min(self.state[0]) + max(self.state[0])) / 2 + self.grid_len // 2
        bd = len(self.seq) / 2
        ax.set_xlim([xmid - bd - 0.5 , xmid + bd + 0.5])
        ax.set_ylim([ymid - bd + 0.5 , ymid + bd + 1.5])
        
        # Plot chain
        dictlist = list(state_dict.items())
        curr_state = dictlist[0]
        mol = plt.Circle(curr_state[0], 0.2, color = 'green' if curr_state[1] == 'H' else 'gray', zorder = 1)
        ax.add_artist(mol)
        mol = plt.Circle(curr_state[0], 0.3, color = 'blue', fill = False, zorder = 2)
        ax.add_artist(mol)
        for i in range(1, len(dictlist)):
            next_state = dictlist[i]
            xdata = [curr_state[0][0], next_state[0][0]]
            ydata = [curr_state[0][1], next_state[0][1]]
            bond = mlines.Line2D(xdata, ydata, color = 'k', zorder = 0)
            ax.add_line(bond)
            mol = plt.Circle(next_state[0], 0.2, color = 'green' if next_state[1] == 'H' else 'gray', zorder = 1)
            ax.add_artist(mol)
            curr_state = next_state
        
        # Show H-H bonds
        ## Compute all pair distances for the bases in the configuration
        state = []
        for i in range(len(dictlist)):
            if dictlist[i][1] == 'H':
                state.append(dictlist[i][0])
            else:
                state.append((-1000, 1000)) #To get rid of P's
        distances = euclidean_distances(state, state)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        for (x,y) in zip(*bond_idx):
            xdata = [state[x][0], state[y][0]]
            ydata = [state[x][1], state[y][1]]
            backbone = mlines.Line2D(xdata, ydata, color = 'r', ls = ':', zorder = 1)
            ax.add_line(backbone)