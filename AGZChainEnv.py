"""
Implements the Chain Environment with pull moves for AlphaGo Zero MCTS. 

Note that this is distinct from Chain_Env.py in that it incorporates
a different structure of the data, as well as a few additional methods
such as stringrep and hyp_max.
"""

import numpy as np
from Chain_Env import ChainEnv
from sklearn.metrics.pairwise import euclidean_distances

POLY_TO_INT = {
    'H' : 1, 'P' : -1, ' ' : 0
}

DIR_TO_ARRAY = {
    0  : np.array([-1, 1]),
    1  : np.array([-1, -1]),
    2  : np.array([1, -1]),
    3  : np.array([1, 1]),
    -2 : np.array([1, -1]),
    -1 : np.array([1, 1])
}


class Chain():
    """A 2-dimensional chain environment with pull moves, inspired by
    N. Lesh, M. Mitzenmacher, and S. Whitesides.

    Refer to Chain_Env.py for additional information on how the
    environment works.

    Attributes
    ----------
    seq : str
        Polymer sequence describing a particular protein.
    env : ChainEnv
        Chain environment for the given sequence.
    grid_len : int
        Length of one side of the grid.
    max_len : int
        Maximum length for a sequence.
    shape : tuple
        Shape of the data
        
    """
    def __init__(self, seq, max_len = 100, grid_len = 201, max_moves = 100):
        self.seq = seq
        self.env = ChainEnv(seq, max_len, grid_len)
        self.max_len = self.env.max_len
        self.grid_len = self.env.grid_len
        self.shape = (max_len + 2, grid_len, grid_len) #2nd to last channel: # actions so far, last channel: 1 if done
        self.max_moves = grid_len
        
    def make_state(self):
        """Returns the starting state in the desired data format
        
        Returns
        -------
        np.array
            Starting state
        """
        state = np.zeros(self.shape)
        for i in range(len(self.seq)):
            state[i][self.grid_len // 2][self.grid_len // 2 - len(self.seq) // 2 + i] = POLY_TO_INT[self.seq[i]]
        return state
    
    def stringrep(self, state):
        """String representation of a state
        
        The string representation is found by simply
        hashing the 2D representation of the sequence.
        
        Parameters
        -------
        state : np.array
            Current state
            
        Returns
        -------
        string
            String representation of state
        """
        # convert to 2D rep, then do tostring()
        state2d = np.zeros((self.grid_len, self.grid_len))
        pos = np.nonzero(state)
        for i in range(len(self.seq)):
            state2d[pos[1][i]][pos[2][i]] = POLY_TO_INT[self.seq[i]]
        return state2d.tostring()
    
    def done(self, state):
        '''
        Return 0 if not ended, 1 if done
        '''
        return state[-1][0][0] or state[-2][0][0] > self.max_moves
        
    def valid_moves(self, state):
        """Returns an np.array of valid moves
        
        val[a] = 1 if action a is valid
        """
        vm = np.zeros(4 * self.max_len + 9)
        for a in range(4 * self.max_len + 9):
            if self.is_valid(a, state):
                vm[a] = 1
            
        return vm
    
    def is_adj(self, x, y):
        return True if abs(np.sum(x[0] - y[0])) + abs(np.sum(x[1] - y[1])) == 1 else False
    
    def fourth_vertex(self, x, y, z):
        return np.array([x[0] ^ y[0] ^ z[0], x[1] ^ y[1] ^ z[1]])
    
    def is_valid(self, action, state):
        '''
        Check if action is a valid action
        '''
        locs = np.nonzero(state[:-2])[1:]
        grid = np.zeros((self.grid_len, self.grid_len))
        for i in range(len(locs[0])):
            grid[locs[0][i]][locs[1][i]] = POLY_TO_INT[self.seq[i]]
        if action > 4 * self.max_len + 8:
            return False
        elif action == 4 * self.max_len + 8: 
            return True
        elif action == 4 * self.max_len + 7:
            return grid[locs[0][-1]][locs[1][-1] + 2] == 0 and grid[locs[0][-1]][locs[1][-1] + 1] == 0
        elif action == 4 * self.max_len + 6:
            return grid[locs[0][-1] + 2][locs[1][-1]] == 0 and grid[locs[0][-1] + 1][locs[1][-1]] == 0
        elif action == 4 * self.max_len + 5:
            return grid[locs[0][-1]][locs[1][-1] - 2] == 0 and grid[locs[0][-1]][locs[1][-1] - 1] == 0
        elif action == 4 * self.max_len + 4:
            return grid[locs[0][-1] - 2][locs[1][-1]] == 0 and grid[locs[0][-1] - 1][locs[1][-1]] == 0
        elif action == 4 * self.max_len + 3:
            return grid[locs[0][0]][locs[1][0] + 2] == 0 and grid[locs[0][0]][locs[1][0] + 1] == 0
        elif action == 4 * self.max_len + 2:
            return grid[locs[0][0] + 2][locs[1][0]] == 0 and grid[locs[0][0] + 1][locs[1][0]] == 0
        elif action == 4 * self.max_len + 1:
            return grid[locs[0][0]][locs[1][0] - 2] == 0 and grid[locs[0][0]][locs[1][0] - 1] == 0
        elif action == 4 * self.max_len:
            return grid[locs[0][0] - 2][locs[1][0]] == 0 and grid[locs[0][0] - 1][locs[1][0]] == 0
        elif action < 4 * self.max_len and action >= 4 * len(self.seq):
            return False
        else:
            i = action // 4
            dir = action % 4
            loc = DIR_TO_ARRAY[dir] + np.array([locs[0][i], locs[1][i]])
            if grid[loc[0]][loc[1]] != 0:
                return False
            if i == 0:
                return self.is_adj(loc, np.array([locs[0][1], locs[1][1]]))
            elif i == len(self.seq) - 1:
                return self.is_adj(loc, np.array([locs[0][-2], locs[1][-2]]))
            elif self.is_adj(loc, np.array([locs[0][i - 1], locs[1][i - 1]])):
                if self.is_adj(loc, np.array([locs[0][i + 1], locs[1][i + 1]])):
                    return True
                loc2 = self.fourth_vertex(loc, np.array([locs[0][i - 1], locs[1][i - 1]]), np.array([locs[0][i], locs[1][i]]))
                return grid[loc2[0]][loc2[1]] == 0
            elif self.is_adj(loc, np.array([locs[0][i + 1], locs[1][i + 1]])):
                loc2 = self.fourth_vertex(loc, np.array([locs[0][i + 1], locs[1][i + 1]]), np.array([locs[0][i], locs[1][i]]))
                return grid[loc2[0]][loc2[1]] == 0
            else:
                return False
        
    def next_state(self, state, action):
        """Computes the next state given the current state and action"""
        locs = np.array(np.nonzero(state[:-2])[1:])
        prev_locs = np.array(np.nonzero(state[:-2])[1:])
        if action == 4 * self.max_len + 8:
            new_state = np.copy(state)
            new_state[-1] = np.ones((self.grid_len, self.grid_len))
            return new_state
        elif action >= 4 * self.max_len + 4: #For the last mol
            if action == 4 * self.max_len + 7:
                locs[1][-1] += 2
                locs[0][-2] = locs[0][-1]
                locs[1][-2] = locs[1][-1] - 1
            elif action == 4 * self.max_len + 6:
                locs[0][-1] += 2
                locs[0][-2] = locs[0][-1] - 1
                locs[1][-2] = locs[1][-1]
            elif action == 4 * self.max_len + 5:
                locs[1][-1] -= 2
                locs[0][-2] = locs[0][-1]
                locs[1][-2] = locs[1][-1] + 1
            else:
                locs[0][-1] -= 2
                locs[0][-2] = locs[0][-1] + 1
                locs[1][-2] = locs[1][-1]
                
            i = len(self.seq) - 3
            locs[:, i] = prev_locs[:, i + 2]
            while not self.is_adj(locs[:, i], locs[:, i - 1]) and i >= 1:
                i -= 1
                locs[:,  i] = prev_locs[:, i + 2]
                
        elif action >= 4 * self.max_len: #For the first mol
            if action == 4 * self.max_len + 3:
                locs[1][0] += 2
                locs[0][1] = locs[0][0]
                locs[1][1] = locs[1][0] - 1
            elif action == 4 * self.max_len + 2:
                locs[0][0] += 2
                locs[0][1] = locs[0][0] - 1
                locs[1][1] = locs[1][0]
            elif action == 4 * self.max_len + 1:
                locs[1][0] -= 2
                locs[0][1] = locs[0][0]
                locs[1][1] = locs[1][0] + 1
            else:
                locs[0][0] -= 2
                locs[0][1] = locs[0][0] + 1
                locs[1][1] = locs[1][0]
                
            i = 2
            locs[:, i] = prev_locs[:, i - 2]
            while i <= len(self.seq) - 2 and not self.is_adj(locs[:, i], locs[:, i + 1]):
                i += 1
                locs[:, i] = prev_locs[:, i - 2]
                
        else: #For all other mols    
            i = action // 4
            dir = action % 4
            loc = DIR_TO_ARRAY[dir] + prev_locs[:, i]
            locs[:, i] = loc
            pre = True if i == len(self.seq) - 1 else False
            post = True if i == 0 else False
            if i != 0 and i != len(self.seq) - 1:
                pre = True if self.is_adj(loc, locs[:, i - 1]) else False # i - 1 adjacent to i's new location
                post = True if self.is_adj(loc, locs[:, i + 1]) else False # i + 1 adjacent to i's new location
            if pre and post:
                new_state = np.copy(state)
                new_state[i][prev_locs[0, i]][prev_locs[1, i]] = 0
                new_state[i][locs[0, i]][locs[1, i]] = POLY_TO_INT[self.seq[i]]
                new_state[-2] = np.full((self.grid_len, self.grid_len), state[-2][0][0] + 1)
                return new_state
            elif pre:
                C = loc + prev_locs[:, i] - prev_locs[:, i - 1]
                if i != len(self.seq) - 1:
                    locs[:, i + 1] = C
                i += 2
                if i <= len(self.seq) - 1:
                    locs[:, i] = prev_locs[:, i - 2]
                    while i <= len(self.seq) - 2 and not self.is_adj(locs[:, i], locs[:, i + 1]):
                        i += 1
                        locs[:, i] = prev_locs[:, i - 2]

            elif post:
                C = loc + prev_locs[:, i] - prev_locs[:, i + 1]
                if i != 0:
                    locs[:, i - 1] = C
                i -= 2
                if i >= 0:
                    locs[:, i] = prev_locs[:, i + 2]
                    while not self.is_adj(locs[:, i], locs[:, i - 1]) and i >= 1:
                        i -= 1
                        locs[:,  i] = prev_locs[:, i + 2]
            else:
                print("Illegal Move")
                return state
        new_state = np.zeros_like(state)
        for i in range(len(self.seq)):
            new_state[i][locs[0][i]][locs[1][i]] = POLY_TO_INT[self.seq[i]]
        new_state[-2] = np.full((self.grid_len, self.grid_len), state[-2][0][0] + 1)
        return new_state
        
    def calc_score(self, state):
        """Computes the reward for a given time step

        For every timestep, we compute the reward using the 
        Gibbs free energy given the chain's state.

        Returns
        -------
        int
            Reward function
        """
        locs = np.array(np.nonzero(state)[1:])
        grid = []
        for i in range(len(self.seq)):
            if self.seq[i] == 'H':
                grid.append((locs[0][i], locs[1][i]))
            else:
                grid.append((-1000, -1000))
        distances = euclidean_distances(grid, grid)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        return len(bond_idx[0])
    
    def hyp_max(self):
        """Hypothetical maximum reward for a sequence
        
        Computed as 2 * min(odd, even), where odd is the number 
        of H polymers in the "odd" positions of the sequence, and
        even is the number of H polymers in the "even" positions of 
        the sequence.        
        """
        odd = 0
        even = 0
        for i in range(len(self.seq)):
            if self.seq[i] == 'H':
                if i % 2 == 1:
                    odd += 1
                else:
                    even += 1
        return 2 * np.minimum(odd, even)