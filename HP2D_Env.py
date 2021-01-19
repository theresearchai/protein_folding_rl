"""
Implements the 2D Lattice Environment for AlphaGo Zero MCTS. 

Note that this is distinct from lattice2d_env_linear.py in that it incorporates
a different structure of the data, as well as a few additional methods
such as stringrep and hyp_max.
"""
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import euclidean_distances

ACTION_TO_ARRAY = {
    0 : np.array([0, -1]),
    1 : np.array([-1, 0]),
    2 : np.array([1, 0]),
    3 : np.array([0, 1])
}

POLY_TO_INT = {
    'H' : 1, 'P' : -1, ' ' : 0
}

class HP2D():
    '''
    2D Lattice Environment for HP Protein Folding for AlphaGo Zero Algorithm
    '''
    
    def __init__(self, seq, shape):
        self.seq = seq
        if len(seq) < 5:
            self.seq += ' '
        self.shape = shape
        
    def make_state(self):
        state = np.zeros(self.shape)
        num_channels = self.shape[0]
        for i in range(6, num_channels):
            state[i] = np.full((self.shape[1], self.shape[2]), POLY_TO_INT[self.seq[i - 5]])
        state[0][self.shape[1] // 2][self.shape[2] // 2] = POLY_TO_INT[self.seq[0]]
        return state
    
    def get_pos(self, state):
        return np.argwhere(np.array([state[0] != state[3]]) == True)[0]     
    
    def stringrep(self, state):
        ''' 
        String representation of state
        Represent state by actions taken to get there
        '''
        actions = []
        num_mol = np.count_nonzero(state[0])
        curr = np.array([self.shape[1] // 2, self.shape[2] // 2])
        hor_adj = np.copy(state[1])
        vert_adj = np.copy(state[2])        
        for i in range(num_mol - 1):
            a = 0
            if int(hor_adj[(curr + np.array([0, -1]))[0], (curr + np.array([0, -1]))[1]]) == 1:
                a = 0
                hor_adj[(curr + np.array([0, -1]))[0], (curr + np.array([0, -1]))[1]] = 0
            if int(hor_adj[curr[0], curr[1]]) == 1:
                a = 3
                hor_adj[curr[0], curr[1]] = 0
            if int(vert_adj[curr[0], curr[1]]) == 1:
                a = 2
                vert_adj[curr[0], curr[1]] = 0
            if int(vert_adj[(curr + np.array([-1, 0]))[0], (curr + np.array([-1, 0]))[1]]) == 1:
                a = 1
                vert_adj[(curr + np.array([-1, 0]))[0], (curr + np.array([-1, 0]))[1]] = 0
            actions.append(a)
            curr += ACTION_TO_ARRAY[a]
                
        actions_str = [str(a) for a in actions]
        return ''.join(actions_str)
    
    def done(self, state):
        '''
        Return 0 if not ended, 1 if done
        '''
        if state[6][0][0] == 0:
            return 1
        if self.valid_moves(state) == [0,0,0,0]:
            return 1
        return 0
        
    def valid_moves(self, state):
        last_pos = np.array(self.get_pos(state)[1:])
        vm = [0, 0, 0, 0]
        for a in range(4):
            if self.is_valid(last_pos + ACTION_TO_ARRAY[a], state):
                vm[a] = 1
            
        return vm
    
    def is_valid(self, pos, state):
        '''
        Check if pos is a valid position
        '''
        if pos[0] not in np.argwhere(np.zeros(state[0].shape) == 0):
            return False
        if pos[1] not in np.argwhere(np.zeros(state[0].shape) == 0):
            return False
        if state[0][pos[0]][pos[1]] != 0:
            return False
        return True
        
    def next_state(self, state, action):
        add_mol = int(state[6][0][0])
        last_pos = np.array(self.get_pos(state)[1:])
        next_pos = last_pos + ACTION_TO_ARRAY[action]
        if not self.is_valid(next_pos, state):
            # print('Illegal action')
            # print(last_pos, action)
            return state
        num_mol = np.count_nonzero(state[0])
        if num_mol + len(state) - 6 < len(self.seq):
            next_mol = POLY_TO_INT[self.seq[num_mol + len(state) - 6]]
        else:
            next_mol = 0
        ns = np.copy(state)
        ns[3:6] = state[0:3]
        ns[6:-1] = state[7:]
        ns[-1] = np.full(state[0].shape, next_mol)
        
        ns[0][next_pos[0]][next_pos[1]] = add_mol
        if action == 0:
            ns[1][next_pos[0]][next_pos[1]] = 1
        if action == 3:
            ns[1][last_pos[0]][last_pos[1]] = 1
        if action == 2:
            ns[2][last_pos[0]][last_pos[1]] = 1
        if action == 1:
            ns[2][next_pos[0]][next_pos[1]] = 1
        return ns
    
    def calc_score(self, state):
        num_mol = np.count_nonzero(state[0])
        # If trapped: penalty = -(length remaining chain)
        trapped_penalty = num_mol - len(self.seq)
        curr = np.array([self.shape[1] // 2, self.shape[2] // 2])
        odict = OrderedDict({(curr[0], curr[1]) : self.seq[0]})
        hor_adj = np.copy(state[1])
        vert_adj = np.copy(state[2])        
        for i in range(num_mol - 1):
            a = 0
            if int(hor_adj[(curr + np.array([0, -1]))[0], (curr + np.array([0, -1]))[1]]) == 1:
                a = 0
                hor_adj[(curr + np.array([0, -1]))[0], (curr + np.array([0, -1]))[1]] = 0
            if int(hor_adj[curr[0], curr[1]]) == 1:
                a = 3
                hor_adj[curr[0], curr[1]] = 0
            if int(vert_adj[curr[0], curr[1]]) == 1:
                a = 2
                vert_adj[curr[0], curr[1]] = 0
            if int(vert_adj[(curr + np.array([-1, 0]))[0], (curr + np.array([-1, 0]))[1]]) == 1:
                a = 1
                vert_adj[(curr + np.array([-1, 0]))[0], (curr + np.array([-1, 0]))[1]] = 0
            curr += ACTION_TO_ARRAY[a]
            odict.update({(curr[0], curr[1]) : self.seq[i + 1]})

        # Show H-H bonds
        ## Compute all pair distances for the bases in the configuration
        state = []
        dictlist = list(odict.items())
        for i in range(len(dictlist)):
            if dictlist[i][1] == 'H':
                state.append(dictlist[i][0])
            else:
                state.append((-1000, 1000)) #To get rid of P's
        distances = euclidean_distances(state, state)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        if self.hyp_max() != 0:
            return (len(bond_idx[0]) + trapped_penalty) / self.hyp_max()
        return 0
    
    def hyp_max(self):
        odd = 0
        even = 0
        for i in range(len(self.seq)):
            if self.seq[i] == 'H':
                if i % 2 == 1:
                    odd += 1
                else:
                    even += 1
        return 2 * np.minimum(odd, even)