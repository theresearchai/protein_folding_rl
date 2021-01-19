import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import euclidean_distances

ACTION_TO_ARRAY = {
    1 : (0, -1),
    2 : (-1, 0),
    3 : (1, 0),
    4 : (0, 1)
}

POLY_TO_INT = {
    'H' : 1, 'P' : -1, ' ' : 0
}

class RNN_Env():
    '''
    2D Lattice Environment for HP Protein Folding for RNN + MCTS
    '''
    
    def __init__(self, seq, max_seq_length):
        self.seq = seq
        self.maxl = max_seq_length - 1
        
    def make_state(self):
        state = np.zeros(self.maxl)
        return state
    
    def stringrep(self, state):
        ''' 
        String representation of state
        Represent state by actions taken to get there
        '''
        
        return ''.join([str(int(s)) for s in state])
    
    def done(self, state):
        '''
        Return 0 if not ended, 1 if done
        '''
        if np.count_nonzero(state) == len(self.seq) - 1:
            return 1
        return 0
        
    def valid_moves(self, state):
        board = {(0, 0) : 1 }
        curr = (0, 0)
        for a in state:
            if a != 0:
                curr = tuple([sum(x) for x in zip(curr, ACTION_TO_ARRAY[a])])
                board.update({ curr : 1 })
            else:
                break
        vm = [1, 1, 1, 1, 1]
        for a in range(1, 5):
            if tuple([sum(x) for x in zip(curr, ACTION_TO_ARRAY[a])]) in board:
                vm[a] = 0
        
        return vm        
        
    def next_state(self, state, action):
        n = np.count_nonzero(state)
        res = state
        res[n] = action
        return res
    
    def calc_score(self, state):
        num_mol = np.count_nonzero(state) + 1
        # If trapped: penalty = -(length remaining chain)
        trapped_penalty = num_mol - len(self.seq)
        
        ctr = 0
        board = { (0, 0) : self.seq[ctr] }
        curr = (0, 0)
        for a in state:
            print(board)
            ctr += 1
            if a != 0:
                curr = tuple([sum(x) for x in zip(curr, ACTION_TO_ARRAY[a])])
                board.update({ curr : self.seq[ctr] })
            else:
                break
        print(board.items())

        # Show H-H bonds
        ## Compute all pair distances for the bases in the configuration
        grid = []
        dictlist = list(board.items())
        for i in range(len(dictlist)):
            if dictlist[i][1] == 'H':
                grid.append(dictlist[i][0])
            else:
                grid.append((-1000, 1000)) #To get rid of P's
        distances = euclidean_distances(grid, grid)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        return len(bond_idx[0]) + trapped_penalty