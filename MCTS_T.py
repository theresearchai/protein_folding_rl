from lattice2d_linear_env import Lattice2DLinearEnv
import numpy as np
import math
import random

C = 1.41 # Constant in MCTS exploration

stoid = { "H" : 1, "P" : 0 }

class MCTSNode():
    def __init__(self, state, parent = None):
        self.state = state # state = tuple of ele in (0,1,2) denoting actions taken thus far
        self.Q = 0 # Total reward
        self.N = 0 # Total times state is visited
        self.children = {} # Dict of children: Action : Child
        self.parent = parent # Parent of node
        self.score = 0.0 # score = average reward + exploration weight
        self.avg_Q = 0.0 # average reward
        
    def add_child(self, child, action):
        self.children.update( {action: child} )
    
    def update(self, Q):
        if self.parent != None:
            self.Q += Q
            self.N += 1
            self.score = self.Q / self.N + 2 * C * math.sqrt(2 * math.log(self.parent.N + 1) / self.N)
            self.avg_Q = self.Q / self.N
            self.parent.update(Q) 
            
class MCTS_T():
    def __init__(self, seq, max_iter, max_len):
        self.env = Lattice2DLinearEnv(seq)
        self.seq = seq
        self.max_iter = max_iter
        self.max_len = max_len
        
    def get_prob(self, root):
        for i in range(self.max_iter):
            leaf = self.traverse(root)
            sim_result = self.rollout(leaf)
            leaf.update(sim_result)
            
        """
        counts = [root.children[0].N, root.children[1].N, root.children[2].N]
        
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
        """
        return self.best_action(root)
    
    def traverse(self, node):
        '''
        Function for node traversal
        '''

        # While node is fully expanded, pick child with best score
        while len(list(node.children.keys())) == 3 and len(node.state) <= len(self.env.seq) - 1:
            node = self.best_uct(node)

        # If the node is terminal
        if len(node.state) == len(self.env.seq) - 1:
            return node

        # If not fully expanded, pick unvisited child.
        if len(list(node.children.values())) != 3: 
            unvisited = [a for a in range(3) if a not in list(node.children.keys())] # list of previously untaken actions
            action = random.choice(unvisited)
            child = MCTSNode(node.state + (action,), node)
            node.add_child(child, action)
            return child
    
    def rollout(self, node):
        '''
        Function for simulating game starting from given node.
        Returns simulated reward
        '''
        trajectory = node.state
        while len(trajectory) != len(self.env.seq) - 1:
            trajectory += (random.choice(range(3)),) # simulate random policy until end of the game

        # Compute reward
        state = self.env.reset()
        a = 0
        reward = 0
        for i in range(len(trajectory)):
            _, reward, _, _ = self.env.step(a)
            a = (3 * a + trajectory[i]) % 4
        return reward
    
    def best_action(self, node):
        '''
        Picks child with the highest count
        '''
        if len(list(node.children.keys())) == 0:
            action = random.choice(range(3))
        action = list(node.children.keys())[0]
        for i in list(node.children.keys()):
            if node.children[i].N > node.children[action].N:
                action = i
        return action
    
    def best_uct(self, node):
        '''
        Picks child with the highest score
        '''
        child = node.children[0]
        for i in range(3):
            if node.children[i].score > child.score:
                child = node.children[i]
        return child
    
    def get_data(self, root):
        data = []
        self.env.reset()
        reward = 0
        a = 0
        traj = [a]
        n = 1
        while True:
            self.env.reset()
            print(traj)
            for i in traj:
                _, reward, done, info = self.env.step(i)
                reward = self.env.get_reward(info['is_trapped'], info['collisions'])
            print(reward)
            if n >= 2:
                src = [(stoid[self.seq[j]] + 1) for j in range(n + 1)] + [0] * (self.max_len - n - 1)
                sym_trg = self.get_syms(traj)
                for t in sym_trg:
                    ti = np.concatenate((t + np.array([1] * (n)), np.array([0] * (self.max_len - n))), 0)
                    data += [(src, ti, reward / self.hyp_max())]
                
            if done:
                break
            best = self.get_prob(root)
            root = root.children[best]
            a = (3 * a + best) % 4
            traj += [a]
            n += 1

        return data
    
    def get_syms(self, trg):
        res = []
        trg = np.array(trg)
        t = np.copy(trg)
        res.append(trg)
        w1 = np.where(trg == 1)
        w2 = np.where(trg == 2)
        t[w1] = 2
        t[w2] = 1
        res.append(t)
        return res
    
    def src_pad(self, x):
        res = np.full()
        
    
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