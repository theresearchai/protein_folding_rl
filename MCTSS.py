"""
MCTS for pretraining for AlphaGo Zero under lattice environment
"""

from lattice2d_linear_env import Lattice2DLinearEnv
from HP2D_Env import HP2D
import numpy as np
import math
import random

C = 1.41 # Constant in MCTS exploration

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
            
class MCTSS():
    def __init__(self, seq, max_iter, shape):
        self.env = Lattice2DLinearEnv(seq)
        self.seq = seq
        self.max_iter = max_iter
        self.shape = shape
        
    def get_prob(self, root):
        for i in range(self.max_iter):
            leaf = self.traverse(root)
            sim_result = self.rollout(leaf)
            leaf.update(sim_result)

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
        Picks child with the highest average reward
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
        '''
        Formats MCTS data into desired form for training
        '''
        states = []
        probs = []
        testenv = HP2D(self.seq, self.shape)
        data = []
        state = self.env.reset()
        teststate = testenv.make_state()
        reward = 0
        a = 0
        traj = [a]
        while True:
            self.env.reset()
            for i in traj:
                _, reward, done, info = self.env.step(i)
            states.append(teststate)
            p = [0,0,0,0]
            p[a] = 1
            probs.append(p)
            teststate = testenv.next_state(teststate, a)
            if done:
                break
            best = self.get_prob(root)
            root = root.children[best]
            a = (3 * a + best) % 4
            traj += [a]
        if testenv.hyp_max() != 0:
            reward /= testenv.hyp_max()
        for i in range(len(states)):
            data += [(states[i], probs[i], reward)]
        return data