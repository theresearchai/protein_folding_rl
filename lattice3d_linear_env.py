# -*- coding: utf-8 -*-

"""
Implements the 3D Lattice Environment
"""
# Import gym modules
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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

# Human-readable
ACTION_TO_STR = {
    0 : 'L', 1 : 'F',
    2 : 'U', 3 : 'D',
    4 : 'B', 5 : 'R'}

POLY_TO_INT = {
    'H' : 1, 'P' : -1
}

class Lattice3DLinearEnv(gym.Env):
    """A 3-dimensional lattice environment from Dill and Lau, 1989
    [dill1989lattice]_.

    It follows an absolute Cartesian coordinate system, the location of
    the polymer is stated independently from one another. Thus, we have
    six actions (left, right, up, down, forward, and backward) and a chance of collision.

    The environment will first place the initial polymer at the origin. Then,
    for each step, agents place another polymer to the lattice. An episode
    ends when all polymers are placed, i.e. when the length of the action
    chain is equal to the length of the input sequence minus 1. We then
    compute the reward using the energy minimization rule while accounting
    for the collisions and traps.

    Attributes
    ----------
    seq : str
        Polymer sequence describing a particular protein.
    state : OrderedDict
        Dictionary of the current polymer chain with coordinates and
        polymer type (H or P).
    actions : list
        List of actions performed by the model.
    collisions : int
        Number of collisions incurred by the model.
    trapped : int
        Number of times the agent was trapped.
    grid_length : int
        Length of one side of the grid.
    midpoint : tuple
        Coordinate containing the midpoint of the grid.
    grid : numpy.ndarray
        Actual grid containing the polymer chain.

    .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
    mechanics model of the conformational and se quence spaces of proteins.
    Marcromolecules 22(10), 3986–3997 (1989)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seq, collision_penalty=-2, trap_penalty=0.5, dp = False):
        """Initializes the lattice

        Parameters
        ----------
        seq : str, must only consist of 'H' or 'P'
            Sequence containing the polymer chain.
        collision_penalty : int, must be a negative value
            Penalty incurred when the agent made an invalid action.
            Default is -2.
        trap_penalty : float, must be between 0 and 1
            Penalty incurred when the agent is trapped. Actual value is
            computed as :code:`floor(length_of_sequence * trap_penalty)`
            Default is 0.5.

        Raises
        ------
        AssertionError
            If a certain polymer is not 'H' or 'P'
        """
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
            
        try:
            if collision_penalty >= 0:
                raise ValueError("%r (%s) must be negative" %
                                 (collision_penalty, type(collision_penalty)))
            if not isinstance(collision_penalty, int):
                raise ValueError("%r (%s) must be of type 'int'" %
                                 (collision_penalty, type(collision_penalty)))
            self.collision_penalty = collision_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'int'" %
                         (collision_penalty, type(collision_penalty)))
            raise

        try:
            if not 0 < trap_penalty < 1:
                raise ValueError("%r (%s) must be between 0 and 1" %
                                 (trap_penalty, type(trap_penalty)))
            self.trap_penalty = trap_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'float'" %
                         (trap_penalty, type(trap_penalty)))
            raise

        self.state = OrderedDict({(0, 0, 0) : self.seq[0]})
        self.actions = []
        self.collisions = 0
        self.trapped = 0

        # Grid attributes
        self.grid_length = 51 #Maximum seq length 25
        self.midpoint = (25, 25, 25)
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length, self.grid_length), dtype=int)

        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]

        # Define action-observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.grid_length * self.grid_length * self.grid_length,),
                                            dtype=int)
        self.last_action = None
        
        if dp:
            # For DP Algorithms
            # P represents the transition probabilities of the environment
            # P[s][a] is a tuple (next_state, reward, done)
            # nS is the number of states
            # nA is the number of actions
            # Denote states by the actions taken to get there (left, straight, up)
            # Encode them as ternary numbers
            # Assume the first step is left
            self.nS = int((5**(len(self.seq)- 1) + 1) / 2)
            self.nA = 5
            self.P = [[(0, 0, False) for i in range(self.nA)] for j in range(self.nS)]

            self.states_dic = {}
            self.fill_P()

    def step(self, action):
        """Updates the current chain with the specified action.

        The action supplied by the agent should be an integer from 0
        to 5. In this case:
            - 0 : left (x, -1)
            - 1 : forward (y, +1)
            - 2 : up (z, +1)
            - 3 : down (z, -1)
            - 4 : backwards (y, -1)
            - 5 : right (x, +1)

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
        action : int, {0, 1, 2, 3, 4, 5}
            Specifies the position where the next polymer will be placed
            relative to the previous one:
                - 0 : left
                - 1 : forward
                - 2 : up
                - 3 : down
                - 4 : backwards
                - 5 : right

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
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        self.last_action = action
        is_trapped = False # Trap signal
        collision = False  # Collision signal
        # Obtain coordinate of previous polymer
        x, y, z = next(reversed(self.state))
        # Get all adjacent coords and next move based on action
        adj_coords = self._get_adjacent_coords((x, y, z))
        next_move = adj_coords[action]
        # Detects for collision or traps in the given coordinate
        idx = len(self.state)
        if set(adj_coords.values()).issubset(self.state.keys()):
            logger.warn('Your agent was trapped! Ending the episode.')
            self.trapped += 1
            is_trapped = True
        elif next_move in self.state:
            self.collisions += 1
            collision = True
        else:
            self.actions.append(action)
            try:
                self.state.update({next_move : self.seq[idx]})
            except IndexError:
                logger.error('All molecules have been placed! Nothing can be added to the protein chain.')
                raise

        # Set-up return values
        grid = self._draw_grid(self.state)
        done = True if (len(self.state) == len(self.seq) or is_trapped) else False
        reward = self._compute_reward(is_trapped, collision, done)
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : len(self.seq),
            'collisions'   : self.collisions,
            'actions'      : [ACTION_TO_STR[i] for i in self.actions],
            'is_trapped'   : is_trapped,
            'state_chain'  : self.state
        }

        return (grid.flatten(), reward, done, info)

    def reset(self):
        """Resets the environment"""
        self.state = OrderedDict({(0, 0, 0) : self.seq[0]})
        self.actions = []
        self.last_action = None
        self.collisions = 0
        self.trapped = 0
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length, self.grid_length), dtype=int)
        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]

        return self.grid.flatten()

    def render(self):
        ''' Renders the environment '''
        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.grid(b = False)
        ax.autoscale(True, 'both')
        if self.last_action is not None:
            plt.title('{}: {}'.format(len(self.actions), ["Left", "Forward", "Up", "Down", "Backward", "Right"][self.last_action]))
        else:
            plt.title('0: Starting position')
        bd = 2
        ax.set_xlim([-0.5 - bd, 0.5 + bd])
        ax.set_ylim([-0.5 - bd, 0.5 + bd])
        ax.set_zlim([-0.5 - bd, 0.5 + bd])
        
        # Plot chain
        dictlist = list(self.state.items())
        curr_state = dictlist[0]
        ax.scatter(curr_state[0][0], curr_state[0][1], curr_state[0][2], color = 'green' if curr_state[1] == 'H' else 'gray', s = 50)
        for i in range(1, len(dictlist)):
            next_state = dictlist[i]
            xdata = [curr_state[0][0], next_state[0][0]]
            ydata = [curr_state[0][1], next_state[0][1]]
            zdata = [curr_state[0][2], next_state[0][2]]
            ax.plot3D(xdata, ydata, zdata, color = 'k')
            ax.scatter(next_state[0][0], next_state[0][1], next_state[0][2], color = 'green' if next_state[1] == 'H' else 'gray', s = 50)
            curr_state = next_state
        
        # Show H-H bonds
        ## Compute all pair distances for the bases in the configuration
        state = []
        for i in range(len(dictlist)):
            if dictlist[i][1] == 'H':
                state.append(dictlist[i][0])
            else:
                state.append((-1000, 1000, 1000)) #To get rid of P's
        distances = euclidean_distances(state, state)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        for (x,y) in zip(*bond_idx):
            xdata = [state[x][0], state[y][0]]
            ydata = [state[x][1], state[y][1]]
            zdata = [state[x][2], state[y][2]]
            ax.plot3D(xdata, ydata, zdata, color = 'r', ls = ':')

    def _get_adjacent_coords(self, coords):
        """Obtains all adjacent coordinates of the current position

        Parameters
        ----------
        coords : 2-tuple
            Coordinates (X-y) of the current position

        Returns
        -------
        dictionary
            All adjacent coordinates
        """
        x, y, z = coords
        adjacent_coords = {
            0 : (x - 1, y, z),
            1 : (x, y - 1, z),
            2 : (x, y, z + 1),
            3 : (x, y, z - 1),
            4 : (x, y + 1, z),
            5 : (x + 1, y, z)
        }

        return adjacent_coords

    def _draw_grid(self, chain):
        """Constructs a grid with the current chain

        Parameters
        ----------
        chain : OrderedDict
            Current chain/state

        Returns
        -------
        numpy.ndarray
            Grid of shape :code:`(n, n)` with the chain inside
        """
        for coord, poly in chain.items():
            trans_x, trans_y, trans_z = tuple(sum(x) for x in zip(self.midpoint, coord))
            # Recall that a numpy array works by indexing the rows first
            # before the columns, that's why we interchange.
            self.grid[(trans_y, trans_x, trans_z)] = POLY_TO_INT[poly]

        return np.flipud(self.grid)
    
    def get_reward(self, is_trapped, collision, done):
        """Alternate implementation of _compute_free_energy"""
        dictlist = list(self.state.items())
        state = []
        for i in range(len(dictlist)):
            if dictlist[i][1] == 'H':
                state.append(dictlist[i][0])
            else:
                state.append((-1000, 1000, 1000)) #To get rid of P's
        distances = euclidean_distances(state, state)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        
        state_reward = len(bond_idx[0]) if done else 0
        collision_penalty = self.collision_penalty if collision else 0
        actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0
        return state_reward + collision_penalty + actual_trap_penalty
    
    def _compute_reward(self, is_trapped, collision, done):
        """Computes the reward for a given time step

        For every timestep, we compute the reward using the following function:

        .. code-block:: python

            reward_t = state_reward 
                       + collision_penalty
                       + actual_trap_penalty

        The :code:`state_reward` is only computed at the end of the episode
        (Gibbs free energy) and its value is :code:`0` for every timestep
        before that.

        The :code:`collision_penalty` is given when the agent makes an invalid
        move, i.e. going to a space that is already occupied.

        The :code:`actual_trap_penalty` is computed whenever the agent
        completely traps itself and has no more moves available. Overall, we
        still compute for the :code:`state_reward` of the current chain but
        subtract that with the following equation:
        :code:`floor(length_of_sequence * trap_penalty)`
        try:

        Parameters
        ----------
        is_trapped : bool
            Signal indicating if the agent is trapped.
        done : bool
            Done signal
        collision : bool
            Collision signal

        Returns
        -------
        int
            Reward function
        """
        state_reward = self._compute_free_energy(self.state) if done else 0
        collision_penalty = self.collision_penalty if collision else 0
        actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0

        # Compute reward at timestep, the state_reward is originally
        # negative (Gibbs), so we invert its sign.
        reward = - state_reward + collision_penalty + actual_trap_penalty

        return reward

    def _compute_free_energy(self, chain):
        """Computes the Gibbs free energy given the lattice's state

        The free energy is only computed at the end of each episode. This
        follow the same energy function given by Dill et. al.
        [dill1989lattice]_

        Recall that the goal is to find the configuration with the lowest
        energy.

        .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
        mechanics model of the conformational and se quence spaces of proteins.
        Marcromolecules 22(10), 3986–3997 (1989)

        Parameters
        ----------
        chain : OrderedDict
            Current chain in the lattice

        Returns
        -------
        int
            Computed free energy
        """
        h_polymers = [x for x in chain if chain[x] == 'H']
        h_pairs = [(x, y) for x in h_polymers for y in h_polymers]

        # Compute distance between all hydrophobic pairs
        h_adjacent = []
        for pair in h_pairs:
            dist = np.linalg.norm(np.subtract(pair[0], pair[1]))
            if dist == 1.0: # adjacent pairs have a unit distance
                h_adjacent.append(pair)

        # Get the number of consecutive H-pairs in the string,
        # these are not included in computing the energy
        h_consecutive = 0
        for i in range(1, len(self.state)):
            if (self.seq[i] == 'H') and (self.seq[i] == self.seq[i-1]):
                h_consecutive += 1

        # Remove duplicate pairs of pairs and subtract the
        # consecutive pairs
        nb_h_adjacent = len(h_adjacent) / 2
        gibbs_energy = nb_h_adjacent - h_consecutive
        reward = - gibbs_energy
        return int(reward)
    
    def fill_P(self):
        states = [(-1,),]
        for i in range(len(self.seq) - 1):
            states += list(itertools.product(range(5), repeat = i))

        states = sorted(states, key = len)
        self.states_dic = {states[i] : i for i in range(len(states))}
        
        self.P[1][0] = (0, 0, False)
        self.P[1][1] = (0, 0, False)
        self.P[1][2] = (0, 0, False)
        self.P[1][3] = (0, 0, False)
        self.P[1][4] = (0, 0, False)
        
        for state in states:
            if state != (-1,):
                if len(state) < len(self.seq) - 2:
                    for i in range(5):
                        a = 0
                        reward = 0
                        self.step(a)
                        for j in range(len(state)):
                            a = (5 * a + state[j]) % 6
                            _, _, done, info = self.step(a)
                            reward = self._compute_reward(info['is_trapped'], info['collisions'], done)
                        a = (5 * a + i) % 6
                        _, _, done, info = self.step(a)
                        reward = self._compute_reward(info['is_trapped'], info['collisions'], done)
                        self.reset()
                        self.P[self.states_dic[state]][i] = (self.states_dic[state + (i,)], reward, False)
                else:
                    a = 0
                    reward = 0
                    self.step(a)
                    for j in range(len(state)):
                        a = (5 * a + state[j]) % 6
                        _, _, done, info = self.step(a)
                        reward = self._compute_reward(info['is_trapped'], info['collisions'], done)
                    self.reset()
                    for i in range(5):
                        self.P[self.states_dic[state]][i] = (self.states_dic[state], reward, True)
        self.reset()
    
