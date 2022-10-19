import numpy as np
import gymnasium as gym
import matrix_mdp

n_states = 4
n_actions = 2
p_0 = np.ones((n_states, )) / n_states
r = np.ones((n_states, n_actions))
p = np.ones((n_states, n_states, n_actions)) / n_states

env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0, p=p, r=r)
env.reset( )
print(env.step(0))