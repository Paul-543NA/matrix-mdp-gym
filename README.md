# Matrix MDP
[![Downloads](https://pepy.tech/badge/matrix-mdp-gym)](https://pepy.tech/project/matrix-mdp-gym)

Easily generate an MDP from transition and reward matricies.

Want to learn more on the story behind this repo? Check the blog post [here](https://www.paul-festor.com/post/i-created-a-python-library)!


## Installation
Assuming you are in the root directory of the project, run the following command:
```bash
pip install matrix-mdp-gym
```

## Usage
```python
import gymnasium as gym
import matrix_mdp
env = gym.make('matrix_mdp/MatrixMDP-v0')
```

## Environment documentation

### Description

A flexible environment to have a gym API for discrete MDPs with `N_s` states and `N_a` actions given:
 - A vector of initial state distribution vector P_0(S)
 - A transition probability matrix P(S' | S, A)
 - A reward matrix R(S', S, A) of the reward for reaching S' after having taken action A in state S

### Action Space

The action is a `ndarray` with shape `(1,)` representing the index of the action to execute.

### Observation Space

The observation is a `ndarray` with shape `(1,)` representing index of the state the agent is in.

### Rewards

The reward function is defined according to the reward matrix given at the creation of the environment.

### Starting State

The starting state is a random state sampled from $P_0$.

### Episode Truncation

The episode truncates when a terminal state is reached.
Terminal states are inferred from the transition probability matrix as
$\sum_{s' \in S} \sum_{s \in S} \sum_{a \in A} P(s' | s, a) = 0$

### Arguments

- `p_à`: `ndarray` of shape `(n_states, )` representing the initial state probability distribution.
- `p`: `ndarray` of shape `(n_states, n_states, n_actions)` representing the transition dynamics $P(S' | S, A)$.
- `r`: `ndarray` of shape `(n_states, n_states, n_actions)` representing the reward matrix.

```python
import gymnasium as gym
import matrix_mdp

gym.make('MatrixMDP-v0', p_0=p_0, p=p, r=r)
```

### Version History

* `v0`: Initial versions release

## Acknowledgements

Thanks to [Will Dudley](https://github.com/WillDudley) for his help on learning how to put a Python package together/