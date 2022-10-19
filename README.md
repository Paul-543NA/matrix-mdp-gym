# Matrix MDP
Easily generate an MDP from transition and reward matricies.

## Installation
Assuming you are in the root directory of the project, run the following command:
```bash
pip install .
```

## Usage
```python
import gymnasium as gym
import matrix_mdp
env = gym.make('matrix_mdp/MatrixMDP-v0')
```