__credits__ = ["Paul Festor"]

from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class MatrixMDPEnv(gym.Env):
    """
    ## Description

    A flexible environment to have a gym API for discrete MDPs with `N_s` states and `N_a` actions given:
     - A vector of initial state distribution vector P_0(S)
     - A transition probability matrix P(S' | S, A)
     - A reward matrix R(S', S, A) of the reward for reaching S' after having taken action A in state S

    ## Action Space

    The action is an `int` state index.

    ## Observation Space

    The observation is an `int` state index.

    ## Rewards

    The reward function is defined according to the reward matrix given at the creation of the environment.

    ## Starting State

    The starting state is a random state sampled from $P_0$.

    ## Episode Truncation

    The episode truncates when a terminal state is reached.
    Terminal states are inferred from the transition probability matrix as
    $\sum_{s' \in S} \sum_{s \in S} \sum_{a \in A} P(s' | s, a) = 0$

    ## Arguments

    - `p_à`: `ndarray` of shape `(n_states, )` representing the initial state probability distribution.
    - `p`: `ndarray` of shape `(n_states, n_states, n_actions)` representing the transition dynamics $P(S' | S, A)$.
    - `r`: `ndarray` of shape `(n_states, n_states, n_actions)` representing the reward matrix.

    ```python
    import gymnasium as gym
    import matrix_mdp

    gym.make('MatrixMDP-v0', p_0=p_0, p=p, r=r)
    ```

    ## Version History

    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", ],
    }

    def __init__(self, p_0, p, r, render_mode: Optional[str] = None):
        self.p_0 = p_0
        self.p = p
        self.r = r

        self.action_space = spaces.Discrete(n=p.shape[2])
        self.observation_space = spaces.Discrete(n=p.shape[0])
        self.states_array = np.arange(p.shape[0])

        # Check that initial probability matrix sums to 1
        if np.around(p_0.sum(), decimals=6) != 1:
            raise ValueError("The provided initial probabilities do not sum tp 1.")

        # Check that state transition probabilities sum to O (terminal state) or 1 for every (s, a) pair
        for s in self.states_array:
            for a in np.arange(self.p.shape[2]):
                if p[:, s, a].sum() != 0 and np.around(p[:, s, a].sum(), decimals=6) != 1:
                    raise ValueError(
                        "The provided transition probabilities are invalid: \sum_s' P(s' | s, a) not in {0, 1} " +\
                        f"for s={s} and a={a}."
                    )

        self.render_mode = render_mode

        # Terminal states are states where no actions lead anywhere
        self.terminal_states = [s for s in self.states_array if self.p[:, s, :].sum() == 0]

        # Initialize the first state
        self.state = np.random.choice(self.states_array, p=self.p_0)

    def step(self, action: np.ndarray):
        """
        Takes a step in the environment.
        Note: Of the agent executes an invalid action (for which $\\sum_{s'} P(s'|s, a) = 0$), the environment will
        remain in the current state and both the reward and done flags will be None.

        Args:
            action (int): Index of the taken action

        Returns:
            obs (int): Next state
            reward (float): The reward for taking the action in the current state and reaching the next state
            done (bool): True if the environment reached a terminal state, otherwise false
            done (bool): For compatibility with other gymnasium environments
            info (dict): Always empty, for compatibility with other gymnasium environments
        """
        if self.p[:, self.state, action].sum() == 0:
            if self.render_mode == "human":
                print(f"/!\\ Warning: the agent took action {action} invalid in state {self.state}" +\
                      f"as p[:, s={self.state}, a={action}].sum() == 0 /!\\")
            new_state = None
            reward = None
            done = None

        else:
            new_state = np.random.choice(self.states_array, p=self.p[:, self.state, action])
            reward = self.r[new_state, self.state, action]
            done = (new_state in self.terminal_states)

        self.state = new_state
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, done, done, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment, draws the state from the initial state distribution matrix.

        Args:
            seed (int?): Seed for random number generator
            options (dict): To override the random state initialisation, set the start state by giving:
                options={"start_state": start_state} (e.g. options={"start_state": 0} sets the start state to S0)

        Returns:
            obs (int): New initial state
            info (dict): Always empty, for compatibility with other gymnasium environments
        """
        super().reset(seed=seed)

        start_state = np.random.choice(self.states_array, p=self.p_0)
        if options is not None:
            start_state = options.get("start_state") if "start_state" in options else start_state

        self.state = start_state

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Returns the current state of the MDP.

        Returns:
            obs (int): Current state of the environment
        """
        return self.state

    def render(self):
        """
        Prints environment information in the console.

        Returns:

        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "human":
            print(f"Current state: {self.state}")

    def close(self):
        pass
