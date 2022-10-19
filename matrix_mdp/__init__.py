from gymnasium.envs.registration import register

register(
    id='matrix_mdp/MatrixMDP-v0',
    entry_point='matrix_mdp.envs:MatrixMDPEnv',
    max_episode_steps=300,
)