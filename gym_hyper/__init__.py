from gym.envs.registration import register

register(
    id='hypersiamese-v0',
    entry_point='gym_hyper.envs:HyperSiameseEnv',
)

