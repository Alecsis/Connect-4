from gym.envs.registration import register

register(
    id='connect4-v0',
    entry_point='gym-connect4.envs:Connect4Env',
)