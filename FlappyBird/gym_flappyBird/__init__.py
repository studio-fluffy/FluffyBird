from gym.envs.registration import register
register(
    #id='flpbird-v0',
    id = 'scienceCampBird-v1',
    entry_point='gym_flappyBird.envs.rev1:birdEnv',
)
