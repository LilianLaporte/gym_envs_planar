from gym.envs.registration import register
register(
    id='nLink-reacher-acc-v0',
    entry_point='nLinkReacher.envs:NLinkAccReacherEnv'
)
register(
    id='nLink-reacher-vel-v0',
    entry_point='nLinkReacher.envs:NLinkVelReacherEnv'
)
register(
    id='nLink-reacher-tor-v0',
    entry_point='nLinkReacher.envs:NLinkTorReacherEnv'
)
