from gymkit.neuro_net import NeatAgent
from gymkit.arena import Arena
from gymkit.environment import BipedalWalkerEnvironment

if __name__ == '__main__':
    stadium = Arena(environment_class=BipedalWalkerEnvironment)
    agent = NeatAgent(elite_size=1, verbose=True)
    stadium.register(agent)

    evaluations = stadium.run(render=True)
