from gymkit.neuro_net import NeatAgent
from gymkit.arena import Arena
from gymkit.config import Config

if __name__ == '__main__':
    config = Config(name='Pendulum-v0')
    stadium = Arena(config)
    agent = NeatAgent(elite_size=3)
    stadium.register(agent)

    evaluations = stadium.run()
