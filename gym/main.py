from gymkit.neuro_net import NeatAgent
from gymkit.neural_network_agent import NeuralNetworkAgent
from gymkit.arena import Arena
from gymkit.environment import *


def run(env_class):
    arena = Arena(environment_class=env_class)
    #stadium.register(NeatAgent(elite_size=1, verbose=True))
    arena.register(NeuralNetworkAgent())
    arena.run(render=True)


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run(CartpoleEnvironment)
    #run(BipedalWalkerEnvironment)
    #run(LunarLanderEnvironment)

