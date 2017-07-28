from gymkit.neuro_net import NeatAgent
from gymkit.neural_network_agent import NeuralNetworkAgent
from gymkit.arena import Arena
from gymkit.environments import *
from gymkit.q_agent import QAgent
from gymkit.ffnn import FeedForwardNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import math
from os import system
from gymkit.neat_q_network import NeatQNetwork


def run(env_class):
    arena = Arena(environment_class=env_class)
    # approximator = FeedForwardNeuralNetwork(8, 4)
    approximator = NeatQNetwork(None)
    agent = QAgent(approximator)
    arena.register(agent)
    approximator.setup(arena.environment(agent))
    episodes = arena.run(render=False, max_episodes=200)[agent.id]
    plt.plot(episodes)
    #plt.plot(agent.q_values)
    plt.show()



# -----------------------------------------------------------------------------

if __name__ == '__main__':
    #run(CartPoleEnvironment)
    #run(BipedalWalkerEnvironment)
    run(LunarLanderEnvironment)

