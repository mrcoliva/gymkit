from os import system
from gymkit.arena import Arena
from gymkit.environments import *
from gymkit.q_agent import QAgent
from gymkit.neuro_net import NeatAgent
from gymkit.neat_q_network import NeatQNetwork


def run(env_class):
    arena = Arena(environment_class=env_class, print_episodes=True)
    # approximator = FeedForwardNeuralNetwork(8, 4)
    agent = NeatAgent(test_episodes=1, verbose=True)
    arena.register(agent)
    episodes = arena.run(render=False, num_episodes=1000)[agent.id]
    # plt.plot(episodes)
    # #plt.plot(agent.q_values)
    # plt.show()
    # system('say Fertig')

    # e = [agent.epsilon(e) for e in range(5000)]
    # plt.plot(e)
    # plt.show()



# -----------------------------------------------------------------------------

if __name__ == '__main__':
    #run(CartPoleEnvironment)
    #run(BipedalWalkerEnvironment)
    run(LunarLanderEnvironment)

