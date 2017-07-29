from gymkit.arena import Arena
from gymkit.environments import *
from gymkit.q_agent import QAgent
from os import system
from gymkit.neat_q_network import NeatQNetwork


def run(env_class):
    arena = Arena(environment_class=env_class, print_episodes=True)
    # approximator = FeedForwardNeuralNetwork(8, 4)
    approximator = NeatQNetwork(None, test_episodes=5, gen_evolve=1)
    agent = QAgent(approximator)
    arena.register(agent)
    approximator.setup(arena.environment(agent))
    episodes = arena.run(render=False, max_episodes=2)[agent.id]
    # plt.plot(episodes)
    # #plt.plot(agent.q_values)
    # plt.show()
    system('say Fertig')



# -----------------------------------------------------------------------------

if __name__ == '__main__':
    #run(CartPoleEnvironment)
    #run(BipedalWalkerEnvironment)
    run(LunarLanderEnvironment)

