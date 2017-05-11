import gym
from gym import wrappers
import neat
import os

outdir = '/tmp/cartpole-experiment'
environment = 'CartPole-v0'
# env = gym.make('LunarLander-v2')
env = gym.make(environment)
# env = wrappers.Monitor(env, outdir, force=True, mode='training')
observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n
min_reward = 0
max_reward = 200
solved_threshhold = 195
total_episodes = 0


def sum_of(x, y):
    return x + y

class Agent(object):

    def __init__(self):
        self.scores = []

    def log_episode(self, score):
        self.scores.append(score)
        if len(self.scores) % 100 == 0:
            print 'Episodes: {}, Average of last 100: {}'.format(len(self.scores), sum(self.scores[-100:]) / 100)

    def solved(self):
        return len(self.scores) >= 100 and sum(self.scores[-100:]) / 100 >= solved_threshhold

    def network(self, genome, config):
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def fittest_networks(self, n, stats, config):
        return map(lambda genome: self.network(genome, config), stats.best_unique_genomes(n))

    def compute_fitness(self, population, config):
        """
        Computes the fitness of each genome in the population and 
        stores it in their fitness properties.
        
        :param population: A list of genome_id, genome tuples.
        :param config: The neat config.
        """
        genome_network_pairs = map(lambda (_, genome): (genome, self.network(genome, config)), population)

        test_episodes = 1
        for genome, network in genome_network_pairs:
            avg_scores = self.average_scores(env, network, test_episodes)
            genome.fitness = self.fitness(avg_scores)

    def fitness(self, average_score):
        # TODO: implement a less naive function.
        """
        :param average_score: The average of the scores reached in the simulated games.
        :return: The fitness of the genome.
        """
        return average_score / max_reward

    def average_scores(self, env, network, episodes):
        total_score = 0

        for _ in range(episodes):
            observation = env.reset()
            game_over = False
            score = 0

            while not game_over:
                network_output = network.activate(observation)[0]
                action = 0 if network_output < 0 else 1
                observation, reward, done, info = env.step(action)
                score += reward
                total_score += reward

                if done or score >= max_reward:
                    game_over = True
                    self.log_episode(score)

        return total_score / episodes

    def run(self):
        local_directory = os.path.dirname(__file__)
        config_path = os.path.join(local_directory, 'neat-config')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        population = neat.Population(config)
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        # population.add_reporter(neat.StdOutReporter(True))

        while not self.solved():
            game_over = False
            episode_reward = 0

            best_genome = population.run(fitness_function=self.compute_fitness, n=3)

            best_networks = self.fittest_networks(3, stats, config)
            observation = env.reset()

            while not game_over:
                vote_balance = reduce(sum_of, map(lambda network: network.activate(observation)[0], best_networks))
                action = 0 if vote_balance < 0 else 1
                observation, reward, done, info = env.step(action)
                episode_reward += reward

                # env.render()
                if done:
                    game_over = True
                    self.log_episode(episode_reward)


#############################################

Agent().run()

env.close()
#gym.upload(outdir, api_key='sk_7tsnNgAgQJmd87v9WC2hg')