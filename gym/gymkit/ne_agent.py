from typing import Optional
import neat, os, time, numpy as np
from neat.nn import FeedForwardNetwork
from gymkit.agent import Agent
from gymkit.environment import Environment
from gym.spaces import Box
from baselines import logger
import itertools


class NeatAgent(Agent):
    def __init__(self, id='NeatAgent', elite_size=3, test_episodes=10, verbose=False):
        super(NeatAgent, self).__init__(id)
        self.env = None
        self.config = None
        self.verbose = verbose
        self.stats = neat.StatisticsReporter()
        self.generation = 0
        self.population = None
        self.elite_size = elite_size
        self.test_episodes = test_episodes
        self.scores = []
        self.elite_scores = []
        self.winner_candidate = None
        # the number of generations a winner candidate must remain the fittest individual before being evaluated
        self.winner_candidate_eval_age = 1
        self.winner_candidate_age = 0
        self.winner = None
        self.dismissed_winner_candidates = []
        self.winner_candidate_min_fitness = 200


    def setup(self, environment: Environment):
        self.results_file = '/tmp/{0}-experiment-{1}'.format(environment.name, time.time())
        # self.env = gym.wrappers.Monitor(environment.env, self.results_file)
        self.env = environment.env
        self.config = self.read_config(environment)
        self.population = neat.Population(self.config)
        self.population.add_reporter(self.stats)

        if self.verbose:
            self.population.add_reporter(neat.StdOutReporter(self.config))


    @staticmethod
    def read_config(environment: Environment):
        config_path = os.path.join(os.path.dirname(__file__), 'neat_config/neat-config-{}'.format(environment.name))
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, config_path)


    def log_episode(self, score: int):
        self.scores.append(score)


    def fittest_networks(self, n: int) -> [neat.nn.FeedForwardNetwork]:
        """
        Returns the fittest n networks of the current population.
        :rtype: list
        :param n: The number of fittest networks to create.
        :return: An array of n neural network phenotypes.
        """
        return [self.phenotype(genome) for genome in self.stats.best_unique_genomes(n)]


    def phenotype(self, genome: neat.DefaultGenome) -> neat.nn.FeedForwardNetwork:
        """
        Creates and returns the neural network phenotype of the given genome. 
        :param genome: The genome encoding the network.
        """
        return FeedForwardNetwork.create(genome, self.config)


    def compute_fitness(self, population: [neat.DefaultGenome], config: neat.Config) -> None:
        """
        Computes the fitness of each genome in the population and 
        stores it in their fitness properties.
        
        :param population: A list of genome_id, genome tuples.
        :param config: The neat config.
        """
        for genome, network in [(genome, self.phenotype(genome)) for _, genome in population]:
            genome.fitness = self.average_score(self.env, network, self.test_episodes)


    def next_winner_candidate(self) -> Optional[neat.DefaultGenome]:
        for n in itertools.count(1):
            candidate = self.stats.best_unique_genomes(n)[-1]
            if candidate not in self.dismissed_winner_candidates:
                print('Using the {0}. fittest genome as winner candidate with fitness {1}.'.format(n, candidate.fitness))
                return candidate
        return None


    def update_winner_candidate(self) -> None:
        genome = self.next_winner_candidate()
        if genome.fitness < -50:
            return

        if self.winner_candidate is not None and self.winner_candidate.key == genome.key:
            self.winner_candidate_age += 1

            if self.winner_candidate_age > self.winner_candidate_eval_age:
                self.evaluate_winner_candidate()
        else:
            self.winner_candidate = genome


    def evaluate_winner_candidate(self) -> None:
        print("\n--\nEvaluating winner candidate with key {}.".format(self.winner_candidate.key))
        candidate_score = self.average_score(self.env, self.phenotype(self.winner_candidate), num_episodes=10)
        if candidate_score >= -80:
            self.winner = self.winner_candidate
        else:
            self.dismiss_winner_candidate()
            print("Dismissed winner candidate with average score of {}.\n--".format(candidate_score))


    def dismiss_winner_candidate(self):
        self.dismissed_winner_candidates.append(self.winner_candidate)
        self.winner_candidate = None
        self.winner_candidate_age = 0


    def average_score(self, env, network: neat.nn.FeedForwardNetwork, num_episodes: int) -> float:
        """
        Runs the network in an environment and measures its success. 
        :param env: The environment to test in.
        :param network: The network to evaluate.
        :param num_episodes: The number of episodes to evaluate and average about.
        :return: The average score reached in the test episodes.
        """
        total_score = 0
        t = 0

        for _ in range(num_episodes):
            state = env.reset()

            while True:
                t += 1
                observation, reward, done, _ = env.step(self.action(state, [network], t))
                state = observation
                total_score += reward

                if done:
                    break

        return total_score / num_episodes


    def evolve(self, generations: int=1):
        """
        Runs the genetic algorithm to evolve the population for the specified number of generations.
        """
        _ = self.population.run(fitness_function=self.compute_fitness, n=generations)
        self.generation += generations
        # print('Evolved to generation {}.'.format(self.generation))
        # self.update_winner_candidate()


    def action(self, state: np.ndarray, networks: [neat.nn.FeedForwardNetwork], t: int):
        # if random.random() < 0.2:  # self.epsilon(t):
        #     return self.env.action_space.sample()

        votes = [network.activate(state.flatten()) for network in networks]
        if isinstance(self.env.action_space, Box):
            return list(map(self.aggregate_output, list(zip(*votes))))
        else:
            return np.argmax(list(map(self.aggregate_output, list(zip(*votes)))))


    @staticmethod
    def aggregate_output(output: np.ndarray) -> np.ndarray:
        """
        The function used to aggregate the output of multiple phenotypes into a single decision. 
        """
        return np.mean(output)


    def log_state(self, steps: int, e: float) -> None:
        # logger.record_tabular("steps", steps)
        logger.record_tabular("episodes", len(self.scores))
        logger.record_tabular("mean episode reward", round(np.mean(self.scores[-101:-1]), 1))
        # logger.record_tabular("% time spent exploring", float(e))
        logger.dump_tabular()


    def run_episode(self, render: bool=False) -> None:
        pass


    def solved(self) -> bool:
        """
        Returns a boolean indicating whether the environment is considered as solved by the agent.
        """
        return len(self.scores) >= 100 and np.mean(self.scores[-101:-1]) >= 195


    def actors(self) -> [FeedForwardNetwork]:
        if self.winner is None:
            # self.evolve(generations=1)
            return self.fittest_networks(self.elite_size)
        else:
            return [self.phenotype(self.winner)]


    def evaluate(self, max_episodes: int):
        self.scores = []
        env = self.env
        t = 0

        while not self.solved() and len(self.scores) < max_episodes:
            state = env.reset()
            episode_reward = 0
            self.evolve(generations=1)

            while True:
                action = self.action(state, self.actors(), t)
                observation, reward, done, _ = env.step(action)
                state = observation
                episode_reward += reward
                t += 1
                env.render()

                if done:
                    self.scores.append(episode_reward)
                    self.log_state(t, self.epsilon(t))
                    break

            if self.solved():
                print('''* Environment solved! *''')
                self.env.close()
                #gym.upload(self.results_file, api_key='sk_7tsnNgAgQJmd87v9WC2hg')
                return self.scores

        return self.scores

