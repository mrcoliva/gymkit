from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

color_cycle = ['k', 'r', 'g', 'b', 'c']


class Visualizer(object):

    @staticmethod
    def plot_species(species_sizes):
        """ 
        Visualizes speciation throughout evolution. 
        """
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        fig, ax = plt.subplots()
        ax.stackplot(range(num_generations), *curves)

        plt.title("Speciation")
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        plt.show()
        plt.close()


    @staticmethod
    def plot_runtimes(neat, dqn):
        def extract(key, evals):
            return list(map(lambda e: e[key], evals))

        neat_mean_runtime = np.mean(extract('runtime', neat))
        dqn_mean_runtime = np.mean(extract('runtime', dqn))

        dqn_trial_episodes = 2000
        neat_trial_episodes = 250 * 150 + 250

        dqn_mean_episode_duration = dqn_mean_runtime / dqn_trial_episodes
        neat_mean_episode_duration = neat_mean_runtime / neat_trial_episodes

        plt.bar([0, 2], [dqn_mean_episode_duration * 1000, neat_mean_episode_duration * 1000], color='#424242')
        plt.xticks([0, 1], ('DQN', 'NEAT'))

        plt.ylabel('Average Episode Duration (in milliseconds)')
        plt.show()
        plt.close()


    @staticmethod
    def plot_dqn_results(evals):
        def extract(key):
            return list(map(lambda e: e[key], evals))

        reward_threshold = np.full(2000, -110)
        scores = extract('scores')
        mean_scores = np.mean(scores, axis=0)
        mean_q_values = np.mean(extract('q_values'), axis=0)
        runtimes = extract('runtime')

        trials_avgs = []
        for trial in scores:
            trials_avgs.append([np.mean(trial[max(0, i - 99):i + 1]) for i in range(len(trial))])

        solved = np.asarray([np.argmax(np.asarray(t) > -111) for t in trials_avgs])
        for i in range(len(solved)): solved[i] = 2000 if solved[i] == 0 else solved[i]

        plt.plot(np.mean(runtimes, axis=0))

        plt.plot(reward_threshold, label='Reward Threshold', linestyle='--', alpha=.3, color='#101010')

        plt.plot(mean_scores, label='Actual Scores', alpha=1.0, color='#FF5F5E')
        plt.plot(mean_q_values, label='Estimated Action-Values (Q-values)', alpha=1.0, color='#626262')

        plt.ylabel('Score')
        plt.xlabel('Episodes')
        plt.legend()

        plt.show()
        plt.close()


    @staticmethod
    def plot_neat_results(evals):
        def extract(key):
            return list(map(lambda e: e[key], evals))

        scores = extract('scores')
        best_fitness = extract('best_fitness')
        avg_fitness = np.asarray(extract('avg_fitness'))
        stdev_fitness = np.asarray(extract('stdev_fitness'))

        reward_threshold = np.full(250, -110)
        mean_scores = np.mean(scores, axis=0)
        mean_best_fitness = np.mean(best_fitness, axis=0)
        mean_avg_fitness = np.mean(avg_fitness, axis=0)
        mean_stdev_fitness = np.mean(stdev_fitness, axis=0)

        plt.ylim(-200, 0)
        plt.plot(reward_threshold, label='Reward Threshold', linestyle='--', alpha=.3, color='#101010')
        plt.plot(mean_scores, label='Agent Scores', color='#FF5F5E')
        plt.plot(mean_avg_fitness, 'b-', label="Population Average", color='#626262')
        plt.plot(mean_avg_fitness + mean_stdev_fitness, 'g-.', label="Population Average +1 sd", color='#626262', alpha=.2)
        plt.plot(mean_best_fitness, 'r-', label="Best Genome", color='#6EC038')

        plt.xlabel("Generations")
        plt.ylabel("Score / Fitness")
        plt.legend()

        plt.show()
        plt.close()
