from gymkit import *

# Welcome to GymKit!
#
# Explore the framework by looking into the 'gymkit' folder.
#
# To run an an agent in an environment, simply execute this file.
# Modify the '__main__' method at the bottom of this file to play run or analyze an evaluation.
# Modify the 'run' method below to play around with different agent, configurations and environments.
# To see what environments are available, take a look into '/gymkit/environments.py'.


def run(env_class):

    arena = Arena(environment_class=env_class)

    arena.register([DQNAgent()])
    evaluations = arena.run(render=True, num_episodes=250)

    arena.register([NeatAgent(elite_size=1)])
    evaluations = arena.run(render=True, num_episodes=2000)

    # Persist the evaluations to disk.
    for evaluation in evaluations.items():
        PersistenceService.persist(evaluation, scope=evaluation.name)


def analyze(scope: str):
    evals = PersistenceService.load_evaluations(scope)

    if scope == 'dqn_1':
        Visualizer.plot_dqn_results(evals)

    if scope == 'neat_1':
        Visualizer.plot_neat_results(evals)


# -----------------------------------------------------------------------------


if __name__ == '__main__':

    run(MountainCarEnvironment)

    # Uncomment one of these lines to plot the respective performance development.
    # analyze('neat_1')
    # analyze('dqn_1')

    # Uncomment the lines below to plot the average episode durations of both algoritms.
    # Visualizer.plot_runtimes(
    #     PersistenceService.load_evaluations('neat_1'),
    #     PersistenceService.load_evaluations('dqn_1')
    # )
