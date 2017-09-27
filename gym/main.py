from gymkit import *


def run(env_class):

    arena = Arena(environment_class=env_class)

    arena.register([DQNAgent()])
    evaluations = arena.run(render=False, num_episodes=250)

    arena.register([NeatAgent(elite_size=1)])
    evaluations = arena.run(render=False, num_episodes=2000)

    # Persist the evaluations to disk.
    for evaluation in evaluations.items():
        PersistenceService.persist(evaluation, scope=evaluation.name)


def analyze(scope: str):
    evals = PersistenceService.load_evaluations(scope)

    if scope == 'DQNAgent':
        Visualizer.plot_dqn_results(evals)

    if scope == 'NEATAgent':
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
