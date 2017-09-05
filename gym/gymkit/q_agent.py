import itertools

import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from baselines import deepq, logger
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from gymkit.agent import Agent
from gymkit.globals import NUM_CORES
from gymkit.q_function_approximator import DeepQNetwork


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class QAgent(Agent):

    def __init__(self, id='QAgent'):
        # type: (DeepQNetwork, str) -> QAgent
        super(QAgent, self).__init__(id)
        self.id = id
        self.env = None
        self.episode_count = 0
        self.scores = []


    def setup(self, environment):
        self.env = environment


    def epsilon(t, min=0.01, max=0.1, max_t=10):
        """
        Returns an epsilon for the epsilon-greedy policy at time step t, linearly annealed 
        between from max to min over max_t time steps, and fixed at min afterwards.
        
        :param min: The minimum epsilon, used as the fixed value after max_t time steps.
        :param max: The maximum epsilon, used as the initial value.
        :param max_t: The number of time steps over which epsilon is linearly annealed from max to min.
        :return: The scalar value of epsilon.
        """
        if t >= max_t:
            return min
        return max - (t * (max - min) / max_t)


    def solved(self):
        """
        Returns a boolean indicating whether the environment is considered as solved by the agent.
        """
        return len(self.scores) >= 100 and np.mean(self.scores[-101:-1]) >= 200


    def log_state(self, steps, e):
        logger.record_tabular("steps", steps)
        logger.record_tabular("episodes", len(self.scores))
        logger.record_tabular("mean episode reward", round(np.mean(self.scores[-101:-1]), 1))
        logger.record_tabular("% time spent exploring", int(100 * e))
        logger.dump_tabular()


    def run_episode(self, render=False):
        pass


    def evaluate(self, num_episodes, render=False):
        with U.make_session(NUM_CORES):
            env = self.env.env
            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                    make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                    q_func=model,
                    num_actions=env.action_space.n,
                    optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            )
            # Create the replay buffer
            replay_buffer = ReplayBuffer(50000)
            # Create the schedule for exploration starting from 1 (every action is random) down to
            # 0.02 (98% of actions are selected according to values predicted by the model).
            exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            self.episode_count += 1
            state = env.reset()
            self.scores = [0.0]

            for t in itertools.count():
                action = act(state[None], update_eps=exploration.value(t))[0]
                observation, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, observation, float(done))

                state = observation
                self.scores[-1] += reward

                if render:
                    env.render()

                if done:
                    state = env.reset()
                    self.scores.append(0)

                is_solved = t > 100 and np.mean(self.scores[-101:-1]) >= 200
                if is_solved:
                    # Show off the result
                    env.render()
                else:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if t > 1000:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # Update target network periodically.
                    if t % 1000 == 0:
                        update_target()

                if done and len(self.scores) % 10 == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(self.scores))
                    logger.record_tabular("mean episode reward", round(np.mean(self.scores[-101:-1]), 1))
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

        # =========================================================================================================
        #
        # with U.make_session(NUM_CORES):
        #     env = self.env.env
        #     act, train, update_target, debug = deepq.build_train(
        #             make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
        #             q_func=model,
        #             num_actions=env.action_space.n,
        #             optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        #     )
        #
        #     U.initialize()
        #     update_target()
        #     replay_buffer = ReplayBuffer(50000)
        #
        #     # Create the schedule for exploration starting from 1 (every action is random) down to
        #     # 0.02 (98% of actions are selected according to values predicted by the model).
        #     exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)
        #
        #     self.episode_count += 1
        #     state = env.reset()
        #     self.scores.append(0)
        #
        #     for t in itertools.count():
        #         action = act(state[None], update_exp=exploration.value(self.episode_count))[0]
        #         observation, reward, done, _ = env.perform(action)
        #         replay_buffer.add(state, action, reward, observation, float(done))
        #
        #         state = observation
        #         self.scores[-1] += reward
        #
        #         if done:
        #             if len(self.scores) % 10 == 0:
        #                 self.log_state(t, exploration.value(t))
        #             break
        #
        #         if t > 1000:
        #             observations, actions, rewards, result_observations, terminals = replay_buffer.sample(32)
        #             train(observations, actions, rewards, result_observations, terminals, np.ones_like(rewards))
        #         if t % 1000:
        #             update_target()
        #
        #     if self.solved():
        #         print("\n=====\nSOLVED AFTER {} EPISODES!\n=====\n".format(len(self.scores)))
        #         env.render()
