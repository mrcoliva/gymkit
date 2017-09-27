import itertools
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

from baselines import deepq
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from gymkit.globals import NUM_CORES
from gymkit.evaluation import Evaluation
from gymkit.agent import Agent


def model(input, num_actions, scope, reuse=False):
    """
    This model takes as input an observation and returns values of all actions.
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class DQNAgent(Agent):

    def __init__(self, id: str='DQNAgent'):
        super(DQNAgent, self).__init__(id)
        self.id = id
        self.env = None
        self.episode_count = 0
        self.scores = []
        self.q_values = []
        self.t0 = None
        self.eps_hist = []
        self.evaluation = Evaluation(self.id, info={
            'scores': [],
            'q_values': []
        })


    def setup(self, environment):
        self.env = environment


    def final_evaluation(self):
        self.evaluation.info['scores'] = self.scores
        self.evaluation.info['runtime'] = time.time() - self.t0
        return self.evaluation


    def evaluate(self, num_episodes, render=False):
        with U.make_session(NUM_CORES):
            self.t0 = time.time()
            env = self.env.env

            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                    make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                    q_func=model,
                    num_actions=env.action_space.n,
                    optimizer=tf.train.AdamOptimizer(learning_rate=5e-4)
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
            episode_q = []

            for t in itertools.count():
                action = act(state[None], update_eps=exploration.value(t))[0]
                observation, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, observation, float(done))

                state = observation
                self.scores[-1] += reward

                episode_q.append(float(debug['q_values'](state[None]).max()))

                if render:
                    env.render()

                if done:
                    print('{0}, score: {1} ({2})'.format(len(self.scores), self.scores[-1], np.mean(self.scores[-100:])))
                    self.evaluation.info['q_values'].append(np.mean(episode_q))

                    if len(self.scores) >= num_episodes:
                        return self.final_evaluation()

                    state = env.reset()
                    episode_q = []
                    self.scores.append(0)

                    if self.env.solved(self.scores):
                        self.evaluation.info['solved'] = len(self.scores)

                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            U.reset()
            return self.final_evaluation()
