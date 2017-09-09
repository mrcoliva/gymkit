import itertools
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from baselines import deepq
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from gymkit.agent import Agent
from gymkit.globals import NUM_CORES


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class QAgent(Agent):

    def __init__(self, id: str='QAgent'):
        super(QAgent, self).__init__(id)
        self.id = id
        self.env = None
        self.episode_count = 0
        self.scores = []


    def setup(self, environment):
        self.env = environment


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
                    if len(self.scores) >= num_episodes:
                        return self.scores
                    state = env.reset()
                    self.scores.append(0)

                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            return self.scores
