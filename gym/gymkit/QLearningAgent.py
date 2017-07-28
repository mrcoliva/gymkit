from gymkit.agent import Agent
import tensorflow as tf


class QLearningAgent(Agent):

    def __init__(self, id='QLearningAgent', hidden_layers=1, quality_score_threshold=25, verbose=False):
        super(QLearningAgent, self).__init__(id)
        self.env = None
        self.model = None
        self.episode_count = 0


    def setup(self, environment):
        self.env = environment
        self.model = self.build_model()


    def build_model(self):
        tf.reset_default_graph()

        inputs = tf.placeholder(shape=[1, 16], dtype=tf.float32)


    def run_episode(self, render=False):
        self.episode_count += 1
        episode_reward = 0
        dataset = []
        observation = self.env.reset()

        while True:
            prediction = self.activate(observation)
            dataset.append((observation, prediction))
            action = self.action(prediction)
            observation, reward, done, _ = self.env.perform(action)
            episode_reward += reward

            if render:
                self.env.render()

            if done:
                #if episode_reward >= self.quality_score_threshold:
                self.train(dataset)
                return episode_reward