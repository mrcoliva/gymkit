from gymkit.agent import Agent
import random
import numpy as np
from gymkit.q_function_approximator import DeepQNetwork
from gymkit.q_models import Experience, Memory


class QAgent(Agent):

    def __init__(self, function_approximator, id='QAgent'):
        # type: (DeepQNetwork, basestring) -> QAgent
        super(QAgent, self).__init__(id)
        self.id = id
        self.env = None
        self.approximator = function_approximator
        self.approximator.q_agent = self
        self.t = 0
        self.memory = Memory(buffer_size=2000)
        self.q_values = []


    def setup(self, environment):
        self.env = environment


    @property
    def epsilon(t, min=0.01, max=0.1, max_t=10000):
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


    def action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        q = self.approximator.activate(state)
        self.q_values.append(np.amax(q))
        return np.argmax(q)


    def run_episode(self, render=False):
        self.approximator.prepare_for_episode()
        state = self.env.reset()
        game_over = False
        score = 0
        self.t = 0

        while not game_over:
            action = self.action(state)
            new_state, reward, done, _ = self.env.perform(action)
            score += reward
            self.t += 1
            self.memory.store(Experience(state, action, reward, new_state, done))
            state = new_state
            game_over = done

        self.approximator.did_finish_episode(self.memory)
        return score
















    # def algorithm_step(self, state):
    #     if random.random() < self.epsilon(self.t):  # explore
    #         return random.choice(self.env.action_space)
    #
    #     q_values = self.action_q_values(state)
    #     action = max(enumerate(q_values), key=itemgetter(1))[1][0]
    #     observation, reward, done, _ = self.env.perform(action)
    #     q_next_values = self.action_q_values(observation)
    #     index, (q, next_action) = max(enumerate(q_values), key=itemgetter(1))
    #     target = reward + self.discount(1) + next_action
    #
    #     self.memory.store(Experience(state, action, reward, observation))