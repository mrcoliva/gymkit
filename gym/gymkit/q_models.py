import random


def clamp(value, lower_bound, upper_bound):
    return max(lower_bound, min(value, upper_bound))


class Experience(object):

    def __init__(self, state, action, reward, result_state=None, terminal=False, clamp_reward=True):
        self.state = state
        self.action = action
        self.reward = clamp(reward, -1, 1)
        self.result_state = result_state
        self.terminal = terminal

    def readable_format(self):
        return "===\nState: {0}\nAction: {1}\nReward: {2}\nResult State: {3}"\
            .format(self.state, self.action, self.reward, self.result_state)


class Memory(object):

    def __init__(self, buffer_size=5000):
        self.experiences = []
        self.buffer_size = buffer_size


    def store(self, experience):
        if len(self.experiences) >= self.buffer_size:
            self.experiences.pop(0)
        self.experiences.append(experience)


    def sample(self, size):
        if len(self.experiences) < size:
            return self.experiences
        return random.sample(self.experiences, size)


    def sequence(self, length):
        """
        Returns a sequence of consecutively stored samples with the given length.
        :param length: The length of the sequence to retrieve. 
        :return: An array containing elements of a sequence of consecutively stored experiences.
        """
        if len(self.experiences) < length:
            return self.experiences
        return self.experiences[-length:]
