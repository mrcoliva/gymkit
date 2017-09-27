from gymkit.environment import Environment
import numpy as np


class BipedalWalkerEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(BipedalWalkerEnvironment, self).__init__('BipedalWalker-v2', mode, monitoring_enabled)


class CartPoleEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(CartPoleEnvironment, self).__init__('CartPole-v0', mode, monitoring_enabled)


class LunarLanderEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(LunarLanderEnvironment, self).__init__('LunarLander-v2', mode, monitoring_enabled)


class PendulumEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(PendulumEnvironment, self).__init__('Pendulum-v0', mode, monitoring_enabled)


class MountainCarEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(MountainCarEnvironment, self).__init__('MountainCar-v0', mode, monitoring_enabled)

    def solved(self, scores):
        if len(scores) < 100:
            return False
        return np.mean(scores[-99:]) >= self.reward_threshold


class Go9x9Environment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(Go9x9Environment, self).__init__('Go9x9-v0', mode, monitoring_enabled)


class InvertedDoublePendulumEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(InvertedDoublePendulumEnvironment, self).__init__('InvertedDoublePendulum-v1', mode, monitoring_enabled)