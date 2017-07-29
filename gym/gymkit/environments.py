from gymkit.environment import Environment


class BipedalWalkerEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(BipedalWalkerEnvironment, self).__init__('BipedalWalker-v2', mode, monitoring_enabled)


class CartPoleEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(CartPoleEnvironment, self).__init__('CartPole-v0', mode, monitoring_enabled)


class LunarLanderEnvironment(Environment):

    def __init__(self, mode='training', monitoring_enabled=False):
        super(LunarLanderEnvironment, self).__init__('LunarLander-v2', mode, monitoring_enabled)

