from environment import Environment
from gym import wrappers


class Config(object):

    def __init__(self, mode='training', monitor=False, name='CartPole-v0'):
        self.env_name = name
        self.mode = mode
        self.monitor = monitor

    def outdir(self):
        return '/tmp/{}'.format(self.env_name)


    def environment(self):
        env = Environment(self)

        if self.monitor:
            return wrappers.Monitor(env, self.outdir(), force=True, mode=self.mode)
        return env


class CartPoleConfig(Config):

    def __init__(self, mode='training', monitor=False):
        super(CartPoleConfig, self).__init__(mode, monitor)
        self.env_name = 'CartPole-v0'


class MountainCarConfig(Config):

    def __init__(self, mode='training', monitor=False):
        super(MountainCarConfig, self).__init__(mode, monitor)
        self.env_name = 'MountainCar-v0'
