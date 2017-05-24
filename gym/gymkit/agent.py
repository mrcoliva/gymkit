class Agent(object):

    def __init__(self, id='UnnamedAgent'):
        self.id = id


    def setup(self, environment):
        raise NotImplementedError


    def run_episode(self, render=False):
        raise NotImplementedError
