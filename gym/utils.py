import random

# returns a random action in the given action space
def random_action(action_space):
    return random.randrange(0, action_space)
