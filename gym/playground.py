import gym
import numpy
import random
import utils
from keras.models import Sequential
from keras.layers import Dense, Activation

env = gym.make('CartPole-v0')
should_render = True

action_space = 2
observation_dimensionality = 4
training_episode_count = 1000
game_episode_count = 10
target_step_count = 1000
quality_score_threshold = 50

# Runs an initial set of games with random decisions to create learning data.
# Returns an array of (observation, action) tuples to be used as learning data.
def run_initial_environment():
    # array of (observation, action) tuples
    training_data = []

    # all scores
    scores = []

    # the scores higher than the quality threshhold
    quality_scores = []

    # actions taken in the current environment
    action_history = []

    # observations made in the current environment
    environment_observations = []

    for episode_index in range(training_episode_count):
        score = 0
        episode_action_history = []
        previous_observation = []

        for tick in range(target_step_count):
            action = utils.random_action(action_space)
            observation, reward, done, info = env.step(action)

            score += reward

            # store the taken action as a result of the previous observation
            if len(previous_observation) > 0:
                episode_action_history.append([previous_observation, action])
            previous_observation = observation

            if done:
                env.reset()
                break

        if score >= quality_score_threshold:
            quality_scores.append(score)
            [training_data.append(sample) for sample in episode_action_history]

        env.reset()
        scores.append(score)

    print 'Median score: ' + str(reduce(lambda x, y: x + y, quality_scores) / len(quality_scores))
    return training_data


# Builds and returns a simple neural network with no hidden layers and the given number of input neurons.
def build_neural_network_model(input_size):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_size))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Trains the neural network model with the given training data.
def train_model(model, training_data):
    input_vector = numpy.array(map(lambda sample: sample[0], training_data)) # observation
    output_vector = numpy.array(map(lambda sample: sample[1], training_data)) # action
    model.fit(input_vector, output_vector, epochs=10)
    return model


# Runs an initial set of games with random decisions to create learning data.
# Returns an array of (observation, action) tuples to be used as learning data.
def run_with_neural_network(model):
    # all scores
    scores = []

    # the scores higher than the quality threshhold
    quality_scores = []

    # actions taken in the current environment
    action_history = []

    # observations made in the current environment
    environment_observations = []

    for episode_index in range(game_episode_count):
        score = 0
        episode_action_history = []
        previous_observation = []

        for tick in range(target_step_count):
            # env.render()

            if len(previous_observation) == 0:
                action = utils.random_action(action_space)
            else:
                decision = model.predict(numpy.array(previous_observation).reshape((1, 4)))
                action = 1 if decision >= 0.5 else 0

            observation, reward, done, info = env.step(action)

            score += reward

            # store the taken action as a result of the previous observation
            if len(previous_observation) > 0:
                episode_action_history.append([previous_observation, action])
            previous_observation = observation

            if score >= quality_score_threshold: quality_scores.append(score)

            if done:
                env.reset()
                break

    print 'Median score: ' + str(reduce(lambda x, y: x + y, quality_scores) / len(quality_scores))


env.reset()

training_data = run_initial_environment()

model = build_neural_network_model(input_size=observation_dimensionality)
model = train_model(model, training_data)

env.reset()
run_with_neural_network(model)










#
