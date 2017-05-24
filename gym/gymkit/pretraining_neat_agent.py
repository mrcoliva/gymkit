from neuro_net import NeatAgent


class PretrainingNeatAgent(NeatAgent):

    def __init__(self, id='PretrainingNeatAgent', elite_size=3, verbose=False):
        super(PretrainingNeatAgent, self).__init__(id, elite_size, verbose)
        self.is_training = False


    def should_evolve(self):
        return self.is_training


    def train(self, episodes, fitness_treshhold, max_generations=None):
        self.is_training = True
        self.test_episodes = episodes
        #print '[PretrainingNeatAgend] Starting training...'

        generations = 0
        fitnesses = []
        #while self.elite_fitness() < fitness_treshhold and (max_generations is None or generations <= max_generations):
        for _ in range(max_generations):
            self.evolve()
            fitnesses.append(self.elite_fitness())
            print self.elite()[0]
            generations += 1
            #print 'Generation {}, Elite Fitness: {}'.format(generations, self.elite_fitness())

        #print('[PretrainingNeatAgend] Finished training after {} generations.'.format(generations))
        self.is_training = False
        return fitnesses

