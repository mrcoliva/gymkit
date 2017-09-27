import os, time, json
from gymkit.evaluation import Evaluation


path = os.path.abspath('../gym/gymkit/eval/')


class PersistenceService(object):

    @staticmethod
    def persist(evaluation: Evaluation, scope: str='') -> str:
        """
        Writes the contents of the given evaluations dict to a .txt file and stores
        the newly created file to disk. 
        
        :param scope: The scope of the evaluation. Files will be persisted in a folder like as the scope.
        :param evaluation: The evaluation object that should be persisted on a file. 
        :return: The name of the newly created file.
        """
        filename = '{0}/{1}/eval_{2}.txt'.format(path, scope, time.time())

        with open(filename, 'w') as f:
            f.write(json.dumps(evaluation.info))

        print('[PersistenceService] Stored evaluation {0} in {1}'.format(evaluation.name, filename))
        return filename


    @staticmethod
    def load_evaluations(scope: str):
        scope_path = '{0}/{1}/'.format(path, scope)
        eval_files = list(filter(lambda f: f[0] == 'e', os.listdir(scope_path)))
        evals = []

        for filename in eval_files:
            with open('{0}/{1}'.format(scope_path, filename), 'r') as f:
                evals.append(json.loads(f.read()))

        return evals
