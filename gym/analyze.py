import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json

path = os.getcwd() + '/gymkit/eval/dqn_1'
eval_files = list(filter(lambda f: f[0] == 'e',  os.listdir(path)))
evals = []

for filename in eval_files:
    with open('{0}/{1}'.format(path, filename), 'r') as f:
        evals.append(json.loads(f.read()))

def extract(key):
    return list(map(lambda e: e[key], evals))

reward_threshold = np.full(2000, -110)
scores = extract('scores')
q_values = extract('q_values')

mean_scores = np.mean(scores, axis=0)
mean_q_values = np.mean(q_values, axis=0)

trials_avgs = []
for trial in scores:
    trials_avgs.append([np.mean(trial[max(0, i-99):i+1]) for i in range(len(trial))])

mean_trials_avgs = np.mean(trials_avgs, axis=0)

solved = np.asarray([np.argmax(np.asarray(t) > -111) for t in trials_avgs])
for i in range(len(solved)): solved[i] = 2000 if solved[i] == 0 else solved [i]

# print(np.argmax(trials_avgs, axis=1))
# print(np.max(trials_avgs, axis=1))

plt.plot(reward_threshold)
plt.plot(mean_scores)
plt.plot(mean_trials_avgs)
plt.plot(mean_q_values)
plt.axvline(np.mean(solved))

print(solved)
print(np.mean(solved), np.std(solved))

#[plt.plot(s) for s in trials_avgs]

plt.show()
