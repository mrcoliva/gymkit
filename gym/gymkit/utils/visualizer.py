from typing import Dict
from gymkit.utils.utils import savitzky_golay
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Visualizer(object):

    @staticmethod
    def visualize(evaluations, colors) -> None:
        for i, (agent, scores) in enumerate(list(evaluations.items())):
            raw = np.array(scores)
            x = np.arange(0, len(scores))
            smooth = savitzky_golay(raw, 101, 3)
            plt.plot(x, raw, linestyle='-', linewidth=1, alpha=.3, color=colors[i])
            plt.plot(x, smooth, label=agent, linewidth=3, color=colors[i])

        plt.ylabel('Episode Scores')
        plt.legend()
        plt.show()





