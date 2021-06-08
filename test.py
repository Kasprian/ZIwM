import numpy as np
from scipy.stats import ttest_rel

if __name__ == '__main__':
    scores = np.load('results.npy')
    # analiza t-studenta
    print(scores)

    alfa = .05
    t_statistic = np.zeros((6, 6))
    p_value = np.zeros((6, 6))
    s = np.array(results)

    for i in range(0, 6):
        for j in range(0, 6):
            t_statistic[i, j], p_value[i, j] = ttest_rel(s[i], s[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)