import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from tabulate import tabulate

if __name__ == '__main__':
    score1 = np.load('results/results_of 1 features.npy')
    score2 = np.load('results/results_of 2 features.npy')
    score3 = np.load('results/results_of 3 features.npy')
    score4 = np.load('results/results_of 4 features.npy')
    score5 = np.load('results/results_of 5 features.npy')
    score6 = np.load('results/results_of 6 features.npy')
    score7 = np.load('results/results_of 7 features.npy')
    score8 = np.load('results/results_of 8 features.npy')
    score9 = np.load('results/results_of 9 features.npy')
    scores = np.array([score1, score2, score3, score4, score5, score6, score7, score8, score9])
    # analiza t-studenta

    best_results = np.zeros((6, 10))
    best_feature = [[""], [""], [""], [""], [""], [""]]
    for i in range(6):
        mean = 0
        for feature in range(9):
            if mean < np.mean(scores[feature, i]):
                mean = np.mean(scores[feature, i])
                best_results[i] = scores[feature, i]
                best_feature[i] = str(feature + 1)

    print(best_feature)

    headers = ["Mom50 F 4", "Mom500 F 5", "Mom1000 F 5", "NoMom50 F 4", "NoMom500 F 3", "NoMom1000 F 2"]
    names_column = np.array(
        [["Mom50 F 4"], ["Mom500 F 5 "], ["Mom1000 F 5"], ["NoMomentum50 F 4"], ["NoMom500 F 3"], ["NoMom1000 F 2"]])

    print(headers)
    print(names_column)
    alfa = .05
    t_statistic = np.zeros((6, 6))
    p_value = np.zeros((6, 6))

    for i in range(0, 6):
        for j in range(0, 6):
            t_statistic[i, j], p_value[i, j] = ttest_ind(best_results[i], best_results[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((6, 6))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((6, 6))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)
