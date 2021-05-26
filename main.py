import matplotlib.pyplot as plt
import pandas as pd
from mlp import mlp, feature_selection
from lists import columns
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
import numpy as np

if __name__ == '__main__':

    # Część 1.

    breast_cancer_data = pd.read_csv('./data/breast-cancer-wisconsin.data', header=None)
    breast_cancer_data.drop(0, axis=1, inplace=True)
    breast_cancer_data.columns = columns

    breast_cancer_data = breast_cancer_data[breast_cancer_data['Bare Nuclei'] != '?']
    breast_cancer_data['Bare Nuclei'] = breast_cancer_data['Bare Nuclei'].astype('int')

    features = breast_cancer_data.drop('Class', axis=1)
    classification = breast_cancer_data['Class']

    fit_x, result = feature_selection(features, classification, 9)
    scores = list(zip(features.columns, result))
    sortedScore = sorted(scores, key=lambda x: x[1], reverse=True)
    print('Ranking:')
    for i, j in enumerate(sortedScore, 1):
        print(f"{i}. {j[0]} {round(j[1], 2)}")
    r = sorted(sortedScore, key=lambda x: x[1])

    X = features.to_numpy()
    y = classification.to_numpy()

    # Część 2.

    results = mlp(X, y)
    means = []

    clf_1 = results[0:90]
    clf_2 = results[90:180]
    clf_3 = results[180:270]
    clf_4 = results[270:360]
    clf_5 = results[360:450]
    clf_6 = results[450:540]

    best_scores = [clf_1, clf_2, clf_3, clf_4, clf_5, clf_6]

    # prezentacja uśrednionych wyników

    for i in range(0, 6):
        mean_score = np.mean(results[i*6:(i+1)*6])
        means.append(mean_score)
        std_score = np.std(results[i*6:(i+1)*6])

    print(means)

    # analiza t-studenta

    alfa = .05
    t_statistic = np.zeros((6, 6))
    p_value = np.zeros((6, 6))

    for i in range(0, 6):
        for j in range(0, 6):
            t_statistic[i, j], p_value[i, j] = ttest_ind(best_scores[i], best_scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    plt.figure(figsize=(15, 8))
    plt.style.use("ggplot")
    plt.barh(range(len(r)), [s[1] for s in r], align='center')
    plt.yticks(range(len(r)), [s[0] for s in r])
    plt.title('Ranking cech')
    plt.rc('ytick', labelsize=14)
    plt.show
    plt.savefig('plot.png')
