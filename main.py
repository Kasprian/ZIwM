import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from tabulate import tabulate
from sklearn.exceptions import ConvergenceWarning

columns = [
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
]

params = [{'hidden_layer_sizes': (10,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False, 'max_iter': 500, 'random_state': 5},
          {'hidden_layer_sizes': (10,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': True, 'max_iter': 500, 'random_state': 5},
          {'hidden_layer_sizes': (15,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False, 'max_iter': 500, 'random_state': 5},
          {'hidden_layer_sizes': (15,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': True, 'max_iter': 500, 'random_state': 5},
          {'hidden_layer_sizes': (20,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False, 'max_iter': 500, 'random_state': 5},
          {'hidden_layer_sizes': (20), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': True, 'max_iter': 500, 'random_state': 5}]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def feature_selection(features, classification, n_best):
    select = SelectKBest(score_func=f_classif, k=n_best).fit(features, classification)
    fit_X = select.transform(features)
    return fit_X, select.scores_


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

    # Esperyment
    results = np.zeros((6, 9))
    for i in range(1, 10):
        j = 0
        for param in params:  # po kolei klasyfikatory
            clf = MLPClassifier(**param)
            rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)
            scores = []
            for train_index, test_index in rkf.split(X, y):
                fit_X, _ = feature_selection(X, y, i)
                X_train, X_test = fit_X[train_index], fit_X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                predict = clf.predict(X_test)
                scores.append(accuracy_score(y_test, predict))  # wynik każdego ksperymentu w ramach jednego foldu
            mean = np.mean(scores)  # średnia z ostatnich foldów dla danego klasyfikatora z daną liczna cech
            results[j, i-1] = mean
            print(results)
            j = j + 1

    np.save('results', results)
    # analiza t-studenta

    alfa = .05
    t_statistic = np.zeros((6, 6))
    p_value = np.zeros((6, 6))
    s = np.array(results)

    #for i in range(0, 6):
    #    for j in range(0, 6):
    #        t_statistic[i, j], p_value[i, j] = ttest_rel(s[i], s[j])
    #print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    # plt.figure(figsize=(15, 8))
    # plt.style.use("ggplot")
    # plt.barh(range(len(r)), [s[1] for s in r], align='center')
    # plt.yticks(range(len(r)), [s[0] for s in r])
    # plt.title('Ranking cech')
    # plt.rc('ytick', labelsize=14)
    # plt.show
    # plt.savefig('plot.png')
