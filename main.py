import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
from matplotlib import ticker
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from sklearn import clone
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
clfs = {
    '50layers_momentum': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, nesterovs_momentum=True, solver='sgd',
                                       random_state=1234, momentum=0.9),
    '500layers_momentum': MLPClassifier(hidden_layer_sizes=(500,), max_iter=500, nesterovs_momentum=True, solver='sgd',
                                        random_state=1234, momentum=0.9),
    '1000layers_momentum': MLPClassifier(hidden_layer_sizes=(1000,), max_iter=500, nesterovs_momentum=True,
                                         solver='sgd', random_state=1234, momentum=0.9),
    '50layers_no_momentum': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, nesterovs_momentum=False,
                                          solver='sgd', random_state=1234, momentum=0),
    '500layers_no_momentum': MLPClassifier(hidden_layer_sizes=(500,), max_iter=500, nesterovs_momentum=False,
                                           solver='sgd', random_state=1234, momentum=0),
    '1000layers_no_momentum': MLPClassifier(hidden_layer_sizes=(1000,), max_iter=500, nesterovs_momentum=False,
                                            solver='sgd', random_state=1234, momentum=0),
}

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
    max_features = 9
    for i in range(1, max_features + 1):
        print(str(i) + " features")
        kfold = RepeatedStratifiedKFold(
            n_splits=2, n_repeats=5, random_state=1)
        scores = np.zeros((len(clfs), 2 * 5))
        for fold_id, (train, test) in enumerate(kfold.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                fit_x, _ = feature_selection(X, y, i)
                clf = clone(clfs[clf_name])
                clf.fit(fit_x[train], y[train])
                prediction = clf.predict(fit_x[test])
                scores[clf_id, fold_id] = accuracy_score(y[test], prediction)
        print(scores)
        np.save('results/results_of ' + str(i) + " features", scores)

