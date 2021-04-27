import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from lists import params
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def feature_selection(features, classification, n_best):
    select = SelectKBest(score_func=f_classif, k=n_best).fit(features, classification)
    fit_X = select.transform(features)
    return fit_X, select.scores_

def mlp(X, y):
    results = []
    for i in range(1, 10):
        j = 0
        print(f"K -> {i}")
        fit_X, _ = feature_selection(X, y, i)
        for param in params:
            j = j + 1
            print(f"Clf -> {j}")
            scores = []
            clf = MLPClassifier(**param)
            rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
            for train_index, test_index in rkf.split(fit_X, y):
                X_train, X_test = fit_X[train_index], fit_X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                predict = clf.predict(X_test)
                scores.append(accuracy_score(y_test, predict))

            mean_score = np.mean(scores)
            # std_score = np.std(scores)
            results.append(mean_score)
    return results
