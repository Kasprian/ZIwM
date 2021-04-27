import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from lists import params


def feature_selection(features, classification, n_best):
    select = SelectKBest(score_func=f_classif, k=n_best).fit(features, classification)
    fit_x = select.transform(features)
    return fit_x, select.scores_

def mlp(X, y):
    for i in range(1, 10):
        fit_x, _ = feature_selection(X, y, i)
        for param in params:
            scores = []
            clf = MLPClassifier(**param)
            rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
            for train_index, test_index in rkf.split(fit_x, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                predict = clf.predict(X_test)
                scores.append(accuracy_score(y_test, predict))

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))
