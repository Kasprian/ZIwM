import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from lists import params


def mlp(X, y):
    for param in params:
        scores = []
        clf = MLPClassifier(**param)
        rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
        for train_index, test_index in rkf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            print(predict)
            scores.append(accuracy_score(y_test, predict))

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))