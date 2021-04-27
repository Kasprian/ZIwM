from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from numpy import mean

params = [{'hidden_layer_sizes': (11,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False},
          {'hidden_layer_sizes': (11,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': False},
          {'hidden_layer_sizes': (15,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False},
          {'hidden_layer_sizes': (15,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': False},
          {'hidden_layer_sizes': (20,), 'solver': 'sgd', 'momentum': 0,
           'nesterovs_momentum': False},
          {'hidden_layer_sizes': (20,), 'solver': 'sgd', 'momentum': 0.9,
           'nesterovs_momentum': False}]

# sprawdzić czy dobre parametry


def evaluate_model(clf, X, y):
    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1)
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=rkf,
                             n_jobs=-1)
    return scores


def mlp(X, y):
    results = []
    for param in params:
        clf = MLPClassifier(**param)
        scores = evaluate_model(clf, X, y)
        # tutaj obrobić tabele z wynikami i uśrednić
        results.append(scores)
    return results
