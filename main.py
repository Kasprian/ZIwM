import matplotlib.pyplot as plt
import pandas as pd
from mlp import mlp, feature_selection
from lists import columns
from scipy.stats import ttest_rel

if __name__ == '__main__':
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

    results = mlp(X, y)

    clf_1 = max(results[0::6])
    clf_2 = max(results[1::6])
    clf_3 = max(results[2::6])
    clf_4 = max(results[3::6])
    clf_5 = max(results[4::6])
    clf_6 = max(results[5::6])

    # analiza t-studenta

    plt.figure(figsize=(15, 8))
    plt.style.use("ggplot")
    plt.barh(range(len(r)), [s[1] for s in r], align='center')
    plt.yticks(range(len(r)), [s[0] for s in r])
    plt.title('Ranking cech')
    plt.rc('ytick', labelsize=14)
    plt.show
    plt.savefig('plot.png')
