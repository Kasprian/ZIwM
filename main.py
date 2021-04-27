import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from .mlp import mlp

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

if __name__ == '__main__':
    breast_cancer_data = pd.read_csv('./data/breast-cancer-wisconsin.data', header=None)
    breast_cancer_data.drop(0, axis=1, inplace=True)
    breast_cancer_data.columns = columns

    breast_cancer_data = breast_cancer_data[breast_cancer_data['Bare Nuclei'] != '?']
    breast_cancer_data['Bare Nuclei'] = breast_cancer_data['Bare Nuclei'].astype('int')

    classification = breast_cancer_data['Class']
    features = breast_cancer_data.drop('Class', axis=1)

    select = SelectKBest(score_func=f_classif, k=9).fit(features, classification)
    scores = list(zip(features.columns, select.scores_))
    sortedScore = sorted(scores, key=lambda x: x[1], reverse=True)
    print('Ranking:')
    for i, j in enumerate(sortedScore, 1):
        print(f"{i}. {j[0]} {round(j[1], 2)}")
    r = sorted(sortedScore, key=lambda x: x[1])

    results = mlp(classification, features)
    # dorobić dodawanie cech, t-student

    plt.figure(figsize=(15, 8))
    plt.style.use("ggplot")
    plt.barh(range(len(r)), [s[1] for s in r], align='center')
    plt.yticks(range(len(r)), [s[0] for s in r])
    plt.title('Ranking cech')
    plt.rc('ytick', labelsize=14)
    plt.show
    plt.savefig('plot.png')

