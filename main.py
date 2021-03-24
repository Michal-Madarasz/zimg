import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

from names_dict import FEATURES

if __name__ == '__main__':
    data = pd.read_csv("./data/prepared_data/diagnosis.csv", encoding='UTF-16')

    predictors = dict(FEATURES[1:])

    selector = SelectKBest(f_classif, k=3)
    for i in predictors:
        data[i] = data[i].map({'yes': 1, 'no': 0})

    selector.fit_transform(data[predictors], data["Inflammation"])
    scores = -np.log10(selector.pvalues_)


    scores.sort()

    plt.figure(figsize=(12,6))
    plt.axes([0.45, 0.35, 0.5, 0.5])
    plt.barh(range(len(predictors)), scores)
    plt.yticks(range(len(predictors)), predictors.values())
    plt.show()
