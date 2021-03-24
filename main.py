import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv("./data/prepared_data/diagnosis.csv", encoding='UTF-16')

    replace_data = ["Nausea", "LumberPain", "ConUrine", "MictPains", "UrethraBurn", "Inflammation", "Nephritis"]
    predictors = ["Nausea", "LumberPain", "ConUrine", "MictPains", "UrethraBurn"]

    selector = SelectKBest(f_classif, k=3)
    for i in predictors:
        data[i] = data[i].map({'yes': 1, 'no': 0})

    selector.fit_transform(data[predictors], data["Inflammation"])
    scores = -np.log10(selector.pvalues_)
    scores.sort()
    scores = scores[::-1]
    print(selector)
    print(scores)

    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='horizontal')
    plt.show()
