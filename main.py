import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate

from names_dict import COLUMNS

alfa = .05

def rebuild_table(tmp_data):
    predictors = COLUMNS[1:]
    for i in predictors:
        tmp_data[i] = tmp_data[i].map({'yes': 1, 'no': 0})


if __name__ == '__main__':
    data = pd.read_csv('./data/prepared_data/diagnosis.csv', encoding='UTF-16')
    num_of_neighbours = [1, 5, 10]
    type_of_metric = ['euclidean', 'manhattan']

    rebuild_table(data)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)

    final_scores = []

    for metric in type_of_metric:
        for neighbor in num_of_neighbours:
            clf = KNeighborsClassifier(n_neighbors=neighbor, metric=metric)
            scores = []
            for train_index, test_index in rkf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                predict = clf.predict(X_test)
                scores.append(accuracy_score(y_test, predict))

            mean_score = np.mean(scores)
            final_scores.append(scores)
            std_score = np.std(scores)
            print(f"Miara odległości: {metric}")
            print(f"Liczba najbliższych sąsiadów: {neighbor}")
            print(f"Średnia jakość {mean_score:.3f}")
            print(f"Odchylenie standardowe jakości {std_score:.3f}")

    # np.save('results', final_scores)

    stats_array_len = len(final_scores)

    t_statistic = np.zeros((stats_array_len, stats_array_len))
    p_value = np.zeros((stats_array_len, stats_array_len))

    print(t_statistic)
    print(p_value)
    print(final_scores)

    for i in range(stats_array_len):
        for j in range(stats_array_len):
            t_statistic[i, j], p_value[i, j] = ttest_rel(final_scores[i], final_scores[j], nan_policy='omit')

    print("t_statistic:\n", t_statistic)
    print("p_value:\n", p_value)

    headers = ['1n_eu','5n_eu','10n_eu','1n_man','5n_man','10n_man']
    names_column = np.array([['1n_eu'],['5n_eu'],['10n_eu'],['1n_man'],['5n_man'],['10n_man']])

    advantage = np.zeros((stats_array_len, stats_array_len))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)
