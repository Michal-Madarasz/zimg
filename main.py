import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate

from names_dict import COLUMNS, FEATURES

alpha = .05


def rebuild_table(tmp_data):
    predictors = COLUMNS[1:]
    for i in predictors:
        tmp_data[i] = tmp_data[i].map({'yes': 1, 'no': 0})


if __name__ == '__main__':
    data = pd.read_csv('./data/prepared_data/diagnosis.csv', encoding='UTF-16')

    num_of_neighbours = [1, 5, 10]
    type_of_metric = ['euclidean', 'manhattan']

    rebuild_table(data)

    print('Data after rebuild_table()')
    print(data)

    # --------------------------------------------------------------------

    predictors = dict(FEATURES)
    selector = SelectKBest(f_classif, k=3)
    selector.fit_transform(data[predictors], data["Inflammation"])
    scores2 = -np.log10(selector.pvalues_)

    scores2.sort()
    print('Ranking cech:')
    print(scores2)

    plt.figure(figsize=(12,6))
    plt.axes([0.45, 0.35, 0.5, 0.5])
    plt.barh(range(len(predictors)), scores2)
    plt.yticks(range(len(predictors)), predictors.values())
    plt.show()

    # --------------------------------------------------------------------

    # rozbicie na tabelę X (cechy 0/1) oraz y wynik 0/1
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)

    final_scores = []

    for features_index in range(0,6):
        selected_features_count = features_index + 1
        selected_data = selector.fit_transform(X, y)
        for metric in type_of_metric:
            for neighbor in num_of_neighbours:
                clf = KNeighborsClassifier(n_neighbors=neighbor, metric=metric)
                scores = []
                # split zwraca numery indeksów, próbek wybranych i podzielonych na
                # podzbiory uczące oraz podzbiory testowe
                for train_index, test_index in rkf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # uczymy klasyfikator
                    clf.fit(X_train, y_train)
                    # testowanie klasyfikatora
                    predict = clf.predict(X_test)
                    # accuracy_score - wyliczanie metryki dokładności
                    scores.append(accuracy_score(y_test, predict))

                print('--------------------------------------')
                print('Wyniki:')
                print(scores)
                mean_score = np.mean(scores)
                final_scores.append(scores)
                std_score = np.std(scores)
                print(f"Liczba cech: {selected_features_count}")
                print(f"Miara odległości: {metric}")
                print(f"Liczba najbliższych sąsiadów: {neighbor}")
                print(f"Średnia jakość {mean_score:.3f}")
                print(f"Odchylenie standardowe jakości {std_score:.3f}")

    # np.save('results', final_scores)

    stats_array_len = len(final_scores)

    t_statistic = np.zeros((stats_array_len, stats_array_len))
    p_value = np.zeros((stats_array_len, stats_array_len))

    print(f"\nZbalansowanie danych: {np.unique(y, return_counts=True)}\n")


    print('Final SCORES:\n')
    print(final_scores)

    for i in range(stats_array_len):
        for j in range(stats_array_len):
            t_statistic[i, j], p_value[i, j] = ttest_ind(final_scores[i], final_scores[j])

    # liczba wierszy odpowiada liczbie testowanych modeli
    # liczba kolumn = liczba wyników uzyskanych w procesie walidacji

    headers = ['1n_eu','5n_eu','10n_eu','1n_man','5n_man','10n_man']
    names_column = np.array([['1n_eu'],['5n_eu'],['10n_eu'],['1n_man'],['5n_man'],['10n_man']])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")

    print("t_statistic:\n", t_statistic_table)
    print("p_value:\n", p_value_table)

    advantage = np.zeros((stats_array_len, stats_array_len))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers, floatfmt=".2f")
    print("Advantage:\n", advantage_table)
