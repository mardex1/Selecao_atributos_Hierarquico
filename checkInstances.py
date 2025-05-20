import os
from read_arff import read_arff
import numpy as np

caminho_base = "Datasets/BasesMain"
for dir_data in os.listdir(caminho_base):
    print(f"=================================== DATASET {dir_data} ===================================")
    caminho_dataset = caminho_base + "/" + dir_data
    f_measures = []
    for particao in range(10):
        print(f"Iteração {particao+1}/10")
        for idx, item in enumerate(os.listdir(caminho_dataset)):
            if f"TRA{particao}" in item:
                train_arff = item
            if f"TES{particao}" in item:
                test_arff = item

        caminho_part_treino = caminho_dataset + "/" + train_arff
        train_data, train_hier, train_cols = read_arff(caminho_part_treino)
        n_cols_original = len(train_data.columns) - 1

        train_data = train_data.to_numpy()[:, :-1]
        y_train = train_data[:, -1]

        caminho_part_teste = caminho_dataset + "/" + test_arff
        test_data, test_hier, test_cols = read_arff(caminho_part_teste)
        X_test = test_data.to_numpy()[:, :-1]
        y_test = test_data.to_numpy()[:, -1]

        print(f"Instances: {train_data.shape[0] + X_test.shape[0]}")
        print(f"Columns: {train_data.shape[1]}")
        break
