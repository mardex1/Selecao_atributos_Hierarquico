import os
import numpy as np
from read_arff import read_arff
from GMNB import NaiveBayesH, f_measure_hierarquica

caminho_base = "Datasets/BasesMain"
caminho_log = "Logs/GMNB_BASE_LOG.txt"

for dir_data in os.listdir(caminho_base):
    print(f"=================================== DATASET {dir_data} ===================================")
    with open(caminho_log, 'a') as f:
        f.write(f"\n\n========================= DATASET {dir_data} =========================\n\n")

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

        train_data = train_data.to_numpy()
        y_train = train_data[:, -1]

        caminho_part_teste = caminho_dataset + "/" + test_arff
        test_data, test_hier, test_cols = read_arff(caminho_part_teste)
        X_test = test_data.to_numpy()[:, :-1]
        y_test = test_data.to_numpy()[:, -1]

        model_cv = NaiveBayesH(train_hier)
        model_cv.fit(train_data, y_train)
        predictions = model_cv.predict(X_test)

        f_measure = f_measure_hierarquica(predictions, y_test, classes=model_cv.classes)
        f_measures.append(f_measure)
        print(f"CV f_measure={f_measure}")
        with open(caminho_log, 'a') as f:
            f.write(f"F-measure Hierárquica para partição {particao+1}/10 = {round(f_measure, 4)}\n")
    with open(caminho_log, 'a') as f:
        f.write(f"\nF-measure Hierárquica média da Validação Cruzada = {round(np.mean(f_measures), 4)}\n")
        f.write(f"Desvio padrão F-measure Hierárquica para partição: {particao+1}/10 = {np.std(f_measures)}")
    print(np.mean(f_measures))

