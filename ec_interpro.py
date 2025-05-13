import os
import numpy as np
from read_arff import read_arff
from GMNB import NaiveBayesH, f_measure_hierarquica
from feature_selection import best_first
"Datasets/BasesMain/EC-Interpro"

caminho_base = "Datasets/BasesMain/EC-Interpro"
caminho_log = "Logs/log_adapted_EC_Interpro.txt"
f_measures = []
nome_dataset = caminho_base.split('/')[-1]
with open(caminho_log, 'a') as f:
    f.write(f"\n\n========================= DATASET {nome_dataset} =========================\n\n")
for particao in range(10):
    print(f"Iteração {particao+1}/10")
    with open(caminho_log, 'a') as f:
        f.write(f"Iteração {particao+1}/10")
    for idx, item in enumerate(os.listdir(caminho_base)):
        if f"TRA{particao}" in item:
            train_arff = item
        if f"TES{particao}" in item:   
            test_arff = item

    caminho_part_treino = caminho_base + "/" + train_arff
    train_data, train_hier, train_cols = read_arff(caminho_part_treino)
    n_cols_original = len(train_data.columns) - 1

    _, melhor_subconjunto, i_r, porcent_melhor_subconjunto = best_first(train_data, 50)

    cols = []
    for i, idx in enumerate(melhor_subconjunto):
        if idx == 1:
            cols.append(i)
    with open(caminho_log, 'a') as f:
        f.write(f'Colunas resultantes da seleção de atributos: {cols}\n')
        f.write(f'Número colunas original = {n_cols_original}\n')
        f.write(f'Número de colunas = {len(cols)}\n')
        f.write(f'Porcentagem padrões únicos: {porcent_melhor_subconjunto}\n')
    print(f'Colunas resultantes da seleção de atributos: {cols}\nNúmero colunas original = {n_cols_original}\nNúmero de colunas = {len(cols)}\nPorcentagem padrões únicos: {porcent_melhor_subconjunto}\n\n')
    
    train_data = train_data.to_numpy()
    X_train_selected = train_data[:, cols]
    y_train = train_data[:, -1]

    caminho_part_teste = caminho_base + "/" + test_arff
    test_data, test_hier, test_cols = read_arff(caminho_part_teste)
    X_test = test_data.to_numpy()[:, cols]
    y_test = test_data.to_numpy()[:, -1]

    model_cv = NaiveBayesH(train_hier)
    model_cv.fit(X_train_selected, y_train)
    predictions = model_cv.predict(X_test)

    f_measure = f_measure_hierarquica(predictions, y_test, classes=model_cv.classes)
    f_measures.append(f_measure)
    print(f"CV f_measure={f_measure}")
    with open(caminho_log, 'a') as f:
        f.write(f"F-measure Hierárquica para partição {particao+1}/10 = {round(f_measure, 4)}\n")
with open(caminho_log, 'a') as f:
    f.write(f"\nF-measure Hierárquica média da Validação Cruzada = {round(np.mean(f_measures), 4)}\n")
    f.write(f"Desvio padrão F-measure Hierárquica = {np.std(f_measures)}")
print(np.mean(f_measures))

