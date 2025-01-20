import pandas as pd
import numpy as np
from read_arff import read_arff
from sklearn.model_selection import train_test_split, StratifiedKFold
from GMNB import NaiveBayesH, f_measure_hierarquica
import os

caminho_registro = "Logs/log_bases_full.txt"
caminho_bases = "Datasets/BasesProcessadas"

for data_file in os.listdir(caminho_bases):
    name_dataset = data_file.split('.')[0]
    data, hier, cols = read_arff(caminho_bases + "/" + data_file)

    with open(caminho_registro, 'a') as f:
        f.write(f"\n============================ {name_dataset.upper()} ============================\n")

    data['mw'], _ = pd.factorize(data['mw'], sort=True)
    data['sl'], _ = pd.factorize(data['sl'], sort=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    X = data.drop('class', axis=1)
    y = data['class']

    f_measures = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xval, yval = X.iloc[val_idx], y.iloc[val_idx]

        Xtr, ytr, Xval, yval = Xtr.to_numpy(), ytr.to_numpy(), Xval.to_numpy(), yval.to_numpy()

        print(f"Split número {i+1} / 10")

        nbh_fold = NaiveBayesH(hier)
        nbh_fold.fit(Xtr, ytr)
        pred_fold = nbh_fold.predict(Xval)

        f_measure = f_measure_hierarquica(pred_fold, yval, nbh_fold.classes)
        f_measures.append(f_measure)
        with open(caminho_registro, 'a') as f:
            f.write(f"\nF-measure Hierárquica fold {i+1} = {f_measure}\n")

    print(np.mean(f_measures))
    with open(caminho_registro, 'a') as f:
        f.write(f"\nF-measure Hierárquica média da Validação Cruzada = {round(np.mean(f_measures), 4)}\n")
