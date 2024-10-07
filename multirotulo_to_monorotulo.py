# Função que lê um arquivo ARFF, e transforma o dataset de multirótulo para monorótulo e retorna um arquivo ARFF. É feito da seguinte forma, percorre cada instância do dataset e escolhe a classe mais frequente dos múltiplos caminhos.

import pandas as pd

from read_arff import read_arff
from dataframe_to_arff import dataframe_to_arff

def make_monorotulo(caminho):
    dataset, hierarquia, columns = read_arff(caminho)
    y = dataset['class']
    x = dataset.drop('class', axis=1)

    class_dict = {}
    y_np = y.to_numpy()
    for classe in hierarquia:
        class_dict[classe] = 0
    for elem in y_np:
        for classe in elem.split('@'):
            class_dict[classe] += 1

    for idx, elem in enumerate(y_np):
        if len(elem.split('@')) > 1:
            freq = -1
            most_freq_class = ""
            for classe in elem.split('@'):
                if class_dict[classe] > freq:
                    most_freq_class = classe
                    freq = class_dict[classe]
            y_np[idx] = most_freq_class

    y_new = pd.DataFrame(y_np, columns=['class'])
    dataset_monorotulo = pd.concat([x, y_new], axis=1)

    dataframe_to_arff(dataset_monorotulo, 'dataset_monorotulo', 'Datasets/dataset_monorotulo.arff', hierarquia)

    return dataset_monorotulo