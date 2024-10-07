import numpy as np
import pandas as pd

from read_arff import read_arff
from dataframe_to_arff import dataframe_to_arff

def discretizacao_nao_supervisionada(caminho):
    dataset, hierarquia, columns = read_arff(caminho)
    for column in dataset.columns[:-1]:
        dataset[column].astype(float)
    print(dataset.dtypes)
    print(dataset.info())
    data_discretizado = pd.DataFrame()
    for column in dataset.columns[:-1]:
        data_discretizado[column] = pd.qcut(dataset[column], q=20, labels=range(1,21))
    data_discretizado['class'] = dataset['class']
    dataframe_to_arff(data_discretizado, 'dataset_discretizado', 'Datasets/dataset_discretizado.arff', hierarquia)

    return data_discretizado