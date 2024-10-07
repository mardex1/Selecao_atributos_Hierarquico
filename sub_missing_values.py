# Função que lê um arquivo ARFF, realiza a substituição de valores ausentes do dataset e retorna um arquivo ARFF. É feito da seguinte forma, quando se encontra um valor ausente, procura pela média de todos os elementos não ausentes de mesma classe, e associa essa média ao valor ausente. Caso não haja elementos não ausentes, então, procuramos por elementos não ausentes nos descendentes da classe da instância e utilizamos a média desses valores, caso não haja descendentes da classe ou não haja valores não ausentes, utilizamos a média global do atributo.

import numpy as np
import pandas as pd

from read_arff import read_arff
from dataframe_to_arff import dataframe_to_arff

def get_global_mean(col, x_np):
    sum = 0
    n = 0
    for idx, valor in enumerate(x_np.T[col]):
        if valor != '?':
            sum += float(valor)
            n += 1
    return sum / n

def gera_descendentes(classe, hierarquia): # Gera os descendentes de uma classe
    descendentes = []
    for c in hierarquia:
        if classe in c and c != classe:
            descendentes.append(c)
    return descendentes

def get_mean_descendentes(classe, col, x_np, y_np, hierarquia):
    classes_descendentes = gera_descendentes(classe, hierarquia)
    if len(classes_descendentes) == 0: # Caso o qual a classe não tem descendentes
        return get_global_mean(col, x_np)
    sum = 0
    n = 0
    # Calcula a média dos valores conhecidos do atributo para todos os descendentes
    for idx, valor in enumerate(x_np.T[col]):
        if valor != '?' and y_np[idx] in classes_descendentes:
            sum += float(valor)
            n += 1
    if n == 0: # Nenhuma instância possui valor conhecido para o atributo para as classes descendentes
        return get_global_mean(col, x_np)
    return sum / n


def get_mean(classe, col, x_np, y_np, hierarquia):
    sum = 0
    n = 0
    for idx, valor in enumerate(x_np.T[col]):
        if valor != '?' and y_np[idx] == classe:
            sum += float(valor)
            n += 1
    if n == 0:
        return get_mean_descendentes(classe, col, x_np, y_np, hierarquia) # calcula a média para todos os descendentes de "classe"
    return sum/n


def sub_missing_values(caminho):
    dataset, hierarquia, colunas = read_arff(caminho)

    y = dataset['class']
    x = dataset.drop('class', axis=1)
    x_np = x.to_numpy()
    y_np = y.to_numpy()

    for col, atributo in enumerate(x_np.T): # Percorre cada atributo
        for row, valor in enumerate(atributo): # Percorre os valores de cada atributo
            class_missing_value = ""
            if valor == '?':
                class_missing_value = y_np[row]
                mean = get_mean(class_missing_value, col, x_np, y_np, hierarquia)
                x_np[row][col] = mean

    x_new = pd.DataFrame(x_np, columns=colunas[:-1])
    x_new['class'] = y
    data = x_new.copy()

    # Feito para controlar os valores númericos, pois certos valores estavam sofrendo de perda de precisão, para isso, arredondamos todos os valores 4 casas decimais.
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='ignore')
    data = data.round(4)

    dataframe_to_arff(data, 'dataset_sem_valores_ausentes', 'Datasets/dataset_sem_valores_ausentes.arff', hierarquia)

    return data

