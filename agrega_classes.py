# Função que lê um arquivo ARFF, realiza a agregação de suas classes para que todas tenham pelo menos 10 instâncias, então retorna o arquivo ARFF. É feito da seguinte forma, percorre cada nível da hierarquia, se encontrar uma classe com menos de 10 instãncias, então a classe se torna a classe pai. Caso a classe não tenha classe pai, então as instâncias são eliminadas do dataset. 

# PARA O FUNCIONAMENTO CORRETO DA FUNÇÃO, É PRECISO QUE O DATASET SEJA MONORÓTULO, ENTÃO CONSIDERE UTILIZAR A FUNÇÃO multirotulo_to_monorotulo.

import numpy as np
import pandas as pd
from collections import Counter

from read_arff import read_arff
from dataframe_to_arff import dataframe_to_arff

def gera_hierarquia_completa(hierarquia):
    hierarquia_completa = []
    for classe in hierarquia:
        for _ in range(len(classe.split('/'))):
            if classe not in hierarquia_completa:
                hierarquia_completa.append(classe)
                classe = classe.split('/')[:-1]
                classe = '/'.join(classe)
    return hierarquia_completa

def agrega_classes(caminho, nome_dataset):
    dataset, hierarquia, columns = read_arff(caminho)
    
    hierarquia_completa = gera_hierarquia_completa(hierarquia)

    # Descobre a profundidade da hierarquia
    y = dataset['class']
    profundidade = 0
    caminho = ""
    dict_count = {}

    for classe in hierarquia_completa:
        dict_count[classe] = 0
        caminho = classe.split('/')
        if len(caminho) > profundidade:
            profundidade = len(caminho)
            caminho = caminho
    # Contagem de cada classe.
            
    for classe in y:
        dict_count[classe] += 1

    nivel_i = profundidade
    while nivel_i >= 1:
        print(f"==================================== NIVEL {nivel_i} ====================================")
        for classe, count in dict_count.items():
            if dict_count[classe] < 10 and len(classe.split('/')) == nivel_i:
                new_classe = classe.split('/')[:-1]
                if len(new_classe) == 0:
                    drop_idxs = []
                    for idx, c in enumerate(dataset['class']):
                        if c == classe:
                            drop_idxs.append(idx)
                    print("dropping ", classe)
                    dataset = dataset.drop(index=drop_idxs)
                    dataset = dataset.reset_index(drop=True)
                else:
                    new_classe = '/'.join(new_classe)
                    print(f"Classe {classe.upper()} reduziu para {new_classe.upper()}")
                    print(f"{classe.upper()}: {dict_count[classe]}, -> {new_classe.upper()}: {dict_count[new_classe]}")
                    dataset.loc[dataset['class'] == classe, 'class'] = new_classe
        nivel_i -= 1
        for classe, count in dict_count.items():
            dict_count[classe] = 0
        for classe in dataset['class']:
            dict_count[classe] += 1
    print(len(dataset))
    class_final_counts = Counter(dataset['class'].tolist())
    print(class_final_counts)
    dataframe_to_arff(dataset, 'dataset_agregado', f'Datasets/processados/{nome_dataset}_agregado.arff', hierarquia)
    return dataset, dict_count

agrega_classes('Datasets/processados/cellcycle_sem_valores_ausentes.arff', 'eisen')