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


    # Supondo que `dataset` e `class_counts` já estejam definidos
    y = dataset['class']
    class_counts = Counter(y)
    profundidade_max = max([len(classe.split('/')) for classe in y])
    y_processed = y.copy()
    instancias_to_remove = set()

    nivel_atual = profundidade_max
    while nivel_atual >= 1:
        alteracoes = Counter()  # Dicionário para acumular alterações temporárias nesta iteração
        
        # Itera sobre as classes no nível atual
        for label in list(class_counts.keys()):
            classe_atual = label
            if class_counts[classe_atual] < 10 and nivel_atual == len(classe_atual.split('/')):
                # Determina a classe pai
                classe_pai_lista = classe_atual.split('/')[:-1]
                classe_pai = '/'.join(classe_pai_lista)
                
                if len(classe_pai) == 0:
                    # Adiciona à lista de instâncias para remover se atingiu a raiz
                    instancias_to_remove.add(classe_atual)
                else:
                    # Agrega instâncias à classe pai no dicionário `alteracoes`
                    alteracoes[classe_pai] += class_counts[classe_atual]
                    alteracoes[classe_atual] = -class_counts[classe_atual]

                    # Atualiza a classe no dataset para o pai
                    dataset.loc[dataset['class'] == classe_atual, 'class'] = classe_pai

        # Aplica as alterações acumuladas ao `class_counts`
        for classe, incremento in alteracoes.items():
            class_counts[classe] += incremento

        # Remove as instâncias que chegaram à raiz sem atingir o número mínimo de instâncias
        drop_idxs = []
        for classe in instancias_to_remove:
            drop_idxs.extend(dataset[dataset['class'] == classe].index)
        dataset = dataset.drop(index=drop_idxs).reset_index(drop=True)

        print(f"Instâncias removidas: {len(drop_idxs)}")
        instancias_to_remove.clear()  # Limpa para a próxima iteração
        nivel_atual -= 1



    print(len(dataset))
    dataframe_to_arff(dataset, 'dataset_agregado', f'Datasets/processados/{nome_dataset}_agregado.arff', hierarquia)
    return dataset

agrega_classes('Datasets/processados/fma_mfcc.arff_sem_valores_ausentes.arff', 'fma_mfcc')