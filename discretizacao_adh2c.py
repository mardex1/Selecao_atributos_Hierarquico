from read_arff import read_arff
from dataframe_to_arff import dataframe_to_arff

import pandas as pd
import numpy as np
import time

# Função que calcula a similaridade entre duas classes. Utilizada no cálculo da /
# distância entre duas classes.
def similaridade(ci, hk, hierarquia):
    n_parentes_similares = 0
    n_parentes_hk = 0
    for classe in hierarquia:
        if classe in hk:
            n_parentes_hk += 1
            if classe in ci:
                n_parentes_similares += 1
    return n_parentes_similares / n_parentes_hk

# Função que calcula a distância entre duas partições adjacentes.
def distancia_particoes_adjacentes(idx_p1, idx_p2, x_agreggated, y, hierarquia): # 1, 2, x, y
    dist = 0
    list_classes = hierarquia
    n_classes = len(list_classes)
    v1 = [0] * n_classes
    v2 = [0] * n_classes
    # Para cada classe na hierarquia, calcular a similaridade entre uma classe
    # qualquer e a classe de cada partição
    for k in range(n_classes):
        hk = list_classes[k]
        nivel_hk = len(hk.split('/'))
        # for i in range(len(x_agreggated)):
        #     if x_agreggated[i] == p1:
        cij = y[idx_p1]
        v1[k] += similaridade(cij, hk, list_classes)
        # if x_agreggated[i] == p2:
        cij2 = y[idx_p2]
        v2[k] += similaridade(cij2, hk, list_classes)
        dist += ((0.8**nivel_hk) * (v1[k] - v2[k])**2)
    return dist**0.5

# Função que conserta o y quando uma partição é removida.
# Exemplo: 1 1 2 4 5 5 -> 1 1 2 3 4 4
def conserta(x_agreggated, idx, n):
    for i in range(len(x_agreggated)-1):
        if x_agreggated[i]+1 < x_agreggated[i+1]:
            while(i < n-1):
                x_agreggated[i+1] -= 1
                i += 1
            break
    return x_agreggated
#Implementação do ADH2C

def ADH2C(data_pd, hierarquia, col):

    data = data_pd.to_numpy()
    # Salvo os indices originais, para que no final, a ordenação mantenha as instâncias como eram inicialmente.
    indices = np.argsort(data[:, 0])
    # Ordenação do atributo contínuo
    data = data[np.argsort(data[:, 0])]
    x_sorted = data[:, 0].astype(float)
    y = data[:, -1]

    # Criação de partições puras, cada valor contínuo recebe um valor.
    partition = 1
    x_partition = []
    for idx in range(len(x_sorted)):
        x_partition.append(partition)
        if idx < len(x_sorted)-1 and x_sorted[idx] != x_sorted[idx+1]:
            partition += 1

    # agregação de partições, se valores contínuos estiverem associados a mesma classe,
    # então elas são agregadas.
    new_partition = 1
    x_agreggated = []
    for idx, (partition, label) in enumerate(zip(x_partition, y)):
        x_agreggated.append(new_partition)
        if idx < len(x_partition)-1 and y[idx] != y[idx+1]: # Verifica_parentesco (só junta as iguais)
            new_partition += 1

    # Construção de soluções candidatas
    n_particoes = len(set(x_agreggated))
    x_inicial = x_agreggated.copy()
    if n_particoes < 4:
        return x_inicial[np.argsort(indices)]

    candidatos = []
    dist_vet = []
    n = len(x_agreggated)
    start = time.time()
    while n_particoes > 1:
       
        dist_min = float('inf')
        idx_min = -1
        # procura a distância miníma entre as partições
        for i in range(n-1):
            if x_agreggated[i] != x_agreggated[i+1]:
                new_dist = distancia_particoes_adjacentes(x_agreggated[i], x_agreggated[i+1], x_agreggated, y, hierarquia)
                
                if new_dist < dist_min:
                    dist_min = new_dist
                    idx_min = i
        
        # a partição que tem a distância mínima se junta a próxima partição
        x_agreggated[idx_min+1] = x_agreggated[idx_min]
        idx_min += 1 # 1 1 1 1 2 3 - 0 1 2 3 4 3
        # sobreescreve toda a próxima partição
        while(idx_min < n-1 and x_agreggated[idx_min]+1 == x_agreggated[idx_min+1]):
            x_agreggated[idx_min+1] = x_agreggated[idx_min]
            idx_min += 1
        

        # Conserta o restante do y, além de colocar a distância mínima no vetor
        # e a solução na lista de candidatos.
        x_agreggated = conserta(x_agreggated, idx_min, n)
        dist_vet.append(dist_min)
        candidatos.append(x_agreggated.copy())
        n_particoes -= 1

    end = time.time()
    print(f'Tempo de execução para a coluna {col}: {end-start}')

    # seleção melhores candidatos
    if min(dist_vet) > 1:
        return x_inicial[np.argsort(indices)]
    melhor = float('-inf')
    k = len(dist_vet)
    for i in range(1, k-1):
        anterior = dist_vet[i-1]
        atual = dist_vet[i]
        proximo = dist_vet[i+1]
        if atual < anterior and atual < proximo:
            maior = max(anterior, proximo)
            qualidade = ((anterior - atual) / maior) + ((proximo-atual)/maior)
            if qualidade >= melhor:
                melhor = qualidade
                id = i
    if melhor == float('-inf'):
        return np.array(x_inicial, dtype='int64')[np.argsort(indices)]
    else:
        return candidatos[id][np.argsort(indices)]


def executa_discretizacao_hierarquica(caminho):
    dataset, hierarquia, columns = read_arff(caminho)
    dataset_discretized = pd.DataFrame(columns=columns)
    y = dataset['class']
    for col in dataset.columns[:-1]:
        data = pd.concat([dataset[col], y], axis=1)
        discretized = ADH2C(data, hierarquia, col)
        dataset_discretized[col] = discretized
    dataset_discretized['class'] = y
    nome_dataset = caminho.split('/')[-1].split('.')[0]
    dataframe_to_arff(dataset_discretized, f'{nome_dataset}_discretizado',
                      f'Datasets/{nome_dataset}_discretizado.arff', hierarquia)

executa_discretizacao_hierarquica('Datasets/nao_processados/Hglass.arff')
