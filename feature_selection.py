import numpy as np
import pandas as pd
from itertools import combinations
from read_arff import read_arff

def expande(arr):
  arr_exp = []
  for idx, elem in enumerate(arr):
    if elem == 0:
      arr_copy = arr.copy()
      arr_copy[idx] = 1
      arr_exp.append(arr_copy)
  return arr_exp


def avalia_plano(arr, cols, x, y):
    subconjunto = []
    columns = []
    for idx, (elem, col) in enumerate(zip(arr, cols)):
        if elem == 1:
            subconjunto.append(x[:, idx])
            columns.append(col)

    data = pd.DataFrame(np.array(subconjunto).T, columns=columns)
    data['class'] = y

    i_r, porcent_padroes_unicos = inconsistency_rate(data, False)
    return i_r, porcent_padroes_unicos


def avalia_h(arr, cols, x, y):
    subconjunto = []
    columns = []
    for idx, (elem, col) in enumerate(zip(arr, cols)):
        if elem == 1:
            subconjunto.append(x[:, idx])
            columns.append(col)

    data = pd.DataFrame(np.array(subconjunto).T, columns=columns)
    data['class'] = y

    i_r, porcent_padroes_unicos = inconsistency_rate_h(data)
    return i_r, porcent_padroes_unicos

def best_first(data, limiar, hierarchical):
    if hierarchical is True:
        avalia = avalia_h
    else:
        avalia = avalia_plano

    x = data.drop('class', axis=1).to_numpy()
    y = data['class'].to_numpy()
    cols = data.columns.tolist()

    explorar = expande([0 for i in range(len(x.T))])
    metric_dict = {tuple(elem): avalia(elem, cols, x, y) for elem in explorar}
    visitados = [[0 for i in range(len(x.T))]]
    min_metric_value = float('inf')
    retornos = 0
    melhor_subconjunto = [0 for i in range(len(x.T))]

    while len(metric_dict) > 0 and retornos <= limiar:
    # Retorna chave com menor valor associado.
        best_solution_tuple = min(metric_dict.items(), key=lambda item: item[1][0]) 
        best_solution = list(best_solution_tuple[0])
        print('Número de features: ', sum(best_solution))
        print('Inconsistency_rate: ', metric_dict[tuple(best_solution)][0])
        print(f'Porcentagem de valores únicos: {round(metric_dict[tuple(best_solution)][1], 2)}%')
        # Se a métrica for menor que o mínimo que se tem até o momento, atualiza o mínimo que se tem até o momento
        if metric_dict[tuple(best_solution)][0] < min_metric_value:
            min_metric_value = metric_dict[tuple(best_solution)][0]
            melhor_subconjunto = best_solution
            porcent_unique_melhor_subconjunto = metric_dict[tuple(best_solution)][1]
        # Se não, checa se houve retorno, ou seja, a nova solução tem o mesmo tanto ou um número menor de atributos que a ultima solução.
        elif sum(best_solution) <= sum(visitados[-1]):
            retornos += 1

        visitados.append(best_solution)
        metric_dict.pop(tuple(best_solution))
        explorar = expande(best_solution)

        # Avalia as soluções que ainda não foram visitadas
        for elem in explorar:
            if elem not in visitados:
                metric_dict[tuple(elem)] = avalia(elem, cols, x, y)

    return visitados, melhor_subconjunto, min_metric_value, porcent_unique_melhor_subconjunto

def find_height(y):
    # Cria uma lista com os tamanhos de cada string de label e pega o maior
    max_len = max(len(label.split('.')) for label in y) - 1
    return max_len

def truncate_data(data_selecionado, nivel_atual):
  data_truncado = data_selecionado.copy()
  data_truncado['class'] = (
      data_truncado['class'].astype(str).str.split('.').apply(lambda parts: '.'.join(parts[:nivel_atual+1]))
  )
  return data_truncado


def inconsistency_rate_h(data):
    nivel_maximo = find_height(data['class'])
    arr_res = []
    porcent_padroes_unicos_arr = []
    # Itera por cada nível
    depth = data['class'].astype(str).str.split('.').str.len() - 1
    weights = [(nivel_maximo - i + 1) * (2 / (nivel_maximo * (nivel_maximo + 1))) for i in range(1, nivel_maximo + 1)]
    for nivel_atual in range(1, nivel_maximo+1):
        # Seleciona um subconjunto a qual todas as classes pertencem ou são descendentes do nível atual 
        data_selecionado = data[depth >= nivel_atual]
        # Todas as classes descendentes são transformadas em classe de um único nível:
        # nivel_atual = 2 - g/w/b -> g/w
        data_truncado = truncate_data(data_selecionado, nivel_atual)
        print(data_truncado)
        print(weights)
        print(nivel_atual)
        print(weights[nivel_atual-1])

        i_r, porcent_padroes_unicos = inconsistency_rate(data_truncado, True)
        arr_res.append(i_r*weights[nivel_atual-1])
        porcent_padroes_unicos_arr.append(porcent_padroes_unicos)
    return sum(arr_res), np.mean(porcent_padroes_unicos)

def get_padroes(data):
    # Dicionário no formato {(padrão), [classes que o padrão aparece]}
    # Exemplo {(0,1,0,1), [1,1,2,1,3,3]}
    x = data.drop("class", axis=1).values.tolist()
    y = data["class"].values.tolist()
    padroes = {}
    for i, padrao in enumerate(x):
      if tuple(padrao) in padroes:
        padroes[tuple(padrao)].append(y[i])
      else:
        padroes[tuple(padrao)] = [y[i]]
    return padroes, x, y


def inconsistency_rate(data, adapted):
    # Retorna um dicionário "padrões", que é composto por um tupla com o padrão e as classes as quais ele aparece(podem haver classes repetidas).
    padroes, x, y = get_padroes(data)
    inconsistency_counts = []
    # (0, 1, 0): [R.1.1, R.2.1, R.1.1] ([R.1.1, R.1.1], [2]) 3-2= 1
    n_padroes_unicos = 0
    for padrao, classes_p in padroes.items():
        # Procuramos por padrões que acontecem mais que uma vez.
        if len(classes_p) == 1:
            n_padroes_unicos += 1
        else:
            # Retorna a contagem de cada classe em que o padrão específico aparece
            uniq_value, class_counts = np.unique(classes_p, return_counts=True)

            # Significa que o padrão tem apenas uma classe predominante, então é consistente
            if len(class_counts) == 1:
                continue

            freq_majoritary_class = np.max(class_counts)
            inconsistency_counts.append(len(classes_p) - freq_majoritary_class)
    porcent_padroes_unicos = n_padroes_unicos / len(padroes) * 100
    if adapted is True:
        inconsistency_rate = (sum(inconsistency_counts) / (len(x) - n_padroes_unicos))
    else:
        inconsistency_rate = sum(inconsistency_counts) / len(x)
    return inconsistency_rate, porcent_padroes_unicos
