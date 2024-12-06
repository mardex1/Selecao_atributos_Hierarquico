import numpy as np
import pandas as pd
from itertools import combinations

from read_arff import read_arff
from dataframe_to_arff import dataframe_to_arff

def expande(arr):
  arr_exp = []
  for idx, elem in enumerate(arr):
    if elem == 0:
      arr_copy = arr.copy()
      arr_copy[idx] = 1
      arr_exp.append(arr_copy)
  return arr_exp

def avalia(arr, cols, x, y):
    subconjunto = []
    columns = []
    for idx, (elem, col) in enumerate(zip(arr, cols)):
        if elem == 1:
            subconjunto.append(x[:, idx])
            columns.append(col)

    data = pd.DataFrame(np.array(subconjunto).T, columns=columns)
    data['class'] = y

    i_r = inconsistency_rate_h(data)
    return i_r

def best_first(data, limiar):
  x = data.drop('class', axis=1).to_numpy()
  y = data['class'].to_numpy()
  cols = data.columns.tolist()


  explorar = expande([0 for i in range(len(x.T))])
  metric_dict = {tuple(elem):avalia(elem, cols, x, y) for elem in explorar}
  
  visitados = [[0 for i in range(len(x.T))]]
  min_metric_value = float('inf')
  retornos = 0
  melhor_subconjunto = [0 for i in range(len(x.T))]
  
  while len(metric_dict) > 0 and retornos <= limiar:
    # Retorna chave com menor valor associado.
    best_solution = min(metric_dict, key=metric_dict.get)
    best_solution = list(best_solution)
    print('Melhor solução atual: ', best_solution)
    print('inconsistency_rate: ', metric_dict[tuple(best_solution)])
    # Se a métrica for menor que o mínimo que se tem até o momento, atualiza o mínimo que se tem até o momento
    if metric_dict[tuple(best_solution)] <= min_metric_value:
      min_metric_value = metric_dict[tuple(best_solution)]
      melhor_subconjunto = best_solution
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
  return visitados, melhor_subconjunto, min_metric_value

def find_subset(S, D):
    y = D['class']
    x = D.drop('class', axis=1)

    # Calcula a inconsistência para o conjunto completo
    valor_conjunto_completo = inConCal(S, x, y)
    
    for tam in range(1, len(S)):
        # Gera todos os subconjuntos de tamanho = tam
        for subconjuntos in combinations(S, tam):
            
            subconjuntos = list(set(subconjuntos))
            # Verifica se o subconjunto tem a inconsistencia menor que o conjunto completo.
            if inConCal(subconjuntos, x, y) <= valor_conjunto_completo:
                return subconjuntos
    return None

def inConCal(features, x, y):
    data = pd.DataFrame(x, columns=features)
    data['class'] = y
    ir = inconsistency_rate_h(data)
    print(f'Número features={len(features)} -> Inconsistency rate={ir}')
    return ir

def find_height(y):
    # Cria uma lista com os tamanhos de cada string de label e pega o maior
    max_len = max(len(label.split('/')) for label in y)
    return max_len

def truncate_data(data_selecionado, nivel_atual):
  data_truncado = data_selecionado.copy()
  
  y_new = []
  for idx, elem in data_truncado['class'].items():
    temp_elem = elem
    while len(temp_elem.split('/')) > nivel_atual:
       temp_elem_list = temp_elem.split('/')[:-1]
       temp_elem = '/'.join(temp_elem_list)
    data_truncado.loc[idx, 'class'] = temp_elem
  return data_truncado


def inconsistency_rate_h(data):
    nivel_maximo = find_height(data['class'])
    arr_res = []
    # Itera por cada nível
    for nivel_atual in range(1, nivel_maximo+1):
        # Seleciona um subconjunto a qual todas as classes pertencem ou são descendentes do nível atual 
        data_selecionado = data[data['class'].str.split('/').apply(len) >= nivel_atual]

        # Todas as classes descendentes são transformadas em classe de um único nível:
        # nivel_atual = 2 - g/w/b -> g/w
        data_truncado = truncate_data(data_selecionado, nivel_atual)

        w_i = (nivel_maximo - nivel_atual + 1) * (2/(nivel_maximo*(nivel_maximo+1))) # pesos somam 1
        i_r = inconsistency_rate(data_truncado)
        arr_res.append(i_r*w_i)
    return sum(arr_res)

def get_padroes(data):
    # Dicionário no formato {(padrão), [classes que o padrão aparece]}
    # Exemplo {(0,1,0,1), [1,1,2,1,3,3]}
    x = data.drop("class", axis=1).values.tolist()
    y = data["class"].values.tolist()
    padroes = {}
    for i, padrao in enumerate(x):
      print(padrao)
      if tuple(padrao) in padroes:
        padroes[tuple(padrao)].append(y[i])
      else:
        padroes[tuple(padrao)] = [y[i]]
    return padroes, x, y

def inconsistency_rate(data):
    # Retorna um dicionário "padrões", que é composto por um tupla com o padrão e as classes as quais ele aparece(podem haver classes repetidas).
    padroes, x, y = get_padroes(data)
    
    padroes_len = {padrao: len(classe_p) for padrao, classe_p in padroes.items()}
    # print("******************************************************************")
    # print(sorted(padroes_len.items(), key=lambda padroes_len: padroes_len[1], reverse=True))
    inconsistency_counts = []
    print(padroes)
    for padrao, classes_p in padroes.items():
        # Procuramos por padrões que acontecem mais que uma vez.
        if len(classes_p) > 1:
            # Retorna a contagem de cada classe em que o padrão específico aparece
            uniq_value, class_counts = np.unique(classes_p, return_counts=True)
            
            # Significa que o padrão tem apenas uma classe predominante, então é consistente
            if len(class_counts) == 1:
                continue
            
            freq_majoritary_class = np.max(class_counts)
            inconsistency_counts.append(len(classes_p) - freq_majoritary_class)
    inconsistency_rate = sum(inconsistency_counts) / len(x)
    return inconsistency_rate

data, hier, cols = read_arff('Datasets/processados/fma_mfcc.arff_discretizado.arff')

# print(find_subset(cols, data)) ['feature3', 'feature12', 'feature9', 'feature7', 'feature5', 'feature13', 'feature1', 'feature8', 'feature6', 'feature4', 'feature10', 'feature11']






data = np.array([[3, 2, 10,"R.1.1"],
    [3, 2, 11,"R.1.1"],
    [1, 4, 10,"R.2"],
    [1, 4, 12,"R.2.1"],
    [4, 2, 11,"R.1.1"],
    [2, 5, 13, "R.2.1"],
    [1, 4, 15, "R.2"],
    [3, 2, 15, "R.1.1.1"],
    [1, 5, 12, "R.2.1"],
    [3, 1, 21, "R.1.2"],
    [3, 2, 21, "R.1.1.1"],
    [1, 4, 11,"R.2.1"],
    [3, 1, 10,"R.1.2"],
    [3, 1, 11,"R.1.2"]])
    # X = data[:, :-1].astype(np.float64)
    # y = data[:, -1]
data = pd.DataFrame(data, columns=['col1', 'col2', 'col3', 'class'])

print(best_first(data, 10))
print(inconsistency_rate_h(data))