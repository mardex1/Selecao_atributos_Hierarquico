from concatena_arquivo_arff import concatena_arquivo_arff
from multirotulo_to_monorotulo import make_monorotulo
from sub_missing_values import sub_missing_values
from agrega_classes import agrega_classes
from discretizacao_nao_supervisionada import discretizacao_nao_supervisionada

def run():
    print('Executando Pré-processamento...')

    dataset_concatenado = concatena_arquivo_arff('Datasets/nao_processados')

    dataset_monorotulo = make_monorotulo('Datasets/dataset_concatenado.arff')

    dataset_no_missing = sub_missing_values('Datasets/dataset_monorotulo.arff')

    dataset_agregado, dict_count = agrega_classes('Datasets/dataset_sem_valores_ausentes.arff')
    print(dataset_agregado.dtypes)
    dataset_discretizado = discretizacao_nao_supervisionada('Datasets/dataset_agregado.arff')

    print('Pré-processamento realizado com sucesso!')


run()
