# Função que executa as funções de uma vez, seguindo uma ordem. Concatena arquivos -> Transforma em monorotulo -> remove valores ausentes -> Torna cada classe com pelo menos 10 elementos -> discretiza a base. 

from concatena_arquivo_arff import concatena_arquivo_arff
from multirotulo_to_monorotulo import make_monorotulo
from sub_missing_values import sub_missing_values
from agrega_classes import agrega_classes
from discretizacao_nao_supervisionada import discretizacao_nao_supervisionada
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def run(caminho):
    nome_dataset = caminho.split('/')[-1]

    dataset_concatenado = concatena_arquivo_arff(caminho, nome_dataset)

    dataset_monorotulo = make_monorotulo(f'Datasets/processados/{nome_dataset}_concatenado.arff', 
                                         nome_dataset)

    dataset_no_missing = sub_missing_values(f'Datasets/processados/{nome_dataset}_monorotulo.arff', 
                                            nome_dataset)

    dataset_agregado, dict_count = agrega_classes(f'Datasets/processados/{nome_dataset}_sem_valores_ausentes.arff', 
                                                  nome_dataset)

    dataset_discretizado = discretizacao_nao_supervisionada(f'Datasets/processados/{nome_dataset}_agregado.arff', 
                                                            nome_dataset)
    

if __name__ == '__main__':
    print('Executando Pré-processamento...')
    
    run('Datasets/nao_processados/cellcycle')
    
    print('Pré-processamento realizado com sucesso!')
