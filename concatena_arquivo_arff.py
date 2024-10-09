from pathlib import Path
import numpy as np
import pandas as pd

from read_arff import read_arff
from dataframe_to_arff import dataframe_to_arff

def concatena_arquivo_arff(caminho, nome_dataset):
    caminho = Path(caminho)

    if caminho.is_dir():
        primeira_iter = True
        for arquivo in caminho.iterdir():
            dataset, hierarquia, cols = read_arff(arquivo)
            if primeira_iter:
                data_full = pd.DataFrame(columns=cols)
                primeira_iter = False
            data_full = pd.concat([data_full, dataset], axis=0, ignore_index=True)
            
        dataframe_to_arff(data_full, 'dataset_concatenado', f'Datasets/processados/{nome_dataset}_concatenado.arff', hierarquia)

        return data_full
    else:
        print('O caminho fornecido não é um diretório.')
