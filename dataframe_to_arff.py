import pandas as pd
import numpy as np

def dataframe_to_arff(df, nome_relacao, caminho_arquivo, hierarquia):
    # Abrindo o arquivo para escrita
    with open(caminho_arquivo, 'w') as f:
        # Escrevendo a relação
        f.write(f"@RELATION '{nome_relacao}'\n\n")

        # Escrevendo os atributos
        for coluna in df.columns:
            tipo = df[coluna].dtype
            if coluna != 'class': # Significa que era uma coluna numérica porém foi discretizada
                f.write(f"@ATTRIBUTE {coluna} \t\t\t\tnumeric\n")
            else:
                f.write(f'@ATTRIBUTE {coluna} \t\t\t\thierarchical ')
                f.write(','.join(hierarquia))
                f.write('\n')

        f.write("\n@DATA\n")

        # Escrevendo os dados
        for index, row in df.iterrows():
            dados = []
            for coluna in df.columns:
                dados.append(str(row[coluna]))
            f.write(f"{','.join(dados)}\n")