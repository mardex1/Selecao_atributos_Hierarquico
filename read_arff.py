import pandas as pd


def read_arff(path):
    """Função que lê um arquivo no formato arff, e o transforma em uma Dataframe do Pandas"""
    with open(path, 'r') as f:
        # Lê o arquivo linha a linha
        lines = f.readlines()

    columns = []
    hierarquia = ""

    for i, line in enumerate(lines):
        line = line.strip() # Remove espaços em branco

        if line.lower().startswith('@relation'):
            relation_name = line.split()[1]  # Nome da relação
        elif line.lower().startswith('@attribute'):
            partes = line.split() # Separa em partes por espaço.
            nome_atributo = partes[1] # A primeira parte é sempre o nome do atributo
            columns.append(partes[1]) # Cria uma lista com os nomes das colunas para a criação de um DataFrame pandas.
            tipo_atributo = partes[2:]# A segunda parte é sempre o seu tipo
            if nome_atributo == 'class':
                hierarquia = partes[2:][0]# se o tipo de atributo for hierarchical, a terceira parte será a hierarquia de fato]
                hierarquia = hierarquia[1:] # removing {
                hierarquia = hierarquia[:-1] # removing }

        elif line.lower().startswith('@data'):
            data_start_index = i + 1 # Armazena o indice onde os dados começam
            break

    data = []
    for line in lines[data_start_index:]:
        line = line.strip() # Remove espaços em branco

        values = line.split(',') # Retorna uma lista com cada elemento separado por virgulas
        data.append(values) # Coloca tudo em um array no formato [[instancia1], [instancia2]], onde cada instancia apresenta 78 colunas, com uma delas sendo a classe.

    dataset = pd.DataFrame(data=data, columns=columns) # Cria um dataFrame do pandas.
    for col in dataset.columns[:-1]:
        dataset[col] = pd.to_numeric(dataset[col], errors='ignore')
    dataset['class'].astype('object')
    return dataset, hierarquia.split(","), columns
