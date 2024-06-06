import pandas as pd

def carregar_dados(caminho_arquivo):
    """
    Carrega os dados do arquivo CSV.
    """
    return pd.read_csv(caminho_arquivo)

def agregar_dados(data):
    """
    Remove duplicatas ou agrega os dados.
    """
    return data.groupby(['pk_TimeStamp', 'pk_fk_Id']).mean().reset_index()

def transformar_dados(data):
    """
    Transforma os dados longos para largos e preenche valores ausentes.
    """
    # mapeamento de IDs para nomes de colunas
    id_to_name = {
        70: 'Vazão m3/h',
        71: 'SP_Velocidade',
        73: 'feed_velocidade',
        74: 'feed_corrente',
        75: 'feed_tensão',
        76: 'SP_Velocidade_BA03',
        78: 'feed_velocidade_BA03',
        79: 'feed_corrente_BA03',
        80: 'feed_tensão_BA03',
        81: 'SP_Velocidade_BA04',
        83: 'feed_velocidade_BA04',
        84: 'feed_corrente_BA04',
        85: 'feed_tensão_BA04',
        86: 'BA01_status',
        87: 'BA03_status',
        88: 'BA04_status',
    }

    # Transformar dados longos para largos
    data_pivot = data.pivot(index='pk_TimeStamp', columns='pk_fk_Id', values='Value').reset_index()

    # Renomear colunas com base nos IDs fornecidos
    data_pivot.rename(columns=id_to_name, inplace=True)

    # Preencher valores ausentes com o último valor válido
    data_pivot.ffill(inplace=True)

    return data_pivot

def main():
    data_path = 'C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/arquivoMotorIA.csv'

    # Carregar os dados
    data = carregar_dados(data_path)

    # Remover duplicatas ou agregá-las
    data_aggregated = agregar_dados(data)

    # Transformar os dados
    data_pivot = transformar_dados(data_aggregated)

    # Excluir linhas que possuem qualquer valor NaN SOLUÇÂO PRECISA ESTAR AQUI NESSE FLUXO
    data_pivot.dropna(inplace=True)

    # Exibir as primeiras valores NaN estão presentes ver como fazer essa parte
    print(data_pivot.head())

    # Exibir as Últimas linhas para confirmar a transformação
    print(data_pivot.tail())

if __name__ == '__main__':
    main()
