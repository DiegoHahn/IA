import pandas as pd

def carregar_dados(data_path):
    """
    Carrega os dados do arquivo CSV.
    """
    return pd.read_csv(data_path)

def agregar_dados(data):
    """
    Remove duplicatas ou agrega os dados.
    """
    return data.groupby(['pk_TimeStamp', 'pk_fk_Id']).mean().reset_index()

def transformar_dados(data):
    """
    Transforma os dados longos para largos e preenche valores ausentes.
    """
    # Exemplo de um mapeamento de IDs para nomes de colunas
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
        88: 'BA04_status'
    }

    # Transformar dados longos para largos
    data_pivot = data.pivot(index='pk_TimeStamp', columns='pk_fk_Id', values='Value').reset_index()

    # Renomear colunas com base nos IDs fornecidos
    data_pivot.rename(columns=id_to_name, inplace=True)

    # Preencher valores ausentes com o último valor válido
    data_pivot.ffill(inplace=True)

    # Excluir linhas que possuem qualquer valor NaN
    data_pivot.dropna(inplace=True)

    return data_pivot

def salvar_dados(data, output_path):
    """
    Salva os dados transformados em um arquivo CSV com encoding UTF-8 e delimitador de vírgula.
    """
    data.to_csv(output_path, index=False, encoding='utf-8', sep=',')

def main():
    data_path = 'C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/arquivoMotorIA.csv'
    output_path = 'C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/data_pivot.csv'

    # Carregar os dados
    data = carregar_dados(data_path)

    # Remover duplicatas ou agregá-las
    data_aggregated = agregar_dados(data)

    # Transformar os dados
    data_pivot = transformar_dados(data_aggregated)

    # Salvar os dados transformados em um arquivo CSV
    salvar_dados(data_pivot, output_path)

    # Exibir as primeiras linhas para confirmar a transformação
    print(data_pivot.head())

if __name__ == '__main__':
    main()
