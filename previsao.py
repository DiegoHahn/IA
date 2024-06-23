import pandas as pd
import joblib
from sqlalchemy import create_engine, text

def carregar_dados_do_banco(engine, query):
    """
    Carrega os dados do banco de dados PostgreSQL usando uma consulta SQL.
    """
    with engine.connect() as connection:
        data = pd.read_sql_query(query, connection)
    return data

def pre_processar_dados(data, scaler, feature_columns):
    """
    Normaliza os dados usando o scaler fornecido.
    """
    # Selecionar apenas as colunas necessárias para previsão
    features = data[feature_columns]

    # Normalizar os dados com o scaler fornecido
    features_scaled = scaler.transform(features)

    return features_scaled

def fazer_predicoes(model, features_scaled):
    """
    Faz previsões de anomalias com o modelo treinado.
    """
    predicoes = model.predict(features_scaled)
    # Converter -1 para 1 (anomalia) e 1 para 0 (normal)
    predicoes = [1 if p == -1 else 0 for p in predicoes]
    return predicoes

def inserir_resultados(engine, data, predicoes):
    """
    Insere os resultados de volta na tabela do banco de dados.
    """
    with engine.begin() as connection:
        for i, pred in enumerate(predicoes):
            result = connection.execute(text("""
                UPDATE log_motor
                SET anomalia = :anomalia
                WHERE "ID Log" = :id_log;
            """), {'anomalia': int(pred), 'id_log': int(data['ID Log'].iloc[i])})
            # Log para verificar se a atualização foi realizada
            print(f"Atualização {i + 1}/{len(predicoes)}: Linhas afetadas = {result.rowcount}")

# cria um arquivo CSV para com os dados e as previsões de anomalias (PODE REMOVER DEPOIS)
def salvar_resultados(data, predicoes, output_path):
    """
    Salva os dados com as previsões de anomalias em um arquivo CSV.
    """
    # Adicionar a coluna de anomalias aos dados originais
    data['anomalia'] = predicoes
    data.to_csv(output_path, index=False, encoding='utf-8', sep=',')

def main():
   # Configurações de conexão (PRECISA ARRUMAR OS DADOS DA CONEXÃO COM O BANCO) esse outh_path é o local onde será salvo o arquivo CSV (se não for mais necessário, pode remover)
    engine = create_engine('postgresql+psycopg2://USER:SENHA@localhost:5432/NOMEDOBANCO')
    # output_path = 'C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/previsao_anomalias.csv'
    
    # Query para selecionar os dados (TALVEZ ADICIONAR UM INDICE NO BANCO PARA MELHORAR A PERFORMANCE)
    query = "SELECT * FROM log_motor WHERE anomalia IS NULL;"

    # Carregar os dados do banco de dados
    data = carregar_dados_do_banco(engine, query)

    # Carregar o modelo treinado e o scaler
    model = joblib.load('C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/isolation_forest_model.pkl')
    scaler = joblib.load('C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/scaler.pkl')

    # Definir os nomes das colunas usadas para o treinamento do modelo
    feature_columns = ['ID Motor', 'Status', 'SP Frequencia', 'Feed Corrente', 'Feed Tensão Entrada']

    # Pré-processar os dados
    features_scaled = pre_processar_dados(data, scaler, feature_columns)

    # Fazer previsões de anomalias
    predicoes = fazer_predicoes(model, features_scaled)

    # Inserir os resultados de volta na tabela do banco de dados
    inserir_resultados(engine, data, predicoes)
    
    # Salvar os resultados em um arquivo CSV para testar (PODE REMOVER DEPOIS)
    # salvar_resultados(data, predicoes, output_path)

if __name__ == '__main__':
    main()
