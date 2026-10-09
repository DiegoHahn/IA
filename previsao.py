"""
Classifica registros de telemetria como normais (0) ou anômalos (1) usando o
modelo treinado por treinamentoModelo.py.

Por padrão lê do PostgreSQL as linhas de `log_motor` com `anomalia IS NULL` e
grava o resultado de volta na coluna `anomalia`. Também pode ler de um CSV e
salvar as previsões em CSV, sem acessar o banco.

Uso:
    python previsao.py                                  # banco -> banco (DATABASE_URL)
    python previsao.py --input-csv dados.csv --output-csv previsoes.csv --no-db-write

ATENÇÃO: os arquivos .pkl são carregados com joblib (pickle), que executa código
ao carregar. Só use modelos que você mesmo treinou ou de fonte confiável.
"""

import argparse
import os
import sys

import joblib
import pandas as pd

from treinamentoModelo import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH, FEATURE_COLUMNS

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv é opcional
    load_dotenv = None

# Nome fixo da tabela: identificadores não podem ser parametrizados com segurança na query.
# TODO: em bases grandes, criar um índice parcial para acelerar esta consulta, por exemplo
#   CREATE INDEX IF NOT EXISTS idx_log_motor_sem_previsao ON log_motor ("ID Log") WHERE anomalia IS NULL;
QUERY_PENDENTES = 'SELECT * FROM log_motor WHERE anomalia IS NULL;'


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
    Grava as previsões na coluna `anomalia` da tabela `log_motor`, em uma única transação.
    """
    from sqlalchemy import text

    parametros = [
        {'anomalia': int(pred), 'id_log': int(id_log)}
        for pred, id_log in zip(predicoes, data['ID Log'])
    ]
    if not parametros:
        return 0

    # executemany: o rowcount não é confiável em todos os drivers, então
    # retornamos quantos registros foram enviados para atualização.
    with engine.begin() as connection:
        connection.execute(text("""
            UPDATE log_motor
            SET anomalia = :anomalia
            WHERE "ID Log" = :id_log;
        """), parametros)
    return len(parametros)


def salvar_resultados(data, predicoes, output_path):
    """
    Salva os dados com as previsões de anomalias em um arquivo CSV.
    """
    data = data.copy()
    data['anomalia'] = predicoes
    data.to_csv(output_path, index=False, encoding='utf-8', sep=',')


def parse_args():
    parser = argparse.ArgumentParser(description='Detecta anomalias na telemetria dos motores.')
    parser.add_argument('--database-url', default=os.getenv('DATABASE_URL'),
                        help='URL SQLAlchemy do PostgreSQL (padrão: $DATABASE_URL)')
    parser.add_argument('--model', default=os.getenv('IA_MODEL_PATH', DEFAULT_MODEL_PATH),
                        help=f'Modelo treinado (padrão: $IA_MODEL_PATH ou {DEFAULT_MODEL_PATH})')
    parser.add_argument('--scaler', default=os.getenv('IA_SCALER_PATH', DEFAULT_SCALER_PATH),
                        help=f'Scaler treinado (padrão: $IA_SCALER_PATH ou {DEFAULT_SCALER_PATH})')
    parser.add_argument('--input-csv',
                        help='Lê os registros deste CSV em vez do banco')
    parser.add_argument('--output-csv',
                        help='Salva os registros com a coluna "anomalia" neste CSV')
    parser.add_argument('--no-db-write', action='store_true',
                        help='Não grava as previsões no banco')
    return parser.parse_args()


def main():
    if load_dotenv:
        load_dotenv()
    args = parse_args()

    precisa_banco = args.input_csv is None or not args.no_db_write
    engine = None
    if precisa_banco:
        if not args.database_url:
            sys.exit('Defina DATABASE_URL (ou --database-url), ou use --input-csv com --no-db-write.')
        from sqlalchemy import create_engine
        engine = create_engine(args.database_url)

    if args.input_csv:
        data = pd.read_csv(args.input_csv)
    else:
        data = carregar_dados_do_banco(engine, QUERY_PENDENTES)

    if data.empty:
        print('Nenhum registro para classificar.')
        return

    # Carregar o modelo treinado e o scaler (pickle: apenas arquivos confiáveis)
    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    features_scaled = pre_processar_dados(data, scaler, FEATURE_COLUMNS)
    predicoes = fazer_predicoes(model, features_scaled)

    total_anomalias = sum(predicoes)
    print(f'{len(predicoes)} registros classificados, {total_anomalias} marcados como anomalia.')

    if not args.no_db_write:
        linhas = inserir_resultados(engine, data, predicoes)
        print(f'Previsões gravadas no banco para {linhas} registros.')

    if args.output_csv:
        salvar_resultados(data, predicoes, args.output_csv)
        print(f'Previsões salvas em {args.output_csv}')


if __name__ == '__main__':
    main()
