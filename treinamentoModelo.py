"""
Treina um Isolation Forest para detectar anomalias na telemetria dos motores.

Lê um CSV exportado da tabela de log dos motores, normaliza as colunas de
telemetria com StandardScaler e salva o modelo e o scaler com joblib.

Uso:
    python treinamentoModelo.py --data data/arquivoMotorIA.csv
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv é opcional
    load_dotenv = None

# Colunas usadas como entrada do modelo. A previsão usa exatamente a mesma lista,
# na mesma ordem, para que o scaler e o modelo recebam os dados no formato do treino.
FEATURE_COLUMNS = ['ID Motor', 'Status', 'SP Frequencia', 'Feed Corrente', 'Feed Tensão Entrada']

DEFAULT_DATA_PATH = 'data/arquivoMotorIA.csv'
DEFAULT_MODEL_PATH = 'models/isolation_forest_model.pkl'
DEFAULT_SCALER_PATH = 'models/scaler.pkl'


def carregar_dados(data_path):
    """
    Carrega os dados de treino a partir de um arquivo CSV.

    TODO: o treino só aceita CSV. Para treinar direto do banco, reaproveitar
    `carregar_dados_do_banco` de previsao.py com uma consulta que traga o
    histórico considerado normal.
    """
    return pd.read_csv(data_path)


def pre_processar_dados(data):
    """
    Seleciona as colunas de telemetria e normaliza com StandardScaler.
    """
    faltando = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if faltando:
        raise ValueError(f'Colunas ausentes no CSV de treino: {faltando}')

    features = data[FEATURE_COLUMNS]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, scaler


def treinar_modelo(features_scaled, contamination, n_estimators, max_samples, max_features, bootstrap):
    """
    Treina o modelo Isolation Forest nos dados fornecidos.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42
    )
    model.fit(features_scaled)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Treina o Isolation Forest de detecção de anomalias.')
    parser.add_argument('--data', default=os.getenv('IA_DATA_PATH', DEFAULT_DATA_PATH),
                        help=f'CSV de treino (padrão: $IA_DATA_PATH ou {DEFAULT_DATA_PATH})')
    parser.add_argument('--model', default=os.getenv('IA_MODEL_PATH', DEFAULT_MODEL_PATH),
                        help=f'Onde salvar o modelo (padrão: $IA_MODEL_PATH ou {DEFAULT_MODEL_PATH})')
    parser.add_argument('--scaler', default=os.getenv('IA_SCALER_PATH', DEFAULT_SCALER_PATH),
                        help=f'Onde salvar o scaler (padrão: $IA_SCALER_PATH ou {DEFAULT_SCALER_PATH})')
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='Proporção esperada de anomalias nos dados de treino (padrão: 0.1)')
    parser.add_argument('--n-estimators', type=int, default=200, help='Número de árvores (padrão: 200)')
    parser.add_argument('--max-samples', type=int, default=256, help='Amostras por árvore (padrão: 256)')
    return parser.parse_args()


def main():
    if load_dotenv:
        load_dotenv()
    args = parse_args()

    data = carregar_dados(args.data)
    features_scaled, scaler = pre_processar_dados(data)

    model = treinar_modelo(
        features_scaled,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=1.0,
        bootstrap=False,
    )

    for path in (args.model, args.scaler):
        pasta = os.path.dirname(path)
        if pasta:
            os.makedirs(pasta, exist_ok=True)

    joblib.dump(model, args.model)
    joblib.dump(scaler, args.scaler)

    rotulos = model.predict(features_scaled)
    anomalias = int((rotulos == -1).sum())
    print(f'Treino concluído com {len(data)} linhas. '
          f'Marcadas como anomalia no próprio treino: {anomalias} ({anomalias / len(data):.1%}).')
    print(f'Modelo salvo em {args.model} e scaler em {args.scaler}.')


if __name__ == '__main__':
    main()
