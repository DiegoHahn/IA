import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def carregar_dados(data_path):
    """
    Carrega os dados do arquivo CSV.
    """
    return pd.read_csv(data_path)

def pre_processar_dados(data):
    """
    Normaliza os dados e separa as características (features) do alvo (target).
    """
    # Remover a coluna de timestamp, pois não será usada no treinamento
    features = data.drop(columns=['Timestamp'])

    # Normalizar os dados
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, scaler

def treinar_modelo(features_scaled, contamination=0.001, n_estimators=100, max_samples='auto', max_features=1.0, bootstrap=False):
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

def fazer_predicoes(model, features_scaled):
    """
    Faz previsões de anomalias com o modelo treinado.
    """
    return model.predict(features_scaled)

def salvar_resultados(data, predicoes, output_path):
    """
    Salva os dados com as previsões de anomalias em um arquivo CSV.
    """
    # Adicionar a coluna de anomalias aos dados originais
    data['anomaly'] = predicoes
    data.to_csv(output_path, index=False, encoding='utf-8', sep=',')

def main():
    data_path = 'C:/Users/Diego/Documents/Programacao/Satc/5fase/IA/IA/arquivoMotorIA.csv'
    output_path = 'C:/Users/Diego/Documents/Programacao/Satc/5fase/IA/IA/data_pivot_with_anomalies.csv'

    # Carregar os dados
    data = carregar_dados(data_path)

    # Pré-processar os dados
    features_scaled, scaler = pre_processar_dados(data)

    # Treinar o modelo com parâmetros ajustáveis
    model = treinar_modelo(features_scaled, contamination=0.05, n_estimators=200, max_samples=256, max_features=1.0, bootstrap=False)

    # Fazer previsões de anomalias
    predicoes = fazer_predicoes(model, features_scaled)

    # Salvar os resultados
    salvar_resultados(data, predicoes, output_path)

    # Exibir as primeiras linhas para confirmar a transformação
    print(data.head())

if __name__ == '__main__':
    main()
