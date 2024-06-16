import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def carregar_dados(data_path):
    """
    Carrega os dados do arquivo CSV. Se for importar do banco de dados precisa ajustar
    """
    return pd.read_csv(data_path)

def pre_processar_dados(data):
    """
    Normaliza os dados e separa as características (features) do alvo (target).
    """
    # Remover colunas que não serão usadas no treinamento
    features = data.drop(columns=['Timestamp', 'ID Log'])

    # Normalizar os dados
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, scaler

def treinar_modelo(features_scaled, contamination=0.01, n_estimators=100, max_samples='auto', max_features=1.0, bootstrap=False):
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

def main():
    data_path = 'C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/arquivoMotorIA.csv'
    model_path = 'C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/isolation_forest_model.pkl'
    scaler_path = 'C:/Users/Diego/OneDrive/Documentos/Programacao/Satc/IA/scaler.pkl'

    # Carregar os dados
    data = carregar_dados(data_path)

    # Pré-processar os dados
    features_scaled, scaler = pre_processar_dados(data)

    # Treinar o modelo com parâmetros ajustáveis
    model = treinar_modelo(features_scaled, contamination=0.05, n_estimators=200, max_samples=256, max_features=1.0, bootstrap=False)

    # Salvar o modelo treinado e o scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
if __name__ == '__main__':
    main()
