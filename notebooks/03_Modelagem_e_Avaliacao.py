import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Caminho para os dados pré-processados (relativo ao diretório de execução /home/ubuntu/ciencia_dados_n3)
DATA_PATH = "data/processed_data.npz"
MODEL_PATH = "modelo_final.pkl"

# 1. Carregamento dos Dados
try:
    data = np.load(DATA_PATH, allow_pickle=True)
    # Reconstruindo o DataFrame com os nomes das features
    X_train = pd.DataFrame(data['X_train'], columns=data['feature_names'])
    X_test = pd.DataFrame(data['X_test'], columns=data['feature_names'])
    y_train = data['y_train']
    y_test = data['y_test']
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {DATA_PATH}.")
    exit()

# 2. Definição dos Modelos
models = {
    "Regressão Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42), # Adicionando random_state para reprodutibilidade
    "Árvore de Decisão": DecisionTreeRegressor(random_state=42)
}

# 3. Treinamento e Avaliação
results = []
best_r2 = -np.inf
best_model = None
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Cálculo das Métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Modelo': name,
        'R² (R-squared)': r2,
        'MAE (Mean Absolute Error)': mae,
        'RMSE (Root Mean Squared Error)': rmse
    })
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results)

# 4. Análise Comparativa dos Resultados
print("Análise Comparativa:")
print(results_df.sort_values(by='R² (R-squared)', ascending=False).to_markdown(index=False))
print(f"\nO melhor modelo (baseado no R²) é: {best_model_name}")

# 5. Salvando o Melhor Modelo (Parte 4.1)
joblib.dump(best_model, MODEL_PATH)
print(f"Modelo '{best_model_name}' salvo em {MODEL_PATH}")

# 6. Demonstração de Uso (Parte 4.2)
loaded_model = joblib.load(MODEL_PATH)

# Pegando a primeira amostra do conjunto de teste para demonstração
sample_X = X_test.iloc[0].values.reshape(1, -1)
sample_y_true = y_test[0]
sample_y_pred = loaded_model.predict(sample_X)[0]

print(f"\nPrevisão de Demonstração (Modelo: {best_model_name}):")
print(f"  Valor Real: R$ {sample_y_true:,.2f}")
print(f"  Previsão do Modelo: R$ {sample_y_pred:,.2f}")