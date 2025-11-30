import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Caminho para o dataset (relativo ao diretório de execução /home/ubuntu/ciencia_dados_n3)
DATA_PATH = "data/Housing.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {DATA_PATH}.")
    exit()

# 2. Codificação de Variáveis Categóricas
categorical_cols = df.select_dtypes(include='object').columns
# Usando drop_first=True para evitar a armadilha da variável dummy
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Separação de Variáveis (X e y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 4. Normalização/Escalonamento de Variáveis Numéricas
# Escalonando apenas as features numéricas
numeric_cols = X.select_dtypes(include=np.number).columns
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 5. Divisão em Conjuntos de Treino e Teste
# Usando random_state=42 para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Salvando os dados pré-processados para a próxima etapa
# Salvando em formato npz para preservar os arrays numpy
np.savez_compressed('data/processed_data.npz', 
                     X_train=X_train.values, X_test=X_test.values, 
                     y_train=y_train.values, y_test=y_test.values, 
                     feature_names=X_train.columns.values)

print("Pré-processamento concluído. Dados de treino e teste salvos em data/processed_data.npz")
