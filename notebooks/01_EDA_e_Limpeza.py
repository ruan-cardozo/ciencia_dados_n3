import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração para melhor visualização
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# Caminho para o dataset (relativo ao diretório de execução /home/ubuntu/ciencia_dados_n3)
DATA_PATH = "data/Housing.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {DATA_PATH}.")
    exit()

# 2. Visão Geral dos Dados
print(f"Formato do Dataset: {df.shape}")

# 4. Verificação de Valores Ausentes
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("Não há valores ausentes no dataset.")
else:
    print("Valores ausentes encontrados:")
    print(missing_values[missing_values > 0])

# 5. Análise da Variável Alvo (price)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribuição do Preço dos Imóveis')
plt.xlabel('Preço (R$)')
plt.ylabel('Contagem')
plt.savefig('notebooks/distribuicao_preco.png')
plt.close()

# 6. Análise de Correlação com a Variável Alvo
df_numeric = df.select_dtypes(include=np.number)
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.savefig('notebooks/matriz_correlacao.png')
plt.close()

# 7. Análise de Variáveis Categóricas
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=col, y='price', data=df)
    plt.title(f'Preço vs {col}')
    plt.savefig(f'notebooks/boxplot_{col}.png')
    plt.close()

print("EDA concluída. Resultados e gráficos salvos na pasta notebooks.")
