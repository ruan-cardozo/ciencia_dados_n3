import kagglehub
import os

# O nome do dataset no Kaggle
DATASET_NAME = "yasserh/housing-prices-dataset"
# O diretório onde o arquivo será salvo
SAVE_DIR = "/home/ubuntu/ciencia_dados_n3/data"

# Cria o diretório se não existir
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Iniciando o download do dataset: {DATASET_NAME}")

# Download latest version
# O kagglehub baixa o arquivo para um cache local e retorna o caminho
path = kagglehub.dataset_download(DATASET_NAME)

print(f"Dataset baixado para: {path}")

# O arquivo CSV está dentro do diretório baixado. Vamos movê-lo para a pasta 'data' do projeto.
# O caminho retornado pelo kagglehub é o diretório raiz do dataset.
# O nome do arquivo é 'Housing.csv'
source_file = os.path.join(path, "Housing.csv")
target_file = os.path.join(SAVE_DIR, "Housing.csv")

if os.path.exists(source_file):
    os.rename(source_file, target_file)
    print(f"Arquivo 'Housing.csv' movido para: {target_file}")
else:
    print(f"Erro: Arquivo 'Housing.csv' não encontrado em {source_file}")

print("Download e organização de dados concluídos.")
