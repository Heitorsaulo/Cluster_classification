import json
import pandas as pd
import numpy as np
import os
import openai
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
 
df = pd.read_csv(f'datasets/breast-cancer-dataset/breast-cancer.csv')
df.head()
 
df.describe().T.round(2)
 
label_encoder = LabelEncoder()
# Transformar todas as colunas não numéricas em labels
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])
    label_legend = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_legend)
 
df.head()
 
df.drop(columns=['id'], inplace=True)
 
def remover_colunas_correlacionadas(df):
    matrix_corr = df.corr().abs()
    remove_columns = set()

    for i_corr in range(len(matrix_corr.columns)):
        for j_corr in range(i_corr + 1, len(matrix_corr.columns)):
            if matrix_corr.iloc[i_corr, j_corr] > 0.87 and matrix_corr.columns[j_corr] != 'close':
                remove_columns.add(matrix_corr.columns[j_corr])
                print(f"Removendo coluna: {matrix_corr.columns[j_corr]} com correlação: {matrix_corr.iloc[i_corr, j_corr]}")

    return df.drop(columns = remove_columns)
 
matrix_corr = df.corr().abs()

# Configurar o tamanho da figura
plt.figure(figsize=(12, 8))

# Criar o heatmap usando matplotlib
plt.imshow(matrix_corr, cmap='coolwarm', interpolation='none')
plt.colorbar()

# Adicionar rótulos
plt.xticks(range(len(matrix_corr.columns)), matrix_corr.columns, rotation=90)
plt.yticks(range(len(matrix_corr.columns)), matrix_corr.columns)

# Adicionar título
plt.title('Matriz de Correlação')

# Exibir o gráfico
plt.show()
 
df_filtrado = remover_colunas_correlacionadas(df)
df_filtrado.head()
 
data_train = df_filtrado.drop(columns=['diagnosis'])
data_train_np = np.array(data_train)
## Treinando o modelo de classificação
 
#modelo_unsupervised = KMeans(n_clusters=4, random_state=42)
#modelo_unsupervised.fit(data_train)
 
gmm_model = GaussianMixture(n_components=2, random_state=42)
gmm_model.fit(data_train_np)
 
cluster_df = data_train

cluster_df['Cluster'] = gmm_model.predict(data_train_np) #modelo_unsupervised.labels_

cluster_0 = cluster_df[cluster_df['Cluster'] == 0]
cluster_1 = cluster_df[cluster_df['Cluster'] == 1]
 
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_train['radius_mean'], data_train['smoothness_mean'], c=cluster_df['Cluster'], cmap='viridis')

plt.xlabel('Radius Mean')
plt.ylabel('Smoothness Mean')
plt.title('GMM Clustering of breast cancer Data')

legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['radius_mean'], df['smoothness_mean'], c=df['diagnosis'], cmap='cividis')

plt.xlabel('Radius Mean')
plt.ylabel('Smoothness Mean')
plt.title('Original breast cancer Data')

legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()



cluster_0_json = cluster_0.to_json(orient='records')
cluster_1_json = cluster_1.to_json(orient='records')

client = openai.OpenAI(
    organization='org-8Q6LduybHrpbAdbT9yJUJDQA',
    project='proj_DjkXbEEukpsg44dLeabG1ZyP',
    api_key=os.environ.get('MY_OPENAI_KEY')
)

def generateResponseLLM(model: str, messages: list , max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# Configurar a chave da OpenAI (pode ser via variável de ambiente)
OPENAI_API_KEY = os.getenv("MY_OPENAI_KEY")
 
prompt = f"""
A seguir, apresento a lista das possíveis classificações:

Maligno

Benigno

Além disso, para cada grupo, disponibilizarei os dados no formato JSON conforme o exemplo abaixo:

Grupo 1: {cluster_0_json}

Grupo 2: {cluster_1_json}

O contexto desses dados são para a classificação de câncer de mama, onde cada grupo representa um conjunto de características de pacientes que fizeram o exame para investigar um possível cancer.

Sua tarefa é analisar as características presentes em cada JSON (dados de cada grupo) e, com base nessas informações e na lista de classificações fornecida, atribuir a cada grupo a classificação que melhor o representa, certifique-se de considerar todas as características presentes em cada json, e classificar todos os grupos fornecidos. Para cada grupo, por favor, inclua:

A classificação escolhida.

Uma breve justificativa explicando a relação entre as características do grupo e a classificação atribuída.

Por favor, apresente os resultados da seguinte forma no formato JSON para eu converter diretamente a resposta para um arquivo JSON em python:

    "classificação":
        "cluster_0": label_0,
        "cluster_1": label_1,
        ...

    "justificativa":
        "cluster_0": "justificativa_0",
        "cluster_1": "justificativa_1",
        ...


"""
 
messages = [
    {
        "role": "system",
        "content": "Você é um especialista em análise de dados e machine learning. Recebi um conjunto de dados que foi segmentado em grupos utilizando um algoritmo de clustering não supervisionado. Cada grupo contém observações com características semelhantes, mas ainda não sabemos a que classificação cada um se encaixa."
    },
    {
        "role": "user",
        "content": prompt
    }
]
 
resposta_llm = generateResponseLLM('gpt-4o-mini', messages, 5600)
print(resposta_llm)


#Automatizar a conversão da resposta para JSON, está assim, pois fui testando manualmente no notebook
resposta_llm_json = {
    "classificação": {
        "cluster_0": "M",
        "cluster_1": "B"
    },
    "justificativa": {
        "cluster_0": "O grupo 0 apresenta características típicas de tumores malignos, como altos valores em 'radius_mean' (média de raio), 'compactness_mean' (compactação média), 'concavity_mean' (concavidade média) e outros parâmetros que estão geralmente associados a um crescimento tumor maligno. Os dados mostram valores elevados em métricas como 'concave points_mean', que podem indicar a presença de formações mais agressivas.",
        "cluster_1": "O grupo 1, por outro lado, demonstra características associadas a tumores benignos, como valores mais baixos em 'radius_mean', 'smoothness_mean' e 'texture_mean', que estão relacionados a uma estrutura mais homogênea e menos agressiva. As características deste grupo indicam menor risco e potencial agressividade, alinhando-se com zero ou uma baixa presença de câncer."
    }
}

cluster_classification_llm = []
for i in range(2):
    cluster_classification_llm.append(resposta_llm_json['classificação'][f'cluster_{i}'])



# Comparar os resultados e verificar a precisão dos clusters
df = remover_colunas_correlacionadas(df)
groups_cluster = []
groups = df.groupby('diagnosis')
for category, group in groups:
    print(f"Category: {category}")
    groups_cluster.append(group)
 
print(groups_cluster[0])
print(groups_cluster[1])

for index in range(1, len(groups_cluster)):
    groups_cluster[index - 1] = groups_cluster[index - 1].drop(columns=['diagnosis'])
    groups_cluster[index - 1].head()
 
Classification_labels_adjusted = cluster_df['Cluster'].astype(object).copy()

for i in range(len(cluster_df['Cluster'])):
    Classification_labels_adjusted[i] = cluster_classification_llm[cluster_df['Cluster'][i]]

label_encoder = LabelEncoder()

Classification_labels_adjusted = label_encoder.fit_transform(Classification_labels_adjusted)
label_legend_cluster = {index: label for index, label in enumerate(label_encoder.classes_)}
print(f'legenda das labels dos clusters: {label_legend_cluster}')
print(f'legenda das labels do dataset: {label_legend}')
 
print(Classification_labels_adjusted)
print(df['diagnosis'].values)
 
ari_score = adjusted_rand_score(df['diagnosis'].values, Classification_labels_adjusted)
print(f"Adjusted Rand Index (ARI): {ari_score:.2f}")

nmi_score = normalized_mutual_info_score(df['diagnosis'].values, Classification_labels_adjusted)
print(f"Normalized Mutual Information (NMI): {nmi_score:.2f}")
 
def calculate_similarity_percentage(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    matching_elements = sum(1 for a, b in zip(list1, list2) if a == b)

    similarity_percentage = (matching_elements / len(list1)) * 100

    return similarity_percentage

brute_similarity_percentage = calculate_similarity_percentage(df['diagnosis'].values, Classification_labels_adjusted)

print(f'comparação de elemento iguais na classificação de clusters e valores do dataset: {brute_similarity_percentage:.2f}%')