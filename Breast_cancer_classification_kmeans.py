import json
import pandas as pd
import numpy as np
import os
import openai
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv(f'datasets/breast-cancer-dataset/breast-cancer.csv')
df.describe().T.round(2)
label_encoder = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])
    label_legend = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_legend)


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

plt.figure(figsize=(12, 8))

plt.imshow(matrix_corr, cmap='coolwarm', interpolation='none')
plt.colorbar()

plt.xticks(range(len(matrix_corr.columns)), matrix_corr.columns, rotation=90)
plt.yticks(range(len(matrix_corr.columns)), matrix_corr.columns)

plt.title('Matriz de Correlação')

plt.show()
df_filtrado = remover_colunas_correlacionadas(df)
df_filtrado.head()
data_train = df_filtrado.drop(columns=['diagnosis'])
data_train_np = np.array(data_train)
kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(data_train_np)

cluster_df = data_train.copy()

cluster_df['Cluster'] = kmeans_model.predict(data_train)

cluster_0 = cluster_df[cluster_df['Cluster'] == 0]
cluster_1 = cluster_df[cluster_df['Cluster'] == 1]
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_train['radius_mean'], data_train['smoothness_mean'], c=cluster_df['Cluster'], cmap='viridis')

plt.xlabel('Radius Mean')
plt.ylabel('Smoothness Mean')
plt.title('KMeans Clustering of breast cancer Data')

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
#%% md
# # Resposta do LLM por modelo
# 
# ## modelo gpt-4o-mini
# ```json
# {
#     "classificação": {
#         "cluster_0": "Benigno",
#         "cluster_1": "Maligno"
#     },
#     "justificativa": {
#         "cluster_0": "Os valores de 'radius_mean', 'smoothness_mean', e 'compactness_mean' são relativamente menores e com variações menores dentro do grupo, o que sugere menos agressividade e características típicas de tumores benignos.",
#         "cluster_1": "As características desse grupo apresentam valores de 'radius_mean', 'compactness_mean', e 'concavity_mean' significativamente maiores, indicando que as células têm um padrão mais agressivo e atributos típicos de tumores malignos."
#     }
# }
# ```
# ## modelo gpt-4o
# 
resposta_llm_json = {
    "classificação": {
        "cluster_0": "Benigno",
        "cluster_1": "Maligno"
    },
    "justificativa": {
        "cluster_0": "O grupo 0 apresenta características como 'radius_mean' e 'smoothness_mean' relativamente altos, além de outros atributos que indicam uma tendência para características benignas. Os valores de 'concavity' e 'concave points' estão dentro de faixas que frequentemente indicam que as amostras são menos agressivas, o que é típico de tumores benignos.",
        "cluster_1": "O grupo 1, por outro lado, possui valores de 'texture_mean' e 'compactness_mean' mais elevados, além de um 'radius_mean' geralmente menor, formando padrões associados a características malignas. Os altos valores de 'concavity' e 'concave points', juntamente com a média de textura, sugerem a presença de tumores mais agressivos e malignos."
    }
}

cluster_classification_llm = []
for i in range(2):
    cluster_classification_llm.append(resposta_llm_json['classificação'][f'cluster_{i}'])
df = remover_colunas_correlacionadas(df)
groups_cluster = []
groups = df.groupby('diagnosis')

for category, group in groups:
    print(f"Category: {category}")
    groups_cluster.append(group)

for index in range(1, len(groups_cluster)):
    groups_cluster[index - 1] = groups_cluster[index - 1].drop(columns=['diagnosis'])
    groups_cluster[index - 1].head()

Classification_labels_adjusted = cluster_df['Cluster'].astype(object).copy()

for i in range(len(cluster_df['Cluster'])):
    Classification_labels_adjusted[i] = cluster_classification_llm[cluster_df['Cluster'][i]]

ari_score = adjusted_rand_score(df['diagnosis'], Classification_labels_adjusted)
print(f"Adjusted Rand Index (ARI): {100*ari_score:.2f}%")

nmi_score = normalized_mutual_info_score(df['diagnosis'], Classification_labels_adjusted)
print(f"Normalized Mutual Information (NMI): {100*nmi_score:.2f}%")
#%% md
# def row_similarity(row1, row2):
#     # Retorna 1 se as linhas forem iguais, caso contrário 0
#     return 1 if row1.equals(row2) else 0
# 
# def remove_tested_lines(df1, df2, row1, row2):
#     # Remove as linhas dos dataframes utilizando os índices atuais (após reset)
#     if row1.name in df1.index and row2.name in df2.index:
#         df1 = df1.drop(index=row1.name).reset_index(drop=True)
#         df2 = df2.drop(index=row2.name).reset_index(drop=True)
#     return df1, df2
# 
# def compare_dataframes(df1, df2):
#     similarities = []
#     # Reinicia os índices para garantir consistência
#     group_1 = df1.copy().reset_index(drop=True)
#     group_2 = df2.copy().reset_index(drop=True)
# 
#     # Enquanto houver linhas em group_1
#     while not group_1.empty:
#         row1 = group_1.iloc[0]  # Pega sempre a primeira linha (índice atual)
#         found_match = False
# 
#         # Procura linha equivalente em group_2
#         for j in range(len(group_2)):
#             row2 = group_2.iloc[j]
#             if row_similarity(row1, row2) == 1:
#                 similarities.append(1)
#                 group_1, group_2 = remove_tested_lines(group_1, group_2, row1, row2)
#                 found_match = True
#                 break
# 
#         # Se não encontrar correspondência, marca como 0 e remove a linha atual de group_1
#         if not found_match:
#             similarities.append(0)
#             group_1 = group_1.drop(index=row1.name).reset_index(drop=True)
# 
#     return np.mean(similarities) * 100

