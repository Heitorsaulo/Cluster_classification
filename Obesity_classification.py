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



df = pd.read_csv(f'datasets/Obesity Classification.csv')
df.head()

labels = df['Label']

train_df = df.drop(columns=['Label', 'ID'])

label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)
label_legend = {index: label for index, label in enumerate(label_encoder.classes_)}
df['Numeric_Label'] = numeric_labels

numeric_gender = label_encoder.fit_transform(df['Gender'])
gender_legend = {index: label for index, label in enumerate(label_encoder.classes_)}
train_df['Gender'] = numeric_gender
print(f'labels legend: {label_legend} || numeric labels: {numeric_labels} || gender legend: {gender_legend} || numeric gender : {numeric_gender}')
## Treinando o modelo de classificação
data_train = np.array(train_df)
classes = np.array(numeric_labels)

#Treinar o modelo de classificação
#modelo_unsupervised = KMeans(n_clusters=4, random_state=42)
#modelo_unsupervised.fit(data_train)
gmm_model = GaussianMixture(n_components=4, random_state=42)
gmm_model.fit(data_train)

cluster_df = train_df

cluster_df['Cluster'] =  gmm_model.predict(data_train) #modelo_unsupervised.labels_

gender_mapping = {0: 'Female', 1: 'Male'}

cluster_df['Gender'] = cluster_df['Gender'].map(gender_mapping)

cluster_0 = cluster_df[cluster_df['Cluster'] == 0]
cluster_1 = cluster_df[cluster_df['Cluster'] == 1]
cluster_2 = cluster_df[cluster_df['Cluster'] == 2]
cluster_3 = cluster_df[cluster_df['Cluster'] == 3]
plt.figure(figsize=(10, 6))
scatter = plt.scatter(train_df['Weight'], train_df['Height'], c=cluster_df['Cluster'], cmap='viridis')

plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('GMM Clustering of Obesity Data')

legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()
cluster_0_json = cluster_0.to_json(orient='records')
cluster_1_json = cluster_1.to_json(orient='records')
cluster_2_json = cluster_2.to_json(orient='records')
cluster_3_json = cluster_3.to_json(orient='records')

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

{label_legend[0]}

{label_legend[1]}

{label_legend[2]}

{label_legend[3]}

Além disso, para cada grupo, disponibilizarei os dados no formato JSON conforme o exemplo abaixo:

Grupo 1: {cluster_0_json}

Grupo 2: {cluster_1_json}

Grupo 3: {cluster_2_json}

Grupo 4: {cluster_3_json}

Sua tarefa é analisar as características presentes em cada JSON (dados de cada grupo) e, com base nessas informações e na lista de classificações fornecida, atribuir a cada grupo a classificação que melhor o representa, certifique-se de considerar todas as características presentes em cada json, e classificar todos os grupos fornecidos. Para cada grupo, por favor, inclua:

A classificação escolhida.

Uma breve justificativa explicando a relação entre as características do grupo e a classificação atribuída.

Por favor, apresente os resultados da seguinte forma no formato JSON, sem detalhes adicionais, somente o formato json na resposta para eu converter diretamente a resposta para um arquivo JSON:

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
resposta_llm = generateResponseLLM('gpt-4o', messages, 5600)

resposta_llm_json = json.loads(resposta_llm)

cluster_classification_llm = []
for i in range(4):
    cluster_classification_llm.append(resposta_llm_json['classificação'][f'cluster_{i}'])

groups_cluster = []
groups = df.groupby('Label')
for category, group in groups:
    print(f"Category: {category}")
    groups_cluster.append(group)

for index in range(1, len(groups_cluster)):
    groups_cluster[index - 1] = groups_cluster[index - 1].drop(columns=['ID', 'Label', 'Numeric_Label'])
    groups_cluster[index - 1].head()

def row_similarity(row1, row2):
    return 1 if row1.equals(row2) else 0

def compare_dataframes(df1, df2):
    # Encontre as colunas comuns
    common_columns = df1.columns.intersection(df2.columns)

    # Filtre os DataFrames para manter apenas as colunas comuns
    df1_common = df1[common_columns]
    df2_common = df2[common_columns]

    # Calcule a similaridade para cada par de linhas
    similarities = []
    for i in range(min(len(df1_common), len(df2_common))):
        row1 = df1_common.iloc[i]
        row2 = df2_common.iloc[i]
        similarity = row_similarity(row1, row2)
        similarities.append(similarity)

    # Retorne a média das similaridades
    return np.mean(similarities) * 100

sim1 = compare_dataframes(cluster_0, groups_cluster[0])
sim2 = compare_dataframes(cluster_1, groups_cluster[1])
sim3 = compare_dataframes(cluster_2, groups_cluster[2])
sim4 = compare_dataframes(cluster_3, groups_cluster[3])

print(sim1)
print(sim2)
print(sim3)
# Calcula a média das similaridades
similarity_average = np.mean([sim1, sim2, sim3, sim4])
print(f"Porcentagem de semelhança: {similarity_average:.2f}%")