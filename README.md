# __Cluster_classification__
Este projeto realiza análise de clustering em um conjunto de dados de câncer de mama, utilizando algoritmos não supervisionados (KMeans e Gaussian Mixture Models) para segmentar os dados em grupos. Um modelo de linguagem (LLM) é então utilizado para interpretar esses clusters e classificá-los como malignos ou benignos.

##  __Integrantes do Grupo__

- **Heitor Saulo Dantas Santos**
- **Itor Carlos Souza Queiroz**
- **Lanna Luara Novaes Silva**
- **Lavínia Louise Rosa Santos**
- **Rômulo Menezes De Santana**

## Visão Geral do Projeto
O projeto implementa duas abordagens de clustering:

 * KMeans - Breast_cancer_classification_kmeans.py
 * Gaussian Mixture Model (GMM) - Breast_cancer_gmm_classification.py

Após o clustering, os dados são enviados para um modelo de linguagem (via API OpenAI) que analisa os grupos formados e os classifica como tumores malignos ou benignos, fornecendo justificativa para cada classificação.

## Funcionalidades
 * Visualização da matriz de correlação
 * Interpretação dos clusters formados utilizando LLM, especificamente os modelos GPT-4o e GPT-4o-mini
 * Avaliação da qualidade do clustering usando métricas como Adjusted Rand Index (ARI) e Normalized Mutual Information (NMI)
 * Pré-processamento de dados (codificação de variáveis categóricas, remoção de colunas altamente correlacionadas)
 * Visualização dos clusters formados

##  Configuração e execução do projeto
### Instalação e Configuração 

1. Clone o repositório

    ```sh
    git clone https://github.com/Heitorsaulo/Cluster_classification.git
    ```

2. Crie um ambiente virtual e instale as dependências:

    ```sh
    python -m venv .venv
    source .venv/bin/activate # No Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    
3. Configure sua chave da API OpenAI:
   ```sh
   export MY_OPENAI_KEY='your-openai-api-key'  # No Windows use `set MY_OPENAI_KEY=sua-chave-openai`
   ```

### Execução do algoritmo Kmeans
1. Execute o seguinte comando:
    ```sh
    python .\Breast_cancer_classification_kmeans.py 
    ```

### Execução do algoritmo GMM:
1. Execute o seguinte comando:
    ```sh
    python .\Breast_cancer_gmm_classification.py 
    ```
