# Machine Learning — Classificação de Níveis de Obesidade

Projeto de Machine Learning desenvolvido como trabalho final da disciplina de
Aprendizado de Máquina da Pós-graduação em Ciência de Dados.

O objetivo do projeto é desenvolver um modelo de classificação multiclasse
capaz de prever o nível de obesidade (`NObeyesdad`) de indivíduos a partir
de características físicas, hábitos alimentares e estilo de vida.

## Objetivo

Construir e avaliar modelos de Machine Learning para classificação dos
indivíduos em sete categorias relacionadas ao nível de obesidade.

O projeto contempla:

- Análise exploratória dos dados (EDA)
- Verificação de valores ausentes e registros duplicados
- Análise de variáveis numéricas e categóricas
- Engenharia de atributos com criação do IMC
- Visualização dos dados
- Separação das bases de treinamento e validação
- Pré-processamento das variáveis categóricas
- Treinamento de Decision Tree
- Treinamento de Random Forest
- Avaliação utilizando métricas de classificação
- Comparação dos modelos
- Treinamento do modelo final
- Predição sobre a base de teste
- Geração do arquivo `submission.csv`

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pathlib

## Engenharia de atributos

Além das variáveis disponíveis originalmente no dataset, foi criada a
variável IMC (Índice de Massa Corporal):

IMC = Peso / Altura²

O novo atributo foi incorporado às bases de treinamento e teste e utilizado
pelos modelos de Machine Learning.

## Preparação dos dados

A variável alvo utilizada foi:

`NObeyesdad`

O identificador `id` foi removido das variáveis explicativas.

A base de treinamento foi dividida em:

- 80% para treinamento
- 20% para validação

Foi utilizado `random_state=42` para garantir a reprodutibilidade da divisão
e `stratify=y` para preservar aproximadamente a distribuição das classes.

Para as variáveis categóricas foi aplicado One-Hot Encoding através do
`OneHotEncoder`.

O pré-processamento e os modelos foram integrados utilizando `Pipeline`
e `ColumnTransformer` do Scikit-learn.

## Modelos avaliados

### Decision Tree

O primeiro modelo utilizado foi uma Árvore de Decisão
(`DecisionTreeClassifier`).

Resultados obtidos na validação:

| Métrica | Resultado |
|---|---:|
| Accuracy | 84,34% |
| Precision | 84,26% |
| Recall | 84,34% |
| F1-score | 84,29% |

### Random Forest

O segundo modelo utilizado foi o `RandomForestClassifier`, configurado
com 300 árvores.

Resultados obtidos na validação:

| Métrica | Resultado |
|---|---:|
| Accuracy | 90,58% |
| Precision | 90,56% |
| Recall | 90,58% |
| F1-score | 90,55% |

A Random Forest apresentou melhor desempenho na base de validação e,
por isso, foi utilizada como modelo final.

## Modelo final

Após a comparação dos modelos, a Random Forest foi novamente treinada
utilizando toda a base de treinamento disponível.

O modelo final foi então aplicado ao conjunto `test.csv` para realizar
as previsões da variável `NObeyesdad`.

Por fim, foi gerado o arquivo:

`submission.csv`

no formato esperado para submissão.

## Estrutura do projeto

    obesity-classification-machine-learning/
    │
    ├── aprendizado_maquina.py
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    │
    ├── graficos/
    │   └── gráficos gerados durante a análise exploratória
    │
    └── docs/
        └── Trabalho-Machine-Learning-outubro-2026.pdf

## Apresentação do projeto

A apresentação completa do trabalho, contendo a análise exploratória,
metodologia, comparação dos modelos e resultados obtidos, está disponível em:

[📄 Visualizar apresentação do projeto](docs/Trabalho-Machine-Learning-outubro-2026.pdf)

## Execução

Após instalar as dependências do projeto:

    pip install -r requirements.txt

Execute:

    python aprendizado_maquina.py

Os arquivos `train.csv`, `test.csv` e `sample_submission.csv` são
necessários para executar todas as etapas do projeto.

## Principais conceitos aplicados

- Machine Learning supervisionado
- Classificação multiclasse
- Análise exploratória de dados
- Feature Engineering
- One-Hot Encoding
- Train/Test Split
- Estratificação
- Pipeline
- Decision Tree
- Random Forest
- Accuracy
- Precision
- Recall
- F1-score

## Autor

Carlos Ferreira

Projeto desenvolvido durante a Pós-graduação em Ciência de Dados.