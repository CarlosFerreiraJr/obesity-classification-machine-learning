# ============================================================================================#
# TRABALHO FINAL - APRENDIZADO DE MÁQUINA                                                     #
# Pós-graduação em Ciência de Dados                                                           #
# Professor: Pedro Igor                                                                       #
# Problema:  Previsão do nível de risco de obesidade                                          #
# Classificação: Multiclasse                                                                  # 
# =============================================================================================


##############################################################################################
#                                    Importação das bibliotecas                              #
#  No terminal do VS Code, execute os comandos abaixo:                                       #
#     C:\python314\python.exe -m pip install pandas numpy matplotlib scikit-learn            #
#                                                                                            #
#  Observações:                                                                              #
#    1. Eu passei o diretório do Python 3.14 para instalação dos pacotes, pois eu uso o      #
#       Python 3.12 no Databricks.                                                           #
#    2. Estou usando o pandas para leitura do csv ao invés do PySpark, pois o dataset é      #
#       pequeno e não há necessidade de usar o PySpark.                                      #
#    3. A biblioteca numpy permitir o processamento rápido e eficiente de grandes volumes de #
#       dados numéricos e cálculos matemáticos complexos                                     #
#    4. A biblioteca matplotlib é usada para criar visualizações de dados, como gráficos e   # 
#       estatísticas descritivas.                                                            #
#    5. A biblioteca pathlib é usada para manipulação de caminhos de arquivos e diretórios,  #
#       tornando o código mais legível e portátil.                                           #
# #############################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configurações do Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ============================================================
# 1. DEFINIÇÃO DOS DIRETÓRIOS
#    O r antes da string significa raw string e evita que o 
#    Python interprete o caractere "\" como caractere especial.
# ============================================================

strDir = Path(
    r"C:\02. Unicarioca - Pós graduação em Ciência de Dados -2026\03. Machine Learning\Trabalho Final"
)

# Diretório onde os gráficos serão armazenados
dirGraficos = strDir / "graficos"

# Cria a pasta caso ela ainda não exista
dirGraficos.mkdir(exist_ok=True)

print("\nDiretório do projeto:")
print(strDir)

print("\nDiretório dos gráficos:")
print(dirGraficos)


# ============================================================
# 2. CARREGAMENTO DOS DATASETS
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("1 - CARREGAMENTO DOS DATASETS")
print("=" * 70)

# Leitura dos arquivos CSV usando o pandas
train = pd.read_csv(strDir / "train.csv")
test = pd.read_csv(strDir / "test.csv")
sample_submission = pd.read_csv(strDir / "sample_submission.csv")

print("\nDatasets carregados com sucesso!")
print("\nDimensões:")
print("Conjunto de dados (dataset) de Treinamento:", train.shape)
print("Conjunto de dados (dataset) de Teste:", test.shape)
print("Conjunto de dados (dataset) de Envio de Amostra:", sample_submission.shape)


# ============================================================
# 3. VISUALIZAÇÃO INICIAL
# ============================================================

# Leitura dos arquivos CSV usando o pandas
print("\n" + "=" * 70)
print("3 - VISÃO GERAL DO DATASET")
print("=" * 70)

print("\nPrimeiros registros do dataset de treinamento:\n")

# Observação: O método head() do pandas exibe as primeiras linhas do DataFrame, 
# permitindo uma visualização rápida dos dados.
# Esta função foi ensinada pela professora de Python.
print(train.head())


# ============================================================
# 4. INFORMAÇÕES SOBRE O DATASET
#    Trazer uma visão geral do dataset
# ============================================================

# Leitura dos arquivos CSV usando o pandas
print("\n" + "=" * 70)
print("4 - INFORMAÇÕES DO DATASET (Visão geral do dataset e Distribuição dos dados)")
print("=" * 70)

# Imprime uma linha em branco para separar a saída do código
print()

# Exibe informações sobre a estrutura do dataset, 
# incluindo colunas, tipos de dados e valores não nulos
train.info()

print("\nQuantidade de linhas :", train.shape[0])
print("Quantidade de colunas:", train.shape[1])

print("\nColunas disponíveis:")

for coluna in train.columns:
    print("-", coluna)

# ============================================================
# 5. VALORES AUSENTES
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("5 - VERIFICAÇÃO DE VALORES AUSENTES (Visão geral do dataset e Distribuição dos dados)")
print("=" * 70)

valores_nulos = train.isnull().sum()

print("\nValores nulos por coluna:\n")
print(valores_nulos)

print("\nTotal de valores nulos:", valores_nulos.sum())

# ============================================================
# 6. REGISTROS DUPLICADOS
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("6 - VERIFICAÇÃO DE DUPLICIDADES (Visão geral do dataset e Distribuição dos dados)")
print("=" * 70)

duplicados = train.duplicated().sum()

print("\nQuantidade de registros duplicados:", duplicados)


# ============================================================
# 7. DEFINIÇÃO DA VARIÁVEL ALVO
# ============================================================

# Retorna somente os valores diferentes existentes nessa coluna. 
# O dataset possui 7 categorias de obesidade, sendo assim teremos 7 classes

target = "NObeyesdad"

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("7 - VARIÁVEL ALVO")
print("=" * 70)

print("\nVariável alvo:", target)
print("Quantidade de classes:", train[target].nunique())
print("\nClasses encontradas:")

# Obtém as classes distintas da variável alvo,
# ordena alfabeticamente e exibe cada classe no terminal.
for classe in sorted(train[target].unique()):
    print("-", classe)

# ============================================================
# 8. CRIAÇÃO DA VARIÁVEL IMC
# ============================================================

# O IMC (Índice de Massa Corporal) é calculado dividindo
# o peso em quilogramas pela altura em metros ao quadrado.
#
# Fórmula:
# IMC = Peso / Altura²

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("8 - CRIAÇÃO DA VARIÁVEL IMC")
print("=" * 70)

train["IMC"] = train["Weight"] / (train["Height"] ** 2)
test["IMC"] = test["Weight"] / (test["Height"] ** 2)

print("\nVariável IMC criada com sucesso!")

print("\nExemplo dos valores calculados:")
print(
    train[
        ["Height", "Weight", "IMC"]
    ].head()
)

# ============================================================
# 9. VARIÁVEIS NUMÉRICAS E CATEGÓRICAS
#    Apresentar quais são as variáveis categóricas e numéricas
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("9 - TIPOS DE VARIÁVEIS")
print("=" * 70)

# Seleciona as colunas numéricas do dataset
# (inteiros e decimais) e armazena seus nomes em uma lista.
numericas = (
    train
    .select_dtypes(include=["int64", "float64"])
    .columns
    .tolist()
)

# Seleciona as colunas categóricas (texto/string)
# e armazena os nomes das colunas em uma lista
categoricas = (
    train
    .select_dtypes(include=["str"])
    .columns
    .tolist()
)

# ID não será considerado atributo explicativo
if "id" in numericas:
    numericas.remove("id")

# A variável alvo também é categórica,
# mas não deve ser considerada variável de entrada.
if target in categoricas:
    categoricas.remove(target)

print("\nVariáveis numéricas:")
for coluna in numericas:
    print("-", coluna)

print("\nVariáveis categóricas:")
for coluna in categoricas:
    print("-", coluna)

print("\nQuantidade de variáveis numéricas:", len(numericas))
print("Quantidade de variáveis categóricas:", len(categoricas))

# ============================================================
# 10. ANÁLISE DESCRITIVA - VARIÁVEIS NUMÉRICAS
#     Trazer uma análise descritiva sobre os atributos
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("10 - ANÁLISE DESCRITIVA DAS VARIÁVEIS NUMÉRICAS")
print("=" * 70)

descricao_numericas = (
    train[numericas]
    .describe()
    .T
)

print("\n")
print(descricao_numericas)


# ============================================================
# 11. ANÁLISE DESCRITIVA - VARIÁVEIS CATEGÓRICAS
#     Trazer uma análise descritiva sobre os atributos
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("11 - ANÁLISE DESCRITIVA DAS VARIÁVEIS CATEGÓRICAS")
print("=" * 70)

descricao_categoricas = (
    train[categoricas]
    .describe()
    .T
)

print("\n")
print(descricao_categoricas)


# ============================================================
# 12. DISTRIBUIÇÃO DA VARIÁVEL ALVO
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("12 - DISTRIBUIÇÃO DAS CLASSES")
print("=" * 70)

distribuicao = train[target].value_counts()

print("\nQuantidade por classe:\n")
print(distribuicao)


percentual = (
    train[target]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPercentual por classe:\n")
print(percentual)


# ============================================================
# 13. GRÁFICO - DISTRIBUIÇÃO DAS CLASSES
#     Plotar gráficos que auxiliem na análise dos dados
# ============================================================

print("\nGerando gráfico de distribuição das classes...")

plt.figure(figsize=(12, 6))

ax = train[target].value_counts().plot(
    kind="bar"
)

# Exibe o valor sobre cada barra
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%d",
        padding=3
    )

plt.title("Distribuição das Classes de Obesidade")
plt.xlabel("Nível de Obesidade")
plt.ylabel("Quantidade de Registros")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    dirGraficos / "01_distribuicao_classes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# 14. HISTOGRAMAS DAS VARIÁVEIS NUMÉRICAS
#     Gera um gráfico individual para cada variável numérica
# ============================================================

print("Gerando histogramas individuais das variáveis numéricas...")

# Tradução e descrição das variáveis numéricas
descricao_variaveis = {
    "Age": {
        "titulo": "Idade (Age)",
        "eixo_x": "Idade em anos",
        "descricao": "Idade do indivíduo"
    },

    "Height": {
        "titulo": "Altura (Height)",
        "eixo_x": "Altura em metros",
        "descricao": "Altura do indivíduo"
    },

    "Weight": {
        "titulo": "Peso (Weight)",
        "eixo_x": "Peso em quilogramas",
        "descricao": "Peso corporal do indivíduo"
    },

    "FCVC": {
        "titulo": "Consumo de vegetais (FCVC)",
        "eixo_x": "Frequência de consumo",
        "descricao": "Frequência de consumo de vegetais"
    },

    "NCP": {
        "titulo": "Número de refeições principais (NCP)",
        "eixo_x": "Número de refeições",
        "descricao": "Número de refeições principais realizadas por dia"
    },

    "CH2O": {
        "titulo": "Consumo de água (CH2O)",
        "eixo_x": "Consumo de água",
        "descricao": "Consumo diário de água"
    },

    "FAF": {
        "titulo": "Atividade física (FAF)",
        "eixo_x": "Frequência de atividade física",
        "descricao": "Frequência de realização de atividade física"
    },

    "TUE": {
        "titulo": "Uso de dispositivos tecnológicos (TUE)",
        "eixo_x": "Tempo de utilização",
        "descricao": "Tempo de utilização de dispositivos tecnológicos"
    },

    "IMC": {
        "titulo": "Índice de Massa Corporal (IMC)",
        "eixo_x": "IMC (kg/m²)",
        "descricao": "Índice calculado a partir do peso e da altura"
    }
}

# Percorre a lista de variáveis numéricas.
# O enumerate() retorna o número da posição e o nome da coluna.
# start=1 faz com que a numeração comece em 1 em vez de 0.
# O índice será utilizado para numerar os arquivos dos gráficos.
for indice, coluna in enumerate(numericas, start=1):

    plt.figure(figsize=(10, 6))

    plt.hist(
        train[coluna],
        bins=30
    )

    plt.title(
        f"Distribuição - {descricao_variaveis[coluna]['titulo']}"
    )

    plt.xlabel(
        descricao_variaveis[coluna]["eixo_x"]
    )

    plt.ylabel(
        "Quantidade de indivíduos"
    )

    # Texto explicativo na parte inferior do gráfico
    plt.figtext(
        0.5,
        0.01,
        f"{coluna}: {descricao_variaveis[coluna]['descricao']}",
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )

    plt.savefig(
        dirGraficos /
        f"02_{indice:02d}_{coluna.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 15. GRÁFICO - PESO MÉDIO POR NÍVEL DE OBESIDADE
#
#     Calcula o peso médio dos indivíduos de cada classe
#     de obesidade e apresenta os resultados em um gráfico
#     de barras.
# ============================================================

print("Gerando gráfico de peso médio por nível de obesidade...")


# Tradução das classes da variável alvo
traducao_classes = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III"
}


# Define a ordem das classes no gráfico
ordem_classes = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III"
]


# Calcula o peso médio de cada classe
peso_medio = (
    train
    .groupby(target)["Weight"]
    .mean()
    .reindex(ordem_classes)
)


# Substitui os nomes originais pelas traduções
peso_medio.index = [
    traducao_classes[classe]
    for classe in peso_medio.index
]


# Criação do gráfico
plt.figure(figsize=(14, 7))

ax = peso_medio.plot(
    kind="bar"
)


# Exibe o peso médio sobre cada barra
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f kg",
        padding=3
    )


# Título e identificação dos eixos
plt.title(
    "Peso Médio por Nível de Obesidade"
)

plt.xlabel(
    "Classificação do Nível de Obesidade"
)

plt.ylabel(
    "Peso Médio (kg)"
)


# Ajuste dos nomes das classes
plt.xticks(
    rotation=45,
    ha="right"
)


# Texto explicativo
plt.figtext(
    0.5,
    0.01,
    "Weight (Peso): peso corporal médio dos indivíduos em cada classe de obesidade.",
    ha="center",
    fontsize=10
)


plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)


# Salva o gráfico
plt.savefig(
    dirGraficos / "03_peso_medio_por_nivel_obesidade.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
plt.close()

# ============================================================
# 16. GRÁFICO - DISTRIBUIÇÃO DO IMC
# ============================================================

print("Gerando gráfico de distribuição do IMC...")

faixas_imc = pd.cut(
    train["IMC"],
    bins=[0, 18.5, 25, 30, 35, 40, float("inf")],
    labels=[
        "Abaixo de 18,5",
        "18,5 a 24,9",
        "25,0 a 29,9",
        "30,0 a 34,9",
        "35,0 a 39,9",
        "40 ou mais"
    ],
    right=False
)

distribuicao_imc = (
    faixas_imc
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))

ax = distribuicao_imc.plot(
    kind="bar"
)

# Exibe o valor sobre cada barra
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%d",
        padding=3
    )

plt.title("Distribuição dos Indivíduos por Faixa de IMC")
plt.xlabel("Faixa de IMC")
plt.ylabel("Quantidade de Indivíduos")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    dirGraficos / "05_distribuicao_imc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 17. GRÁFICO - IMC MÉDIO POR NÍVEL DE OBESIDADE
#
#     Calcula o IMC médio dos indivíduos de cada classe
#     de obesidade.
# ============================================================

print("Gerando gráfico de IMC médio por nível de obesidade...")


# Calcula o IMC médio de cada classe
imc_medio = (
    train
    .groupby(target)["IMC"]
    .mean()
    .reindex(ordem_classes)
)


# Substitui os nomes originais pelas traduções
imc_medio.index = [
    traducao_classes[classe]
    for classe in imc_medio.index
]


# Criação do gráfico
plt.figure(figsize=(14, 7))

ax = imc_medio.plot(
    kind="bar"
)


# Exibe o IMC médio sobre cada barra
for container in ax.containers:

    ax.bar_label(
        container,
        fmt="%.1f",
        padding=3
    )


# Título e identificação dos eixos
plt.title(
    "IMC Médio por Nível de Obesidade"
)

plt.xlabel(
    "Classificação do Nível de Obesidade"
)

plt.ylabel(
    "IMC Médio (kg/m²)"
)


plt.xticks(
    rotation=45,
    ha="right"
)


# Texto explicativo
plt.figtext(
    0.5,
    0.01,
    "IMC: Índice de Massa Corporal calculado a partir do peso e da altura.",
    ha="center",
    fontsize=10
)


plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)


# Salva o gráfico
plt.savefig(
    dirGraficos / "04_imc_medio_por_nivel_obesidade.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
plt.close()


# ============================================================
# 18. FINALIZAÇÃO DA ANÁLISE EXPLORATÓRIA
#  
#  Observação: Os gráficos gerados estão salvos na pasta 
#              "graficos" dentro do diretório do projeto para
#              que seja possível visualizá-los posteriormente.
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("ANÁLISE EXPLORATÓRIA CONCLUÍDA")
print("=" * 70)

print("\nGráficos salvos em:")
print(dirGraficos)

print("\nArquivos gerados:")

for arquivo in sorted(dirGraficos.glob("*.png")):
    print("-", arquivo.name)

print("\nA análise exploratória foi finalizada com sucesso!")

# ============================================================
# 19. PREPARAÇÃO DOS DADOS PARA MACHINE LEARNING
# ============================================================

# Gera uma linha com 70 caracteres "=" para separar as seções do código
print("\n" + "=" * 70)
print("PREPARAÇÃO DOS DADOS PARA MACHINE LEARNING")
print("=" * 70)


# ------------------------------------------------------------
# 19.1 Separação das variáveis de entrada (X) e variável-alvo (y)
# ------------------------------------------------------------

# A variável-alvo (target) é a informação que o modelo de Machine 
# Learning deverá aprender a prever.
target = "NObeyesdad"


# X representa as variáveis de entrada utilizadas pelo modelo.
#
# Remoção de variáveis que não serão utilizadas como atributos explicativos:
# - NObeyesdad: porque é justamente a variável que queremos prever.
# - id: porque é apenas um identificador de cada registro e não
#       representa uma característica relevante do indivíduo.
X = train.drop(columns=[target, "id"])

# y representa a variável-alvo. Corresponde ao nível de obesidade do indivíduo.
y = train[target]

# ------------------------------------------------------------
# 19.2 Exibição das informações para conferência
# ------------------------------------------------------------

print("\nDimensão da base original:")
print(train.shape)

print("\nDimensão de X (variáveis utilizadas pelo modelo):")
print(X.shape)

print("\nDimensão de y (variável-alvo):")
print(y.shape)

print("\nVariáveis presentes em X:")
print(X.columns.tolist())

print("\nVariável-alvo:")
print(target)

print("\nPrimeiros valores da variável-alvo:")
print(y.head())

# ============================================================
# 20. DIVISÃO DA BASE EM TREINAMENTO E VALIDAÇÃO
# ============================================================

# Para realizar o treinamento, será utilizado o arquivo train.csv, 
# que possui as respostas e o mesmo será divido da seguinte forma:

#               Arquivo train.csv
#               20.758 registros
#                      │
#              x + y já separados
#                      │
#             ┌────────┴────────┐
#             │                 │
#             ▼                 ▼
#       TREINAMENTO          VALIDAÇÃO
#           80%                 20%
#      modelo aprende      modelo é avaliado

# Importa a função responsável por dividir os dados.
from sklearn.model_selection import train_test_split


print("\n" + "=" * 70)
print("DIVISÃO DA BASE EM TREINAMENTO E VALIDAÇÃO")
print("=" * 70)


# ------------------------------------------------------------
# 20.1 Divisão dos dados
# ------------------------------------------------------------

# A base será dividida em:
#
# 80% -> treinamento
# 20% -> validação
#
# X_train -> características utilizadas para treinar o modelo
# X_valid -> características utilizadas para validar o modelo
# y_train -> respostas corretas utilizadas no treinamento
# y_valid -> respostas corretas utilizadas para avaliar o modelo
#
# test_size=0.20
# Define que 20% dos registros serão destinados à validação.
#
# random_state=42
# Garante que a divisão seja reproduzível.
# Isso significa que, ao executar novamente o programa,
# obteremos a mesma divisão dos registros.
#
# stratify=y
# Mantém aproximadamente a mesma proporção das classes
# de NObeyesdad nos conjuntos de treinamento e validação.

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 20.2 Conferência das dimensões
# ------------------------------------------------------------

print("\nQuantidade total de registros:")
print(len(X))

print("\nRegistros utilizados para treinamento:")
print(len(X_train))

print("\nRegistros utilizados para validação:")
print(len(X_valid))

print("\nDimensões das bases:")
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)


# ------------------------------------------------------------
# 20.3 Conferência da distribuição das classes
# ------------------------------------------------------------

print("\nDistribuição percentual das classes na base original:")
print(
    (y.value_counts(normalize=True) * 100)
    .round(2)
    .sort_index()
)

print("\nDistribuição percentual das classes no treinamento:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
    .sort_index()
)

print("\nDistribuição percentual das classes na validação:")
print(
    (y_valid.value_counts(normalize=True) * 100)
    .round(2)
    .sort_index()
)

# ============================================================
# 21. PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================

# ColumnTransformer:
# Permite aplicar tratamentos diferentes para grupos diferentes
# de colunas.
#
# OneHotEncoder:
# Converte as variáveis categóricas em colunas numéricas
# utilizando a técnica One-Hot Encoding.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


print("\n" + "=" * 70)
print("PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 70)


# ------------------------------------------------------------
# 21.1 Identificação das variáveis numéricas e categóricas
# ------------------------------------------------------------

# Seleciona automaticamente as colunas numéricas presentes em X.
variaveis_numericas = X.select_dtypes(
    include=["number"]
).columns.tolist()


# Seleciona automaticamente as colunas categóricas presentes em X.
variaveis_categoricas = X.select_dtypes(
    include=["str"]
).columns.tolist()


print("\nVariáveis numéricas:")
print(variaveis_numericas)

print("\nQuantidade de variáveis numéricas:")
print(len(variaveis_numericas))

print("\nVariáveis categóricas:")
print(variaveis_categoricas)

print("\nQuantidade de variáveis categóricas:")
print(len(variaveis_categoricas))


# ------------------------------------------------------------
# 21.2 Criação do pré-processador
# ------------------------------------------------------------

# Para as variáveis categóricas utilizaremos One-Hot Encoding.
#
# handle_unknown="ignore":
# Caso apareça futuramente uma categoria que não estava presente
# durante o treinamento, o programa não apresentará erro.
#
# As variáveis numéricas não precisam ser transformadas neste
# momento e serão mantidas como estão através de:
#
# remainder="passthrough"

preprocessador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            variaveis_categoricas
        )
    ],
    remainder="passthrough"
)

print("\nPré-processador criado com sucesso.")

# ============================================================
# 22. MODELO 1 - DECISION TREE (ÁRVORE DE DECISÃO)
# ============================================================

# Pipeline:
# Permite unir o pré-processamento dos dados e o modelo
# de Machine Learning em uma única estrutura.
#
# DecisionTreeClassifier:
# Algoritmo de classificação baseado em Árvore de Decisão.

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Métricas utilizadas para avaliar o desempenho do modelo.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


print("\n" + "=" * 70)
print("MODELO 1 - DECISION TREE")
print("=" * 70)


# ------------------------------------------------------------
# 22.1 Criação do Pipeline
# ------------------------------------------------------------

# O Pipeline executará automaticamente duas etapas:
#
# 1 - preprocessador:
#     Aplica One-Hot Encoding nas variáveis categóricas
#     e mantém as variáveis numéricas.
#
# 2 - modelo:
#     Treina uma Árvore de Decisão utilizando os dados
#     já preparados.

modelo_decision_tree = Pipeline(
    steps=[
        ("preprocessador", preprocessador),

        (
            "modelo",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# 22.2 Treinamento do modelo
# ------------------------------------------------------------

print("\nTreinando o modelo Decision Tree...")

# O modelo aprende utilizando:
#
# X_train -> características dos indivíduos
# y_train -> classificação correta de cada indivíduo

modelo_decision_tree.fit(
    X_train,
    y_train
)

print("Treinamento concluído com sucesso.")


# ------------------------------------------------------------
# 22.3 Predição utilizando a base de validação
# ------------------------------------------------------------

# Agora apresentamos ao modelo registros que ficaram separados
# do treinamento.
#
# O modelo recebe somente X_valid e deverá prever
# o nível de obesidade de cada indivíduo.

pred_decision_tree = modelo_decision_tree.predict(
    X_valid
)


print("\nQuantidade de previsões realizadas:")
print(len(pred_decision_tree))


# ------------------------------------------------------------
# 22.4 Cálculo das métricas de avaliação
# ------------------------------------------------------------

# Accuracy:
# Percentual total de previsões corretas.

accuracy_dt = accuracy_score(
    y_valid,
    pred_decision_tree
)


# Precision:
# Mede a precisão das classificações realizadas pelo modelo.
#
# average="weighted":
# Calcula a métrica considerando a quantidade de registros
# existente em cada uma das sete classes.

precision_dt = precision_score(
    y_valid,
    pred_decision_tree,
    average="weighted",
    zero_division=0
)


# Recall:
# Mede a capacidade do modelo de identificar corretamente
# os registros pertencentes a cada classe.

recall_dt = recall_score(
    y_valid,
    pred_decision_tree,
    average="weighted",
    zero_division=0
)


# F1-score:
# Combina Precision e Recall em uma única medida.

f1_dt = f1_score(
    y_valid,
    pred_decision_tree,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# 22.5 Exibição das métricas
# ------------------------------------------------------------

print("\nResultados - Decision Tree")

print(f"Accuracy : {accuracy_dt:.4f} ({accuracy_dt * 100:.2f}%)")
print(f"Precision: {precision_dt:.4f} ({precision_dt * 100:.2f}%)")
print(f"Recall   : {recall_dt:.4f} ({recall_dt * 100:.2f}%)")
print(f"F1-score : {f1_dt:.4f} ({f1_dt * 100:.2f}%)")


# ------------------------------------------------------------
# 22.6 Relatório detalhado por classe
# ------------------------------------------------------------

print("\nRelatório de classificação por classe:")

print(
    classification_report(
        y_valid,
        pred_decision_tree,
        zero_division=0
    )
)

# ============================================================
# 23. MODELO 2 - RANDOM FOREST (FLORESTA ALEATÓRIA)
# ============================================================

# RandomForestClassifier:
# Algoritmo de classificação baseado em um conjunto
# de Árvores de Decisão.
#
# Em vez de utilizar apenas uma árvore, como no modelo
# anterior, o Random Forest cria várias árvores.
#
# Cada árvore realiza uma classificação e o Random Forest
# combina os resultados dessas árvores para obter
# a classificação final.

from sklearn.ensemble import RandomForestClassifier


print("\n" + "=" * 70)
print("MODELO 2 - RANDOM FOREST")
print("=" * 70)


# ------------------------------------------------------------
# 23.1 Criação do Pipeline
# ------------------------------------------------------------

# Assim como foi feito na Decision Tree, será utilizado um Pipeline.
#
# O Pipeline executará automaticamente duas etapas:
#
# 1 - preprocessador:
#     Aplica One-Hot Encoding nas variáveis categóricas
#     e mantém as variáveis numéricas.
#
# 2 - modelo:
#     Treina o Random Forest utilizando os dados
#     já preparados.

modelo_random_forest = Pipeline(
    steps=[
        ("preprocessador", preprocessador),

        (
            "modelo",
            RandomForestClassifier(

                # Quantidade de árvores que serão criadas
                # dentro da floresta.
                n_estimators=300,

                # Mantém a aleatoriedade do treinamento
                # reproduzível.
                random_state=42,

                # Permite utilizar todos os núcleos disponíveis
                # do processador durante o treinamento.
                n_jobs=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# 23.2 Treinamento do modelo
# ------------------------------------------------------------

print("\nTreinando o modelo Random Forest...")

# Foi Utilizado exatamente os mesmos dados de treinamento
# usados anteriormente na Decision Tree.
#
# X_train -> características dos indivíduos
# y_train -> classificação correta de cada indivíduo

modelo_random_forest.fit(
    X_train,
    y_train
)

print("Treinamento concluído com sucesso.")


# ------------------------------------------------------------
# 23.3 Predição utilizando a base de validação
# ------------------------------------------------------------

# Foram utilizados os mesmos registros de validação
# utilizados na Decision Tree.
#
# O Random Forest recebe X_valid e deverá prever
# o nível de obesidade de cada indivíduo.

pred_random_forest = modelo_random_forest.predict(
    X_valid
)


print("\nQuantidade de previsões realizadas:")
print(len(pred_random_forest))


# ------------------------------------------------------------
# 23.4 Cálculo das métricas de avaliação
# ------------------------------------------------------------

# Accuracy (Acurácia):
# Percentual total de previsões corretas.

accuracy_rf = accuracy_score(
    y_valid,
    pred_random_forest
)


# Precision (Precisão):
# Mede a precisão das classificações realizadas pelo modelo.

precision_rf = precision_score(
    y_valid,
    pred_random_forest,
    average="weighted",
    zero_division=0
)


# Recall (Revocação):
# Mede a capacidade do modelo de identificar corretamente
# os registros pertencentes a cada classe.

recall_rf = recall_score(
    y_valid,
    pred_random_forest,
    average="weighted",
    zero_division=0
)


# F1-score:
# Combina Precision e Recall em uma única medida.

f1_rf = f1_score(
    y_valid,
    pred_random_forest,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# 23.5 Exibição das métricas
# ------------------------------------------------------------

print("\nResultados - Random Forest")

print(f"Accuracy : {accuracy_rf:.4f} ({accuracy_rf * 100:.2f}%)")
print(f"Precision: {precision_rf:.4f} ({precision_rf * 100:.2f}%)")
print(f"Recall   : {recall_rf:.4f} ({recall_rf * 100:.2f}%)")
print(f"F1-score : {f1_rf:.4f} ({f1_rf * 100:.2f}%)")


# ------------------------------------------------------------
# 23.6 Relatório detalhado por classe
# ------------------------------------------------------------

print("\nRelatório de classificação por classe:")

print(
    classification_report(
        y_valid,
        pred_random_forest,
        zero_division=0
    )
)

# ============================================================
# 24. MODELO FINAL - RANDOM FOREST
# ============================================================

# Após comparar os modelos Decision Tree e Random Forest,
# o Random Forest apresentou o melhor desempenho na base
# de validação.
#
# Agora o modelo será treinado novamente utilizando TODOS
# os registros disponíveis no arquivo train.csv.
#
# Até agora utilizamos:
#
# X_train e y_train -> 80% da base para treinamento
# X_valid e y_valid -> 20% da base para validação
#
# Como a avaliação dos modelos já foi realizada, podemos
# utilizar toda a base conhecida (X e y) para treinar
# o modelo que será aplicado ao test.csv.


print("\n" + "=" * 70)
print("MODELO FINAL - RANDOM FOREST")
print("=" * 70)


# ------------------------------------------------------------
# 24.1 Criação do Pipeline do modelo final
# ------------------------------------------------------------

# O Pipeline continuará realizando duas etapas:
#
# 1 - preprocessador:
#     Aplica One-Hot Encoding nas variáveis categóricas
#     e mantém as variáveis numéricas.
#
# 2 - modelo:
#     Random Forest com 300 Árvores de Decisão.

modelo_final = Pipeline(
    steps=[
        ("preprocessador", preprocessador),

        (
            "modelo",
            RandomForestClassifier(

                # Quantidade de árvores da floresta.
                n_estimators=300,

                # Garante a reprodução dos resultados.
                random_state=42,

                # Utiliza todos os núcleos disponíveis
                # do processador.
                n_jobs=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# 24.2 Treinamento utilizando toda a base train.csv
# ------------------------------------------------------------

print("\nTreinando o modelo final com toda a base de treinamento...")

# X contém as características dos 20.758 indivíduos.
#
# y contém a classificação correta (NObeyesdad)
# desses mesmos indivíduos.
#
# Diferentemente da etapa anterior, agora NÃO utilizamos
# apenas X_train e y_train.
#
# Utilizamos X e y completos.

modelo_final.fit(
    X,
    y
)

print("Treinamento do modelo final concluído com sucesso.")


# ------------------------------------------------------------
# 24.3 Conferência da quantidade de registros utilizados
# ------------------------------------------------------------

print("\nQuantidade de registros utilizados no treinamento final:")
print(len(X))

print("\nQuantidade de características utilizadas pelo modelo:")
print(X.shape[1])

# ============================================================
# 25. APLICAÇÃO DO MODELO FINAL AO TEST.CSV
# ============================================================

# O arquivo test.csv contém os indivíduos para os quais
# será previsto o nível de obesidade (NObeyesdad).
#
# IMPORTANTE:
# O test.csv possui a coluna "id", mas essa coluna não foi
# utilizada durante o treinamento do modelo.
#
# Portanto, precisamos retirar o "id" antes de realizar
# as previsões.


print("\n" + "=" * 70)
print("APLICAÇÃO DO MODELO FINAL AO TEST.CSV")
print("=" * 70)


# ------------------------------------------------------------
# 25.1 Preparação da base de teste
# ------------------------------------------------------------

# Remoção da coluna "id".
#
# A variável IMC já foi criada anteriormente também no
# DataFrame test, portanto ela continuará sendo utilizada
# pelo modelo.

X_test = test.drop(
    columns=["id"]
)


# ------------------------------------------------------------
# 25.2 Conferência da base de teste
# ------------------------------------------------------------

print("\nDimensão da base utilizada para previsão:")
print(X_test.shape)

print("\nQuantidade de registros:")
print(len(X_test))

print("\nQuantidade de características:")
print(X_test.shape[1])


# ------------------------------------------------------------
# 25.3 Conferência das colunas
# ------------------------------------------------------------

# X foi utilizado no treinamento final.
# X_test será utilizado para as previsões.
#
# Os dois precisam possuir as mesmas características.

print("\nAs colunas de X e X_test são iguais?")
print(list(X.columns) == list(X_test.columns))


# ------------------------------------------------------------
# 25.4 Realização das previsões
# ------------------------------------------------------------

print("\nRealizando as previsões...")

pred_test = modelo_final.predict(
    X_test
)

print("Previsões concluídas com sucesso.")


# ------------------------------------------------------------
# 25.5 Conferência das previsões
# ------------------------------------------------------------

print("\nQuantidade de previsões realizadas:")
print(len(pred_test))

print("\nPrimeiras 10 previsões:")

print(
    pred_test[:10]
)

# ============================================================
# 26. GERAÇÃO DO ARQUIVO SUBMISSION.CSV
# ============================================================

print("\n" + "=" * 70)
print("GERAÇÃO DO ARQUIVO SUBMISSION.CSV")
print("=" * 70)


# ------------------------------------------------------------
# 26.1 Criar o arquivo de submissão
# ------------------------------------------------------------

# Será feita uma cópia do arquivo sample_submission.csv fornecido
# pelo Kaggle, gerando um novo arquivo chamado submission.csv.
#
# Esse arquivo já possui a estrutura correta:
#
# id | NObeyesdad
#
# Vamos substituir os valores da coluna NObeyesdad
# pelas previsões realizadas pelo nosso Random Forest.

submission = sample_submission.copy()

submission["NObeyesdad"] = pred_test


# ------------------------------------------------------------
# 26.2 Conferências antes de salvar
# ------------------------------------------------------------

print("\nDimensão do arquivo de submissão:")
print(submission.shape)

print("\nColunas do arquivo:")
print(submission.columns.tolist())

print("\nQuantidade de valores nulos:")
print(submission.isnull().sum())

print("\nOs IDs do submission são iguais aos IDs do test.csv?")
print(
    submission["id"].equals(test["id"])
)

print("\nQuantidade de previsões:")
print(len(submission))

print("\nPrimeiras 10 linhas:")
print(submission.head(10))


# ------------------------------------------------------------
# 26.3 Conferir as classes previstas
# ------------------------------------------------------------

print("\nClasses encontradas nas previsões:")

print(
    sorted(submission["NObeyesdad"].unique())
)


# ------------------------------------------------------------
# 26.4 Salvar o arquivo
# ------------------------------------------------------------

caminho_submission = strDir / "submission.csv"

submission.to_csv(
    caminho_submission,
    index=False
)

print("\nArquivo gerado com sucesso:")
print(caminho_submission)