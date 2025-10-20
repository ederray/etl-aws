# ETL na AWS  — Tech Challenge ML Engineer Fase 2  


Este projeto foi desenvolvido como parte do **Tech Challenge**, com foco na construção de um **pipeline ETL automatizado** para as ações da **Carteira Ibovespa**, incluindo a captura de dados via **API YFinance**, etapas de **limpeza e transformação**, e o consumo de dados na **Amazon Web Services (AWS)**.

Com base nesse pipeline, foi implementada uma **arquitetura MLOps serverless**, projetada para **prever o preço de fechamento futuro das ações** ($\text{Close}_{t+1}$), utilizando técnicas de **Machine Learning aplicadas a séries temporais financeiras**.

O projeto combina **engenharia de dados, automação em nuvem e modelagem preditiva**, garantindo **escalabilidade, rastreabilidade de experimentos e reprodutibilidade completa do pipeline**.

---

## 🏗️ Arquitetura do Projeto

![Arquitetura do Projeto](reports\Arquitetura_ETL_na_AWS.png)

---

## 🎯 Objetivo do Projeto

### Construção do Pipeline e Previsão de Preço ($\text{Close}_{t+1}$) 

O objetivo é construir um pipeline robusto de captura do preço das ações da carteira Ibovespa e prever o **preço de fechamento do próximo dia** com base em dados técnicos e contextuais.

* **Regressão (Nível):** minimizar o Erro Absoluto Médio (MAE), alcançando **MAE ≈ 0.81**.
---

## 📊 Dataset e Features

O dataset fornece uma visão abrangente do **mercado de capitais brasileiro (B3)**, incluindo:

* **Preços Históricos:** Abertura, Fechamento, Máxima, Mínima, Volume.
* **Contexto:** Setor, Indústria e Tipo de Ativo.
* **Features de Momentum:** Indicadores técnicos como **MACD** e **RSI**.
* **Features de Nível e Variação:** Lags e variações acumuladas (`lag_1_Close`, `lag_30_Close`, `lag_X_close_diff`).

---

## 📁 Estrutura do Repositório

```
├── data/                       
│   ├── raw/                    # Dados brutos (B3)
│   └── processed/              # Dados com Feature Engineering (MACD, RSI, Lags)
├── notebooks/                  
│   ├── 02_EDA                  # Análise Exploratória de Dados (EDA)
│   └── 02_MODEL_VALIDATION     # Validação e Análises Gráficas
├── reports/                    
│   └── figures/models/         # Plots e gráficos (Curvas, SHAP)
├── src/                        
│   ├── config/                 # Configurações do projeto
│   ├── data/                   # Processamento e ETL
│   ├── features/               # Cálculo de indicadores técnicos
│   ├── modeling/               # Treinamento e tuning de modelos
│   ├── models/                 # Modelos serializados
│   ├── utils/                  # Funções utilitárias
│   └── visualization/          # Visualizações e gráficos
├── tests/                      # Testes unitários e integração
├── glue.py                     # Script ETL AWS Glue
├── main.py                     # Script de automação da captura de ações
├── project.toml                # Configuração Poetry
└── requirements.txt            # Dependências
```

---

## ⚙️ Funcionalidades Implementadas

### 🧩 Pipeline de Dados

* ✅ Extração e limpeza de dados históricos.
* ✅ Engenharia de Features de séries temporais (Lags, Retornos).
* ✅ Cálculo de Indicadores Técnicos (MACD, RSI).

### 🤖 Modelagem de Machine Learning

* ✅ Otimização de hiperparâmetros com **HalvingGridSearchCV**.
* ✅ Treinamento com **XGBoost Regressor** (modelo vencedor).
* ✅ Rastreamento de experimentos e versionamento com **MLflow**.
* ✅ Análise de interpretabilidade com **SHAP Values**.

---

## 🧰 Tecnologias Utilizadas

* **Python 3.11.x**
* **Pandas** e **NumPy** — manipulação de dados.
* **Scikit-learn** — pipelines e validação.
* **Category-Encoder** — feature target encoder.
* **XGBoost** e **LightGBM** — modelagem preditiva.
* **SHAP** — interpretabilidade de modelos.
* **Matplotlib e Seaborn** — visualizações.

---

## ⚙️ Setup e Instalação

### Pré-requisitos

* Python 3.11+
* [Poetry](https://python-poetry.org/) instalado globalmente.

### Passos

```bash
# 1. Clonar o repositório
git clone https://github.com/ederray/etl-aws.git
cd etl-aws

# 2. Instalar dependências
poetry install

# 3. Ativar ambiente virtual
poetry shell
```


## ☁️ Implantação AWS Serverless

A arquitetura MLOps foi projetada de forma **serverless**, aproveitando os serviços gerenciados da AWS para garantir **escalabilidade**, **baixo custo** e **observabilidade**:

| Serviço AWS                       | Função               | Descrição                                                |
| --------------------------------- | -------------------- | -------------------------------------------------------- |
| **S3**                            | Data Lake            | Armazena dados brutos, processados e previsão.           |
| **Lambda**                        | Trigger              | Funções automáticas para ETL e previsão.                 |
| **Glue**                          | ETL                  | ETL e catálogo de dados.                                 |
| **Athena**                        | Análise de dados     | Agendamento de execuções e reprocessamentos.             |
| **CloudWatch**                    | Logs e Métricas      | Monitoramento de pipelines e funções.                    |
| **ECR**                           | Repositório Docker   | Armazena e versiona imagens Docker usadas em jobs de ML. |

Fluxo simplificado:

```
S3 (Raw Data)
   ↓
Lambda (Preprocessing)
   ↓
Glue (Preprocessing)
   ↓
Glue Catalog (Preprocessing)
   ↓
S3 (Processed)
   ↓
Lambda (Prediction)
   ↓
S3 (Predicted)
   ↓
Athena (Data Ananyliss)

```

---

## 📈 Resultados e Insights Finais

| Métrica                 | Valor                  | Insights                                                    |
| ----------------------- | ---------------------- | ----------------------------------------------------------- |
| **MAE (Teste)**         | ≈ 0.81                 | Baixo erro absoluto, indicando previsões de preço precisas. |
| **Overfitting**         | Controlado (Gamma=1.0) | Regularização forte garantiu robustez e generalização.      |


---

## 🧪 Testes e Validação

Notebooks de validação (Curva de Aprendizagem e análise SHAP) estão disponíveis em:
`notebooks/02_MODEL_VALIDATION/`

---

## 🤝 Contribuições

Este projeto foi desenvolvido como parte do **Tech Challenge**.
Pull requests e issues são bem-vindos!

---

## 📄 Licença

Este projeto está sob a licença especificada no arquivo **LICENSE**.



