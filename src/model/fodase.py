# train.py

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from utils import gerar_metricas  # Importamos a função de métricas do arquivo utils.py

# ================================================================
# FUNÇÕES DO FLUXO DE TREINAMENTO
# ================================================================

def carregar_e_preparar_dados(caminho_treino, caminho_teste):
    """
    Carrega os datasets de treino e teste e os prepara para o modelo.
    Retorna X_treino, y_treino, X_teste, y_teste.
    """
    df_treino = pd.read_csv(caminho_treino)
    df_teste = pd.read_csv(caminho_teste)

    # Identificar features e target
    X_treino = df_treino.drop('Close_diff', axis=1)
    y_treino = df_treino['Close_diff']
    X_teste = df_teste.drop('Close_diff', axis=1)
    y_teste = df_teste['Close_diff']

    return X_treino, y_treino, X_teste, y_teste


def criar_pipeline_catboost(X_treino):
    """
    Cria e retorna o pipeline completo para o modelo CatBoost.
    """
    # Define colunas numéricas e categóricas
    colunas_numericas = X_treino.select_dtypes(include=np.number).columns.tolist()
    colunas_categoricas = ['ticker', 'tipo', 'setor', 'industria']
    
    # Define o pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), colunas_numericas),
            ('cat', 'passthrough', colunas_categoricas)
        ],
        remainder='drop'
    )
    
    # Define os índices das colunas categóricas para o CatBoost
    cat_features_indices = list(range(len(colunas_numericas), len(colunas_numericas) + len(colunas_categoricas)))
    
    # Cria a instância do modelo CatBoost
    catboost_model = CatBoostRegressor(
        random_state=42, 
        verbose=0,
        cat_features=cat_features_indices
    )
    
    # Cria e retorna o pipeline
    pipeline = Pipeline(steps=[
        ('preprocessador', preprocessor),
        ('modelo', catboost_model)
    ])
    
    return pipeline


def treinar_e_avaliar_modelo(pipeline, X_treino, y_treino, X_teste, y_teste):
    """
    Treina o pipeline, faz previsões e exibe as métricas.
    """
    print("Iniciando o treinamento do modelo CatBoost...")
    pipeline.fit(X_treino, y_treino)
    print("Treinamento concluído.")
    
    y_pred = pipeline.predict(X_teste)
    
    print("\n--- Métricas de Desempenho (conjunto de teste) ---")
    gerar_metricas(y_teste, y_pred)
    
    return pipeline





