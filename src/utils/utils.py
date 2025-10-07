"""Funções utilitárias para uso do modelo"""
import logging
import pickle
import pandas as pd
# instância do objeto logger
logger = logging.getLogger(__name__)

def salvar_modelo(modelo, caminho_salvar):
    """
    Salva o modelo treinado em um arquivo.
    """
    pickle.dump(modelo, caminho_salvar)
    print(f"\nModelo salvo em: {caminho_salvar}")

def carregar_modelo():
    pass



def gerar_dataset_validacao(X_teste, y_teste, y_pred, coluna_acao="acao"):
    """
    Gera um dataset de validação com valores reais e previstos.
    
    Parâmetros:
        X_teste (DataFrame): Conjunto de features de teste.
        y_teste (array-like ou Series): Valores reais do target.
        y_pred (array-like): Previsões do modelo.
        coluna_acao (str): Nome da coluna identificadora da ação.
    
    Retorna:
        DataFrame com colunas: [acao, y_real, y_pred].
    """
    df = X_teste[[coluna_acao]].copy()
    df["y_real"] = pd.Series(y_teste).values
    df["y_pred"] = pd.Series(y_pred).values
    return df



