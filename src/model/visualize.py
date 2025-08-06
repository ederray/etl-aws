"""Funções de criação de gráficos para análise de validação de modelos"""
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, ...
import shap
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import plot_tree

# instância do objeto logger
logger = logging.getLogger(__name__)


def grafico_shap_value_beeswarm(model, X_df):
    """
    Cria um gráfico Beeswarm de SHAP values.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)
    shap.plots.beeswarm(shap_values)


def grafico_shap_value_scatterplot():
    pass



def grafico_shap_value_barplot():
    pass


# Você pode adicionar as outras funções aqui...
def grafico_arvore_decisao(model, X_df):
    """
    Cria um gráfico de uma árvore de decisão.
    """
    # A função plot_tree é para estimadores de árvore individuais.
    # Para modelos de ensemble como XGBoost, CatBoost ou RandomForest,
    # você precisaria extrair uma única árvore para plotar.
    plt.figure(figsize=(20,10))
    # Exemplo para Random Forest
    # plot_tree(model.estimators_[0], feature_names=X_df.columns, filled=True, rounded=True)
    plt.show()



