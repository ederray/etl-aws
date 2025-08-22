"""Funções de criação de gráficos para análise de validação de modelos"""
import logging
import numpy as np
from pandas import Series
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.model_selection import learning_curve

# instância do objeto logger
logger = logging.getLogger(__name__)


def grafico_curva_aprendizagem(estimator, X:Series, y:Series, cv, train_size:list[float],scoring:str='r2') -> plt.plot:
    """Função para construção do gráfico de dispersão dos resíduos com os valores previstos."""
    train_sizes, train_scores, test_scores = learning_curve(
    estimator=estimator,
    X=X,
    y=y,
    cv=cv,
    scoring=scoring,
    train_sizes=train_size 
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plote a curva de aprendizado
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Pontuação de Treino")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Pontuação de Validação Cruzada")
    plt.xlabel("Número de exemplos de treinamento")
    plt.ylabel("Pontuação (R²)")
    plt.title("Curva de Aprendizagem")
    plt.legend(loc="best")
    plt.grid()
    plt.show() 

def grafico_analise_dispersao_residuo(y_pred:Series, res:Series) -> plt.plot:
    """Função para construção do gráfico de dispersão dos resíduos com os valores previstos."""
    plt.scatter(y_pred, res)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Valor previsto')
    plt.ylabel('Resíduos')
    plt.title(f'Gráfico de dispersão dos valores previstos x resíduo')
    plt.show()

def grafico_analise_dispersao_target(y_pred:Series, target:Series) -> plt.plot:
    """Função para construção do gráfico de dispersão dos resíduos com os valores previstos."""
    sns.regplot(x=y_pred, y=target, scatter_kws={"alpha": 0.8}, line_kws={"color": "red"})
    plt.xlabel('Valor previsto')
    plt.ylabel('Target')
    plt.title(f'Gráfico de dispersão dos valores previstos x target')
    plt.show()


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



