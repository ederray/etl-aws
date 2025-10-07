"""Modulo de funções geradoras de gráfico para avaliação de modelo"""

import logging
from typing import Any, List

import graphviz
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, learning_curve
from sklearn.tree import export_graphviz

# inicializa o javascript para renderizar a imagem
shap.initjs()

# instância do objeto logger
logger = logging.getLogger(__name__)


def grafico_curva_aprendizagem(
    estimator: Any,
    X: DataFrame,
    y: Series,
    train_size: List[float],
    titulo: str = "Curva de Aprendizagem",
    scoring: str = "neg_mean_absolute_error",
    cv: Any = KFold,
    path: str = None,
    ax=None,
) -> None:
    """
    Gera um gráfico da curva de aprendizado,comparando a pontuação de treino e validação.

    params:
        estimator (Any): O estimador (modelo) a ser usado.
        X (DataFrame): DataFrame completo com todas as features, incluindo 'acao'.
        y (Series): Series completa com o valor alvo ('y_real').
        cv (Any): Estratégia de validação cruzada (ex: KFold).
        train_size (list[float]): Lista de frações de dados para treinamento.
        scoring (str): Métrica de pontuação.
        path (str): caminho para salvamento da imagem.
        ax (matplotlib axis): Eixo para plotar. Se None, cria novo.

    return:
        None: gráfico curva de aprendizagem.
    """

    was_called_alone = ax is None

    try:
        if was_called_alone:
            fig, ax = plt.subplots(figsize=(10, 6))

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            train_sizes=train_size,
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        ax.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Pontuação de Treino"
        )
        ax.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Pontuação de Validação Cruzada",
        )

        ax.set_xlabel("Número de exemplos de treinamento")
        ax.set_ylabel(f"Pontuação ({scoring})")
        ax.set_title(f"{titulo}")

        ax.legend(loc="best")

        if was_called_alone:
            fig.tight_layout()

            if path:
                fig.savefig(path)

            plt.show()
            plt.close(fig)

    except Exception as e:
        logger.error(f"Erro ao gerar a curva de aprendizado: {e}")
        return


def gerar_grafico_shap(
    model, X_df: DataFrame, n_features: int = 20, kind: str = "bar", path: str = None
) -> None:
    """
    Gera um gráfico de barras com a importância global das features.

    params:
        model: O modelo LightGBM treinado.
        X_df (pd.DataFrame): O DataFrame de features (ex: X_test).
        n_features (int): O número de features a serem plotadas.
        kind (str): Tipo de gráfico shap: bar ou beeswarm.
        path (str): caminho para salvamento da imagem.

    return:
        None: gráfico shap plot.

    """
    try:
        if kind == "bar":
            explainer = shap.Explainer(model)
            shap_values = explainer.shap_values(X_df)
            shap.summary_plot(
                shap_values, X_df, plot_type=kind, max_display=n_features, show=False
            )

        elif kind == "beeswarm":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            shap.summary_plot(
                shap_values, X_df, max_display=n_features, plot_type="dot", show=False
            )

    except Exception as e:
        raise logger.error(e)

    if path:
        try:
            plt.gcf().tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            raise logger.error(e)

    plt.title(
        "Importância Global das Features (Média Absoluta dos SHAP Values)", fontsize=16
    )
    plt.show()


def gerar_arvore(
    modelo: Any,
    feature_list: list,
    model_type: str = "decisiontree",
    filename: str = None,
) -> graphviz.Source:
    """
    Função que gera um gráfico de árvore de decisão para modelos Decision Tree,
    Random Forest ou LightGBM, usando create_tree_digraph() para LightGBM.

    params:
        model: O modelo treinado.
        feature_list (list): A lista de features.
        model_type (str): Tipo de gráfico de árvore: decisiontree,lightgbm ou randomforest.
        filename (str): caminho para salvamento da imagem.

    return:
        None: gráfico de árvore.

    """

    model_type_lower = model_type.lower()

    # --- 1. Lógica para LightGBM (Usando create_tree_digraph) ---
    if model_type_lower == "lightgbm":
        logger.info("Plotando a primeira árvore (índice 0) do modelo LightGBM.")

        # Obtém o booster (o modelo LightGBM puro)
        booster = modelo.booster_

        # PASSO CRUCIAL: Força a atualização do nome das features no booster
        # Isso corrige a ausência de nomes ('Column_X')
        # E permite que a próxima chamada funcione sem o argumento 'feature_names'
        booster.feature_name = feature_list

        # Chama a função create_tree_digraph (preferida pela documentação)
        # NENHUM 'feature_names' AQUI para evitar o TypeError de argumento duplicado
        graph = lgb.create_tree_digraph(
            booster=booster, tree_index=0, orientation="vertical"
        )

        # Renderiza e salva
        if filename:
            graph.render(filename=filename, format="png", cleanup=True)

        return graph

    # --- 2. Lógica para Modelos Scikit-learn (Decision Tree e Random Forest) ---
    # ... (Seu código original para Random Forest e Decision Tree, que usa export_graphviz) ...

    arvore_para_plotar = None

    if model_type_lower == "randomforest":
        # Seleciona a primeira árvore do ensemble
        arvore_para_plotar = modelo.estimators_[0]
        logger.info("Plotando a primeira árvore (índice 0) do modelo Random Forest.")

    elif model_type_lower == "decisiontree":
        # Usa o modelo DT diretamente
        arvore_para_plotar = modelo

    else:
        logger.warning(f"model_type '{model_type}' não reconhecido. Falha na plotagem.")
        return graphviz.Source("")

    # --- 3. Executa a Exportação do Scikit-learn ---

    data = export_graphviz(
        arvore_para_plotar,
        out_file=None,
        filled=True,
        rounded=True,
        class_names=None,  # Mantemos None para Regressão
        feature_names=feature_list,
    )

    graph = graphviz.Source(data)

    if filename:
        graph.render(filename=filename, format="png", cleanup=True)

    return graph

def gerar_previsao_valores(df: DataFrame):
    """
    Compara visualmente os valores reais com os previstos para uma ação, usando um Output widget.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'acao', 'y_real' e 'y_pred'.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Selecionar Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_grafico(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        
        with output:
            if df_acao.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return

            plt.figure(figsize=(12, 6))
            plt.plot(df_acao['y_real'].values, label='Valor Real', color='blue', marker='o', markersize=4)
            plt.plot(df_acao['y_pred'].values, label='Previsão', color='red', marker='x', markersize=4)
            
            plt.title(f'Comparação: Valor Real vs. Previsão | Ação: {ticker}')
            plt.xlabel('Observação')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_grafico, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))







 

def grafico_residuos_ao_longo_do_tempo(df: DataFrame):
    """
    Plota os resíduos de um modelo ao longo do tempo para uma ação selecionada,
    ajudando a identificar erros sistemáticos.

    Args:
        df (pd.DataFrame): DataFrame contendo 'acao', 'y_real' e 'y_pred'.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_grafico(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        
        with output:
            if df_acao.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return

            residuos = df_acao['y_real'] - df_acao['y_pred']

            plt.figure(figsize=(12, 6))
            plt.plot(residuos.values, marker='o', linestyle='-', markersize=4, alpha=0.7)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Resíduo Zero')
            plt.title(f'Resíduos ao Longo do Tempo | Ação: {ticker}')
            plt.xlabel('Observação')
            plt.ylabel('Resíduos')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_grafico, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))


def grafico_metricas_ao_longo_dos_folds(df: DataFrame, model: Any, n_splits: int = 5):
    """
    Plota métricas de desempenho (RMSE e MAE) ao longo dos folds do TimeSeriesSplit
    para uma ação selecionada, mostrando a estabilidade do modelo.

    Args:
        df (pd.DataFrame): DataFrame completo com todas as ações.
        model (Any): O modelo de machine learning treinado.
        n_splits (int): Número de splits para o TimeSeriesSplit.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_grafico(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        
        with output:
            if len(df_acao) < n_splits + 1:
                print(f"Série para {ticker} é muito curta para {n_splits} splits.")
                return

            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            rmse_scores = []
            mae_scores = []
            
            X_acao = df_acao.drop(columns=['y_real', 'acao'], errors='ignore')
            y_acao = df_acao['y_real']
            
            # Treina e avalia o modelo em cada fold
            for train_index, test_index in tscv.split(X_acao):
                X_train, X_test = X_acao.iloc[train_index], X_acao.iloc[test_index]
                y_train, y_test = y_acao.iloc[train_index], y_acao.iloc[test_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae_scores.append(mean_absolute_error(y_test, y_pred))

            plt.figure(figsize=(12, 6))
            plt.plot(range(1, n_splits + 1), rmse_scores, marker='o', label='RMSE', color='blue')
            plt.plot(range(1, n_splits + 1), mae_scores, marker='o', label='MAE', color='red')
            
            plt.title(f'Métricas de Validação Cruzada | Ação: {ticker}')
            plt.xlabel('Fold')
            plt.ylabel('Valor da Métrica')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_grafico, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))


def grafico_rolling_forecast(df: DataFrame, model: Any, n_splits: int = 5):
    """
    Plota uma visualização de backtesting com previsões em múltiplos folds do TimeSeriesSplit,
    mostrando a robustez do modelo ao longo do tempo.

    Args:
        df (pd.DataFrame): DataFrame completo com todas as ações.
        model (Any): O modelo de machine learning treinado.
        n_splits (int): Número de splits para o TimeSeriesSplit.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_grafico(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        
        with output:
            if len(df_acao) < n_splits + 1:
                print(f"Série para {ticker} é muito curta para {n_splits} splits.")
                return

            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            X_acao = df_acao.drop(columns=['y_real', 'acao'], errors='ignore')
            y_acao = df_acao['y_real']
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_acao.values, label='Valor Real', color='blue')

            # Realiza e plota as previsões para cada fold
            for i, (train_index, test_index) in enumerate(tscv.split(X_acao)):
                X_train, X_test = X_acao.iloc[train_index], X_acao.iloc[test_index]
                y_train, y_test = y_acao.iloc[train_index], y_acao.iloc[test_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                plt.plot(test_index, y_pred, linestyle='--', color='red', label=f'Previsão Fold {i+1}')
            
            plt.title(f'Backtesting com Previsão Contínua | Ação: {ticker}')
            plt.xlabel('Observação')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_grafico, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))
