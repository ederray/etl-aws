"""Funções de criação de gráficos para análise de validação de modelos"""
import logging
import numpy as np
from pandas import Series, DataFrame

import shap
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
import scipy.stats as stats
from sklearn.tree import plot_tree
from sklearn.preprocessing import power_transform
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from ipywidgets import Dropdown, VBox, Output
from IPython.display import display
from typing import Any, List


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
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plote a curva de aprendizado
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Pontuação de Treino")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Pontuação de Validação Cruzada")
    plt.xlabel("Número de exemplos de treinamento")
    plt.ylabel("Pontuação (R²)")
    plt.title("Curva de Aprendizagem")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def grafico_curva_aprendizagem_interativo(
    estimator: Any,X: DataFrame, y: Series, cv: Any, train_size: List[float], scoring: str = 'neg_mean_absolute_error'):
    """
    Gera um gráfico interativo da curva de aprendizado para uma ação,
    comparando a pontuação de treino e validação.
    
    Args:
        estimator (Any): O estimador (modelo) a ser usado.
        X (pd.DataFrame): DataFrame completo com todas as features, incluindo 'acao'.
        y (pd.Series): Series completa com o valor alvo ('y_real').
        cv (Any): Estratégia de validação cruzada (ex: TimeSeriesSplit).
        train_size (list[float]): Lista de frações de dados para treinamento.
        scoring (str): Métrica de pontuação.
    """
    if 'acao' not in X.columns:
        print("Erro: A coluna 'acao' não foi encontrada no DataFrame X.")
        return

    tickers = sorted(X['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_curva(change):
        output.clear_output(wait=True)
        ticker = change['new']
        
        # Filtra os dados de X e y para a ação selecionada
        X_acao = X[X['acao'] == ticker]
        y_acao = y[X['acao'] == ticker]

        with output:
            if X_acao.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return
            
            if len(X_acao) <= cv.get_n_splits():
                print(f"Série para {ticker} é muito curta para o número de splits ({cv.get_n_splits()}).")
                return

            train_sizes, train_scores, test_scores = learning_curve(
                estimator=estimator,
                X=X_acao,
                y=y_acao,
                cv=cv,
                scoring=scoring,
                train_sizes=train_size
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Pontuação de Treino")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Pontuação de Validação Cruzada")
            plt.xlabel("Número de exemplos de treinamento")
            plt.ylabel(f"Pontuação ({scoring})")
            plt.title(f"Curva de Aprendizagem | Ação: {ticker}")
            plt.legend(loc="best")
            plt.grid()
            plt.show()

    dropdown.observe(atualizar_curva, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))

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


def grafico_qq_plot_funcao_normal(serie: Series, scale=False, metodo_scale='box-cox'):
    """
    Plota QQ-Plots para as colunas numéricas de um DataFrame.

    Parâmetros:
    - serie: pd.Series.
    - scale: bool, se True aplica power_transform.
    - metodo_scale: str, método de transformação: 'box-cox' ou 'yeo-johnson'.

    Retorna:
    - None, apenas plota os gráficos.
    """

    # Aplica transformação se necessário
    if scale:
        serie_scaled = serie.copy()
        serie_scaled = power_transform(
            serie_scaled.to_numpy().reshape(-1, 1), method=metodo_scale).flatten()
        serie_final = Series(serie_scaled, index=serie.index, name=serie.name)
        title_suffix = f'(Scaled: {metodo_scale})'
    else:
        serie_final = serie.copy()
        title_suffix = '(Original)'
    
    # construção da figura
    _, ax = plt.subplots(1, 1, figsize=(10,4))
    
    # Use stats from scipy, not statsmodels
    stats.probplot(serie_final, dist="norm", plot=ax)
    ax.set_title(f'QQ-Plot do resíduo {title_suffix}')
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.show()


def teste_ljung_box(residuos, lags=20, alpha=0.05):
    """
    Executa o teste de Ljung-Box para verificar autocorrelação nos resíduos.
    
    Parâmetros:
        residuos (array-like): Série de resíduos do modelo.
        lags (int): Número de defasagens a serem testadas.
        alpha (float): Nível de significância do teste.

    Retorna:
        DataFrame com estatística Q, p-valor e decisão de rejeição da hipótese nula.
    """
    ljung_box = acorr_ljungbox(residuos, lags=lags, return_df=True)
    ljung_box["Rejeita_H0"] = ljung_box["lb_pvalue"] < alpha
    return ljung_box


def gerar_previsao_valores_interativo(df: DataFrame):
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

def histograma_residuos_interativo(df: DataFrame):
    """
    Gera um histograma interativo dos resíduos para cada ticker.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'acao', 'y_real' e 'y_pred'.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Selecionar Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_histograma(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        
        with output:
            if df_acao.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return

            residuos = df_acao['y_real'] - df_acao['y_pred']

            plt.figure(figsize=(10, 5))
            residuos.plot(kind='hist',title=f'Histograma dos Resíduos | Ação: {ticker}')
            plt.xlabel('Resíduos')
            plt.ylabel('Frequência')
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_histograma, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))

def grafico_analise_dispersao_residuo_interativo(df: DataFrame):
    """
    Gera um gráfico de dispersão interativo para resíduos vs. valores previstos por ticker.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Selecionar Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_plot(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        y_pred = df_acao['y_pred']
        res = df_acao['y_real'] - y_pred

        with output:
            if y_pred.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return
            
            plt.figure(figsize=(10, 5))
            plt.scatter(y_pred, res, alpha=0.6)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Valor previsto')
            plt.ylabel('Resíduos')
            plt.title(f'Dispersão (Previsão vs Resíduo) | Ação: {ticker}')
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_plot, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))


def grafico_analise_dispersao_target_interativo(df: DataFrame):
    """
    Gera um gráfico de dispersão interativo para valores previstos vs. target por ticker.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Selecionar Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_plot(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        y_pred = df_acao['y_pred']
        target = df_acao['y_real']
        
        with output:
            if y_pred.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return
            
            plt.figure(figsize=(10, 5))
            sns.regplot(x=y_pred, y=target, scatter_kws={"alpha": 0.8}, line_kws={"color": "red"})
            plt.xlabel('Valor previsto')
            plt.ylabel('Target')
            plt.title(f'Dispersão (Previsão vs Target) | Ação: {ticker}')
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_plot, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))


def grafico_qq_plot_funcao_normal_interativo(df: DataFrame, scale=False, metodo_scale='yeo-johnson'):
    """
    Gera um Q-Q Plot interativo para os resíduos por ticker.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Selecionar Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_plot(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        residuos = df_acao['y_real'] - df_acao['y_pred']

        with output:
            if residuos.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return
            
            # Lógica de transformação
            serie_final = residuos.copy()
            title_suffix = '(Original)'
            if scale:
                serie_scaled = residuos.to_numpy().reshape(-1, 1)
                serie_scaled = power_transform(serie_scaled, method=metodo_scale).flatten()
                serie_final = Series(serie_scaled, index=residuos.index, name=residuos.name)
                title_suffix = f'(Scaled: {metodo_scale})'
            
            plt.figure(figsize=(10, 5))
            stats.probplot(serie_final, dist="norm", plot=plt.gca())
            plt.title(f'QQ-Plot do resíduo | Ação: {ticker} {title_suffix}')
            plt.xlabel('Quantis Teóricos')
            plt.ylabel('Quantis Amostrais')
            plt.tight_layout()
            plt.show()

    dropdown.observe(atualizar_plot, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))


def teste_ljung_box_interativo(df: DataFrame, lags=20, alpha=0.05):
    """
    Executa o teste de Ljung-Box interativamente para os resíduos por ticker.
    """
    tickers = sorted(df['acao'].dropna().unique())
    dropdown = Dropdown(options=tickers, description='Selecionar Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_teste(change):
        output.clear_output(wait=True)
        ticker = change['new']
        df_acao = df[df['acao'] == ticker].dropna()
        residuos = df_acao['y_real'] - df_acao['y_pred']

        with output:
            if residuos.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return
            
            ljung_box = acorr_ljungbox(residuos, lags=lags, return_df=True)
            ljung_box["Rejeita_H0"] = ljung_box["lb_pvalue"] < alpha
            
            print(f"Resultados do Teste de Ljung-Box para a Ação: {ticker}\n")
            display(ljung_box)
            print("---")
            
            p_value_geral = ljung_box.iloc[-1]["lb_pvalue"]
            if p_value_geral > alpha:
                print(f"O p-valor ({p_value_geral:.4f}) é maior que {alpha}. Não há autocorrelação significativa.")
            else:
                print(f"O p-valor ({p_value_geral:.4f}) é menor que {alpha}. Há autocorrelação significativa.")
    
    dropdown.observe(atualizar_teste, names='value')
    dropdown.value = tickers[0]
    display(VBox([dropdown, output]))

def grafico_acf_interativo(df: DataFrame, max_lags: int, coluna_valor: str = 'y_pred'):
    """
    Gera gráfico de autocorrelação (ACF) para uma série temporal, interativamente por ticker.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'acao' e a coluna numérica de interesse.
        max_lags (int): Número máximo de lags para o gráfico.
        coluna_valor (str): Nome da coluna numérica (ex: 'close', 'trailingPE', etc.).
    """
    tickers = sorted(df['acao'].dropna().unique())

    dropdown = Dropdown(
        options=tickers,
        description='Ação:',
        layout={'width': '300px'}
    )

    output = Output()

    def atualizar_acf(change):
        output.clear_output(wait=True)
        acao = change['new']

        serie = df[df['acao'] == acao][coluna_valor].dropna()

        with output:
            if serie.empty:
                print(f"Nenhuma série disponível para {acao}")
                return

            plt.figure(figsize=(10, 4))
            plot_acf(serie, lags=max_lags)
            plt.title(f'ACF - {coluna_valor} | Ação: {acao}')
            plt.tight_layout()
            plt.show()
            plt.close()  # <- fecha a figura para evitar acumulação

    dropdown.observe(atualizar_acf, names='value')
    dropdown.value = tickers[0]  # força exibição inicial

    display(VBox([dropdown, output]))

def grafico_pacf_interativo(df: DataFrame, max_lags: int, coluna_valor: str = 'y_pred', metodo: str = 'ywm'):
    """
    Gera gráfico de autocorrelação parcial (PACF) interativo por ticker.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'acao' e a coluna numérica.
        max_lags (int): Número máximo de lags.
        coluna_valor (str): Nome da coluna numérica.
        metodo (str): Método do PACF ('ywm' por padrão).
    """
    # Corrigido: Usando a coluna 'acao' para o dropdown
    acoes = sorted(df['acao'].dropna().unique())

    dropdown = Dropdown(
        options=acoes,
        description='Ação:',
        layout={'width': '300px'}
    )

    output = Output()

    def atualizar_pacf(change):
        output.clear_output(wait=True)
        # Corrigido: Usando 'acao' para capturar o valor do dropdown
        acao = change['new']
        # Corrigido: Usando 'acao' para filtrar o DataFrame
        serie = df[df['acao'] == acao][coluna_valor].dropna()

        with output:
            if serie.empty:
                print(f"Nenhuma série disponível para {acao}")
                return

            limite = len(serie) // 2
            lags_ajustado = min(max_lags, limite)
            if lags_ajustado < max_lags:
                print(f"[Atenção] Série muito curta. Reduzindo lags de {max_lags} para {lags_ajustado}.")

            plt.figure(figsize=(10, 4))
            plot_pacf(serie, lags=lags_ajustado, method=metodo)
            # Corrigido: Usando 'acao' no título
            plt.title(f'PACF - {coluna_valor} | Ação: {acao}')
            plt.tight_layout()
            plt.show()
            plt.close()

    dropdown.observe(atualizar_pacf, names='value')
    dropdown.value = acoes[0]
    display(VBox([dropdown, output]))

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



def grafico_residuos_ao_longo_do_tempo_interativo(df: DataFrame):
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


def grafico_metricas_ao_longo_dos_folds_interativo(df: DataFrame, model: Any, n_splits: int = 5):
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


def grafico_rolling_forecast_interativo(df: DataFrame, model: Any, n_splits: int = 5):
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
