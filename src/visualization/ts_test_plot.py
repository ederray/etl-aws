"""Funções de criação de gráficos para análise de validação de modelos"""
import logging
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from ipywidgets import Dropdown, VBox, Output
from IPython.display import display
from typing import Any, List

# instância do objeto logger
logger = logging.getLogger(__name__)


from ipywidgets import Dropdown, VBox, Output
from IPython.display import display

def teste_ljung_box(df: DataFrame, lags=20, alpha=0.05):
    """
    Executa o teste de Ljung-Box interativamente para os resíduos (y_real - y_pred) por ticker.
    Este teste verifica se os resíduos são ruído branco (sem autocorrelação).
    """
    # Verifica se as colunas necessárias existem
    if 'acao' not in df.columns or 'y_real' not in df.columns or 'y_pred' not in df.columns:
        print("Erro: O DataFrame deve conter as colunas 'acao', 'y_real' e 'y_pred'.")
        return

    tickers = sorted(df['acao'].dropna().unique())
    
    if not tickers:
        print("Erro: Nenhuma ação válida encontrada na coluna 'acao'.")
        return

    dropdown = Dropdown(options=tickers, description='Selecionar Ação:', layout={'width': '300px'})
    output = Output()

    def atualizar_teste(change):
        output.clear_output(wait=True)
        ticker = change['new']
        # Filtra a ação e calcula os resíduos
        df_acao = df[df['acao'] == ticker].dropna(subset=['y_real', 'y_pred'])
        residuos = df_acao['y_real'] - df_acao['y_pred']

        with output:
            if residuos.empty:
                print(f"Nenhuma série de resíduos disponível para {ticker}")
                return
            
            # Executa o teste de Ljung-Box
            ljung_box = acorr_ljungbox(residuos, lags=lags, return_df=True)
            ljung_box["Rejeita_H0"] = ljung_box["lb_pvalue"] < alpha
            
            print(f"Resultados do Teste de Ljung-Box para a Ação: {ticker}\n")
            display(ljung_box)
            print("---")
            
            # Interpretação do resultado no último lag
            p_value_geral = ljung_box.iloc[-1]["lb_pvalue"]
            if p_value_geral > alpha:
                print(f"O p-valor ({p_value_geral:.4f}) é maior que {alpha}. Não há autocorrelação significativa (indicativo de bom modelo).")
            else:
                print(f"O p-valor ({p_value_geral:.4f}) é menor que {alpha}. Há autocorrelação significativa (o modelo não capturou toda a informação).")
    
    dropdown.observe(atualizar_teste, names='value')
    
    # Força a execução inicial para o primeiro ticker
    if tickers:
        dropdown.value = tickers[0] 
        display(VBox([dropdown, output]))

def grafico_acf(df: DataFrame, max_lags: int, coluna_valor: str):
    """
    Gera gráfico de autocorrelação (ACF) para uma série temporal, interativamente por ticker.
    Usado para identificar a ordem do componente MA (média móvel) ou a sazonalidade.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'acao' e a coluna numérica de interesse.
        max_lags (int): Número máximo de lags para o gráfico.
        coluna_valor (str): Nome da coluna numérica (ex: 'close', 'y_pred' ou 'resíduos').
    """
    if 'acao' not in df.columns or coluna_valor not in df.columns:
        print(f"Erro: O DataFrame deve conter a coluna 'acao' e a coluna de valor '{coluna_valor}'.")
        return

    tickers = sorted(df['acao'].dropna().unique())
    if not tickers:
        print("Erro: Nenhuma ação válida encontrada na coluna 'acao'.")
        return

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
            plot_acf(serie, lags=max_lags, title=f'ACF - {coluna_valor} | Ação: {acao}')
            plt.tight_layout()
            plt.show()
            plt.close() 

    dropdown.observe(atualizar_acf, names='value')
    
    # Força a execução inicial
    if tickers:
        dropdown.value = tickers[0]  
        display(VBox([dropdown, output]))


def grafico_pacf(df: DataFrame, max_lags: int, coluna_valor: str, metodo: str = 'ywm'):
    """
    Gera gráfico de autocorrelação parcial (PACF) interativo por ticker.
    Usado para identificar a ordem do componente AR (auto-regressivo).

    Args:
        df (pd.DataFrame): DataFrame com colunas 'acao' e a coluna numérica.
        max_lags (int): Número máximo de lags.
        coluna_valor (str): Nome da coluna numérica.
        metodo (str): Método do PACF ('ywm' por padrão, para evitar warnings de séries curtas).
    """
    if 'acao' not in df.columns or coluna_valor not in df.columns:
        print(f"Erro: O DataFrame deve conter a coluna 'acao' e a coluna de valor '{coluna_valor}'.")
        return

    acoes = sorted(df['acao'].dropna().unique())
    if not acoes:
        print("Erro: Nenhuma ação válida encontrada na coluna 'acao'.")
        return

    dropdown = Dropdown(
        options=acoes,
        description='Ação:',
        layout={'width': '300px'}
    )

    output = Output()

    def atualizar_pacf(change):
        output.clear_output(wait=True)
        acao = change['new']
        serie = df[df['acao'] == acao][coluna_valor].dropna()

        with output:
            if serie.empty:
                print(f"Nenhuma série disponível para {acao}")
                return

            # Ajuste de Lags (melhora a robustez em séries curtas)
            limite = len(serie) // 2 - 1
            lags_ajustado = min(max_lags, limite)
            if lags_ajustado < max_lags:
                print(f"[Atenção] Série curta. Reduzindo lags de {max_lags} para {lags_ajustado}.")
                
            if lags_ajustado <= 0:
                 print("Não foi possível calcular o PACF. A série é muito curta ou vazia.")
                 return

            plt.figure(figsize=(10, 4))
            plot_pacf(serie, lags=lags_ajustado, method=metodo, title=f'PACF - {coluna_valor} | Ação: {acao}')
            plt.tight_layout()
            plt.show()
            plt.close()

    dropdown.observe(atualizar_pacf, names='value')
    
    # Força a execução inicial
    if acoes:
        dropdown.value = acoes[0]
        display(VBox([dropdown, output]))

