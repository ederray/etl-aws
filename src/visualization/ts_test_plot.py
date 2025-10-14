"""Funções de criação de gráficos para análise de validação de modelos"""
import logging
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from ipywidgets import Dropdown, VBox, Output, interact, HTML
from statsmodels.tsa.stattools import adfuller
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
      
        df_acao = df[df['acao'] == ticker].dropna(subset=['y_real', 'y_pred'])
        residuos = df_acao['y_real'] - df_acao['y_pred']

        with output:
            if residuos.empty:
                print(f"Nenhuma série de resíduos disponível para {ticker}")
                return
            
      
            ljung_box = acorr_ljungbox(residuos, lags=lags, return_df=True)
            ljung_box["Rejeita_H0"] = ljung_box["lb_pvalue"] < alpha
            
            print(f"Resultados do Teste de Ljung-Box para a Ação: {ticker}\n")
            display(ljung_box)
            print("---")
            
            p_value_geral = ljung_box.iloc[-1]["lb_pvalue"]
            if p_value_geral > alpha:
                print(f"O p-valor ({p_value_geral:.4f}) é maior que {alpha}. Não há autocorrelação significativa (indicativo de bom modelo).")
            else:
                print(f"O p-valor ({p_value_geral:.4f}) é menor que {alpha}. Há autocorrelação significativa (o modelo não capturou toda a informação).")
    
    dropdown.observe(atualizar_teste, names='value')
    
    if tickers:
        dropdown.value = tickers[0] 
        display(VBox([dropdown, output]))

def grafico_acf_interativo(df: DataFrame, max_lags: int, coluna_valor: str = 'close'):
    """
    Gera gráfico de autocorrelação (ACF) para uma série temporal, interativamente por ticker.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'ticker' e a coluna numérica de interesse.
        max_lags (int): Número máximo de lags para o gráfico.
        coluna_valor (str): Nome da coluna numérica (ex: 'close', 'trailingPE', etc.).
    """
    tickers = sorted(df['ticker'].dropna().unique())

    dropdown = Dropdown(
        options=tickers,
        description='Ticker:',
        layout={'width': '300px'}
    )

    output = Output()

    def atualizar_acf(change):
        output.clear_output(wait=True)
        ticker = change['new']

        serie = df[df['ticker'] == ticker][coluna_valor].dropna()

        with output:
            if serie.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return

            plt.figure(figsize=(10, 4))
            plot_acf(serie, lags=max_lags)
            plt.title(f'ACF - {coluna_valor} | Ticker: {ticker}')
            plt.tight_layout()
            plt.show()
            plt.close()  # <- fecha a figura para evitar acumulação

    dropdown.observe(atualizar_acf, names='value')
    dropdown.value = tickers[0]  # força exibição inicial

    display(VBox([dropdown, output]))

def grafico_pacf_interativo(df: DataFrame, max_lags: int, coluna_valor: str = 'close', metodo: str = 'ywm'):
    """
    Gera gráfico de autocorrelação parcial (PACF) interativo por ticker.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'ticker' e a coluna numérica.
        max_lags (int): Número máximo de lags.
        coluna_valor (str): Nome da coluna numérica.
        metodo (str): Método do PACF ('ywm' por padrão).
    """
    tickers = sorted(df['acao'].dropna().unique())

    dropdown = Dropdown(
        options=tickers,
        description='Ação:',
        layout={'width': '300px'}
    )

    output = Output()

    def atualizar_pacf(change):
        output.clear_output(wait=True)
        ticker = change['new']
        serie = df[df['acao'] == ticker][coluna_valor].dropna()

        with output:
            if serie.empty:
                print(f"Nenhuma série disponível para {ticker}")
                return

            limite = len(serie) // 2
            lags_ajustado = min(max_lags, limite)
            if lags_ajustado < max_lags:
                print(f"[Atenção] Série muito curta. Reduzindo lags de {max_lags} para {lags_ajustado}.")

            plt.figure(figsize=(10, 4))
            plot_pacf(serie, lags=lags_ajustado, method=metodo)
            plt.title(f'PACF - {coluna_valor} | Ticker: {ticker}')
            plt.tight_layout()
            plt.show()
            plt.close()

    dropdown.observe(atualizar_pacf, names='value')
    dropdown.value = tickers[0]  # força execução inicial

    display(VBox([dropdown, output]))



def teste_estacionariedade_interativo(df: DataFrame, coluna_valor: str = 'close'):
    """
    Interface interativa para testar estacionariedade de uma série temporal por ticker.
    """
    tickers = sorted(df['ticker'].dropna().unique())
    
    # Cria um widget HTML vazio para exibir o resultado
    resultado_html = HTML()

    def analisar(ticker):
        serie = df[df['ticker'] == ticker][coluna_valor]
        resultado = adfuller(serie.dropna())

        # Constrói o texto do resultado como uma string formatada
        resultado_string = f"""
        <p>🔍 <strong>Teste ADF - {coluna_valor} | Ticker: {ticker}</strong></p>
        <ul>
            <li>ADF Statistic: {resultado[0]:.4f}</li>
            <li>p-value: {resultado[1]:.4f}</li>
        """
        for k, v in resultado[4].items():
            resultado_string += f"<li>Critério {k}%: {v:.4f}</li>"
        
        if resultado[1] < 0.05:
            resultado_string += "</ul><p>✅ Série estacionária (rejeita H₀)</p>"
        else:
            resultado_string += "</ul><p>⚠️ Série NÃO estacionária (não rejeita H₀)</p>"

        # Atualiza o conteúdo do widget HTML
        resultado_html.value = resultado_string

    # Exibe o widget interativo e o widget HTML
    interact(analisar, ticker=tickers)
    display(resultado_html)

