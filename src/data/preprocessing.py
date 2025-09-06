"""Funções de tratamento dos dados"""
from IPython.display import display
import holidays
from ipywidgets import interact, HTML, Output, Dropdown, VBox
import logging
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# instância do objeto logger
logger = logging.getLogger(__name__)

def amostra_dados(df: DataFrame) -> DataFrame:
    """Função para retornar a amostragem dos dados"""
    return df.sample(3)


def remover_duplicados(df: DataFrame, coluna: str) -> DataFrame:
    """Função para remoção de valores duplicados."""
    df.drop_duplicates(subset=[coluna], keep='first', inplace=True)
    return df


def selecao_colunas(df: DataFrame, colunas: list) -> DataFrame:
    """Função que seleciona as colunas para montagem do dataset"""
    return df[colunas]


def agrupar_dados(df: DataFrame, cols_agrup: list, cols_filter: list=None, agr=None) -> DataFrame:
    """Função que agrupa as colunas para montagem do dataset."""
    try:
        if not cols_filter:
            logger.info(f'Agrupamento selecionado: {cols_agrup}, método: {agr}')
            df = df.groupby(by=cols_agrup).agg(agr)
        else:
            logger.info(f'Agrupamento selecionado: {cols_agrup}, filtragem dataset:{cols_filter}, método: {agr}')
            df = df.groupby(by=cols_agrup)[cols_filter].agg(agr)

    except Exception as e:
        logger.error(e)

    return df


def dados_temporais(df: DataFrame) -> DataFrame:
    """Função que insere colunas com dados temporais a partir do index do Dataframe"""
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    # criação do objeto com os feriados brasileiros
    br_holidays = holidays.BR()
    df['Feriado'] = df.index.to_series().apply(lambda x: x in br_holidays)

    return df


def transformacao_ciclica(df: DataFrame, dias_uteis:bool=False) -> DataFrame:
    """Transformação cíclica"""
    
    try:
        if not dias_uteis:
            logger.info(f'Transformação cíclica para as colunas de dados temporais.')
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


        else:
            logger.info(f'Transformação cíclica com dias úteis para as colunas de dados temporais.')
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    except Exception as e:
        logger.error(e)
    return df

def grafico_decomposicao_temporal_interativo(df: pd.DataFrame, target: str, period: int = 7):
    """
    Cria um gráfico de decomposição temporal interativo, selecionando o ticker.

    Args:
        df (pd.DataFrame): DataFrame contendo a série temporal com índice de datas.
        target (str): Nome da coluna de valores (por exemplo, 'Close').
        period (int): Período para decomposição sazonal (ex: 7 dias, 12 meses, etc).
    """

    tickers = sorted(df['ticker'].dropna().unique())
    
    dropdown = Dropdown(
        options=tickers,
        description='Ticker:',
        layout={'width': '300px'}
    )

    output = Output()

    def atualizar_grafico(change):
        output.clear_output(wait=True)

        ticker = change['new']
        serie = df[df['ticker'] == ticker].sort_index()

        with output:
            if serie.empty:
                print(f"Nenhum dado encontrado para {ticker}")
                return
            
            if target not in serie.columns:
                print(f"Coluna {target} não encontrada.")
                return

            try:
                resultado = seasonal_decompose(serie[target], model='additive', period=period)
                fig = resultado.plot()
                fig.suptitle(f'Decomposição Temporal: {ticker}', fontsize=14)
                fig.tight_layout()
                plt.show()
                plt.close(fig)

            except Exception as e:
                print(f"Erro na decomposição: {e}")

    # Conecta o dropdown ao handler
    dropdown.observe(atualizar_grafico, names='value')

    # Força render inicial com o primeiro ticker
    dropdown.value = tickers[0]

    display(VBox([dropdown, output]))



def testar_estacionariedade(serie, nome="Série"):
    """Função para análise de estacionriedade da série de dados"""
    resultado = adfuller(serie.dropna())
    print(f"\n🔍 Teste ADF - {nome}")
    print(f"ADF Statistic: {resultado[0]:.4f}")
    print(f"p-value: {resultado[1]:.4f}")
    for k, v in resultado[4].items():
        print(f"Critério {k}%: {v:.4f}")
    
    if resultado[1] < 0.05:
        print("✅ Série estacionária (rejeita H₀)")
    else:
        print("⚠️ Série NÃO estacionária (não rejeita H₀)")


def diferenciar_serie_temporal(df: DataFrame, target: str) -> DataFrame:
    """
    Aplica diferenciação de primeira ordem na coluna target agrupando por 'ticker'.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'ticker' e 'Close'.

    Returns:
        pd.DataFrame: DataFrame com nova coluna 'target_diff'.
    """
    df = df.copy()
    df[f'{target}_diff'] = df.groupby('ticker')[target].diff()
    return df

def gerar_features_temporais(df: pd.DataFrame, 
                              coluna_valor: str,
                              lags: list = [1,3,5,7,15,30,60,90],
                              janelas_rolling: list = [3,5,7,15,30,60,90],
                              grupo: str = 'ticker') -> pd.DataFrame:
    """
    Gera lags, médias móveis e volatilidades para séries temporais diferenciadas.

    Args:
        df (pd.DataFrame): DataFrame com coluna temporal e coluna de valor diferenciada.
        coluna_valor (str): Nome da coluna diferenciada (ex: 'close_diff').
        lags (list): Lista de lags desejados (ex: [1, 2, 3]).
        janelas_rolling (list): Janelas para rolling mean e std.
        grupo (str): Nome da coluna de identificação da série (ex: 'ticker').

    Returns:
        pd.DataFrame: DataFrame com novas features adicionadas.
    """
    df = df.copy()
    df = df.reset_index()
    df = df.sort_values(by=[grupo,'Date'])
    df = df.set_index('Date')

    for lag in lags:
        df[f'lag_{lag}_{coluna_valor}'] = df.groupby(grupo)[coluna_valor].shift(lag)

    for janela in janelas_rolling:
        df[f'rolling_mean_{janela}_{coluna_valor}'] = df.groupby(grupo)[coluna_valor].transform(lambda x: x.shift(1).rolling(janela).mean())
        df[f'volatility_{janela}_{coluna_valor}'] = df.groupby(grupo)[coluna_valor].transform(lambda x: x.shift(1).rolling(janela).std())
        df[f'retorno_acumulado_{janela}_{coluna_valor}'] = df.groupby(grupo)[coluna_valor].transform(lambda x: x.shift(1).rolling(janela).sum())

    return df


# novas features técnicas
def calcular_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calcular_macd(df, fast_period=12, slow_period=26, signal_period=9):
    exp1 = df['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

