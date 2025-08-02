"""Funções de tratamento dos dados"""
import logging
import holidays
from pandas import DataFrame, Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

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


def agrupar_dados(df: DataFrame, colunas: list, agr) -> DataFrame:
    """Função que agrupa as colunas para montagem do dataset."""
    df = df.groupby(by=colunas).agg(agr)
    return df

def media_movel(df: DataFrame, coluna: str, p: list[int]) -> DataFrame:
    """Função que gera coluna de média móvel"""
    # verifica o comprimento da lista para seguir o processo.
    if len(p) > 1:
        for valor in p:
            df[f'{coluna}_rm_{valor}'] = df[coluna].rolling(valor).mean()
    else:
        df[f'{coluna}_rm_{p}'] = df[coluna].rolling(p).mean()
    return df


def lag_data(df: DataFrame, coluna: str, lag: list[int]) -> DataFrame:
    """Função que gera coluna de lags"""
    # verifica o comprimento da lista para seguir o processo.
    if len(lag) > 1:
        for valor in lag:
            df[f'{coluna}_lag_{valor}'] = df[coluna].shift(valor)
    else:
        df[f'{coluna}_lag_{lag}'] = df[coluna].shift(lag)
    return df

def dados_temporais(df: DataFrame) -> DataFrame:
    """Função que insere colunas com dados temporais a partir do index do Dataframe"""
    df['Dia_Semana'] = df.index.day
    df['Mês'] = df.index.month

    # criação do objeto com os feriados brasileiros
    br_holidays = holidays.BR()
    df['Feriado'] = df.index.to_series().apply(lambda x: x in br_holidays)

    return df

def grafico_decomposicao_temporal(df: DataFrame, target: str, n: int):
    """Função para construção de uma decomposição de série temporal."""
    decomposicao = seasonal_decompose(df[target], model='additive', period=n)
    decomposicao.plot()
    plt.tight_layout()
    return plt.show()

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

def grafico_acf(coluna_target:Series, n_lag:int):
    """Função para gerar o gráfico de autocorrelação"""
    plot_acf(coluna_target, lags=n_lag, title=f'Autocorrelação de {n_lag}lags') 
    return plt.show()


def grafico_pacf(coluna_target:Series, n_lag:int, metodo:str='ywm'):
    """Função para gerar o gráfico de autocorrelação parcial."""
    plot_pacf(coluna_target, lags=n_lag, method=metodo)  # método estável para séries reais
    return plt.show()