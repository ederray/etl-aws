"""Arquivo de funções e classes de pré-processamento dos dados"""

import logging
from typing import Optional

import holidays
import numpy as np
from pandas import DataFrame

# instância do objeto logger
logger = logging.getLogger(__name__)


def dados_temporais(df: DataFrame, data: str) -> DataFrame:
    """
    Função que insere colunas com dados temporais a partir do index do Dataframe.

    df (DataFrame): DataFrame de entrada.
    data (str): Coluna de data.

    return:
        DataFrame: DataFrame com as colunas dayofweek, months, weekend e feriado.

    """
    df["dayofweek"] = df[data].dt.dayofweek
    df["month"] = df[data].dt.month
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)

    # criação do objeto com os feriados brasileiros
    br_holidays = holidays.BR()
    df["Feriado"] = df[data].apply(lambda x: x in br_holidays)

    return df


def transformacao_ciclica(df: DataFrame, dias_uteis: bool = False) -> DataFrame:
    """
    Função que realiza a transformação cíclica das colunas de tempo para correção de escala do efeito de fim dos períodos,
    adicionando uma escala cíclica orientada pelas funções de seno e cosseno.

    params:

        df (DataFrame): DataFrame de entrada.
        dias_uteis (bool): Define se a transformação será orientada por dias úteis ou da semana.

    return:
        DataFrame: DataFrame com as colunas day_sin, day_cos, month_sin, month_cos.
    """

    try:
        if not dias_uteis:
            logger.info(f"Transformação cíclica para as colunas de dados temporais.")
            df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        else:
            logger.info(
                f"Transformação cíclica com dias úteis para as colunas de dados temporais."
            )
            df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 5)
            df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 5)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    except Exception as e:
        logger.error(e)
    return df
