"""Arquivo de funções geradoras de gráficos estáticos"""

import logging
import math
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sidetable as stb
import statsmodels.stats.multicomp as mc
from ipywidgets import interact
from pandas import DataFrame, Series
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

sns.set_style("darkgrid")


# instância do objeto logger
logger = logging.getLogger(__name__)




def grafico_dispersao(
    df: DataFrame,
    y: Series,
    x: Series,
    titulo: str,
    xlabel: str,
    ylabel: str,
    res: bool = None,
    regr: bool = None,
    hue: str = None,
    size: str = None,
    path: str = None,
    ax=None,
) -> None:
    """
    Gera um gráfico de dispersão para comparar.

    params:

        df (DataFrame): Dataframe de entrada.
        y (Series): coluna de valores do eixo y.
        x (Series): coluna de valores do eixo x.
        title (str): O título do gráfico.
        xlabel (str): O título do do eixo x.
        ylabel (str): O título do eixo y.
        res (bool)=True: adiciona linha horizontal y=0.
        regr (bool) = True: adiciona a linha de regressão do
        hue (str): coluna para gerar categorização da dispersão dos dados.
        size (str): coluna para definir o tamanho da dispersão dos dados.
        path (str): caminho para salvamento da imagem.
        ax (matplotlib axis): Eixo para plotar. Se None, cria novo.

    return:
        None: gráfico dispersão.

    """

    was_called_alone = ax is None

    try:
        if was_called_alone:
            fig, ax = plt.subplots(figsize=(10, 5))

        if not regr:
            sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, legend="full", ax=ax)
        else:
            sns.regplot(data=df, x=x, y=y, line_kws={"color": "red"}, ax=ax)

        if res and not regr:
            ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)

        if was_called_alone:
            fig.tight_layout()

            if path:
                fig.savefig(path)

            plt.show()
            plt.close(fig)

    except Exception as e:
        logger.error(f"Erro ao gerar o gráfico de dispersão: {e}")
        return




