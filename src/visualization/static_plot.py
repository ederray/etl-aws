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



def grafico_pairplot_target(
    df: DataFrame,
    target: str,
    lista_features: list[str],
    path: str = None,
    tipo="str",
) -> None:
    """Função que retorna um gráfico pairplot das variáveis numéricas correlacionadas com o target indicado.

    params:
    df (DataFrame): Dataframe de entrada.
    target (str): feature alvo da previsão.
    lista_features (List[str]): lista de features para avaliar a correlação dos dados com o target.
    path (str): caminho para salvamento da imagem.
    tipo (str): tipo de gráfico pairplot.


    return:
    None: gráfico pairplot com o target.

    """
    ax = sns.pairplot(data=df, y_vars=target, x_vars=lista_features, kind=tipo)
    ax.figure.suptitle("Gráfico de dispersão das variáveis", y=1.05)
    if path:
        plt.savefig(path)
    return plt.show()


def grafico_coluna(
    df,
    x_col,
    y_col,
    hue_col=None,
    title=None,
    path: str = None,
    palette: str = None,
) -> None:
    """
    Cria um gráfico de colunas com a opção de um hue categórico.

    params:
        df (pd.DataFrame): O DataFrame com os dados.
        x_col (str): O nome da coluna para o eixo X (variável categórica).
        y_col (str): O nome da coluna para o eixo Y (variável numérica).
        hue_col (str, opcional): O nome da coluna para a cor (hue). Padrão é None.
        title (str, opcional): O título do gráfico.
        path (str): caminho para salvamento da imagem.
        palette (str): paleta de cores do gráfico.

    return:
        None: gráfico de colunas.

    """
    # Define o tamanho da figura
    plt.figure(figsize=(10, 6))

    # Cria o gráfico de colunas
    ax = sns.barplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        errorbar=None,  # Remove a barra de erro para simplificar o exemplo
        palette=palette,
    )

    # Adiciona o título, se fornecido
    if title:
        plt.title(title, fontsize=16)

    # Melhora a visualização
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(f"Média de {y_col}", fontsize=12)
    plt.xticks(
        rotation=45, ha="right"
    )  # Rotaciona os rótulos do eixo X para melhor visualização
    plt.tight_layout()  # Ajusta o layout para evitar sobreposições
    if path:
        plt.savefig(path)
    plt.show()


def grafico_replot(
    df: DataFrame,
    x: str,
    y: str,
    col_div: str,
    linha_div: str,
    hue: str,
    tipo: str = "scatter",
    titulo: str = None,
    path: str = None,
) -> None:
    """
    Cria um gráfico relacional (relplot) usando Seaborn, com divisões em linhas e colunas
    para visualização de subgrupos de dados.

    params:
        df (DataFrame): O DataFrame a ser usado para plotagem.
        x (str): A coluna para o eixo x.
        y (str): A coluna para o eixo y.
        col_div (str): A coluna para dividir o gráfico em subplots por coluna.
        linha_div (str): A coluna para dividir o gráfico em subplots por linha.
        hue (str): A coluna para diferenciar as cores dos pontos/linhas.
        tipo (str, opcional): O tipo de gráfico a ser gerado ('scatter' ou 'line').
                               Padrão é 'scatter'.
        titulo (str, opcional): Título principal para o gráfico. Padrão é None.
        path (str, opcional): Caminho completo para salvar a imagem do gráfico. Padrão é None.

    return:
        None: gráfico relacional.
    """
    g = sns.relplot(data=df, x=x, y=y, col=col_div, row=linha_div, hue=hue, kind=tipo)

    if titulo:
        g.fig.suptitle(titulo, fontsize=16, fontweight="bold")
        g.fig.subplots_adjust(top=0.9)

    if path:
        plt.savefig(path)

    plt.show()


def grafico_catplot(
    df: DataFrame,
    x: str,
    y: str,
    col_div: str = None,
    linha_div: str = None,
    hue: str = None,
    tipo: str = "box",
    titulo: str = None,
    path: str = None,
) -> None:
    """
    Cria um gráfico categórico (catplot) usando Seaborn, ideal para visualizações
    que envolvem variáveis categóricas.

    Args:
        df (DataFrame): O DataFrame a ser usado para plotagem.
        x (str): A coluna para o eixo x. Pode ser categórica ou numérica.
        y (str): A coluna para o eixo y. Pode ser categórica ou numérica.
        col_div (str, opcional): A coluna para dividir o gráfico em subplots por coluna.
                                 Padrão é None.
        linha_div (str, opcional): A coluna para dividir o gráfico em subplots por linha.
                                   Padrão é None.
        hue (str, opcional): A coluna para diferenciar as cores das categorias. Padrão é None.
        tipo (str, opcional): O tipo de gráfico categórico a ser gerado
                              (ex: 'box', 'violin', 'swarm', 'bar'). Padrão é 'box'.
        titulo (str, opcional): Título principal para o gráfico. Padrão é None.
        path (str, opcional): Caminho completo para salvar a imagem do gráfico. Padrão é None.

    Returns:
        None: gráfico categórico relacional.
    """
    g = sns.catplot(data=df, x=x, y=y, col=col_div, row=linha_div, hue=hue, kind=tipo)

    if titulo:
        g.fig.suptitle(titulo, fontsize=16, fontweight="bold")
        g.fig.subplots_adjust(top=0.9)

    if path:
        plt.savefig(path)

    plt.show()


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




