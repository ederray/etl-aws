"""Arquivo de funções geradoras de gráficos interativos"""

import logging
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from ipywidgets import interact
from pandas import DataFrame, Series
from scipy import stats
from sklearn.preprocessing import scale
import math
sns.set_style("darkgrid")


# instância do objeto logger
logger = logging.getLogger(__name__)

def grafico_boxplot(
    df: DataFrame, interativo: bool = None, cat_col: str = None, path: str = None, scaler=True, 
) -> None:
    """
    Cria um gráfico com múltiplos subplots, onde cada subplot exibe um boxplot
    de uma coluna numérica, agrupado pelas categorias da feature indicada.

    params:
        df (DataFrame): DataFrame com os dados.
        cat_col (str): O nome da coluna categórica para agrupar os dados (ex: 'room_type').
        path (str): caminho para salvamento da imagem.

    return:
        None: Gráfico box-plot geral ou interativo com filtro a partir de uma feature categórica
    """
    try:

        def plot(df:DataFrame, scaler=None, title=str):

            df_numeric = df.select_dtypes(include='number')
            plt.figure(figsize=(14, 7))
            if scaler:
                data_to_plot = df_numeric.apply(scale)
            else:
                data_to_plot = df_numeric
            sns.boxplot(data=data_to_plot)
            plt.xticks(rotation=60)
            plt.title(title)
            plt.tight_layout()
            if path:
                plt.savefig(path)

            return plt.show()


        if not interativo:
            plot(df, scaler=scaler, title='Gráfico boxplot')
            
        else:
        
            opcoes = sorted(df[cat_col].dropna().unique())

            @interact(filtro=opcoes)
            def _plot(filtro):
                plot(df=df[df[cat_col] == filtro], scaler=scaler,title=f"Gráfico boxplot interativo pela feature: ticker")


    except Exception as e:
        logger.error(f"Erro: {e}")


def grafico_qq_plot(
    df: pd.DataFrame, interativo: bool = False, feature: str = None, path: str = None
) -> None:
    """
    Gera gráficos Q-Q Plot para as colunas numéricas.
    Se interativo=True e col_cat for informado, permite filtrar por valores dessa coluna.

    params:
        df (DataFrame): DataFrame de entrada.
        feature (str): Nome da coluna categórica para agrupar (modo interativo).
        interativo (bool): Se True, a análise é por categoria com um widget. Se False, a análise é geral.
        path (str): caminho para salvamento da imagem.

    return:
        None: gráfico qq-plot.

    """
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        def plot(dataframe):
            nrows = 3
            ncols = int(np.ceil(len(numeric_cols) / nrows))
            _, axs = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows)
            )
            axs = axs.ravel()

            for i, col in enumerate(numeric_cols):
                data = dataframe[col].dropna()
                stats.probplot(data, dist="norm", plot=axs[i])
                axs[i].set_title(f"Q-Q Plot: {col}")
                axs[i].set_xlabel("Quantis Teóricos")
                axs[i].set_ylabel("Quantis da Amostra")
            for j in range(len(numeric_cols), len(axs)):
                axs[j].axis("off")

            plt.suptitle("Q-Q Plots Colunas Numéricas", fontsize=20, y=1.02)
            plt.tight_layout()
            if path:
                plt.savefig(path)
            plt.show()

        if interativo:
            opcoes = sorted(df[feature].dropna().unique())

            @interact(filtro=opcoes)
            def _plot(filtro):
                plot(df[df[feature] == filtro])

        else:
            plot(df)

    except Exception as e:
        logger.error(f"Erro: {e}")


def grafico_heatmap(
    df: pd.DataFrame, interativo: bool = False, feature: str = None, path: str = None
) -> None:
    """
    Cria heatmap de correlação dos dados numéricos.
    Se interativo=True e feature informado, cria heatmap filtrado por valores dessa coluna.

    params:
        df (DataFrame): DataFrame de entrada.
        feature (str): Nome da coluna categórica para agrupar (modo interativo).
        interativo (bool): Se True, a análise é por categoria com um widget. Se False, a análise é geral.
        path (str): caminho para salvamento da imagem.

    return:
        None: gráfico heatmap.
    """
    try:

        def plot(dataframe, titulo_extra=""):
            df_corr = dataframe.select_dtypes(include="number").corr()
            plt.figure(figsize=(10, 7))
            mask = np.triu(df_corr)
            sns.heatmap(df_corr, linewidths=0.5, cmap="vlag", mask=mask, annot=True)
            plt.title(f"Heatmap Correlação {titulo_extra}", fontsize=14,)
            if path:
                plt.savefig(path)
            plt.show()

        if interativo and feature and feature in df.columns:
            opcoes = sorted(df[feature].dropna().unique())

            @interact(filtro=opcoes)
            def _plot(filtro):
                plot(df[df[feature] == filtro], f"- {feature}: {filtro}")

        else:
            plot(df)

    except Exception as e:
        logger.error(f"Erro: {e}")



def grafico_histograma(
    df: pd.DataFrame,
    interativo: bool = False, 
    cat_col: str = None,    
    path: str = None,
    titulo: str = None,
    scaler=False, 
    ax=None,
) -> None:
    """
    Exibe histogramas de uma ou mais colunas numéricas em grid de 2 colunas.
    Pode ser usado no modo interativo para filtrar por uma coluna categórica.

    params:
        df (DataFrame): DataFrame de entrada.
        feature (str | list[str]): Nome da coluna ou lista de colunas a serem plotadas.
        interativo (bool): Se True, usa o widget @interact para filtrar por cat_col.
        cat_col (str): Nome da coluna categórica para filtro no modo interativo.
        path (str): Caminho para salvar a imagem (opcional).
        titulo (str): Título geral (usado apenas quando múltiplos plots).
        ax (matplotlib axis): Eixo para plotar. Se None, cria novo.

    return:
        None: gráfico histograma.
    """
    try:
 
        def _plot_hist(df_to_plot: DataFrame, scaler=scaler):
            
            cols_numericas = df_to_plot.select_dtypes(include='number')
            n_features = cols_numericas.shape[1]
            n_cols = 2
            n_rows = math.ceil(n_features / n_cols)
            
            fig, axes = None, None
            if ax is None:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
                axes = axes.flatten()
            else:
                axes = [ax]
            
            for i, col in enumerate(cols_numericas.columns):
    
                if col in cols_numericas.columns:

                    if scaler:
                        sns.histplot(cols_numericas[[col]].apply(scale), color="steelblue", alpha=0.7, ax=axes[i], kde=True)
                        axes[i].set_title(f"{col}", fontsize=12)
                    else:
                        sns.histplot(cols_numericas[col], color="steelblue", alpha=0.7, ax=axes[i], kde=True)
                        axes[i].set_title(f"{col}", fontsize=12)

            if ax is None:
                for j in range(n_features, len(axes)):
                    axes[j].set_visible(False)

            if titulo:
                fig.suptitle(f"{titulo}", fontsize=16, y=1.02)

            plt.tight_layout()

            if path:
                plt.savefig(path, bbox_inches="tight")

            if ax is None:
                plt.show()


        if not interativo:
            _plot_hist(df, scaler=False)

        else:
            
            opcoes = sorted(df[cat_col].unique().tolist())
            @interact(filtro=opcoes)
            def _plot_interativo(filtro):
                _plot_hist(df, scaler=True)
                
    except Exception as e:
        print(f"Erro ao gerar o gráfico: {e}")


def grafico_lineplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = None,  
    hue_col: str = None, 
    interativo: bool = False,
    cat_col: str = None,
    path: str = None,
    titulo: str = None,
) -> None:
    """
    Exibe um line plot com opção de modo interativo para selecionar as features.

    params:
        df (DataFrame): DataFrame de entrada.
        x_col (str): Nome da coluna para o eixo X (geralmente a coluna temporal, é fixo).
        y_col (str, opcional): Nome da coluna para o eixo Y. Obrigatório no modo estático.
        hue_col (str, opcional): Nome da coluna que define as diferentes linhas de agrupamento.
        cat_col (str): Nome da coluna categórica para filtro no modo interativo.
        interativo (bool): Se True, ativa a interface com Dropdown para y_col e hue_col.
        path (str, opcional): Caminho para salvar a imagem (apenas modo estático).
        titulo (str, opcional): Título principal do gráfico.

    return:
        None: gráfico de linha (estático ou interativo).
    """

    def _plot_line(df_plot, x, y, hue, titulo, path=None):
        plt.figure(figsize=(12, 4))
        
        label_x = x.replace("_", " ").title() if isinstance(x, str) else str(x)

        sns.lineplot(
            data=df_plot, x=x, y=y, hue=hue, marker="o", dashes=False
        )

        final_title = titulo if titulo else f"Série temporal da coluna {y}"
        plt.title(final_title, fontsize=16, pad=20)
        plt.xlabel(label_x, fontsize=12) 
        plt.ylabel(y, fontsize=12)

        if hue:
            plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc=2)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        else:
            plt.tight_layout()

        if path:
            plt.savefig(path, bbox_inches="tight")
        
        plt.show()
        plt.close()

    try:
  
        if not interativo:
            
            df_plot = df.copy()
            if x_col not in df_plot.columns and df_plot.index.name == x_col:
                df_plot = df_plot.reset_index()

            _plot_line(df_plot, x_col, y_col, hue_col, titulo, path)

        else:

            opcoes = sorted(df[cat_col].dropna().unique().tolist())

            @interact(filtro=opcoes)
            def plot(filtro):
                df_filtrado = df[df[cat_col] == filtro].copy()

                if isinstance(df_filtrado.index, pd.DatetimeIndex):
                    df_filtrado = df_filtrado.sort_index()
                    _plot_line(
                        df_filtrado.reset_index(),
                        x=df_filtrado.index.name or x_col,
                        y=y_col,
                        hue=hue_col,
                        titulo=f"{titulo or 'Série temporal'} | {cat_col}: {filtro}",
                        path=None
                    )
                else:
                
                    if x_col not in df_filtrado.columns:
                        df_filtrado = df_filtrado.reset_index(names=[x_col])

                    _plot_line(
                        df_filtrado,
                        x_col,
                        y_col,
                        hue_col,
                        titulo=f"{titulo or 'Série temporal'} | {cat_col}: {filtro}",
                        path=None
                    )
    except Exception as e:
        print(f"Erro: {e}")