"""Funções de visualização de dados no pré-processamento"""
from IPython.display import display
import holidays
from ipywidgets import interact, HTML, Output, Dropdown, VBox
import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# instância do objeto logger
logger = logging.getLogger(__name__)

def boxplot_analise_descritiva(df: DataFrame, distribuicao: list[float]) -> plt.Figure:
    """Constroi um gráfico combinado de histograma das features presentes na análise descritiva."""
    
    try:
        logger.info("Histograma unificado.")
        return df.describe(percentiles=distribuicao).boxplot(figsize=(10,4))

    except Exception as e:
        logger.error(e)


def boxplot_analise_descritiva_categorica(df: pd.DataFrame, distribuicao: list[float], feature: str):
    """
    Exibe um boxplot das features numéricas para cada valor selecionado de uma feature categórica.

    Args:
        df (pd.DataFrame): DataFrame com dados.
        distribuicao (list[float]): Lista de percentis para a análise (não usada diretamente aqui, mas mantida).
        feature (str): Coluna categórica a ser usada para seleção (ex: 'ticker', 'setor', etc.).
    """
    opcoes = sorted(df[feature].dropna().unique())

    @interact(coluna=opcoes)
    def plot(coluna):
        try:
            logger.info(f"Gerando boxplot para {feature} = {coluna}")

            # Filtra os dados com base na seleção
            dados_filtrados = df[df[feature] == coluna]
            dados_numericos = dados_filtrados.select_dtypes(include='number')

            if dados_numericos.empty:
                logger.warning(f"Nenhuma coluna numérica para {feature} = {coluna}")
                print("Nenhum dado numérico disponível.")
                return

            # Aplica minmax_scale
            scaler = StandardScaler()
            norm = PowerTransformer(method='yeo-johnson')
            dados_normalizados = pd.DataFrame(scaler.fit_transform(norm.fit_transform(dados_numericos)), 
                columns=dados_numericos.columns)

            # Cria boxplot com as colunas no eixo X
            plt.figure(figsize=(14, 6))
            dados_normalizados.boxplot()
            plt.xticks(rotation=60)
            plt.title(f"Análise Descritiva - {feature}: {coluna}")
            plt.ylabel("Valor Normalizado (MinMax)")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}")
            print("Erro:", e)


def histograma_feature_categorica(df: pd.DataFrame, feature: str):
    """
    Exibe histogramas das colunas numéricas para os registros filtrados por uma feature categórica.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        feature (str): Coluna categórica usada para filtrar os dados (ex: 'ticker').
    """
    opcoes = sorted(df[feature].dropna().unique())

    @interact(coluna=opcoes)
    def plot(coluna):
        try:
            logger.info(f"Gerando histograma para {feature} = {coluna}")

            # Filtra os dados
            dados_filtrados = df[df[feature] == coluna]
            dados_numericos = dados_filtrados.select_dtypes(include='number')

            transform_box_cox = PowerTransformer(method='yeo-johnson').fit_transform(dados_numericos)

            if dados_numericos.empty:
                logger.warning(f"Nenhuma coluna numérica para {feature} = {coluna}")
                print("Nenhum dado numérico disponível.")
                return

            # Normaliza com MinMaxScaler
            normalizado = pd.DataFrame(
                StandardScaler().fit_transform(transform_box_cox),
                columns=dados_numericos.columns
            )

            # Define layout dos subplots
            num_colunas = len(normalizado.columns)
            cols = 3  # Número de colunas de plots por linha
            rows = int(np.ceil(num_colunas / cols))

            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axs = axs.flatten()  # Para indexar linearmente

            for i, col in enumerate(normalizado.columns):
                axs[i].hist(normalizado[col], color='steelblue', alpha=0.7)
                axs[i].set_title(col)
                axs[i].set_xlabel('Valor Normalizado')
                axs[i].set_ylabel('Frequência')

            # Remove eixos não utilizados
            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(f"Distribuição de Features Numéricas - {feature}: {coluna}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}")
            print("Erro:", e)


def grafico_acf(coluna_target:Series, n_lag:int):
    """Função para gerar o gráfico de autocorrelação"""
    plot_acf(coluna_target, lags=n_lag, title=f'Autocorrelação de {n_lag}lags') 
    return plt.show()

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

def grafico_pacf(coluna_target:Series, n_lag:int, metodo:str='ywm'):
    """Função para gerar o gráfico de autocorrelação parcial."""
    plot_pacf(coluna_target, lags=n_lag, method=metodo)  # método estável para séries reais
    return plt.show()

def grafico_acf_interativo(df: pd.DataFrame, max_lags: int, coluna_valor: str = 'close'):
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

def grafico_pacf_interativo(df: pd.DataFrame, max_lags: int, coluna_valor: str = 'close', metodo: str = 'ywm'):
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

def testar_estacionariedade_interativo(df: DataFrame, coluna_valor: str = 'close'):
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


def grafico_correlacao(corr):
    # transforma os valores em uma matriz booleana para indicar presença dos dados
    mask =  np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True) 
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)
    plt.show()
