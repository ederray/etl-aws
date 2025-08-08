"""Funções de tratamento dos dados"""
from IPython.display import display
import holidays
from ipywidgets import interact, HTML, Output, Dropdown, VBox
import logging
import matplotlib.pyplot as plt
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



def boxplot_analise_descritiva(df: DataFrame, distribuicao: list[float]) -> plt.Figure:
    """onstroi um gráfico combinado de histograma das features presentes na análise descritiva."""
    
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



def tratamento_nulo_dados_setor_industria(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """
    Aplica interpolação em cascata (Indústria -> Setor -> Geral) para preencher valores nulos
    em colunas numéricas, ideal para a camada Refined de um pipeline de dados financeiros.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'ticker', 'setor', 'industria' e colunas numéricas.
                           O índice deve ser do tipo DatetimeIndex ordenado.
        colunas (list[str]): Lista de nomes das colunas numéricas a interpolar.

    Returns:
        pd.DataFrame: DataFrame com os valores nulos preenchidos.
    """
    df_transf = df.copy()

    try:
        logger.info("Iniciando a preparação do dataset para interpolação.")

        if not isinstance(df_transf.index, pd.DatetimeIndex):
            raise ValueError("O índice do DataFrame deve ser um DatetimeIndex para interpolação temporal.")

        df_transf = df_transf.reset_index().sort_values(by=['ticker', 'Date']).set_index('Date')


        empresas_por_industria = (
            df_transf.groupby(['setor', 'industria'])['ticker']
            .nunique()
            .reset_index(name='count_ticker_industria')
        )
        empresas_por_setor = (
            df_transf.groupby('setor')['ticker']
            .nunique()
            .reset_index(name='count_ticker_setor')
        )
        industrias_por_setor = (
            df_transf.groupby('setor')['industria']
            .nunique()
            .reset_index(name='count_industria_setor')
        )

        # reset de index para preservação da feature de data
        df_transf = df_transf.reset_index()

        df_transf = df_transf.merge(empresas_por_industria, on=['setor', 'industria'], how='left')
        df_transf = df_transf.merge(empresas_por_setor, on='setor', how='left')
        df_transf = df_transf.merge(industrias_por_setor, on='setor', how='left')

        logger.info("Contagens por setor e indústria adicionadas ao DataFrame.")

    except Exception as e:
        logger.error(f"Erro na fase de preparação: {e}")
        raise

    try:
        for coluna in colunas:
            logger.info(f"Processando coluna: {coluna}")

            # Preenchimento inicial
            df_transf[coluna] = df_transf.groupby('ticker')[coluna].ffill()

            # Regra 1: Interpolação por indústria
            cond_industria = df_transf['count_ticker_industria'] > 1
            df_ind = df_transf[cond_industria].copy()
            df_ind_interp = (
                df_ind.groupby(['setor', 'industria'])[coluna]
                .apply(lambda x: x.sort_index().interpolate(method='polynomial', order=2))
                .reset_index(level=[0, 1], drop=True)
            )
            df_transf.loc[df_ind_interp.index, coluna] = df_ind_interp

            # Regra 2: Interpolação por setor
            cond_setor = (
                (df_transf['count_ticker_industria'] == 1) &
                (df_transf['count_industria_setor'] > 1)
            )
            df_set = df_transf[cond_setor].copy()
            df_set_interp = (
                df_set.groupby('setor')[coluna]
                .apply(lambda x: x.sort_index().interpolate(method='polynomial', order=2))
                .reset_index(level=0, drop=True)
            )
            df_transf.loc[df_set_interp.index, coluna] = df_set_interp

            # Regra 3: Interpolação geral
            cond_geral = (
                (df_transf['count_ticker_industria'] == 1) &
                (df_transf['count_industria_setor'] == 1)
            )
            df_geral = df_transf[cond_geral].copy()
            df_geral_interp = (
                df_geral[coluna]
                .sort_index()
                .interpolate(method='polynomial', order=2)
            )
            df_transf.loc[df_geral_interp.index, coluna] = df_geral_interp

        # ffill e bfill finais
        logger.info("Aplicando ffill() e bfill() finais.")
        for coluna in colunas:
            df_transf[coluna] = df_transf.groupby('ticker')[coluna].ffill()
            df_transf[coluna] = df_transf.groupby('ticker')[coluna].bfill()

        # Fallback final
        for coluna in colunas:
            if df_transf[coluna].isna().any():
                logger.warning(f"Ainda há NaNs em '{coluna}'. Preenchendo com a média global.")
                media = df_transf[coluna].mean()
                df_transf[coluna].fillna(media, inplace=True)

        # filtragem das colunas construídas no processo de tratamento dos dados nulos
        df_transf.drop(columns=['count_ticker_industria','count_ticker_setor','count_industria_setor'], inplace=True)
        # retorno do datetimeIndex
        df_transf = df_transf.set_index('Date')

        logger.info("Preenchimento de nulos concluído com sucesso.")
        return df_transf

    except Exception as e:
        logger.error(f"Erro durante interpolação de valores: {e}")
        raise

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


def diferenciar_serie_temporal(df: pd.DataFrame, target:str) -> pd.DataFrame:
    """
    Aplica diferenciação de primeira ordem na coluna 'Close' agrupando por 'ticker'.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'ticker' e 'Close'.

    Returns:
        pd.DataFrame: DataFrame com nova coluna 'Close_diff'.
    """
    df = df.copy()
    df['Close_diff'] = df.groupby('ticker')[target].diff()
    df.dropna(inplace=True)
    return df


def grafico_acf(coluna_target:Series, n_lag:int):
    """Função para gerar o gráfico de autocorrelação"""
    plot_acf(coluna_target, lags=n_lag, title=f'Autocorrelação de {n_lag}lags') 
    return plt.show()


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
    tickers = sorted(df['ticker'].dropna().unique())

    dropdown = Dropdown(
        options=tickers,
        description='Ticker:',
        layout={'width': '300px'}
    )

    output = Output()

    def atualizar_pacf(change):
        output.clear_output(wait=True)
        ticker = change['new']
        serie = df[df['ticker'] == ticker][coluna_valor].dropna()

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





