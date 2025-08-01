"""Funções de webscrapping"""
import os
import pandas as pd
import logging
import yfinance as yf
import time
import requests
from datetime import datetime
from src.config.logging_config import setup_logging 
from src.config.constants import CAMPOS_FUNDAMENTALISTAS_SELECIONADOS 
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, wait_exponential

# instância do objeto logger
logger = logging.getLogger(__name__)

# construção de diretório e arquivo cache para o diretório /tmp/
os.makedirs("/tmp/py-yfinance-cache", exist_ok=True)
yf.set_tz_cache_location("/tmp/py-yfinance-cache")

# configuração de proxy para conexão com a página
proxy_url = os.environ.get("HTTP_PROXY")
proxies = None
if proxy_url:
    proxies = {
        "http": proxy_url,
        "https": proxy_url,
    }
    logger.info(f"Proxy configurado na URL: {proxy_url}")
else:
    logger.warning("Nenhuma variável de ambiente 'HTTP_PROXY' encontrada. Requisições yfinance não usarão proxy.")

# configuração da sessão requests
requests_session = requests.Session()
if proxies:
    requests_session.proxies.update(proxies)

# decorador de retry para a função de obtenção de dados do yfinance
@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception),
       reraise=True)
def _get_yfinance_data_with_retry(ticker_symbol, period, session_object):
    """Função auxiliar para obter dados do yfinance com retries."""
    logger.info(f"Tentando obter dados para {ticker_symbol} (tentativa retry)...")
    ticker = yf.Ticker(ticker_symbol, session=session_object)
    historico = ticker.history(period=period)
    info = ticker.info
    
    # verifica o conteúdo retornado para forçar o retry
    if historico.empty and not info: 
        raise ValueError(f"No data or info found for {ticker_symbol}, forcing retry.")

    return historico, info


def gerar_tabela_cotacao_diaria_ibovespa(lista_acoes: list[str], data_cotacao: datetime.date, periodo: str) -> pd.DataFrame:
    """Função que constroi uma dataframe com os dados de cotação da carteira de ações."""

    try:
        logger.info("Iniciando o processo de captação dos dados.")
        lista_df = []

        for acao in lista_acoes:
            try:
                # captura as informações da api com retry
                historico, info = _get_yfinance_data_with_retry(acao, periodo, requests_session)
                # verifica o conteúdo da tabela
                if historico.empty: 
                    logger.warning(f"Nenhum dado retornado para a ação: {acao} após múltiplas tentativas.")
                    
                    continue

                # inserção da data_cotacao no index
                historico["data_cotacao"] = data_cotacao
                historico.set_index("data_cotacao", inplace=True)

                # captura dos dados fundamentalistas da ação
                dict_dados_info = {k: info.get(k) for k in CAMPOS_FUNDAMENTALISTAS_SELECIONADOS}

                # construção do DataFrame de fundamentos com mesma quantidade de linhas que o histórico
                dados_fundamentalistas = pd.DataFrame([dict_dados_info] * len(historico))
                dados_fundamentalistas["data_cotacao"] = data_cotacao
                dados_fundamentalistas.set_index("data_cotacao", inplace=True)

                # concatenação dos dados de preço e fundamentos
                tbl_acao = pd.concat([historico, dados_fundamentalistas], axis=1)

                # adiciona a coluna da ação na primeira posicao do dataframe
                tbl_acao.insert(0, 'ticker', acao.replace(".SA", ""))

                # adiciona ao acumulador
                lista_df.append(tbl_acao)
                logger.info(f"Dados coletados para: {acao}")

            except Exception as e:
                # Este bloco catcha a exceção final se *todas* as retries de tenacity falharem
                logger.error(f"Erro persistente ao processar a ação {acao} após todas as tentativas de retry: {e}", exc_info=True)

            # delay entre requisições
            time.sleep(0.5)

        # concatenação final de todas as ações
        if lista_df:
            tbl_cotacao_diaria = pd.concat(lista_df)
            logger.info(f"Ações capturadas com sucesso. Dataset final: {tbl_cotacao_diaria.shape[0]} linhas.")
        else:
            logger.warning("Nenhuma ação retornou dados válidos.")
            tbl_cotacao_diaria = pd.DataFrame()

    except Exception as e:
        logger.error("Erro na construção do dataset principal", exc_info=True)
        tbl_cotacao_diaria = pd.DataFrame()

    return tbl_cotacao_diaria


def gerar_tabela_cotacao_diaria_ibovespa__http_proxy(lista_acoes: list[str], data_cotacao: datetime.date, periodo: str) -> pd.DataFrame:
    """Função que constroi uma dataframe com os dados de cotação da carteira de ações."""

    try:
        logger.info("Iniciando o processo de captação dos dados.")
        lista_df = []

        for acao in lista_acoes:
            try:
                # captura as informações da api com retry
                historico, info = _get_yfinance_data_with_retry(acao, periodo, requests_session)
                # verifica o conteúdo da tabela
                if historico.empty: 
                    logger.warning(f"Nenhum dado retornado para a ação: {acao} após múltiplas tentativas.")
                    
                    continue

                # inserção da data_cotacao no index
                historico["data_cotacao"] = data_cotacao
                historico.set_index("data_cotacao", inplace=True)

                # captura dos dados fundamentalistas da ação
                dict_dados_info = {k: info.get(k) for k in CAMPOS_FUNDAMENTALISTAS_SELECIONADOS}

                # construção do DataFrame de fundamentos com mesma quantidade de linhas que o histórico
                dados_fundamentalistas = pd.DataFrame([dict_dados_info] * len(historico))
                dados_fundamentalistas["data_cotacao"] = data_cotacao
                dados_fundamentalistas.set_index("data_cotacao", inplace=True)

                # concatenação dos dados de preço e fundamentos
                tbl_acao = pd.concat([historico, dados_fundamentalistas], axis=1)

                # adiciona a coluna da ação na primeira posicao do dataframe
                tbl_acao.insert(0, 'ticker', acao.replace(".SA", ""))

                # adiciona ao acumulador
                lista_df.append(tbl_acao)
                logger.info(f"Dados coletados para: {acao}")

            except Exception as e:
                # Este bloco catcha a exceção final se *todas* as retries de tenacity falharem
                logger.error(f"Erro persistente ao processar a ação {acao} após todas as tentativas de retry: {e}", exc_info=True)

            # delay entre requisições
            time.sleep(0.5)

        # concatenação final de todas as ações
        if lista_df:
            tbl_cotacao_diaria = pd.concat(lista_df)
            logger.info(f"Ações capturadas com sucesso. Dataset final: {tbl_cotacao_diaria.shape[0]} linhas.")
        else:
            logger.warning("Nenhuma ação retornou dados válidos.")
            tbl_cotacao_diaria = pd.DataFrame()

    except Exception as e:
        logger.error("Erro na construção do dataset principal", exc_info=True)
        tbl_cotacao_diaria = pd.DataFrame()

    return tbl_cotacao_diaria