"""Funções de webscrapping"""
import pandas as pd
import logging
import yfinance as yf
from datetime import datetime
from src.config.logging_config import setup_logging
from src.config.constants import CAMPOS_FUNDAMENTALISTAS_SELECIONADOS

# instância do objeto logger
logger = logging.getLogger(__name__)


def gerar_tabela_cotacao_diaria_ibovespa(lista_acoes: list[str], data_cotacao: datetime, periodo: str) -> pd.DataFrame:
    """Função que constroi uma dataframe com os dados de cotação da carteira de ações."""

    try:
        logger.info("Iniciando o processo de captação dos dados.")

        # lista para armazenar os resultados de cada ação
        lista_df = []

        # laço de repetição para captar os dados da api yfinance
        for acao in lista_acoes:

            # captura dos dados históricos da ação
            ticker = yf.Ticker(acao)
            historico = ticker.history(period=periodo)

            # verifica se retornou dados
            if historico.empty:
                logger.warning(f"Nenhum dado retornado para a ação: {acao}")
                continue

            # inserção da data_cotacao no index
            historico["data_cotacao"] = data_cotacao
            historico.set_index("data_cotacao", inplace=True)

            # captura dos dados fundamentalistas da ação
            info = ticker.info
            dict_dados_info = {k: info.get(k) for k in CAMPOS_FUNDAMENTALISTAS_SELECIONADOS}

            # construção do DataFrame de fundamentos com mesma quantidade de linhas que o histórico
            dados_fundamentalistas = pd.DataFrame([dict_dados_info] * len(historico))
            dados_fundamentalistas["data_cotacao"] = data_cotacao
            dados_fundamentalistas.set_index("data_cotacao", inplace=True)

            # concatenação dos dados de preço e fundamentos
            tbl_acao = pd.concat([historico, dados_fundamentalistas], axis=1)

            # adiciona a coluna da ação na primeira posicao do dataframe
            tbl_acao.insert(0, 'ticker', acao)

            # adiciona ao acumulador
            lista_df.append(tbl_acao)

        # concatenação final de todas as ações
        if lista_df:
            tbl_cotacao_diaria = pd.concat(lista_df)
            logger.info(f"Ações capturadas com sucesso. Dataset final: {tbl_cotacao_diaria.shape[0]} linhas.")
        else:
            logger.warning("Nenhuma ação retornou dados válidos.")
            tbl_cotacao_diaria = pd.DataFrame()

    except Exception as e:
        logger.error("Erro na construção do dataset", exc_info=True)
        tbl_cotacao_diaria = pd.DataFrame()

    return tbl_cotacao_diaria