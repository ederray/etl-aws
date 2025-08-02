"""Funções de webscrapping"""
import pandas as pd
import logging
import yfinance as yf
import time
from datetime import datetime
from src.config.constants import CAMPOS_FUNDAMENTALISTAS_SELECIONADOS 

# instância do objeto logger
logger = logging.getLogger(__name__)

def gerar_tabela_cotacao_diaria_ibovespa(lista_acoes: list[str], data_inicio: datetime.date, data_cotacao: datetime.date) -> pd.DataFrame:
    """Função que constroi uma dataframe com os dados de cotação da carteira de ações."""

    try:
        logger.info("Iniciando o processo de captação dos dados.")
        lista_df = []

        for acao in lista_acoes:
            try:
                # captura as informações da api do yfinance
                ticker = yf.Ticker(acao)
                time.sleep(0.2)
                historico = ticker.history(start=data_inicio, end=data_cotacao)
                info = ticker.info

                # verifica o conteúdo da tabela
                if historico.empty: 
                    logger.warning(f"Nenhum dado retornado para a ação: {acao} após múltiplas tentativas.")
                    
                    continue

                # inserção da data_cotacao no index
                # historico["data_cotacao"] = data_cotacao
                # historico.set_index("data_cotacao", inplace=True)

                # captura dos dados fundamentalistas da ação
                dict_dados_info = {k: info.get(k) for k in CAMPOS_FUNDAMENTALISTAS_SELECIONADOS}

                # construção do DataFrame de fundamentos com mesma quantidade de linhas que o histórico
                dados_fundamentalistas = pd.DataFrame([dict_dados_info] * len(historico), index=historico.index)
                #dados_fundamentalistas["data_cotacao"] = data_cotacao
                #dados_fundamentalistas.set_index("data_cotacao", inplace=True)

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