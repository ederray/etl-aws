"""Arquivo de execução das tarefas"""
import logging
import pandas as pd 
from datetime import date, timedelta
from src.config.logging_config import setup_logging
from src.etl.webscrapping import gerar_tabela_cotacao_diaria_ibovespa
from src.etl.s3 import carregar_dados_parquet_particao_diaria_s3
from dotenv import load_dotenv

# instância do objeto logger
setup_logging()
logger = logging.getLogger(__name__)

# carregamento das variáveis de ambiente
load_dotenv()

# captura dos dados para a data de hoje
hoje = date.today()
tbl_carteira_ibovespa = pd.read_csv("./data/external/tbl_acoes_ibovespa.csv", sep=';')
tbl_cotacao_diaria = gerar_tabela_cotacao_diaria_ibovespa(lista_acoes = tbl_carteira_ibovespa['ticker'], 
                                     data_cotacao= hoje)
tbl_cotacao_diaria.head()

# envio dos dados para o bucket s3
carregar_dados_parquet_particao_diaria_s3(df=tbl_cotacao_diaria,data_cotacao=hoje)