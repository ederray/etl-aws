"""Arquivo de configuração do lambda de captura de ações"""
import os
import boto3
import pandas as pd
from io import BytesIO
from datetime import date
import logging
from src.config.logging_config import setup_logging
from etl.webscrapping import gerar_tabela_cotacao_diaria_ibovespa
from src.etl.s3 import carregar_lista_acoes_do_s3

# carregamento das configurações de logging
setup_logging()
logger = logging.getLogger(__name__)

# Variáveis de ambiente
BUCKET_NAME = os.environ.get("BUCKET_NAME")
CAMINHO_CSV = os.environ["CAMINHO_CSV"]
PASTA_RAW = os.environ.get("PASTA_RAW")


# Função lambda
def lambda_handler(event, context):
    logger.info("Lambda iniciada.")

    # seleção da data do dia para definir a cotacao diária
    data_cotacao = date.today()
    # definição da partição dos dados no S3
    str_data_particao = data_cotacao.strftime('%Y-%m-%d')

    # Carrega lista de ações do S3
    lista_acoes = carregar_lista_acoes_do_s3(bucket=BUCKET_NAME, key=CAMINHO_CSV)
    
    if not lista_acoes:
        logger.warning("Lista de ações vazia ou não carregada.")
        return {
            "statusCode": 400,
            "body": "Erro ao carregar lista de ações do S3."
        }

    logger.info(f"{len(lista_acoes)} ações carregadas para cotação.")

    # Construção do dataset com a cotação diária das ações
    df = gerar_tabela_cotacao_diaria_ibovespa(
        lista_acoes=lista_acoes,
        data_cotacao=data_cotacao,
        periodo="1d"
    )

    # Etapa de verificação de linhas do dataset
    if df.empty:
        logger.warning("DataFrame vazio retornado pela coleta de dados.")
        return {
            "statusCode": 204,
            "body": "Nenhum dado coletado para os tickers fornecidos."
        }

    # Transformação dos dados no formato parquet
    buffer = BytesIO()
    df.to_parquet(buffer, index=True, engine="pyarrow")
    buffer.seek(0)

    caminho_final = f"{PASTA_RAW}/data_cotacao={str_data_particao}/cotacoes.parquet"

    # etapa de carregamento dos dados no formato parquet no s3.
    try:
        s3 = boto3.client('s3')
        s3.upload_fileobj(buffer, BUCKET_NAME, caminho_final)
        logger.info(f"Arquivo salvo em s3://{BUCKET_NAME}/{caminho_final}")
        return {
            "statusCode": 200,
            "body": f"Arquivo salvo com sucesso em s3://{BUCKET_NAME}/{caminho_final}"
        }
    except Exception as e:
        logger.error(f"Erro ao salvar Parquet no S3: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": "Erro ao salvar arquivo no S3."
        }