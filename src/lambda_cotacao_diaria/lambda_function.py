"""Arquivo de configuração do lambda de captura de ações"""
import os
import boto3
from io import BytesIO
from datetime import date, timedelta
import logging
from urllib.parse import urlparse
from src.config.logging_config import setup_logging
from src.etl.webscrapping import gerar_tabela_cotacao_diaria_ibovespa
from src.etl.s3 import carregar_lista_acoes_do_s3

# carregamento das configurações de logging
logger = logging.getLogger(__name__)

# instância da variável com o path completo
CAMINHO_CSV_S3_URL = os.environ["CAMINHO_CSV"]
PASTA_RAW_PREFIX = os.environ.get("PASTA_RAW", "raw").strip('/')


# Função lambda
def lambda_handler(event, context):
    logger.info("Lambda iniciada.")

    # tentativa de instancia do caminho do bucket
    try:
        parsed_s3_path = urlparse(CAMINHO_CSV_S3_URL)
        bucket_name_for_s3 = parsed_s3_path.netloc
        object_key_for_s3 = parsed_s3_path.path.lstrip('/')

        logger.info(f"Variável de Ambiente CAMINHO_CSV (URL S3 completo): {CAMINHO_CSV_S3_URL}")
        logger.info(f"Bucket extraído para S3: {bucket_name_for_s3}")
        logger.info(f"Chave (Key) extraída para S3: {object_key_for_s3}")

    except Exception as e:
        logger.error(f"Erro ao parsear o URL S3 da variável de ambiente CAMINHO_CSV: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'body': "Erro interno: Falha ao processar variável CAMINHO_CSV."
        }

    # seleção da data do dia para definir a cotacao diária
    data_cotacao = date.today()-timedelta(days=1)

    # Carrega lista de ações do S3
    lista_acoes = carregar_lista_acoes_do_s3(bucket=bucket_name_for_s3, key=object_key_for_s3)
    lista_acoes = ['PETR3.SA', 'ABEV3.SA']
    
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

    # Instâncias das variáveis para partição do arquivo no s3
    ano = data_cotacao.strftime('%Y')
    mes = data_cotacao.strftime('%m')
    dia = data_cotacao.strftime('%d')

    # A construção do caminho final usa o prefixo PASTA_RAW_PREFIX corrigido
    caminho_final = f"{PASTA_RAW_PREFIX}/ano={ano}/mes={mes}/dia={dia}/cotacoes.parquet"


    # etapa de carregamento dos dados no formato parquet no s3.
    try:
        s3 = boto3.client('s3')
        # BUCKET_NAME também será o 'bucket_name_for_s3' (o nome do bucket extraído)
        s3.upload_fileobj(buffer, bucket_name_for_s3, caminho_final)
        logger.info(f"Arquivo salvo em s3://{bucket_name_for_s3}/{caminho_final}")
        return {
            "statusCode": 200,
            "body": f"Arquivo salvo com sucesso em s3://{bucket_name_for_s3}/{caminho_final}"
        }
    except Exception as e:
        logger.error(f"Erro ao salvar Parquet no S3: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": "Erro ao salvar arquivo no S3."
        }