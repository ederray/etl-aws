"""Arquivo de funções do S3"""
import boto3
import os
import pandas as pd
import datetime 
from io import StringIO, BytesIO
import logging
from urllib.parse import urlparse

# instância do objeto logger
logger = logging.getLogger(__name__)

def carregar_lista_acoes_do_s3(bucket: str, key: str) -> list[str]:
    """Função para construir uma lista de ações componentes do Ibovespa armazenada no s3."""
    try:
        logger.info(f"Iniciando o processo de carregamento da lista de ações.")
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=key)
        conteudo_csv = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(conteudo_csv), sep=";")

        logger.info(f"Arquivo {key} carregado com sucesso do bucket {bucket}")
        return df["ticker"].tolist()
    
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo {key} do bucket {bucket}: {e}", exc_info=True)
        return []


def carregar_dados_parquet_particao_diaria_s3(df: pd.DataFrame, data_cotacao: datetime.date) -> bool:
    """
    Função para carregar os dados de cotação diária no formato parquet
    com partição diária no bucket S3.
    Retorna True em caso de sucesso, False em caso de falha.
    Exemplo: raw/ano=2025/mes=08/dia=01/cotacoes.parquet
    """
    logger.info("Início da transformação e carregamento dos dados no formato parquet para S3.")

    CAMINHO_CSV_S3_URL = os.environ.get("CAMINHO_CSV")
    PASTA_RAW = os.environ.get("PASTA_RAW")

    logger.info(f"DEBUG: CAMINHO_CSV lido: '{CAMINHO_CSV_S3_URL}'")
    logger.info(f"DEBUG: PASTA_RAW lido: '{PASTA_RAW}'")

    if not CAMINHO_CSV_S3_URL:
        logger.error("Erro de Configuração: Variável de ambiente 'CAMINHO_CSV' (URL S3 completo) não definida ou vazia.")
        raise ValueError("CAMINHO_CSV (URL S3 completo) não definido na variável de ambiente.")
    
    if not PASTA_RAW:
        logger.error("Erro de Configuração: Variável de ambiente 'PASTA_RAW' não definida ou vazia. Define o prefixo base no S3.")
        raise ValueError("PASTA_RAW não definida na variável de ambiente.")

    # instância do caminho do bucket
    try:
        parsed_s3_path = urlparse(CAMINHO_CSV_S3_URL)
        bucket_name_for_s3 = parsed_s3_path.netloc.strip()
    
        logger.info(f"Variável de Ambiente CAMINHO_CSV (URL S3 completo): {CAMINHO_CSV_S3_URL}")
        logger.info(f"Bucket extraído para S3: {bucket_name_for_s3}")

        if not bucket_name_for_s3:
            logger.error(f"Erro de Parse: 'netloc' vazio. URL processada: {CAMINHO_CSV_S3_URL}")
            raise ValueError(f"Não foi possível extrair o nome do bucket do URL S3: {CAMINHO_CSV_S3_URL}. 'netloc' está vazio.")

        logger.info(f"Variável de Ambiente CAMINHO_CSV (URL S3 completo): {CAMINHO_CSV_S3_URL}")
        logger.info(f"Bucket de destino extraído para S3: {bucket_name_for_s3}")

    except Exception as e:
        logger.error(f"Erro ao parsear o URL S3 da variável de ambiente CAMINHO_CSV: {e}", exc_info=True)
        return False

    # instância das variáveis de particionamento dos dados.
    ano = data_cotacao.strftime('%Y')
    mes = data_cotacao.strftime('%m')
    dia = data_cotacao.strftime('%d')

    # A construção do caminho final usa o prefixo PASTA_RAW corrigido
    caminho_final = f"{PASTA_RAW}/ano={ano}/mes={mes}/dia={dia}/cotacao_ibovespa.parquet"

    logger.info(f"Caminho S3 de destino: s3://{bucket_name_for_s3}/{caminho_final}")

    # Transformação dos dados no formato parquet
    buffer = BytesIO()
    df.to_parquet(buffer, engine="pyarrow") 
    buffer.seek(0)

    # etapa de carregamento dos dados no formato parquet no s3. Retorno booleano para validar o processo.
    try:
        s3 = boto3.client('s3')
        s3.upload_fileobj(buffer, bucket_name_for_s3, caminho_final)
        logger.info(f"Arquivo salvo com sucesso em s3://{bucket_name_for_s3}/{caminho_final}")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar Parquet no S3: {e}", exc_info=True)
        return False