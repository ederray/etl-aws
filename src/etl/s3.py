"""Arquivo de funções do S3"""
import boto3
import pandas as pd
from io import StringIO
import logging

# instância do objeto logger
logger = logging.getLogger(__name__)

def carregar_lista_acoes_do_s3(bucket: str, key: str) -> list[str]:
    """Função para construir uma lista de ações componentes do Ibovespa armazenada no s3."""
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=key)
        conteudo_csv = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(conteudo_csv), sep=";")

        logger.info(f"Arquivo {key} carregado com sucesso do bucket {bucket}")
        return df["ticker"].tolist()
    
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo {key} do bucket {bucket}: {e}", exc_info=True)
        return []
