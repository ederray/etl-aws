import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import SparkSession, Window
from awsglue.job import Job
from pyspark.sql.functions import col, dayofweek, month, sin, cos, lag, mean, stddev, lit, concat, lower, trim, length
from datetime import datetime, timedelta
import re
import math
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, DateType, IntegerType
import botocore.session
from botocore.config import Config
from botocore.exceptions import ClientError
import time
import boto3

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================================
# 1. Inicialização do ambiente Glue e argumentos
# ========================================================================
try:
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'PROCESSING_DATE', 'BUCKET_NAME'])

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    BUCKET_NAME = args.get('BUCKET_NAME')
    PROCESSING_DATE_STR = args.get('PROCESSING_DATE')

    if not BUCKET_NAME or not PROCESSING_DATE_STR:
        raise ValueError("Argumentos BUCKET_NAME e PROCESSING_DATE são obrigatórios.")

    PROCESSING_DATE = datetime.strptime(PROCESSING_DATE_STR, '%Y-%m-%d')
    S3_PATH_REF = f"s3://{BUCKET_NAME}/references/tbl_acoes_ibovespa.csv"
    S3_PATH_RAW = f"s3://{BUCKET_NAME}/raw/"
    S3_PATH_REFINED = f"s3://{BUCKET_NAME}/refined/"
    DIAS_HISTORICO = 10
    data_inicio = PROCESSING_DATE - timedelta(days=DIAS_HISTORICO)

    logger.info("Iniciando o job de ETL para a data: %s", PROCESSING_DATE_STR)
    logger.info("Período de histórico a ser processado: %s a %s", data_inicio.strftime('%Y-%m-%d'), PROCESSING_DATE_STR)

    explicit_schema_without_date = StructType([
        StructField("ticker", StringType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Volume", LongType(), True),
        StructField("Dividends", DoubleType(), True),
        StructField("Stock Splits", DoubleType(), True),
        StructField("trailingPE", DoubleType(), True),
        StructField("priceToBook", DoubleType(), True),
        StructField("returnOnEquity", DoubleType(), True),
        StructField("profitMargins", DoubleType(), True),
        StructField("revenueGrowth", DoubleType(), True),
        StructField("dividendYield", DoubleType(), True),
        StructField("totalDebt", LongType(), True),
        StructField("freeCashflow", LongType(), True),
    ])

    # ========================================================================
    # 2. Carregamento dos dados brutos e da tabela de referência
    # ========================================================================
    logger.info("Lendo dados Parquet particionados do S3 do caminho raiz: %s", S3_PATH_RAW)
    
    # Filtra por partição no S3
    df = spark.read.parquet(S3_PATH_RAW).filter(
        (col("ano") >= data_inicio.year) &
        (col("ano") <= PROCESSING_DATE.year) &
        (col("mes") >= data_inicio.month) &
        (col("mes") <= PROCESSING_DATE.month)
    )
    
    df = df.withColumn("date", concat(col("ano"), lit("-"), col("mes"), lit("-"), col("dia")))
    df = df.withColumn("date", col("date").cast(DateType()))
    
    logger.info("Inspecionando dados brutos após a leitura.")
    if isinstance(df, DataFrame) and df.count() > 0:
        df.printSchema()
        df.show(5, truncate=False)
    
    df_count = df.count()
    logger.info("CHECKPOINT 1: Total de registros lidos do raw: %d", df_count)
    
    df_count_for_date = df.filter(col("date") == PROCESSING_DATE).count()
    logger.info("CHECKPOINT 1.1: Total de registros lidos para a data %s: %d", PROCESSING_DATE_STR, df_count_for_date)

    logger.info("Carregando a tabela de referência do S3 em %s...", S3_PATH_REF)
    df_ref = spark.read.csv(S3_PATH_REF, header=True, sep=";", inferSchema=True)
    
    df_ref_columns = [col_name.lower() for col_name in df_ref.columns]
    df_ref = df_ref.toDF(*df_ref_columns)
    
    df_ref_count = df_ref.count()
    logger.info("CHECKPOINT 2: Total de registros lidos da referência: %d", df_ref_count)
    
    # ========================================================================
    # 3. Renomeação, padronização e junção
    # ========================================================================
    logger.info("Iniciando a renomeação, padronização e junção dos DataFrames...")
    
    df = df.withColumnRenamed("ticker", "acao")
    df = df.withColumnRenamed("Open", "abertura")
    df = df.withColumnRenamed("High", "maximo")
    df = df.withColumnRenamed("Low", "minimo")
    df = df.withColumnRenamed("Close", "fechamento")
    df = df.withColumnRenamed("Volume", "volume")
    df = df.withColumnRenamed("Dividends", "dividendos")
    df = df.withColumnRenamed("Stock Splits", "desdobramentos")
    df = df.withColumnRenamed("trailingPE", "trailing_pe")
    df = df.withColumnRenamed("priceToBook", "price_to_book")
    df = df.withColumnRenamed("returnOnEquity", "return_on_equity")
    df = df.withColumnRenamed("profitMargins", "profit_margins")
    df = df.withColumnRenamed("revenueGrowth", "revenue_growth")
    df = df.withColumnRenamed("dividendYield", "dividend_yield")
    df = df.withColumnRenamed("totalDebt", "total_debt")
    df = df.withColumnRenamed("freeCashflow", "free_cashflow")
    
    # CORREÇÃO FINAL ROBUSTA: CRIA CHAVES DE JUNÇÃO EXPLÍCITAS E PADRONIZADAS
    df = df.withColumn("join_key", lower(trim(col("acao").cast(StringType()))))
    df_ref = df_ref.withColumn("join_key", lower(trim(col("codigo").cast(StringType()))))
    
    df_merged = df.join(df_ref, df.join_key == df_ref.join_key, "left")
    
    # Verificação final da junção
    registros_com_join = df_merged.filter(col("tipo").isNotNull()).count()
    logger.info(f"VERIFICAÇÃO FINAL: Número de registros com colunas 'tipo' e 'setor' preenchidas: {registros_com_join}")

    df_merged = df_merged.select(
        col("acao"),
        col("date"),
        col("abertura"),
        col("maximo"),
        col("minimo"),
        col("fechamento"),
        col("volume"),
        col("dividendos"),
        col("desdobramentos"),
        col("trailing_pe"),
        col("price_to_book"),
        col("return_on_equity"),
        col("profit_margins"),
        col("revenue_growth"),
        col("dividend_yield"),
        col("total_debt"),
        col("free_cashflow"),
        col("tipo"),
        col("setor"),
        col("industria"),
        col("ticker")
    )
    
    logger.info("Renomeação, padronização e junção concluídas. Colunas finais: %s", df_merged.columns)
      
    df_merged_count = df_merged.count()
    logger.info("CHECKPOINT 3: Total de registros após a junção: %d", df_merged_count)

    df_merged_count_for_date = df_merged.filter(col("date") == PROCESSING_DATE).count()
    logger.info("CHECKPOINT 3.1: Registros após junção para a data %s: %d", PROCESSING_DATE_STR, df_merged_count_for_date)
    
    # 4. Geração de features de tempo e cíclicas
    logger.info("Gerando features temporais e cíclicas...")
    df_merged = df_merged.withColumn('dayofweek', dayofweek('date')).withColumn('month', month('date'))
    df_merged = df_merged.withColumn('day_sin', sin(2 * math.pi * ((col('dayofweek')-2) % 5) / 5))
    df_merged = df_merged.withColumn('day_cos', cos(2 * math.pi * ((col('dayofweek')-2) % 5) / 5))
    df_merged = df_merged.withColumn('month_sin', sin(2 * math.pi * col('month') / 12))
    df_merged = df_merged.withColumn('month_cos', cos(2 * math.pi * col('month') / 12))
    logger.info("Features cíclicas geradas.")
    
    # 5. Diferenciação e Features de Janela
    logger.info("Aplicando diferenciação e gerando features de janela...")
    window_spec = Window.partitionBy("acao").orderBy("date")
    df_merged = df_merged.withColumn("close_diff", col("fechamento") - lag("fechamento", 1).over(window_spec))
    
    logger.info("CHECKPOINT 4: Total de registros após a criação de close_diff: %d", df_merged.count())
    
    lags = [1, 2, 3, 5]
    janelas_rolling = [3, 5]
    for lag_val in lags:
        df_merged = df_merged.withColumn(f'lag_{lag_val}_close_diff', lag("close_diff", lag_val).over(window_spec))
    for janela in janelas_rolling:
        window_rolling = Window.partitionBy("acao").orderBy("date").rowsBetween(-janela, -1)
        df_merged = df_merged.withColumn(f'rolling_mean_{janela}_close_diff', mean("close_diff").over(window_rolling))
        df_merged = df_merged.withColumn(f'volatility_{janela}_close_diff', stddev("close_diff").over(window_rolling))
        
    logger.info("Diferenciação e features de janela concluídas.")

    # 6. Tratamento de Nulos
    logger.info("Iniciando o tratamento de nulos com fallback zero...")
    
    cols_to_fill = [
        "close_diff",
        "lag_1_close_diff",
        "lag_2_close_diff",
        "lag_3_close_diff",
        "lag_5_close_diff",
        "rolling_mean_3_close_diff",
        "volatility_3_close_diff",
        "rolling_mean_5_close_diff",
        "volatility_5_close_diff"
    ]
    
    df_final = df_merged.fillna(0, subset=cols_to_fill)
    
    logger.info("Tratamento de nulos concluído. Contando o DataFrame final para validação...")
    df_final_count = df_final.count()
    logger.info("CHECKPOINT 6: Linhas finais: %d", df_final_count)
    
    # ========================================================================
    # 7. Salvamento e Catalogação
    # ========================================================================
    if df_final_count > 0:
        
        # Remove as colunas ano, mes e dia definitivamente, mantendo apenas a date
        df_to_write = df_final.drop("ano", "mes", "dia").dropDuplicates(["date", "acao"])
        
        df_to_write_count = df_to_write.count()
        logger.info("CHECKPOINT 7: Total de registros a serem escritos para o período: %d", df_to_write_count)
        
        if df_to_write_count > 0:
            logger.info("Esquema do DataFrame final a ser salvo:")
            df_to_write.printSchema()
            
            logger.info("Salvando o DataFrame refinado em %s...", S3_PATH_REFINED)
            # Salvamos o DataFrame sem as colunas redundantes
            df_to_write.write.partitionBy("date", "acao").mode("overwrite").parquet(S3_PATH_REFINED)
            
            logger.info("Dados refinados salvos no S3 com sucesso!")
        
        # --- BLOCO DE CÓDIGO PARA AUTOMAÇÃO DO ATHENA ---
        try:
            # Nomes do banco de dados e tabela no Glue Data Catalog
            DATABASE_NAME = "seu_banco_de_dados_glue"  # <<< Mude para o nome do seu banco de dados
            TABLE_NAME = "sua_tabela_refined"        # <<< Mude para o nome da sua tabela
            
            query = f"MSCK REPAIR TABLE {DATABASE_NAME}.{TABLE_NAME}"
            
            logger.info(f"Executando MSCK REPAIR TABLE para {DATABASE_NAME}.{TABLE_NAME}")
            
            # Inicializa o cliente Athena
            athena_config = Config(
                region_name=boto3.session.Session().region_name,
                retries={'max_attempts': 5}
            )
            athena_client = botocore.session.get_session().create_client('athena', config=athena_config)
            
            response = athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': DATABASE_NAME
                },
                ResultConfiguration={
                    'OutputLocation': f's3://{BUCKET_NAME}/athena-query-results/'
                }
            )
            
            query_execution_id = response['QueryExecutionId']
            
            while True:
                status_response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                status = status_response['QueryExecution']['Status']['State']
                if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                time.sleep(1)
            
            if status == 'SUCCEEDED':
                logger.info(f"MSCK REPAIR TABLE executado com sucesso.")
            else:
                error_message = status_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                logger.error(f"Falha ao executar MSCK REPAIR TABLE. Status: {status}, Motivo: {error_message}")
        
        except ClientError as e:
            logger.error(f"Erro no cliente Athena: {e}")
        except Exception as e:
            logger.error(f"Erro ao executar MSCK REPAIR TABLE: {e}")
        # --- FIM DO BLOCO DE AUTOMAÇÃO ---
        
    else:
        logger.warning("DataFrame final para o período está vazio. Nenhum dado será salvo no S3.")
        
    job.commit()

except Exception as e:
    logger.error("ERRO CRÍTICO NO PROCESSO: %s", e)
    if 'job' in locals() and job:
        job.commit()