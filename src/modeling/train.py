"""Módulo de funções e classes de treinamento"""

import logging
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import category_encoders as ce
from pandas import DataFrame, Series
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor


# instância do objeto logger
logger = logging.getLogger(__name__)


def split_dados_periodo_acao(df: pd.DataFrame, cutoff_date: str, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa o DataFrame em conjuntos de treino e teste usando uma data de corte, 
    garantindo que não haja sobreposição de datas.

    params:
        df (pd.DataFrame): DataFrame com a coluna 'ticker' e com DatetimeIndex.
        cutoff_date (str): Data de corte no formato 'YYYY-MM-DD'.
        target (str): Nome da coluna da variável target (y).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Garante que 'Date' e 'ticker' estejam como colunas antes de criar o MultiIndex
    if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Cria o MultiIndex com 'ticker' e 'Date'
    df = df.set_index(['ticker', 'Date']).sort_index()

    # Converte o cutoff_date para o tipo de dado correto
    cutoff_datetime = pd.to_datetime(cutoff_date)
    
    # Separa os dados com o MultiIndex, garantindo que o cutoff_date pertença
    # apenas ao conjunto de teste.
    train_df = df.loc[(slice(None), slice(None, cutoff_datetime - pd.Timedelta(days=1))), :]
    test_df = df.loc[(slice(None), slice(cutoff_datetime, None)), :]
    
    # Separa X e y
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    logger.info(f"\nDados de treino:{y_train.index.get_level_values('Date').min().date()} a {y_train.index.get_level_values('Date').max().date()}\nDados de teste:{y_test.index.get_level_values('Date').min().date()} a {y_test.index.get_level_values('Date').max().date()}")
    logger.info(f"\nShape X_treino:{X_train.shape}\nShape y_treino{y_train.shape}\nShape X_teste:{X_test.shape}\nShape y_teste{y_test.shape}\n")
    
    return X_train, X_test, y_train, y_test


def criar_pipeline(
    #colunas_target_encoder: list[str],
    colunas_categoricas: list[str],
    colunas_numericas: list[str],
    modelo=None,
    usar_onehot: bool = True,
) -> Pipeline:
    """
    Cria um pipeline de pré-processamento + modelo.

    params:

        colunas_categoricas : list[str] Colunas categóricas.
        colunas_numericas : list[str] Colunas numéricas.
        modelo : estimator, default=None Modelo a ser usado no final do pipeline.
        usar_onehot : bool, default=True Se True aplica OneHotEncoder; se False mantém as categorias apenas imputadas.

    return:
        pipeline com modelo.

    """
    logger.info("Iniciando construção do pipeline de pré-processamento.")

    try:

        if usar_onehot:
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
        else:
            cat_pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="most_frequent"))]
            )


        numeric_passthrough = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaling", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
            
                #('target_encoder', ce.LeaveOneOutEncoder(cols=colunas_target_encoder), colunas_target_encoder), 
                ("cat_features", cat_pipeline, colunas_categoricas),
                ("num_passthrough", numeric_passthrough, colunas_numericas),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
            )

        pipeline = Pipeline(steps=[("preprocessing", preprocessor),("model", modelo)])

        logger.info(f"Pipeline de pré-processamento construído com sucesso.{pipeline}")
        return pipeline

    except Exception as e:
        logger.error(f"Erro ao construir o pipeline: {e}")
        raise e


def treinar_sarimax_por_acao_com_exog(
    df: pd.DataFrame,
    coluna_acao: str,
    coluna_target: str,
    colunas_exogenas: list[str]
) -> dict:
    """
    Treina modelos SARIMAX por ação com variáveis exógenas padronizadas via StandardScaler.

    params:
    
        df : pd.DataFrame
            DataFrame contendo as séries temporais e as variáveis exógenas.
            O índice deve ser um DatetimeIndex.
        coluna_acao : str
            Nome da coluna que identifica cada ação (ou grupo).
        coluna_target : str
            Nome da coluna alvo (variável endógena).
        colunas_exogenas : list[str]
            Lista com os nomes das colunas exógenas.

    return:
  
        dict
            Dicionário com o nome da ação como chave e o modelo SARIMAX ajustado como valor.
    """
    
    modelos_sarimax = {}
    ORDEM_ARIMA = (0, 1, 0)
    ORDEM_SAZONAL = (0, 0, 0, 0)

    logger.info(f"Usando parâmetros ARIMA: {ORDEM_ARIMA}. Sazonalidade removida.")

    for acao in df[coluna_acao].unique():
        logger.info(f"Treinando modelo SARIMAX para a ação: {acao}")

        df_acao = df[df[coluna_acao] == acao].copy()

        endog = df_acao[coluna_target]
        exog = df_acao[colunas_exogenas] if colunas_exogenas else None

        if exog is not None:
            scaler = StandardScaler()
            exog_scaled = pd.DataFrame(
                scaler.fit_transform(exog),
                index=exog.index,
                columns=exog.columns
            )

        else:
            exog_scaled = None

        try:
            modelo = SARIMAX(
                endog=endog,
                exog=exog_scaled,
                order=ORDEM_ARIMA,
                seasonal_order=ORDEM_SAZONAL,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            resultados = modelo.fit(disp=False)
            modelos_sarimax[acao] = {
                "modelo": resultados,
                "scaler": scaler if exog is not None else None
            }

            logger.info(f"Modelo SARIMAX para {acao} treinado com sucesso.")

        except Exception as e:
            logger.error(f"Erro ao treinar SARIMAX para {acao}: {e}. Pulando esta ação.")
            continue

    return modelos_sarimax


def gerar_halving_random_search_cv(
    pipeline=Pipeline,
    param_grid=dict,
    cv=None,
    scoring: str = "r2",
    n_jobs: int = None,
    verbose: int = 1,
) -> HalvingRandomSearchCV:
    """
    Gera um grid de validação e ajuste de hiperparâmetros com seleção randômica para cross-validação (Kfold) para os hiperpâmetros do modelo.

    params:
        pipeline (Pipeline): Modelo acoplado ao pipeline de pré-processamento dos dados.
        param_grid (dict): Grid com hiperparâmetros de ajuste.
        cv (Kfold=5): Objeto de cross-validação dos dados.
        scoring (str='r2'): Métrica de avaliação.
        n_jobs (int=None): Capacidade de processamento definido.
        verbose (int=0): Descrição textual dos processos de validação.

    return:
        HalvingSearchCV: Objeto com modelo ajustado e com pipeline de pré-processamento definido.
    """
    try:
        halv = HalvingRandomSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=cv,
            scoring=scoring,
            n_candidates=100,
            factor=2,
            verbose=verbose,
            min_resources=500,
            random_state=42,
            n_jobs=n_jobs,
            error_score='raise'
        )
        return halv

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise


def gerar_metricas(y_true, y_pred):
    """
    Gera e imprime métricas de avaliação para modelos de regressão.

    params:

        y_true: valor real.
        y_pred: valor predição.

    return:
        mae: erro médio absoluto.
        mse: erro médio quadrático.
        rmse: raiz do erro médio quadrático.
        r2: coeficiente de de determinação.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")

    return mae, mse, rmse, r2

