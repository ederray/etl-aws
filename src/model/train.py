"""Funções de treino e construção do pipeline do modelo"""
import logging
import pandas as pd
import shap
import category_encoders as ce
import graphviz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.tree import plot_tree 
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


# instância do objeto logger
logger = logging.getLogger(__name__)


def separar_dados_treino_teste_loc_acao(df: pd.DataFrame,target: str,coluna_acao: str,dias_treino: int = 6,dias_teste: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa os dados em conjuntos de treino e teste para CADA AÇÃO.
    Cada ação terá os primeiros 'dias_treino' para treino e os últimos 'dias_teste' para teste.
    
    Args:
        df (pd.DataFrame): DataFrame contendo todas as ações e seus dados.
                           O índice DEVE ser DatetimeIndex.
        target (str): Nome da coluna da variável target (y).
        coluna_acao (str): Nome da coluna que identifica cada ação.
        dias_treino (int): Número de dias para treino por ação.
        dias_teste (int): Número de dias para teste por ação.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) combinando os dados de todas as ações.
    """
    logger.info("Iniciando a separação dos dados em treino e teste por ação.")

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("O índice do DataFrame deve ser um DatetimeIndex.")
        raise ValueError("O índice do DataFrame deve ser um DatetimeIndex.")

    if dias_treino <= 0 or dias_teste <= 0:
        logger.error("'dias_treino' e 'dias_teste' devem ser maiores que zero.")
        raise ValueError("'dias_treino' e 'dias_teste' devem ser maiores que zero.")

    total_dias = dias_treino + dias_teste

    df_sorted = df.sort_values(by=[coluna_acao, df.index.name or 'Date'])

    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

    for acao, grupo in df_sorted.groupby(coluna_acao):
        if len(grupo) < total_dias:
            logger.warning(f"Ação '{acao}' ignorada por ter menos de {total_dias} registros.")
            continue

        # CORREÇÃO: Remova APENAS a coluna target do DataFrame de features
        X_group = grupo.drop(columns=[target])
        y_group = grupo[target]

        X_train_list.append(X_group.iloc[:dias_treino])
        X_test_list.append(X_group.iloc[dias_treino : total_dias])
        y_train_list.append(y_group.iloc[:dias_treino])
        y_test_list.append(y_group.iloc[dias_treino : total_dias])

    if not X_train_list:
        logger.error("Nenhuma ação possui dados suficientes para a separação.")
        raise ValueError("Nenhuma ação possui dados suficientes para a separação.")

    X_train = pd.concat(X_train_list)
    X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)

    logger.info(f"Separação concluída. X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def criar_preprocessor(target_encoder: str,colunas_categoricas_onehot: list[str],colunas_numericas: list[str]) -> ColumnTransformer:
    """
    Cria um ColumnTransformer com:
    - TargetEncoder para a coluna do ticker.
    - OneHotEncoder para colunas categóricas restantes.
    - PowerTransformer + StandardScaler para colunas numéricas.
    
    Args:
        target_encoder (str): Nome da coluna a ser codificada via TargetEncoder.
        colunas_categoricas_onehot (list[str]): Lista de colunas para OneHotEncoding.
        colunas_numericas (list[str]): Lista de colunas numéricas para transformação.

    Returns:
        ColumnTransformer: Pré-processador completo.
    """
    logger.info("Iniciando construção do pipeline de pré-processamento.")

    try:
        # Pipeline para colunas categóricas (exceto ticker)
        onehot_encoder_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Pipeline para colunas numéricas
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('power_transform', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ])

        # Construção do ColumnTransformer final
        preprocessor = ColumnTransformer(
            transformers=[
                ('target_encoder_ticker', ce.TargetEncoder(cols=[target_encoder]), [target_encoder]),
                ('onehot_encoder_others', onehot_encoder_pipeline, colunas_categoricas_onehot),
                ('numeric_features', numeric_pipeline, colunas_numericas)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        logger.info("Pipeline de pré-processamento construído com sucesso.")
        return preprocessor

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise

def treinar_sarimax_por_acao_com_exog(df: pd.DataFrame,coluna_acao: str,coluna_target: str,colunas_exogenas: list[str], ordem: tuple = (1, 1, 1),ordem_sazonal: tuple = (0, 0, 0, 0)) -> dict:
    """
    Treina e armazena um modelo SARIMAX para cada ação em um DataFrame,
    incluindo variáveis exógenas (features).

    Args:
        df (pd.DataFrame): DataFrame contendo todas as ações e seus dados.
                           O índice DEVE ser DatetimeIndex.
        coluna_acao (str): Nome da coluna que identifica cada ação (ex: 'ticker').
        coluna_target (str): Nome da coluna da variável target (y).
        colunas_exogenas (list[str]): Uma lista com os nomes das colunas de features (X).
        ordem (tuple): A tupla (p, d, q) para a ordem do SARIMAX.
        ordem_sazonal (tuple): A tupla (P, D, Q, S) para a ordem sazonal do SARIMAX.

    Returns:
        dict: Um dicionário onde as chaves são os nomes das ações (tickers)
              e os valores são os modelos SARIMAX treinados.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("O índice do DataFrame deve ser um DatetimeIndex para usar SARIMAX.")
        raise ValueError("O índice do DataFrame deve ser um DatetimeIndex.")
    
    modelos_sarimax = {}
    
    # Itera sobre cada grupo (ação) para treinar um modelo separado
    for acao, grupo in df.groupby(coluna_acao):
        logger.info(f"Treinando modelo SARIMAX para a ação: {acao}")
        
        # Separa a série target (endógena) e o DataFrame de features (exógenas)
        serie_endog = grupo[coluna_target]
        df_exog = grupo[colunas_exogenas] 
        
        try:
            # Tenta ajustar o modelo SARIMAX para a série
            modelo = SARIMAX(
                endog=serie_endog,
                exog=df_exog, # Variáveis exógenas são passadas aqui
                order=ordem,
                freq='B',
                seasonal_order=ordem_sazonal,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            resultados = modelo.fit(disp=False)
            
            # Armazena o modelo treinado no dicionário
            modelos_sarimax[acao] = resultados
            logger.info(f"Modelo SARIMAX para {acao} treinado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao treinar SARIMAX para {acao}: {e}")
            continue

    return modelos_sarimax



def criar_pipeline(preprocessor, modelo_final):
    """
    Cria um pipeline completo com pré-processador e modelo final.

    Args:
        preprocessor (ColumnTransformer): Pipeline de pré-processamento das features.
        modelo_final (sklearn.base.BaseEstimator): Estimador final (ex: Regressor ou Classifier).

    Returns:
        sklearn.pipeline.Pipeline: Pipeline pronto para treino ou GridSearchCV.
    """
    logger.info("Iniciando criação do pipeline completo.")

    try:
        pipeline = Pipeline([
            ('preprocessador', preprocessor),
            ('modelo', modelo_final)
        ])
        logger.info(f"{pipeline.named_steps['preprocessador']}")
        logger.info("Pipeline criado com sucesso.")
        return pipeline

    except Exception as e:
        logger.error(f"Erro ao criar pipeline: {e}")
        raise



def ajuste_pipeline_com_grid_search():
    pass


def gerar_metricas(y_true, y_pred):
    """
    Gera e imprime métricas de avaliação para modelos de regressão.
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Adicione outras métricas que você precisar
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")


