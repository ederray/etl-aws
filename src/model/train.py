"""Funções de treino e construção do pipeline do modelo"""
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from typing import List


# instância do objeto logger
logger = logging.getLogger(__name__)



def selecionar_features(df: pd.DataFrame, 
                       features: List[str],
                       target: str,
                       k: int = 20) -> List[str]:
    """
    Seleciona as K melhores features usando o teste estatístico f_regression.

    Args:
        df (pd.DataFrame): DataFrame contendo as features e o target.
        features (List[str]): Lista de nomes das features a serem avaliadas.
        target (str): Nome da coluna do target.
        k (int): Número de melhores features a serem selecionadas.

    Returns:
        List[str]: Uma lista com os nomes das features selecionadas.
    """
    df = df.copy().dropna()
    
    X = df[features]
    y = df[target]

    # Configura o seletor para encontrar as K melhores features
    selector = SelectKBest(score_func=f_regression, k=k)
    
    # Aplica a seleção
    selector.fit(X, y)
    
    # Obtém as features selecionadas
    features_selecionadas = X.columns[selector.get_support()].tolist()
    
    return features_selecionadas


def split_data_by_date(df: pd.DataFrame, cutoff_date: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa o DataFrame em conjuntos de treino e teste usando uma data de corte.

    Args:
        df (pd.DataFrame): DataFrame com DatetimeIndex e dados de várias ações.
        cutoff_date (str): Data de corte no formato 'YYYY-MM-DD'.
        target (str): Nome da coluna da variável target (y).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Certifique-se de que o índice é um DatetimeIndex e está ordenado
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("O índice do DataFrame deve ser um DatetimeIndex.")
    
    df_sorted = df.sort_index()

    # Filtra os dados de treino
    train_df = df_sorted.loc[df_sorted.index < cutoff_date]
    
    # Filtra os dados de teste
    test_df = df_sorted.loc[df_sorted.index >= cutoff_date]
    
    # Separa X e y para treino
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    # Separa X e y para teste
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    
    print(f"Dados de treino: {train_df.index.min().date()} a {train_df.index.max().date()}")
    print(f"Dados de teste: {test_df.index.min().date()} a {test_df.index.max().date()}")
    
    return X_train, X_test, y_train, y_test

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
                ('target_encoder', ce.TargetEncoder(cols=[target_encoder]), [target_encoder]),
                ('cat_features', onehot_encoder_pipeline, colunas_categoricas_onehot),
                ('num_features', numeric_pipeline, colunas_numericas)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

        logger.info("Pipeline de pré-processamento construído com sucesso.")
        return preprocessor

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise

def criar_preprocessor_sem_target_encoder(colunas_categoricas_onehot: list[str],colunas_numericas: list[str]) -> ColumnTransformer:
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
                ('cat_features', onehot_encoder_pipeline, colunas_categoricas_onehot),
                ('num_features', numeric_pipeline, colunas_numericas)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        logger.info("Pipeline de pré-processamento construído com sucesso.")
        return preprocessor

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise

def treinar_sarimax_por_acao_com_exog(df: pd.DataFrame, coluna_acao: str, coluna_target: str,
                                     colunas_exogenas: list[str]) -> dict:
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("O índice do DataFrame deve ser um DatetimeIndex.")
        raise ValueError("O índice do DataFrame deve ser um DatetimeIndex.")
    
    modelos_sarimax = {}
    ORDEM_ARIMA = (1, 0, 1)
    
    logger.info(f"Usando parâmetros ARIMA: {ORDEM_ARIMA}. Sazonalidade removida.")
    
    # Itera sobre cada ação única
    for acao in df[coluna_acao].unique():
        logger.info(f"Treinando modelo SARIMAX para a ação: {acao}")
        
        # Filtra dados para a ação
        df_acao = df[df[coluna_acao] == acao]
        
        # Endog: série temporal da coluna target
        endog = df_acao[coluna_target]
        
        # Exog: dataframe das colunas exógenas
        exog = df_acao[colunas_exogenas] if colunas_exogenas else None
        
        try:
            modelo = SARIMAX(
                endog=endog,
                exog=exog,
                order=ORDEM_ARIMA,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            resultados = modelo.fit(disp=False)
            modelos_sarimax[acao] = resultados
            logger.info(f"Modelo SARIMAX para {acao} treinado com sucesso.")
        
        except Exception as e:
            logger.error(f"Erro ao treinar SARIMAX para {acao}: {e}. Pulando esta ação.")
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

def criar_pipeline_catboost(X_treino):
    """
    Cria e retorna o pipeline completo para o modelo CatBoost.
    """
    # Define colunas numéricas e categóricas
    colunas_numericas = X_treino.select_dtypes(include=np.number).columns.tolist()
    colunas_categoricas = ['ticker', 'tipo', 'setor', 'industria']
    
    # Define o pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), colunas_numericas),
            ('cat', 'passthrough', colunas_categoricas)
        ],
        remainder='drop'
    )
    
    # Define os índices das colunas categóricas para o CatBoost
    cat_features_indices = list(range(len(colunas_numericas), len(colunas_numericas) + len(colunas_categoricas)))
    
    # Cria a instância do modelo CatBoost
    catboost_model = CatBoostRegressor(
        random_state=42, 
        verbose=0,
        cat_features=cat_features_indices
    )
    
    # Cria e retorna o pipeline
    pipeline = Pipeline(steps=[
        ('preprocessador', preprocessor),
        ('modelo', catboost_model)
    ])
    
    return pipeline

def ajuste_pipeline_com_grid_search():
    pass


def gerar_metricas(y_true, y_pred):
    """
    Gera e imprime métricas de avaliação para modelos de regressão.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Adicione outras métricas que você precisar
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


def ajustar_hiper_parametros():
    pass