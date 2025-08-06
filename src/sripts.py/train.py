# ================================================================
# FLUXO PRINCIPAL DO SCRIPT
# ================================================================

if __name__ == "__main__":
    # Defina os caminhos dos seus arquivos de dados e onde salvar o modelo
    CAMINHO_DADOS_TREINO = 'caminho/para/seu/treino.csv'
    CAMINHO_DADOS_TESTE = 'caminho/para/seu/teste.csv'
    CAMINHO_MODELO = 'modelo_catboost.pkl'
    
    # 1. Carregar dados
    X_treino, y_treino, X_teste, y_teste = carregar_e_preparar_dados(CAMINHO_DADOS_TREINO, CAMINHO_DADOS_TESTE)
    
    # 2. Criar pipeline
    pipeline_cb = criar_pipeline_catboost(X_treino)
    
    # 3. Treinar e avaliar
    pipeline_cb = treinar_e_avaliar_modelo(pipeline_cb, X_treino, y_treino, X_teste, y_teste)
    
    # 4. Salvar o modelo treinado
    salvar_modelo(pipeline_cb, CAMINHO_MODELO)