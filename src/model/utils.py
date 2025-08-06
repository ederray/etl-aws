"""Funções utilitárias para uso do modelo"""
import logging
import pickle

# instância do objeto logger
logger = logging.getLogger(__name__)

def salvar_modelo(modelo, caminho_salvar):
    """
    Salva o modelo treinado em um arquivo.
    """
    pickle.dump(modelo, caminho_salvar)
    print(f"\nModelo salvo em: {caminho_salvar}")

def carregar_modelo():
    pass






