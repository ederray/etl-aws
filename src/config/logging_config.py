import logging
import logging.config
from pathlib import Path

# construção de uma pasta que registra os logs em um arquivo
LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# função para setup de logging
def setup_logging(log_level=logging.INFO):
    logging_config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            },
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': log_level,
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': LOGS_DIR / 'project.log',
                'mode': 'a',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': log_level,
        },
    }

    # aplicação global das configurações
    logging.config.dictConfig(logging_config)
