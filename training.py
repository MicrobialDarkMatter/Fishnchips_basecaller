import src.api as api
import src.data_api as data_api
from src.utils.config_loader import load_config

config = load_config('./configs/catcaller.json')

api.train(config, 'debug_clean', new_training=True)