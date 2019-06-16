import logging
import pandas as pd
from .utils import log_config, setup_logging
from .classifier import SimpleCNN
from grabsafety import cutoff


setup_logging(log_config)
logger = logging.getLogger(__name__)
safety_model = SimpleCNN()


def predict_danger(df, return_prob=True, cutoff=cutoff):
    """ The function guesses industries and probabilities based on company names. """
    try:
        if isinstance(df, pd.DataFrame):
            return safety_model.predict(df, return_prob, cutoff)
        else:
            logger.error('Input trip data is not in pd.DataFrame format.')
    except Exception as e:
        logger.exception(f'predict_industries: {e}')
