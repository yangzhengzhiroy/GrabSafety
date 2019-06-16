# -*- coding: UTF-8 -*-
"""
This module creates model object.
"""
import os
import logging
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from .utils import log_config, setup_logging
from grabsafety import PARENT_DIR, cutoff
from .datahandler import TripRecord


setup_logging(log_config)
logger = logging.getLogger(__name__)


class SimpleCNN(object):
    """ Simple 1D CNN model. """
    _classifier_weights_file_name = 'safety_weights.h5'
    _classifier_graph_file_name = 'safety_graph.json'
    _classifier_weights_path = os.path.join(PARENT_DIR, 'grabsafety/models', _classifier_weights_file_name)
    _classifier_graph_path = os.path.join(PARENT_DIR, 'grabsafety/models', _classifier_graph_file_name)

    def __init__(self, pad_size=3600, padding='post'):
        self._pad_size = pad_size
        self._padding = padding
        self._trip_record = TripRecord(pad_size=self._pad_size, padding=self._padding)
        self._model = None

    def _prepare_data(self, df):
        """ Prepare the data input with raw df. """
        prepared_data = self._trip_record.trip_clean(df)
        return prepared_data

    def load(self, model_weights_path=_classifier_weights_path, model_graph_path=_classifier_graph_path):
        """ Load the existing model. """
        K.clear_session()
        with open(model_graph_path, 'r') as f:
            model_graph = f.read()
        self._model = model_from_json(model_graph)
        self._model.load_weights(model_weights_path)

    def predict(self, df, return_prob=True, cutoff=cutoff):
        """ This function predicts danger probability based on input. """
        if not self._model:
            self.load()
        data_arr = self._prepare_data(df)
        y_pred_prob = self._model.predict(data_arr)
        y_pred_prob = y_pred_prob.flatten()
        if return_prob:
            return y_pred_prob
        else:
            y_pred_prob = np.where(y_pred_prob >= cutoff, 1, 0)
            return y_pred_prob
