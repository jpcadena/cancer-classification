"""
Model prediction script
"""
import logging

import numpy as np
import pandas as pd
from core.decorators import with_logging, benchmark
from modelling.preprocessing import random_oversample
from modelling.train import training

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def predict_model(
        dataframe: pd.DataFrame, ml_model,
        target_column: str, boost: bool = False) -> tuple:
    """
    Predicts the target variable values using the provided model and
     returns the predicted and actual values.
    :param dataframe: The pandas dataframe containing the text data and
     the target variable
    :type dataframe: pd.DataFrame
    :param ml_model: The machine learning model to use for prediction
    :type ml_model: Any
    :param target_column: The name of the target variable column in the
     dataframe
    :type target_column: str
    :param boost: Whether to boost the model training by converting
     data to float32
    :type boost: bool
    :return: A tuple of predicted and actual values for the target
     variable
    :rtype: tuple
    """
    x_train, x_test, y_train, y_test = training(dataframe, target_column)
    x_train, y_train = random_oversample(x_train, y_train)
    # x_train, y_train = smote(x_train, y_train)  # equals to ROS
    # x_train, y_train = adasyn(x_train, y_train)  # less efficient
    if boost:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
    ml_model.fit(x_train, y_train)
    y_pred: np.ndarray = ml_model.predict(x_test)
    logger.info(x_train.shape, y_train.shape)
    return y_pred, y_test
