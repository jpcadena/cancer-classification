"""
Neural Network using TensorFlow script
"""
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras import backend as K
from keras.layers import LSTM, GRU, Dense
from analysis.visualization import plot_confusion_matrix
from core.decorators import benchmark, with_logging
from modelling.evaluation import evaluate_model
from modelling.preprocessing import random_oversample
from modelling.train import training

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def predict_nn(
        dataframe: pd.DataFrame, target_column: str, layer: str,
        gpu: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the output for the neural network using the Bag of Words
     matrix and a pandas DataFrame
    :param dataframe: The pandas DataFrame with the target column
    :type dataframe: pd.DataFrame
    :param target_column: The target column name
    :type target_column: str
    :param layer: The type of layer to use (LSTM or GRU)
    :type layer: str
    :param gpu: True if GPU is available to use, False otherwise
    :type gpu: bool
    :return: A tuple with the predicted output and the true output
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if gpu:
        print(tf.test.is_built_with_cuda())
        print(tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ",
              len(tf.config.list_physical_devices('GPU')))
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)
        session = tf.compat.v1.Session()
        K.set_session(session)
        tf.config.experimental.set_visible_devices(
            [tf.config.experimental.list_physical_devices('GPU')[0]], 'GPU')
    sequential: Sequential = Sequential()
    x_train, x_test, y_train, y_test = training(dataframe, target_column)
    x_train, y_train = random_oversample(x_train, y_train)
    input_shape = input_shape = (x_train.shape[1], 1)
    if layer == 'LSTM':
        sequential.add(
            LSTM(100, input_shape=input_shape,
                 return_sequences=False))
    elif layer == 'GRU':
        sequential.add(
            GRU(100, input_shape=input_shape,
                return_sequences=False))
    sequential.add(Dense(1, activation='sigmoid'))
    sequential.compile(loss='binary_crossentropy', optimizer='adam',
                       metrics=['accuracy'])
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    sequential.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    y_pred: np.ndarray = sequential.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    return y_pred, y_test


@with_logging
def model_nn(
        dataframe: pd.DataFrame, target_column: str, layer: str, gpu: bool
) -> None:
    """
    Model Neural Network
    :param dataframe: The pandas DataFrame with the target column
    :type dataframe: pd.DataFrame
    :param target_column: The target column name
    :type target_column: str
    :param layer: The type of layer to use (LSTM or GRU)
    :type layer: str
    :param gpu: True if GPU is available to use, False otherwise
    :type gpu: bool
    :return: None
    :rtype: NoneType
    """
    y_pred_lstm, y_test = predict_nn(dataframe, target_column, layer, gpu)
    conf_matrix: np.ndarray = evaluate_model(y_pred_lstm, y_test)
    plot_confusion_matrix(
        conf_matrix, ['Maligno', 'Benigno'], 'Tensorflow NN')
