"""
Models iteration script
"""
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from analysis.visualization import plot_confusion_matrix
from core.decorators import with_logging
from modelling.evaluation import evaluate_model
from modelling.modelling import predict_model
from models.nn_model import model_nn

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
def iterate_models(
        dataframe: pd.DataFrame, target_column: str = 'resultado_diagnóstico',
        gpu: bool = True
) -> None:
    """
    Iterates through a list of machine learning models and evaluates
     their performance on the input data
    :param dataframe: A pandas DataFrame containing the input data
    :type dataframe: pd.DataFrame
    :param target_column: The name of the target column in the input
     data
    :type target_column: str
    :param gpu: True if GPU is available to use, False otherwise
    :type gpu: bool
    :return: None
    :rtype: NoneType
    """
    boost_obj: list
    if gpu:
        boost_obj = [
            XGBClassifier(tree_method='gpu_hist', gpu_id=0),
            CatBoostClassifier(task_type="GPU", devices='0'),
            LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0)
        ]
    else:
        boost_obj = [XGBClassifier(), CatBoostClassifier(), LGBMClassifier()]
    models: list = [LogisticRegression(), SVC(), RandomForestClassifier(),
                    MultinomialNB(), DecisionTreeClassifier(),
                    KNeighborsClassifier(),
                    AdaBoostClassifier()]
    models.extend(boost_obj)
    model_names: list[str] = []
    boost_models: list[bool] = []
    for model in models:
        if isinstance(
                model, (XGBClassifier, CatBoostClassifier, LGBMClassifier)):
            model_names.append(model.__class__.__name__)
            boost_models.append(True)
        else:
            model_names.append(type(model).__name__)
            boost_models.append(False)
    for model, model_name, boost in zip(models, model_names, boost_models):
        print('\n\n', model_name)
        y_pred, y_test = predict_model(dataframe, model, target_column, boost)
        conf_matrix: np.ndarray = evaluate_model(y_pred, y_test)
        plot_confusion_matrix(
            conf_matrix, ['Maligno', 'Benigno'], model_name)


def iterate_nn_models(
        dataframe: pd.DataFrame, target_column: str = 'resultado_diagnóstico',
        gpu: bool = True
) -> None:
    """
    Iterate Neuronal Network Models
    :param dataframe: The dataframe to use
    :type dataframe: pd.DataFrame
    :param target_column: The target column to classify. The default is
     'resultado_diagnóstico'
    :type target_column: str
    :param gpu: True if GPU is available to use, False otherwise. The
    default is True
    :type gpu: bool
    :return: None
    :rtype: NoneType
    """
    for layer in ['LSTM', 'GRU']:
        print(layer)
        logger.info(layer)
        model_nn(dataframe, target_column, layer, gpu)
