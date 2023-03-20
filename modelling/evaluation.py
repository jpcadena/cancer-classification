"""
Evaluation script including improvement and tests.
"""
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, confusion_matrix, classification_report
from core.decorators import with_logging, benchmark

logger: logging.Logger = logging.getLogger(__name__)


@with_logging
@benchmark
def evaluate_model(y_pred: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Evaluate a binary classification ml_model based on several metrics.
    :param y_pred: Predicted binary labels
    :type y_pred: np.ndarray
    :param y_test: True binary values'
    :type y_test: np.ndarray
    :return: The confusion matrix
    :rtype: np.ndarray
    """
    accuracy: float = accuracy_score(y_test, y_pred)
    precision: float = precision_score(y_test, y_pred)
    recall: float = recall_score(y_test, y_pred)
    f1_s: float = f1_score(y_test, y_pred)
    roc_auc: float = roc_auc_score(y_test, y_pred)
    conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)
    logger.info('Accuracy: %s', accuracy)
    logger.info('Precision: %s', precision)
    logger.info('Recall: %s', recall)
    logger.info('F1 Score: %s', f1_s)
    logger.info('ROC AUC: %s', roc_auc)
    logger.info('Confusion Matrix: %s', conf_matrix)
    print(conf_matrix)
    report = classification_report(y_test, y_pred)
    logger.info('Classification report: %s', report)
    print(report)
    return conf_matrix
