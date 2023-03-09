"""
Preprocessing section including: Formatting, Cleaning, Anonymization, Sampling
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from matplotlib import pyplot as plt
from numpy import float32
from sklearn.neighbors import LocalOutlierFactor
from core.config import NUMERICS
from core.decorators import with_logging, benchmark
from engineering.persistence_manager import DataType

@with_logging
@benchmark
def lof_observation(
        dataframe: pd.DataFrame, data_type: DataType = DataType.FIGURES
) -> pd.DataFrame:
    """
    This function identifies outliers with LOF method
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is FIGURES
    :type data_type: DataType
    :return: clean dataframe without outliers from LOF
    :rtype: pd.DataFrame
    """
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=NUMERICS)
    df_outlier: pd.DataFrame = df_num_cols.astype(float32)
    clf: LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=20, contamination=0.1)
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df: pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(stacked=True, xlim=[0, 20], color='r',
                   title='Visualization of outliers according to the LOF '
                         'method', style='.-')
    plt.savefig(f'{data_type.value}outliers.png')
    plt.show()
    th_val = np.sort(df_scores)[2]
    outliers: bool = df_scores > th_val
    dataframe: pd.DataFrame = dataframe.drop(
        df_outlier[~outliers].index).reset_index()
    return dataframe


def clear_outliers(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function remove the outliers from specific column
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :param column: Column name
    :type column: str
    :return: clean dataframe from outliers using IQR
    :rtype: pd.DataFrame
    """
    first_quartile: float = dataframe[column].quantile(0.25)
    third_quartile: float = dataframe[column].quantile(0.75)
    iqr: float = third_quartile - first_quartile
    lower: float = first_quartile - 1.5 * iqr
    upper: float = third_quartile + 1.5 * iqr
    print(f"{column}- Lower score: ", lower, "and upper score: ", upper)
    df_outlier = dataframe[column][(dataframe[column] > upper)]
    print(df_outlier)
    return dataframe


@with_logging
@benchmark
def random_oversample(x_train, y_train):
    """
    Random oversample the minority class using RandomOverSampler from
    the imbalanced-learn library.
    :param x_train: Feature matrix of the training set
    :type x_train: numpy.ndarray or pandas.DataFrame
    :param y_train: Target vector of the training set
    :type y_train: numpy.ndarray or pandas.Series
    :return: Random oversampled feature matrix and target vector
    :rtype: numpy.ndarray or pandas.DataFrame, numpy.ndarray or pandas.Series
    """
    random_over_sampler: RandomOverSampler = RandomOverSampler(random_state=42)
    x_ros, y_ros = random_over_sampler.fit_resample(x_train, y_train)
    return x_ros, y_ros


def smote(x_train, y_train):
    """
    Perform synthetic minority oversampling technique (SMOTE) using the
    imbalanced-learn library.
    :param x_train: Feature matrix of the training set
    :type x_train: numpy.ndarray or pandas.DataFrame
    :param y_train: Target vector of the training set
    :type y_train: numpy.ndarray or pandas.Series
    :return: SMOTE-sampled feature matrix and target vector
    :rtype: numpy.ndarray or pandas.DataFrame, numpy.ndarray or pandas.Series
    """
    smote_ob: SMOTE = SMOTE(random_state=42)
    x_sm, y_sm = smote_ob.fit_resample(x_train, y_train)
    return x_sm, y_sm


def adasyn(x_train, y_train):
    """
    Perform Adaptive Synthetic (ADASYN) oversampling using the
     imbalanced-learn library.
    :param x_train: Feature matrix of the training set
    :type x_train: numpy.ndarray or pandas.DataFrame
    :param y_train: Target vector of the training set
    :type y_train: numpy.ndarray or pandas.Series
    :return: ADASYN-sampled feature matrix and target vector
    :rtype: numpy.ndarray or pandas.DataFrame, numpy.ndarray or pandas.Series
    """
    adasyn_obj: ADASYN = ADASYN(random_state=42)
    x_ada, y_ada = adasyn_obj.fit_resample(x_train, y_train)
    return x_ada, y_ada
