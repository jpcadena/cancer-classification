"""
First Analysis script
"""
import logging
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 101)


def analyze_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Analyze the dataframe and its columns with inference statistics
    :param dataframe: DataFrame to analyze
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    print(dataframe.head())
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.info(memory_usage='deep'))
    print(dataframe.memory_usage(deep=True))
    print(dataframe.describe(include='all', datetime_is_numeric=True))
    non_numeric_df = dataframe.select_dtypes(exclude=[
        'uint8', 'uint16', 'uint32', 'uint64',
        'int8', 'int16', 'int32',
        'int64',
        'float16', 'float32', 'float64'])
    for column in non_numeric_df.columns:
        print(non_numeric_df[column].value_counts())
        print(non_numeric_df[column].unique())
        print(non_numeric_df[column].value_counts(normalize=True) * 100)


def find_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove missing values from the dataframe
    :param dataframe: Dirty dataframe to remove missing values from
    :type dataframe: pd.DataFrame
    :return: Cleaned dataframe
    :rtype: pd.DataFrame
    """
    print("Missing values")
    missing_values: pd.Series = (dataframe.isnull().sum())
    print(missing_values)
    if missing_values.any():
        logger.warning("FOUND MISSING VALUES")
        print(missing_values[missing_values > 0])
        print(missing_values[missing_values > 0] / dataframe.shape[0] * 100)
        dataframe = dataframe.dropna()
    return dataframe
