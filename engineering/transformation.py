"""
script
"""
import pandas as pd
from numpy import uint8


def convert_diagnostic_column(column: str) -> pd.DataFrame:
    """
    Converts the column from 'M'/'B' to 1/0
    :param column: The column to convert
    :type column: str
    :return: The converted dataframe
    :rtype: pd.DataFrame
    """
    return uint8(1) if column == 'M' else uint8(0)
