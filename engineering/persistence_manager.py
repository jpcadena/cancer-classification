"""
Persistence script
"""
import logging
from enum import Enum
from typing import Union, Optional
import pandas as pd
from pandas.io.parsers import TextFileReader
from core.config import ENCODING

logger: logging.Logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    Data Type class based on Enum
    """
    RAW: str = 'data/raw/'
    PROCESSED: str = 'data/processed/'
    FIGURES: str = 'reports/figures/'


class PersistenceManager:
    """
    Persistence Manager class
    """

    @staticmethod
    def save_to_csv(
            data: Union[list[dict], pd.DataFrame],
            data_type: DataType = DataType.PROCESSED, filename: str = 'data'
    ) -> bool:
        """
        Save list of dictionaries as csv file
        :param data: list of tweets as dictionaries
        :type data: list[dict]
        :param data_type: folder where data will be saved
        :type data_type: DataType
        :param filename: name of the file
        :type filename: str
        :return: confirmation for csv file created
        :rtype: bool
        """
        dataframe: pd.DataFrame
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            if not data:
                return False
            dataframe = pd.DataFrame(data)
        dataframe.to_csv(f'{str(data_type)}{filename}.csv', index=False,
                         encoding=ENCODING)
        return True

    @staticmethod
    def load_from_csv(
            filename: str, data_type: DataType, chunk_size: int,
            dtypes: Optional[dict], converter: Optional[dict]
    ) -> pd.DataFrame:
        """
        Load dataframe from CSV using chunk scheme
        :param filename: name of the file
        :type filename: str
        :param data_type: Path where data will be saved
        :type data_type: DataType
        :param chunk_size: Number of chunks to split dataset
        :type chunk_size: int
        :param dtypes: Dictionary of columns and datatypes
        :type dtypes: dict
        :param converter: Dictionary with converter functions
        :type converter: dict
        :return: dataframe retrieved from CSV after optimization with chunks
        :rtype: pd.DataFrame
        """
        filepath: str = f'{data_type.value}{filename}'
        text_file_reader: TextFileReader = pd.read_csv(
            filepath, header=0, chunksize=chunk_size, encoding=ENCODING,
            converters=converter)
        dataframe: pd.DataFrame = pd.concat(
            text_file_reader, ignore_index=True)
        if dtypes:
            for key, value in dtypes.items():
                if value in (int, float):
                    dataframe[key] = pd.to_numeric(
                        dataframe[key], errors='coerce')
                    dataframe[key] = dataframe[key].astype(value)
                else:
                    dataframe[key] = dataframe[key].astype(value)
        return dataframe

    @staticmethod
    def save_to_pickle(
            dataframe: pd.DataFrame, filename: str = 'optimized_df.pkl'
    ) -> None:
        """
        Save dataframe to pickle file
        :param dataframe: dataframe
        :type dataframe: pd.DataFrame
        :param filename: name of the file
        :type filename: str
        :return: None
        :rtype: NoneType
        """
        dataframe.to_pickle(f'data/processed/{filename}')

    @staticmethod
    def load_from_pickle(filename: str = 'optimized_df.pkl') -> pd.DataFrame:
        """
        Load dataframe from Pickle file
        :param filename: name of the file to search and load
        :type filename: str
        :return: dataframe read from pickle
        :rtype: pd.DataFrame
        """
        dataframe: pd.DataFrame = pd.read_pickle(f'data/processed/{filename}')
        return dataframe
