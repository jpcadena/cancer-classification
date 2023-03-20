"""
Extraction script
"""
from typing import Optional
import pandas as pd
from numpy import uint8, uint16, float32
from core.config import CHUNK_SIZE
from engineering.persistence_manager import PersistenceManager, DataType
from engineering.transformation import convert_diagnostic_column


def extract_raw_data(
        filename: str = 'RegistroCancer.csv',
        data_type: DataType = DataType.RAW, chunk_size: int = CHUNK_SIZE,
        d_types: Optional[dict] = None, converter: Optional[dict] = None
) -> pd.DataFrame:
    """
    Engineering method to extract raw data from csv file
    :param filename: Filename to extract data from. The default is
     'drugs_train.csv'
    :type filename: str
    :param data_type: Path where data will be saved: RAW or
     PROCESSED. The default is RAW
    :type data_type: DataType
    :param chunk_size: Number of chunks to split dataset. The default
     is CHUNK_SIZE
    :type chunk_size: int
    :param d_types: Optional dictionary to handle data types of columns.
     The default is None
    :type d_types: dict
    :param converter: Dictionary with converter functions. The default
     is None
    :type converter: dict
    :return: Dataframe with raw data
    :rtype: pd.DataFrame
    """
    if not d_types:
        d_types = {
            'id': uint8, 'radio': uint8, 'textura': uint8,
            'perímetro': uint8, 'área': uint16, 'suavidad': float32,
            'compacidad': float32, 'simetría': float32,
            'dimensión_fractal': float32}
    if not converter:
        converter: dict = {'resultado_diagnóstico': convert_diagnostic_column}
    dataframe: pd.DataFrame = PersistenceManager.load_from_csv(
        filename=filename, data_type=data_type, chunk_size=chunk_size,
        dtypes=d_types, converter=converter)
    return dataframe
