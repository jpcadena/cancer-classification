"""
Main script
"""
import logging
import pandas as pd
from analysis import numerical_eda, visualize_data
from core import logging_config
from engineering.extraction import extract_raw_data
from engineering.persistence_manager import PersistenceManager
from modelling.preprocessing import lof_observation
from models.models import iterate_models, iterate_nn_models

logging_config.setup_logging()
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to execute
    :return: None
    :rtype: NoneType
    """
    logger.info("Running main method")
    dataframe: pd.Dataframe = extract_raw_data()
    dataframe = numerical_eda(dataframe)
    dataframe = dataframe.drop('id', axis=1)
    dataframe = lof_observation(dataframe)
    visualize_data(dataframe)
    PersistenceManager.save_to_pickle(dataframe)
    iterate_models(dataframe)
    iterate_nn_models(dataframe)


if __name__ == '__main__':
    logger.info("First log message")
    main()
    logger.info("End of the program execution")
