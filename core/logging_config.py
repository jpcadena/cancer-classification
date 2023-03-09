"""
Logging script for Core module
"""
import logging
import os
from datetime import datetime


def setup_logging(log_level: int = logging.DEBUG) -> None:
    """
    Setup logging
    :param log_level: Level of logging
    :type log_level: int
    :return: None
    :rtype: NoneType
    """
    current_date: str = datetime.today().strftime('%d-%b-%Y-%H-%M-%S')
    current_file_directory: str = os.path.dirname(os.path.abspath(__file__))
    project_root: str = current_file_directory
    while os.path.basename(project_root) != "cancer-classification":
        project_root = os.path.dirname(project_root)
    log_filename: str = f'log-{current_date}.log'
    filename_path: str = f'{project_root}/logs/{log_filename}'

    logger: logging.Logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    formatter = logging.Formatter(
        '[%(name)s][%(asctime)s][%(levelname)s][%(module)s][%(funcName)s][%('
        'lineno)d]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler: logging.FileHandler = logging.FileHandler(filename_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Logger started')
