"""
Centralized logging module for the Binance Futures trading bot.

This module provides logging functionality to console and rotating log files,
with different log levels (debug, info, warning, error, critical).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

import config

# Create logger
logger = logging.getLogger("trading_bot")

# Set the log level based on config
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setup_logger() -> logging.Logger:
    """
    Set up and configure the logger with console and file handlers.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Set the log level
    log_level = LOG_LEVELS.get(config.LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatters
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Logger initialized with level: %s", config.LOG_LEVEL)
    return logger

def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if not logger.handlers:
        return setup_logger()
    return logger

# Convenience functions for logging
def debug(msg: str, *args, **kwargs) -> None:
    """Log a debug message."""
    get_logger().debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs) -> None:
    """Log an info message."""
    get_logger().info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs) -> None:
    """Log a warning message."""
    get_logger().warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs) -> None:
    """Log an error message."""
    get_logger().error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs) -> None:
    """Log a critical message."""
    get_logger().critical(msg, *args, **kwargs)

def exception(msg: str, *args, **kwargs) -> None:
    """Log an exception with traceback."""
    get_logger().exception(msg, *args, **kwargs)
