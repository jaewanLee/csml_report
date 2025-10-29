"""
Logging configuration for data processing pipeline.

This module provides centralized logging configuration for the entire pipeline.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO",
                  log_dir: Optional[Path] = None,
                  console_output: bool = True) -> logging.Logger:
    """
    Set up logging configuration for the data processing pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/data_processing/)
        console_output: Whether to output logs to console
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs/data_processing")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("data_processing")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "pipeline.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=2,
        encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

    logger.info(f"Logging configured: level={log_level}, dir={log_dir}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"data_processing.{name}")


def log_stage_start(logger: logging.Logger, stage_name: str, **kwargs) -> None:
    """
    Log the start of a processing stage.
    
    Args:
        logger: Logger instance
        stage_name: Name of the stage
        **kwargs: Additional context to log
    """
    logger.info(f"🚀 Starting {stage_name}")
    if kwargs:
        for key, value in kwargs.items():
            logger.info(f"   {key}: {value}")


def log_stage_end(logger: logging.Logger,
                  stage_name: str,
                  duration: Optional[float] = None,
                  **kwargs) -> None:
    """
    Log the end of a processing stage.
    
    Args:
        logger: Logger instance
        stage_name: Name of the stage
        duration: Duration in seconds
        **kwargs: Additional context to log
    """
    if duration is not None:
        logger.info(f"✅ Completed {stage_name} in {duration:.2f}s")
    else:
        logger.info(f"✅ Completed {stage_name}")

    if kwargs:
        for key, value in kwargs.items():
            logger.info(f"   {key}: {value}")


def log_data_info(logger: logging.Logger, data_name: str, shape: tuple,
                  memory_mb: float, **kwargs) -> None:
    """
    Log data information.
    
    Args:
        logger: Logger instance
        data_name: Name of the data
        shape: Data shape (rows, cols)
        memory_mb: Memory usage in MB
        **kwargs: Additional context to log
    """
    logger.info(
        f"📊 {data_name}: {shape[0]:,} rows × {shape[1]:,} cols ({memory_mb:.2f} MB)"
    )
    if kwargs:
        for key, value in kwargs.items():
            logger.info(f"   {key}: {value}")


def log_validation_result(logger: logging.Logger,
                          validation_name: str,
                          passed: bool,
                          details: Optional[str] = None) -> None:
    """
    Log validation results.
    
    Args:
        logger: Logger instance
        validation_name: Name of the validation
        passed: Whether validation passed
        details: Additional details
    """
    if passed:
        logger.info(f"✅ {validation_name} validation passed")
    else:
        logger.error(f"❌ {validation_name} validation failed")

    if details:
        logger.info(f"   Details: {details}")
