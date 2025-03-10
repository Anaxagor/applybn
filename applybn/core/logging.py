import logging
import sys
from rich.logging import RichHandler
from typing import Optional, Dict, Any, Union


# Default log format
DEFAULT_FORMAT = "%(message)s"
DEFAULT_DATE_FORMAT = "[%X]"


class LoggerFactory:
    """Factory class to create and configure loggers across the library."""

    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: Optional[str] = None,
                   level: Union[int, str] = logging.INFO,
                   format_string: str = DEFAULT_FORMAT,
                   date_format: str = DEFAULT_DATE_FORMAT,
                   rich_handler: bool = True) -> logging.Logger:
        """
        Get or create a logger with the specified name.

        Args:
            name: Logger name (defaults to calling module)
            level: Log level (defaults to INFO)
            format_string: Log format string
            date_format: Date format string
            rich_handler: Whether to use Rich for formatting

        Returns:
            Configured logger instance
        """
        if name is None:
            # Get the name of the calling module
            import inspect
            frame = inspect.stack()[1]
            name = frame.frame.f_globals["__name__"]

        # Return existing logger if already configured
        if name in cls._loggers:
            return cls._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Configure handler
        if rich_handler:
            handler = RichHandler(rich_tracebacks=True)
        else:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
            handler.setFormatter(formatter)

        logger.addHandler(handler)
        cls._loggers[name] = logger
        return logger

    @classmethod
    def configure_root_logger(cls, level: Union[int, str] = logging.INFO,
                              format_string: str = DEFAULT_FORMAT,
                              date_format: str = DEFAULT_DATE_FORMAT,
                              rich_handler: bool = True) -> None:
        """Configure the root logger with the given settings."""
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add new handler
        if rich_handler:
            handler = RichHandler(rich_tracebacks=True)
        else:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
            handler.setFormatter(formatter)

        root_logger.addHandler(handler)
        cls._loggers["root"] = root_logger

get_logger = LoggerFactory.get_logger
configure_root_logger = LoggerFactory.configure_root_logger
