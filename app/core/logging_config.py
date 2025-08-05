# app/core/logging_config.py - Enhanced with better formatting
import logging
import sys
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Colored formatter for better log visibility"""
    
    grey = "\x1b[38;21m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        log_color = self.FORMATS.get(record.levelno, self.grey)
        record.levelname = f"{log_color}{record.levelname}{self.reset}"
        return super().format(record)

def setup_logging():
    """Configures structured logging for the application with colors."""
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = ColoredFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Silence overly verbose logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)