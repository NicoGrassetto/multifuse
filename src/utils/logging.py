"""
Logging utilities for the MultiFuse project.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any


from typing import Optional

def setup_logging(logging_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        logging_config: Dictionary containing logging configuration
    """
    if logging_config is None:
        logging_config = {}
    
    # Default configuration
    level = logging_config.get('level', 'INFO')
    format_str = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if 'file' in logging_config:
        log_file = Path(logging_config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
