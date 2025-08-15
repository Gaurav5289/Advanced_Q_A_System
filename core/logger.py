import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standard logger.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a handler to write to standard output (console)
    # Check if handlers are already added to prevent duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
        
    return logger