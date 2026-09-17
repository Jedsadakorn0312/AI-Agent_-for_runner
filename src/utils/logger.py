import logging
import sys


def setup_logger():
    logger = logging.getLogger("AgenticRAG_Logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "\n[%(asctime)s] 🤖 %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()