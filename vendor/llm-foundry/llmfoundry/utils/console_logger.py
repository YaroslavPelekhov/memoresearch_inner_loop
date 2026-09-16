import logging


def setup_logger(name: str, logging_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s |  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        validate=True,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
