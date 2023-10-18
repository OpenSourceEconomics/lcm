import logging


def get_logger(debug_mode):
    """Get a logger that logs to stdout.

    Args:
        debug_mode (bool): Whether to log debug messages.

    Returns:
        logging.Logger: Logger that logs to stdout.

    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("lcm")

    if debug_mode:
        logger.setLevel(logging.DEBUG)

    return logger
