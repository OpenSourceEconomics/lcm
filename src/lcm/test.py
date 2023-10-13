from lcm.logger import get_logger

if __name__ == "__main__":
    logger = get_logger(debug_mode=False)

    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
