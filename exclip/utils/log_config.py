import logging


def configure_logging(logPath, fileName="output"):
    """
    Configures logging for the application, setting up both file and console handlers
    with a specific log format. Logs are written to a file and also output to the console.
        logPath (str): The directory path where the log file will be saved.
        fileName (str, optional): The name of the log file (without extension). Defaults to "output".
    Returns:
        None
    """
    logFormatter = logging.Formatter("[%(asctime)s] %(message)s")
    rootLogger = logging.getLogger("xclip")

    fileHandler = logging.FileHandler(f"{logPath}/{fileName}.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.DEBUG)
    return None
