import json

import orjson


def save_json(filename: str, file: dict) -> None:
    """Save a JSON file.

    Args:
        filename (str): The path to the file to save.
        file (dict): The data to save.
    """
    with open(filename, "w") as outfile:
        json.dump(file, outfile)
    return None

def load_json(file_path: str) -> dict:
    """Load a JSON file.

    Args:
        file_path (str): The path to the file to load.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
    return data
