import json

import orjson


def save_json(filename, file):
    with open(filename, "w") as outfile:
        json.dump(file, outfile)


def load_json(file_path):
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
    return data
