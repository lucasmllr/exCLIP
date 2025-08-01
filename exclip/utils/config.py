import argparse
import json
import logging

from dotwiz import DotWiz


def get_config(args: argparse.Namespace) -> DotWiz:
    """
    Loads and parses a configuration file in JSON format, wraps it in a DotWiz object for attribute-style access, and logs the configuration.

    args (argparse.Namespace): Command-line arguments containing the path to the configuration file as 'config'.

    DotWiz: Configuration object with attribute-style access to configuration parameters.
    """
    config = json.load(open(args.config))
    config = DotWiz(config)

    logger = logging.getLogger("exclip")
    logger.info(config)

    return config
