"""YAML configuration loader with validation."""

import yaml


def load_config(path: str) -> dict:
    """Load and return configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist at *path*.
        KeyError: If a required top-level key is absent from the config.
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            "Please create a config.yaml at the specified path."
        )

    if config is None:
        config = {}

    required_keys = [
        "data",
        "features",
        "windows",
        "diffusion",
        "env",
        "agents",
        "backtest",
        "evaluation",
        "seed",
    ]

    for key in required_keys:
        if key not in config:
            raise KeyError(
                f"Required configuration key '{key}' is missing from '{path}'."
            )

    return config
