from collections import defaultdict
from functools import wraps
from typing import cast

from omegaconf import OmegaConf, SCMode

from neurovista.config.base import PlotConfig, BrainSurfaceConfig, ElectrodesConfig, SceneConfig, DataConfig


class RuntimeConfig:
    """Singleton for managing global runtime configuration."""
    _config = None

    @classmethod
    def initialize(cls, config=None):
        """Initialize the global configuration."""
        if isinstance(config, dict):
            cls._config = OmegaConf.create(config)
        elif config is None:
            cls._config = OmegaConf.structured(PlotConfig())
        else:
            cls._config = OmegaConf.create(OmegaConf.structured(config))

    @classmethod
    def get(cls):
        """Get the current global configuration."""
        if cls._config is None:
            raise RuntimeError("RuntimeConfig is not initialized.")
        return cls._config

    @classmethod
    def update(cls, **kwargs):
        """Update configuration dynamically."""
        if cls._config is None:
            raise RuntimeError("RuntimeConfig is not initialized.")
        cls._config.merge_with(OmegaConf.create(kwargs))

    @classmethod
    def load_from_yaml(cls, file_path: str):
        """Load configuration from a YAML file."""
        yaml_config = OmegaConf.load(file_path)
        if cls._config is None:
            cls._config = yaml_config
        else:
            cls._config.merge_with(yaml_config)



def route_kwargs_to_config(config, kwargs):
    """
    Route kwargs to the appropriate parts of a structured configuration.

    Args:
        config: The structured configuration object.
        kwargs: The keyword arguments to route.

    Returns:
        An overrides dict config which can be used with `OmegaConf.merge`.
    """
    # Iterate over each field in the config
    matched = {kwarg: False for kwarg in kwargs}
    overrides = {}
    for section, section_config in config.items():
        for key, value in kwargs.items():
            if key in section_config.keys():
                if section not in overrides:
                    overrides[section] = {}
                # Override the section's field
                overrides[section][key] = value
                matched[key] = True

    # Check for unmatched kwargs
    unmatched = [kwarg for kwarg, matched in matched.items() if not matched]
    if unmatched:
        raise ValueError(f"Unmatched kwargs: {unmatched}")
    return overrides


def get_config(kwargs: dict) -> PlotConfig:
    """
    Fetch configuration for a function invocation, merging kwarg config items
    with the global runtime config.
    """
    # Get the global configuration and apply overrides
    cfg = RuntimeConfig.get()
    overrides = route_kwargs_to_config(cfg, kwargs)
    cfg = OmegaConf.merge(cfg, overrides)
    
    return cast(PlotConfig,
                OmegaConf.to_container(cfg, structured_config_mode=SCMode.INSTANTIATE))