from collections import defaultdict
from functools import wraps

from neurovista.config.base import BrainSurface, Electrodes, SceneConfig, DataConfig



# Fancy decorators for converting kwargs into structured configs.
# All config propreties must have unique names
def takes_config(*config_classes):
    # Validate unique names
    names = defaultdict(list)
    for config_cls in config_classes:
        for prop in config_cls.__dataclass_fields__:
            names[prop].append(config_cls)
    for prop, classes in names.items():
        if len(classes) > 1:
            raise ValueError(f"Property {prop} is defined in multiple config classes: {classes}")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            configs = {config_cls: config_cls()
                       for cls, config_cls in zip(config_classes, config_classes)}
            
            to_delete = []
            for kwarg in kwargs:
                for config_cls in config_classes:
                    if hasattr(config_cls, kwarg):
                        configs[config_cls].__dict__[kwarg] = kwargs[kwarg]
                        to_delete.append(kwarg)
                        break
            for kwarg in to_delete:
                del kwargs[kwarg]
            
            # Make sure there are no clashes
            for config_cls in config_classes:
                if f"{config_cls._label}_config" in kwargs:
                    raise ValueError(f"Cannot pass {config_cls} as a keyword argument")
            config_kwargs = {f"{config_cls._label}_config": config
                             for config_cls, config in configs.items()}
            new_kwargs = {**kwargs, **config_kwargs}

            return func(*args, **new_kwargs)
        return wrapper
    return decorator