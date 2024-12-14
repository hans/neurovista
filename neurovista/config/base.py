from dataclasses import dataclass
import re

from omegaconf import MISSING

from neurovista.types import Hemisphere


@dataclass
class DataConfig:
    _label = "data"

    subjects_dir: str = MISSING


@dataclass
class SceneConfig:
    _label = "scene"

    background_color: str = 'white'


@dataclass
class PlotElement:
    _label = "plot"


@dataclass
class BrainSurface(PlotElement):
    _label = "brain_surface"

    hemi: Hemisphere = 'lh'
    surf: str = 'pial'

    surface_color: str = 'lightgrey'
    surface_opacity: float = 1.0


@dataclass
class Electrodes(PlotElement):
    _label = "electrodes"

    ambient: float = 0.3261
    specular: float = 1.0
    specular_power: float = 16.0
    diffuse: float = 0.6995
    size: float = 10.0

    shift: tuple[float, float, float] = (-1.0, 0, 0)
    """
    Shift electrode positions by this amount before plotting.
    This can be useful to clearly separate electrodes from the brain surface.
    """