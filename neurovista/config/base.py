from dataclasses import dataclass

from omegaconf import MISSING

from neurovista.types import Hemisphere


@dataclass
class DataConfig:
    subjects_dir: str = MISSING


@dataclass
class SceneConfig:
    background_color: str = 'white'


@dataclass
class PlotElement:
    pass


@dataclass
class BrainSurface(PlotElement):
    hemi: Hemisphere = 'lh'
    surf: str = 'pial'

    color: str = 'lightgrey'
    opacity: float = 1.0


@dataclass
class Electrodes(PlotElement):
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


@dataclass
class Scene:
    elements: list[PlotElement] = MISSING