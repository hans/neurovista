from dataclasses import dataclass, field

from omegaconf import MISSING

from neurovista.types import Hemisphere


@dataclass
class DataConfig:

    subjects_dir: str = MISSING


@dataclass
class SceneConfig:
    background_color: str = 'white'


class PlotElement:
    pass


@dataclass
class BrainSurfaceConfig(PlotElement):
    _label = "brain_surface"

    hemi: Hemisphere = Hemisphere.LEFT
    surf: str = 'pial'

    surface_color: str = 'lightgrey'
    surface_opacity: float = 1.0


@dataclass
class ElectrodesConfig(PlotElement):
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
class PlotConfig:
    data: DataConfig = field(default_factory=DataConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    brain_surface: BrainSurfaceConfig = field(default_factory=BrainSurfaceConfig)
    electrodes: ElectrodesConfig = field(default_factory=ElectrodesConfig)