from dataclasses import dataclass, field
from typing import Optional

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
class ScalarBarConfig(PlotElement):
    """
    See pyvista.plotter.add_scalar_bar for more documentation.
    """
    scalar_bar_title: str = ""
    scalar_bar_n_labels: int = 5
    scalar_bar_italic: bool = False
    scalar_bar_bold: bool = False
    scalar_bar_title_font_size: Optional[int] = None
    scalar_bar_label_font_size: Optional[int] = None
    scalar_bar_color: str = 'black'
    scalar_bar_font_family: str = 'arial'
    scalar_bar_shadow: bool = False

    scalar_bar_width: Optional[float] = None
    scalar_bar_height: Optional[float] = None
    scalar_bar_position_x: Optional[float] = None
    scalar_bar_position_y: Optional[float] = None
    scalar_bar_vertical: Optional[bool] = None
    
    scalar_bar_fmt: Optional[str] = None
    scalar_bar_use_opacity: bool = True
    scalar_bar_outline: bool = False
    scalar_bar_nan_annotation: bool = False
    scalar_bar_below_label: Optional[str] = None
    scalar_bar_above_label: Optional[str] = None

    scalar_bar_background_color: Optional[str] = None
    scalar_bar_fill: bool = False

    def as_pv_dict(self):
        return {
            'title': self.scalar_bar_title,
            'n_labels': self.scalar_bar_n_labels,
            'italic': self.scalar_bar_italic,
            'bold': self.scalar_bar_bold,
            'title_font_size': self.scalar_bar_title_font_size,
            'label_font_size': self.scalar_bar_label_font_size,
            'color': self.scalar_bar_color,
            'font_family': self.scalar_bar_font_family,
            'shadow': self.scalar_bar_shadow,

            'width': self.scalar_bar_width,
            'height': self.scalar_bar_height,
            'position_x': self.scalar_bar_position_x,
            'position_y': self.scalar_bar_position_y,
            'vertical': self.scalar_bar_vertical,

            'fmt': self.scalar_bar_fmt,
            'use_opacity': self.scalar_bar_use_opacity,
            'outline': self.scalar_bar_outline,
            'nan_annotation': self.scalar_bar_nan_annotation,
            'below_label': self.scalar_bar_below_label,
            'above_label': self.scalar_bar_above_label,

            'background_color': self.scalar_bar_background_color,
            'fill': self.scalar_bar_fill,
        }


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
    scalar_bar: ScalarBarConfig = field(default_factory=ScalarBarConfig)
    brain_surface: BrainSurfaceConfig = field(default_factory=BrainSurfaceConfig)
    electrodes: ElectrodesConfig = field(default_factory=ElectrodesConfig)