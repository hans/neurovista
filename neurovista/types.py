from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np


Hemisphere: TypeAlias = Literal['lh', 'rh']



@dataclass
class ElectrodeAnatomy:
    coordinates: np.ndarray
    anatomy_labels: np.ndarray

    def __post_init__(self):
        assert self.coordinates.shape[1] == 3
        assert len(self.coordinates) == len(self.anatomy_labels)

    def __len__(self):
        return len(self.coordinates)