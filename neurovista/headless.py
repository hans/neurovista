"""
Helpers for rendering static images from a headless machine.
"""


from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv


def plot_matplotlib(pl, **savefig_kwargs):
    """Render a matplotlib plot to Jupyter output"""
    from base64 import b64encode
    from io import BytesIO
    from IPython.display import display, HTML
    figdata = BytesIO()
    pl.show(screenshot=figdata)
    pl.close()
    return display(HTML('<img src="data:image/png;base64,{0}"/>'.format(b64encode(figdata.getvalue()).decode())))


def trim_img(img: np.ndarray):
    nonzero_cols = ((img != 255).any(axis=2).sum(axis=0) != 0).nonzero()[0]
    trim_start_x, trim_end_x = nonzero_cols[0], nonzero_cols[-1] + 1

    nonzero_rows = ((img != 255).any(axis=2).sum(axis=1) != 0).nonzero()[0]
    trim_start_y, trim_end_y = nonzero_rows[0], nonzero_rows[-1] + 1

    return img[trim_start_y:trim_end_y, trim_start_x:trim_end_x]


class MplPlotter(object):
    def __init__(self):
        self._figures = []

    def __enter__(self):
        return self
    
    def add(self, fig):
        self._figures.append(fig)
    
    def __exit__(self, type, value, traceback):
        for fig in self._figures:
            plot_matplotlib(fig)
        self._figures = []


class HeadlessPlotter(object):
    def __init__(self, outdir):
        self.outdir = outdir

    def __enter__(self):
        return self
    
    def add(self, name: str, pl):
        pl.screenshot(str(Path(self.outdir) / name))
        
    def __exit__(self, type, value, traceback):
        pass