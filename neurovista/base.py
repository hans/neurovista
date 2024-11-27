from pathlib import Path
from typing import cast

from matplotlib import cm
import matplotlib.colors
import mne
from mne.surface import _read_mri_surface
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.io import loadmat
import seaborn as sns

from neurovista import colors
from neurovista.types import Hemisphere
from neurovista.types import ElectrodeAnatomy


def load_electrode_anatomy(subject, subjects_dir, warped=False):
    if warped:
        electrode_path = Path(subjects_dir) / subject / 'elecs' / 'TDT_elecs_all_warped.mat'
    else:
        electrode_path = Path(subjects_dir) / subject / 'elecs' / 'TDT_elecs_all.mat'

    elec_data = loadmat(electrode_path, simplify_cells=True)

    coordinates = elec_data['elecmatrix'][:, :3]
    anatomy_labels = elec_data['anatomy'][:, 3]
    return ElectrodeAnatomy(coordinates, anatomy_labels)


def _check_subjects_dir(subjects_dir) -> str:
    if subjects_dir is None:
        subjects_dir = cast(str, mne.get_config('SUBJECTS_DIR', None))
    assert subjects_dir is not None, "subjects_dir must be provided or set in MNE config"
    return subjects_dir


def plot_results(results: pd.DataFrame, subjects_dir=None,
                 hemi: Hemisphere = 'lh', surf='pial', show=False,
                 cmap="Oranges",
                 **kwargs):
    assert {"subject", "channel", "value"} <= set(results.columns)
    assert results.subject.nunique() == 1
    assert results.channel.value_counts().max() == 1
    if results.value.dtype != float:
        raise NotImplementedError()

    if len(results) == 0:
        raise ValueError("No results to plot")

    subjects_dir = _check_subjects_dir(subjects_dir)
    subject = results.subject.iloc[0]
    electrodes = load_electrode_anatomy(subject, subjects_dir, warped=False)

    assert len(electrodes) >= results.channel.max()

    surface_path = Path(subjects_dir) / subject / 'surf' / f'{hemi}.{surf}'
    surf = _read_mri_surface(surface_path)

    pl = pv.Plotter()

    brain_mesh = pv.PolyData(surf['rr'] * 1000, np.concatenate([np.array([3] * len(surf['tris']))[:, None], surf['tris']], axis=1))
    pl.add_mesh(brain_mesh, color='lightgrey', opacity=0.95)
    pl.camera_position = "yz"  # DEV assumes lh
    pl.camera.azimuth = 180
    pl.camera.zoom(1.5)

    # plot electrodes
    ambient = 0.3261
    specular = 1
    specular_power = 16
    diffuse = 0.6995
    
    if "background" in results.columns:
        plot_electrode_idx = results[~results.background].index
        background_idx = results[results.background].index
    else:
        plot_electrode_idx = results.index
        background_idx = None
    
    plot_data = results.loc[plot_electrode_idx]
    if len(plot_data) > 0:
        plot_electrodes = electrodes.coordinates[plot_data.channel - 1, :3]

        # pull out from surface
        plot_electrodes += np.array([-1, 0, 0])[None, :]

        elec_mesh = pv.PolyData(plot_electrodes)
        elec_mesh['value'] = plot_data.value

        pl.add_mesh(
            elec_mesh,
            scalars='value',
            point_size=10,
            render_points_as_spheres=True,
            cmap=cmap,
            ambient=ambient,
            specular=specular,
            specular_power=specular_power,
            diffuse=diffuse,
            show_scalar_bar=True,
        )

    if background_idx is not None and len(background_idx) > 0:
        background_data = results.loc[background_idx]
        background_electrodes = electrodes.coordinates[background_data.channel - 1, :3]

        # pull out from surface
        background_electrodes += np.array([-1, 0, 0])[None, :]

        background_mesh = pv.PolyData(background_electrodes)

        pl.add_mesh(
            background_mesh,
            point_size=5,
            render_points_as_spheres=True,
            color='grey',
            ambient=ambient,
            specular=specular,
            specular_power=specular_power,
            diffuse=diffuse,
            show_scalar_bar=False,
        )

    if show:
        pl.show()
    return pl



def plot_reconstruction(subject, subjects_dir=None,
                        hemi: Hemisphere = 'lh',
                        surf='pial', show=False, **kwargs):
    subjects_dir = _check_subjects_dir(subjects_dir)

    electrodes = load_electrode_anatomy(subject, subjects_dir, warped=False)

    surface_path = Path(subjects_dir) / subject / 'surf' / f'{hemi}.{surf}'
    surf = _read_mri_surface(surface_path)

    all_labels = sorted(np.unique(electrodes.anatomy_labels))

    pl: pv.plotting.plotter.Plotter = pv.Plotter()
    
    brain_mesh = pv.PolyData(surf['rr'] * 1000, np.concatenate([np.array([3] * len(surf['tris']))[:, None], surf['tris']], axis=1))
    pl.add_mesh(brain_mesh, color='lightgrey')
    pl.camera_position = "yz"  # DEV assumes lh
    pl.camera.azimuth = 180
    pl.camera.zoom(1.5)

    # plot electrodes
    ambient = 0.3261
    specular = 1
    specular_power = 16
    diffuse = 0.6995

    cmap = sns.color_palette("viridis", len(all_labels))

    for i, label in enumerate(all_labels):
        indices = np.where(electrodes.anatomy_labels == label)[0]
        point_cloud = pv.PolyData(electrodes.coordinates[indices, :3])
        glyph = point_cloud.glyph(orient=False, scale=False, geom=pv.Sphere(radius=1))

        color = [int(x) for x in colors.freesurfer.get(f"ctx-lh-{label}", cmap[i])]
        pl.add_mesh(glyph, color=color, ambient=ambient, specular=specular, specular_power=specular_power, diffuse=diffuse)

    # annotate with numbers
    pl.add_point_labels(electrodes.coordinates + np.array([-1, 0, 0]),
                        [f"{i + 1}" for i in range(len(electrodes))],
                        font_size=12, point_size=1)
    
    if show:
        pl.show()

    return pl