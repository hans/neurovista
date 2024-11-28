from pathlib import Path
from typing import cast, Optional, TypeAlias

from matplotlib import cm
import matplotlib.colors
import mne
from mne.surface import _read_mri_surface
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.io import loadmat
import seaborn as sns

from neurovista import colors, config
from neurovista.types import ElectrodeAnatomy
from neurovista.types import Hemisphere



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


def _check_results(results: pd.DataFrame) -> pd.DataFrame:
    assert {"subject", "channel", "value"} <= set(results.columns)
    assert results.subject.nunique() == 1
    assert results.channel.value_counts().max() == 1
    if results.value.dtype != float:
        raise NotImplementedError()
    return results


def make_plotter(config: config.SceneConfig) -> pv.Plotter:
    pl = pv.Plotter()
    # pl.background_color = config.background_color

    return pl


def render_brain_surface(plotter, subject: str, subjects_dir: str,
                         surface_config: config.BrainSurface):
    surface_path = Path(subjects_dir) / subject / 'surf' / f'{surface_config.hemi}.{surface_config.surf}'
    surf = _read_mri_surface(surface_path)

    vertices = surf['rr'] * 1000  # type: ignore
    tris = np.concatenate([np.array([3] * len(surf['tris']))[:, None], surf['tris']], axis=1)  # type: ignore
    brain_mesh = pv.PolyData(vertices, tris)
    plotter.add_mesh(brain_mesh, color=surface_config.color, opacity=surface_config.opacity)

    plotter.camera_position = "yz"  # DEV assumes lh
    plotter.camera.azimuth = 180
    plotter.camera.zoom(1.5)


def plot_results(results: pd.DataFrame,
                 surface_config: config.BrainSurface = config.BrainSurface(),
                 electrode_config: config.Electrodes = config.Electrodes(),
                 scene_config: config.SceneConfig = config.SceneConfig(),
                 subjects_dir=None,
                 show=True, cmap="Oranges"):
    results = _check_results(results)
    if len(results) == 0:
        raise ValueError("No results to plot")

    subjects_dir = _check_subjects_dir(subjects_dir)
    subject = results.subject.iloc[0]
    electrodes = load_electrode_anatomy(subject, subjects_dir, warped=False)

    assert len(electrodes) >= results.channel.max()

    pl = make_plotter(scene_config)
    render_brain_surface(pl, subject, subjects_dir, surface_config)

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
        plot_electrodes += np.array(electrode_config.shift)[None, :]

        elec_mesh = pv.PolyData(plot_electrodes)
        elec_mesh['value'] = plot_data.value

        pl.add_mesh(
            elec_mesh,
            scalars='value',
            point_size=electrode_config.size,
            render_points_as_spheres=True,
            cmap=cmap,
            ambient=electrode_config.ambient,
            specular=electrode_config.specular,
            specular_power=electrode_config.specular_power,
            diffuse=electrode_config.diffuse,
            show_scalar_bar=True,
        )

    if background_idx is not None and len(background_idx) > 0:
        background_data = results.loc[background_idx]
        background_electrodes = electrodes.coordinates[background_data.channel - 1, :3]

        # pull out from surface
        background_electrodes += np.array(electrode_config.shift)[None, :]

        background_mesh = pv.PolyData(background_electrodes)

        pl.add_mesh(
            background_mesh,
            point_size=electrode_config.size / 2,
            render_points_as_spheres=True,
            color='grey',
            ambient=electrode_config.ambient,
            specular=electrode_config.specular,
            specular_power=electrode_config.specular_power,
            diffuse=electrode_config.diffuse,
            show_scalar_bar=False,
        )

    if show:
        pl.show()
    return pl



def plot_reconstruction(subject, subjects_dir=None,
                        brain_surface_config: config.BrainSurface = config.BrainSurface(),
                        electrodes_config: config.Electrodes = config.Electrodes(),
                        scene_config: config.SceneConfig = config.SceneConfig(),
                        show=True):
    subjects_dir = _check_subjects_dir(subjects_dir)
    electrodes = load_electrode_anatomy(subject, subjects_dir, warped=False)

    pl = make_plotter(scene_config)
    render_brain_surface(pl, subject, subjects_dir, brain_surface_config)

    all_labels = sorted(np.unique(electrodes.anatomy_labels))

    cmap = sns.color_palette("viridis", len(all_labels))

    for i, label in enumerate(all_labels):
        indices = np.where(electrodes.anatomy_labels == label)[0]
        point_cloud = pv.PolyData(electrodes.coordinates[indices, :3])
        glyph = point_cloud.glyph(orient=False, scale=False, geom=pv.Sphere(radius=1))

        color = [int(x) for x in colors.freesurfer.get(f"ctx-lh-{label}", cmap[i])]
        pl.add_mesh(glyph, color=color,
                    ambient=electrodes_config.ambient,
                    specular=electrodes_config.specular,
                    specular_power=electrodes_config.specular_power,
                    diffuse=electrodes_config.diffuse,
                    show_scalar_bar=False)

    # annotate with numbers
    pl.add_point_labels(electrodes.coordinates + np.array([-1, 0, 0]),
                        [f"{i + 1}" for i in range(len(electrodes))],
                        font_size=12, point_size=1)
    
    if show:
        pl.show()

    return pl