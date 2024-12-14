from pathlib import Path
from typing import cast, Optional, TypeAlias

from matplotlib import cm
import matplotlib.colors
import mne
from mne.surface import _read_mri_surface
import numpy as np
from omegaconf import OmegaConf, SCMode
import pandas as pd
import pyvista as pv
from scipy.io import loadmat
import seaborn as sns

from neurovista import colors, config
from neurovista.config import get_config
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


def _check_subjects_dir(data_config: config.DataConfig) -> str:
    subjects_dir = data_config.subjects_dir
    if subjects_dir is None:
        subjects_dir = cast(str, mne.get_config('SUBJECTS_DIR', None))
    assert subjects_dir is not None, "subjects_dir must be provided or set in MNE config"
    return subjects_dir


def _check_results(results: pd.DataFrame, multi_subject=False) -> pd.DataFrame:
    assert {"subject", "channel", "value"} <= set(results.columns)

    assert results.channel.min() >= 1, "Channels should be 1-indexed"
    assert results.groupby("subject").channel.value_counts().max() == 1
    if not multi_subject:
        assert results.subject.nunique() == 1
    if results.value.dtype != float:
        raise NotImplementedError()
    return results


def make_plotter(config: config.SceneConfig) -> pv.Plotter:
    pl = pv.Plotter()
    # pl.background_color = config.background_color

    return pl


def render_brain_surface(plotter, subject: str, subjects_dir: str,
                         surface_config: config.BrainSurfaceConfig):
    surface_path = Path(subjects_dir) / subject / 'surf' / f'{surface_config.hemi.value}.{surface_config.surf}'
    surf = _read_mri_surface(surface_path)

    vertices = surf['rr'] * 1000  # type: ignore
    tris = np.concatenate([np.array([3] * len(surf['tris']))[:, None], surf['tris']], axis=1)  # type: ignore
    brain_mesh = pv.PolyData(vertices, tris)
    plotter.add_mesh(brain_mesh, color=surface_config.surface_color, opacity=surface_config.surface_opacity)

    plotter.camera_position = "yz"
    if surface_config.hemi.value == "lh":
        plotter.camera.azimuth = 180
    elif surface_config.hemi.value == "rh":
        plotter.camera.azimuth = 0
    plotter.camera.zoom(1.5)


def plot_results_multi_subject(results: pd.DataFrame, show=True, cmap="Oranges",
                               **kwargs):
    cfg = get_config(kwargs)

    subjects_dir = _check_subjects_dir(cfg.data)
    results = _check_results(results, multi_subject=True)
    if len(results) == 0:
        raise ValueError("No results to plot")

    subjects = sorted(results.subject.unique())
    results = results.set_index("subject")
    electrodes = {subject: load_electrode_anatomy(subject, subjects_dir, warped=True)
                  for subject in subjects}
    for subject, subject_electrodes in electrodes.items():
        assert len(subject_electrodes) >= results.loc[subject].channel.max()
    results = results.set_index("channel", append=True)

    pl = make_plotter(cfg.scene)
    render_brain_surface(pl, "cvs_avg35_inMNI152", subjects_dir, cfg.brain_surface)

    if "background" in results.columns:
        plot_electrode_idx = results[~results.background].index
        background_idx = results[results.background].index
    else:
        plot_electrode_idx = results.index
        background_idx = None
    
    plot_data = results.loc[plot_electrode_idx]
    for plot_subject, plot_subject_results in plot_data.groupby("subject"):
        channel_idxs = plot_subject_results.index.get_level_values("channel") - 1
        plot_electrodes = electrodes[plot_subject].coordinates[channel_idxs, :3]

        # pull out from surface
        plot_electrodes += np.array(cfg.electrodes.shift)[None, :]

        elec_mesh = pv.PolyData(plot_electrodes)
        elec_mesh['value'] = plot_subject_results.value.to_numpy()

        pl.add_mesh(
            elec_mesh,
            scalars='value',
            point_size=cfg.electrodes.size,
            render_points_as_spheres=True,
            cmap=cmap,
            ambient=cfg.electrodes.ambient,
            specular=cfg.electrodes.specular,
            specular_power=cfg.electrodes.specular_power,
            diffuse=cfg.electrodes.diffuse,
            show_scalar_bar=True,
        )

    if background_idx is not None and len(background_idx) > 0:
        background_data = results.loc[background_idx]

        for plot_subject, plot_subject_background in background_data.groupby("subject"):
            channel_idxs = plot_subject_background.index.get_level_values("channel") - 1
            background_electrodes = electrodes[plot_subject].coordinates[channel_idxs, :3]

            # pull out from surface
            background_electrodes += np.array(cfg.electrodes.shift)[None, :]

            background_mesh = pv.PolyData(background_electrodes)

            pl.add_mesh(
                background_mesh,
                point_size=cfg.electrodes.size / 2,
                render_points_as_spheres=True,
                color='grey',
                ambient=cfg.electrodes.ambient,
                specular=cfg.electrodes.specular,
                specular_power=cfg.electrodes.specular_power,
                diffuse=cfg.electrodes.diffuse,
                show_scalar_bar=False,
            )

    if show:
        pl.show()
    return pl


def plot_results(results: pd.DataFrame, warped=False, show=True, cmap="Oranges",
                 **kwargs):
    cfg = get_config(kwargs)

    subjects_dir = _check_subjects_dir(cfg.data)
    results = _check_results(results)
    if len(results) == 0:
        raise ValueError("No results to plot")

    subject = results.subject.iloc[0]
    electrodes = load_electrode_anatomy(subject, subjects_dir, warped=warped)

    assert len(electrodes) >= results.channel.max()

    pl = make_plotter(cfg.scene)
    render_brain_surface(pl, subject, subjects_dir, cfg.brain_surface)

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
        plot_electrodes += np.array(cfg.electrodes.shift)[None, :]

        elec_mesh = pv.PolyData(plot_electrodes)
        elec_mesh['value'] = plot_data.value.to_numpy()

        pl.add_mesh(
            elec_mesh,
            scalars='value',
            point_size=cfg.electrodes.size,
            render_points_as_spheres=True,
            cmap=cmap,
            ambient=cfg.electrodes.ambient,
            specular=cfg.electrodes.specular,
            specular_power=cfg.electrodes.specular_power,
            diffuse=cfg.electrodes.diffuse,
            show_scalar_bar=True,
        )

    if background_idx is not None and len(background_idx) > 0:
        background_data = results.loc[background_idx]
        background_electrodes = electrodes.coordinates[background_data.channel - 1, :3]

        # pull out from surface
        background_electrodes += np.array(cfg.electrodes.shift)[None, :]

        background_mesh = pv.PolyData(background_electrodes)

        pl.add_mesh(
            background_mesh,
            point_size=cfg.electrodes.size / 2,
            render_points_as_spheres=True,
            color='grey',
            ambient=cfg.electrodes.ambient,
            specular=cfg.electrodes.specular,
            specular_power=cfg.electrodes.specular_power,
            diffuse=cfg.electrodes.diffuse,
            show_scalar_bar=False,
        )

    if show:
        pl.show()
    return pl


def plot_reconstruction(subject, show=True, **kwargs):
    cfg = get_config(kwargs)

    subjects_dir = _check_subjects_dir(cfg.data)
    electrodes = load_electrode_anatomy(subject, subjects_dir, warped=False)

    pl = make_plotter(cfg.scene)
    render_brain_surface(pl, subject, subjects_dir, cfg.brain_surface)

    all_labels = sorted(np.unique(electrodes.anatomy_labels))

    cmap = sns.color_palette("viridis", len(all_labels))

    for i, label in enumerate(all_labels):
        indices = np.where(electrodes.anatomy_labels == label)[0]
        point_cloud = pv.PolyData(electrodes.coordinates[indices, :3])
        glyph = point_cloud.glyph(orient=False, scale=False, geom=pv.Sphere(radius=1))

        color = [int(x) for x in colors.freesurfer.get(f"ctx-lh-{label}", cmap[i])]
        pl.add_mesh(glyph, color=color,
                    ambient=cfg.electrodes.ambient,
                    specular=cfg.electrodes.specular,
                    specular_power=cfg.electrodes.specular_power,
                    diffuse=cfg.electrodes.diffuse,
                    show_scalar_bar=False)

    # annotate with numbers
    pl.add_point_labels(electrodes.coordinates + np.array([-1, 0, 0]),
                        [f"{i + 1}" for i in range(len(electrodes))],
                        font_size=12, point_size=1)
    
    if show:
        pl.show()

    return pl