# ============================================================================
# Enhanced Nanotwinned Cu Phase-Field Simulator (FFT) + LLM Plasticity Recommender
# ============================================================================
import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib
matplotlib.use("Agg")                       # headless safety
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
import matplotlib.animation as animation
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import zipfile
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import hashlib
import traceback
import warnings
import struct
import re
import functools
import requests
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from scipy import stats
from io import BytesIO, StringIO
import tempfile
import os
from pathlib import Path
import pandas as pd
import logging
import networkx as nx

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR HANDLING DECORATOR (now with functools.wraps)
# ============================================================================
def handle_errors(default_return=None):
    """Decorator factory. Use as @handle_errors() or @handle_errors(default_return=(None, None))."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"❌ Error in {func.__name__}: {str(e)}"
                st.error(error_msg)
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


# ============================================================================
# SPECTRAL DERIVATIVE HELPERS
# ============================================================================
def make_k_vectors(N, dx):
    kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1)
    ky = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-12
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    fh = fft2(field)
    gx = np.real(ifft2(1j * kx * fh))
    gy = np.real(ifft2(1j * ky * fh))
    return gx, gy


def spectral_laplacian(field, k2):
    fh = fft2(field)
    lap = np.real(ifft2(-k2 * fh))
    return lap


def compute_twin_spacing_from_gradient(phi_gx, phi_gy):
    grad_mag = np.sqrt(phi_gx**2 + phi_gy**2)
    h = np.where(grad_mag > 1e-12, 2.0 / np.maximum(grad_mag, 1e-12), 1e6)
    return h


def compute_anisotropic_properties(phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso,
                                   L_CTB, L_ITB, n_mob):
    grad_mag = np.sqrt(phi_gx**2 + phi_gy**2 + 1e-12)
    mask = grad_mag > 1e-6
    mx = np.where(mask, phi_gx / grad_mag, 0.0)
    my = np.where(mask, phi_gy / grad_mag, 0.0)
    dot = mx * nx + my * ny
    kappa_phi = kappa0 * (1.0 + gamma_aniso * (1.0 - dot**2))
    aniso_factor = (1.0 - dot**2) ** n_mob
    L_phi = L_CTB + (L_ITB - L_CTB) * aniso_factor
    kappa_phi = np.where(mask, kappa_phi, kappa0)
    L_phi = np.where(mask, L_phi, L_CTB)
    return kappa_phi, L_phi


def compute_transformation_strain(phi, eta1, gamma_tw, ax, ay, nx, ny):
    f_phi = 0.25 * (phi**3 - phi**2 - phi + 1)
    eta1_clamped = np.clip(eta1, 0.0, 1.0)
    exx_star = gamma_tw * nx * ax * f_phi * eta1_clamped
    eyy_star = gamma_tw * ny * ay * f_phi * eta1_clamped
    exy_star = 0.5 * gamma_tw * (nx * ay + ny * ax) * f_phi * eta1_clamped
    return exx_star, eyy_star, exy_star


def compute_yield_stress(h, sigma0, mu, b, nu, rho0=None, alpha_taylor=0.3):
    """Hall–Petch + optional Taylor hardening. h in meters."""
    safe = h > 2 * b
    sigma_y = np.empty_like(h)
    log_term = np.log(np.maximum(h, 2.001 * b) / b)
    sigma_y[safe] = sigma0 + (mu * b / (2 * np.pi * h[safe] * (1 - nu))) * log_term[safe]
    sigma_y[~safe] = sigma0 + mu / (2 * np.pi * (1 - nu))
    if rho0 is not None and rho0 > 0:
        sigma_y = sigma_y + alpha_taylor * mu * b * np.sqrt(rho0)
    return sigma_y


def update_plastic_strain(sigma_eq, sigma_y, sxx, syy, sxy,
                          eps_p_xx, eps_p_yy, eps_p_xy,
                          gamma0_dot, m, dt):
    """J2 power-law update aligned with the deviatoric stress direction."""
    MAX_OVERSTRESS = 1.0
    MAX_PLASTIC_STRAIN = 0.1
    MAX_GAMMA_DOT = 1e6

    overstress = np.maximum(sigma_eq - sigma_y, 0.0) / np.maximum(sigma_y, 1e-9)
    overstress = np.minimum(overstress, MAX_OVERSTRESS)
    gamma_dot = gamma0_dot * np.power(overstress, m)
    gamma_dot = np.minimum(gamma_dot, MAX_GAMMA_DOT)

    # Deviatoric direction
    p = (sxx + syy) / 2.0
    s_dev_xx = sxx - p
    s_dev_yy = syy - p
    s_dev_xy = sxy
    s_mag = np.sqrt(s_dev_xx**2 + s_dev_yy**2 + 2.0 * s_dev_xy**2 + 1e-15)

    scale = (2.0 / 3.0) * gamma_dot * dt / s_mag
    d_xx = scale * s_dev_xx
    d_yy = scale * s_dev_yy
    d_xy = scale * s_dev_xy

    eps_p_xx_new = np.clip(eps_p_xx + d_xx, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_yy_new = np.clip(eps_p_yy + d_yy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_xy_new = np.clip(eps_p_xy + d_xy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    return eps_p_xx_new, eps_p_yy_new, eps_p_xy_new


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================
class MetadataManager:
    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        if run_time is None:
            run_time = 0.0
        return {
            'run_time': run_time,
            'frames': len(history) if history else 0,
            'grid_size': kwargs.get('grid_size', sim_params.get('N', 256)),
            'dx': kwargs.get('dx', sim_params.get('dx', 0.5)),
            'dt': sim_params.get('dt', 1e-4),
            'created_at': datetime.now().isoformat(),
            'colormaps': {
                'phi': sim_params.get('cmap_phi', 'RdBu_r'),
                'sigma_eq': sim_params.get('cmap_stress', 'hot'),
                'sigma_h': sim_params.get('cmap_hydro', 'RdBu'),
                'h': sim_params.get('cmap_h', 'plasma'),
                'eta1': sim_params.get('cmap_eta1', 'Reds')
            },
            'material_properties': sim_params.get('material_properties', {}),
            'plasticity_overrides': sim_params.get('plasticity', {}),
            'recommender_provenance': sim_params.get('recommender_provenance', {}),
            'simulation_parameters': {
                'dt': sim_params.get('dt', 1e-4),
                'N': sim_params.get('N', 256),
                'dx': sim_params.get('dx', 0.5),
                'twin_spacing': sim_params.get('twin_spacing', 20.0),
                'applied_stress': sim_params.get('applied_stress', 300e6),
                'applied_stress_angle': sim_params.get('applied_stress_angle', 0.0),
                'n_steps': sim_params.get('n_steps', 100),
                'solver_method': 'semi_implicit_fourier_spectral'
            }
        }

    @staticmethod
    def validate_metadata(metadata):
        if not isinstance(metadata, dict):
            metadata = {}
        defaults = {'run_time': 0.0, 'frames': 0, 'grid_size': 256, 'dx': 0.5,
                    'dt': 1e-4, 'created_at': datetime.now().isoformat()}
        for k, v in defaults.items():
            metadata.setdefault(k, v)
        metadata.setdefault('colormaps', {'phi': 'RdBu_r', 'sigma_eq': 'hot',
                                         'sigma_h': 'RdBu', 'h': 'plasma', 'eta1': 'Reds'})
        return metadata


# ============================================================================
# JOURNAL TEMPLATES
# ============================================================================
class JournalTemplates:
    @staticmethod
    def get_journal_styles():
        return {
            'nature': dict(figure_width_single=8.9, figure_width_double=18.3,
                font_family='Arial', font_size_small=7, font_size_medium=8,
                font_size_large=9, line_width=0.5, axes_linewidth=0.5,
                tick_width=0.5, tick_length=2, grid_alpha=0.1, dpi=600,
                color_cycle=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
            'science': dict(figure_width_single=5.5, figure_width_double=11.4,
                font_family='Helvetica', font_size_small=8, font_size_medium=9,
                font_size_large=10, line_width=0.75, axes_linewidth=0.75,
                tick_width=0.75, tick_length=3, grid_alpha=0.15, dpi=600,
                color_cycle=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
                             '#4DBEEE', '#A2142F', '#FF00FF', '#00FFFF', '#FFA500']),
            'advanced_materials': dict(figure_width_single=8.6, figure_width_double=17.8,
                font_family='Arial', font_size_small=8, font_size_medium=9,
                font_size_large=10, line_width=1.0, axes_linewidth=1.0,
                tick_width=1.0, tick_length=4, grid_alpha=0.2, dpi=600,
                color_cycle=['#004488', '#DDAA33', '#BB5566', '#000000', '#44AA99',
                             '#882255', '#117733', '#999933', '#AA4499', '#88CCEE']),
            'prl': dict(figure_width_single=3.4, figure_width_double=7.0,
                font_family='Times New Roman', font_size_small=8,
                font_size_medium=10, font_size_large=12, line_width=1.0,
                axes_linewidth=1.0, tick_width=1.0, tick_length=4, grid_alpha=0,
                dpi=600,
                color_cycle=['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                             '#0072B2', '#D55E00', '#CC79A7', '#999999', '#FFFFFF']),
            'custom': dict(figure_width_single=6.0, figure_width_double=12.0,
                font_family='DejaVu Sans', font_size_small=10, font_size_medium=12,
                font_size_large=14, line_width=1.5, axes_linewidth=1.5,
                tick_width=1.0, tick_length=5, grid_alpha=0.3, dpi=300,
                color_cycle=plt.get_cmap('Set2').colors if hasattr(plt.get_cmap('Set2'), 'colors')
                            else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
        }

    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
        styles = JournalTemplates.get_journal_styles()
        style = styles.get(journal_name, styles['nature'])
        rcParams.update({
            'font.family': style['font_family'],
            'font.size': style['font_size_medium'],
            'axes.linewidth': style['axes_linewidth'],
            'axes.labelsize': style['font_size_medium'],
            'axes.titlesize': style['font_size_large'],
            'xtick.labelsize': style['font_size_small'],
            'ytick.labelsize': style['font_size_small'],
            'legend.fontsize': style['font_size_small'],
            'figure.titlesize': style['font_size_large'],
            'lines.linewidth': style['line_width'],
            'lines.markersize': 4,
            'xtick.major.width': style['tick_width'],
            'ytick.major.width': style['tick_width'],
            'xtick.minor.width': style['tick_width'] * 0.5,
            'ytick.minor.width': style['tick_width'] * 0.5,
            'xtick.major.size': style['tick_length'],
            'ytick.major.size': style['tick_length'],
            'xtick.minor.size': style['tick_length'] * 0.6,
            'ytick.minor.size': style['tick_length'] * 0.6,
            'axes.grid': False,
            'savefig.dpi': style['dpi'],
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.prop_cycle': plt.cycler(color=style['color_cycle'])
        })
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        for ax in axes_flat:
            if ax is not None:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(which='both', direction='in', top=True, right=True)
        return fig, style


# ============================================================================
# COLORMAP LIBRARY
# ============================================================================
COLORMAPS = {
    'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno',
    'magma': 'magma', 'cividis': 'cividis', 'hot': 'hot', 'cool': 'cool',
    'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn', 'winter': 'winter',
    'copper': 'copper', 'bone': 'bone', 'gray': 'gray', 'pink': 'pink',
    'afmhot': 'afmhot', 'gist_heat': 'gist_heat', 'gist_gray': 'gist_gray',
    'binary': 'binary', 'coolwarm': 'coolwarm', 'bwr': 'bwr', 'seismic': 'seismic',
    'RdBu': 'RdBu', 'RdBu_r': 'RdBu_r', 'RdGy': 'RdGy', 'PiYG': 'PiYG',
    'PRGn': 'PRGn', 'BrBG': 'BrBG', 'PuOr': 'PuOr', 'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted', 'hsv': 'hsv', 'tab10': 'tab10',
    'tab20': 'tab20', 'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3',
    'Paired': 'Paired', 'Accent': 'Accent', 'Dark2': 'Dark2', 'jet': 'jet',
    'turbo': 'turbo', 'rainbow': 'rainbow', 'nipy_spectral': 'nipy_spectral',
    'gist_ncar': 'gist_ncar', 'gist_rainbow': 'gist_rainbow',
    'gist_earth': 'gist_earth', 'gist_stern': 'gist_stern', 'ocean': 'ocean',
    'terrain': 'terrain', 'gnuplot': 'gnuplot', 'gnuplot2': 'gnuplot2',
    'CMRmap': 'CMRmap', 'cubehelix': 'cubehelix', 'brg': 'brg',
}
cmap_list = list(COLORMAPS.keys())


# ============================================================================
# LINE PROFILER
# ============================================================================
class EnhancedLineProfiler:
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

    @handle_errors(default_return=(np.array([]), np.array([]), None))
    def extract_profile(self, data, profile_type, position_ratio=0.5, angle_deg=45):
        profile_type = self._normalize_profile_type(profile_type)
        ny, nx = data.shape
        center_x, center_y = nx // 2, ny // 2
        if profile_type in ['horizontal', 'vertical']:
            offset = int(min(nx, ny) * 0.4 * position_ratio)
        else:
            offset = int(min(nx, ny) * 0.3 * position_ratio)

        if profile_type == 'horizontal':
            row_idx = min(max(center_y + offset, 0), ny - 1)
            profile = data[row_idx, :]
            distance = np.linspace(self.extent[0], self.extent[1], nx)
            endpoints = (self.extent[0], row_idx * self.dx + self.extent[2],
                         self.extent[1], row_idx * self.dx + self.extent[2])
        elif profile_type == 'vertical':
            col_idx = min(max(center_x + offset, 0), nx - 1)
            profile = data[:, col_idx]
            distance = np.linspace(self.extent[2], self.extent[3], ny)
            endpoints = (col_idx * self.dx + self.extent[0], self.extent[2],
                         col_idx * self.dx + self.extent[0], self.extent[3])
        elif profile_type == 'diagonal':
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x - diag_length//2, center_y - diag_length//2)
            profile, distances = [], []
            for i in range(diag_length):
                x, y = start_idx[0] + i, start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    distances.append(i * self.dx * np.sqrt(2) - (diag_length//2) * self.dx * np.sqrt(2))
            distance = np.array(distances); profile = np.array(profile)
            endpoints = (start_idx[0]*self.dx + self.extent[0], start_idx[1]*self.dx + self.extent[2],
                         (start_idx[0]+diag_length-1)*self.dx + self.extent[0],
                         (start_idx[1]+diag_length-1)*self.dx + self.extent[2])
        elif profile_type == 'anti_diagonal':
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x + diag_length//2, center_y - diag_length//2)
            profile, distances = [], []
            for i in range(diag_length):
                x, y = start_idx[0] - i, start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    distances.append(i * self.dx * np.sqrt(2) - (diag_length//2) * self.dx * np.sqrt(2))
            distance = np.array(distances); profile = np.array(profile)
            endpoints = (start_idx[0]*self.dx + self.extent[0], start_idx[1]*self.dx + self.extent[2],
                         (start_idx[0]-diag_length+1)*self.dx + self.extent[0],
                         (start_idx[1]+diag_length-1)*self.dx + self.extent[2])
        elif profile_type == 'custom':
            angle_rad = np.deg2rad(angle_deg)
            length = int(min(nx, ny) * 0.8)
            profile, distances = [], []
            for t in np.linspace(-length/2, length/2, length):
                x = center_x + t * np.cos(angle_rad) + offset * np.cos(angle_rad + np.pi/2)
                y = center_y + t * np.sin(angle_rad) + offset * np.sin(angle_rad + np.pi/2)
                if 0 <= x < nx-1 and 0 <= y < ny-1:
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0+1, nx-1), min(y0+1, ny-1)
                    wx, wy = x - x0, y - y0
                    val = (data[y0, x0]*(1-wx)*(1-wy) + data[y0, x1]*wx*(1-wy) +
                           data[y1, x0]*(1-wx)*wy + data[y1, x1]*wx*wy)
                    profile.append(val); distances.append(t * self.dx)
            distance = np.array(distances); profile = np.array(profile)
            endpoints = (0, 0, 0, 0)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        return distance, profile, endpoints

    def _normalize_profile_type(self, profile_type):
        normalized = str(profile_type).lower().replace('-', '_')
        mapping = {'horizontal': 'horizontal', 'h': 'horizontal', 'x': 'horizontal',
                   'vertical': 'vertical', 'v': 'vertical', 'y': 'vertical',
                   'diagonal': 'diagonal', 'd': 'diagonal', 'diag': 'diagonal',
                   'anti_diagonal': 'anti_diagonal', 'antidiagonal': 'anti_diagonal',
                   'ad': 'anti_diagonal',
                   'custom': 'custom', 'c': 'custom', 'angled': 'custom'}
        return mapping.get(normalized, normalized)


class PublicationEnhancer:
    @staticmethod
    def create_custom_colormaps():
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        return {
            'plasma_enhanced': LinearSegmentedColormap.from_list('plasma_enhanced', [
                (0.0, '#0c0887'), (0.1, '#4b03a1'), (0.3, '#8b0aa5'),
                (0.5, '#b83289'), (0.7, '#db5c68'), (0.9, '#f48849'), (1.0, '#fec325')]),
            'coolwarm_enhanced': LinearSegmentedColormap.from_list('coolwarm_enhanced', [
                (0.0, '#3a4cc0'), (0.25, '#8abcdd'), (0.5, '#f7f7f7'),
                (0.75, '#f0b7a4'), (1.0, '#b40426')]),
            'twin_categorical': ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c',
                                                 '#d62728', '#9467bd', '#8c564b']),
            'stress_map': LinearSegmentedColormap.from_list('stress_map', [
                (0.0, '#2c7bb6'), (0.2, '#abd9e9'), (0.4, '#ffffbf'),
                (0.6, '#fdae61'), (0.8, '#d7191c'), (1.0, '#800026')])
        }

    @staticmethod
    def add_scale_bar(ax, length_nm, location='lower right', color='black',
                      linewidth=2, fontsize=8):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x_range = xlim[1] - xlim[0]; y_range = ylim[1] - ylim[0]
        bar_x_start = xlim[1] - x_range * 0.15
        bar_x_end = bar_x_start - length_nm
        bar_y = ylim[0] + y_range * 0.05
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
                color=color, linewidth=linewidth, solid_capstyle='butt')
        ax.text((bar_x_start + bar_x_end) / 2, bar_y + y_range * 0.02,
                f'{length_nm} nm', ha='center', va='bottom',
                color=color, fontsize=fontsize, fontweight='bold')
        return ax


# ============================================================================
# SIMULATION DATABASE
# ============================================================================
class SimulationDatabase:
    @staticmethod
    @handle_errors(default_return="00000000")
    def generate_id(sim_params):
        param_str = json.dumps({k: v for k, v in sim_params.items()
                              if k not in ['history', 'results', 'geom_viz']},
                             sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    @staticmethod
    @handle_errors(default_return=None)
    def save_simulation(sim_params, results_history, geometry_data,
                        metadata=None, run_time=0.0):
        if 'twin_simulations' not in st.session_state:
            st.session_state.twin_simulations = {}
        sim_id = SimulationDatabase.generate_id(sim_params)
        if metadata is None:
            metadata = MetadataManager.create_metadata(sim_params, results_history, run_time)
        else:
            metadata = MetadataManager.validate_metadata(metadata)
        st.session_state.twin_simulations[sim_id] = {
            'id': sim_id, 'params': sim_params,
            'results_history': results_history,
            'geometry_data': geometry_data, 'metadata': metadata,
            'created_at': metadata.get('created_at', datetime.now().isoformat()),
            'last_modified': datetime.now().isoformat()
        }
        return sim_id

    @staticmethod
    @handle_errors(default_return=None)
    def get_simulation(sim_id):
        if 'twin_simulations' in st.session_state and sim_id in st.session_state.twin_simulations:
            sim_data = st.session_state.twin_simulations[sim_id]
            if 'metadata' in sim_data:
                sim_data['metadata'] = MetadataManager.validate_metadata(sim_data['metadata'])
            return sim_data
        return None

    @staticmethod
    @handle_errors(default_return=False)
    def delete_simulation(sim_id):
        if 'twin_simulations' in st.session_state and sim_id in st.session_state.twin_simulations:
            del st.session_state.twin_simulations[sim_id]
            return True
        return False

    @staticmethod
    def get_all_simulations():
        if 'twin_simulations' in st.session_state:
            for sim_id, sim_data in st.session_state.twin_simulations.items():
                if 'metadata' in sim_data:
                    sim_data['metadata'] = MetadataManager.validate_metadata(sim_data['metadata'])
            return st.session_state.twin_simulations
        return {}

    @staticmethod
    def get_simulation_list():
        if 'twin_simulations' not in st.session_state:
            return []
        simulations = []
        for sim_id, sim_data in st.session_state.twin_simulations.items():
            try:
                params = sim_data.get('params', {})
                metadata = sim_data.get('metadata', {})
                name = (f"λ={params.get('twin_spacing', 0):.1f}nm | "
                        f"σ={params.get('applied_stress', 0)/1e6:.0f}MPa | "
                        f"θ={params.get('applied_stress_angle', 0):.0f}° | "
                        f"W={params.get('W', 0):.1f}")
                simulations.append({'id': sim_id, 'name': name, 'params': params,
                                    'metadata': metadata,
                                    'results': sim_data['results_history'][-1]
                                                if sim_data['results_history'] else None})
            except Exception:
                continue
        return simulations


# ============================================================================
# FILENAME HELPERS
# ============================================================================
@handle_errors(default_return="unknown")
def sanitize_token(text: str) -> str:
    s = str(text)
    for ch in [" ", "{", "}", "/", "\\", ",", ";", "(", ")", "[", "]", "°"]:
        s = s.replace(ch, "")
    return s


@handle_errors(default_return="0.0")
def fmt_num_trim(x, ndigits=3):
    s = f"{x:.{ndigits}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


@handle_errors(default_return="twin_simulation")
def build_sim_name(params: dict, sim_id: str = None) -> str:
    geom_type = params.get("geometry_type", "standard")
    if geom_type == "defect":
        defect = params.get("defect_type", "dislocation")
        geom_token = f"twin_grain_with_{defect}"
    else:
        geom_token = "standard_twin_grain"
    twin_spacing = fmt_num_trim(params.get("twin_spacing", 20.0), ndigits=1)
    W = fmt_num_trim(params.get("W", 2.0), ndigits=1)
    stress_mpa = fmt_num_trim(params.get("applied_stress", 300e6)/1e6, ndigits=0)
    theta = fmt_num_trim(params.get("applied_stress_angle", 0.0), ndigits=0)
    name = f"twin_lambda_{twin_spacing}_W_{W}_stress_{stress_mpa}MPa_theta_{theta}_{geom_token}"
    return f"{name}_{sim_id}" if sim_id else name


# ============================================================================
# MATERIAL PROPERTIES
# ============================================================================
class MaterialProperties:
    @staticmethod
    def get_cu_properties():
        return {
            'elastic': {'C11': 168.4e9, 'C12': 121.4e9, 'C44': 75.4e9,
                        'source': 'Phys. Rev. B 73, 064112 (2006)'},
            'twinning': {
                'gamma_tw': 1/np.sqrt(2),
                'n_111': np.array([1, 1, 1])/np.sqrt(3),
                'a_112': np.array([1, 1, -2])/np.sqrt(6),
                'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])},
            'plasticity': {'mu': 48e9, 'nu': 0.34, 'b': 0.256e-9,
                          'sigma0': 50e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
        }

    @staticmethod
    def get_al_properties():
        return {
            'elastic': {'C11': 106.8e9, 'C12': 60.4e9, 'C44': 28.3e9,
                        'source': 'J. Appl. Phys. 88, 3287 (2000)'},
            'twinning': {
                'gamma_tw': 1/np.sqrt(2),
                'n_111': np.array([1, 1, 1])/np.sqrt(3),
                'a_112': np.array([1, 1, -2])/np.sqrt(6),
                'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])},
            'plasticity': {'mu': 26e9, 'nu': 0.33, 'b': 0.286e-9,
                          'sigma0': 30e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
        }

    @staticmethod
    def get_ni_properties():
        return {
            'elastic': {'C11': 246.5e9, 'C12': 147.3e9, 'C44': 124.7e9,
                        'source': 'Phys. Rev. B 94, 014110 (2016)'},
            'twinning': {
                'gamma_tw': 1/np.sqrt(2),
                'n_111': np.array([1, 1, 1])/np.sqrt(3),
                'a_112': np.array([1, 1, -2])/np.sqrt(6),
                'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])},
            'plasticity': {'mu': 80e9, 'nu': 0.31, 'b': 0.249e-9,
                          'sigma0': 70e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
        }

    @staticmethod
    @handle_errors(default_return=None)
    def get_material(material_name='Cu'):
        if material_name == 'Cu': return MaterialProperties.get_cu_properties()
        if material_name == 'Al': return MaterialProperties.get_al_properties()
        if material_name == 'Ni': return MaterialProperties.get_ni_properties()
        return MaterialProperties.get_cu_properties()

    @staticmethod
    @handle_errors(default_return=([], []))
    def validate_parameters(params):
        errors, warnings_list = [], []
        if params.get('dt', 0) <= 0: errors.append("Time step dt must be positive")
        if params.get('dx', 0) <= 0: errors.append("Grid spacing dx must be positive")
        if params.get('N', 0) < 32:
            warnings_list.append("Grid resolution N < 32 may produce inaccurate results")
        if params.get('twin_spacing', 0) < 5:
            warnings_list.append("Twin spacing < 5nm may be physically unrealistic")
        if params.get('applied_stress', 0) > 2e9:
            warnings_list.append("Applied stress > 2GPa may cause unrealistic deformation")
        return errors, warnings_list


# ============================================================================
# INITIAL GEOMETRY
# ============================================================================
class InitialGeometryVisualizer:
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.x = np.linspace(-N*dx/2, N*dx/2, N)
        self.y = np.linspace(-N*dx/2, N*dx/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

    @handle_errors(default_return=(None, None, None))
    def create_twin_grain_geometry(self, twin_spacing=20.0, grain_boundary_pos=0.0,
                                   gb_width=3.0, buffer_width=5.0, left_buffer_width=5.0,
                                   gb_profile='plane', gb_curvature=0.0):
        eta1 = np.zeros((self.N, self.N)); eta2 = np.zeros((self.N, self.N))
        phi = np.zeros((self.N, self.N))

        def gb_x_func(y):
            if gb_profile == 'plane': return grain_boundary_pos
            sigma = (self.N * self.dx) / 4.0
            if gb_profile == 'concave':
                return grain_boundary_pos - gb_curvature * np.exp(-y**2 / (2 * sigma**2))
            if gb_profile == 'convex':
                return grain_boundary_pos + gb_curvature * np.exp(-y**2 / (2 * sigma**2))
            return grain_boundary_pos

        left_edge = self.extent[0]
        for i in range(self.N):
            for j in range(self.N):
                x_val, y_val = self.X[i, j], self.Y[i, j]
                gb_x = gb_x_func(y_val)
                dist_from_gb = x_val - gb_x
                if dist_from_gb < -gb_width:
                    eta1[i, j] = 1.0; eta2[i, j] = 0.0
                elif dist_from_gb > gb_width:
                    eta1[i, j] = 0.0; eta2[i, j] = 1.0
                else:
                    transition = 0.5 * (1 - np.tanh(dist_from_gb / (gb_width/3)))
                    eta1[i, j] = transition; eta2[i, j] = 1 - transition
        for i in range(self.N):
            for j in range(self.N):
                if eta1[i, j] > 0.5:
                    x_val, y_val = self.X[i, j], self.Y[i, j]
                    gb_x = gb_x_func(y_val)
                    if (abs(x_val - gb_x) > buffer_width and
                        abs(x_val - left_edge) > left_buffer_width):
                        phase = 2 * np.pi * y_val / twin_spacing
                        phi[i, j] = np.tanh(np.sin(phase) * 3.0)
                    else:
                        phi[i, j] = 1.0
        return phi, eta1, eta2

    @handle_errors(default_return=(None, None, None))
    def create_defect_geometry(self, twin_spacing=20.0, defect_type='dislocation',
                               defect_pos=(0, 0), defect_radius=10.0,
                               grain_boundary_pos=0.0, gb_width=3.0,
                               buffer_width=5.0, left_buffer_width=5.0,
                               gb_profile='plane', gb_curvature=0.0):
        phi, eta1, eta2 = self.create_twin_grain_geometry(
            twin_spacing, grain_boundary_pos, gb_width,
            buffer_width, left_buffer_width, gb_profile, gb_curvature)
        center_x, center_y = defect_pos
        if defect_type == 'dislocation':
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i,j]-center_x)**2 + (self.Y[i,j]-center_y)**2)
                    if dist < defect_radius:
                        phase_shift = np.exp(-dist**2 / (defect_radius**2)) * np.pi
                        phase = 2*np.pi*self.Y[i,j]/twin_spacing + phase_shift
                        phi_candidate = np.tanh(np.sin(phase) * 3.0)
                        if eta1[i, j] > 0.5:
                            phi[i, j] = phi_candidate
        elif defect_type == 'void':
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i,j]-center_x)**2 + (self.Y[i,j]-center_y)**2)
                    if dist < defect_radius:
                        eta1[i,j] = 0.0; eta2[i,j] = 0.0; phi[i,j] = 0.0
        return phi, eta1, eta2


# ============================================================================
# SPECTRAL ELASTICITY SOLVER
# ============================================================================
class EnhancedSpectralSolver:
    def __init__(self, N, dx, elastic_params, kx=None, ky=None, k2=None):
        self.N = N; self.dx = dx
        if kx is None or ky is None or k2 is None:
            self.kx, self.ky, self.k2 = make_k_vectors(N, dx)
        else:
            self.kx, self.ky, self.k2 = kx, ky, k2
        C11 = elastic_params['C11']; C12 = elastic_params['C12']; C44 = elastic_params['C44']
        C11_2d = (C11 + C12 + 2*C44) / 2
        C12_2d = (C11 + C12 - 2*C44) / 2
        lambda_2d = C12_2d
        mu_2d = (C11_2d - C12_2d) / 2
        self.C11_2d = C11_2d; self.C12_2d = C12_2d; self.C44_2d = C44
        denom = mu_2d * (lambda_2d + 2*mu_2d) * self.k2 + 1e-15
        self.G11 = (mu_2d*(self.kx**2 + 2*self.ky**2) + lambda_2d*self.ky**2) / denom
        self.G12 = -mu_2d * self.kx * self.ky / denom
        self.G22 = (mu_2d*(self.ky**2 + 2*self.kx**2) + lambda_2d*self.kx**2) / denom
        # DC mode for Green's function should be zero
        self.G11[0, 0] = 0.0; self.G12[0, 0] = 0.0; self.G22[0, 0] = 0.0

    @handle_errors(default_return=(None,)*8)
    def solve(self, eigenstrain_xx, eigenstrain_yy, eigenstrain_xy,
              applied_stress_xx=0, applied_stress_yy=0, applied_stress_xy=0):
        assert eigenstrain_xx.shape == (self.N, self.N)
        eps_xx_hat = fft2(eigenstrain_xx)
        eps_yy_hat = fft2(eigenstrain_yy)
        eps_xy_hat = fft2(eigenstrain_xy)
        # Correct Green's-function contraction for a full 2-D eigenstrain
        ux_hat = 1j * (self.G11 * self.kx * eps_xx_hat +
                       self.G12 * self.ky * eps_xx_hat +
                       self.G12 * self.kx * eps_yy_hat +
                       self.G22 * self.ky * eps_yy_hat +
                       2.0 * self.G12 * self.kx * eps_xy_hat +
                       2.0 * self.G22 * self.ky * eps_xy_hat)
        uy_hat = 1j * (self.G12 * self.kx * eps_xx_hat +
                       self.G22 * self.ky * eps_xx_hat +
                       self.G11 * self.kx * eps_yy_hat +
                       self.G12 * self.ky * eps_yy_hat +
                       2.0 * self.G11 * self.kx * eps_xy_hat +
                       2.0 * self.G12 * self.ky * eps_xy_hat)
        eps_xx_el = np.real(ifft2(1j * self.kx * ux_hat))
        eps_yy_el = np.real(ifft2(1j * self.ky * uy_hat))
        eps_xy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_hat + self.ky * ux_hat)))
        # Constitutive law: sigma = C : (eps_el + eps*) is NOT correct.
        # Correct: sigma = C : eps_el (compatible strain from u already accounts for eps*).
        eps_xx = eps_xx_el
        eps_yy = eps_yy_el
        eps_xy = eps_xy_el
        sxx = applied_stress_xx + self.C11_2d * eps_xx + self.C12_2d * eps_yy
        syy = applied_stress_yy + self.C12_2d * eps_xx + self.C11_2d * eps_yy
        sxy = applied_stress_xy + 2 * self.C44_2d * eps_xy
        sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + syy**2 + sxx**2 + 6 * sxy**2))
        sigma_eq = np.clip(sigma_eq, 0, 5e9)
        sigma_h = (sxx + syy) / 2
        return sigma_eq, sxx, syy, sxy, sigma_h, eps_xx, eps_yy, eps_xy


# ============================================================================
# VISUALIZATION
# ============================================================================
class EnhancedTwinVisualizer:
    def __init__(self, N, dx, dt=1e-4):
        self.N = N; self.dx = dx; self.dt = dt
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
        self.line_profiler = EnhancedLineProfiler(N, dx)
        self.COLORMAPS = COLORMAPS.copy()
        self.COLORMAPS.update(PublicationEnhancer.create_custom_colormaps())

    @handle_errors(default_return=None)
    def get_colormap(self, cmap_name):
        if cmap_name in self.COLORMAPS:
            entry = self.COLORMAPS[cmap_name]
            return plt.get_cmap(entry) if isinstance(entry, str) else entry
        return plt.get_cmap('viridis')

    @handle_errors(default_return=None)
    def create_multi_field_comparison(self, results_dict, style_params=None):
        if style_params is None: style_params = {}
        defaults = {'title_font_size': 10, 'label_font_size': 8,
                    'scalebar_color': 'black', 'scalebar_fontsize': 8,
                    'phi_cmap': 'RdBu_r', 'sigma_eq_cmap': 'hot',
                    'sigma_h_cmap': 'RdBu', 'h_cmap': 'plasma',
                    'eps_p_mag_cmap': 'YlOrRd', 'sigma_y_cmap': 'viridis',
                    'eta1_cmap': 'Reds'}
        for k, v in defaults.items(): style_params.setdefault(k, v)
        title_fs = float(style_params['title_font_size'])
        label_fs = float(style_params['label_font_size'])
        sb_fs = float(style_params['scalebar_fontsize'])
        fields_to_plot = [
            ('phi', 'Twin Order Parameter φ', style_params['phi_cmap'], [-1.2, 1.2]),
            ('eta1', 'Grain η₁', style_params['eta1_cmap'], [0, 1]),
            ('sigma_eq', 'Von Mises Stress (GPa)', style_params['sigma_eq_cmap'], None),
            ('sigma_h', 'Hydrostatic Stress (GPa)', style_params['sigma_h_cmap'], None),
            ('h', 'Twin Spacing (nm)', style_params['h_cmap'], [0, 30]),
            ('eps_p_mag', 'Plastic Strain', style_params['eps_p_mag_cmap'], None),
            ('sigma_y', 'Yield Stress (MPa)', style_params['sigma_y_cmap'], None)]
        available = [(f, t, c, v) for f, t, c, v in fields_to_plot if f in results_dict]
        if not available: return None
        n_fields = len(available)
        cols = min(3, n_fields); rows = (n_fields + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
        if rows == 1 and cols == 1: axes = np.array([axes])
        else: axes = axes.flatten()
        for idx, (fname, title, cmap_default, vrange) in enumerate(available):
            ax = axes[idx]
            data = results_dict[fname].copy()
            if fname in ['sigma_eq', 'sigma_h']: data = data / 1e9
            elif fname == 'sigma_y': data = data / 1e6
            cmap = self.get_colormap(style_params.get(f'{fname}_cmap', cmap_default))
            if vrange is not None: vmin, vmax = vrange
            else:
                vmin = np.percentile(data, 2); vmax = np.percentile(data, 98)
                if fname == 'sigma_h':
                    vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
            im = ax.imshow(data, extent=self.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                          origin='lower', aspect='equal', interpolation='bilinear')
            if fname == 'phi':
                ax.contour(np.linspace(self.extent[0], self.extent[1], self.N),
                          np.linspace(self.extent[2], self.extent[3], self.N),
                          data, levels=[0], colors='white', linewidths=1, alpha=0.8)
            ax.set_title(title, fontsize=title_fs)
            ax.set_xlabel('x (nm)', fontsize=label_fs)
            ax.set_ylabel('y (nm)', fontsize=label_fs)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if fname in ['sigma_eq', 'sigma_h']: cbar.set_label('Stress (GPa)')
            elif fname == 'sigma_y': cbar.set_label('Stress (MPa)')
            elif fname == 'h': cbar.set_label('Spacing (nm)')
            if fname in ['phi', 'eta1', 'sigma_eq', 'sigma_h']:
                PublicationEnhancer.add_scale_bar(
                    ax, 10.0, 'lower right',
                    color=style_params['scalebar_color'], fontsize=sb_fs)
        for idx in range(n_fields, len(axes)): axes[idx].axis('off')
        plt.tight_layout()
        return fig

    @handle_errors(default_return=None)
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0):
        if field_name not in results_dict: return None
        data = results_dict[field_name].copy()
        unit = ""
        if field_name in ['sigma_eq', 'sigma_h']: data = data/1e9; unit = " (GPa)"
        elif field_name == 'sigma_y': data = data/1e6; unit = " (MPa)"
        elif field_name == 'h': unit = " (nm)"
        if field_name == 'phi': colorscale, zmid = 'RdBu', 0
        elif field_name == 'eta1': colorscale, zmid = 'Reds', None
        elif field_name in ['sigma_eq', 'sigma_h']:
            colorscale = 'Viridis' if field_name == 'sigma_eq' else 'RdBu'
            zmid = 0 if field_name == 'sigma_h' else None
        else: colorscale, zmid = 'Plasma', None
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=data,
            x=np.linspace(self.extent[0], self.extent[1], self.N),
            y=np.linspace(self.extent[2], self.extent[3], self.N),
            colorscale=colorscale, zmid=zmid,
            colorbar=dict(title=f"{field_name}{unit}"),
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.3f}<extra></extra>'))
        if field_name != 'phi' and 'phi' in results_dict:
            fig.add_trace(go.Contour(z=results_dict['phi'],
                x=np.linspace(self.extent[0], self.extent[1], self.N),
                y=np.linspace(self.extent[2], self.extent[3], self.N),
                contours=dict(start=0, end=0, size=0, coloring='none', showlabels=False),
                line=dict(color='white', width=2), showscale=False, hoverinfo='skip'))
        fig.update_layout(title=f"{field_name} (Frame {frame_idx})",
                         xaxis_title="x (nm)", yaxis_title="y (nm)",
                         width=600, height=500, template="plotly_white")
        return fig

    @handle_errors(default_return=None)
    def create_plotly_line_profiles(self, results_dict, field_name, profile_types,
                                    position_ratio=0.5):
        fig = go.Figure()
        for ptype in profile_types:
            distance, profile, _ = self.line_profiler.extract_profile(
                results_dict[field_name], ptype, position_ratio)
            if field_name in ['sigma_eq', 'sigma_h']:
                profile = profile / 1e9; ylabel = 'Stress (GPa)'
            elif field_name == 'sigma_y':
                profile = profile / 1e6; ylabel = 'Stress (MPa)'
            else: ylabel = field_name
            fig.add_trace(go.Scatter(x=distance, y=profile, mode='lines',
                                    name=ptype.replace('_', ' ').title()))
        fig.update_layout(title=f"{field_name} Line Profiles",
                         xaxis_title="Position (nm)", yaxis_title=ylabel,
                         hovermode='x unified', template="plotly_white")
        return fig

    @handle_errors(default_return=None)
    def create_plotly_3d_surface(self, results_dict, field_name, frame_idx=0):
        if field_name not in results_dict: return None
        data = results_dict[field_name].copy(); unit = ""
        if field_name in ['sigma_eq', 'sigma_h']: data = data/1e9; unit = " (GPa)"
        elif field_name == 'sigma_y': data = data/1e6; unit = " (MPa)"
        elif field_name == 'h': unit = " (nm)"
        x = np.linspace(self.extent[0], self.extent[1], self.N)
        y = np.linspace(self.extent[2], self.extent[3], self.N)
        X, Y = np.meshgrid(x, y)
        if field_name == 'phi': colorscale, cmin, cmax = 'RdBu', -1.2, 1.2
        elif field_name == 'eta1': colorscale, cmin, cmax = 'Reds', 0, 1
        elif field_name == 'sigma_h':
            colorscale = 'RdBu'; cmin, cmax = -np.max(np.abs(data)), np.max(np.abs(data))
        else: colorscale, cmin, cmax = 'Viridis', None, None
        fig = go.Figure(data=[go.Surface(z=data, x=X, y=Y, colorscale=colorscale,
                                         cmin=cmin, cmax=cmax,
                                         colorbar=dict(title=f"{field_name}{unit}"),
                                         hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>z: %{z:.3f}<extra></extra>')])
        fig.update_layout(title=f"3D Surface: {field_name} (Frame {frame_idx})",
                         scene=dict(xaxis_title='x (nm)', yaxis_title='y (nm)',
                                    zaxis_title=f'{field_name}{unit}',
                                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
                         width=700, height=600, template="plotly_white")
        return fig

    @handle_errors(default_return=None)
    def create_animation(self, history, field_name, output_format='gif', fps=5, dpi=150):
        if not history: return None
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
        try:
            first = history[0][field_name].copy()
            if field_name in ['sigma_eq', 'sigma_h']: first = first / 1e9
            elif field_name == 'sigma_y': first = first / 1e6
            vmin = np.percentile(first, 2); vmax = np.percentile(first, 98)
            if field_name == 'phi': vmin, vmax = -1.2, 1.2
            elif field_name == 'eta1': vmin, vmax = 0, 1
            elif field_name == 'sigma_h':
                vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
            im = ax.imshow(first, extent=self.extent, cmap=self.get_colormap('viridis'),
                          vmin=vmin, vmax=vmax, origin='lower', interpolation='bilinear')
            cbar = plt.colorbar(im, ax=ax)
            if field_name in ['sigma_eq', 'sigma_h']: cbar.set_label('Stress (GPa)')
            elif field_name == 'sigma_y': cbar.set_label('Stress (MPa)')
            elif field_name == 'h': cbar.set_label('Spacing (nm)')
            else: cbar.set_label(field_name)
            title = ax.set_title(f"{field_name} - t = 0.000 ns")
            dt_local = self.dt
            def update_frame(frame_idx):
                d = history[frame_idx][field_name].copy()
                if field_name in ['sigma_eq', 'sigma_h']: d = d / 1e9
                elif field_name == 'sigma_y': d = d / 1e6
                im.set_array(d)
                title.set_text(f"{field_name} - t = {frame_idx*dt_local*1e3:.3f} ns")
                return [im, title]
            ani = animation.FuncAnimation(fig, update_frame, frames=len(history),
                                         interval=1000/fps, blit=True)
            buffer = BytesIO()
            if output_format == 'gif':
                ani.save(buffer, writer='pillow', fps=fps, dpi=dpi)
            else:
                ani.save(buffer, writer='ffmpeg', fps=fps, dpi=dpi)
            buffer.seek(0)
            return buffer
        finally:
            plt.close(fig)


# ============================================================================
# SOLVER
# ============================================================================
class NanotwinnedCuSolver:
    def __init__(self, params):
        self.params = dict(params)   # shallow copy to avoid mutating caller
        self.N = self.params['N']; self.dx = self.params['dx']; self.dt = self.params['dt']
        material_name = self.params.get('material', 'Cu')
        self.mat_props = MaterialProperties.get_material(material_name)
        self.params['material'] = material_name

        # Merge user overrides into the plasticity block
        user_plast = self.params.get('plasticity', {}) or {}
        for k in ('mu', 'sigma0', 'gamma0_dot', 'm', 'rho0'):
            if user_plast.get(k) is not None:
                self.mat_props['plasticity'][k] = user_plast[k]

        errors, warnings_list = MaterialProperties.validate_parameters(self.params)
        if errors: raise ValueError(f"Parameter validation failed: {', '.join(errors)}")
        if warnings_list: st.warning(f"Parameter warnings: {', '.join(warnings_list)}")

        self.geom_viz = InitialGeometryVisualizer(self.N, self.dx)
        self.kx, self.ky, self.k2 = make_k_vectors(self.N, self.dx)

        try:
            self.phi, self.eta1, self.eta2 = self.initialize_fields()
        except Exception as e:
            st.error(f"Failed to initialize fields: {e}")
            self.phi = np.zeros((self.N, self.N))
            self.eta1 = np.zeros((self.N, self.N))
            self.eta2 = np.zeros((self.N, self.N))

        self.eps_p_xx = np.zeros((self.N, self.N))
        self.eps_p_yy = np.zeros((self.N, self.N))
        self.eps_p_xy = np.zeros((self.N, self.N))

        self.spectral_solver = EnhancedSpectralSolver(
            self.N, self.dx, self.mat_props['elastic'],
            kx=self.kx, ky=self.ky, k2=self.k2)

        self.history = {'phi_norm': [], 'energy': [], 'max_stress': [],
                        'plastic_work': [], 'avg_stress': [], 'twin_spacing_avg': []}

        kappa0 = float(self.params.get('kappa0', 1.0))
        L_CTB = float(self.params.get('L_CTB', 0.05))
        self.implicit_diff_phi = L_CTB * kappa0
        kappa_eta = float(self.params.get('kappa_eta', 2.0))
        L_eta = float(self.params.get('L_eta', 1.0))
        self.implicit_diff_eta = L_eta * kappa_eta
        self.confine_twin = self.params.get('confine_twin', True)

    @handle_errors(default_return=(None, None, None))
    def initialize_fields(self):
        geom_type = self.params.get('geometry_type', 'standard')
        twin_spacing = self.params['twin_spacing']
        gb_pos = self.params['grain_boundary_pos']
        gb_width = self.params.get('gb_width', 3.0)
        buffer_width = self.params.get('buffer_width', 5.0)
        left_buffer_width = self.params.get('left_buffer_width', 5.0)
        gb_profile = self.params.get('gb_profile', 'plane')
        gb_curvature = self.params.get('gb_curvature', 0.0)
        if geom_type == 'defect':
            return self.geom_viz.create_defect_geometry(
                twin_spacing, self.params.get('defect_type', 'dislocation'),
                self.params.get('defect_pos', (0, 0)),
                self.params.get('defect_radius', 10.0),
                gb_pos, gb_width, buffer_width, left_buffer_width,
                gb_profile, gb_curvature)
        return self.geom_viz.create_twin_grain_geometry(
            twin_spacing, gb_pos, gb_width, buffer_width,
            left_buffer_width, gb_profile, gb_curvature)

    @handle_errors(default_return=(None, None, None))
    def compute_local_energy_derivatives(self):
        W = self.params['W']; A = self.params['A']; B = self.params['B']
        df_dphi = 4 * W * self.phi * (self.phi**2 - 1) * self.eta1**2
        df_deta1 = (2*A*self.eta1*(1-self.eta1)*(1-2*self.eta1) +
                    2*B*self.eta1*self.eta2**2 +
                    2*W*(self.phi**2 - 1)**2 * self.eta1)
        df_deta2 = (2*A*self.eta2*(1-self.eta2)*(1-2*self.eta2) +
                    2*B*self.eta2*self.eta1**2)
        return df_dphi, df_deta1, df_deta2

    @handle_errors(default_return=None)
    def compute_elastic_driving_force(self, sxx, syy, sxy):
        gamma_tw = self.mat_props['twinning']['gamma_tw']
        n = self.mat_props['twinning']['n_2d']
        a = self.mat_props['twinning']['a_2d']
        dh_dphi = 0.25 * (3*self.phi**2 - 2*self.phi - 1)
        nx, ny = n[0], n[1]; ax, ay = a[0], a[1]
        deps_xx_dphi = gamma_tw * nx * ax * dh_dphi * self.eta1
        deps_yy_dphi = gamma_tw * ny * ay * dh_dphi * self.eta1
        deps_xy_dphi = 0.5 * gamma_tw * (nx * ay + ny * ax) * dh_dphi * self.eta1
        return -(sxx*deps_xx_dphi + syy*deps_yy_dphi + 2*sxy*deps_xy_dphi)

    @handle_errors(default_return=None)
    def evolve_twin_field(self, sxx, syy, sxy, eps_p_mag):
        kappa0 = float(self.params['kappa0'])
        gamma_aniso = float(self.params['gamma_aniso'])
        L_CTB = float(self.params.get('L_CTB', 0.05))
        L_ITB = float(self.params.get('L_ITB', 5.0))
        n_mob = int(self.params.get('n_mob', 4))
        zeta = float(self.params.get('zeta', 0.3))
        n_twin = self.mat_props['twinning']['n_2d']
        nx, ny = float(n_twin[0]), float(n_twin[1])
        phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
        lap_phi = spectral_laplacian(self.phi, self.k2)
        kappa_phi, L_phi = compute_anisotropic_properties(
            phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso, L_CTB, L_ITB, n_mob)
        df_loc_dphi, _, _ = self.compute_local_energy_derivatives()
        df_el_dphi = self.compute_elastic_driving_force(sxx, syy, sxy)
        diss_p = zeta * eps_p_mag * self.phi
        NLF = df_loc_dphi + df_el_dphi + diss_p
        correction = (L_phi * kappa_phi - self.implicit_diff_phi) * lap_phi
        R = -L_phi * NLF + correction
        if self.confine_twin: R = R * self.eta1**2
        phi_hat = fft2(self.phi); R_hat = fft2(R)
        denom = 1.0 + self.dt * self.implicit_diff_phi * self.k2
        phi_new = np.real(ifft2((phi_hat + self.dt * R_hat) / denom))
        return np.clip(phi_new, -1.1, 1.1)

    @handle_errors(default_return=(None, None))
    def evolve_grain_fields(self):
        L_eta = float(self.params.get('L_eta', 1.0))
        _, df_deta1, df_deta2 = self.compute_local_energy_derivatives()
        R_eta1 = -L_eta * df_deta1
        R_eta2 = -L_eta * df_deta2
        eta1_hat = fft2(self.eta1); eta2_hat = fft2(self.eta2)
        R1_hat = fft2(R_eta1); R2_hat = fft2(R_eta2)
        denom = 1.0 + self.dt * self.implicit_diff_eta * self.k2
        eta1_new = np.real(ifft2((eta1_hat + self.dt * R1_hat) / denom))
        eta2_new = np.real(ifft2((eta2_hat + self.dt * R2_hat) / denom))
        eta1_new = np.clip(eta1_new, 0, 1); eta2_new = np.clip(eta2_new, 0, 1)
        norm = np.sqrt(eta1_new**2 + eta2_new**2 + 1e-12)
        mask = norm > 1
        eta1_new[mask] = eta1_new[mask] / norm[mask]
        eta2_new[mask] = eta2_new[mask] / norm[mask]
        return eta1_new, eta2_new

    @handle_errors(default_return=None)
    def compute_plastic_strain(self, sigma_eq, sigma_y, sxx, syy, sxy):
        pp = self.mat_props['plasticity']
        self.eps_p_xx, self.eps_p_yy, self.eps_p_xy = update_plastic_strain(
            sigma_eq, sigma_y, sxx, syy, sxy,
            self.eps_p_xx, self.eps_p_yy, self.eps_p_xy,
            pp['gamma0_dot'], int(pp['m']), self.dt)
        eps_p_mag = np.sqrt(2/3 * (self.eps_p_xx**2 + self.eps_p_yy**2 +
                                   2*self.eps_p_xy**2 + 1e-15))
        return np.clip(eps_p_mag, 0, 0.5)

    @handle_errors(default_return=0.0)
    def compute_total_energy(self):
        W = self.params['W']; A = self.params['A']; B = self.params['B']
        f_loc = (W*(self.phi**2 - 1)**2 * self.eta1**2 +
                 A*(self.eta1**2*(1-self.eta1)**2 + self.eta2**2*(1-self.eta2)**2) +
                 B*self.eta1**2*self.eta2**2)
        phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
        eta1_gx, eta1_gy = spectral_gradients(self.eta1, self.kx, self.ky)
        eta2_gx, eta2_gy = spectral_gradients(self.eta2, self.kx, self.ky)
        kappa0 = self.params['kappa0']; kappa_eta = self.params['kappa_eta']
        f_grad = (0.5*kappa0*(phi_gx**2 + phi_gy**2) +
                  0.5*kappa_eta*(eta1_gx**2 + eta1_gy**2 + eta2_gx**2 + eta2_gy**2))
        dx_m = self.dx * 1e-9   # nm -> m for the volume element
        return float(np.sum(f_loc + f_grad) * (dx_m**2))

    @handle_errors(default_return=None)
    def step(self):
        sigma_mag = self.params.get('applied_stress', 0.0)
        theta = np.deg2rad(self.params.get('applied_stress_angle', 0.0))
        applied_xx = sigma_mag * np.cos(theta)**2
        applied_yy = sigma_mag * np.sin(theta)**2
        applied_xy = sigma_mag * np.sin(theta) * np.cos(theta)
        gamma_tw = self.mat_props['twinning']['gamma_tw']
        n = self.mat_props['twinning']['n_2d']; a = self.mat_props['twinning']['a_2d']
        exx_star, eyy_star, exy_star = compute_transformation_strain(
            self.phi, self.eta1, gamma_tw, a[0], a[1], n[0], n[1])
        eigenstrain_xx = exx_star + self.eps_p_xx
        eigenstrain_yy = eyy_star + self.eps_p_yy
        eigenstrain_xy = exy_star + self.eps_p_xy
        (sigma_eq, sxx, syy, sxy, sigma_h,
         eps_xx, eps_yy, eps_xy) = self.spectral_solver.solve(
            eigenstrain_xx, eigenstrain_yy, eigenstrain_xy,
            applied_xx, applied_yy, applied_xy)
        phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
        h_nm = compute_twin_spacing_from_gradient(phi_gx, phi_gy)
        h_m = h_nm * 1e-9    # FIX: convert to meters for Hall-Petch
        pp = self.mat_props['plasticity']
        sigma_y = compute_yield_stress(h_m, pp['sigma0'], pp['mu'],
                                       pp['b'], pp['nu'], rho0=pp.get('rho0', 0.0))
        eps_p_mag = self.compute_plastic_strain(sigma_eq, sigma_y, sxx, syy, sxy)
        self.phi = self.evolve_twin_field(sxx, syy, sxy, eps_p_mag)
        self.eta1, self.eta2 = self.evolve_grain_fields()
        phi_norm = np.linalg.norm(self.phi)
        total_energy = self.compute_total_energy()
        max_stress = np.max(sigma_eq); avg_stress = np.mean(sigma_eq)
        valid_h = h_nm[(h_nm > 5) & (h_nm < 50)]
        avg_spacing = float(np.mean(valid_h)) if valid_h.size > 0 else 0.0
        plastic_work = float(np.sum(eps_p_mag) * (self.dx**2))
        self.history['phi_norm'].append(phi_norm)
        self.history['energy'].append(total_energy)
        self.history['max_stress'].append(max_stress)
        self.history['avg_stress'].append(avg_stress)
        self.history['plastic_work'].append(plastic_work)
        self.history['twin_spacing_avg'].append(avg_spacing)
        return {
            'phi': self.phi.copy(), 'eta1': self.eta1.copy(),
            'eta2': self.eta2.copy(), 'sigma_eq': sigma_eq.copy(),
            'sigma_h': sigma_h.copy(), 'sigma_xx': sxx.copy(),
            'sigma_yy': syy.copy(), 'sigma_xy': sxy.copy(),
            'h': h_nm.copy(), 'sigma_y': sigma_y.copy(),
            'eps_p_mag': eps_p_mag.copy(), 'eps_xx': eps_xx.copy(),
            'eps_yy': eps_yy.copy(), 'eps_xy': eps_xy.copy(),
            'convergence': {'phi_norm': phi_norm, 'energy': total_energy,
                           'max_stress': max_stress, 'avg_stress': avg_stress,
                           'plastic_work': plastic_work, 'avg_spacing': avg_spacing}
        }


# ============================================================================
# MONITORING
# ============================================================================
class SimulationMonitor:
    @staticmethod
    @handle_errors(default_return=None)
    def create_convergence_plots(history, timesteps):
        history_length = len(history['phi_norm'])
        if len(timesteps) >= history_length:
            plot_timesteps = timesteps[:history_length]
        else:
            plot_timesteps = np.linspace(0, timesteps[-1] if len(timesteps) else 1.0,
                                         history_length)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].plot(plot_timesteps, history['phi_norm'], 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Time (ns)'); axes[0, 0].set_ylabel('||φ||')
        axes[0, 0].set_title('Twin Order Parameter Norm'); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(plot_timesteps, history['energy'], 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Time (ns)'); axes[0, 1].set_ylabel('Total Energy (J)')
        axes[0, 1].set_title('System Energy'); axes[0, 1].grid(True, alpha=0.3)
        axes[0, 2].plot(plot_timesteps, np.array(history['max_stress'])/1e9,
                        'g-', linewidth=2, label='Max')
        axes[0, 2].plot(plot_timesteps, np.array(history['avg_stress'])/1e9,
                        'g--', linewidth=1.5, label='Avg')
        axes[0, 2].set_xlabel('Time (ns)'); axes[0, 2].set_ylabel('Stress (GPa)')
        axes[0, 2].set_title('Stress Evolution'); axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[1, 0].plot(plot_timesteps, history['plastic_work'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Time (ns)'); axes[1, 0].set_ylabel('Plastic Work (J)')
        axes[1, 0].set_title('Plastic Work'); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(plot_timesteps, history['twin_spacing_avg'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Time (ns)'); axes[1, 1].set_ylabel('Avg Spacing (nm)')
        axes[1, 1].set_title('Average Twin Spacing'); axes[1, 1].grid(True, alpha=0.3)
        axes[1, 2].text(0.5, 0.5, 'Plastic strain history\nnot saved',
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Plastic Strain Evolution')
        plt.tight_layout()
        return fig


# ============================================================================
# EXPORTERS
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


class DataExporter:
    @staticmethod
    @handle_errors(default_return=(None, None))
    def export_pkl(sim_data, params, history, sim_name):
        buffer = BytesIO()
        pickle.dump({'params': params, 'history': history,
                     'metadata': sim_data.get('metadata', {}), 'sim_name': sim_name}, buffer)
        buffer.seek(0)
        return buffer, f"{sim_name}.pkl"

    @staticmethod
    @handle_errors(default_return=(None, None))
    def export_pt(sim_data, params, history, sim_name):
        buffer = BytesIO()
        def to_tensor(x):
            return torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)
        tensor_data = {'params': params, 'metadata': sim_data.get('metadata', {}), 'history': []}
        for frame in history:
            tensor_data['history'].append({k: to_tensor(frame[k]) for k in
                ['phi', 'eta1', 'eta2', 'sigma_eq', 'sigma_h', 'h', 'eps_p_mag']})
        torch.save(tensor_data, buffer); buffer.seek(0)
        return buffer, f"{sim_name}.pt"

    @staticmethod
    @handle_errors(default_return=(None, None))
    def export_sql(sim_data, params, history, sim_name, sim_id, N, dx):
        conn = sqlite3.connect(':memory:'); c = conn.cursor()
        c.execute('''CREATE TABLE simulations (id TEXT PRIMARY KEY, sim_name TEXT,
                     twin_spacing REAL, applied_stress REAL, applied_stress_angle REAL,
                     W REAL, geometry_type TEXT, created_at TEXT, grid_size INTEGER, dx REAL)''')
        c.execute('''CREATE TABLE frames (sim_id TEXT, frame_idx INTEGER, phi BLOB,
                     eta1 BLOB, eta2 BLOB, sigma_eq BLOB, sigma_h BLOB, h BLOB, eps_p_mag BLOB)''')
        created_at = sim_data.get('metadata', {}).get('created_at', datetime.now().isoformat())
        c.execute("INSERT INTO simulations VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (sim_id, sim_name, params.get('twin_spacing', 0.0),
                   params.get('applied_stress', 0.0), params.get('applied_stress_angle', 0.0),
                   params.get('W', 0.0), params.get('geometry_type', 'standard'),
                   created_at, params.get('N', N), params.get('dx', dx)))
        for idx, frame in enumerate(history):
            c.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?)",
                      (sim_id, idx, pickle.dumps(frame['phi']), pickle.dumps(frame['eta1']),
                       pickle.dumps(frame['eta2']), pickle.dumps(frame['sigma_eq']),
                       pickle.dumps(frame['sigma_h']), pickle.dumps(frame['h']),
                       pickle.dumps(frame['eps_p_mag'])))
        conn.commit()
        dump_buffer = StringIO()
        for line in conn.iterdump(): dump_buffer.write(f'{line}\n')
        conn.close()
        return BytesIO(dump_buffer.getvalue().encode()), f"{sim_name}.sql"

    @staticmethod
    @handle_errors(default_return=(None, None))
    def export_csv(history, sim_name, extent, N, dx):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], N); y = np.linspace(extent[2], extent[3], N)
            X, Y = np.meshgrid(x, y)
            for idx, frame in enumerate(history):
                df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(),
                    'phi': frame['phi'].flatten(), 'eta1': frame['eta1'].flatten(),
                    'eta2': frame['eta2'].flatten(),
                    'sigma_eq_GPa': (frame['sigma_eq']/1e9).flatten(),
                    'sigma_h_GPa': (frame['sigma_h']/1e9).flatten(),
                    'h_nm': frame['h'].flatten(),
                    'eps_p_mag': frame['eps_p_mag'].flatten()})
                zf.writestr(f"{sim_name}_frame_{idx:04d}.csv", df.to_csv(index=False))
        zip_buffer.seek(0)
        return zip_buffer, f"{sim_name}_csv.zip"

    @staticmethod
    @handle_errors(default_return=(None, None))
    def export_json(sim_data, params, history, sim_name):
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(i) for i in obj]
            return obj
        return (BytesIO(json.dumps({'sim_name': sim_name, 'params': params,
                'metadata': sim_data.get('metadata', {}), 'history': convert(history)},
                indent=2, cls=NumpyEncoder).encode()), f"{sim_name}.json")

    @staticmethod
    @handle_errors(default_return=(None, None))
    def export_hdf5(sim_data, params, history, sim_name, N, dx):
        if not H5PY_AVAILABLE:
            st.error("HDF5 export requires h5py. Install: pip install h5py")
            return None, None
        buffer = BytesIO()
        with h5py.File(buffer, 'w') as f:
            pg = f.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (int, float, str)): pg.attrs[k] = v
                elif isinstance(v, np.ndarray): pg.create_dataset(k, data=v)
            mg = f.create_group('metadata')
            for k, v in sim_data.get('metadata', {}).items():
                if isinstance(v, (int, float, str)): mg.attrs[k] = v
                elif isinstance(v, dict):
                    sg = mg.create_group(k)
                    for sk, sv in v.items():
                        if isinstance(sv, (int, float, str)): sg.attrs[sk] = sv
            f.create_dataset('x', data=np.linspace(-N*dx/2, N*dx/2, N))
            f.create_dataset('y', data=np.linspace(-N*dx/2, N*dx/2, N))
            fg = f.create_group('frames')
            for idx, frame in enumerate(history):
                g = fg.create_group(f'frame_{idx:04d}')
                for field in ['phi', 'eta1', 'eta2', 'sigma_eq', 'sigma_h',
                              'h', 'eps_p_mag', 'sigma_y']:
                    if field in frame: g.create_dataset(field, data=frame[field])
        buffer.seek(0)
        return buffer, f"{sim_name}.h5"

    @staticmethod
    @handle_errors(default_return=(None, None))
    def bulk_export_all_simulations(N, dx, extent):
        all_sims = SimulationDatabase.get_all_simulations()
        if not all_sims: return None, None
        bulk_buffer = BytesIO()
        with zipfile.ZipFile(bulk_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            summary = f"MULTI-SIMULATION EXPORT\nGenerated: {datetime.now().isoformat()}\n"
            summary += f"Total Simulations: {len(all_sims)}\n\n"
            for sim_id, sim_data in all_sims.items():
                try:
                    params = sim_data.get('params', {}); history = sim_data.get('results_history', [])
                    metadata = sim_data.get('metadata', {})
                    sim_name = build_sim_name(params, sim_id); sim_dir = f"simulation_{sim_id}"
                    zf.writestr(f"{sim_dir}/parameters.json", json.dumps(params, indent=2, cls=NumpyEncoder))
                    zf.writestr(f"{sim_dir}/metadata.json", json.dumps(metadata, indent=2, cls=NumpyEncoder))
                    x = np.linspace(extent[0], extent[1], N); y = np.linspace(extent[2], extent[3], N)
                    X, Y = np.meshgrid(x, y)
                    for idx, frame in enumerate(history):
                        df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(),
                            'phi': frame['phi'].flatten(),
                            'sigma_eq_GPa': (frame['sigma_eq']/1e9).flatten(),
                            'sigma_h_GPa': (frame['sigma_h']/1e9).flatten(),
                            'h_nm': frame['h'].flatten()})
                        zf.writestr(f"{sim_dir}/frame_{idx:04d}.csv", df.to_csv(index=False))
                    summary += f"\nSim {sim_id}:\n  Name: {sim_name}\n"
                except Exception as e:
                    summary += f"\nSim {sim_id}: ERROR - {e}\n"
            zf.writestr("EXPORT_SUMMARY.txt", summary)
        bulk_buffer.seek(0)
        return bulk_buffer, f"twin_all_simulations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"


# ============================================================================
# PARAMETER SWEEP
# ============================================================================
class ParameterSweep:
    @staticmethod
    @handle_errors(default_return=[])
    def run_sweep(base_params, param_name, values, save=True):
        results = []
        progress_bar = st.progress(0); status_text = st.empty()
        save_freq = base_params.get('save_frequency', 10)
        for i, val in enumerate(values):
            status_text.text(f"Running {param_name}={val:.3f} ({i+1}/{len(values)})")
            params = base_params.copy()
            params[param_name] = val
            for k in ('history', 'geom_viz'): params.pop(k, None)
            try:
                solver = NanotwinnedCuSolver(params)
                n_steps = params.get('n_steps', 100)
                frames = []
                for step in range(n_steps):
                    frame = solver.step()
                    if step % save_freq == 0:
                        frames.append(frame)
                final_conv = frames[-1].get('convergence', {}) if frames else {}
                if save:
                    SimulationDatabase.save_simulation(params, frames, None)
                results.append({'param_value': val, 'convergence': final_conv,
                                'solver': solver})
            except Exception as e:
                st.error(f"Failed for {param_name}={val}: {e}")
                results.append({'param_value': val, 'convergence': None, 'error': str(e)})
            progress_bar.progress((i + 1) / len(values))
        status_text.text("Sweep completed!")
        return results


# ============================================================================
# ============================================================================
# PLASTICITY PARAMETER RECOMMENDER  —  STAGES A–F
# ============================================================================
# ============================================================================

JSON_META_DIR = os.environ.get("JSON_META_DIR", "./json_metadatabase")

PARAM_FILES = {
    "mu":         "shear_modulus_metadatabase.json",
    "sigma0":     "friction_lattice_stress_metadatabase.json",
    "rho0":       "initial_dislocation_density_metadatabase.json",
    "srs":        "inverse_strain_rate_sensitivity_metadatabase.json",
    "gamma0_dot": "reference_strain_rate_metadatabase.json",
}

# Candidate snippet headers per file (assigned at load time by fuzzy match)
CAND_FIELDS = {
    "mu":         ["Candidate G Values", "Candidate Shear Modulus Values",
                   "Candidate μ Values"],
    "rho0":       ["Candidate ρ0 Values", "Candidate rho0 Values",
                   "Candidate Dislocation Density Values"],
    "srs":        ["Candidate SRS Values", "Candidate m Values",
                   "Candidate Strain Rate Sensitivity Values"],
    "sigma0":     ["Candidate σ0 Values", "Candidate sigma0 Values",
                   "Candidate Friction Stress Values"],
    "gamma0_dot": ["Candidate γ̇0 Values", "Candidate gamma0_dot Values",
                   "Candidate Reference Strain Rate Values"],
}


def decode_bytes_score(s) -> Optional[float]:
    """Decode a b'..' float32 repr'd as a string back to its float value."""
    if not isinstance(s, str):
        try: return float(s)
        except Exception: return None
    if s.startswith("b'") and s.endswith("'"):
        try:
            raw = s[2:-1].encode("latin-1").decode("unicode_escape").encode("latin-1")
            return round(struct.unpack("<f", raw[:4])[0], 3)
        except Exception:
            return None
    try: return float(s)
    except Exception: return None


def robust_load_json(fp: Path):
    """Load JSON, falling back to JSONL and CSV. Returns a list of dicts."""
    if not fp.exists(): return []
    try:
        text = fp.read_text(encoding="utf-8-sig")
    except Exception:
        return []
    if not text.strip(): return []
    try:
        data = json.loads(text)
        if isinstance(data, list): return data
        if isinstance(data, dict): return [data]
    except json.JSONDecodeError:
        pass
    # JSONL fallback
    records = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line: continue
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
    if records: return records
    # CSV fallback
    try: return pd.read_csv(fp).to_dict(orient="records")
    except Exception: return []


def _pick_candidate_field(rec: dict, param: str) -> str:
    for key in CAND_FIELDS[param]:
        if key in rec and rec[key]:
            return str(rec[key])
    # Fuzzy: any key containing 'candidate'
    for key in rec.keys():
        if isinstance(key, str) and 'candidate' in key.lower():
            return str(rec[key])
    return ""


# ----------------------------------------------------------------------------
# STAGE A — Ingest & normalize
# ----------------------------------------------------------------------------
@dataclass
class Evidence:
    param: str
    value: Optional[float] = None
    unit: Optional[str] = None
    uncertainty: Optional[float] = None
    quantity_kind: str = "unknown"
    material: str = "unknown"
    method: str = "unknown"
    temperature: Optional[str] = None
    strain_rate: Optional[str] = None
    paper_id: str = ""
    doi: str = ""
    year: int = 0
    title: str = ""
    snippet: str = ""
    full_text: str = ""
    relevance: float = 0.0
    extractor: str = "preextracted"
    confidence: float = 0.0
    reason: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            'param': self.param, 'value': self.value, 'unit': self.unit,
            'uncertainty': self.uncertainty, 'quantity_kind': self.quantity_kind,
            'material': self.material, 'method': self.method, 'year': self.year,
            'doi': self.doi, 'title': (self.title or "")[:80],
            'relevance': self.relevance, 'extractor': self.extractor,
            'confidence': self.confidence, 'reason': self.reason,
        }


@st.cache_data(show_spinner=False)
def load_evidence_corpus(json_dir: str = None) -> List[Dict[str, Any]]:
    """Return list of Evidence dicts (serializable for st.cache_data)."""
    if json_dir is None:
        json_dir = JSON_META_DIR
    seen, corpus = set(), []
    for param, fname in PARAM_FILES.items():
        fp = Path(json_dir) / fname
        if not fp.exists():
            continue
        for rec in robust_load_json(fp):
            if not isinstance(rec, dict): continue
            uid = (rec.get("unique_id") or rec.get("DOI") or
                   rec.get("doi") or
                   hashlib.md5(str(rec.get("Title", "")).encode()).hexdigest()[:12])
            key = (uid, param)
            if key in seen: continue
            seen.add(key)
            snippet = _pick_candidate_field(rec, param)
            ev = Evidence(
                param=param, paper_id=str(uid),
                doi=str(rec.get("DOI", rec.get("doi", ""))),
                year=int(rec.get("Year", 0) or 0),
                title=str(rec.get("Title", "")),
                snippet=snippet,
                full_text=str(rec.get("Full Text", rec.get("full_text", "")))[:20000],
                relevance=decode_bytes_score(rec.get("Relevance Score")) or 0.0,
            )
            corpus.append(asdict(ev))
    return corpus


# ----------------------------------------------------------------------------
# STAGE B — Domain categorization (multi-axis, closed-set)
# ----------------------------------------------------------------------------
class PlasticityOntology:
    PARAMS = {
        "mu": dict(syn=["shear modulus", "rigidity modulus", " g value",
                        "shear rigidity", "c44", "c'", "c prime"],
                   units={"gpa", "pa"}, rng=(5e9, 200e9),
                   confusions=["young", "elastic modulus", "bulk modulus"]),
        "sigma0": dict(syn=["friction stress", "lattice friction", "peierls",
                            "back stress", "sigma0", "σ0", "friction stress"],
                       units={"mpa", "pa"}, rng=(1e6, 5e8)),
        "rho0": dict(syn=["dislocation density", "threading dislocation",
                          "initial dislocation density", "rho0", "ρ0"],
                     units={"m-2", "cm-2", "m^-2", "cm^-2", "1/m2", "1/cm2"},
                     rng=(1e8, 1e17)),
        "srs": dict(syn=["strain rate sensitivity", "rate sensitivity",
                         "srs", "strain-rate sensitivity", "m value"],
                    units={"-", "dimensionless"}, rng=(0.001, 0.2)),
        "gamma0_dot": dict(syn=["reference strain rate", "characteristic strain rate",
                                "reference shear rate", "gamma0_dot",
                                "γ̇0", "gamma dot 0"],
                           units={"s-1", "s^-1", "1/s", "/s"}, rng=(1e-9, 1e9)),
    }
    QUANTITY_KINDS = ["shear_modulus", "youngs_modulus", "bulk_modulus",
                      "c44", "c_prime", "friction_stress", "dislocation_density",
                      "rate_sensitivity", "strain_rate", "other"]
    MATERIALS = ["nt_cu", "cu", "ni", "al", "other_fcc", "non_metallic", "unknown"]
    METHODS = ["experiment", "md", "dft", "calphad", "review", "model", "unknown"]


METHOD_TRUST = {"experiment": 1.0, "review": 0.9, "dft": 0.8, "calphad": 0.8,
                "model": 0.6, "md": 0.5, "unknown": 0.4}


def rule_categorize(rec: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic pre-pass. Uses title + snippet + full_text[:2000]."""
    text = " ".join([
        str(rec.get("title", "")),
        str(rec.get("snippet", ""))[:800],
        str(rec.get("full_text", ""))[:2000],
    ]).lower()
    out = {"parameter": None, "quantity_kind": "unknown",
           "material": "unknown", "method": "unknown",
           "reason": "rule_based"}
    # Parameter detection
    for p, spec in PlasticityOntology.PARAMS.items():
        if any(s in text for s in spec["syn"]):
            out["parameter"] = p
            break
    # Quantity disambiguation (the InP trap)
    if "young" in text or "elastic modulus" in text:
        out["quantity_kind"] = "youngs_modulus"
    elif "shear modulus" in text or "rigidity" in text:
        out["quantity_kind"] = "shear_modulus"
    elif "bulk modulus" in text:
        out["quantity_kind"] = "bulk_modulus"
    elif "dislocation density" in text:
        out["quantity_kind"] = "dislocation_density"
    elif "strain rate sensitivity" in text or "rate sensitivity" in text:
        out["quantity_kind"] = "rate_sensitivity"
    # Material
    if any(s in text for s in ["indium phosphide", " inp ", " inp,", " inp.", "inas", "gaas"]):
        out["material"] = "non_metallic"
    elif "nanotwin" in text and ("copper" in text or re.search(r"\bcu\b", text)):
        out["material"] = "nt_cu"
    elif "copper" in text or re.search(r"\bcu\b", text):
        out["material"] = "cu"
    elif "nickel" in text or re.search(r"\bni\b", text):
        out["material"] = "ni"
    elif "aluminum" in text or "aluminium" in text or re.search(r"\bal\b", text):
        out["material"] = "al"
    # Method
    method_map = {
        "md": ["molecular dynamics", "lammps", "eam potential", "interatomic potential"],
        "dft": ["density functional", "dft", "first-principles", "first principles", "vasp"],
        "calphad": ["calphad", "thermodynamic database"],
        "experiment": ["in situ", "tensile test", "nanoindentation", "measured",
                      "experimental", "micropillar"],
        "review": ["review", "survey", "meta-analysis"],
    }
    for m, keys in method_map.items():
        if any(k in text for k in keys):
            out["method"] = m
            break
    return out


# ----------------------------------------------------------------------------
# Ollama helper (JSON-mode)
# ----------------------------------------------------------------------------
def ollama_json(prompt: str, model: str = "qwen2.5:7b", timeout: int = 120) -> Optional[Dict]:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "format": "json",
                  "options": {"temperature": 0, "num_predict": 400}},
            timeout=timeout)
        r.raise_for_status()
        return json.loads(r.json().get("response", "{}"))
    except Exception as e:
        logger.warning(f"Ollama JSON call failed: {e}")
        return None


CATEGORIZE_PROMPT = """You curate a plasticity-parameter database for a nanotwinned-COPPER
phase-field model. Classify this literature record. Answer ONLY with JSON.

Title: {title}
Keywords: {keywords}
Candidate snippet: {snippet}

JSON schema:
{{"parameter": "mu|sigma0|rho0|srs|gamma0_dot|null",
  "quantity_kind": "shear_modulus|youngs_modulus|bulk_modulus|c44|friction_stress|dislocation_density|rate_sensitivity|strain_rate|other",
  "material": "nt_cu|cu|ni|al|other_fcc|non_metallic|unknown",
  "method": "experiment|md|dft|calphad|review|model|unknown",
  "usable_for_cu_plasticity": true|false,
  "reason": "one sentence"}}"""


def llm_categorize(rec: Dict[str, Any], model: str = "qwen2.5:7b") -> Dict[str, str]:
    prompt = CATEGORIZE_PROMPT.format(
        title=str(rec.get("title", ""))[:300],
        keywords=str(rec.get("snippet", ""))[:200],
        snippet=str(rec.get("snippet", ""))[:400])
    result = ollama_json(prompt, model=model)
    if not result: return {"parameter": None, "quantity_kind": "unknown",
                           "material": "unknown", "method": "unknown",
                           "reason": "llm_failed"}
    return {
        "parameter": result.get("parameter"),
        "quantity_kind": result.get("quantity_kind", "unknown"),
        "material": result.get("material", "unknown"),
        "method": result.get("method", "unknown"),
        "reason": result.get("reason", "llm_categorized"),
    }


# ----------------------------------------------------------------------------
# STAGE C — NER verification + regex recall
# ----------------------------------------------------------------------------
VERIFY_PROMPT = """Extract ONE physical quantity from the sentence. Answer ONLY with JSON.

Sentence: "{sentence}"
Context: paper about {material}; method: {method}.

JSON: {{"value": number|null, "uncertainty": number|null,
       "unit": "GPa|MPa|m-2|cm-2|s-1|dimensionless",
       "quantity_kind": "shear_modulus|youngs_modulus|bulk_modulus|friction_stress|dislocation_density|rate_sensitivity|strain_rate|other",
       "is_error_bar": true|false, "confidence": 0.0-1.0}}

Rules:
- If the number follows "±", it is an error bar, NOT the value.
- Young's modulus is not shear modulus.
- If the quantity is not about {target_param}, set value=null."""


SUP_TRANS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻×", "0123456789-*")
QTY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:±\s*(\d+(?:\.\d+)?))?\s*"
    r"(?:[×x*]\s*10\s*(?:\^)?\s*([+-]?\d+)|e([+-]?\d+))?\s*"
    r"(GPa|MPa|Pa|s[-−]1|s\^[-−]1|m[-−]2|cm[-−]2|m\^[-−]2|cm\^[-−]2)\b",
    re.IGNORECASE)


def regex_recall(full_text: str) -> List[Dict[str, Any]]:
    hits = []
    text_norm = full_text.translate(SUP_TRANS)
    for m in QTY_RE.finditer(text_norm):
        try:
            v = float(m.group(1))
            if m.group(3) or m.group(4):
                v *= 10 ** int(m.group(3) or m.group(4))
            hits.append({
                "value": v,
                "uncertainty": float(m.group(2)) if m.group(2) else None,
                "unit": m.group(5), "span": m.span(),
                "context": full_text[max(0, m.start()-120): m.end()+120]})
        except Exception:
            continue
    return hits


def verify_snippet_llm(snippet: str, material: str, method: str,
                       target_param: str, model: str = "qwen2.5:7b") -> Optional[Dict]:
    prompt = VERIFY_PROMPT.format(sentence=snippet[:400], material=material,
                                  method=method, target_param=target_param)
    return ollama_json(prompt, model=model)


def verify_snippet_regex(snippet: str) -> Optional[Dict[str, Any]]:
    """Fallback verification: regex over the snippet."""
    hits = regex_recall(snippet)
    if not hits: return None
    # Take the first plausible hit
    h = hits[0]
    return {"value": h["value"], "uncertainty": h["uncertainty"],
            "unit": h["unit"], "confidence": 0.5, "is_error_bar": False,
            "quantity_kind": "unknown"}


def verify_candidate(rec: Dict[str, Any], use_llm: bool = True,
                     model: str = "qwen2.5:7b") -> Optional[Dict[str, Any]]:
    snippet = rec.get("snippet", "") or ""
    if not snippet.strip(): return None
    if use_llm:
        parsed = verify_snippet_llm(snippet, rec.get("material", "unknown"),
                                    rec.get("method", "unknown"),
                                    rec.get("param", "unknown"), model)
        if parsed and parsed.get("value") is not None:
            return parsed
    return verify_snippet_regex(snippet)


# ----------------------------------------------------------------------------
# STAGE D — Graph, cross-attention, LatentMoE
# ----------------------------------------------------------------------------
def build_evidence_graph(verified: List[Evidence]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for ev in verified:
        p = f"paper:{ev.paper_id}"
        if not G.has_node(p):
            G.add_node(p, kind="paper", material=ev.material,
                       method=ev.method, year=ev.year,
                       relevance=ev.relevance, doi=ev.doi)
        v = f"value:{hash((ev.paper_id, ev.param, ev.value))}"
        G.add_node(v, kind="value", param=ev.param, value=ev.value,
                   unit=ev.unit, unc=ev.uncertainty,
                   kind_q=ev.quantity_kind)
        G.add_edge(p, v, rel="REPORTS")
        G.add_edge(v, f"param:{ev.param}", rel="OF_PARAM")
        G.add_edge(p, f"material:{ev.material}", rel="OF_MATERIAL")
    return G


def collect_for_param(G: nx.MultiDiGraph, param: str,
                      target_material: str = "cu") -> List[Tuple[Dict, float, Dict]]:
    out = []
    for n, d in G.nodes(data=True):
        if d.get("kind") != "value" or d.get("param") != param:
            continue
        try:
            paper = next(G.predecessors(n))
        except StopIteration:
            continue
        pd_ = G.nodes[paper]
        if pd_.get("material") not in (target_material, "nt_" + target_material):
            continue
        w = (pd_.get("relevance", 0) / 100.0) * METHOD_TRUST.get(pd_.get("method", "unknown"), 0.5)
        out.append((d, w, pd_))
    return out


class EvidenceAttentionRanker(nn.Module):
    """Untrained cross-attention = interpretable retrieval. Trains later via LOO."""
    def __init__(self, d: int = 384):
        super().__init__()
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)

    def forward(self, query_emb: torch.Tensor, ev_embs: torch.Tensor):
        q = self.q(query_emb).unsqueeze(0)
        a = torch.softmax(q @ self.k(ev_embs).T / (q.shape[-1] ** 0.5), dim=-1)
        return (a @ self.v(ev_embs)).squeeze(0), a.squeeze(0)


class LatentMoE(nn.Module):
    """Trainable MoE: per-expert heads over the pooled latent."""
    def __init__(self, d: int = 384, n_experts: int = 5):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
            for _ in range(n_experts)])
        self.gate = nn.Linear(d, n_experts)

    def forward(self, z: torch.Tensor):
        w = torch.softmax(self.gate(z), dim=-1)
        heads = torch.stack([e(z) for e in self.experts], 1)
        mu = (w * heads[..., 0]).sum(-1)
        var = torch.exp((w * heads[..., 1]).sum(-1))
        return mu, var, w


# Deterministic MoE v1 (aggregation strategy mixture)
def weighted_median(vals: List[float], weights: List[float]) -> float:
    if not vals: return 0.0
    order = np.argsort(vals)
    v = np.array(vals)[order]; w = np.array(weights)[order]
    cw = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(cw, 0.5)])


def moe_fuse_v1(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"value": None, "gate": {}, "n": 0}
    strategies = {
        "consensus":    lambda: weighted_median([r['value'] for r in records],
                                                [r.get('relevance', 1.0) for r in records]),
        "method_trust": lambda: float(np.average([r['value'] for r in records],
                                    weights=[r.get('relevance', 1.0) *
                                             METHOD_TRUST.get(r.get('method', 'unknown'), 0.5)
                                             for r in records])),
        "recent":       lambda: float(np.median([r['value'] for r in
                                    sorted(records, key=lambda x: -x.get('year', 0))[:5]])),
        "exact_match":  lambda: float(np.median([r['value'] for r in records
                                    if r.get('material') in ('cu', 'nt_cu')])) or 0.0,
    }
    gate = {"consensus": 0.35, "method_trust": 0.35, "recent": 0.10, "exact_match": 0.20}
    components = {}
    for k, fn in strategies.items():
        try: components[k] = fn()
        except Exception: components[k] = None
    total_w = 0.0; acc = 0.0
    for k, w in gate.items():
        if components.get(k) is not None:
            acc += w * components[k]; total_w += w
    value = acc / total_w if total_w > 0 else None
    return {"value": value, "gate": gate, "components": components,
            "n": len(records)}


# ----------------------------------------------------------------------------
# STAGE E — Convention mapping, physics guards
# ----------------------------------------------------------------------------
def vrh_average(material_props: Dict) -> float:
    C11 = material_props['elastic']['C11']
    C12 = material_props['elastic']['C12']
    C44 = material_props['elastic']['C44']
    Gv = (C11 - C12 + 3 * C44) / 5
    Gr = 5 * (C11 - C12) * C44 / (4 * C44 + 3 * (C11 - C12))
    return 0.5 * (Gv + Gr)


def to_model_params(raw: Dict[str, Optional[float]],
                    material_props: Dict) -> Tuple[Dict[str, Any], List[str]]:
    """Map raw recommendations → model-ready values. Returns (out, warnings)."""
    out = {}
    warns = []

    if raw.get("mu") is not None:
        mu_pa = raw["mu"] * 1e9 if raw["mu"] < 1e4 else raw["mu"]  # GPa → Pa heuristic
        mu_vrh = vrh_average(material_props)
        if abs(mu_pa - mu_vrh) / mu_vrh > 0.30:
            warns.append(f"μ={mu_pa/1e9:.1f} GPa deviates >30% from VRH({mu_vrh/1e9:.1f} GPa); "
                         f"possible Young's-modulus confusion.")
        out["mu"] = mu_pa

    if raw.get("sigma0") is not None:
        s = raw["sigma0"]
        out["sigma0"] = s * 1e6 if s < 1e4 else s

    if raw.get("rho0") is not None:
        out["rho0"] = raw["rho0"]  # assume already in m^-2 (unit harmonizer handled earlier)

    if raw.get("gamma0_dot") is not None:
        out["gamma0_dot"] = raw["gamma0_dot"]

    if raw.get("srs") is not None:
        srs = max(raw["srs"], 1e-4)
        m_model = max(1, int(round(1.0 / srs)))
        out["m"] = m_model
        out["srs_original"] = srs
        warns.append(f"SRS m_phys={srs:.4f} → overstress exponent m_model={m_model}.")

    # Hard range guards
    ranges = {"mu": (5e9, 200e9), "sigma0": (1e6, 5e8),
              "rho0": (1e8, 1e17), "m": (1, 200), "gamma0_dot": (1e-9, 1e9)}
    for k, (lo, hi) in ranges.items():
        if k in out and not (lo <= out[k] <= hi):
            warns.append(f"{k}={out[k]:.3e} outside [{lo:.1e}, {hi:.1e}] — clamped.")
            out[k] = float(np.clip(out[k], lo, hi))
    return out, warns


# ----------------------------------------------------------------------------
# STAGE F — Orchestrator
# ----------------------------------------------------------------------------
def recommend_plasticity_params(material_name: str = "Cu",
                                llm_model: str = "qwen2.5:7b",
                                use_llm: bool = True,
                                top_k_per_param: int = 30) -> Dict[str, Any]:
    """Full pipeline A–F. Returns a dict with recommended values + provenance."""
    corpus = load_evidence_corpus(JSON_META_DIR)
    if not corpus:
        return {"error": f"No evidence files found in {JSON_META_DIR}",
                "n_evidence": 0}

    # STAGE B: categorize all records (rule first, LLM only when rule is uncertain)
    for rec in corpus:
        rule = rule_categorize(rec)
        # Override param by file if rule says None
        if rule["parameter"] is None:
            rule["parameter"] = rec["param"]
        # LLM only if rule has low confidence for quantity_kind/material
        if use_llm and (rule["quantity_kind"] == "unknown" or rule["material"] == "unknown"):
            llm_out = llm_categorize(rec, model=llm_model)
            for k in ("quantity_kind", "material", "method"):
                if llm_out.get(k) and llm_out[k] != "unknown":
                    rule[k] = llm_out[k]
            rule["reason"] = "llm_categorized"
        rec.update(rule)

    # STAGE C: verify snippets for the target material
    verified: List[Evidence] = []
    excluded = 0
    for rec in corpus:
        # Reject wrong material / wrong quantity_kind for the target
        target_key = material_name.lower()
        if rec["material"] not in (target_key, f"nt_{target_key}"):
            excluded += 1
            continue
        # Quantity-kind sanity per parameter
        qk = rec["quantity_kind"]
        param = rec["param"]
        if param == "mu" and qk == "youngs_modulus":
            excluded += 1
            continue
        # Verify the snippet
        ver = verify_candidate(rec, use_llm=use_llm, model=llm_model)
        if not ver or ver.get("value") is None:
            excluded += 1
            continue
        v_val = ver["value"]
        unit = ver.get("unit", "")
        # Unit harmonization
        if param == "mu" and unit == "MPa":
            v_val = v_val / 1e3  # MPa → GPa
        elif param == "mu" and unit == "Pa":
            v_val = v_val / 1e9
        elif param == "sigma0" and unit == "Pa":
            v_val = v_val / 1e6
        # Range sanity
        lo, hi = PlasticityOntology.PARAMS[param]["rng"]
        if param == "mu":
            v_val_pa = v_val * 1e9 if v_val < 1e4 else v_val
            if not (lo <= v_val_pa <= hi):
                excluded += 1
                continue
            v_val = v_val_pa / 1e9  # store in GPa
        elif param == "sigma0":
            v_val_pa = v_val * 1e6 if v_val < 1e4 else v_val
            if not (lo <= v_val_pa <= hi):
                excluded += 1
                continue
            v_val = v_val_pa / 1e6  # store in MPa
        elif not (lo <= v_val <= hi):
            excluded += 1
            continue

        ev = Evidence(
            param=param, value=v_val, unit=unit,
            uncertainty=ver.get("uncertainty"),
            quantity_kind=qk, material=rec["material"],
            method=rec["method"], year=rec["year"],
            paper_id=rec["paper_id"], doi=rec.get("doi", ""),
            title=rec.get("title", ""),
            snippet=rec.get("snippet", "")[:200],
            relevance=rec.get("relevance", 0.0),
            extractor="llm" if use_llm else "regex",
            confidence=ver.get("confidence", 0.5),
            reason=rec.get("reason", ""))
        verified.append(ev)

    # STAGE D: graph + MoE aggregation per parameter
    G = build_evidence_graph(verified)
    recommendations = {}
    provenance = {}
    for param in PARAM_FILES.keys():
        recs_for_p = [e for e in verified if e.param == param]
        if not recs_for_p:
            recommendations[param] = None
            provenance[param] = {"n": 0, "gate": {}, "top_papers": []}
            continue
        records = [{'value': e.value, 'relevance': e.relevance, 'method': e.method,
                    'year': e.year, 'material': e.material} for e in recs_for_p]
        fused = moe_fuse_v1(records)
        recommendations[param] = fused["value"]
        top_papers = sorted(recs_for_p, key=lambda e: e.relevance, reverse=True)[:3]
        provenance[param] = {
            "n": len(recs_for_p),
            "gate": fused["gate"],
            "components": fused["components"],
            "top_papers": [{"doi": e.doi, "year": e.year,
                            "value": e.value, "unit": e.unit,
                            "method": e.method} for e in top_papers]}

    # STAGE E: convention mapping
    material_props = MaterialProperties.get_material(material_name)
    model_params, warns = to_model_params(recommendations, material_props)

    # Evidence table for UI
    evidence_table = pd.DataFrame([e.to_row() for e in verified])
    if not evidence_table.empty:
        evidence_table = evidence_table.sort_values(
            ["param", "relevance"], ascending=[True, False])

    return {
        "recommendations": recommendations,
        "model_params": model_params,
        "warnings": warns,
        "n_evidence": len(verified),
        "n_excluded": excluded,
        "evidence_table": evidence_table,
        "provenance": provenance,
        "graph": G,
        "raw_corpus_size": len(corpus),
    }


# ----------------------------------------------------------------------------
# SIDEBAR UI — recommender block
# ----------------------------------------------------------------------------
def render_plasticity_recommender_sidebar(material_name: str):
    """Renders the 5 plastic-parameter inputs + the Let-LLM-recommend button."""
    st.subheader("🧠 Plasticity Parameters (LLM-assisted)")

    defaults = MaterialProperties.get_material(material_name)['plasticity']

    # Seed the session state once
    if "p_mu" not in st.session_state:
        st.session_state["p_mu"] = float(defaults['mu'])
    if "p_sigma0" not in st.session_state:
        st.session_state["p_sigma0"] = float(defaults['sigma0'])
    if "p_rho0" not in st.session_state:
        st.session_state["p_rho0"] = float(defaults['rho0'])
    if "p_gamma0" not in st.session_state:
        st.session_state["p_gamma0"] = float(defaults['gamma0_dot'])
    if "p_m" not in st.session_state:
        st.session_state["p_m"] = int(defaults['m'])

    col1, col2 = st.columns(2)
    with col1:
        mu_in = st.number_input("μ (Pa)", value=st.session_state["p_mu"],
                                format="%.3e", key="mu_input_widget")
        sigma0_in = st.number_input("σ₀ (Pa)", value=st.session_state["p_sigma0"],
                                    format="%.3e", key="sigma0_input_widget")
        rho0_in = st.number_input("ρ₀ (m⁻²)", value=st.session_state["p_rho0"],
                                  format="%.3e", key="rho0_input_widget")
    with col2:
        gamma0_in = st.number_input("γ̇₀ (s⁻¹)", value=st.session_state["p_gamma0"],
                                    format="%.3e", key="gamma0_input_widget")
        m_in = st.number_input("m (overstress exp.)", value=st.session_state["p_m"],
                               min_value=1, max_value=200, step=1,
                               key="m_input_widget")

    # Sync back to session state
    st.session_state["p_mu"] = float(mu_in)
    st.session_state["p_sigma0"] = float(sigma0_in)
    st.session_state["p_rho0"] = float(rho0_in)
    st.session_state["p_gamma0"] = float(gamma0_in)
    st.session_state["p_m"] = int(m_in)

    with st.expander("⚙️ Recommender settings", expanded=False):
        llm_model = st.text_input("Ollama model", value="qwen2.5:7b", key="rec_model")
        use_llm_flag = st.checkbox("Use LLM verification (falls back to regex)",
                                   value=True, key="rec_use_llm")
        st.caption(f"Metadatabase folder: `{JSON_META_DIR}`")

    if st.button("🦙 Let LLM recommend", type="primary", use_container_width=True,
                 key="llm_recommend_btn"):
        with st.spinner("Categorizing → NER → graph reasoning → MoE fusion…"):
            result = recommend_plasticity_programmatic(
                material_name, llm_model=st.session_state.get("rec_model", "qwen2.5:7b"),
                use_llm=st.session_state.get("rec_use_llm", True))
        if result.get("error"):
            st.error(result["error"])
        else:
            mp = result["model_params"]
            # Fill input boxes via session state
            if "mu" in mp: st.session_state["p_mu"] = float(mp["mu"])
            if "sigma0" in mp: st.session_state["p_sigma0"] = float(mp["sigma0"])
            if "rho0" in mp: st.session_state["p_rho0"] = float(mp["rho0"])
            if "gamma0_dot" in mp: st.session_state["p_gamma0"] = float(mp["gamma0_dot"])
            if "m" in mp: st.session_state["p_m"] = int(mp["m"])

            # Persist provenance for the solver params
            st.session_state["recommender_provenance"] = {
                "n_evidence": result["n_evidence"],
                "n_excluded": result["n_excluded"],
                "warnings": result["warnings"],
                "provenance": result["provenance"],
            }

            st.success(f"✅ Recommendation ready ({result['n_evidence']} evidence, "
                       f"{result['n_excluded']} excluded)")
            if result["warnings"]:
                for w in result["warnings"]:
                    st.warning(w)
            # Provenance first (code 2's philosophy)
            with st.expander(f"📚 Evidence ({result['n_evidence']} records)"):
                et = result["evidence_table"]
                if isinstance(et, pd.DataFrame) and not et.empty:
                    st.dataframe(et, use_container_width=True, height=250)
                else:
                    st.info("No evidence table available.")
                st.markdown("**MoE gate per parameter:**")
                for p, prov in result["provenance"].items():
                    if prov["n"] > 0:
                        st.caption(f"`{p}` (n={prov['n']}, "
                                   f"gate={prov['gate']})")
            st.rerun()

    return {
        "mu": st.session_state["p_mu"],
        "sigma0": st.session_state["p_sigma0"],
        "rho0": st.session_state["p_rho0"],
        "gamma0_dot": st.session_state["p_gamma0"],
        "m": st.session_state["p_m"],
    }


@st.cache_data(show_spinner=False, ttl=600)
def recommend_plasticity_programmatic(material_name: str,
                                      llm_model: str = "qwen2.5:7b",
                                      use_llm: bool = True) -> Dict[str, Any]:
    """Cache-friendly wrapper: exclude the graph from the returned dict."""
    result = recommend_plasticity_params(material_name, llm_model, use_llm)
    if "graph" in result:
        result = {k: v for k, v in result.items() if k != "graph"}
    return result


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.set_page_config(page_title="Enhanced Nanotwinned Cu Phase-Field Simulator",
                       layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { height: 3rem; white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px; padding: 0.5rem 1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🔬 Enhanced Nanotwinned Cu Phase-Field Simulator (FFT)</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #F0F9FF; padding: 1.5rem; border-radius: 10px;
    border-left: 5px solid #3B82F6; margin-bottom: 1rem;">
    <strong>✅ PURE FFT SPECTRAL METHOD + LLM PLASTICITY RECOMMENDER</strong><br>
    • <span style="color: green;">NO FDM/NUMBA:</span> all spatial derivatives use exact spectral operators.<br>
    • <span style="color: green;">SEMI-IMPLICIT FOURIER:</span> Laplacian implicit → large Δt stability.<br>
    • <span style="color: green;">ANISOTROPY RETAINED:</span> κ_φ(m̂), L_φ(m̂) as explicit corrections.<br>
    • <span style="color: green;">LLM RECOMMENDER:</span> Reads 5 JSON metadatabases (μ, σ₀, ρ₀, γ̇₀, SRS)
      and recommends values via categorization → NER → graph → attention → LatentMoE → physics guards.
    </div>
    """, unsafe_allow_html=True)

    # Session state init
    for k, default in [("initialized", False), ("twin_simulations", {}),
                       ("results_history", None), ("comparison_config", None),
                       ("recommender_provenance", {})]:
        if k not in st.session_state:
            st.session_state[k] = default

    with st.sidebar:
        st.header("🔄 Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", type="secondary"):
                if 'twin_simulations' in st.session_state:
                    del st.session_state.twin_simulations
                st.success("All simulations cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
        if st.button("🔄 Reset to Defaults", type="secondary"):
            keys = ['N','dx','dt','twin_spacing','W','A','B','kappa0','gamma_aniso',
                    'kappa_eta','L_CTB','L_ITB','n_mob','L_eta','zeta','applied_stress',
                    'geom_type','defect_type','defect_x','defect_y','defect_radius',
                    'left_buffer_width','buffer_width','gb_profile','gb_curvature',
                    'grain_boundary_pos','stability_factor','enable_monitoring',
                    'auto_adjust_dt','n_steps','save_freq','confine_twin',
                    'p_mu','p_sigma0','p_rho0','p_gamma0','p_m']
            for k in keys:
                st.session_state.pop(k, None)
            st.success("Parameters reset to defaults!")
            st.rerun()

        st.markdown("---")
        operation_mode = st.radio(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations",
             "Single Simulation View", "Parameter Sweep"],
            index=0)

        if operation_mode == "Run New Simulation":
            st.header("🎛️ New Simulation Setup")
            st.subheader("🧪 Material")
            material_choice = st.selectbox("Select material", ["Cu", "Al", "Ni"],
                                           key="material")

            # >>>>>> LLM RECOMMENDER BLOCK <<<<<<
            plasticity_values = render_plasticity_recommender_sidebar(material_choice)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            st.subheader("🧩 Geometry Configuration")
            geometry_type = st.selectbox("Geometry Type",
                ["Standard Twin Grain", "Twin Grain with Defect"], key="geom_type")
            left_buffer_width = st.slider("Left buffer width (nm)", 0.0, 20.0, 5.0, 0.5,
                                          key="left_buffer_width")
            buffer_width = st.slider("Twin‑free buffer near GB (nm)", 0.0, 20.0, 5.0, 0.5,
                                     key="buffer_width")
            gb_profile = st.selectbox("Grain Boundary Profile",
                                     ["Plane", "Concave", "Convex"], key="gb_profile")
            gb_curvature = st.slider("GB Curvature Amplitude (nm)", 0.0, 20.0, 5.0, 0.5,
                                     key="gb_curvature")
            st.subheader("📊 Grid Configuration")
            N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64, key="N")
            dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1, key="dx")
            dt = st.slider("Time step (ns)", 1e-5, 1e-2, 1e-3, 1e-5, key="dt",
                          format="%.5f")
            st.subheader("🔬 Material Parameters")
            twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0,
                                     key="twin_spacing")
            grain_boundary_pos = st.slider("Grain boundary nominal position (nm)",
                                          -50.0, 50.0, 0.0, 1.0, key="grain_boundary_pos")
            if geometry_type == "Twin Grain with Defect":
                st.subheader("⚠️ Defect Parameters")
                defect_type = st.selectbox("Defect Type", ["Dislocation", "Void"],
                                          key="defect_type")
                defect_x = st.slider("Defect X (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_x")
                defect_y = st.slider("Defect Y (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_y")
                defect_radius = st.slider("Defect radius (nm)", 5.0, 30.0, 10.0, 1.0,
                                         key="defect_radius")
            st.subheader("⚡ Thermodynamic Parameters")
            W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1, key="W")
            A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5, key="A")
            B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5, key="B")
            st.subheader("🌀 Gradient Energy")
            kappa0 = st.slider("κ₀", 0.01, 10.0, 1.0, 0.1, key="kappa0")
            gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05,
                                   key="gamma_aniso")
            kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1, key="kappa_eta")
            st.subheader("⚡ Kinetic Parameters")
            L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001, key="L_CTB")
            L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1, key="L_ITB")
            n_mob = st.slider("n (mobility exponent)", 1, 10, 4, 1, key="n_mob")
            L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1, key="L_eta")
            zeta = st.slider("ζ (dislocation pinning)", 0.0, 2.0, 0.3, 0.05, key="zeta")
            st.subheader("🏋️ Loading Conditions")
            applied_stress_MPa = st.slider("Applied stress magnitude (MPa)",
                                          0.0, 1000.0, 300.0, 10.0, key="applied_stress")
            loading_angle = st.slider("Loading angle θ (deg)", 0.0, 180.0, 0.0, 5.0,
                                     key="loading_angle")
            st.subheader("⏯️ Simulation Control")
            n_steps = st.slider("Number of steps", 10, 1000, 100, 10, key="n_steps")
            save_frequency = st.slider("Save frequency", 1, 100, 10, 1, key="save_freq")
            with st.expander("🔧 Advanced Options"):
                stability_factor = st.slider("Stability factor", 0.1, 1.0, 0.5, 0.1,
                                            key="stability_factor")
                enable_monitoring = st.checkbox("Enable real-time monitoring", True,
                                               key="enable_monitoring")
                auto_adjust_dt = st.checkbox("Auto-adjust time step", True,
                                            key="auto_adjust_dt")
                confine_twin = st.checkbox("Confine twin evolution to twinned grain",
                                          True, key="confine_twin")
            st.subheader("🎨 Visualization Settings")
            global_cmap_phi = st.selectbox("Global φ colormap", cmap_list,
                index=cmap_list.index('RdBu_r') if 'RdBu_r' in cmap_list else 0,
                key="global_cmap_phi")
            global_cmap_stress = st.selectbox("Global σ_eq colormap", cmap_list,
                index=cmap_list.index('hot') if 'hot' in cmap_list else 0,
                key="global_cmap_stress")
            global_cmap_hydro = st.selectbox("Global σ_h colormap", cmap_list,
                index=cmap_list.index('RdBu') if 'RdBu' in cmap_list else 0,
                key="global_cmap_hydro")
            sim_cmap_phi = st.selectbox("Simulation φ colormap", cmap_list,
                index=cmap_list.index(global_cmap_phi) if global_cmap_phi in cmap_list else 0,
                key="sim_cmap_phi")
            sim_cmap_stress = st.selectbox("Simulation σ_eq colormap", cmap_list,
                index=cmap_list.index(global_cmap_stress) if global_cmap_stress in cmap_list else 0,
                key="sim_cmap_stress")
            sim_cmap_hydro = st.selectbox("Simulation σ_h colormap", cmap_list,
                index=cmap_list.index(global_cmap_hydro) if global_cmap_hydro in cmap_list else 0,
                key="sim_cmap_hydro")
            st.subheader("📏 Scale Bar Settings")
            scalebar_color = st.color_picker("Scale bar color", "#000000",
                                            key="scalebar_color")
            scalebar_fontsize = st.slider("Scale bar font size", 6, 20, 10, 1,
                                         key="scalebar_fontsize")

            if st.button("🚀 Initialize Simulation", type="primary", use_container_width=True):
                params = {
                    'material': material_choice,
                    'N': N, 'dx': dx, 'dt': dt,
                    'W': W, 'A': A, 'B': B,
                    'kappa0': kappa0, 'gamma_aniso': gamma_aniso,
                    'kappa_eta': kappa_eta,
                    'L_CTB': L_CTB, 'L_ITB': L_ITB, 'n_mob': n_mob,
                    'L_eta': L_eta, 'zeta': zeta,
                    'twin_spacing': twin_spacing,
                    'grain_boundary_pos': grain_boundary_pos,
                    'gb_width': 3.0, 'buffer_width': buffer_width,
                    'left_buffer_width': left_buffer_width,
                    'gb_profile': gb_profile.lower(),
                    'gb_curvature': gb_curvature,
                    'geometry_type': 'defect' if geometry_type == "Twin Grain with Defect" else 'standard',
                    'applied_stress': applied_stress_MPa * 1e6,
                    'applied_stress_angle': loading_angle,
                    'n_steps': n_steps, 'save_frequency': save_frequency,
                    'stability_factor': stability_factor,
                    'confine_twin': confine_twin,
                    'cmap_phi': sim_cmap_phi, 'cmap_stress': sim_cmap_stress,
                    'cmap_hydro': sim_cmap_hydro,
                    'global_cmap_phi': global_cmap_phi,
                    'global_cmap_stress': global_cmap_stress,
                    'global_cmap_hydro': global_cmap_hydro,
                    'scalebar_color': scalebar_color,
                    'scalebar_fontsize': scalebar_fontsize,
                    'plasticity': {
                        'mu': plasticity_values['mu'],
                        'sigma0': plasticity_values['sigma0'],
                        'rho0': plasticity_values['rho0'],
                        'gamma0_dot': plasticity_values['gamma0_dot'],
                        'm': plasticity_values['m'],
                    },
                    'recommender_provenance': st.session_state.get("recommender_provenance", {}),
                }
                if geometry_type == "Twin Grain with Defect":
                    params['defect_type'] = defect_type.lower()
                    params['defect_pos'] = (defect_x, defect_y)
                    params['defect_radius'] = defect_radius
                errors, warnings_list = MaterialProperties.validate_parameters(params)
                if errors:
                    st.error(f"Validation errors: {', '.join(errors)}")
                else:
                    if warnings_list:
                        st.warning(f"Parameter warnings: {', '.join(warnings_list)}")
                    geom_viz = InitialGeometryVisualizer(N, dx)
                    if geometry_type == "Twin Grain with Defect":
                        phi, eta1, eta2 = geom_viz.create_defect_geometry(
                            twin_spacing, defect_type.lower(), (defect_x, defect_y),
                            defect_radius, grain_boundary_pos, 3.0,
                            buffer_width, left_buffer_width,
                            gb_profile.lower(), gb_curvature)
                    else:
                        phi, eta1, eta2 = geom_viz.create_twin_grain_geometry(
                            twin_spacing, grain_boundary_pos, 3.0,
                            buffer_width, left_buffer_width,
                            gb_profile.lower(), gb_curvature)
                    st.session_state.initial_geometry = {
                        'phi': phi, 'eta1': eta1, 'eta2': eta2,
                        'geom_viz': geom_viz, 'params': params}
                    st.session_state.initialized = True
                    st.success("✅ Simulation initialized successfully!")

        elif operation_mode == "Compare Saved Simulations":
            st.header("🔍 Comparison Configuration")
            simulations = SimulationDatabase.get_simulation_list()
            if not simulations:
                st.warning("No simulations saved yet. Run some first!")
            else:
                sim_options = {sim['name']: sim['id'] for sim in simulations}
                selected_sim_ids = st.multiselect(
                    "Select Simulations to Compare",
                    options=list(sim_options.keys()),
                    default=list(sim_options.keys())[:min(3, len(sim_options))])
                comparison_type = st.selectbox("Comparison Type",
                    ["Side-by-Side Heatmaps", "Overlay Line Profiles",
                     "Statistical Summary", "Correlation Analysis",
                     "Evolution Timeline"], index=0)
                field_to_compare = st.selectbox("Field to Compare",
                    ["phi (Twin Order)", "eta1 (Grain)",
                     "sigma_eq (Von Mises Stress)",
                     "sigma_h (Hydrostatic Stress)",
                     "h (Twin Spacing)", "sigma_y (Yield Stress)"], index=2)
                field_key = field_to_compare.split()[0]
                if comparison_type == "Overlay Line Profiles":
                    profile_direction = st.selectbox("Profile Direction",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal", "Custom"],
                        index=0)
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                    profile_map = {"Horizontal": "horizontal", "Vertical": "vertical",
                                   "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal",
                                   "Custom": "custom"}
                    internal_direction = profile_map.get(profile_direction, "horizontal")
                    custom_angle = (st.slider("Custom angle (deg)", -180, 180, 45, 5)
                                    if profile_direction == "Custom" else 45)
                if st.button("🔬 Run Comparison", type="primary"):
                    comparison_config = {
                        'sim_ids': [sim_options[name] for name in selected_sim_ids],
                        'type': comparison_type, 'field': field_key}
                    if comparison_type == "Overlay Line Profiles":
                        comparison_config.update({
                            'profile_direction': internal_direction,
                            'position_ratio': position_ratio,
                            'custom_angle': custom_angle if profile_direction == "Custom" else None})
                    st.session_state.comparison_config = comparison_config
                    st.rerun()

        elif operation_mode == "Single Simulation View":
            st.header("🔍 Single Simulation View")
            simulations = SimulationDatabase.get_simulation_list()
            if not simulations:
                st.warning("No simulations saved yet.")
            else:
                sim_options = {sim['name']: sim['id'] for sim in simulations}
                selected_sim = st.selectbox("Select Simulation", list(sim_options.keys()))
                if selected_sim:
                    st.session_state.selected_sim_id = sim_options[selected_sim]

        elif operation_mode == "Parameter Sweep":
            st.header("📈 Parameter Sweep")
            st.subheader("Base Configuration")
            sweep_material = st.selectbox("Material", ["Cu", "Al", "Ni"],
                                          key="sweep_material")
            geom_type_sweep = st.selectbox("Geometry Type",
                ["Standard Twin Grain", "Twin Grain with Defect"], key="sweep_geom")
            col1, col2 = st.columns(2)
            with col1:
                N_sweep = st.slider("Grid size N", 64, 256, 128, 32, key="sweep_N")
                dx_sweep = st.slider("dx (nm)", 0.2, 1.0, 0.5, 0.1, key="sweep_dx")
                dt_sweep = st.slider("dt (ns)", 1e-5, 1e-2, 1e-3, 1e-5,
                                     format="%.5f", key="sweep_dt")
                W_sweep = st.slider("W (J/m³)", 0.5, 5.0, 2.0, 0.1, key="sweep_W")
            with col2:
                n_steps_sweep = st.slider("Number of steps", 20, 200, 50, 10,
                                          key="sweep_steps")
                save_freq_sweep = st.slider("Save frequency", 1, 50, 10, 1,
                                            key="sweep_savefreq")
                twin_spacing_sweep = st.slider("Twin spacing (nm)", 10.0, 50.0, 20.0, 1.0,
                                              key="sweep_twin_spacing")
                applied_stress_sweep = st.slider("Applied stress (MPa)", 0.0, 600.0, 300.0,
                                                10.0, key="sweep_stress")
            st.subheader("Sweep Parameter")
            sweep_param = st.selectbox("Choose parameter to vary",
                ["twin_spacing", "applied_stress", "applied_stress_angle",
                 "W", "L_CTB", "L_ITB", "kappa0"], index=0)
            col1, col2 = st.columns(2)
            with col1:
                sweep_min = st.number_input(f"Min {sweep_param}",
                    value=10.0 if sweep_param=="twin_spacing" else 0.0, key="sweep_min")
                sweep_max = st.number_input(f"Max {sweep_param}",
                    value=50.0 if sweep_param=="twin_spacing" else 500.0, key="sweep_max")
            with col2:
                sweep_steps = st.number_input("Number of steps", 2, 20, 5, 1,
                                              key="sweep_steps")
            if sweep_param == "applied_stress":
                sweep_values = np.linspace(sweep_min*1e6, sweep_max*1e6, sweep_steps)
            else:
                sweep_values = np.linspace(sweep_min, sweep_max, sweep_steps)
            st.write(f"Sweep values: {sweep_values}")
            base_params = {
                'material': sweep_material,
                'N': N_sweep, 'dx': dx_sweep, 'dt': dt_sweep, 'W': W_sweep,
                'A': 5.0, 'B': 10.0,
                'kappa0': 1.0, 'gamma_aniso': 0.7, 'kappa_eta': 2.0,
                'L_CTB': 0.05, 'L_ITB': 5.0, 'n_mob': 4, 'L_eta': 1.0, 'zeta': 0.3,
                'twin_spacing': twin_spacing_sweep,
                'grain_boundary_pos': 0.0, 'gb_width': 3.0,
                'buffer_width': 5.0, 'left_buffer_width': 5.0,
                'gb_profile': 'plane', 'gb_curvature': 0.0,
                'geometry_type': 'defect' if geom_type_sweep == "Twin Grain with Defect" else 'standard',
                'applied_stress': applied_stress_sweep * 1e6,
                'applied_stress_angle': 0.0,
                'n_steps': n_steps_sweep, 'save_frequency': save_freq_sweep,
                'stability_factor': 0.5, 'confine_twin': True}
            if geom_type_sweep == "Twin Grain with Defect":
                base_params['defect_type'] = 'dislocation'
                base_params['defect_pos'] = (0.0, 0.0)
                base_params['defect_radius'] = 10.0
            if st.button("🚀 Run Parameter Sweep", type="primary"):
                with st.spinner("Running parameter sweep..."):
                    sweep_results = ParameterSweep.run_sweep(
                        base_params, sweep_param, sweep_values, save=True)
                st.session_state.sweep_results = sweep_results
                st.session_state.sweep_param = sweep_param
                st.success("Parameter sweep completed!")
                st.rerun()

    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    if operation_mode == "Compare Saved Simulations" and st.session_state.get('comparison_config'):
        config = st.session_state.comparison_config
        simulations = []
        for sim_id in config['sim_ids']:
            sim = SimulationDatabase.get_simulation(sim_id)
            if sim: simulations.append(sim)
        if not simulations:
            st.error("No valid simulations found.")
        else:
            st.success(f"Loaded {len(simulations)} simulations")
            sim_names = [build_sim_name(sim['params'], sim['id']) for sim in simulations]
            if config['type'] == "Side-by-Side Heatmaps":
                last_frames = [sim['results_history'][-1] if sim['results_history'] else None
                               for sim in simulations]
                valid_indices = [i for i, f in enumerate(last_frames) if f is not None]
                if not valid_indices:
                    st.warning("No frame data available.")
                else:
                    n_sims = len(valid_indices)
                    cols = min(3, n_sims); rows = (n_sims + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
                    if rows == 1 and cols == 1: axes = np.array([axes])
                    else: axes = axes.flatten()
                    for idx, sim_idx in enumerate(valid_indices):
                        ax = axes[idx]; sim = simulations[sim_idx]
                        frame = last_frames[sim_idx]; field = config['field']
                        if field in frame:
                            data = frame[field].copy()
                            if field in ['sigma_eq', 'sigma_h']: data = data / 1e9
                            elif field == 'sigma_y': data = data / 1e6
                            N = sim['params']['N']; dx = sim['params']['dx']
                            extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
                            im = ax.imshow(data, extent=extent, cmap='viridis',
                                          origin='lower')
                            ax.set_title(sim_names[sim_idx][:30] + "...", fontsize=8)
                            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    for idx in range(len(valid_indices), len(axes)):
                        axes[idx].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)
            elif config['type'] == "Overlay Line Profiles":
                fig = go.Figure()
                ref = simulations[0]
                N = ref['params']['N']; dx = ref['params']['dx']
                visualizer = EnhancedTwinVisualizer(N, dx)
                for sim_idx, sim in enumerate(simulations):
                    if not sim['results_history']: continue
                    frame = sim['results_history'][-1]
                    field = config['field']
                    if field not in frame: continue
                    distance, profile, _ = visualizer.line_profiler.extract_profile(
                        frame[field], config['profile_direction'],
                        config['position_ratio'], config.get('custom_angle', 45))
                    if field in ['sigma_eq', 'sigma_h']:
                        profile = profile / 1e9; ylabel = 'Stress (GPa)'
                    elif field == 'sigma_y':
                        profile = profile / 1e6; ylabel = 'Stress (MPa)'
                    else: ylabel = field
                    fig.add_trace(go.Scatter(x=distance, y=profile, mode='lines',
                                            name=sim_names[sim_idx][:30]))
                fig.update_layout(title=f"{field} Line Profiles Comparison",
                                 xaxis_title="Position (nm)", yaxis_title=ylabel,
                                 hovermode='x unified', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif config['type'] == "Statistical Summary":
                data = []
                for sim in simulations:
                    params = sim['params']; hist = sim['results_history']
                    if hist:
                        last = hist[-1].get('convergence', {})
                        data.append({
                            'Name': build_sim_name(params, sim['id'])[:40],
                            'λ (nm)': params.get('twin_spacing', 0),
                            'σ_app (MPa)': params.get('applied_stress', 0)/1e6,
                            'θ (deg)': params.get('applied_stress_angle', 0),
                            'W (J/m³)': params.get('W', 0),
                            'Avg σ_eq (GPa)': last.get('avg_stress', 0)/1e9,
                            'Max σ_eq (GPa)': last.get('max_stress', 0)/1e9,
                            'Avg h (nm)': last.get('avg_spacing', 0),
                            'Plastic Work (J)': last.get('plastic_work', 0),
                            'Energy (J)': last.get('energy', 0)})
                if data:
                    df = pd.DataFrame(data); st.dataframe(df)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df['Name'], y=df['Avg σ_eq (GPa)'],
                                        name='Avg Stress'))
                    fig.add_trace(go.Bar(x=df['Name'], y=df['Max σ_eq (GPa)'],
                                        name='Max Stress'))
                    fig.update_layout(title="Stress Comparison",
                                     xaxis_title="Simulation", yaxis_title="Stress (GPa)")
                    st.plotly_chart(fig, use_container_width=True)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df['λ (nm)'], y=df['Avg σ_eq (GPa)'],
                                             mode='markers+text', text=df['Name'],
                                             textposition='top center'))
                    fig2.update_layout(title="Twin Spacing vs. Avg Stress",
                                      xaxis_title="λ (nm)", yaxis_title="Avg Stress (GPa)")
                    st.plotly_chart(fig2, use_container_width=True)
            elif config['type'] == "Correlation Analysis":
                data = []
                for sim in simulations:
                    params = sim['params']; hist = sim['results_history']
                    if hist and 'convergence' in hist[-1]:
                        conv = hist[-1]['convergence']
                        data.append({
                            'twin_spacing': params.get('twin_spacing', 0),
                            'applied_stress': params.get('applied_stress', 0)/1e6,
                            'W': params.get('W', 0),
                            'avg_stress': conv.get('avg_stress', 0)/1e9,
                            'max_stress': conv.get('max_stress', 0)/1e9,
                            'avg_spacing': conv.get('avg_spacing', 0),
                            'plastic_work': conv.get('plastic_work', 0)})
                if data:
                    df = pd.DataFrame(data)
                    fig = go.Figure(data=go.Splom(
                        dimensions=[dict(label=k, values=df[k]) for k in df.columns],
                        showupperhalf=False, marker=dict(size=8)))
                    fig.update_layout(title="Correlation Matrix", width=800, height=800)
                    st.plotly_chart(fig, use_container_width=True)
            elif config['type'] == "Evolution Timeline":
                fig = go.Figure()
                for sim_idx, sim in enumerate(simulations):
                    hist_frames = sim.get('results_history', [])
                    if not hist_frames: continue
                    dt = sim['params'].get('dt', 1e-4)
                    times = np.arange(len(hist_frames)) * dt * sim['params'].get('save_frequency', 1)
                    key_map = {'phi': 'phi_norm', 'sigma_eq': 'avg_stress',
                               'h': 'twin_spacing_avg', 'energy': 'energy',
                               'plastic_work': 'plastic_work'}
                    metric_key = key_map.get(config['field'], 'avg_stress')
                    values = [f.get('convergence', {}).get(metric_key, 0)
                              for f in hist_frames]
                    if metric_key in ('avg_stress', 'max_stress'):
                        values = np.array(values) / 1e9
                    fig.add_trace(go.Scatter(x=times, y=values, mode='lines',
                                            name=sim_names[sim_idx][:30]))
                fig.update_layout(title=f"{config['field']} Evolution Comparison",
                                 xaxis_title="Time (ns)", yaxis_title=config['field'],
                                 hovermode='x unified', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    elif operation_mode == "Single Simulation View" and st.session_state.get('selected_sim_id'):
        sim_id = st.session_state.selected_sim_id
        sim_data = SimulationDatabase.get_simulation(sim_id)
        if sim_data:
            st.header(f"📊 Single Simulation: {build_sim_name(sim_data['params'], sim_id)}")
            params = sim_data['params']
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("λ (twin spacing)", f"{params.get('twin_spacing', 0):.1f} nm")
            with col2:
                st.metric("σ_app / θ",
                         f"{params.get('applied_stress', 0)/1e6:.0f} MPa / "
                         f"{params.get('applied_stress_angle', 0):.0f}°")
            with col3: st.metric("W (well depth)", f"{params.get('W', 0):.2f} J/m³")
            with col4: st.metric("κ₀", f"{params.get('kappa0', 0):.2f}")
            history = sim_data.get('results_history', [])
            if history:
                num_frames = len(history)
                frame_idx = st.slider("Frame", 0, num_frames-1, num_frames-1,
                                     key=f"frame_slider_{sim_id}")
                results = history[frame_idx]
                visualizer = EnhancedTwinVisualizer(
                    params['N'], params['dx'], dt=params.get('dt', 1e-4))
                style_params = {
                    'phi_cmap': params.get('cmap_phi', 'RdBu_r'),
                    'eta1_cmap': params.get('cmap_eta1', 'Reds'),
                    'sigma_eq_cmap': params.get('cmap_stress', 'hot'),
                    'sigma_h_cmap': params.get('cmap_hydro', 'RdBu'),
                    'scalebar_color': params.get('scalebar_color', 'black'),
                    'scalebar_fontsize': params.get('scalebar_fontsize', 10)}
                fig = visualizer.create_multi_field_comparison(results, style_params)
                if fig: st.pyplot(fig); plt.close(fig)
                if st.button("🗑️ Delete This Simulation", key=f"delete_{sim_id}"):
                    SimulationDatabase.delete_simulation(sim_id)
                    st.session_state.pop('selected_sim_id', None)
                    st.success(f"Simulation {sim_id} deleted!")
                    st.rerun()
        else:
            st.error("Simulation not found.")

    elif operation_mode == "Run New Simulation" and st.session_state.get('initialized'):
        params = st.session_state.initial_geometry['params']
        N = params['N']; dx = params['dx']
        visualizer = EnhancedTwinVisualizer(N, dx, dt=params.get('dt', 1e-4))
        tabs = st.tabs(["📐 Initial Geometry", "▶️ Run Simulation", "📊 Basic Results",
                        "🔍 Advanced Analysis", "📊 Plotly Interactive",
                        "🖥️ 3D Interactive", "📤 Enhanced Export"])
        with tabs[0]:
            st.header("Initial Geometry Visualization")
            phi = st.session_state.initial_geometry['phi']
            eta1 = st.session_state.initial_geometry['eta1']
            kx, ky, k2 = make_k_vectors(N, dx)
            phi_gx, phi_gy = spectral_gradients(phi, kx, ky)
            h = compute_twin_spacing_from_gradient(phi_gx, phi_gy)
            initial_results = {'phi': phi, 'eta1': eta1, 'h': h}
            style_params = {'eta1_cmap': 'Reds',
                           'scalebar_color': params.get('scalebar_color', 'black'),
                           'scalebar_fontsize': params.get('scalebar_fontsize', 10)}
            fig = visualizer.create_multi_field_comparison(initial_results, style_params)
            if fig: st.pyplot(fig); plt.close(fig)
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_spacing = np.mean(h[(h>5)&(h<50)]) if np.any((h>5)&(h<50)) else 0
                st.metric("Avg Twin Spacing", f"{avg_spacing:.1f} nm")
            with col2:
                st.metric("Twin Grain Area", f"{np.sum(eta1 > 0.5) * dx**2:.0f} nm²")
            with col3:
                st.metric("Number of Twins", f"{np.sum(h < 20):.0f}")

        with tabs[1]:
            st.header("Run Simulation (Pure FFT Spectral Method)")
            if st.button("▶️ Start Evolution", type="secondary", use_container_width=True):
                with st.spinner("Running phase-field simulation (FFT)…"):
                    try:
                        solver = NanotwinnedCuSolver(params)
                        solver.phi = st.session_state.initial_geometry['phi'].copy()
                        solver.eta1 = st.session_state.initial_geometry['eta1'].copy()
                        solver.eta2 = st.session_state.initial_geometry['eta2'].copy()
                        progress_bar = st.progress(0); status_text = st.empty()
                        results_history = []; timesteps = []
                        monitoring_cols = st.columns(4)
                        n_steps = params['n_steps']; dt = params['dt']
                        save_freq = params['save_frequency']
                        for step in range(n_steps):
                            status_text.text(f"Step {step+1}/{n_steps} | "
                                            f"t={(step+1)*dt:.4f} ns")
                            results = solver.step()
                            if step % save_freq == 0:
                                results_history.append(results)
                                timesteps.append(step * dt)
                            progress_bar.progress((step + 1) / n_steps)
                            if step % 10 == 0 and results:
                                with monitoring_cols[0]:
                                    st.metric("Avg Stress",
                                              f"{np.mean(results['sigma_eq'])/1e9:.2f} GPa")
                                with monitoring_cols[1]:
                                    valid_h = results['h'][(results['h']>5)&(results['h']<50)]
                                    st.metric("Avg Spacing",
                                              f"{np.mean(valid_h) if len(valid_h)>0 else 0:.1f} nm")
                                with monitoring_cols[2]:
                                    st.metric("Max Plastic Strain",
                                              f"{np.max(results['eps_p_mag']):.4f}")
                                with monitoring_cols[3]:
                                    st.metric("Energy",
                                              f"{results['convergence']['energy']:.2e} J")
                        st.success(f"✅ Simulation completed! {len(results_history)} frames.")
                        st.session_state.results_history = results_history
                        st.session_state.timesteps = timesteps
                        st.session_state.solver = solver
                        start_time = datetime.now()
                        SimulationDatabase.save_simulation(
                            params, results_history, st.session_state.initial_geometry,
                            run_time=(datetime.now()-start_time).total_seconds())
                        st.balloons()
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")
                        st.exception(e)

        with tabs[2]:
            if st.session_state.get('results_history'):
                st.header("Basic Results Visualization")
                results_history = st.session_state.results_history
                frame_idx = st.slider("Select frame", 0, len(results_history)-1,
                                     len(results_history)-1)
                results = results_history[frame_idx]
                style_params = {
                    'phi_cmap': params.get('cmap_phi', 'RdBu_r'),
                    'eta1_cmap': params.get('cmap_eta1', 'Reds'),
                    'sigma_eq_cmap': params.get('cmap_stress', 'hot'),
                    'sigma_h_cmap': params.get('cmap_hydro', 'RdBu'),
                    'scalebar_color': params.get('scalebar_color', 'black'),
                    'scalebar_fontsize': params.get('scalebar_fontsize', 10)}
                fig = visualizer.create_multi_field_comparison(results, style_params)
                if fig: st.pyplot(fig); plt.close(fig)
                st.subheader("Convergence Monitoring")
                if hasattr(st.session_state, 'solver') and \
                   st.session_state.solver.history['phi_norm']:
                    full_timesteps = (np.arange(len(st.session_state.solver.history['phi_norm']))
                                     * params['dt'])
                    conv_fig = SimulationMonitor.create_convergence_plots(
                        st.session_state.solver.history, full_timesteps)
                    if conv_fig: st.pyplot(conv_fig); plt.close(conv_fig)
            else:
                st.info("Run a simulation first.")

        with tabs[3]:
            if st.session_state.get('results_history'):
                st.header("Advanced Analysis Tools")
                results = st.session_state.results_history[-1]
                col1, col2 = st.columns(2)
                with col1:
                    profile_types = st.multiselect("Profile Directions",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"],
                        default=["Horizontal", "Vertical"])
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                with col2:
                    field_to_profile = st.selectbox("Field to Profile",
                        ["phi", "eta1", "sigma_eq", "sigma_h", "h", "sigma_y"],
                        index=2)
                profile_map = {"Horizontal": "horizontal", "Vertical": "vertical",
                               "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal"}
                internal_types = [profile_map[pt] for pt in profile_types]
                profiler = EnhancedLineProfiler(N, dx)
                if internal_types:
                    fig_profiles, axes = plt.subplots(len(internal_types), 1,
                                                     figsize=(10, 4*len(internal_types)))
                    if len(internal_types) == 1: axes = [axes]
                    for idx, ptype in enumerate(internal_types):
                        ax = axes[idx]
                        distance, profile, _ = profiler.extract_profile(
                            results[field_to_profile], ptype, position_ratio)
                        if field_to_profile in ['sigma_eq', 'sigma_h']:
                            profile = profile / 1e9; ylabel = 'Stress (GPa)'
                        elif field_to_profile == 'sigma_y':
                            profile = profile / 1e6; ylabel = 'Stress (MPa)'
                        else: ylabel = field_to_profile
                        ax.plot(distance, profile, 'b-', linewidth=2)
                        ax.set_xlabel('Position (nm)'); ax.set_ylabel(ylabel)
                        ax.set_title(f'{ptype.replace("_", " ").title()} Profile')
                        ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_profiles); plt.close(fig_profiles)
            else:
                st.info("Run a simulation first.")

        with tabs[4]:
            if st.session_state.get('results_history'):
                st.header("📊 Plotly Interactive Visualization (2D)")
                results_history = st.session_state.results_history
                plotly_field = st.selectbox("Select field to visualize",
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h", "eps_p_mag", "sigma_y"],
                    index=0, key="plotly_2d_field")
                frame_idx = st.slider("Frame", 0, len(results_history)-1,
                                     len(results_history)-1, key="plotly_2d_frame")
                results = results_history[frame_idx]
                fig_heatmap = visualizer.create_plotly_heatmap(results, plotly_field, frame_idx)
                if fig_heatmap: st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown("---")
                st.subheader("Interactive Line Profiles")
                col1, col2 = st.columns(2)
                with col1:
                    profile_type_plotly = st.selectbox("Profile direction",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"],
                        key="plotly_2d_profile")
                with col2:
                    position_ratio_plotly = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                                     key="plotly_2d_pos")
                profile_map = {"Horizontal": "horizontal", "Vertical": "vertical",
                               "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal"}
                internal_pt = profile_map.get(profile_type_plotly, "horizontal")
                fig_line = visualizer.create_plotly_line_profiles(
                    results, plotly_field, [internal_pt], position_ratio_plotly)
                if fig_line: st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Run a simulation first.")

        with tabs[5]:
            st.header("🖥️ 3D Interactive Surface Visualization")
            if st.session_state.get('results_history'):
                results_history = st.session_state.results_history
                field_3d = st.selectbox("Select field for 3D surface",
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h", "eps_p_mag", "sigma_y"],
                    index=1, key="3d_field")
                frame_idx_3d = st.slider("Frame", 0, len(results_history)-1,
                                        len(results_history)-1, key="3d_frame")
                results = results_history[frame_idx_3d]
                fig_3d = visualizer.create_plotly_3d_surface(results, field_3d, frame_idx_3d)
                if fig_3d: st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("Run a simulation first.")

        with tabs[6]:
            st.header("📤 Enhanced Export")
            if st.session_state.get('results_history'):
                results_history = st.session_state.results_history
                params = st.session_state.initial_geometry['params']
                sim_id = SimulationDatabase.generate_id(params)
                sim_name = build_sim_name(params, sim_id)
                sim_data = {'metadata': MetadataManager.create_metadata(params, results_history),
                            'params': params}
                st.subheader("Export Simulation Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📦 Pickle (PKL)"):
                        b, f = DataExporter.export_pkl(sim_data, params, results_history, sim_name)
                        if b: st.download_button("Download PKL", b, f)
                    if st.button("🔥 PyTorch (PT)"):
                        b, f = DataExporter.export_pt(sim_data, params, results_history, sim_name)
                        if b: st.download_button("Download PT", b, f)
                    if st.button("📄 SQL Dump"):
                        b, f = DataExporter.export_sql(sim_data, params, results_history,
                                                       sim_name, sim_id, params['N'], params['dx'])
                        if b: st.download_button("Download SQL", b, f)
                with col2:
                    if st.button("📊 CSV (ZIP)"):
                        vis = EnhancedTwinVisualizer(params['N'], params['dx'])
                        b, f = DataExporter.export_csv(results_history, sim_name, vis.extent,
                                                      params['N'], params['dx'])
                        if b: st.download_button("Download CSV ZIP", b, f)
                    if st.button("📋 JSON"):
                        b, f = DataExporter.export_json(sim_data, params, results_history, sim_name)
                        if b: st.download_button("Download JSON", b, f)
                    if st.button("📁 HDF5"):
                        b, f = DataExporter.export_hdf5(sim_data, params, results_history,
                                                        sim_name, params['N'], params['dx'])
                        if b: st.download_button("Download HDF5", b, f)
                with col3:
                    st.subheader("Animation Export")
                    anim_field = st.selectbox("Field for animation",
                        ["phi", "eta1", "sigma_eq", "sigma_h", "h", "eps_p_mag"],
                        key="anim_field")
                    anim_format = st.selectbox("Format", ["gif", "mp4"], key="anim_format")
                    fps = st.slider("FPS", 1, 30, 5, key="anim_fps")
                    if st.button("🎬 Generate Animation"):
                        with st.spinner("Creating animation…"):
                            vis = EnhancedTwinVisualizer(params['N'], params['dx'],
                                                        dt=params.get('dt', 1e-4))
                            buf = vis.create_animation(results_history, anim_field,
                                                      anim_format, fps)
                            if buf:
                                st.download_button(f"Download {anim_format.upper()}",
                                                  buf, f"{sim_name}_{anim_field}.{anim_format}")
                st.markdown("---")
                st.subheader("Bulk Export All Simulations")
                if st.button("📦 Export All Simulations"):
                    vis = EnhancedTwinVisualizer(params['N'], params['dx'])
                    b, f = DataExporter.bulk_export_all_simulations(params['N'], params['dx'],
                                                                   vis.extent)
                    if b: st.download_button("Download All Simulations ZIP", b, f)
            else:
                st.info("Run a simulation first.")

    elif operation_mode == "Parameter Sweep" and st.session_state.get('sweep_results'):
        st.header("📊 Parameter Sweep Results")
        sweep_results = st.session_state.sweep_results
        sweep_param = st.session_state.sweep_param
        param_vals, avg_stress, max_stress, avg_spacing, plastic_work = [], [], [], [], []
        for res in sweep_results:
            if res['convergence'] is not None:
                param_vals.append(res['param_value'])
                conv = res['convergence']
                avg_stress.append(conv.get('avg_stress', 0) / 1e9)
                max_stress.append(conv.get('max_stress', 0) / 1e9)
                avg_spacing.append(conv.get('avg_spacing', 0))
                plastic_work.append(conv.get('plastic_work', 0))
        if sweep_param == 'applied_stress':
            param_display = np.array(param_vals) / 1e6
            param_label = "Applied Stress (MPa)"
        else:
            param_display = param_vals; param_label = sweep_param.replace('_', ' ').title()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0,0].plot(param_display, avg_stress, 'o-', linewidth=2)
        axes[0,0].set_xlabel(param_label); axes[0,0].set_ylabel("Avg σ_eq (GPa)")
        axes[0,0].set_title("Stress vs Parameter"); axes[0,0].grid(True, alpha=0.3)
        axes[0,1].plot(param_display, avg_spacing, 's-', color='green', linewidth=2)
        axes[0,1].set_xlabel(param_label); axes[0,1].set_ylabel("Avg Twin Spacing (nm)")
        axes[0,1].set_title("Twin Spacing vs Parameter"); axes[0,1].grid(True, alpha=0.3)
        axes[1,0].plot(param_display, plastic_work, 'd-', color='red', linewidth=2)
        axes[1,0].set_xlabel(param_label); axes[1,0].set_ylabel("Plastic Work (J)")
        axes[1,0].set_title("Plastic Work vs Parameter"); axes[1,0].grid(True, alpha=0.3)
        axes[1,1].plot(param_display, max_stress, '^-', color='purple', linewidth=2,
                      label='Max')
        axes[1,1].plot(param_display, avg_stress, 'o-', color='blue', linewidth=2,
                      label='Avg')
        axes[1,1].set_xlabel(param_label); axes[1,1].set_ylabel("Stress (GPa)")
        axes[1,1].set_title("Stress Extremes vs Parameter"); axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        df_sweep = pd.DataFrame({param_label: param_display,
                                 'Avg Stress (GPa)': avg_stress,
                                 'Max Stress (GPa)': max_stress,
                                 'Avg Spacing (nm)': avg_spacing,
                                 'Plastic Work (J)': plastic_work})
        st.dataframe(df_sweep)
        if st.button("Clear Sweep Results"):
            st.session_state.pop('sweep_results', None)
            st.session_state.pop('sweep_param', None)
            st.rerun()


if __name__ == "__main__":
    main()
