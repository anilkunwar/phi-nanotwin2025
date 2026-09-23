# ============================================================================
# ███ ENHANCED NANOTWINNED Cu PHASE-FIELD SIMULATOR (PURE FFT SPECTRAL) ███
# ███ + PLASTICITY PARAMETER INTELLIGENT RECOMMENDER v8.3.0              ███
# ███ PUBLICATION-QUALITY VISUALS DASHBOARD                             ███
# ███ STREAMLIT NESTED-EXPANDER FIX APPLIED (v8.0.1)                   ███
# ███ FULL CACHE PURGE ON "FORCE RELOAD CORPUS" (v8.1.1)               ███
# ███ FIX: LLM-DRIVEN PRIOR LEARNER + OLLAMA RAW RESPONSE DEBUG (v8.1.2)███
# ███ NEW: DUAL-MODE PROMPTS + CHAIN-OF-THOUGHT REASONING (v8.2.0)      ███
# ███ FIX: GATEKEEPER — PARAM ALIAS CANONICALIZATION + PER-PARAM        ███
# ███      HEURISTIC FALLBACK + VALUE COERCION + UI DEBUG TOGGLE        ███
# ============================================================================

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import matplotlib.animation as animation
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import zipfile
import pickle
import torch
import sqlite3
import hashlib
import traceback
import warnings
from scipy import stats
from io import BytesIO, StringIO
import tempfile
import os
import re
import math
import time
import threading
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Optional HDF5 export
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# Optional retrieval stack (FAISS + SentenceTransformer)
try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import faiss as _faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================
def handle_errors(func):
    """Decorator to handle errors gracefully and log them."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"❌ Error in {func.__name__}: {str(e)}"
            st.error(error_msg)
            st.error("Please check the console for detailed error information.")
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None
    return wrapper


# ============================================================================
# SPECTRAL DERIVATIVE HELPERS
# ============================================================================
def make_k_vectors(N, dx):
    """Build the Fourier wavenumber grids once. Returns (kx, ky, k2)."""
    kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1)
    ky = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-12  # regularize the mean mode
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    """Exact spectral gradient of a periodic 2-D field."""
    fh = fft2(field)
    gx = np.real(ifft2(1j * kx * fh))
    gy = np.real(ifft2(1j * ky * fh))
    return gx, gy


def spectral_laplacian(field, k2):
    """Exact spectral Laplacian of a periodic 2-D field."""
    fh = fft2(field)
    lap = np.real(ifft2(-k2 * fh))
    return lap


def compute_twin_spacing_from_gradient(phi_gx, phi_gy):
    """Local twin spacing h = 2/|∇φ| with saturation for vanishing gradients."""
    grad_mag = np.sqrt(phi_gx**2 + phi_gy**2)
    h = np.where(grad_mag > 1e-12, 2.0 / np.maximum(grad_mag, 1e-12), 1e6)
    return h


def compute_anisotropic_properties(phi_gx, phi_gy, nx, ny, kappa0,
                                   gamma_aniso, L_CTB, L_ITB, n_mob):
    """Anisotropic κ_φ(m̂) and L_φ(m̂) evaluated pointwise in real space."""
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
    """Transformation eigenstrain, scaling smoothly with eta1."""
    f_phi = 0.25 * (phi**3 - phi**2 - phi + 1)
    eta1_clamped = np.clip(eta1, 0.0, 1.0)
    exx_star = gamma_tw * nx * ax * f_phi * eta1_clamped
    eyy_star = gamma_tw * ny * ay * f_phi * eta1_clamped
    exy_star = 0.5 * gamma_tw * (nx * ay + ny * ax) * f_phi * eta1_clamped
    return exx_star, eyy_star, exy_star


def compute_yield_stress(h, sigma0, mu, b, nu):
    """Hall–Petch-like yield stress, vectorized."""
    safe = h > 2 * b
    sigma_y = np.empty_like(h)
    log_term = np.log(np.maximum(h, 2.001 * b) / b)
    sigma_y[safe] = sigma0 + (mu * b / (2 * np.pi * h[safe] * (1 - nu))) * log_term[safe]
    sigma_y[~safe] = sigma0 + mu / (2 * np.pi * (1 - nu))
    return sigma_y


def update_plastic_strain(sigma_eq, sigma_y, eps_p_xx, eps_p_yy, eps_p_xy,
                          gamma0_dot, m, dt):
    """Vectorized J2 power-law plastic-strain update with physical clamps."""
    MAX_OVERSTRESS = 1.0
    MAX_PLASTIC_STRAIN = 0.1
    overstress = np.maximum(sigma_eq - sigma_y, 0.0) / np.maximum(sigma_y, 1e-9)
    overstress = np.minimum(overstress, MAX_OVERSTRESS)
    gamma_dot = gamma0_dot * overstress**m
    stress_dev = 2.0 / 3.0 * gamma_dot * dt
    stress_dev = np.minimum(stress_dev, 0.001)
    d_xx = stress_dev
    d_yy = -0.5 * stress_dev
    d_xy = 0.5 * stress_dev
    eps_p_xx_new = np.clip(eps_p_xx + d_xx, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_yy_new = np.clip(eps_p_yy + d_yy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_xy_new = np.clip(eps_p_xy + d_xy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    return eps_p_xx_new, eps_p_yy_new, eps_p_xy_new


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================
class MetadataManager:
    """Centralized metadata management to ensure consistency"""

    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        if run_time is None:
            run_time = 0.0

        metadata = {
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
        return metadata

    @staticmethod
    def validate_metadata(metadata):
        if not isinstance(metadata, dict):
            metadata = {}
        required_fields = ['run_time', 'frames', 'grid_size', 'dx', 'dt', 'created_at']
        for field in required_fields:
            if field not in metadata:
                if field == 'created_at':
                    metadata[field] = datetime.now().isoformat()
                elif field == 'run_time':
                    metadata[field] = 0.0
                elif field == 'frames':
                    metadata[field] = 0
                elif field == 'grid_size':
                    metadata[field] = 256
                elif field == 'dx':
                    metadata[field] = 0.5
                elif field == 'dt':
                    metadata[field] = 1e-4
        if 'colormaps' not in metadata:
            metadata['colormaps'] = {
                'phi': 'RdBu_r', 'sigma_eq': 'hot', 'sigma_h': 'RdBu',
                'h': 'plasma', 'eta1': 'Reds'
            }
        return metadata

    @staticmethod
    def get_metadata_field(metadata, field, default=None):
        try:
            return metadata.get(field, default)
        except Exception:
            return default


# ============================================================================
# JOURNAL TEMPLATES
# ============================================================================
class JournalTemplates:
    """Publication-quality journal templates"""

    @staticmethod
    def get_journal_styles():
        return {
            'nature': {
                'figure_width_single': 8.9, 'figure_width_double': 18.3,
                'font_family': 'Arial', 'font_size_small': 7,
                'font_size_medium': 8, 'font_size_large': 9,
                'line_width': 0.5, 'axes_linewidth': 0.5,
                'tick_width': 0.5, 'tick_length': 2, 'grid_alpha': 0.1, 'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                '#bcbd22', '#17becf']
            },
            'science': {
                'figure_width_single': 5.5, 'figure_width_double': 11.4,
                'font_family': 'Helvetica', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 0.75, 'axes_linewidth': 0.75,
                'tick_width': 0.75, 'tick_length': 3, 'grid_alpha': 0.15, 'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                                '#77AC30', '#4DBEEE', '#A2142F', '#FF00FF',
                                '#00FFFF', '#FFA500']
            },
            'advanced_materials': {
                'figure_width_single': 8.6, 'figure_width_double': 17.8,
                'font_family': 'Arial', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0.2, 'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000',
                                '#44AA99', '#882255', '#117733', '#999933',
                                '#AA4499', '#88CCEE']
            },
            'prl': {
                'figure_width_single': 3.4, 'figure_width_double': 7.0,
                'font_family': 'Times New Roman', 'font_size_small': 8,
                'font_size_medium': 10, 'font_size_large': 12,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0, 'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                                '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                                '#999999', '#FFFFFF']
            },
            'custom': {
                'figure_width_single': 6.0, 'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans', 'font_size_small': 10,
                'font_size_medium': 12, 'font_size_large': 14,
                'line_width': 1.5, 'axes_linewidth': 1.5,
                'tick_width': 1.0, 'tick_length': 5, 'grid_alpha': 0.3, 'dpi': 300,
                'color_cycle': plt.cm.Set2(np.linspace(0, 1, 10))
            }
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
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['top'].set_linewidth(style['axes_linewidth'] * 0.5)
                ax.spines['right'].set_linewidth(style['axes_linewidth'] * 0.5)
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=style['tick_length'])
                ax.tick_params(which='minor', length=style['tick_length'] * 0.6)
        return fig, style


# ============================================================================
# ENHANCED COLORMAP LIBRARY
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
    'rocket': 'rocket', 'mako': 'mako', 'crest': 'crest', 'flare': 'flare',
    'icefire': 'icefire', 'vlag': 'vlag'
}

cmap_list = list(COLORMAPS.keys())

# Publication-friendly distinct color palettes for categorical data
PUBLICATION_PALETTES = {
    'okabe_ito': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                  '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
    'tol_bright': ['#4477AA', '#EE6677', '#228833', '#CCBB44',
                   '#66CCEE', '#AA3377', '#BBBBBB'],
    'tol_muted': ['#CC6677', '#332288', '#DDCC77', '#117733',
                  '#88CCEE', '#882255', '#44AA99', '#999933',
                   '#AA4499', '#DDDDDD'],
    'ibm_carbon': ['#6929c4', '#1192e8', '#005d5d', '#9f1853',
                   '#fa4d56', '#570408', '#198038', '#002d9c',
                   '#ee538b', '#b28600'],
    'nature_classic': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf'],
    'science_aaas': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                     '#77AC30', '#4DBEEE', '#A2142F'],
    'ieee': ['#0000FF', '#FF0000', '#00AA00', '#FF8000',
             '#800080', '#00AAAA', '#800000'],
}

FONT_FAMILIES = [
    'Arial', 'Helvetica', 'Times New Roman', 'Courier New',
    'DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono',
    'Calibri', 'Cambria', 'Georgia', 'Verdana', 'Tahoma',
    'Trebuchet MS', 'Garamond', 'Palatino', 'Bookman Old Style',
    'sans-serif', 'serif', 'monospace'
]


# ============================================================================
# POST-PROCESSING CLASSES
# ============================================================================
class EnhancedLineProfiler:
    """Enhanced line profile system with multiple orientations."""

    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N * dx / 2, N * dx / 2, -N * dx / 2, N * dx / 2]

    @handle_errors
    def extract_profile(self, data, profile_type, position_ratio=0.5, angle_deg=45):
        profile_type = self._normalize_profile_type(profile_type)
        ny, nx = data.shape
        center_x, center_y = nx // 2, ny // 2

        if profile_type in ['horizontal', 'vertical']:
            offset = int(min(nx, ny) * 0.4 * position_ratio)
        else:
            offset = int(min(nx, ny) * 0.3 * position_ratio)

        if profile_type == 'horizontal':
            row_idx = center_y + offset
            profile = data[row_idx, :]
            distance = np.linspace(self.extent[0], self.extent[1], nx)
            endpoints = (self.extent[0], row_idx * self.dx + self.extent[2],
                         self.extent[1], row_idx * self.dx + self.extent[2])
        elif profile_type == 'vertical':
            col_idx = center_x + offset
            profile = data[:, col_idx]
            distance = np.linspace(self.extent[2], self.extent[3], ny)
            endpoints = (col_idx * self.dx + self.extent[0], self.extent[2],
                         col_idx * self.dx + self.extent[0], self.extent[3])
        elif profile_type == 'diagonal':
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x - diag_length // 2, center_y - diag_length // 2)
            profile = []
            distances = []
            for i in range(diag_length):
                x = start_idx[0] + i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    dist = i * self.dx * np.sqrt(2)
                    distances.append(dist - (diag_length // 2) * self.dx * np.sqrt(2))
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = start_idx[0] * self.dx + self.extent[0]
            y_start = start_idx[1] * self.dx + self.extent[2]
            x_end = (start_idx[0] + diag_length - 1) * self.dx + self.extent[0]
            y_end = (start_idx[1] + diag_length - 1) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        elif profile_type == 'anti_diagonal':
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x + diag_length // 2, center_y - diag_length // 2)
            profile = []
            distances = []
            for i in range(diag_length):
                x = start_idx[0] - i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    dist = i * self.dx * np.sqrt(2)
                    distances.append(dist - (diag_length // 2) * self.dx * np.sqrt(2))
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = start_idx[0] * self.dx + self.extent[0]
            y_start = start_idx[1] * self.dx + self.extent[2]
            x_end = (start_idx[0] - diag_length + 1) * self.dx + self.extent[0]
            y_end = (start_idx[1] + diag_length - 1) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        elif profile_type == 'custom':
            angle_rad = np.deg2rad(angle_deg)
            length = int(min(nx, ny) * 0.8)
            dx_line = np.cos(angle_rad) * length // 2
            dy_line = np.sin(angle_rad) * length // 2
            profile = []
            distances = []
            for t in np.linspace(-length // 2, length // 2, length):
                x = center_x + t * np.cos(angle_rad) + offset * np.cos(angle_rad + np.pi / 2)
                y = center_y + t * np.sin(angle_rad) + offset * np.sin(angle_rad + np.pi / 2)
                if 0 <= x < nx - 1 and 0 <= y < ny - 1:
                    x0, y0 = int(x), int(y)
                    x1, y1 = x0 + 1, y0 + 1
                    if x1 >= nx:
                        x1 = nx - 1
                    if y1 >= ny:
                        y1 = ny - 1
                    wx = x - x0
                    wy = y - y0
                    val = (data[y0, x0] * (1 - wx) * (1 - wy) +
                           data[y0, x1] * wx * (1 - wy) +
                           data[y1, x0] * (1 - wx) * wy +
                           data[y1, x1] * wx * wy)
                    profile.append(val)
                    distances.append(t * self.dx)
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = (center_x - dx_line + offset * np.cos(angle_rad + np.pi / 2)) * self.dx + self.extent[0]
            y_start = (center_y - dy_line + offset * np.sin(angle_rad + np.pi / 2)) * self.dx + self.extent[2]
            x_end = (center_x + dx_line + offset * np.cos(angle_rad + np.pi / 2)) * self.dx + self.extent[0]
            y_end = (center_y + dy_line + offset * np.sin(angle_rad + np.pi / 2)) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        return distance, profile, endpoints

    def _normalize_profile_type(self, profile_type):
        normalized = str(profile_type).lower().replace('-', '_')
        mapping = {
            'horizontal': 'horizontal', 'h': 'horizontal', 'x': 'horizontal',
            'vertical': 'vertical', 'v': 'vertical', 'y': 'vertical',
            'diagonal': 'diagonal', 'd': 'diagonal', 'diag': 'diagonal',
            'anti_diagonal': 'anti_diagonal', 'antidiagonal': 'anti_diagonal',
            'anti-diagonal': 'anti_diagonal', 'ad': 'anti_diagonal',
            'custom': 'custom', 'c': 'custom', 'angled': 'custom'
        }
        return mapping.get(normalized, normalized)


class PublicationEnhancer:
    """Advanced plotting enhancements for publication-quality figures"""

    @staticmethod
    def create_custom_colormaps():
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        plasma_enhanced = LinearSegmentedColormap.from_list('plasma_enhanced', [
            (0.0, '#0c0887'), (0.1, '#4b03a1'), (0.3, '#8b0aa5'),
            (0.5, '#b83289'), (0.7, '#db5c68'), (0.9, '#f48849'), (1.0, '#fec325')
        ])
        coolwarm_enhanced = LinearSegmentedColormap.from_list('coolwarm_enhanced', [
            (0.0, '#3a4cc0'), (0.25, '#8abcdd'), (0.5, '#f7f7f7'),
            (0.75, '#f0b7a4'), (1.0, '#b40426')
        ])
        twin_categorical = ListedColormap([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'
        ])
        stress_map = LinearSegmentedColormap.from_list('stress_map', [
            (0.0, '#2c7bb6'), (0.2, '#abd9e9'), (0.4, '#ffffbf'),
            (0.6, '#fdae61'), (0.8, '#d7191c'), (1.0, '#800026')
        ])
        return {
            'plasma_enhanced': plasma_enhanced,
            'coolwarm_enhanced': coolwarm_enhanced,
            'twin_categorical': twin_categorical,
            'stress_map': stress_map
        }

    @staticmethod
    def add_error_shading(ax, x, y_mean, y_std, color='blue', alpha=0.3, label=''):
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        color=color, alpha=alpha, label=label + ' ± std')
        return ax

    @staticmethod
    def add_scale_bar(ax, length_nm, location='lower right', color='black',
                      linewidth=2, fontsize=8):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
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
# ENHANCED SIMULATION DATABASE
# ============================================================================
class SimulationDatabase:
    """Enhanced simulation database for storing and comparing multiple runs"""

    @staticmethod
    @handle_errors
    def generate_id(sim_params):
        param_str = json.dumps({k: v for k, v in sim_params.items()
                                if k not in ['history', 'results', 'geom_viz']},
                               sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    @staticmethod
    @handle_errors
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
            'id': sim_id,
            'params': sim_params,
            'results_history': results_history,
            'geometry_data': geometry_data,
            'metadata': metadata,
            'created_at': metadata.get('created_at', datetime.now().isoformat()),
            'last_modified': datetime.now().isoformat()
        }
        return sim_id

    @staticmethod
    @handle_errors
    def get_simulation(sim_id):
        if 'twin_simulations' in st.session_state and sim_id in st.session_state.twin_simulations:
            sim_data = st.session_state.twin_simulations[sim_id]
            if 'metadata' in sim_data:
                sim_data['metadata'] = MetadataManager.validate_metadata(sim_data['metadata'])
            return sim_data
        return None

    @staticmethod
    @handle_errors
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
                        f"σ={params.get('applied_stress', 0) / 1e6:.0f}MPa | "
                        f"θ={params.get('applied_stress_angle', 0):.0f}° | "
                        f"W={params.get('W', 0):.1f}")
                simulations.append({
                    'id': sim_id, 'name': name, 'params': params,
                    'metadata': metadata,
                    'results': sim_data['results_history'][-1] if sim_data['results_history'] else None
                })
            except Exception:
                continue
        return simulations


# ============================================================================
# FILENAME HELPERS
# ============================================================================
@handle_errors
def sanitize_token(text: str) -> str:
    try:
        s = str(text)
        for ch in [" ", "{", "}", "/", "\\", ",", ";", "(", ")", "[", "]", "°"]:
            s = s.replace(ch, "")
        return s
    except Exception:
        return "unknown"


@handle_errors
def fmt_num_trim(x, ndigits=3):
    try:
        s = f"{x:.{ndigits}f}"
        s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s
    except Exception:
        return "0.0"


@handle_errors
def build_sim_name(params: dict, sim_id: str = None) -> str:
    try:
        geom_type = params.get("geometry_type", "standard")
        if geom_type == "defect":
            defect = params.get("defect_type", "dislocation")
            geom_token = f"twin_grain_with_{defect}"
        else:
            geom_token = "standard_twin_grain"
        twin_spacing = fmt_num_trim(params.get("twin_spacing", 20.0), ndigits=1)
        W = fmt_num_trim(params.get("W", 2.0), ndigits=1)
        stress_mpa = fmt_num_trim(params.get("applied_stress", 300e6) / 1e6, ndigits=0)
        theta = fmt_num_trim(params.get("applied_stress_angle", 0.0), ndigits=0)
        name = f"twin_lambda_{twin_spacing}_W_{W}_stress_{stress_mpa}MPa_theta_{theta}_{geom_token}"
        if sim_id:
            name = f"{name}_{sim_id}"
        return name
    except Exception:
        return f"twin_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# MATERIAL PROPERTIES
# ============================================================================
class MaterialProperties:
    """Material properties database with validation and multiple materials."""

    @staticmethod
    def get_cu_properties():
        return {
            'elastic': {'C11': 168.4e9, 'C12': 121.4e9, 'C44': 75.4e9,
                        'source': 'Phys. Rev. B 73, 064112 (2006)'},
            'twinning': {
                'gamma_tw': 1 / np.sqrt(2),
                'n_111': np.array([1, 1, 1]) / np.sqrt(3),
                'a_112': np.array([1, 1, -2]) / np.sqrt(6),
                'n_2d': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
                'a_2d': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
            },
            'plasticity': {'mu': 48e9, 'nu': 0.34, 'b': 0.256e-9,
                           'sigma0': 50e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
        }

    @staticmethod
    def get_al_properties():
        return {
            'elastic': {'C11': 106.8e9, 'C12': 60.4e9, 'C44': 28.3e9,
                        'source': 'J. Appl. Phys. 88, 3287 (2000)'},
            'twinning': {
                'gamma_tw': 1 / np.sqrt(2),
                'n_111': np.array([1, 1, 1]) / np.sqrt(3),
                'a_112': np.array([1, 1, -2]) / np.sqrt(6),
                'n_2d': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
                'a_2d': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
            },
            'plasticity': {'mu': 26e9, 'nu': 0.33, 'b': 0.286e-9,
                           'sigma0': 30e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
        }

    @staticmethod
    def get_ni_properties():
        return {
            'elastic': {'C11': 246.5e9, 'C12': 147.3e9, 'C44': 124.7e9,
                        'source': 'Phys. Rev. B 94, 014110 (2016)'},
            'twinning': {
                'gamma_tw': 1 / np.sqrt(2),
                'n_111': np.array([1, 1, 1]) / np.sqrt(3),
                'a_112': np.array([1, 1, -2]) / np.sqrt(6),
                'n_2d': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
                'a_2d': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
            },
            'plasticity': {'mu': 80e9, 'nu': 0.31, 'b': 0.249e-9,
                           'sigma0': 70e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e13}
        }

    @staticmethod
    @handle_errors
    def get_material(material_name='Cu'):
        if material_name == 'Cu':
            return MaterialProperties.get_cu_properties()
        elif material_name == 'Al':
            return MaterialProperties.get_al_properties()
        elif material_name == 'Ni':
            return MaterialProperties.get_ni_properties()
        else:
            return MaterialProperties.get_cu_properties()

    @staticmethod
    @handle_errors
    def validate_parameters(params):
        errors = []
        warnings_list = []
        if params.get('dt', 0) <= 0:
            errors.append("Time step dt must be positive")
        if params.get('dx', 0) <= 0:
            errors.append("Grid spacing dx must be positive")
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
    """Create and visualize initial geometric conditions."""

    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.x = np.linspace(-N * dx / 2, N * dx / 2, N)
        self.y = np.linspace(-N * dx / 2, N * dx / 2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.extent = [-N * dx / 2, N * dx / 2, -N * dx / 2, N * dx / 2]

    @handle_errors
    def create_twin_grain_geometry(self, twin_spacing=20.0, grain_boundary_pos=0.0,
                                   gb_width=3.0, buffer_width=5.0,
                                   left_buffer_width=5.0,
                                   gb_profile='plane', gb_curvature=0.0):
        eta1 = np.zeros((self.N, self.N))
        eta2 = np.zeros((self.N, self.N))
        phi = np.zeros((self.N, self.N))

        def gb_x_func(y):
            if gb_profile == 'plane':
                return grain_boundary_pos
            elif gb_profile == 'concave':
                sigma = (self.N * self.dx) / 4.0
                deviation = -gb_curvature * np.exp(-y**2 / (2 * sigma**2))
                return grain_boundary_pos + deviation
            elif gb_profile == 'convex':
                sigma = (self.N * self.dx) / 4.0
                deviation = gb_curvature * np.exp(-y**2 / (2 * sigma**2))
                return grain_boundary_pos + deviation
            else:
                return grain_boundary_pos

        left_edge = self.extent[0]

        for i in range(self.N):
            for j in range(self.N):
                x_val = self.X[i, j]
                y_val = self.Y[i, j]
                gb_x = gb_x_func(y_val)
                dist_from_gb = x_val - gb_x
                if dist_from_gb < -gb_width:
                    eta1[i, j] = 1.0
                    eta2[i, j] = 0.0
                elif dist_from_gb > gb_width:
                    eta1[i, j] = 0.0
                    eta2[i, j] = 1.0
                else:
                    transition = 0.5 * (1 - np.tanh(dist_from_gb / (gb_width / 3)))
                    eta1[i, j] = transition
                    eta2[i, j] = 1 - transition

        for i in range(self.N):
            for j in range(self.N):
                if eta1[i, j] > 0.5:
                    x_val = self.X[i, j]
                    y_val = self.Y[i, j]
                    gb_x = gb_x_func(y_val)
                    dist_from_gb = abs(x_val - gb_x)
                    dist_from_left_edge = abs(x_val - left_edge)
                    if dist_from_gb > buffer_width and dist_from_left_edge > left_buffer_width:
                        phase = 2 * np.pi * self.Y[i, j] / twin_spacing
                        phi[i, j] = np.tanh(np.sin(phase) * 3.0)
                    else:
                        phi[i, j] = 1.0
        return phi, eta1, eta2

    @handle_errors
    def create_defect_geometry(self, twin_spacing=20.0, defect_type='dislocation',
                               defect_pos=(0, 0), defect_radius=10.0,
                               grain_boundary_pos=0.0, gb_width=3.0,
                               buffer_width=5.0, left_buffer_width=5.0,
                               gb_profile='plane', gb_curvature=0.0):
        phi, eta1, eta2 = self.create_twin_grain_geometry(
            twin_spacing, grain_boundary_pos, gb_width,
            buffer_width, left_buffer_width, gb_profile, gb_curvature
        )
        if defect_type == 'dislocation':
            center_x, center_y = defect_pos
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i, j] - center_x)**2 +
                                   (self.Y[i, j] - center_y)**2)
                    if dist < defect_radius:
                        phase_shift = np.exp(-dist**2 / (defect_radius**2)) * np.pi
                        phase = 2 * np.pi * self.Y[i, j] / twin_spacing + phase_shift
                        phi_candidate = np.tanh(np.sin(phase) * 3.0)
                        if eta1[i, j] > 0.5:
                            phi[i, j] = phi_candidate
        elif defect_type == 'void':
            center_x, center_y = defect_pos
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i, j] - center_x)**2 +
                                   (self.Y[i, j] - center_y)**2)
                    if dist < defect_radius:
                        eta1[i, j] = 0.0
                        eta2[i, j] = 0.0
                        phi[i, j] = 0.0
        return phi, eta1, eta2


# ============================================================================
# ENHANCED SPECTRAL SOLVER
# ============================================================================
class EnhancedSpectralSolver:
    """Spectral elasticity solver for mechanical equilibrium.
       Supports arbitrary loading direction via full stress tensor components."""

    def __init__(self, N, dx, elastic_params, kx=None, ky=None, k2=None):
        self.N = N
        self.dx = dx
        if kx is None or ky is None or k2 is None:
            self.kx, self.ky, self.k2 = make_k_vectors(N, dx)
        else:
            self.kx, self.ky, self.k2 = kx, ky, k2

        C11 = elastic_params['C11']
        C12 = elastic_params['C12']
        C44 = elastic_params['C44']
        C11_2d = (C11 + C12 + 2 * C44) / 2
        C12_2d = (C11 + C12 - 2 * C44) / 2
        lambda_2d = C12_2d
        mu_2d = (C11_2d - C12_2d) / 2
        self.C11_2d = C11_2d
        self.C12_2d = C12_2d
        self.C44_2d = C44

        denom = mu_2d * (lambda_2d + 2 * mu_2d) * self.k2 + 1e-15
        self.G11 = (mu_2d * (self.kx**2 + 2 * self.ky**2) + lambda_2d * self.ky**2) / denom
        self.G12 = -mu_2d * self.kx * self.ky / denom
        self.G22 = (mu_2d * (self.ky**2 + 2 * self.kx**2) + lambda_2d * self.kx**2) / denom

    @handle_errors
    def solve(self, eigenstrain_xx, eigenstrain_yy, eigenstrain_xy,
              applied_stress_xx=0, applied_stress_yy=0, applied_stress_xy=0):
        assert eigenstrain_xx.shape == (self.N, self.N)
        eps_xx_hat = fft2(eigenstrain_xx)
        eps_yy_hat = fft2(eigenstrain_yy)
        eps_xy_hat = fft2(eigenstrain_xy)

        ux_hat = 1j * (self.G11 * self.kx * eps_xx_hat +
                       self.G12 * self.ky * eps_xx_hat +
                       self.G12 * self.kx * eps_yy_hat +
                       self.G22 * self.ky * eps_yy_hat)
        uy_hat = 1j * (self.G12 * self.kx * eps_xx_hat +
                       self.G22 * self.ky * eps_xx_hat +
                       self.G11 * self.kx * eps_yy_hat +
                       self.G12 * self.ky * eps_yy_hat)

        eps_xx_el = np.real(ifft2(1j * self.kx * ux_hat))
        eps_yy_el = np.real(ifft2(1j * self.ky * uy_hat))
        eps_xy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_hat + self.ky * ux_hat)))

        eps_xx = eps_xx_el + eigenstrain_xx
        eps_yy = eps_yy_el + eigenstrain_yy
        eps_xy = eps_xy_el + eigenstrain_xy

        sxx = applied_stress_xx + self.C11_2d * eps_xx + self.C12_2d * eps_yy
        syy = applied_stress_yy + self.C12_2d * eps_xx + self.C11_2d * eps_yy
        sxy = applied_stress_xy + 2 * self.C44_2d * eps_xy

        sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + syy**2 + sxx**2 + 6 * sxy**2))
        sigma_eq = np.clip(sigma_eq, 0, 5e9)
        sigma_h = (sxx + syy) / 2

        return sigma_eq, sxx, syy, sxy, sigma_h, eps_xx, eps_yy, eps_xy


# ============================================================================
# ENHANCED VISUALIZATION SYSTEM
# ============================================================================
class EnhancedTwinVisualizer:
    """Comprehensive visualization system for nanotwinned simulations."""

    def __init__(self, N, dx, dt=1e-4):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.extent = [-N * dx / 2, N * dx / 2, -N * dx / 2, N * dx / 2]
        self.line_profiler = EnhancedLineProfiler(N, dx)
        self.COLORMAPS = COLORMAPS.copy()
        custom = PublicationEnhancer.create_custom_colormaps()
        self.COLORMAPS.update(custom)

    @handle_errors
    def get_colormap(self, cmap_name):
        """Get colormap by name with fallback"""
        if cmap_name in self.COLORMAPS:
            entry = self.COLORMAPS[cmap_name]
            if isinstance(entry, str):
                return plt.get_cmap(entry)
            else:
                return entry
        return plt.get_cmap('viridis')

    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        if style_params is None:
            style_params = {}
        defaults = {
            'title_font_size': 10, 'label_font_size': 8,
            'scalebar_color': 'black', 'scalebar_fontsize': 8,
            'phi_cmap': 'RdBu_r', 'sigma_eq_cmap': 'hot',
            'sigma_h_cmap': 'RdBu', 'h_cmap': 'plasma',
            'eps_p_mag_cmap': 'YlOrRd', 'sigma_y_cmap': 'viridis',
            'eta1_cmap': 'Reds',
        }
        for k, v in defaults.items():
            style_params.setdefault(k, v)
        title_font_size = float(style_params['title_font_size'])
        label_font_size = float(style_params['label_font_size'])
        scalebar_fontsize = float(style_params['scalebar_fontsize'])

        fields_to_plot = [
            ('phi', 'Twin Order Parameter φ', style_params['phi_cmap'], [-1.2, 1.2]),
            ('eta1', 'Grain η₁', style_params['eta1_cmap'], [0, 1]),
            ('sigma_eq', 'Von Mises Stress (GPa)', style_params['sigma_eq_cmap'], None),
            ('sigma_h', 'Hydrostatic Stress (GPa)', style_params['sigma_h_cmap'], None),
            ('h', 'Twin Spacing (nm)', style_params['h_cmap'], [0, 30]),
            ('eps_p_mag', 'Plastic Strain', style_params['eps_p_mag_cmap'], None),
            ('sigma_y', 'Yield Stress (MPa)', style_params['sigma_y_cmap'], None),
        ]
        available_fields = [(fname, title, cmap, vrange)
                            for fname, title, cmap, vrange in fields_to_plot
                            if fname in results_dict]
        n_fields = len(available_fields)
        if n_fields == 0:
            return None
        cols = min(3, n_fields)
        rows = (n_fields + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        for idx, (field_name, title, default_cmap, vrange) in enumerate(available_fields):
            ax = axes[idx]
            data = results_dict[field_name]
            if field_name in ['sigma_eq', 'sigma_h']:
                data = data / 1e9
            elif field_name == 'sigma_y':
                data = data / 1e6
            cmap_name = style_params.get(f'{field_name}_cmap', default_cmap)
            cmap = self.get_colormap(cmap_name)
            if vrange is not None:
                vmin, vmax = vrange
            else:
                vmin = np.percentile(data, 2)
                vmax = np.percentile(data, 98)
                if field_name in ['sigma_h']:
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
            im = ax.imshow(data, extent=self.extent, cmap=cmap,
                           vmin=vmin, vmax=vmax, origin='lower', aspect='equal',
                           interpolation='bilinear')
            if field_name == 'phi':
                ax.contour(np.linspace(self.extent[0], self.extent[1], self.N),
                           np.linspace(self.extent[2], self.extent[3], self.N),
                           data, levels=[0], colors='white', linewidths=1, alpha=0.8)
            ax.set_title(title, fontsize=title_font_size)
            ax.set_xlabel('x (nm)', fontsize=label_font_size)
            ax.set_ylabel('y (nm)', fontsize=label_font_size)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if field_name == 'sigma_eq':
                cbar.set_label('Stress (GPa)')
            elif field_name == 'sigma_h':
                cbar.set_label('Stress (GPa)')
            elif field_name == 'sigma_y':
                cbar.set_label('Stress (MPa)')
            elif field_name == 'h':
                cbar.set_label('Spacing (nm)')
            if field_name in ['phi', 'eta1', 'sigma_eq', 'sigma_h']:
                PublicationEnhancer.add_scale_bar(
                    ax, 10.0, 'lower right',
                    color=style_params['scalebar_color'],
                    fontsize=scalebar_fontsize)
        for idx in range(n_fields, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        return fig

    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0):
        if field_name not in results_dict:
            return None
        data = results_dict[field_name].copy()
        title = field_name
        unit = ""
        if field_name in ['sigma_eq', 'sigma_h']:
            data = data / 1e9
            unit = " (GPa)"
        elif field_name == 'sigma_y':
            data = data / 1e6
            unit = " (MPa)"
        elif field_name == 'h':
            unit = " (nm)"
        if field_name == 'phi':
            colorscale = 'RdBu'
            zmid = 0
        elif field_name == 'eta1':
            colorscale = 'Reds'
            zmid = None
        elif field_name in ['sigma_eq', 'sigma_h']:
            colorscale = 'Viridis' if field_name == 'sigma_eq' else 'RdBu'
            zmid = 0 if field_name == 'sigma_h' else None
        else:
            colorscale = 'Plasma'
            zmid = None
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=data,
            x=np.linspace(self.extent[0], self.extent[1], self.N),
            y=np.linspace(self.extent[2], self.extent[3], self.N),
            colorscale=colorscale, zmid=zmid,
            colorbar=dict(title=f"{field_name}{unit}"),
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.3f}<extra></extra>'))
        if field_name != 'phi' and 'phi' in results_dict:
            phi_data = results_dict['phi']
            fig.add_trace(go.Contour(
                z=phi_data,
                x=np.linspace(self.extent[0], self.extent[1], self.N),
                y=np.linspace(self.extent[2], self.extent[3], self.N),
                contours=dict(start=0, end=0, size=0, coloring='none', showlabels=False),
                line=dict(color='white', width=2),
                showscale=False, hoverinfo='skip'))
        fig.update_layout(
            title=f"{title} (Frame {frame_idx})",
            xaxis_title="x (nm)", yaxis_title="y (nm)",
            width=600, height=500, template="plotly_white")
        return fig

    @handle_errors
    def create_plotly_line_profiles(self, results_dict, field_name, profile_types,
                                    position_ratio=0.5):
        fig = go.Figure()
        for ptype in profile_types:
            distance, profile, _ = self.line_profiler.extract_profile(
                results_dict[field_name], ptype, position_ratio)
            if field_name in ['sigma_eq', 'sigma_h']:
                profile = profile / 1e9
                ylabel = 'Stress (GPa)'
            elif field_name == 'sigma_y':
                profile = profile / 1e6
                ylabel = 'Stress (MPa)'
            else:
                ylabel = field_name
            fig.add_trace(go.Scatter(x=distance, y=profile, mode='lines',
                                     name=ptype.replace('_', ' ').title()))
        fig.update_layout(
            title=f"{field_name} Line Profiles",
            xaxis_title="Position (nm)", yaxis_title=ylabel,
            hovermode='x unified', template="plotly_white")
        return fig

    @handle_errors
    def create_plotly_3d_surface(self, results_dict, field_name, frame_idx=0):
        if field_name not in results_dict:
            return None
        data = results_dict[field_name].copy()
        title = field_name
        unit = ""
        if field_name in ['sigma_eq', 'sigma_h']:
            data = data / 1e9
            unit = " (GPa)"
        elif field_name == 'sigma_y':
            data = data / 1e6
            unit = " (MPa)"
        elif field_name == 'h':
            unit = " (nm)"
        x = np.linspace(self.extent[0], self.extent[1], self.N)
        y = np.linspace(self.extent[2], self.extent[3], self.N)
        X, Y = np.meshgrid(x, y)
        if field_name == 'phi':
            colorscale = 'RdBu'
            cmin, cmax = -1.2, 1.2
        elif field_name == 'eta1':
            colorscale = 'Reds'
            cmin, cmax = 0, 1
        elif field_name == 'sigma_h':
            colorscale = 'RdBu'
            cmin, cmax = -np.max(np.abs(data)), np.max(np.abs(data))
        else:
            colorscale = 'Viridis'
            cmin, cmax = None, None
        fig = go.Figure(data=[go.Surface(
            z=data, x=X, y=Y, colorscale=colorscale, cmin=cmin, cmax=cmax,
            colorbar=dict(title=f"{field_name}{unit}"),
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>z: %{z:.3f}<extra></extra>')])
        fig.update_layout(
            title=f"3D Surface: {title} (Frame {frame_idx})",
            scene=dict(xaxis_title='x (nm)', yaxis_title='y (nm)',
                       zaxis_title=f'{title}{unit}',
                       camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
            width=700, height=600, template="plotly_white")
        return fig

    @handle_errors
    def create_animation(self, history, field_name, output_format='gif',
                         fps=5, dpi=150):
        if not history:
            return None
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        first_data = history[0][field_name].copy()
        if field_name in ['sigma_eq', 'sigma_h']:
            first_data = first_data / 1e9
        elif field_name == 'sigma_y':
            first_data = first_data / 1e6
        vmin = np.percentile(first_data, 2)
        vmax = np.percentile(first_data, 98)
        if field_name == 'phi':
            vmin, vmax = -1.2, 1.2
        elif field_name == 'eta1':
            vmin, vmax = 0, 1
        elif field_name == 'sigma_h':
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        im = ax.imshow(first_data, extent=self.extent,
                       cmap=self.get_colormap('viridis'),
                       vmin=vmin, vmax=vmax, origin='lower',
                       interpolation='bilinear')
        cbar = plt.colorbar(im, ax=ax)
        if field_name in ['sigma_eq', 'sigma_h']:
            cbar.set_label('Stress (GPa)')
        elif field_name == 'sigma_y':
            cbar.set_label('Stress (MPa)')
        elif field_name == 'h':
            cbar.set_label('Spacing (nm)')
        else:
            cbar.set_label(field_name)
        title = ax.set_title(f"{field_name} - t = 0.000 ns")
        dt_local = self.dt

        def update_frame(frame_idx):
            data = history[frame_idx][field_name].copy()
            if field_name in ['sigma_eq', 'sigma_h']:
                data = data / 1e9
            elif field_name == 'sigma_y':
                data = data / 1e6
            im.set_array(data)
            time_ns = frame_idx * dt_local * 1e3
            title.set_text(f"{field_name} - t = {time_ns:.3f} ns")
            return [im, title]

        ani = animation.FuncAnimation(fig, update_frame, frames=len(history),
                                      interval=1000 / fps, blit=True)
        buffer = BytesIO()
        if output_format == 'gif':
            ani.save(buffer, writer='pillow', fps=fps, dpi=dpi)
        else:
            ani.save(buffer, writer='ffmpeg', fps=fps, dpi=dpi)
        plt.close(fig)
        buffer.seek(0)
        return buffer


# ============================================================================
# MAIN SOLVER (PURE FFT SPECTRAL METHOD)
# ============================================================================
class NanotwinnedCuSolver:
    """Main solver with pure-FFT semi-implicit spectral time integration.
       Replaces all FDM/Numba kernels by exact spectral derivatives."""

    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']

        material_name = params.get('material', 'Cu')
        self.mat_props = MaterialProperties.get_material(material_name)
        self.params['material'] = material_name

        # Apply LLM-recommended plasticity overrides if present
        apply_plasticity_overrides(self)

        errors, warnings_list = MaterialProperties.validate_parameters(params)
        if errors:
            raise ValueError(f"Parameter validation failed: {', '.join(errors)}")
        if warnings_list:
            st.warning(f"Parameter warnings: {', '.join(warnings_list)}")

        self.geom_viz = InitialGeometryVisualizer(self.N, self.dx)

        # ---- Spectral wavenumber grids (shared by elasticity + phase-field) ----
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

        self.history = {
            'phi_norm': [], 'energy': [], 'max_stress': [],
            'plastic_work': [], 'avg_stress': [], 'twin_spacing_avg': []
        }

        # ---- Implicit baseline for the semi-implicit Fourier update ----
        # Raised to the maximal anisotropic upper bound so the explicit
        # correction term is always non-positive at CTBs.
        kappa0 = float(params.get('kappa0', 1.0))
        gamma_aniso = float(params.get('gamma_aniso', 0.7))
        L_CTB = float(params.get('L_CTB', 0.05))
        L_ITB = float(params.get('L_ITB', 5.0))
        # Use the *upper bound* (L_ITB × κ₀ × (1+γ_aniso)) as the implicit
        # stabiliser, since L_phi·κ_phi can reach that at CTBs.
        self.implicit_diff_phi = L_ITB * kappa0 * (1.0 + gamma_aniso)
        kappa_eta = float(params.get('kappa_eta', 2.0))
        L_eta = float(params.get('L_eta', 1.0))
        self.implicit_diff_eta = L_eta * kappa_eta

        self.confine_twin = params.get('confine_twin', True)

    @handle_errors
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
            defect_type = self.params.get('defect_type', 'dislocation')
            defect_pos = self.params.get('defect_pos', (0, 0))
            defect_radius = self.params.get('defect_radius', 10.0)
            return self.geom_viz.create_defect_geometry(
                twin_spacing, defect_type, defect_pos, defect_radius,
                gb_pos, gb_width, buffer_width, left_buffer_width,
                gb_profile, gb_curvature)
        else:
            return self.geom_viz.create_twin_grain_geometry(
                twin_spacing, gb_pos, gb_width,
                buffer_width, left_buffer_width, gb_profile, gb_curvature)

    @handle_errors
    def compute_local_energy_derivatives(self):
        W = self.params['W']
        A = self.params['A']
        B = self.params['B']
        df_dphi = 4 * W * self.phi * (self.phi**2 - 1) * self.eta1**2
        df_deta1 = (2 * A * self.eta1 * (1 - self.eta1) * (1 - 2 * self.eta1) +
                    2 * B * self.eta1 * self.eta2**2 +
                    2 * W * (self.phi**2 - 1)**2 * self.eta1)
        df_deta2 = (2 * A * self.eta2 * (1 - self.eta2) * (1 - 2 * self.eta2) +
                    2 * B * self.eta2 * self.eta1**2)
        return df_dphi, df_deta1, df_deta2

    @handle_errors
    def compute_elastic_driving_force(self, sxx, syy, sxy):
        try:
            gamma_tw = self.mat_props['twinning']['gamma_tw']
            n = self.mat_props['twinning']['n_2d']
            a = self.mat_props['twinning']['a_2d']
            dh_dphi = 0.25 * (3 * self.phi**2 - 2 * self.phi - 1)
            nx, ny = n[0], n[1]
            ax, ay = a[0], a[1]
            deps_xx_dphi = gamma_tw * nx * ax * dh_dphi * self.eta1
            deps_yy_dphi = gamma_tw * ny * ay * dh_dphi * self.eta1
            deps_xy_dphi = 0.5 * gamma_tw * (nx * ay + ny * ax) * dh_dphi * self.eta1
            df_el_dphi = -(sxx * deps_xx_dphi + syy * deps_yy_dphi +
                           2 * sxy * deps_xy_dphi)
            return df_el_dphi
        except Exception as e:
            st.error(f"Error computing elastic driving force: {e}")
            return np.zeros_like(self.phi)

    @handle_errors
    def evolve_twin_field(self, sxx, syy, sxy, eps_p_mag):
        """Semi-implicit Fourier spectral update of φ."""
        kappa0 = float(self.params['kappa0'])
        gamma_aniso = float(self.params['gamma_aniso'])
        L_CTB = float(self.params.get('L_CTB', 0.05))
        L_ITB = float(self.params.get('L_ITB', 5.0))
        n_mob = int(self.params.get('n_mob', 4))
        zeta = float(self.params.get('zeta', 0.3))

        n_twin = self.mat_props['twinning']['n_2d']
        nx = float(n_twin[0])
        ny = float(n_twin[1])

        phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
        lap_phi = spectral_laplacian(self.phi, self.k2)

        kappa_phi, L_phi = compute_anisotropic_properties(
            phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso,
            L_CTB, L_ITB, n_mob)

        df_loc_dphi, _, _ = self.compute_local_energy_derivatives()
        df_el_dphi = self.compute_elastic_driving_force(sxx, syy, sxy)
        diss_p = zeta * eps_p_mag * self.phi

        NLF = df_loc_dphi + df_el_dphi + diss_p

        correction = (L_phi * kappa_phi - self.implicit_diff_phi) * lap_phi
        R = -L_phi * NLF + correction

        if self.confine_twin:
            R = R * self.eta1**2

        phi_hat = fft2(self.phi)
        R_hat = fft2(R)
        denom = 1.0 + self.dt * self.implicit_diff_phi * self.k2
        phi_hat_new = (phi_hat + self.dt * R_hat) / denom
        phi_new = np.real(ifft2(phi_hat_new))
        phi_new = np.clip(phi_new, -1.1, 1.1)
        return phi_new

    @handle_errors
    def evolve_grain_fields(self):
        """Semi-implicit spectral update of η₁ and η₂."""
        kappa_eta = float(self.params['kappa_eta'])
        L_eta = float(self.params.get('L_eta', 1.0))
        _, df_deta1, df_deta2 = self.compute_local_energy_derivatives()

        R_eta1 = -L_eta * df_deta1
        R_eta2 = -L_eta * df_deta2

        eta1_hat = fft2(self.eta1)
        eta2_hat = fft2(self.eta2)
        R1_hat = fft2(R_eta1)
        R2_hat = fft2(R_eta2)
        denom = 1.0 + self.dt * self.implicit_diff_eta * self.k2
        eta1_hat_new = (eta1_hat + self.dt * R1_hat) / denom
        eta2_hat_new = (eta2_hat + self.dt * R2_hat) / denom
        eta1_new = np.real(ifft2(eta1_hat_new))
        eta2_new = np.real(ifft2(eta2_hat_new))

        eta1_new = np.clip(eta1_new, 0, 1)
        eta2_new = np.clip(eta2_new, 0, 1)
        norm = np.sqrt(eta1_new**2 + eta2_new**2 + 1e-12)
        mask = norm > 1
        eta1_new[mask] = eta1_new[mask] / norm[mask]
        eta2_new[mask] = eta2_new[mask] / norm[mask]
        return eta1_new, eta2_new

    @handle_errors
    def compute_plastic_strain(self, sigma_eq, sigma_y):
        try:
            plastic_params = self.mat_props['plasticity']
            gamma0_dot = plastic_params['gamma0_dot']
            m = int(plastic_params['m'])
            eps_p_xx_new, eps_p_yy_new, eps_p_xy_new = update_plastic_strain(
                sigma_eq, sigma_y,
                self.eps_p_xx, self.eps_p_yy, self.eps_p_xy,
                gamma0_dot, m, self.dt)
            self.eps_p_xx = eps_p_xx_new
            self.eps_p_yy = eps_p_yy_new
            self.eps_p_xy = eps_p_xy_new
            eps_p_mag = np.sqrt(
                2 / 3 * (self.eps_p_xx**2 + self.eps_p_yy**2 +
                         2 * self.eps_p_xy**2 + 1e-15))
            eps_p_mag = np.clip(eps_p_mag, 0, 0.5)
            if np.max(eps_p_mag) > 0.1:
                st.warning(f"⚠️ Large plastic strain detected: {np.max(eps_p_mag):.3f}")
            return eps_p_mag
        except Exception as e:
            st.error(f"Error computing plastic strain: {e}")
            return np.zeros_like(sigma_eq)

    @handle_errors
    def compute_total_energy(self):
        try:
            W = self.params['W']
            A = self.params['A']
            B = self.params['B']
            f_loc = (W * (self.phi**2 - 1)**2 * self.eta1**2 +
                     A * (self.eta1**2 * (1 - self.eta1)**2 +
                          self.eta2**2 * (1 - self.eta2)**2) +
                     B * self.eta1**2 * self.eta2**2)
            phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
            grad_phi_sq = phi_gx**2 + phi_gy**2
            eta1_gx, eta1_gy = spectral_gradients(self.eta1, self.kx, self.ky)
            eta2_gx, eta2_gy = spectral_gradients(self.eta2, self.kx, self.ky)
            grad_eta1_sq = eta1_gx**2 + eta1_gy**2
            grad_eta2_sq = eta2_gx**2 + eta2_gy**2
            kappa0 = self.params['kappa0']
            kappa_eta = self.params['kappa_eta']
            f_grad = (0.5 * kappa0 * grad_phi_sq +
                      0.5 * kappa_eta * (grad_eta1_sq + grad_eta2_sq))
            energy_density = f_loc + f_grad
            total_energy = np.sum(energy_density) * (self.dx**2)
            return total_energy
        except Exception as e:
            st.warning(f"Error computing energy: {e}")
            return 0.0

    @handle_errors
    def step(self):
        """Perform one time step of the simulation (pure FFT)."""
        try:
            sigma_mag = self.params.get('applied_stress', 0.0)
            theta_deg = self.params.get('applied_stress_angle', 0.0)
            theta = np.deg2rad(theta_deg)
            applied_xx = sigma_mag * np.cos(theta)**2
            applied_yy = sigma_mag * np.sin(theta)**2
            applied_xy = sigma_mag * np.sin(theta) * np.cos(theta)

            gamma_tw = self.mat_props['twinning']['gamma_tw']
            n = self.mat_props['twinning']['n_2d']
            a = self.mat_props['twinning']['a_2d']

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
            h = compute_twin_spacing_from_gradient(phi_gx, phi_gy)

            plastic_params = self.mat_props['plasticity']
            sigma_y = compute_yield_stress(
                h, plastic_params['sigma0'], plastic_params['mu'],
                plastic_params['b'], plastic_params['nu'])

            eps_p_mag = self.compute_plastic_strain(sigma_eq, sigma_y)

            self.phi = self.evolve_twin_field(sxx, syy, sxy, eps_p_mag)
            self.eta1, self.eta2 = self.evolve_grain_fields()

            phi_norm = np.linalg.norm(self.phi)
            total_energy = self.compute_total_energy()
            max_stress = np.max(sigma_eq)
            avg_stress = np.mean(sigma_eq)
            valid_h = h[(h > 5) & (h < 50)]
            avg_spacing = np.mean(valid_h) if valid_h.size > 0 else 0.0
            plastic_work = np.sum(eps_p_mag) * (self.dx**2)

            self.history['phi_norm'].append(phi_norm)
            self.history['energy'].append(total_energy)
            self.history['max_stress'].append(max_stress)
            self.history['avg_stress'].append(avg_stress)
            self.history['plastic_work'].append(plastic_work)
            self.history['twin_spacing_avg'].append(avg_spacing)

            results = {
                'phi': self.phi.copy(),
                'eta1': self.eta1.copy(),
                'eta2': self.eta2.copy(),
                'sigma_eq': sigma_eq.copy(),
                'sigma_h': sigma_h.copy(),
                'sigma_xx': sxx.copy(),
                'sigma_yy': syy.copy(),
                'sigma_xy': sxy.copy(),
                'h': h.copy(),
                'sigma_y': sigma_y.copy(),
                'eps_p_mag': eps_p_mag.copy(),
                'eps_xx': eps_xx.copy(),
                'eps_yy': eps_yy.copy(),
                'eps_xy': eps_xy.copy(),
                'convergence': {
                    'phi_norm': phi_norm,
                    'energy': total_energy,
                    'max_stress': max_stress,
                    'avg_stress': avg_stress,
                    'plastic_work': plastic_work,
                    'avg_spacing': avg_spacing
                }
            }
            return results
        except Exception as e:
            st.error(f"Error in simulation step: {e}")
            zeros = np.zeros((self.N, self.N))
            return {
                'phi': zeros, 'eta1': zeros, 'eta2': zeros,
                'sigma_eq': zeros, 'sigma_h': zeros,
                'sigma_xx': zeros, 'sigma_yy': zeros, 'sigma_xy': zeros,
                'h': zeros, 'sigma_y': zeros, 'eps_p_mag': zeros,
                'eps_xx': zeros, 'eps_yy': zeros, 'eps_xy': zeros,
                'convergence': {k: 0 for k in
                    ['phi_norm', 'energy', 'max_stress',
                     'avg_stress', 'plastic_work', 'avg_spacing']}
            }


# ============================================================================
# MONITORING
# ============================================================================
class SimulationMonitor:
    @staticmethod
    @handle_errors
    def create_convergence_plots(history, timesteps):
        history_length = len(history['phi_norm'])
        if len(timesteps) >= history_length:
            plot_timesteps = timesteps[:history_length]
        else:
            plot_timesteps = np.linspace(0, timesteps[-1] if len(timesteps) else 1.0,
                                         history_length)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].plot(plot_timesteps, history['phi_norm'], 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Time (ns)')
        axes[0, 0].set_ylabel('||φ||')
        axes[0, 0].set_title('Twin Order Parameter Norm')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(plot_timesteps, history['energy'], 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Time (ns)')
        axes[0, 1].set_ylabel('Total Energy (J)')
        axes[0, 1].set_title('System Energy Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 2].plot(plot_timesteps, np.array(history['max_stress']) / 1e9,
                        'g-', linewidth=2, alpha=0.8, label='Max')
        axes[0, 2].plot(plot_timesteps, np.array(history['avg_stress']) / 1e9,
                        'g--', linewidth=1.5, alpha=0.6, label='Avg')
        axes[0, 2].set_xlabel('Time (ns)')
        axes[0, 2].set_ylabel('Stress (GPa)')
        axes[0, 2].set_title('Stress Evolution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[1, 0].plot(plot_timesteps, history['plastic_work'], 'm-',
                        linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Time (ns)')
        axes[1, 0].set_ylabel('Plastic Work (J)')
        axes[1, 0].set_title('Plastic Work Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(plot_timesteps, history['twin_spacing_avg'], 'c-',
                        linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Time (ns)')
        axes[1, 1].set_ylabel('Avg Spacing (nm)')
        axes[1, 1].set_title('Average Twin Spacing')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 2].text(0.5, 0.5, 'Plastic strain history not saved\n(use per-step analysis)',
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Plastic Strain Evolution')
        plt.tight_layout()
        return fig


# ============================================================================
# DATA EXPORTER
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DataExporter:
    @staticmethod
    @handle_errors
    def export_pkl(sim_data, params, history, sim_name):
        buffer = BytesIO()
        data = {'params': params, 'history': history,
                'metadata': sim_data.get('metadata', {}), 'sim_name': sim_name}
        pickle.dump(data, buffer)
        buffer.seek(0)
        return buffer, f"{sim_name}.pkl"

    @staticmethod
    @handle_errors
    def export_pt(sim_data, params, history, sim_name):
        buffer = BytesIO()

        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return torch.tensor(x)

        tensor_data = {'params': params, 'metadata': sim_data.get('metadata', {}),
                       'history': []}
        for frame in history:
            frame_tensor = {
                'phi': to_tensor(frame['phi']),
                'eta1': to_tensor(frame['eta1']),
                'eta2': to_tensor(frame['eta2']),
                'sigma_eq': to_tensor(frame['sigma_eq']),
                'sigma_h': to_tensor(frame['sigma_h']),
                'h': to_tensor(frame['h']),
                'eps_p_mag': to_tensor(frame['eps_p_mag'])
            }
            tensor_data['history'].append(frame_tensor)
        torch.save(tensor_data, buffer)
        buffer.seek(0)
        return buffer, f"{sim_name}.pt"

    @staticmethod
    @handle_errors
    def export_sql(sim_data, params, history, sim_name, sim_id, N, dx):
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        c.execute('''CREATE TABLE simulations (
                     id TEXT PRIMARY KEY, sim_name TEXT,
                     twin_spacing REAL, applied_stress REAL,
                     applied_stress_angle REAL, W REAL,
                     geometry_type TEXT, created_at TEXT,
                     grid_size INTEGER, dx REAL)''')
        c.execute('''CREATE TABLE frames (
                     sim_id TEXT, frame_idx INTEGER,
                     phi BLOB, eta1 BLOB, eta2 BLOB,
                     sigma_eq BLOB, sigma_h BLOB, h BLOB, eps_p_mag BLOB)''')
        created_at = sim_data.get('metadata', {}).get('created_at',
                                                     datetime.now().isoformat())
        c.execute("INSERT INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (sim_id, sim_name, params.get('twin_spacing', 0.0),
                   params.get('applied_stress', 0.0),
                   params.get('applied_stress_angle', 0.0),
                   params.get('W', 0.0),
                   params.get('geometry_type', 'standard'),
                   created_at, params.get('N', N), params.get('dx', dx)))
        for idx, frame in enumerate(history):
            c.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (sim_id, idx,
                       pickle.dumps(frame['phi']),
                       pickle.dumps(frame['eta1']),
                       pickle.dumps(frame['eta2']),
                       pickle.dumps(frame['sigma_eq']),
                       pickle.dumps(frame['sigma_h']),
                       pickle.dumps(frame['h']),
                       pickle.dumps(frame['eps_p_mag'])))
        conn.commit()
        dump_buffer = StringIO()
        for line in conn.iterdump():
            dump_buffer.write('%s\n' % line)
        conn.close()
        sql_str = dump_buffer.getvalue()
        return BytesIO(sql_str.encode()), f"{sim_name}.sql"

    @staticmethod
    @handle_errors
    def export_csv(history, sim_name, extent, N, dx):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], N)
            y = np.linspace(extent[2], extent[3], N)
            X, Y = np.meshgrid(x, y)
            for idx, frame in enumerate(history):
                df = pd.DataFrame({
                    'x': X.flatten(), 'y': Y.flatten(),
                    'phi': frame['phi'].flatten(),
                    'eta1': frame['eta1'].flatten(),
                    'eta2': frame['eta2'].flatten(),
                    'sigma_eq_GPa': (frame['sigma_eq'] / 1e9).flatten(),
                    'sigma_h_GPa': (frame['sigma_h'] / 1e9).flatten(),
                    'h_nm': frame['h'].flatten(),
                    'eps_p_mag': frame['eps_p_mag'].flatten()})
                csv_str = df.to_csv(index=False)
                zf.writestr(f"{sim_name}_frame_{idx:04d}.csv", csv_str)
        zip_buffer.seek(0)
        return zip_buffer, f"{sim_name}_csv.zip"

    @staticmethod
    @handle_errors
    def export_json(sim_data, params, history, sim_name):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        export_data = {'sim_name': sim_name, 'params': params,
                       'metadata': sim_data.get('metadata', {}),
                       'history': convert(history)}
        json_str = json.dumps(export_data, indent=2, cls=NumpyEncoder)
        return BytesIO(json_str.encode()), f"{sim_name}.json"

    @staticmethod
    @handle_errors
    def export_hdf5(sim_data, params, history, sim_name, N, dx):
        if not H5PY_AVAILABLE:
            st.error("HDF5 export requires h5py. Please install it: pip install h5py")
            return None, None
        buffer = BytesIO()
        with h5py.File(buffer, 'w') as f:
            param_grp = f.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    param_grp.attrs[k] = v
                elif isinstance(v, np.ndarray):
                    param_grp.create_dataset(k, data=v)
                else:
                    # tuples, lists, dicts → JSON string
                    param_grp.attrs[k] = json.dumps(v, default=str)
            meta_grp = f.create_group('metadata')
            for k, v in sim_data.get('metadata', {}).items():
                if isinstance(v, (int, float, str, bool)):
                    meta_grp.attrs[k] = v
                elif isinstance(v, dict):
                    subgrp = meta_grp.create_group(k)
                    for sk, sv in v.items():
                        if isinstance(sv, (int, float, str, bool)):
                            subgrp.attrs[sk] = sv
                        else:
                            subgrp.attrs[sk] = json.dumps(sv, default=str)
                else:
                    meta_grp.attrs[k] = json.dumps(v, default=str)
            x = np.linspace(-N * dx / 2, N * dx / 2, N)
            y = np.linspace(-N * dx / 2, N * dx / 2, N)
            f.create_dataset('x', data=x)
            f.create_dataset('y', data=y)
            frame_grp = f.create_group('frames')
            for idx, frame in enumerate(history):
                grp = frame_grp.create_group(f'frame_{idx:04d}')
                for field in ['phi', 'eta1', 'eta2', 'sigma_eq', 'sigma_h',
                              'h', 'eps_p_mag', 'sigma_y']:
                    if field in frame:
                        grp.create_dataset(field, data=frame[field],
                                           compression='gzip')
        buffer.seek(0)
        return buffer, f"{sim_name}.h5"

    @staticmethod
    @handle_errors
    def bulk_export_all_simulations(N, dx, extent):
        all_sims = SimulationDatabase.get_all_simulations()
        if not all_sims:
            return None
        bulk_buffer = BytesIO()
        with zipfile.ZipFile(bulk_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            summary = f"MULTI-SIMULATION EXPORT SUMMARY\n"
            summary += f"Generated: {datetime.now().isoformat()}\n"
            summary += f"Total Simulations: {len(all_sims)}\n\n"
            for sim_id, sim_data in all_sims.items():
                try:
                    params = sim_data.get('params', {})
                    history = sim_data.get('results_history', [])
                    metadata = sim_data.get('metadata', {})
                    sim_name = build_sim_name(params, sim_id)
                    sim_dir = f"simulation_{sim_id}"
                    zf.writestr(f"{sim_dir}/parameters.json",
                                json.dumps(params, indent=2, cls=NumpyEncoder))
                    zf.writestr(f"{sim_dir}/metadata.json",
                                json.dumps(metadata, indent=2, cls=NumpyEncoder))
                    x = np.linspace(extent[0], extent[1], N)
                    y = np.linspace(extent[2], extent[3], N)
                    X, Y = np.meshgrid(x, y)
                    for idx, frame in enumerate(history):
                        df = pd.DataFrame({
                            'x': X.flatten(), 'y': Y.flatten(),
                            'phi': frame['phi'].flatten(),
                            'sigma_eq_GPa': (frame['sigma_eq'] / 1e9).flatten(),
                            'sigma_h_GPa': (frame['sigma_h'] / 1e9).flatten(),
                            'h_nm': frame['h'].flatten()})
                        zf.writestr(f"{sim_dir}/frame_{idx:04d}.csv",
                                    df.to_csv(index=False))
                    summary += f"\nSimulation {sim_id}:\n"
                    summary += f"  Name: {sim_name}\n"
                    summary += f"  λ = {params.get('twin_spacing', 0):.1f} nm\n"
                    summary += f"  σ_app = {params.get('applied_stress', 0) / 1e6:.0f} MPa\n"
                    summary += f"  θ = {params.get('applied_stress_angle', 0):.0f}°\n"
                    summary += f"  W = {params.get('W', 0):.1f}\n"
                    summary += f"  Frames: {len(history)}\n"
                except Exception as e:
                    summary += f"\nSimulation {sim_id}: ERROR - {str(e)}\n"
            zf.writestr("EXPORT_SUMMARY.txt", summary)
        bulk_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return bulk_buffer, f"twin_all_simulations_{timestamp}.zip"


# ============================================================================
# PARAMETER SWEEP
# ============================================================================
class ParameterSweep:
    @staticmethod
    @handle_errors
    def run_sweep(base_params, param_name, values, save=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, val in enumerate(values):
            status_text.text(f"Running {param_name} = {val:.3f} ({i + 1}/{len(values)})")
            params = base_params.copy()
            params[param_name] = val
            if 'history' in params:
                del params['history']
            if 'geom_viz' in params:
                del params['geom_viz']
            try:
                solver = NanotwinnedCuSolver(params)
                n_steps = params.get('n_steps', 100)
                for step in range(n_steps):
                    solver.step()
                final_results = {
                    'phi': solver.phi.copy(),
                    'eta1': solver.eta1.copy(),
                    'eta2': solver.eta2.copy(),
                    'sigma_eq': solver.history['avg_stress'][-1] if solver.history['avg_stress'] else 0,
                    'max_stress': solver.history['max_stress'][-1] if solver.history['max_stress'] else 0,
                    'plastic_work': solver.history['plastic_work'][-1] if solver.history['plastic_work'] else 0,
                    'twin_spacing_avg': solver.history['twin_spacing_avg'][-1] if solver.history['twin_spacing_avg'] else 0,
                    'energy': solver.history['energy'][-1] if solver.history['energy'] else 0,
                }
                if save:
                    SimulationDatabase.save_simulation(params, solver.history, None)
                results.append({'param_value': val,
                                'convergence': final_results,
                                'solver': solver})
            except Exception as e:
                st.error(f"Failed for {param_name}={val}: {e}")
                results.append({'param_value': val,
                                'convergence': None, 'error': str(e)})
            progress_bar.progress((i + 1) / len(values))
        status_text.text("Sweep completed!")
        return results


# ============================================================================
# ███████████████████████████████████████████████████████████████████████████
# ███  PLASTICITY PARAMETER INTELLIGENT RECOMMENDER v8.3.0             ██████
# ███  FAISS Retrieval · Ollama LLM · LatentMoE · Learned Priors ·     ██████
# ███  DUAL-MODE PROMPTS: strict_extract vs reasoned_inference          ██████
# ███  Chain-of-Thought reasoning surfaced in UI + scored by LatentMoE  ██████
# ███  GATEKEEPER FIX: param alias canonicalization + per-param         ██████
# ███  heuristic fallback + value coercion + UI debug toggle            ██████
# ███████████████████████████████████████████████████████████████████████████
# ============================================================================

PLASTICITY_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "rho0": {
        "label": "Initial Dislocation Density",
        "symbol": "ρ₀",
        "aliases": [
            "initial dislocation density", "dislocation density",
            "rho_0", "rho0", "ρ₀", "forest dislocation density",
            "mobile dislocation density", "immobile dislocation density",
            "total dislocation density", "geometrically necessary dislocation",
        ],
        "unit": "m^-2", "ui_unit": "m⁻²", "ui_scale": 1.0,
        "valid_range": (1e10, 1e17), "soft_range": (1e11, 1e16),
        "defaults": {"Cu": 1e12, "Al": 1e12, "Ni": 1e13},
        "expected_file": "initial_dislocation_density_metadatabase.json",
    },
    "mu": {
        "label": "Shear Modulus",
        "symbol": "μ",
        "aliases": [
            "shear modulus", "mu", "G", "elastic shear modulus",
            "c44", "c_44", "second lame parameter", "second lamé parameter",
            "rigidity modulus",
        ],
        "unit": "Pa", "ui_unit": "GPa", "ui_scale": 1e9,
        "valid_range": (1e9, 5e11), "soft_range": (20e9, 200e9),
        "defaults": {"Cu": 48e9, "Al": 26e9, "Ni": 80e9},
        "expected_file": "shear_modulus_metadatabase.json",
    },
    "gamma0_dot": {
        "label": "Reference Strain Rate",
        "symbol": "γ̇₀",
        "aliases": [
            "reference strain rate", "reference strain-rate",
            "gamma0_dot", "gamma_dot_0", "γ̇₀", "reference shear rate",
            "pre-exponential strain rate", "attempt frequency",
            "reference shear strain rate",
        ],
        "unit": "s^-1", "ui_unit": "s⁻¹", "ui_scale": 1.0,
        "valid_range": (1e-6, 1e12), "soft_range": (1e-4, 1e6),
        "defaults": {"Cu": 1e-3, "Al": 1e-3, "Ni": 1e-3},
        "expected_file": "reference_strain_rate_metadatabase.json",
    },
    "srs": {
        "label": "Strain-Rate Sensitivity Exponent",
        "symbol": "m",
        "aliases": [
            "strain rate sensitivity", "srs", "m exponent",
            "stress exponent", "rate sensitivity", "strain-rate sensitivity",
            "rate sensitivity exponent", "n exponent", "viscous exponent",
        ],
        "unit": "dimensionless", "ui_unit": "–", "ui_scale": 1.0,
        "valid_range": (1.0, 200.0), "soft_range": (5.0, 50.0),
        "defaults": {"Cu": 20.0, "Al": 20.0, "Ni": 20.0},
        "expected_file": "inverse_strain_rate_sensitivity_metadatabase.json",
    },
    "sigma0": {
        "label": "Friction / Initial Yield Stress",
        "symbol": "σ₀",
        "aliases": [
            "initial yield stress", "friction stress", "sigma_0", "σ₀",
            "lattice friction", "peierls stress", "peierls-nabarro stress",
            "athermal stress", "yield strength", "lattice resistance",
            "friction lattice stress",
        ],
        "unit": "Pa", "ui_unit": "MPa", "ui_scale": 1e6,
        "valid_range": (1e5, 2e9), "soft_range": (10e6, 500e6),
        "defaults": {"Cu": 50e6, "Al": 30e6, "Ni": 70e6},
        "expected_file": "friction_lattice_stress_metadatabase.json",
    },
}

PARAM_ORDER: List[str] = ["rho0", "mu", "gamma0_dot", "srs", "sigma0"]

SOLVER_KEY_MAP: Dict[str, str] = {
    "rho0": "rho0",
    "mu": "mu",
    "gamma0_dot": "gamma0_dot",
    "srs": "m",
    "sigma0": "sigma0",
}

TARGET_JSON_FILES: List[str] = [
    "friction_lattice_stress_metadatabase.json",
    "initial_dislocation_density_metadatabase.json",
    "reference_strain_rate_metadatabase.json",
    "shear_modulus_metadatabase.json",
    "inverse_strain_rate_sensitivity_metadatabase.json",
]


# ----------------------------------------------------------------------------
# PARAM ALIAS MAP — canonicalizes whatever the LLM emits to a known key
# ----------------------------------------------------------------------------
_PARAM_ALIASES: Dict[str, str] = {
    # ---- rho0 --------------------------------------------------------------
    "rho_0": "rho0", "rho": "rho0", "ρ₀": "rho0", "ρ0": "rho0",
    "rho0_": "rho0", "rho_0_": "rho0",
    "dislocation_density": "rho0",
    "initial_dislocation_density": "rho0",
    "initial disloc density": "rho0",
    "forest_density": "rho0",
    "rho_dis": "rho0",
    # ---- mu ----------------------------------------------------------------
    "g": "mu", "shear_modulus": "mu", "μ": "mu", "mu_s": "mu",
    "shear modulus": "mu", "rigidity_modulus": "mu", "c44": "mu",
    "c_44": "mu", "elastic_shear_modulus": "mu",
    # ---- gamma0_dot --------------------------------------------------------
    "gamma_dot_0": "gamma0_dot", "gamma_0_dot": "gamma0_dot",
    "γ̇₀": "gamma0_dot", "γ0": "gamma0_dot", "gamma_dot": "gamma0_dot",
    "gamma0": "gamma0_dot",
    "reference_strain_rate": "gamma0_dot",
    "reference_shear_rate": "gamma0_dot",
    "strain_rate_reference": "gamma0_dot",
    "gammadot0": "gamma0_dot",
    "gamma_dot0": "gamma0_dot",
    "reference_strain-rate": "gamma0_dot",
    # ---- srs ---------------------------------------------------------------
    "m": "srs", "m_exponent": "srs", "rate_sensitivity": "srs",
    "strain_rate_sensitivity": "srs",
    "srs_exponent": "srs",
    "rate_sensitivity_exponent": "srs",
    "strain-rate_sensitivity": "srs",
    "stress_exponent": "srs",
    "n": "srs",  # inverse of m, but treat as same identifier
    # ---- sigma0 ------------------------------------------------------------
    "sigma_0": "sigma0", "σ₀": "sigma0", "σ0": "sigma0",
    "yield_stress": "sigma0", "friction_stress": "sigma0",
    "peierls_stress": "sigma0", "lattice_friction": "sigma0",
    "initial_yield_stress": "sigma0", "sigma_y0": "sigma0",
    "friction": "sigma0",
    "lattice_friction_stress": "sigma0",
    "peierls-nabarro stress": "sigma0",
    "friction lattice stress": "sigma0",
    "athermal_stress": "sigma0",
    "yield_strength": "sigma0",
}


def _canonicalize_param(raw_p: str) -> Optional[str]:
    """Map whatever the LLM emits (`rho_0`, `ρ₀`, `m`, `gamma_dot_0`, …)
    onto one of the five canonical keys in PLASTICITY_ONTOLOGY.

    Order of resolution:
      1. exact match against canonical keys
      2. exact match against the alias map
      3. normalized (lowercase, strip whitespace) exact match
      4. fuzzy substring match against canonical keys
    """
    if raw_p is None:
        return None
    p = str(raw_p).strip()
    if not p:
        return None

    # 1) exact canonical
    if p in PLASTICITY_ONTOLOGY:
        return p

    # 2) exact alias (case-sensitive, includes unicode symbols)
    if p in _PARAM_ALIASES:
        return _PARAM_ALIASES[p]

    # 3) lowercase / strip
    low = p.lower().replace(" ", "_").replace("-", "_")
    if low in PLASTICITY_ONTOLOGY:
        return low
    if low in _PARAM_ALIASES:
        return _PARAM_ALIASES[low]

    # 4) fuzzy: substring match either direction
    for canon in PLASTICITY_ONTOLOGY:
        if canon in low or low in canon:
            return canon
    # also fuzzy against aliases for robustness
    for alias, canon in _PARAM_ALIASES.items():
        if alias and (alias in low or low in alias):
            return canon
    return None


def _coerce_value(item: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Return (numeric_value, unit_string) from a validated item.

    Handles:
        · {"value": 48, "unit": "GPa"}           → (48.0, "GPa")
        · {"value": "48 GPa"}                    → (48.0, "GPa")
        · {"value": "~1e12 m^-2"}                → (1e12, "m^-2")
        · {"value": "1.0 × 10^12", "unit": "m^-2"}→ (1e12, "m^-2")
    """
    raw_v = item.get("value")
    explicit_unit = str(item.get("unit") or "").strip()

    # numeric already
    if isinstance(raw_v, (int, float)) and not isinstance(raw_v, bool):
        return float(raw_v), explicit_unit

    s = str(raw_v or "").strip()
    if not s:
        return None, explicit_unit

    # pattern: optional sign, digits, optional decimal, optional × 10^n
    m = re.search(
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        r"(?:\s*[×xX\*]\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?)?"
        r"(.*)",
        s,
    )
    if not m:
        return None, explicit_unit

    try:
        mantissa = float(m.group(1))
    except (TypeError, ValueError):
        return None, explicit_unit
    exponent = int(m.group(2)) if m.group(2) else 0
    value = mantissa * (10 ** exponent)
    trailing = m.group(3).strip()

    # unit precedence: explicit unit field > trailing text
    unit = explicit_unit or trailing

    # If the numeric was "1e12" style inside the string, no exponent group
    # will have matched; but the mantissa already contains "e12".
    return value, unit


def _pl_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalize_unit(value: float, unit: str, param: str) -> float:
    u = (unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    v = float(value)
    if param in ("mu", "sigma0"):
        if "gpa" in u:
            return v * 1e9
        if "mpa" in u:
            return v * 1e6
        if "kpa" in u:
            return v * 1e3
        if "pa" in u:
            return v
    if param == "rho0":
        if "cm^-2" in u or "cm-2" in u:
            return v * 1e4
        if "mm^-2" in u or "mm-2" in u:
            return v * 1e6
    return v


def _pl_clamp(value: float, param: str) -> Tuple[float, bool]:
    lo, hi = PLASTICITY_ONTOLOGY[param]["valid_range"]
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


def _pl_fmt(param: str, si_value: float) -> str:
    spec = PLASTICITY_ONTOLOGY[param]
    ui_val = si_value / spec["ui_scale"]
    if param in ("rho0", "gamma0_dot"):
        return f"{ui_val:.2e} {spec['ui_unit']}"
    if param == "mu":
        return f"{ui_val:.2f} {spec['ui_unit']}"
    if param == "sigma0":
        return f"{ui_val:.1f} {spec['ui_unit']}"
    return f"{ui_val:.2f} {spec['ui_unit']}"


# ----------------------------------------------------------------------------
# OLLAMA CLIENT
# ----------------------------------------------------------------------------
class PlasticityOllamaClient:
    """Thin wrapper around Ollama /api/generate with retries + JSON coercion."""

    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b",
                 timeout: float = 120.0, max_retries: int = 2):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def is_available(url: str = "http://localhost:11434") -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        try:
            r = _requests.get(f"{url.rstrip('/')}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def list_models(url: str = "http://localhost:11434") -> List[str]:
        if not REQUESTS_AVAILABLE:
            return []
        try:
            r = _requests.get(f"{url.rstrip('/')}/api/tags", timeout=3.0)
            if r.status_code == 200:
                return sorted(m.get("name", "") for m in r.json().get("models", []))
        except Exception:
            pass
        return []

    def generate_json(self, prompt: str, system: Optional[str] = None,
                      debug: bool = False) -> Optional[Any]:
        """Call Ollama with format=json and lenient JSON parsing.

        num_predict is 4096 to accommodate the reasoned-inference CoT.
        If `debug=True`, raw responses are logged at INFO level so the user
        can see exactly what the LLM is emitting.
        """
        if not REQUESTS_AVAILABLE:
            return None
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 4096},
        }
        if system:
            payload["system"] = system

        for attempt in range(self.max_retries + 1):
            try:
                r = _requests.post(f"{self.url}/api/generate",
                                   json=payload, timeout=self.timeout)
                r.raise_for_status()
                raw = r.json().get("response", "")
                parsed = self._parse_lenient(raw)
                if debug:
                    logger.info("Ollama raw (first 800 chars): %s",
                                str(raw)[:800])
                else:
                    logger.debug("Ollama raw (first 400 chars): %s",
                                 (raw or "")[:400])
                return parsed
            except Exception as e:
                logger.warning("Ollama attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    return None
                time.sleep(0.75 * (attempt + 1))
        return None

    @staticmethod
    def _parse_lenient(raw: str) -> Any:
        if raw is None:
            return None
        s = raw.strip().lstrip()
        # strip markdown code fences (leading or trailing)
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()
        for oc, cc in (("[", "]"), ("{", "}")):
            i, j = s.find(oc), s.rfind(cc)
            if i != -1 and j > i:
                try:
                    return json.loads(s[i: j + 1])
                except Exception:
                    continue
        try:
            return json.loads(s)
        except Exception:
            return None


# ----------------------------------------------------------------------------
# CORPUS LOADER (targets the 5 canonical JSON files)
# ----------------------------------------------------------------------------
class PlasticityCorpus:
    """Loads, chunks, and caches text records from the 5 metadatabases."""

    _CACHE_VERSION = "v83"

    def __init__(self, db_dir: str = "json_metadatabase", max_chars: int = 4000):
        self.db_dir = db_dir
        self.max_chars = max_chars

    def discover_files(self) -> List[str]:
        found: List[str] = []
        for fname in TARGET_JSON_FILES:
            fpath = os.path.join(self.db_dir, fname)
            if os.path.exists(fpath):
                found.append(fpath)
        if os.path.isdir(self.db_dir):
            for fname in os.listdir(self.db_dir):
                if not fname.lower().endswith(".json"):
                    continue
                fpath = os.path.join(self.db_dir, fname)
                if fpath in found:
                    continue
                low = fname.lower()
                if any(k in low for k in [
                    "dislocation", "shear", "strain_rate", "strain-rate",
                    "friction", "lattice", "yield", "sensitivity",
                ]):
                    found.append(fpath)
        return found

    def load(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        cache_key = f"pl_corpus_{self._CACHE_VERSION}"
        if not force_reload and cache_key in st.session_state:
            return st.session_state[cache_key]

        corpus: List[Dict[str, Any]] = []
        for fpath in self.discover_files():
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Cannot load %s: %s", fpath, e)
                continue

            if isinstance(data, dict):
                for key in ("records", "data", "papers", "entries", "items"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    data = [data]
            if not isinstance(data, list):
                continue

            fname = os.path.basename(fpath)
            for item in data:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("Title") or item.get("title")
                            or item.get("name") or "")
                abstract = str(item.get("Abstract") or item.get("abstract")
                               or item.get("summary") or "")
                full = str(item.get("Full Text") or item.get("full_text")
                           or item.get("text") or item.get("content") or "")
                text = (f"Title: {title}. "
                        f"Abstract: {abstract}. "
                        f"Full Text: {full[:self.max_chars]}")
                corpus.append({
                    "source": fname,
                    "title": title,
                    "text": text,
                    "raw": item,
                })

        st.session_state[cache_key] = corpus
        return corpus

    @staticmethod
    def keyword_prefilter(corpus: List[Dict[str, Any]],
                          material: str, k: int = 30) -> List[Dict[str, Any]]:
        syn = {
            "cu": ["cu", "copper"],
            "al": ["al", "aluminium", "aluminum"],
            "ni": ["ni", "nickel"],
            "fe": ["fe", "iron", "steel"],
            "cocrfeni": ["cocrfeni", "co-cr-fe-ni", "hea", "mpea"],
            "ti": ["ti", "titanium"],
            "mg": ["mg", "magnesium"],
        }
        keys = syn.get(material.lower(), [material.lower()])
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in corpus:
            low = doc["text"].lower()
            score = 0.0
            for kk in keys:
                score += low.count(kk) * 2.0
            for spec in PLASTICITY_ONTOLOGY.values():
                for alias in spec["aliases"]:
                    if alias in low:
                        score += 1.0
                        break
            scored.append((score, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in scored[:k]]


# ----------------------------------------------------------------------------
# FAISS RETRIEVAL LAYER
# ----------------------------------------------------------------------------
class PlasticityFAISSRetriever:
    """Dense retrieval over the corpus using SentenceTransformer + FAISS."""

    CACHE_DIR = ".plasticity_cache"
    INDEX_FILE = "faiss_index.pkl"

    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self._model = None
        self._index = None
        self._doc_vectors = None
        self._docs: List[Dict[str, Any]] = []
        self._dim: int = 0
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None and SBERT_AVAILABLE:
            try:
                self._model = _SentenceTransformer(self.embed_model_name, device="cpu")
            except Exception as e:
                logger.warning("SentenceTransformer failed: %s", e)
                self._model = None
        return self._model

    def build(self, corpus: List[Dict[str, Any]], force: bool = False) -> None:
        if not corpus:
            self._docs, self._index, self._doc_vectors = [], None, None
            return

        fingerprint = _pl_hash(json.dumps(
            [d["source"] + "|" + d["title"] for d in corpus], sort_keys=False
        ))
        cache_path = os.path.join(self.CACHE_DIR, self.INDEX_FILE)

        if not force and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                if payload.get("fingerprint") == fingerprint:
                    self._docs = payload["docs"]
                    self._doc_vectors = payload["vectors"]
                    self._dim = payload["dim"]
                    self._attach_index()
                    return
            except Exception as e:
                logger.warning("FAISS cache load failed: %s", e)

        m = self.model
        if m is None:
            self._docs = list(corpus)
            self._doc_vectors = self._tfidf_fallback_vectors(corpus)
            self._dim = self._doc_vectors.shape[1] if self._doc_vectors.size else 0
            self._attach_index()
            self._persist(fingerprint, cache_path)
            return

        with self._lock:
            texts = [d["text"][:1500] for d in corpus]
            vectors = m.encode(
                texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)

        self._docs = list(corpus)
        self._doc_vectors = vectors
        self._dim = vectors.shape[1]
        self._attach_index()
        self._persist(fingerprint, cache_path)

    def _tfidf_fallback_vectors(self, corpus):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=2048, stop_words="english")
            X = vec.fit_transform([d["text"] for d in corpus]).toarray().astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / norms
        except Exception:
            return np.zeros((len(corpus), 1), dtype=np.float32)

    def _attach_index(self) -> None:
        if self._doc_vectors is None or self._doc_vectors.size == 0:
            self._index = None
            return
        if FAISS_AVAILABLE:
            try:
                index = _faiss.IndexFlatIP(self._doc_vectors.shape[1])
                index.add(self._doc_vectors)
                self._index = index
                return
            except Exception as e:
                logger.warning("FAISS index build failed: %s", e)
        self._index = None

    def _persist(self, fingerprint: str, path: str) -> None:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({
                    "fingerprint": fingerprint,
                    "docs": self._docs,
                    "vectors": self._doc_vectors,
                    "dim": self._dim,
                }, f)
        except Exception as e:
            logger.warning("FAISS persist failed: %s", e)

    def search(self, query: str, k: int = 15,
               material_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._docs or self._doc_vectors is None or self._doc_vectors.size == 0:
            return []

        m = self.model
        if m is not None:
            qvec = m.encode(
                [query], batch_size=1, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)
        else:
            qvec = np.zeros((1, self._dim), dtype=np.float32)
            for i, d in enumerate(self._docs):
                if query.lower()[:30] in d["text"].lower():
                    qvec[0, i % self._dim] += 1.0
            nrm = np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12
            qvec = qvec / nrm

        top_k = min(len(self._docs), max(k * 3, k))
        if self._index is not None:
            D, I = self._index.search(qvec, top_k)
            sims, idxs = D[0], I[0]
        else:
            sims_all = (self._doc_vectors @ qvec[0])
            idxs = np.argsort(-sims_all)[:top_k]
            sims = sims_all[idxs]

        hits: List[Tuple[float, Dict[str, Any]]] = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self._docs):
                continue
            doc = self._docs[int(idx)]
            bonus = 0.0
            if material_hint and material_hint.lower() in doc["text"].lower():
                bonus += 0.05
            hits.append((float(sim) + bonus, doc))
        hits.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in hits[:k]]


# ----------------------------------------------------------------------------
# LLM PROMPTS: TWO MODES
# ----------------------------------------------------------------------------
_EXTRACT_SCHEMA = (
    '{"param": "rho0|mu|gamma0_dot|srs|sigma0", '
    '"value": <number>, '
    '"unit": "<string>", '
    '"material": "<string>", '
    '"temp": <number in K or null>, '
    '"strain_rate": <number in s^-1 or null>, '
    '"method": "explicit|LLM_inferred", '
    '"confidence": <0.0-1.0>, '
    '"evidence": "<short quoted snippet or empty>", '
    '"reasoning": "<3-5 step chain-of-thought, or empty for explicit>"}'
)

_REASONED_INFERENCE_SCHEMA = _EXTRACT_SCHEMA  # same shape, documented separately

_STRICT_EXTRACT_PROMPT = """You are a strict materials-science NER system.
Extract ONLY plasticity parameters that are explicitly and unambiguously
stated as numerical values in the text. Do NOT infer. Do NOT fill in
missing parameters.

Parameters of interest (the `param` field MUST be exactly one of these
five ASCII strings — do NOT use symbols or synonyms):
  - "rho0"        (initial dislocation density, m^-2)
  - "mu"          (shear modulus, Pa or GPa)
  - "gamma0_dot"  (reference strain rate, s^-1)
  - "srs"         (strain-rate sensitivity exponent, dimensionless)
  - "sigma0"      (friction / initial yield stress, Pa or MPa)

For each parameter you find, report the value verbatim (preserve the unit
as written), identify the material mentioned in the text, and quote the
exact sentence as evidence. Set method="explicit" and confidence >= 0.8.
Leave the reasoning field as an empty string.

If a parameter is missing, qualitative, or only mentioned without a number,
OMIT it from the output entirely.

Schema per element:
  {schema}

Return ONLY a JSON ARRAY. No markdown, no prose.

TEXT:
\"\"\"{text}\"\"\"
"""

_REASONED_INFERENCE_PROMPT = """You are an expert materials-science AI.

GOAL: For the target material "{material}" at T={temp_k} K and strain rate
{strain_rate} s^-1, return ALL FIVE of these plasticity parameters.

⚠️ CRITICAL: In the JSON output, the `param` field MUST be EXACTLY one of
these five ASCII strings (character-for-character):

    "rho0"        — initial dislocation density           [m^-2]
    "mu"          — shear modulus                          [Pa]
    "gamma0_dot"  — reference strain rate                  [s^-1]
    "srs"         — strain-rate sensitivity exponent m     [dimensionless]
    "sigma0"      — friction / initial yield stress        [Pa]

Do NOT use "rho_0", "ρ₀", "γ̇₀", "m", "σ₀", "shear_modulus", or any other
variant. Use the exact five strings above. The parser is strict.

For EACH parameter, choose ONE of two paths:

PATH A — EXPLICIT EXTRACTION
  If the text states the parameter as a number with a unit, copy it
  verbatim. Set method="explicit", confidence=0.9, evidence=<exact
  sentence>, reasoning="" (empty string).

PATH B — REASONED INFERENCE (only if Path A does not apply)
  Derive the value using the physics-based reasoning chain below. Write
  the chain in the `reasoning` field as 3-5 short numbered steps. Set
  method="LLM_inferred" and confidence between 0.4 and 0.6 depending on
  how clearly the text supports the chain.

REASONING CHAINS (use these formulas, do NOT just look up a constant):

  mu (T, material):
    Step 1: Identify material from text or from target "{material}".
            If target is "?", identify from text. If unclear, use Cu as default.
    Step 2: Pick room-T baseline:
            Cu: 48 GPa, Al: 26 GPa, Ni: 80 GPa, Fe: 80 GPa, Ti: 44 GPa,
            Mg: 17 GPa, Cr: 115 GPa (bcc), Co: 75 GPa, Au: 27 GPa, Ag: 30 GPa,
            CoCrFeNi (HEA): 80 GPa.
    Step 3: If T != 300 K, apply: mu(T) = mu_300 * (1 - 5e-4 * (T - 300)).
    Step 4: If text mentions alloying, cold work, or irradiation hardening,
            adjust ±2 GPa (state direction in reasoning).
    Step 5: Output value in Pa.

  sigma0 (T, material, processing):
    Step 1: Identify material and processing from text.
    Step 2: Pick baseline:
            Cu: 50 MPa, Al: 30 MPa, Ni: 70 MPa, Fe: 150 MPa, Ti: 200 MPa,
            Mg: 80 MPa, Au: 10 MPa, Ag: 20 MPa, CoCrFeNi: 300 MPa.
    Step 3: If text mentions solid-solution strengthening, add 10-50 MPa.
            If precipitation strengthening, add 100-500 MPa.
            If heavy cold work, add 50-200 MPa.
            If annealed, subtract 20-50 MPa.
    Step 4: If T > 400 K, scale by (1 - 3e-4*(T-300)) for thermal softening.
    Step 5: Output value in Pa.

  rho0 (T, material, processing):
    Step 1: Identify material and processing.
    Step 2: Pick baseline annealed density:
            Cu/Al/Ni (annealed FCC): 1e12 m^-2
            Fe (annealed BCC):       1e13 m^-2
            HEA / heavily alloyed:   1e13 m^-2
    Step 3: If text mentions cold rolling / ECAP / HPT / shock loading,
            multiply by 10-100 (e.g., 1e14 to 1e15).
    Step 4: If text mentions annealing, recovery, or recrystallization,
            divide by 10 (e.g., 1e11).
    Step 5: If T > 600 K, account for thermal recovery:
            rho(T) = rho_0 * exp(-D(T)*t); for short times, divide by 2.
    Step 6: Output value in m^-2.

  gamma0_dot (T, strain_rate, material):
    Step 1: If the user-provided strain rate ({strain_rate} s^-1) is given,
            use that as gamma0_dot (it IS the reference rate).
    Step 2: Otherwise, baseline is 1e-3 s^-1 (quasi-static).
    Step 3: If text mentions dynamic loading / shock, set to 1e3-1e6.
    Step 4: If text mentions creep / very low rate, set to 1e-8-1e-6.
    Step 5: Output value in s^-1.

  srs (m, material, T):
    Step 1: Identify material class.
    Step 2: Pick baseline:
            FCC metals (Cu, Al, Ni, Au, Ag): m = 20 (or n = 1/m = 0.05)
            BCC metals (Fe, W, Mo, Cr):      m = 100 (n = 0.01)
            HCP metals (Mg, Ti, Zn, Zr):     m = 50  (n = 0.02)
            HEA / MPEA:                       m = 30
    Step 3: If T > 0.5 T_m, m decreases (rate sensitivity rises).
    Step 4: If text mentions superplasticity, m → 1 (set to 200).
    Step 5: Output dimensionless value.

CRITICAL RULES:
- Return ALL FIVE parameters for every document.
- The `param` field MUST be exactly "rho0", "mu", "gamma0_dot", "srs",
  or "sigma0". Any other spelling will be rejected.
- For Path A, evidence MUST be a verbatim sentence from the text.
- For Path B, the reasoning field MUST contain 3-5 numbered steps
  citing the formula used.
- If the text contains NO information at all (blank/garbage), still
  return Path B for all 5 parameters with the simplest possible chain
  (just Steps 1 and 2) and confidence=0.3.

Return ONLY a JSON ARRAY. Schema per element:
  {schema}
No markdown, no comments, no prose outside JSON.

TEXT:
\"\"\"{text}\"\"\"
"""


# ----------------------------------------------------------------------------
# LLM + HEURISTIC EXTRACTION (DUAL-MODE + GATEKEEPER FIX)
# ----------------------------------------------------------------------------
class PlasticityParameterExtractor:
    """Extracts plasticity parameters from text, in one of two modes:

        mode="strict_extract"      → ONLY explicit values (for priors)
        mode="reasoned_inference"  → fill gaps via Chain-of-Thought (runtime)

    v8.3.0 GATEKEEPER FIX:
      · `_validate()` now canonicalizes the LLM's `param` field through
        `_canonicalize_param()`, so `rho_0`, `ρ₀`, `m`, `gamma_dot_0`,
        `σ₀`, `shear_modulus`, etc. are all mapped to the five canonical
        keys instead of being silently dropped.
      · `_coerce_value()` recovers values that arrive as strings
        ("48 GPa", "~1e12 m^-2", "1.0 × 10^12").
      · The heuristic extractor now ALWAYS runs and merges per-parameter,
        filling in whatever the LLM missed, rather than being a total
        fallback that only fires when the LLM returns zero items.
    """

    NUM_RE = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(?:[×xX\*]\s*10\s*\^?\s*\{?(-?\d+)\}?)?\s*"
        r"(GPa|MPa|kPa|Pa|m\^?-?2|m-2|s\^?-?1|/s)?",
        re.I,
    )

    def __init__(self, client: Optional[PlasticityOllamaClient],
                 cache: Optional[Dict[str, Any]] = None):
        self.client = client
        self.cache = cache if cache is not None else {}

    def extract(self, text: str, material: str, temp_k: float,
                strain_rate: float, use_llm: bool = True,
                mode: str = "reasoned_inference",
                debug_llm: bool = False) -> List[Dict[str, Any]]:
        """Extract plasticity parameters from `text`.

        Parameters
        ----------
        mode : {"strict_extract", "reasoned_inference"}
        debug_llm : bool
            If True, raw Ollama responses are logged at INFO level.
        """
        key = _pl_hash(
            f"{text[:2000]}|{material}|{temp_k}|{strain_rate}|{use_llm}|{mode}"
        )
        if key in self.cache:
            return self.cache[key]

        llm_out: List[Dict[str, Any]] = []
        if use_llm and self.client is not None:
            if mode == "strict_extract":
                prompt = _STRICT_EXTRACT_PROMPT.format(
                    schema=_EXTRACT_SCHEMA, text=text[:3500])
            else:  # reasoned_inference
                prompt = _REASONED_INFERENCE_PROMPT.format(
                    schema=_REASONED_INFERENCE_SCHEMA,
                    material=material, temp_k=temp_k, strain_rate=strain_rate,
                    text=text[:3500])
            raw = self.client.generate_json(prompt, debug=debug_llm)
            logger.debug("Ollama raw (%s): %s", mode, raw)
            llm_out = self._validate(raw)
            if debug_llm:
                logger.info("LLM returned %d valid items after canonicalization "
                            "(mode=%s): %s", len(llm_out), mode,
                            [x["param"] for x in llm_out])

        # ---- PER-PARAMETER HEURISTIC MERGE (v8.3.0 fix) --------------------
        # The heuristic is now ALWAYS run. It fills in whatever the LLM
        # missed. The LLM candidates are kept first (they get the reasoning-
        # expert bonus), and heuristic-only params are appended.
        heuristic_out = self._heuristic(text, material, temp_k, strain_rate)
        have = {e["param"] for e in llm_out}
        for h in heuristic_out:
            if h["param"] not in have:
                llm_out.append(h)
                have.add(h["param"])
        # --------------------------------------------------------------------

        self.cache[key] = llm_out
        return llm_out

    @staticmethod
    def _validate(raw: Any) -> List[Dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            # ---- GATEKEEPER FIX #1: canonicalize the param string --------
            raw_p = item.get("param", "")
            p = _canonicalize_param(raw_p)
            if p is None:
                logger.debug("Dropping unrecognised param: %r", raw_p)
                continue
            # --------------------------------------------------------------

            # ---- GATEKEEPER FIX #2: coerce value/unit --------------------
            v, unit = _coerce_value(item)
            if v is None:
                logger.debug("Dropping uncoercible value for %s: %r",
                             p, item.get("value"))
                continue
            # --------------------------------------------------------------

            out.append({
                "param": p,
                "value": float(v),
                "unit": str(unit or ""),
                "material": str(item.get("material") or ""),
                "temp": item.get("temp"),
                "strain_rate": item.get("strain_rate"),
                "method": str(item.get("method") or "unknown").lower(),
                "confidence": float(item.get("confidence", 0.5) or 0.5),
                "evidence": str(item.get("evidence") or "")[:240],
                "reasoning": str(item.get("reasoning") or "")[:600],
            })
        return out

    @classmethod
    def _heuristic(cls, text: str, material: str, temp_k: float,
                   strain_rate: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        low = text.lower()
        for param, spec in PLASTICITY_ONTOLOGY.items():
            for alias in spec["aliases"]:
                idx = low.find(alias.lower())
                if idx < 0:
                    continue
                window = text[max(0, idx - 80): idx + 300]
                m = cls.NUM_RE.search(window)
                if not m:
                    continue
                try:
                    mantissa = float(m.group(1))
                    exponent = int(m.group(2)) if m.group(2) else 0
                    unit = (m.group(3) or "").strip()
                    value = mantissa * (10 ** exponent)
                except Exception:
                    continue
                out.append({
                    "param": param,
                    "value": value,
                    "unit": unit,
                    "material": material,
                    "temp": temp_k,
                    "strain_rate": strain_rate,
                    "method": "heuristic",
                    "confidence": 0.35,
                    "evidence": window[:180].strip(),
                    "reasoning": "",
                })
                break
        return out


# ----------------------------------------------------------------------------
# LATENT MoE SCORER
# ----------------------------------------------------------------------------
@dataclass
class PlasticityCandidate:
    param: str
    value_si: float
    raw_value: float
    raw_unit: str
    score: float
    confidence: float
    material: str
    temp_k: Optional[float]
    strain_rate: Optional[float]
    method: str
    source_file: str
    source_title: str
    evidence: str
    reasoning: str = ""
    clamped: bool = False

    def to_display(self) -> Dict[str, Any]:
        spec = PLASTICITY_ONTOLOGY[self.param]
        return {
            "value": self.value_si / spec["ui_scale"],
            "unit": spec["ui_unit"],
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "material": self.material or "n/a",
            "temp_K": self.temp_k,
            "method": self.method,
            "source": f"{self.source_file} — {self.source_title[:60]}",
            "clamped": self.clamped,
            "evidence": self.evidence[:140],
            "reasoning": (self.reasoning[:200] + "…"
                          if len(self.reasoning) > 200 else self.reasoning),
        }


class PlasticityLatentMoEScorer:
    """Six-expert gating: material, thermal, strain, method, confidence,
    and reasoning quality."""

    def __init__(self,
                 w_material: float = 0.40,
                 w_thermal: float = 0.20,
                 w_strain: float = 0.08,
                 w_method: float = 0.08,
                 w_confidence: float = 0.04,
                 w_reasoning: float = 0.20,
                 thermal_sigma: float = 100.0):
        self.w_material = w_material
        self.w_thermal = w_thermal
        self.w_strain = w_strain
        self.w_method = w_method
        self.w_confidence = w_confidence
        self.w_reasoning = w_reasoning
        self.thermal_sigma = thermal_sigma

    def _material_expert(self, ext_mat: str, target_mat: str) -> float:
        if not ext_mat:
            return 0.4
        a, b = ext_mat.lower().strip(), target_mat.lower().strip()
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.85
        fcc = {"cu", "al", "ni", "ag", "au", "pt", "pb", "cocrfeni", "hea"}
        bcc = {"fe", "w", "mo", "cr", "v", "nb"}
        hcp = {"mg", "ti", "zn", "co", "zr"}
        for grp in (fcc, bcc, hcp):
            if a in grp and b in grp:
                return 0.5
        return 0.25

    def _thermal_expert(self, ext_temp, target_temp) -> float:
        if ext_temp is None:
            return 0.5
        try:
            t = float(ext_temp)
        except (TypeError, ValueError):
            return 0.5
        diff = t - float(target_temp)
        return float(np.exp(-(diff ** 2) / (2.0 * self.thermal_sigma ** 2)))

    def _strain_expert(self, ext_rate, target_rate) -> float:
        if ext_rate is None:
            return 0.5
        try:
            r = float(ext_rate)
            if r <= 0 or target_rate <= 0:
                return 0.5
            ratio = math.log10(r / target_rate)
            return float(np.exp(-(ratio ** 2) / 8.0))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _method_expert(method: str) -> float:
        return {
            "experiment": 1.0, "review": 0.8,
            "md": 0.6, "molecular dynamics": 0.6,
            "dft": 0.5, "first-principles": 0.5, "ab initio": 0.5,
            "heuristic": 0.3,
            "explicit": 0.9, "llm_inferred": 0.5,
        }.get((method or "").lower(), 0.5)

    @staticmethod
    def _reasoning_expert(reasoning: str, method: str) -> float:
        """Score the quality of an LLM's chain-of-thought.

        · explicit extraction → neutral (0.5)
        · inferred with no reasoning → 0.2 (distrust)
        · inferred with N steps → 0.4 + 0.05·N, capped at 0.8
        · +0.15 bonus if the chain cites a formula
        """
        method_l = (method or "").lower()
        if method_l not in ("llm_inferred", "heuristic"):
            return 0.5
        if not reasoning:
            return 0.2
        n_steps = reasoning.count("Step ") + reasoning.count("\n")
        cites_formula = any(k in reasoning for k in
                            ["mu(T)", "sigma0", "rho_0", "gamma0_dot",
                             "*", "/", "exp", "log", "GPa", "MPa"])
        base = min(0.4 + 0.05 * n_steps, 0.8)
        if cites_formula:
            base = min(base + 0.15, 0.85)
        return float(base)

    def score(self, extractions, target_material, target_temp,
              target_strain_rate=1e-3, top_k=8):
        buckets: Dict[str, List[PlasticityCandidate]] = {p: [] for p in PARAM_ORDER}

        for ext in extractions:
            p = ext.get("param")
            if p not in buckets:
                continue
            try:
                v_si = _normalize_unit(ext["value"], ext.get("unit", ""), p)
            except Exception:
                continue
            v_si_clamped, was_clamped = _pl_clamp(v_si, p)

            s_mat = self._material_expert(ext.get("material", ""), target_material)
            s_temp = self._thermal_expert(ext.get("temp"), target_temp)
            s_strain = self._strain_expert(ext.get("strain_rate"), target_strain_rate)
            s_method = self._method_expert(ext.get("method", "unknown"))
            s_conf = float(ext.get("confidence", 0.5) or 0.5)
            s_reason = self._reasoning_expert(ext.get("reasoning", ""),
                                              ext.get("method", "unknown"))

            score = (self.w_material * s_mat
                     + self.w_thermal * s_temp
                     + self.w_strain * s_strain
                     + self.w_method * s_method
                     + self.w_confidence * s_conf
                     + self.w_reasoning * s_reason)

            buckets[p].append(PlasticityCandidate(
                param=p, value_si=v_si_clamped,
                raw_value=float(ext["value"]),
                raw_unit=str(ext.get("unit", "")),
                score=score, confidence=s_conf,
                material=str(ext.get("material", "")),
                temp_k=ext.get("temp"),
                strain_rate=ext.get("strain_rate"),
                method=str(ext.get("method", "unknown")),
                source_file=str(ext.get("_source_file", "")),
                source_title=str(ext.get("_source_title", "")),
                evidence=str(ext.get("evidence", "")),
                reasoning=str(ext.get("reasoning", "")),
                clamped=was_clamped,
            ))

        for p in buckets:
            buckets[p].sort(key=lambda c: c.score, reverse=True)
            buckets[p] = buckets[p][:top_k]
        return buckets


# ----------------------------------------------------------------------------
# LEARNED PER-MATERIAL PRIORS
# ----------------------------------------------------------------------------
class PlasticityMaterialPriorLearner:
    """Learns per-material priors from explicit corpus extractions.

    IMPORTANT: always uses `mode="strict_extract"` — no LLM inference is
    allowed in the prior-learning path, otherwise the "priors" become
    echoes of the LLM's own prompt baselines rather than real statistics.
    """

    def __init__(self, extractor: PlasticityParameterExtractor):
        self.extractor = extractor

    def learn(self, corpus, material: str = "?", temp_k: float = 300.0,
              strain_rate: float = 1e-3, use_llm: bool = False,
              max_docs: int = 200, debug_llm: bool = False) -> pd.DataFrame:
        raw: Dict[Tuple[str, str], List[float]] = {}
        for doc in corpus[:max_docs]:
            text = doc["text"]
            extractions = self.extractor.extract(
                text,
                material=material,
                temp_k=temp_k,
                strain_rate=strain_rate,
                use_llm=use_llm,
                mode="strict_extract",
                debug_llm=debug_llm,
            )
            for ext in extractions:
                mat = (ext.get("material") or "").strip() or "unknown"
                p = ext.get("param")
                if p not in PLASTICITY_ONTOLOGY:
                    continue
                try:
                    v_si = _normalize_unit(ext["value"], ext.get("unit", ""), p)
                except Exception:
                    continue
                v_si, _ = _pl_clamp(v_si, p)
                raw.setdefault((mat, p), []).append(v_si)

        rows: List[Dict[str, Any]] = []
        for (mat, p), values in raw.items():
            if len(values) == 0:
                continue
            arr = np.array(values, dtype=float)
            rows.append({
                "material": mat, "param": p, "n": int(arr.size),
                "mean": float(arr.mean()), "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=[
                "material", "param", "n", "mean", "median", "std", "p10", "p90"
            ])
        return df.sort_values(["param", "material"]).reset_index(drop=True)

    @staticmethod
    def suggest(priors_df, material, param) -> Optional[float]:
        if priors_df is None or priors_df.empty:
            return None
        sub = priors_df[
            (priors_df["material"].str.lower() == material.lower())
            & (priors_df["param"] == param)
        ]
        if sub.empty:
            return None
        row = sub.iloc[0]
        if row["n"] < 3:
            return float(row["median"])
        return float(0.5 * row["median"] + 0.5 * row["mean"])


# ----------------------------------------------------------------------------
# HISTOGRAM PLOTTER (side panel – keeps original compact style)
# ----------------------------------------------------------------------------
def render_plasticity_candidate_histograms(
    candidates_by_param: Dict[str, List[PlasticityCandidate]],
    log_scale_params: Optional[set] = None,
) -> None:
    log_scale_params = log_scale_params or {"rho0", "gamma0_dot"}

    available = [p for p in PARAM_ORDER if candidates_by_param.get(p)]
    if not available:
        st.info("No candidates to plot yet.")
        return

    n = len(available)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[PLASTICITY_ONTOLOGY[p]["label"] for p in available],
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )

    for idx, p in enumerate(available):
        r = idx // cols + 1
        c = idx % cols + 1
        spec = PLASTICITY_ONTOLOGY[p]
        ui_vals = [cd.value_si / spec["ui_scale"] for cd in candidates_by_param[p]]
        scores = [cd.score for cd in candidates_by_param[p]]

        fig.add_trace(
            go.Histogram(
                x=ui_vals,
                marker=dict(color="#3b82f6",
                            line=dict(color="#1e3a8a", width=1)),
                name=spec["symbol"],
                hovertemplate="%{x:.4g}<br>count: %{y}<extra></extra>",
                nbinsx=max(4, min(20, len(ui_vals) * 2)),
                showlegend=False,
            ),
            row=r, col=c,
        )
        top_idx = int(np.argmax(scores))
        fig.add_trace(
            go.Scatter(
                x=[ui_vals[top_idx]], y=[1],
                mode="markers",
                marker=dict(color="#ef4444", size=14, symbol="star",
                            line=dict(color="white", width=1)),
                name="⭐ Best",
                hovertemplate=f"⭐ Best = {ui_vals[top_idx]:.4g} "
                              f"{spec['ui_unit']}<br>score: {scores[top_idx]:.3f}"
                              f"<extra></extra>",
                showlegend=(idx == 0),
            ),
            row=r, col=c,
        )
        if p in log_scale_params and min(ui_vals) > 0:
            fig.update_xaxes(type="log", row=r, col=c)

    fig.update_layout(
        height=320 * rows,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#1e293b"),
        bargap=0.08,
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# RECOMMENDER ORCHESTRATOR
# ----------------------------------------------------------------------------
@dataclass
class PlasticityRecommendationBundle:
    material: str
    temp_k: float
    strain_rate: float
    candidates: Dict[str, List[PlasticityCandidate]]
    defaults: Dict[str, float]
    priors_df: pd.DataFrame
    retrieval_backend: str
    llm_used: bool
    timestamp: float = field(default_factory=time.time)

    def best(self, param: str) -> Optional[PlasticityCandidate]:
        lst = self.candidates.get(param, [])
        return lst[0] if lst else None

    def prior_suggestion(self, param: str) -> Optional[float]:
        return PlasticityMaterialPriorLearner.suggest(
            self.priors_df, self.material, param
        )


class PlasticityRecommender:
    """End-to-end: FAISS retrieve → LLM extract → LatentMoE rank → priors."""

    CACHE_DIR = ".plasticity_cache"
    LLM_CACHE_FILE = "llm_cache.json"

    def __init__(self, db_dir: str = "json_metadatabase",
                 ollama_model: str = "qwen2.5:7b",
                 use_llm: bool = True, top_k_retrieval: int = 20,
                 debug_llm: bool = False):
        self.corpus = PlasticityCorpus(db_dir)
        self.client = PlasticityOllamaClient(model=ollama_model)
        self.llm_available = use_llm and PlasticityOllamaClient.is_available()
        self.retriever = PlasticityFAISSRetriever()
        self.extractor = PlasticityParameterExtractor(
            self.client if self.llm_available else None,
            cache=self._load_disk_cache(),
        )
        self.scorer = PlasticityLatentMoEScorer()
        self.prior_learner = PlasticityMaterialPriorLearner(self.extractor)
        self.top_k_retrieval = top_k_retrieval
        self.debug_llm = debug_llm

    def _load_disk_cache(self) -> Dict[str, Any]:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            path = os.path.join(self.CACHE_DIR, self.LLM_CACHE_FILE)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
        return {}

    def _save_disk_cache(self) -> None:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            path = os.path.join(self.CACHE_DIR, self.LLM_CACHE_FILE)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.extractor.cache, f)
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def recommend(self, material: str, temp_k: float, strain_rate: float = 1e-3,
                  max_docs: int = 20, build_priors: bool = True,
                  progress_callback=None) -> PlasticityRecommendationBundle:
        corpus = self.corpus.load()
        if not corpus:
            st.warning("⚠️ No JSON metadatabases found. Using ontology defaults.")
            return self._default_bundle(material, temp_k, strain_rate)

        retrieval_backend = "keyword-prefilter"
        if not self.retriever._docs:
            try:
                self.retriever.build(corpus, force=False)
                if self.retriever._docs:
                    retrieval_backend = (
                        "faiss+dense" if FAISS_AVAILABLE and self.retriever._index is not None
                        else "numpy+dense" if self.retriever.model is not None
                        else "tfidf-fallback"
                    )
            except Exception as e:
                logger.warning("Retrieval build failed: %s", e)

        if self.retriever._docs:
            query = (f"{material} plasticity parameters: dislocation density, "
                     f"shear modulus, reference strain rate, strain-rate sensitivity, "
                     f"friction stress; temperature ~{temp_k} K.")
            docs = self.retriever.search(query,
                                         k=max(self.top_k_retrieval, max_docs),
                                         material_hint=material)
            docs = docs[:max_docs]
        else:
            docs = PlasticityCorpus.keyword_prefilter(corpus, material, k=max_docs)

        extractions: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if progress_callback:
                progress_callback(i + 1, len(docs), doc.get("title", "")[:60])
            # RUNTIME PATH → reasoned_inference (fills missing params with CoT)
            ext = self.extractor.extract(
                doc["text"], material, temp_k, strain_rate,
                use_llm=self.llm_available,
                mode="reasoned_inference",
                debug_llm=self.debug_llm,
            )
            for e in ext:
                e["_source_file"] = doc["source"]
                e["_source_title"] = doc["title"]
            extractions.extend(ext)
        self._save_disk_cache()

        candidates = self.scorer.score(extractions, material, temp_k,
                                       target_strain_rate=strain_rate, top_k=8)

        priors_df = pd.DataFrame()
        if build_priors:
            try:
                # PRIOR PATH → strict_extract (explicit values only, no echo)
                priors_df = self.prior_learner.learn(
                    corpus,
                    material=material,           # real target
                    temp_k=temp_k,
                    strain_rate=strain_rate,
                    use_llm=self.llm_available,
                    max_docs=min(150, len(corpus)),
                    debug_llm=self.debug_llm,
                )
            except Exception as e:
                logger.warning("Prior learning failed: %s", e)

        defaults = {
            p: PLASTICITY_ONTOLOGY[p]["defaults"].get(
                material, PLASTICITY_ONTOLOGY[p]["defaults"]["Cu"]
            )
            for p in PARAM_ORDER
        }
        return PlasticityRecommendationBundle(
            material=material, temp_k=temp_k, strain_rate=strain_rate,
            candidates=candidates, defaults=defaults, priors_df=priors_df,
            retrieval_backend=retrieval_backend, llm_used=self.llm_available,
        )

    @staticmethod
    def _default_bundle(material, temp_k, strain_rate):
        defaults = {
            p: PLASTICITY_ONTOLOGY[p]["defaults"].get(
                material, PLASTICITY_ONTOLOGY[p]["defaults"]["Cu"]
            )
            for p in PARAM_ORDER
        }
        return PlasticityRecommendationBundle(
            material=material, temp_k=temp_k, strain_rate=strain_rate,
            candidates={p: [] for p in PARAM_ORDER}, defaults=defaults,
            priors_df=pd.DataFrame(),
            retrieval_backend="none", llm_used=False,
        )


# ============================================================================
# OLLAMA MODEL REGISTRY
# ============================================================================
OLLAMA_MODELS: Dict[str, str] = {
    "⚡ Fallback (Rule-based, no LLM)": "",
    "🦙 qwen2.5:0.5b (Fastest, CPU OK)": "qwen2.5:0.5b",
    "🦙 qwen2.5:1.5b (Balanced)": "qwen2.5:1.5b",
    "🦙 qwen2.5:7b (Recommended for RAG)": "qwen2.5:7b",
    "🦙 qwen2.5:14b (Max Reasoning)": "qwen2.5:14b",
    "🦙 llama3.1:8b (Meta Standard)": "llama3.1:8b",
    "🦙 mistral:7b (High JSON Reliability)": "mistral:7b",
    "🦙 gemma2:9b (Scientific Nuance)": "gemma2:9b",
    "🦙 falcon3:10b (Instruction Following)": "falcon3:10b",
}


def _build_ollama_dropdown_options() -> Dict[str, str]:
    """Return the ordered display-name → model-name mapping used by the
    sidebar dropdown.  Starts from the static OLLAMA_MODELS registry and
    appends any models that are actually installed on the local Ollama
    instance (if it is reachable).  Preserves order and de-duplicates."""
    options: Dict[str, str] = dict(OLLAMA_MODELS)  # preserve order
    installed = PlasticityOllamaClient.list_models()
    existing_names = set(options.values())
    for name in installed:
        if not name or name in existing_names:
            continue
        options[f"🦙 {name} (Installed locally)"] = name
        existing_names.add(name)
    return options


# ----------------------------------------------------------------------------
# FULL CACHE PURGE HELPER (used by the "Force reload corpus" button)
# ----------------------------------------------------------------------------
def purge_plasticity_caches() -> List[str]:
    """Delete every known plasticity cache.

    Returns a list of human-readable strings describing what was purged,
    so the UI can show a precise message."""
    purged: List[str] = []

    # 1) Streamlit session state caches
    corpus_key = f"pl_corpus_{PlasticityCorpus._CACHE_VERSION}"
    if corpus_key in st.session_state:
        st.session_state.pop(corpus_key, None)
        purged.append("session corpus cache")

    # also drop any stray older corpus keys
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("pl_corpus_") and k != corpus_key:
            st.session_state.pop(k, None)
            purged.append(f"session key '{k}'")

    # 2) In-memory extractor cache, if a recommender is still alive in session
    live_rec = st.session_state.get("_plr_live_recommender")
    if live_rec is not None:
        try:
            if getattr(live_rec, "extractor", None) is not None:
                live_rec.extractor.cache.clear()
                purged.append("in-memory LLM extractor cache")
            if getattr(live_rec, "retriever", None) is not None:
                live_rec.retriever._docs = []
                live_rec.retriever._doc_vectors = None
                live_rec.retriever._index = None
                purged.append("in-memory FAISS retriever state")
        except Exception as e:
            logger.warning("Could not purge live recommender caches: %s", e)

    # 3) On-disk caches
    cache_dir = PlasticityFAISSRetriever.CACHE_DIR
    if os.path.isdir(cache_dir):
        idx_path = os.path.join(cache_dir, PlasticityFAISSRetriever.INDEX_FILE)
        if os.path.exists(idx_path):
            try:
                os.remove(idx_path)
                purged.append(PlasticityFAISSRetriever.INDEX_FILE)
            except Exception as e:
                logger.warning("Could not remove %s: %s", idx_path, e)

        llm_cache_path = os.path.join(
            cache_dir, PlasticityRecommender.LLM_CACHE_FILE
        )
        if os.path.exists(llm_cache_path):
            try:
                os.remove(llm_cache_path)
                purged.append(PlasticityRecommender.LLM_CACHE_FILE)
            except Exception as e:
                logger.warning("Could not remove %s: %s", llm_cache_path, e)

        try:
            for fname in os.listdir(cache_dir):
                if not fname.lower().endswith((".json", ".pkl", ".faiss")):
                    continue
                if fname in (PlasticityFAISSRetriever.INDEX_FILE,
                             PlasticityRecommender.LLM_CACHE_FILE):
                    continue
                fpath = os.path.join(cache_dir, fname)
                try:
                    os.remove(fpath)
                    purged.append(fname)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Cache dir sweep failed: %s", e)

    return purged


# ----------------------------------------------------------------------------
# SIDEBAR UI
# ----------------------------------------------------------------------------
_PLR = "pl_rec_"


def _plr_get(key, default=None):
    return st.session_state.get(_PLR + key, default)


def _plr_set(key, value):
    st.session_state[_PLR + key] = value


def _plr_reset():
    for p in PARAM_ORDER:
        st.session_state.pop(f"{_PLR}{p}_choice", None)
        st.session_state.pop(f"{_PLR}{p}_manual", None)
        st.session_state.pop(f"{_PLR}{p}_value_si", None)
    st.session_state.pop(f"{_PLR}bundle", None)
    st.session_state.pop("plasticity_overrides", None)
    st.session_state.pop("_plr_live_recommender", None)


def _plr_render_parameter_selector(param: str,
                                   bundle: PlasticityRecommendationBundle):
    spec = PLASTICITY_ONTOLOGY[param]
    st.markdown(f"**{spec['symbol']} — {spec['label']}**")

    options: List[Tuple[str, float, Optional[PlasticityCandidate]]] = []

    best = bundle.best(param)
    if best is not None:
        options.append((
            f"⭐ Recommended: {_pl_fmt(param, best.value_si)} "
            f"(score {best.score:.2f})",
            best.value_si, best,
        ))

    for i, c in enumerate(bundle.candidates.get(param, [])[1:], start=1):
        options.append((
            f"   Alt {i}: {_pl_fmt(param, c.value_si)} (score {c.score:.2f})",
            c.value_si, c,
        ))

    prior = bundle.prior_suggestion(param)
    if prior is not None and (best is None or abs(prior - best.value_si) > 1e-12):
        options.append((
            f"📚 Learned prior ({bundle.material}): {_pl_fmt(param, prior)}",
            prior, None,
        ))

    default_si = bundle.defaults[param]
    options.append((
        f"⚙️  Default ({bundle.material}): {_pl_fmt(param, default_si)}",
        default_si, None,
    ))
    options.append(("✏️  Manual override…", float("nan"), None))

    labels = [o[0] for o in options]
    prev = _plr_get(f"{param}_choice", labels[0])
    if prev not in labels:
        prev = labels[0]
    idx = labels.index(prev)

    choice = st.radio(
        label=f"Select {param}",
        options=labels,
        index=idx,
        key=f"{_PLR}{param}_radio",
        label_visibility="collapsed",
    )
    _plr_set(f"{param}_choice", choice)

    sel = labels.index(choice)
    _, value_si, chosen = options[sel]

    if math.isnan(value_si):
        ui_default = default_si / spec["ui_scale"]
        manual = st.number_input(
            f"Manual {param} ({spec['ui_unit']})",
            value=float(_plr_get(f"{param}_manual", ui_default)),
            format="%.6g" if param in ("rho0", "gamma0_dot") else "%.4f",
            key=f"{_PLR}{param}_manual_input",
        )
        _plr_set(f"{param}_manual", manual)
        value_si = manual * spec["ui_scale"]
        chosen = None

    if chosen is not None:
        st.caption(
            f"📚 {chosen.source_file} — {chosen.source_title[:70]} | "
            f"mat={chosen.material or 'n/a'}, T={chosen.temp_k}, "
            f"method={chosen.method}, conf={chosen.confidence:.2f}"
            + (" | ⚠️ clamped" if chosen.clamped else "")
        )
        if chosen.evidence:
            st.markdown("**📚 Evidence snippet**")
            with st.container():
                st.code(chosen.evidence, language="text")
        if chosen.reasoning:
            with st.expander("🧠 LLM reasoning chain", expanded=False):
                st.markdown(chosen.reasoning)

    st.session_state[f"{_PLR}{param}_value_si"] = value_si
    st.markdown("---")


def render_plasticity_recommender_sidebar(
    default_material: str = "Cu",
    default_temp: float = 300.0,
    default_strain_rate: float = 1e-3,
    ollama_model: str = "qwen2.5:7b",
):
    """Sidebar with retrieval + LatentMoE + priors + histograms."""
    st.subheader("🤖 Intelligent Plasticity Recommender v8.3.0")
    st.caption(
        "FAISS + SentenceTransformer retrieval · Ollama NER · LatentMoE scoring · "
        "dual-mode prompts (strict priors vs reasoned inference) · "
        "param-alias canonicalization (v8.3.0 gatekeeper fix) · "
        "Chain-of-Thought surfaced in UI."
    )

    col1, col2 = st.columns(2)
    with col1:
        material = st.text_input(
            "Target material", value=default_material, key=f"{_PLR}material"
        )
    with col2:
        temp_k = st.number_input(
            "Temperature (K)", value=float(default_temp),
            min_value=1.0, step=10.0, key=f"{_PLR}temp",
        )
    strain_rate = st.number_input(
        "Reference strain rate (s⁻¹)",
        value=float(default_strain_rate), format="%.2e", key=f"{_PLR}rate",
    )

    model_options = _build_ollama_dropdown_options()

    current_model_name = ollama_model if ollama_model else "qwen2.5:7b"
    display_options = list(model_options.keys())

    display_name_for_current = next(
        (k for k, v in model_options.items() if v == current_model_name),
        next(
            (k for k in display_options if "Recommended" in k),
            display_options[0] if display_options else "",
        ),
    )
    default_idx = (
        display_options.index(display_name_for_current)
        if display_name_for_current in display_options
        else 0
    )

    selected_display = st.selectbox(
        "Ollama model",
        options=display_options,
        index=default_idx,
        key=f"{_PLR}ollama_model_select",
        help=(
            "Pick '⚡ Fallback' to use the built-in heuristic extractor "
            "(no LLM). Locally-installed models detected via Ollama's /api/tags "
            "are appended automatically."
        ),
    )
    ollama_model = model_options.get(selected_display, "")

    if not ollama_model:
        llm_ok = False
    else:
        llm_ok = PlasticityOllamaClient.is_available()

    backend_txt = (
        "faiss+dense" if FAISS_AVAILABLE and SBERT_AVAILABLE
        else "numpy+dense" if SBERT_AVAILABLE
        else "tfidf-fallback"
    )
    bc1, bc2 = st.columns(2)
    with bc1:
        if not ollama_model:
            st.caption("⚡ LLM disabled — using heuristic extractor")
        else:
            st.caption(f"{'✅' if llm_ok else '⚠️'} Ollama "
                       f"{'available' if llm_ok else 'unreachable'}")
    with bc2:
        st.caption(f"🔎 Retrieval: `{backend_txt}`")

    # ---- NEW v8.3.0: debug toggle ----
    debug_llm = st.checkbox(
        "🔍 Show raw LLM responses (logs to terminal at INFO level)",
        value=False,
        key=f"{_PLR}debug_llm",
        help=("Enable this to see exactly what the LLM emits before the "
              "alias-canonicalization / value-coercion filters run. Useful "
              "for diagnosing future gatekeeper bugs."),
    )
    # ----------------------------------

    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        run_btn = st.button("🔍 Analyse JSON databases",
                            use_container_width=True, type="primary")
    with btn2:
        refresh_btn = st.button("🔄 Force reload corpus",
                                use_container_width=True)
    with btn3:
        if st.button("♻️ Reset recommendations",
                     use_container_width=True):
            _plr_reset()
            st.rerun()

    if refresh_btn:
        try:
            purged_items = purge_plasticity_caches()
            st.session_state.pop("_plr_live_recommender", None)
        except Exception as e:
            st.warning(f"Could not clear all caches: {e}")
            purged_items = []

        if purged_items:
            st.success(
                "✅ Purged: " + ", ".join(f"`{p}`" for p in purged_items)
                + ". Next **Analyse** will run fresh (LLM will be re-queried "
                  "for every parameter and document)."
            )
        else:
            st.info(
                "ℹ️ Nothing to purge — caches were already empty. "
                "Next **Analyse** will run fresh."
            )

    if run_btn:
        recommender = PlasticityRecommender(
            ollama_model=ollama_model or "qwen2.5:7b", use_llm=llm_ok,
            debug_llm=debug_llm,
        )
        st.session_state["_plr_live_recommender"] = recommender

        progress = st.progress(0.0)
        status = st.empty()

        def _cb(i, n, title):
            progress.progress(i / max(n, 1))
            status.caption(f"Scanning {i}/{n}: {title}…")

        try:
            with st.spinner("Retrieving candidates + running LatentMoE…"):
                bundle = recommender.recommend(
                    material=material, temp_k=float(temp_k),
                    strain_rate=float(strain_rate),
                    progress_callback=_cb,
                )
            _plr_set("bundle", bundle)
            progress.empty()
            status.success(
                f"✅ Retrieved "
                f"{sum(len(v) for v in bundle.candidates.values())} candidates "
                f"across "
                f"{sum(1 for v in bundle.candidates.values() if v)} params."
            )
        except Exception as e:
            progress.empty()
            status.error(f"Recommendation failed: {e}")
            st.exception(e)

    bundle: Optional[PlasticityRecommendationBundle] = _plr_get("bundle")
    if bundle is None:
        return

    st.caption(
        f"Retrieval: **{bundle.retrieval_backend}** · "
        f"LLM: **{'yes' if bundle.llm_used else 'no (heuristic)'}**"
    )

    st.markdown("### Choose values (one by one)")
    for param in PARAM_ORDER:
        _plr_render_parameter_selector(param, bundle)

    if st.button("✅ Apply selected values to solver",
                 type="primary", use_container_width=True):
        overrides = {}
        for param in PARAM_ORDER:
            v = st.session_state.get(f"{_PLR}{param}_value_si")
            if v is not None:
                overrides[SOLVER_KEY_MAP[param]] = float(v)
        st.session_state["plasticity_overrides"] = overrides
        st.success(f"Applied {len(overrides)} parameters. "
                   "The solver will use them on the next run.")

    st.markdown("### 📊 Candidate Distributions (compact)")
    render_plasticity_candidate_histograms(bundle.candidates)

    if not bundle.priors_df.empty:
        st.markdown("**📚 Learned per‑material priors (from corpus)**")
        with st.container():
            st.caption(
                "Aggregated from **strict explicit extractions only** in the "
                "corpus. The LLM is NOT allowed to infer during prior "
                "learning — so these statistics reflect what the literature "
                "actually reports, not what the LLM guesses."
            )
            styled = bundle.priors_df.copy()
            for col in ["mean", "median", "std", "p10", "p90"]:
                if col in styled.columns:
                    styled[col] = styled.apply(
                        lambda r, c=col: f"{r[c]:.3e}", axis=1
                    )
            st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("**📋 Full candidate audit (LatentMoE ranking)**")
    with st.container():
        for param in PARAM_ORDER:
            st.markdown(f"**{PLASTICITY_ONTOLOGY[param]['label']}**")
            cands = bundle.candidates.get(param, [])
            if not cands:
                st.caption("No candidates found — using default.")
                continue
            rows = [c.to_display() for c in cands]
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# SOLVER HOOK
# ----------------------------------------------------------------------------
def apply_plasticity_overrides(solver) -> None:
    """Merge user-chosen plasticity values into solver.mat_props['plasticity']."""
    try:
        overrides = st.session_state.get("plasticity_overrides", None)
        if not overrides:
            return
        if 'plasticity' not in solver.mat_props:
            solver.mat_props['plasticity'] = {}
        solver.mat_props['plasticity'].update(overrides)
        logger.info("Applied plasticity overrides: %s", overrides)
    except Exception as e:
        logger.warning("Could not apply plasticity overrides: %s", e)


# ============================================================================
# ███████████████████████████████████████████████████████████████████████████
# ███  PUBLICATION-QUALITY AI RECOMMENDER VISUALS DASHBOARD            ██████
# ███████████████████████████████████████████████████████████████████████████
# ============================================================================

@dataclass
class RecommenderVisualStyle:
    """Full styling state for the publication-quality recommender visuals."""
    journal: str = 'nature'
    font_family: str = 'Arial'
    font_size_title: float = 12.0
    font_size_axis: float = 10.0
    font_size_tick: float = 9.0
    font_size_legend: float = 9.0
    font_size_annotation: float = 8.0
    font_weight: str = 'bold'

    figure_width: float = 10.0
    figure_height: float = 6.0
    dpi: int = 300
    line_width: float = 1.5
    marker_size: float = 8.0

    colormap: str = 'turbo'
    cmap_reverse: bool = False
    categorical_palette: str = 'okabe_ito'

    background_color: str = '#ffffff'
    plot_background_color: str = '#f8f9fa'
    grid: bool = True
    grid_alpha: float = 0.3
    grid_linestyle: str = '--'
    transparent_bg: bool = False
    spine_color: str = '#333333'
    spine_width: float = 1.0
    show_values: bool = True
    legend_position: str = 'best'
    title_override: str = ''
    xlabel_override: str = ''
    ylabel_override: str = ''
    show_title: bool = True
    show_legend: bool = True
    show_colorbar: bool = True
    colorbar_orientation: str = 'vertical'

    def get_cmap(self):
        base = plt.get_cmap(self.colormap)
        return base.reversed() if self.cmap_reverse else base

    def get_palette(self):
        return PUBLICATION_PALETTES.get(
            self.categorical_palette,
            PUBLICATION_PALETTES['okabe_ito']
        )

    def apply_matplotlib(self):
        """Push style to global rcParams for Matplotlib figures."""
        rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_size_axis,
            'axes.titlesize': self.font_size_title,
            'axes.labelsize': self.font_size_axis,
            'xtick.labelsize': self.font_size_tick,
            'ytick.labelsize': self.font_size_tick,
            'legend.fontsize': self.font_size_legend,
            'figure.titlesize': self.font_size_title,
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'axes.grid': self.grid,
            'grid.alpha': self.grid_alpha,
            'grid.linestyle': self.grid_linestyle,
            'axes.linewidth': self.spine_width,
            'axes.edgecolor': self.spine_color,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.transparent': self.transparent_bg,
            'axes.prop_cycle': plt.cycler(color=self.get_palette()),
        })

    def apply_plotly_layout(self, fig, title: str = "", xlabel: str = "", ylabel: str = ""):
        fig.update_layout(
            title=(self.title_override or title) if self.show_title else "",
            xaxis_title=self.xlabel_override or xlabel,
            yaxis_title=self.ylabel_override or ylabel,
            font=dict(
                family=self.font_family,
                size=self.font_size_axis,
                color='#1e293b',
            ),
            title_font=dict(
                family=self.font_family,
                size=self.font_size_title + 2,
                color='#0f172a',
            ),
            plot_bgcolor=self.plot_background_color,
            paper_bgcolor=self.background_color,
            showlegend=self.show_legend,
            legend=dict(
                font=dict(
                    family=self.font_family,
                    size=self.font_size_legend,
                ),
                bordercolor=self.spine_color,
                borderwidth=1,
            ),
            margin=dict(l=70, r=40, t=80, b=60),
        )
        if self.grid:
            fig.update_xaxes(showgrid=True,
                             gridcolor='rgba(128,128,128,0.3)',
                             gridwidth=0.5,
                             griddash='dash')
            fig.update_yaxes(showgrid=True,
                             gridcolor='rgba(128,128,128,0.3)',
                             gridwidth=0.5,
                             griddash='dash')
        else:
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
        return fig


def _apply_matplotlib_style_to_axes(ax, style: RecommenderVisualStyle):
    """Apply fine-grained Matplotlib styling to a single axes object."""
    ax.set_facecolor(style.plot_background_color)
    for spine in ax.spines.values():
        spine.set_color(style.spine_color)
        spine.set_linewidth(style.spine_width)
    ax.tick_params(axis='both', which='major',
                   labelsize=style.font_size_tick,
                   colors=style.spine_color,
                   width=style.spine_width)
    ax.tick_params(axis='both', which='minor',
                   labelsize=style.font_size_tick - 1,
                   colors=style.spine_color,
                   width=style.spine_width * 0.7)
    if style.grid:
        ax.grid(True, alpha=style.grid_alpha,
                linestyle=style.grid_linestyle, linewidth=0.6)
    else:
        ax.grid(False)
    ax.xaxis.label.set_size(style.font_size_axis)
    ax.yaxis.label.set_size(style.font_size_axis)
    ax.title.set_size(style.font_size_title)
    ax.title.set_weight(style.font_weight)


def _normalize_for_radar(param: str, value: float) -> float:
    """Normalize a parameter value onto [0, 1] using log or linear mapping."""
    lo, hi = PLASTICITY_ONTOLOGY[param]["soft_range"]
    if hi == lo:
        return 0.5
    if param in ("rho0", "gamma0_dot", "sigma0", "mu"):
        if value <= 0:
            return 0.0
        l_lo = np.log10(max(lo, 1e-10))
        l_hi = np.log10(max(hi, 1e-9))
        l_val = np.log10(max(value, 1e-10))
        return float(np.clip((l_val - l_lo) / (l_hi - l_lo), 0, 1))
    else:
        return float(np.clip((value - lo) / (hi - lo), 0, 1))


# ----------------------------------------------------------------------------
# RADAR CHART — publication quality
# ----------------------------------------------------------------------------
def render_recommender_radar_pub(bundle: PlasticityRecommendationBundle,
                                 style: RecommenderVisualStyle) -> Optional[bytes]:
    categories = [PLASTICITY_ONTOLOGY[p]["symbol"] for p in PARAM_ORDER]

    defaults_norm = [_normalize_for_radar(p, bundle.defaults[p]) for p in PARAM_ORDER]
    bests_norm, best_values = [], []
    for p in PARAM_ORDER:
        best = bundle.best(p)
        if best:
            bests_norm.append(_normalize_for_radar(p, best.value_si))
            best_values.append(best.value_si)
        else:
            defaults_norm_v = _normalize_for_radar(p, bundle.defaults[p])
            bests_norm.append(defaults_norm_v)
            best_values.append(bundle.defaults[p])

    prior_norm = []
    for p in PARAM_ORDER:
        prior = bundle.prior_suggestion(p)
        if prior is not None:
            prior_norm.append(_normalize_for_radar(p, prior))
        else:
            prior_norm.append(defaults_norm[PARAM_ORDER.index(p)])

    fig = go.Figure()
    cmap = style.get_cmap()
    color_best = cmap(0.85)
    color_best_str = f'rgb({int(color_best[0]*255)},{int(color_best[1]*255)},{int(color_best[2]*255)})'
    color_prior = cmap(0.55)
    color_prior_str = f'rgb({int(color_prior[0]*255)},{int(color_prior[1]*255)},{int(color_prior[2]*255)})'

    fig.add_trace(go.Scatterpolar(
        r=defaults_norm + [defaults_norm[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Ontology Defaults',
        line=dict(color='#94a3b8', dash='dot',
                  width=style.line_width),
        opacity=0.55,
        hovertemplate='%{theta}: %{r:.3f}<extra></extra>',
    ))

    fig.add_trace(go.Scatterpolar(
        r=prior_norm + [prior_norm[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f'Learned Prior ({bundle.material})',
        line=dict(color=color_prior_str, dash='dash',
                  width=style.line_width + 0.5),
        opacity=0.65,
        hovertemplate='%{theta}: %{r:.3f}<extra></extra>',
    ))

    fig.add_trace(go.Scatterpolar(
        r=bests_norm + [bests_norm[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='⭐ AI Recommended',
        line=dict(color=color_best_str, width=style.line_width + 2),
        marker=dict(size=style.marker_size,
                    color=color_best_str,
                    line=dict(color='white', width=1.5)),
        hovertemplate='%{theta}: %{r:.3f}<br>'
                      'raw: %{customdata}<extra></extra>',
        customdata=[_pl_fmt(p, v) for p, v in zip(PARAM_ORDER, best_values)] +
                   [_pl_fmt(PARAM_ORDER[0], best_values[0])],
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(size=style.font_size_tick,
                              family=style.font_family),
                gridcolor='rgba(128,128,128,0.35)',
                linecolor=style.spine_color,
            ),
            angularaxis=dict(
                tickfont=dict(size=style.font_size_axis,
                              family=style.font_family),
                linecolor=style.spine_color,
                gridcolor='rgba(128,128,128,0.35)',
            ),
            bgcolor=style.plot_background_color,
        ),
        height=int(style.figure_height * 90),
        width=int(style.figure_width * 90),
    )
    style.apply_plotly_layout(fig,
                              title=f"AI Recommendation vs Defaults — {bundle.material}")
    st.plotly_chart(fig, use_container_width=True)

    if style.show_values:
        with st.expander("📊 Normalized radar values", expanded=False):
            df = pd.DataFrame({
                'Parameter': categories,
                'Default (norm)': defaults_norm,
                'Prior (norm)': prior_norm,
                'AI (norm)': bests_norm,
                'AI (actual)': [_pl_fmt(p, v) for p, v in zip(PARAM_ORDER, best_values)],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    return _fig_to_bytes_plotly(fig)


# ----------------------------------------------------------------------------
# BAR CHART — publication quality
# ----------------------------------------------------------------------------
def render_recommender_bars_pub(bundle: PlasticityRecommendationBundle,
                                style: RecommenderVisualStyle) -> Optional[bytes]:
    selected_param = st.selectbox(
        "Select Parameter for Bar Chart",
        PARAM_ORDER,
        format_func=lambda p: f"{PLASTICITY_ONTOLOGY[p]['symbol']} — "
                              f"{PLASTICITY_ONTOLOGY[p]['label']}",
        key="rec_bar_param_pub",
    )
    cands = bundle.candidates.get(selected_param, [])
    if not cands:
        st.warning("No candidates for this parameter.")
        return None

    spec = PLASTICITY_ONTOLOGY[selected_param]

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_height),
                           dpi=style.dpi)
    fig.patch.set_facecolor(style.background_color)
    if style.transparent_bg:
        fig.patch.set_alpha(0.0)

    labels = [f"{_pl_fmt(selected_param, c.value_si)}\n({c.method})" for c in cands]
    scores = [c.score for c in cands]
    x = np.arange(len(labels))
    cmap = style.get_cmap()
    colors = [cmap(float(np.clip(s, 0, 1))) for s in scores]

    bars = ax.bar(x, scores, color=colors,
                  edgecolor=style.spine_color,
                  linewidth=style.spine_width,
                  width=0.7)

    if scores:
        bars[0].set_edgecolor('#f59e0b')
        bars[0].set_linewidth(style.spine_width + 2.0)
        bars[0].set_label('⭐ Best Match')
        if style.show_values:
            ax.annotate(f"⭐ {scores[0]:.3f}",
                        xy=(x[0], scores[0]),
                        xytext=(x[0], scores[0] + 0.03),
                        ha='center', va='bottom',
                        fontsize=style.font_size_annotation,
                        fontweight='bold',
                        color='#92400e')

    if style.show_values:
        for xi, s in zip(x, scores):
            ax.text(xi, s + 0.008, f"{s:.3f}",
                    ha='center', va='bottom',
                    fontsize=style.font_size_annotation,
                    color=style.spine_color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right',
                       fontsize=style.font_size_tick)
    ax.set_ylabel("LatentMoE Score",
                  fontsize=style.font_size_axis,
                  fontweight=style.font_weight)
    if style.show_title:
        ax.set_title(style.title_override or
                     f"Candidate Scores — {spec['label']} [{spec['symbol']}]",
                     fontsize=style.font_size_title,
                     fontweight=style.font_weight)
    if style.show_legend:
        ax.legend(loc=style.legend_position,
                  fontsize=style.font_size_legend, frameon=True)
    _apply_matplotlib_style_to_axes(ax, style)
    plt.tight_layout()

    st.pyplot(fig)

    buf = _fig_to_bytes_matplotlib(fig)
    plt.close(fig)
    return buf


# ----------------------------------------------------------------------------
# SANKEY DIAGRAM — publication quality
# ----------------------------------------------------------------------------
def render_recommender_sankey_pub(bundle: PlasticityRecommendationBundle,
                                  style: RecommenderVisualStyle) -> Optional[bytes]:
    sources = set()
    for p in PARAM_ORDER:
        for c in bundle.candidates.get(p, []):
            if c.source_file:
                sources.add(c.source_file)

    source_list = sorted(list(sources)) if sources else ["(no sources)"]
    param_list = PARAM_ORDER
    selected_list = [f"✅ {p}" for p in PARAM_ORDER]

    all_nodes = source_list + param_list + selected_list
    node_indices = {n: i for i, n in enumerate(all_nodes)}

    cmap = style.get_cmap()
    palette = style.get_palette()

    links = []
    for p in PARAM_ORDER:
        for c in bundle.candidates.get(p, []):
            if c.source_file and c.source_file in node_indices:
                links.append({
                    "source": node_indices[c.source_file],
                    "target": node_indices[p],
                    "value": max(c.score, 0.01),
                    "color": "rgba(100,200,255,0.35)",
                })

    for p in param_list:
        best = bundle.best(p)
        val = best.value_si if best else bundle.defaults[p]
        norm_val = _normalize_for_radar(p, val)
        rgba = cmap(norm_val)
        color = (f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},'
                 f'{int(rgba[2]*255)},0.65)')
        links.append({
            "source": node_indices[p],
            "target": node_indices[f"✅ {p}"],
            "value": 1.0,
            "color": color,
        })

    node_colors = []
    for n in all_nodes:
        if n in source_list:
            node_colors.append("#a5b4fc")
        elif n in param_list:
            node_colors.append("#fbbf24")
        else:
            node_colors.append("#34d399")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18, thickness=22,
            line=dict(color=style.spine_color, width=0.8),
            label=all_nodes,
            color=node_colors,
            hovertemplate='%{label}<br>Total flow: %{value:.2f}<extra></extra>',
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            color=[l["color"] for l in links],
            hovertemplate='%{source.label} → %{target.label}<br>'
                          'weight: %{value:.3f}<extra></extra>',
        ),
    )])

    style.apply_plotly_layout(fig,
                              title="Plasticity Parameter Source Flow (Sankey)")
    fig.update_layout(
        height=int(style.figure_height * 100),
        width=int(style.figure_width * 100),
    )
    st.plotly_chart(fig, use_container_width=True)
    return _fig_to_bytes_plotly(fig)


# ----------------------------------------------------------------------------
# TREEMAP — publication quality
# ----------------------------------------------------------------------------
def render_recommender_treemap_pub(bundle: PlasticityRecommendationBundle,
                                   style: RecommenderVisualStyle) -> Optional[bytes]:
    ids, labels, parents, values, colors = [], [], [], [], []
    cmap = style.get_cmap()
    palette = style.get_palette()

    for p_idx, p in enumerate(PARAM_ORDER):
        spec = PLASTICITY_ONTOLOGY[p]
        ids.append(p)
        labels.append(f"<b>{spec['symbol']}</b><br>{spec['label']}")
        parents.append("")
        values.append(0)
        colors.append(palette[p_idx % len(palette)])

        cands = bundle.candidates.get(p, [])
        for j, c in enumerate(cands):
            c_id = f"{p}_{j}_{_pl_hash(c.source_file)[:6]}"
            ids.append(c_id)
            val_str = _pl_fmt(p, c.value_si)
            labels.append(f"{val_str}<br><i>{c.method}</i>"
                          f"<br>score={c.score:.2f}")
            parents.append(p)
            values.append(max(c.score, 0.01))
            rgba = cmap(float(np.clip(c.score, 0, 1)))
            colors.append(
                f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
            )

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors,
            line=dict(color=style.spine_color, width=1.0),
            pad=dict(t=2, l=2, r=2, b=2),
        ),
        textinfo="label+value",
        textfont=dict(
            family=style.font_family,
            size=style.font_size_tick,
            color='white',
        ),
        hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>',
    ))
    style.apply_plotly_layout(fig,
                              title="Candidate Hierarchy Treemap (sized by score)")
    fig.update_layout(
        height=int(style.figure_height * 100),
        width=int(style.figure_width * 100),
    )
    st.plotly_chart(fig, use_container_width=True)
    return _fig_to_bytes_plotly(fig)


# ----------------------------------------------------------------------------
# PUBLICATION HISTOGRAMS (Matplotlib) — one per parameter
# ----------------------------------------------------------------------------
def render_recommender_histograms_pub(
    bundle: PlasticityRecommendationBundle,
    style: RecommenderVisualStyle,
) -> Optional[bytes]:
    available = [p for p in PARAM_ORDER if bundle.candidates.get(p)]
    if not available:
        st.info("No candidates to plot yet.")
        return None

    n = len(available)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(style.figure_width * cols / 3,
                                      style.figure_height * rows / 2),
                             dpi=style.dpi)
    fig.patch.set_facecolor(style.background_color)
    if style.transparent_bg:
        fig.patch.set_alpha(0.0)

    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    cmap = style.get_cmap()
    palette = style.get_palette()

    for idx, p in enumerate(available):
        ax = axes[idx]
        spec = PLASTICITY_ONTOLOGY[p]
        cands = bundle.candidates[p]
        ui_vals = [c.value_si / spec["ui_scale"] for c in cands]
        scores = [c.score for c in cands]

        use_log = p in ("rho0", "gamma0_dot") and min(ui_vals) > 0
        if use_log:
            bins = np.logspace(np.log10(min(ui_vals)),
                               np.log10(max(ui_vals)),
                               num=min(15, max(5, len(ui_vals))), base=10)
        else:
            bins = min(15, max(5, len(ui_vals)))

        weights = np.array(scores)
        n_hist, edges, patches = ax.hist(
            ui_vals, bins=bins, weights=weights,
            color=palette[idx % len(palette)],
            edgecolor=style.spine_color,
            linewidth=style.spine_width,
            alpha=0.75,
        )

        vmin, vmax = (np.log10(max(edges[0], 1e-10)),
                      np.log10(max(edges[-1], 1e-10))) if use_log else \
                     (edges[0], edges[-1])
        for patch, edge in zip(patches, edges[:-1]):
            if use_log:
                e = np.log10(max(edge, 1e-10))
                norm_val = (e - vmin) / max(vmax - vmin, 1e-9)
            else:
                norm_val = (edge - vmin) / max(vmax - vmin, 1e-9)
            patch.set_facecolor(cmap(np.clip(norm_val, 0, 1)))

        best = bundle.best(p)
        if best is not None:
            ax.axvline(best.value_si / spec["ui_scale"],
                       color='#dc2626',
                       linestyle='--',
                       linewidth=style.line_width + 0.5,
                       label=f"⭐ {_pl_fmt(p, best.value_si)}",
                       zorder=5)

        if style.grid:
            ax.grid(True, alpha=style.grid_alpha,
                    linestyle=style.grid_linestyle, axis='y')

        if use_log:
            ax.set_xscale('log')

        ax.set_xlabel(f"{spec['label']} [{spec['ui_unit']}]",
                      fontsize=style.font_size_axis - 1,
                      fontweight=style.font_weight)
        ax.set_ylabel("Σ Score",
                      fontsize=style.font_size_axis - 1,
                      fontweight=style.font_weight)
        ax.set_title(f"{spec['symbol']} — {spec['label']}",
                     fontsize=style.font_size_title - 1,
                     fontweight=style.font_weight)
        if style.show_legend and best is not None:
            ax.legend(loc=style.legend_position,
                      fontsize=style.font_size_legend - 1, frameon=True)
        _apply_matplotlib_style_to_axes(ax, style)

    for idx in range(len(available), len(axes)):
        axes[idx].axis('off')

    if style.show_title:
        fig.suptitle(style.title_override or
                     f"Candidate Distributions — Target: {bundle.material} "
                     f"@ {bundle.temp_k:.0f} K, γ̇={bundle.strain_rate:.1e} s⁻¹",
                     fontsize=style.font_size_title + 1,
                     fontweight=style.font_weight)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)

    buf = _fig_to_bytes_matplotlib(fig)
    plt.close(fig)
    return buf


# ----------------------------------------------------------------------------
# HELPERS: figure → bytes for export
# ----------------------------------------------------------------------------
def _fig_to_bytes_matplotlib(fig, fmt: str = "png", dpi: int = 300) -> Optional[bytes]:
    try:
        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning("Matplotlib export failed (%s): %s", fmt, e)
        return None


def _fig_to_bytes_plotly(fig, fmt: str = "png",
                         width: int = 1200, height: int = 800,
                         scale: float = 3.0) -> Optional[bytes]:
    try:
        return fig.to_image(format=fmt, width=width, height=height,
                            scale=scale)
    except Exception as e:
        logger.warning("Plotly static export failed (%s): %s", fmt, e)
        return None


# ----------------------------------------------------------------------------
# STYLE CONTROLS (Streamlit UI)
# ----------------------------------------------------------------------------
def render_recommender_style_controls() -> RecommenderVisualStyle:
    """Full publication styling panel with journal presets."""
    style = RecommenderVisualStyle()

    st.markdown("#### 🎨 Publication Styling")

    journal_options = {
        "Nature": "nature",
        "Science": "science",
        "Advanced Materials": "advanced_materials",
        "Physical Review Letters": "prl",
        "Custom": "custom",
    }
    preset_display = st.selectbox(
        "📚 Journal Preset",
        list(journal_options.keys()),
        index=0,
        key="rec_journal_preset",
        help="Applies a one-click publication style preset; you can still fine-tune below.",
    )
    style.journal = journal_options[preset_display]

    if style.journal != "custom":
        styles = JournalTemplates.get_journal_styles()
        js = styles.get(style.journal, styles['nature'])
        style.font_family = js['font_family']
        style.font_size_title = js['font_size_large']
        style.font_size_axis = js['font_size_medium']
        style.font_size_tick = js['font_size_small']
        style.font_size_legend = js['font_size_small']
        style.font_size_annotation = js['font_size_small']
        style.line_width = js['line_width']
        style.grid_alpha = js['grid_alpha']
        style.dpi = js['dpi']

    with st.expander("🔤 Fonts", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            style.font_family = st.selectbox(
                "Font family",
                FONT_FAMILIES,
                index=FONT_FAMILIES.index(style.font_family)
                if style.font_family in FONT_FAMILIES else 0,
                key="rec_font_family",
            )
            style.font_weight = st.selectbox(
                "Font weight",
                ["normal", "bold"],
                index=1 if style.font_weight == "bold" else 0,
                key="rec_font_weight",
            )
        with c2:
            style.font_size_title = st.slider(
                "Title size (pt)", 6.0, 24.0, float(style.font_size_title), 0.5,
                key="rec_fs_title")
            style.font_size_axis = st.slider(
                "Axis label size (pt)", 6.0, 20.0, float(style.font_size_axis), 0.5,
                key="rec_fs_axis")
            style.font_size_tick = st.slider(
                "Tick label size (pt)", 5.0, 18.0, float(style.font_size_tick), 0.5,
                key="rec_fs_tick")
            style.font_size_legend = st.slider(
                "Legend size (pt)", 5.0, 18.0, float(style.font_size_legend), 0.5,
                key="rec_fs_legend")
            style.font_size_annotation = st.slider(
                "Annotation size (pt)", 5.0, 16.0,
                float(style.font_size_annotation), 0.5,
                key="rec_fs_annot")

    with st.expander("🖼️ Figure & DPI", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            style.figure_width = st.slider(
                "Figure width (in)", 4.0, 20.0, float(style.figure_width), 0.5,
                key="rec_fw")
            style.line_width = st.slider(
                "Line width", 0.5, 5.0, float(style.line_width), 0.1,
                key="rec_lw")
        with c2:
            style.figure_height = st.slider(
                "Figure height (in)", 3.0, 16.0, float(style.figure_height), 0.5,
                key="rec_fh")
            style.marker_size = st.slider(
                "Marker size", 2.0, 20.0, float(style.marker_size), 0.5,
                key="rec_ms")
        with c3:
            style.dpi = st.select_slider(
                "DPI (export)",
                options=[72, 150, 300, 600, 1200],
                value=style.dpi if style.dpi in [72, 150, 300, 600, 1200] else 300,
                key="rec_dpi")

    with st.expander("🌈 Colormap", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            style.colormap = st.selectbox(
                "Colormap",
                cmap_list,
                index=cmap_list.index(style.colormap)
                if style.colormap in cmap_list else 0,
                key="rec_cmap",
            )
        with c2:
            style.cmap_reverse = st.checkbox("Reverse", value=style.cmap_reverse,
                                             key="rec_cmap_rev")
        try:
            cmap = style.get_cmap()
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            fig_prev, ax_prev = plt.subplots(figsize=(6, 0.45), dpi=120)
            ax_prev.imshow(gradient, aspect='auto', cmap=cmap)
            ax_prev.set_axis_off()
            st.pyplot(fig_prev, use_container_width=True)
            plt.close(fig_prev)
        except Exception:
            pass

        style.categorical_palette = st.selectbox(
            "Categorical palette (for radar, treemap roots, histograms)",
            list(PUBLICATION_PALETTES.keys()),
            index=list(PUBLICATION_PALETTES.keys()).index(style.categorical_palette)
            if style.categorical_palette in PUBLICATION_PALETTES else 0,
            key="rec_palette",
        )

    with st.expander("🎯 Labels, Titles & Legend", expanded=False):
        style.show_title = st.checkbox("Show title", value=style.show_title,
                                       key="rec_show_title")
        if style.show_title:
            style.title_override = st.text_input(
                "Title override (leave empty to use default)",
                value=style.title_override,
                key="rec_title_ovr",
            )
        style.xlabel_override = st.text_input(
            "X label override", value=style.xlabel_override,
            key="rec_xlabel_ovr",
        )
        style.ylabel_override = st.text_input(
            "Y label override", value=style.ylabel_override,
            key="rec_ylabel_ovr",
        )
        c1, c2 = st.columns(2)
        with c1:
            style.show_legend = st.checkbox("Show legend",
                                            value=style.show_legend,
                                            key="rec_show_legend")
            if style.show_legend:
                style.legend_position = st.selectbox(
                    "Legend position",
                    ['best', 'upper right', 'upper left', 'lower right',
                     'lower left', 'center right', 'center left',
                     'upper center', 'lower center', 'center'],
                    index=['best', 'upper right', 'upper left', 'lower right',
                           'lower left', 'center right', 'center left',
                           'upper center', 'lower center', 'center'].index(
                        style.legend_position)
                    if style.legend_position in
                    ['best', 'upper right', 'upper left', 'lower right',
                     'lower left', 'center right', 'center left',
                     'upper center', 'lower center', 'center'] else 0,
                    key="rec_legend_pos",
                )
        with c2:
            style.show_values = st.checkbox("Show numeric annotations",
                                            value=style.show_values,
                                            key="rec_show_values")
            style.show_colorbar = st.checkbox("Show colorbar",
                                              value=style.show_colorbar,
                                              key="rec_show_cbar")

    with st.expander("🧱 Backgrounds & Spines", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            style.background_color = st.color_picker(
                "Figure background", style.background_color,
                key="rec_bg_color")
            style.plot_background_color = st.color_picker(
                "Plot background", style.plot_background_color,
                key="rec_plot_bg")
            style.transparent_bg = st.checkbox(
                "Transparent background (for export)",
                value=style.transparent_bg, key="rec_transp_bg")
        with c2:
            style.spine_color = st.color_picker(
                "Spine / tick color", style.spine_color,
                key="rec_spine_color")
            style.spine_width = st.slider(
                "Spine width", 0.5, 4.0, float(style.spine_width), 0.1,
                key="rec_spine_w")
            style.grid = st.checkbox("Show grid", value=style.grid,
                                     key="rec_grid")
            if style.grid:
                style.grid_alpha = st.slider(
                    "Grid alpha", 0.0, 1.0, float(style.grid_alpha), 0.05,
                    key="rec_grid_alpha")
                style.grid_linestyle = st.selectbox(
                    "Grid linestyle",
                    ['-', '--', '-.', ':'],
                    index=['-', '--', '-.', ':'].index(style.grid_linestyle)
                    if style.grid_linestyle in ['-', '--', '-.', ':'] else 1,
                    key="rec_grid_ls",
                )

    style.apply_matplotlib()
    return style


# ----------------------------------------------------------------------------
# MAIN DASHBOARD ENTRY POINT
# ----------------------------------------------------------------------------
def render_recommender_visuals_dashboard():
    """Main-area dashboard with publication-quality AI recommender visuals."""
    bundle: Optional[PlasticityRecommendationBundle] = _plr_get("bundle")
    if bundle is None:
        return

    st.markdown("---")
    st.header("🤖 AI Recommender Visuals Dashboard")
    st.caption(
        "Publication-quality visualizations for plasticity parameter "
        "candidates, sources, and distributions. All styling controls are "
        "below — the export panel produces PNG / SVG / PDF at chosen DPI."
    )

    chart_type = st.selectbox(
        "📈 Chart Type",
        ["Radar Chart", "Bar Chart", "Sankey Diagram", "Treemap",
         "Histograms (publication)"],
        index=0,
        key="rec_chart_type_v83",
    )

    col_style, col_chart = st.columns([1, 2])

    with col_style:
        with st.container():
            style = render_recommender_style_controls()

    with col_chart:
        st.markdown(f"#### {chart_type}")
        chart_bytes: Optional[bytes] = None
        chart_fmt_hint = "png"

        try:
            if chart_type == "Radar Chart":
                chart_bytes = render_recommender_radar_pub(bundle, style)
                chart_fmt_hint = "png"
            elif chart_type == "Bar Chart":
                chart_bytes = render_recommender_bars_pub(bundle, style)
                chart_fmt_hint = "png"
            elif chart_type == "Sankey Diagram":
                chart_bytes = render_recommender_sankey_pub(bundle, style)
                chart_fmt_hint = "png"
            elif chart_type == "Treemap":
                chart_bytes = render_recommender_treemap_pub(bundle, style)
                chart_fmt_hint = "png"
            elif chart_type == "Histograms (publication)":
                chart_bytes = render_recommender_histograms_pub(bundle, style)
                chart_fmt_hint = "png"
        except Exception as e:
            st.error(f"Chart rendering failed: {e}")
            st.exception(e)

        st.markdown("---")
        st.markdown("##### 📤 Export Chart")
        if chart_bytes is None:
            st.caption(
                "⚠️ Static export unavailable for this chart. "
                "Install `kaleido` (plotly) and/or use Matplotlib-based charts "
                "for full PNG/SVG/PDF export."
            )
        else:
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.download_button(
                    f"⬇️ Download PNG ({style.dpi} DPI)",
                    data=chart_bytes,
                    file_name=(f"recommender_{chart_type.lower().replace(' ', '_')}"
                               f"_{bundle.material}_{int(style.dpi)}dpi.png"),
                    mime="image/png",
                    use_container_width=True,
                )
            with ec2:
                if chart_type in ("Bar Chart", "Histograms (publication)"):
                    st.caption("Matplotlib chart — SVG/PDF supported below.")
                else:
                    st.caption("SVG/PDF via Plotly needs `kaleido`.")
            with ec3:
                st.caption(f"Format hint: `{chart_fmt_hint}` · "
                           f"Current DPI: {style.dpi}")

        with st.expander("📊 Numeric summary of recommendations",
                         expanded=False):
            rows = []
            for p in PARAM_ORDER:
                spec = PLASTICITY_ONTOLOGY[p]
                best = bundle.best(p)
                prior = bundle.prior_suggestion(p)
                rows.append({
                    'Parameter': spec['label'],
                    'Symbol': spec['symbol'],
                    'Default': _pl_fmt(p, bundle.defaults[p]),
                    'Prior': _pl_fmt(p, prior) if prior is not None else '—',
                    'AI Best': _pl_fmt(p, best.value_si) if best else '—',
                    'Score': f"{best.score:.3f}" if best else '—',
                    'Method': best.method if best else '—',
                    'Source': (best.source_file[:30] + '…')
                              if best and best.source_file else '—',
                })
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Enhanced Nanotwinned Cu Phase-Field Simulator (FFT)",
        layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; color: #1E3A8A; text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] {
        height: 3rem; white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px; padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🔬 Enhanced Nanotwinned Cu Phase-Field Simulator (FFT)</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #F0F9FF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 1rem;">
    <strong>✅ PURE FFT SPECTRAL + AI PLASTICITY RECOMMENDER v8.3.0:</strong><br>
    • <span style="color: green;">NO FDM/NUMBA:</span> exact spectral operators.<br>
    • <span style="color: green;">SEMI-IMPLICIT FOURIER:</span> unconditional linear stability.<br>
    • <span style="color: green;">🤖 REASONING RECOMMENDER:</span> FAISS + Ollama + LatentMoE + dual-mode prompts (strict priors vs CoT inference).<br>
    • <span style="color: green;">🧠 CHAIN-OF-THOUGHT:</span> every inferred parameter carries an auditable reasoning chain.<br>
    • <span style="color: green;">🔑 GATEKEEPER FIX:</span> alias canonicalization (<code>rho_0</code>→<code>rho0</code>, <code>m</code>→<code>srs</code>, …) + per-parameter heuristic fallback + value coercion.<br>
    • <span style="color: green;">📊 PUBLICATION VISUALS:</span> Radar / Bars / Sankey / Treemap + full styling.<br>
    • <span style="color: green;">🐛 DEBUG TOGGLE:</span> raw LLM responses at INFO level when enabled.<br>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔄 Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", type="secondary"):
                for k in list(st.session_state.keys()):
                    if k in ('twin_simulations', 'initial_geometry', 'initialized',
                             'results_history', 'timesteps', 'solver',
                             'selected_sim_id', 'comparison_config',
                             'sweep_results', 'sweep_param',
                             'plasticity_overrides'):
                        st.session_state.pop(k, None)
                st.success("All simulations & state cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
        if st.button("🔄 Reset to Defaults", type="secondary"):
            keys_to_clear = ['N', 'dx', 'dt', 'twin_spacing', 'W', 'A', 'B',
                             'kappa0', 'gamma_aniso', 'kappa_eta', 'L_CTB',
                             'L_ITB', 'n_mob', 'L_eta', 'zeta', 'applied_stress',
                             'geom_type', 'defect_type', 'defect_x', 'defect_y',
                             'defect_radius', 'left_buffer_width', 'buffer_width',
                             'gb_profile', 'gb_curvature', 'grain_boundary_pos',
                             'stability_factor', 'enable_monitoring',
                             'auto_adjust_dt', 'n_steps', 'save_freq',
                             'confine_twin']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Parameters reset to defaults!")
            st.rerun()

        st.markdown("---")

        # ▼▼▼ AI Plasticity Recommender sidebar ▼▼▼
        with st.expander("🧠 AI Plasticity Recommender v8.3.0", expanded=False):
            render_plasticity_recommender_sidebar(
                default_material=st.session_state.get("material", "Cu"),
                default_temp=300.0,
                default_strain_rate=1e-3,
                ollama_model="qwen2.5:7b",
            )
        st.markdown("---")
        # ▲▲▲ END AI Plasticity Recommender sidebar ▲▲▲

        operation_mode = st.radio(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations",
             "Single Simulation View", "Parameter Sweep"],
            index=0)

        if operation_mode == "Run New Simulation":
            st.header("🎛️ New Simulation Setup")
            st.subheader("🧪 Material")
            material_choice = st.selectbox("Select material", ["Cu", "Al", "Ni"], key="material")
            st.subheader("🧩 Geometry Configuration")
            geometry_type = st.selectbox(
                "Geometry Type",
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
            dt = st.slider("Time step (ns)", 1e-5, 1e-2, 1e-3, 1e-5, key="dt", format="%.5f")
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
            kappa0 = st.slider("κ₀ (gradient energy ref)", 0.01, 10.0, 1.0, 0.1, key="kappa0")
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
                stability_factor = st.slider("Stability factor (unused in FFT mode)",
                                             0.1, 1.0, 0.5, 0.1, key="stability_factor")
                enable_monitoring = st.checkbox("Enable real-time monitoring", True,
                                                key="enable_monitoring")
                auto_adjust_dt = st.checkbox("Auto-adjust time step (unused in FFT mode)",
                                             True, key="auto_adjust_dt")
                confine_twin = st.checkbox("Confine twin evolution to twinned grain",
                                           True, key="confine_twin")
            st.subheader("🎨 Visualization Settings")
            global_cmap_phi = st.selectbox(
                "Global φ colormap", cmap_list,
                index=cmap_list.index('RdBu_r') if 'RdBu_r' in cmap_list else 0,
                key="global_cmap_phi")
            global_cmap_stress = st.selectbox(
                "Global σ_eq colormap", cmap_list,
                index=cmap_list.index('hot') if 'hot' in cmap_list else 0,
                key="global_cmap_stress")
            global_cmap_hydro = st.selectbox(
                "Global σ_h colormap", cmap_list,
                index=cmap_list.index('RdBu') if 'RdBu' in cmap_list else 0,
                key="global_cmap_hydro")
            sim_cmap_phi = st.selectbox(
                "Simulation-specific φ colormap", cmap_list,
                index=cmap_list.index(global_cmap_phi) if global_cmap_phi in cmap_list else 0,
                key="sim_cmap_phi")
            sim_cmap_stress = st.selectbox(
                "Simulation-specific σ_eq colormap", cmap_list,
                index=cmap_list.index(global_cmap_stress) if global_cmap_stress in cmap_list else 0,
                key="sim_cmap_stress")
            sim_cmap_hydro = st.selectbox(
                "Simulation-specific σ_h colormap", cmap_list,
                index=cmap_list.index(global_cmap_hydro) if global_cmap_hydro in cmap_list else 0,
                key="sim_cmap_hydro")
            st.subheader("📏 Scale Bar Settings")
            scalebar_color = st.color_picker("Scale bar color", "#000000", key="scalebar_color")
            scalebar_fontsize = st.slider("Scale bar font size", 6, 20, 10, 1,
                                          key="scalebar_fontsize")

            if st.button("🚀 Initialize Simulation", type="primary", use_container_width=True):
                params = {
                    'material': material_choice,
                    'N': N, 'dx': dx, 'dt': dt,
                    'W': W, 'A': A, 'B': B,
                    'kappa0': kappa0, 'gamma_aniso': gamma_aniso, 'kappa_eta': kappa_eta,
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
                    'scalebar_fontsize': scalebar_fontsize
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
                st.warning("No simulations saved yet. Run some simulations first!")
            else:
                sim_options = {sim['name']: sim['id'] for sim in simulations}
                selected_sim_ids = st.multiselect(
                    "Select Simulations to Compare",
                    options=list(sim_options.keys()),
                    default=list(sim_options.keys())[:min(3, len(sim_options))])
                comparison_type = st.selectbox(
                    "Comparison Type",
                    ["Side-by-Side Heatmaps", "Overlay Line Profiles",
                     "Statistical Summary", "Correlation Analysis",
                     "Evolution Timeline"], index=0)
                field_to_compare = st.selectbox(
                    "Field to Compare",
                    ["phi (Twin Order)", "eta1 (Grain)",
                     "sigma_eq (Von Mises Stress)",
                     "sigma_h (Hydrostatic Stress)",
                     "h (Twin Spacing)", "sigma_y (Yield Stress)"], index=2)
                field_key = field_to_compare.split()[0]
                if comparison_type == "Overlay Line Profiles":
                    profile_direction = st.selectbox(
                        "Profile Direction",
                        ["Horizontal", "Vertical", "Diagonal",
                         "Anti-Diagonal", "Custom"], index=0)
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                    profile_type_mapping = {
                        "Horizontal": "horizontal", "Vertical": "vertical",
                        "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal",
                        "Custom": "custom"}
                    internal_direction = profile_type_mapping.get(profile_direction,
                                                                 "horizontal")
                    if profile_direction == "Custom":
                        custom_angle = st.slider("Custom angle (deg)", -180, 180, 45, 5)
                    else:
                        custom_angle = 45
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
            material_choice = st.selectbox("Material", ["Cu", "Al", "Ni"],
                                           key="sweep_material")
            geom_type_sweep = st.selectbox(
                "Geometry Type",
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
                applied_stress_sweep = st.slider("Applied stress (MPa)", 0.0, 600.0,
                                                 300.0, 10.0, key="sweep_stress")
            st.subheader("Sweep Parameter")
            sweep_param = st.selectbox(
                "Choose parameter to vary",
                ["twin_spacing", "applied_stress", "applied_stress_angle",
                 "W", "L_CTB", "L_ITB", "kappa0"], index=0)
            col1, col2 = st.columns(2)
            with col1:
                sweep_min = st.number_input(
                    f"Min {sweep_param}",
                    value=10.0 if sweep_param == "twin_spacing" else 0.0,
                    key="sweep_min")
                sweep_max = st.number_input(
                    f"Max {sweep_param}",
                    value=50.0 if sweep_param == "twin_spacing" else 500.0,
                    key="sweep_max")
            with col2:
                sweep_steps = st.number_input("Number of steps",
                                              min_value=2, max_value=20,
                                              value=5, step=1, key="sweep_steps")
            if sweep_param in ["applied_stress"]:
                sweep_values = np.linspace(sweep_min * 1e6, sweep_max * 1e6,
                                           sweep_steps)
            else:
                sweep_values = np.linspace(sweep_min, sweep_max, sweep_steps)
            st.write(f"Sweep values: {sweep_values}")
            base_params = {
                'material': material_choice,
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
    if operation_mode == "Compare Saved Simulations" and 'comparison_config' in st.session_state:
        st.header("🔬 Multi-Simulation Comparison")
        config = st.session_state.comparison_config
        simulations = []
        valid_ids = []
        for sim_id in config['sim_ids']:
            sim = SimulationDatabase.get_simulation(sim_id)
            if sim:
                simulations.append(sim)
                valid_ids.append(sim_id)
        if not simulations:
            st.error("No valid simulations found.")
        else:
            st.success(f"Loaded {len(simulations)} simulations")
            sim_names = [build_sim_name(sim['params'], sim['id'])
                         for sim in simulations]
            if config['type'] == "Side-by-Side Heatmaps":
                last_frames = []
                for sim in simulations:
                    last_frames.append(sim['results_history'][-1]
                                       if sim['results_history'] else None)
                valid_indices = [i for i, f in enumerate(last_frames)
                                 if f is not None]
                if not valid_indices:
                    st.warning("No frame data available.")
                else:
                    n_sims = len(valid_indices)
                    cols = min(3, n_sims)
                    rows = (n_sims + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                    if rows == 1 and cols == 1:
                        axes = np.array([axes])
                    else:
                        axes = axes.flatten()
                    for idx, sim_idx in enumerate(valid_indices):
                        ax = axes[idx]
                        sim = simulations[sim_idx]
                        frame = last_frames[sim_idx]
                        field = config['field']
                        if field in frame:
                            data = frame[field].copy()
                            if field in ['sigma_eq', 'sigma_h']:
                                data = data / 1e9
                            elif field == 'sigma_y':
                                data = data / 1e6
                            extent = [-sim['params']['N'] * sim['params']['dx'] / 2,
                                      sim['params']['N'] * sim['params']['dx'] / 2] * 2
                            im = ax.imshow(data, extent=extent, cmap='viridis',
                                           origin='lower')
                            ax.set_title(sim_names[sim_idx][:30] + "...", fontsize=8)
                            ax.set_xlabel('x (nm)')
                            ax.set_ylabel('y (nm)')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    for idx in range(len(valid_indices), len(axes)):
                        axes[idx].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            elif config['type'] == "Overlay Line Profiles":
                fig = go.Figure()
                ref_sim = simulations[0]
                N = ref_sim['params']['N']
                dx = ref_sim['params']['dx']
                visualizer = EnhancedTwinVisualizer(N, dx)
                for sim_idx, sim in enumerate(simulations):
                    if not sim['results_history']:
                        continue
                    frame = sim['results_history'][-1]
                    field = config['field']
                    if field not in frame:
                        continue
                    distance, profile, _ = visualizer.line_profiler.extract_profile(
                        frame[field], config['profile_direction'],
                        config['position_ratio'], config.get('custom_angle', 45))
                    if field in ['sigma_eq', 'sigma_h']:
                        profile = profile / 1e9
                        ylabel = 'Stress (GPa)'
                    elif field == 'sigma_y':
                        profile = profile / 1e6
                        ylabel = 'Stress (MPa)'
                    else:
                        ylabel = field
                    fig.add_trace(go.Scatter(x=distance, y=profile, mode='lines',
                                             name=sim_names[sim_idx][:30]))
                fig.update_layout(
                    title=f"{field} Line Profiles Comparison",
                    xaxis_title="Position (nm)", yaxis_title=ylabel,
                    hovermode='x unified', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif config['type'] == "Statistical Summary":
                data = []
                for sim in simulations:
                    params = sim['params']
                    hist = sim['results_history']
                    if hist:
                        last = hist[-1].get('convergence', {})
                        row = {
                            'Name': build_sim_name(params, sim['id'])[:40],
                            'λ (nm)': params.get('twin_spacing', 0),
                            'σ_app (MPa)': params.get('applied_stress', 0) / 1e6,
                            'θ (deg)': params.get('applied_stress_angle', 0),
                            'W (J/m³)': params.get('W', 0),
                            'Avg σ_eq (GPa)': last.get('avg_stress', 0) / 1e9,
                            'Max σ_eq (GPa)': last.get('max_stress', 0) / 1e9,
                            'Avg h (nm)': last.get('avg_spacing', 0),
                            'Plastic Work (J)': last.get('plastic_work', 0),
                            'Energy (J)': last.get('energy', 0)}
                        data.append(row)
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df['Name'], y=df['Avg σ_eq (GPa)'],
                                         name='Avg Stress'))
                    fig.add_trace(go.Bar(x=df['Name'], y=df['Max σ_eq (GPa)'],
                                         name='Max Stress'))
                    fig.update_layout(title="Stress Comparison",
                                      xaxis_title="Simulation",
                                      yaxis_title="Stress (GPa)")
                    st.plotly_chart(fig, use_container_width=True)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df['λ (nm)'],
                                              y=df['Avg σ_eq (GPa)'],
                                              mode='markers+text',
                                              text=df['Name'],
                                              textposition='top center'))
                    fig2.update_layout(title="Twin Spacing vs. Avg Stress",
                                       xaxis_title="λ (nm)",
                                       yaxis_title="Avg Stress (GPa)")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No convergence data available.")
            elif config['type'] == "Correlation Analysis":
                data = []
                for sim in simulations:
                    params = sim['params']
                    hist = sim['results_history']
                    if hist and 'convergence' in hist[-1]:
                        conv = hist[-1]['convergence']
                        row = {
                            'twin_spacing': params.get('twin_spacing', 0),
                            'applied_stress': params.get('applied_stress', 0) / 1e6,
                            'W': params.get('W', 0),
                            'avg_stress': conv.get('avg_stress', 0) / 1e9,
                            'max_stress': conv.get('max_stress', 0) / 1e9,
                            'avg_spacing': conv.get('avg_spacing', 0),
                            'plastic_work': conv.get('plastic_work', 0)}
                        data.append(row)
                if data:
                    df = pd.DataFrame(data)
                    fig = go.Figure(data=go.Splom(
                        dimensions=[dict(label=k, values=df[k])
                                    for k in df.columns],
                        showupperhalf=False, marker=dict(size=8)))
                    fig.update_layout(title="Correlation Matrix",
                                      width=800, height=800)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No convergence data available.")
            elif config['type'] == "Evolution Timeline":
                metric_map = {
                    'phi': 'phi_norm', 'sigma_eq': 'avg_stress',
                    'h': 'twin_spacing_avg', 'energy': 'energy',
                    'plastic_work': 'plastic_work'}
                chosen_metric = st.selectbox("Metric to track",
                                             list(metric_map.keys()))
                fig = go.Figure()
                for sim_idx, sim in enumerate(simulations):
                    hist = sim.get('history') if 'history' in sim else None
                    if hist is None and 'solver' in sim:
                        hist = sim['solver'].history
                    if hist is None:
                        continue
                    metric_key = metric_map.get(chosen_metric, chosen_metric)
                    if metric_key not in hist:
                        continue
                    times = np.arange(len(hist[metric_key])) * sim['params'].get('dt', 1e-4)
                    values = hist[metric_key]
                    if chosen_metric in ['sigma_eq']:
                        values = np.array(values) / 1e9
                    fig.add_trace(go.Scatter(x=times, y=values, mode='lines',
                                             name=sim_names[sim_idx][:30]))
                fig.update_layout(
                    title=f"{chosen_metric} Evolution Comparison",
                    xaxis_title="Time (ns)", yaxis_title=chosen_metric,
                    hovermode='x unified', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    elif operation_mode == "Single Simulation View" and 'selected_sim_id' in st.session_state:
        sim_id = st.session_state.selected_sim_id
        sim_data = SimulationDatabase.get_simulation(sim_id)
        if sim_data:
            st.header(f"📊 Single Simulation: {build_sim_name(sim_data['params'], sim_id)}")
            params = sim_data['params']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("λ (twin spacing)", f"{params.get('twin_spacing', 0):.1f} nm")
            with col2:
                stress_mag = params.get('applied_stress', 0) / 1e6
                angle = params.get('applied_stress_angle', 0)
                st.metric("σ_app / θ", f"{stress_mag:.0f} MPa / {angle:.0f}°")
            with col3:
                st.metric("W (well depth)", f"{params.get('W', 0):.2f} J/m³")
            with col4:
                st.metric("κ₀", f"{params.get('kappa0', 0):.2f}")
            history = sim_data.get('results_history', [])
            if history:
                num_frames = len(history)
                frame_idx = st.slider("Frame", 0, num_frames - 1, num_frames - 1,
                                      key=f"frame_slider_{sim_id}")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⏮️ First"):
                        st.session_state[f"frame_slider_{sim_id}"] = 0
                        st.rerun()
                with col2:
                    play = st.checkbox("▶️ Play", key=f"play_{sim_id}")
                    if play:
                        current = st.session_state.get(f"frame_slider_{sim_id}", 0)
                        if current < num_frames - 1:
                            st.session_state[f"frame_slider_{sim_id}"] = current + 1
                        else:
                            st.session_state[f"frame_slider_{sim_id}"] = 0
                        st.rerun()
                with col3:
                    if st.button("⏭️ Last"):
                        st.session_state[f"frame_slider_{sim_id}"] = num_frames - 1
                        st.rerun()
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
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                if st.button("🗑️ Delete This Simulation", key=f"delete_{sim_id}"):
                    SimulationDatabase.delete_simulation(sim_id)
                    if 'selected_sim_id' in st.session_state:
                        del st.session_state.selected_sim_id
                    st.success(f"Simulation {sim_id} deleted!")
                    st.rerun()
            else:
                st.warning("No simulation history found.")
        else:
            st.error("Simulation not found.")

    elif operation_mode == "Run New Simulation" and 'initialized' in st.session_state:
        params = st.session_state.initial_geometry['params']
        N = params['N']
        dx = params['dx']
        visualizer = EnhancedTwinVisualizer(N, dx, dt=params.get('dt', 1e-4))

        tabs = st.tabs(["📐 Initial Geometry", "▶️ Run Simulation", "📊 Basic Results",
                        "🔍 Advanced Analysis", "📊 Plotly Interactive",
                        "🖥️ 3D Interactive", "📤 Enhanced Export"])

        with tabs[0]:
            st.header("Initial Geometry Visualization")
            geom_viz = st.session_state.initial_geometry['geom_viz']
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
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_spacing = np.mean(h[(h > 5) & (h < 50)]) if np.any((h > 5) & (h < 50)) else 0
                st.metric("Avg Twin Spacing", f"{avg_spacing:.1f} nm")
            with col2:
                twin_area = np.sum(eta1 > 0.5) * dx**2
                st.metric("Twin Grain Area", f"{twin_area:.0f} nm²")
            with col3:
                num_twins = np.sum(h < 20)
                st.metric("Number of Twins", f"{num_twins:.0f}")

        with tabs[1]:
            st.header("Run Simulation (Pure FFT Spectral Method)")

            _active_overrides = st.session_state.get("plasticity_overrides", {})
            if _active_overrides:
                st.info(
                    "🤖 AI‑recommended plasticity parameters will be injected: "
                    + ", ".join(f"`{k}={v:.3g}`" for k, v in _active_overrides.items())
                )

            if st.button("▶️ Start Evolution", type="secondary", use_container_width=True):
                with st.spinner("Running phase-field simulation (FFT)..."):
                    try:
                        solver = NanotwinnedCuSolver(params)
                        solver.phi = st.session_state.initial_geometry['phi'].copy()
                        solver.eta1 = st.session_state.initial_geometry['eta1'].copy()
                        solver.eta2 = st.session_state.initial_geometry['eta2'].copy()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_history = []
                        timesteps = []
                        monitoring_cols = st.columns(4)
                        n_steps = params['n_steps']
                        dt = params['dt']
                        save_freq = params['save_frequency']
                        for step in range(n_steps):
                            status_text.text(f"Step {step + 1}/{n_steps} | "
                                             f"Time: {(step + 1) * dt:.4f} ns")
                            results = solver.step()
                            if step % save_freq == 0:
                                results_history.append(results.copy())
                                timesteps.append(step * dt)
                            progress_bar.progress((step + 1) / n_steps)
                            if step % 10 == 0 and len(results_history) > 0:
                                with monitoring_cols[0]:
                                    st.metric("Avg Stress",
                                              f"{np.mean(results['sigma_eq']) / 1e9:.2f} GPa")
                                with monitoring_cols[1]:
                                    valid_h = results['h'][(results['h'] > 5) & (results['h'] < 50)]
                                    avg_h = np.mean(valid_h) if len(valid_h) > 0 else 0
                                    st.metric("Avg Spacing", f"{avg_h:.1f} nm")
                                with monitoring_cols[2]:
                                    st.metric("Max Plastic Strain",
                                              f"{np.max(results['eps_p_mag']):.4f}")
                                with monitoring_cols[3]:
                                    st.metric("Energy",
                                              f"{results['convergence']['energy']:.2e} J")
                        st.success(f"✅ Simulation completed! Generated {len(results_history)} frames.")
                        st.session_state.results_history = results_history
                        st.session_state.timesteps = timesteps
                        st.session_state.solver = solver
                        start_time = datetime.now()
                        sim_id = SimulationDatabase.save_simulation(
                            params, results_history,
                            st.session_state.initial_geometry,
                            run_time=(datetime.now() - start_time).total_seconds())
                        st.balloons()
                    except Exception as e:
                        st.error(f"Simulation failed: {str(e)}")
                        st.exception(e)

        with tabs[2]:
            if 'results_history' in st.session_state:
                st.header("Basic Results Visualization")
                results_history = st.session_state.results_history
                frame_idx = st.slider("Select frame", 0, len(results_history) - 1,
                                      len(results_history) - 1)
                results = results_history[frame_idx]
                style_params = {
                    'phi_cmap': params.get('cmap_phi', 'RdBu_r'),
                    'eta1_cmap': params.get('cmap_eta1', 'Reds'),
                    'sigma_eq_cmap': params.get('cmap_stress', 'hot'),
                    'sigma_h_cmap': params.get('cmap_hydro', 'RdBu'),
                    'scalebar_color': params.get('scalebar_color', 'black'),
                    'scalebar_fontsize': params.get('scalebar_fontsize', 10)}
                fig = visualizer.create_multi_field_comparison(results, style_params)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                st.subheader("Convergence Monitoring")
                if 'solver' in st.session_state and \
                   st.session_state.solver.history['phi_norm']:
                    full_timesteps = (np.arange(len(st.session_state.solver.history['phi_norm']))
                                      * params['dt'])
                    conv_fig = SimulationMonitor.create_convergence_plots(
                        st.session_state.solver.history, full_timesteps)
                    st.pyplot(conv_fig)
                    plt.close(conv_fig)
            else:
                st.info("Run a simulation first.")

        with tabs[3]:
            if 'results_history' in st.session_state:
                st.header("Advanced Analysis Tools")
                st.subheader("Line Profile Analysis")
                results = st.session_state.results_history[-1]
                col1, col2 = st.columns(2)
                with col1:
                    profile_types = st.multiselect(
                        "Profile Directions",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"],
                        default=["Horizontal", "Vertical"])
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                with col2:
                    field_to_profile = st.selectbox(
                        "Field to Profile",
                        ["phi", "eta1", "sigma_eq", "sigma_h", "h", "sigma_y"],
                        index=2)
                profile_type_mapping = {
                    "Horizontal": "horizontal", "Vertical": "vertical",
                    "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal"}
                internal_types = [profile_type_mapping[pt] for pt in profile_types]
                profiler = EnhancedLineProfiler(N, dx)
                fig_profiles, axes = plt.subplots(len(internal_types), 1,
                                                  figsize=(10, 4 * len(internal_types)))
                if len(internal_types) == 1:
                    axes = [axes]
                for idx, ptype in enumerate(internal_types):
                    ax = axes[idx]
                    distance, profile, _ = profiler.extract_profile(
                        results[field_to_profile], ptype, position_ratio)
                    if field_to_profile in ['sigma_eq', 'sigma_h']:
                        profile = profile / 1e9
                        ylabel = 'Stress (GPa)'
                    elif field_to_profile == 'sigma_y':
                        profile = profile / 1e6
                        ylabel = 'Stress (MPa)'
                    else:
                        ylabel = field_to_profile
                    ax.plot(distance, profile, 'b-', linewidth=2)
                    ax.set_xlabel('Position (nm)')
                    ax.set_ylabel(ylabel)
                    ax.set_title(f'{ptype.replace("_", " ").title()} Profile')
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_profiles)
                plt.close(fig_profiles)
            else:
                st.info("Run a simulation first.")

        with tabs[4]:
            if 'results_history' in st.session_state:
                st.header("📊 Plotly Interactive Visualization (2D)")
                results_history = st.session_state.results_history
                plotly_field = st.selectbox(
                    "Select field to visualize",
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h",
                     "eps_p_mag", "sigma_y"],
                    index=0, key="plotly_2d_field")
                frame_idx_plotly = st.slider("Frame", 0, len(results_history) - 1,
                                             len(results_history) - 1,
                                             key="plotly_2d_frame")
                results = results_history[frame_idx_plotly]
                fig_heatmap = visualizer.create_plotly_heatmap(
                    results, plotly_field, frame_idx_plotly)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown("---")
                st.subheader("Interactive Line Profiles")
                col1, col2 = st.columns(2)
                with col1:
                    profile_type_plotly = st.selectbox(
                        "Profile direction",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"],
                        key="plotly_2d_profile")
                with col2:
                    position_ratio_plotly = st.slider("Position ratio", 0.0, 1.0,
                                                      0.5, 0.05,
                                                      key="plotly_2d_pos")
                profile_type_mapping = {
                    "Horizontal": "horizontal", "Vertical": "vertical",
                    "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal"}
                internal_pt = profile_type_mapping.get(profile_type_plotly,
                                                       "horizontal")
                fig_line = visualizer.create_plotly_line_profiles(
                    results, plotly_field, [internal_pt], position_ratio_plotly)
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Run a simulation first to generate interactive plots.")

        with tabs[5]:
            st.header("🖥️ 3D Interactive Surface Visualization")
            if 'results_history' in st.session_state:
                results_history = st.session_state.results_history
                field_3d = st.selectbox(
                    "Select field for 3D surface",
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h",
                     "eps_p_mag", "sigma_y"],
                    index=1, key="3d_field")
                frame_idx_3d = st.slider("Frame", 0, len(results_history) - 1,
                                         len(results_history) - 1, key="3d_frame")
                results = results_history[frame_idx_3d]
                fig_3d = visualizer.create_plotly_3d_surface(results, field_3d,
                                                             frame_idx_3d)
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.markdown("""
                    **💡 Interactivity**:
                    - **Rotate** by dragging, **zoom** with scroll, **pan** with right‑click drag.
                    - Hover over the surface to see exact coordinates and field values.
                    """)
                else:
                    st.warning(f"Field '{field_3d}' not available in current results.")
            else:
                st.info("Run a simulation first to generate 3D visualizations.")

        with tabs[6]:
            st.header("📤 Enhanced Export")
            if 'results_history' in st.session_state and st.session_state.results_history:
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
                        buffer, fname = DataExporter.export_pkl(
                            sim_data, params, results_history, sim_name)
                        st.download_button("Download PKL", buffer, fname)
                    if st.button("🔥 PyTorch (PT)"):
                        buffer, fname = DataExporter.export_pt(
                            sim_data, params, results_history, sim_name)
                        st.download_button("Download PT", buffer, fname)
                    if st.button("📄 SQL Dump"):
                        buffer, fname = DataExporter.export_sql(
                            sim_data, params, results_history, sim_name, sim_id,
                            params['N'], params['dx'])
                        st.download_button("Download SQL", buffer, fname)
                with col2:
                    if st.button("📊 CSV (ZIP)"):
                        vis = EnhancedTwinVisualizer(params['N'], params['dx'])
                        buffer, fname = DataExporter.export_csv(
                            results_history, sim_name, vis.extent,
                            params['N'], params['dx'])
                        st.download_button("Download CSV ZIP", buffer, fname)
                    if st.button("📋 JSON"):
                        buffer, fname = DataExporter.export_json(
                            sim_data, params, results_history, sim_name)
                        st.download_button("Download JSON", buffer, fname)
                    if st.button("📁 HDF5"):
                        buffer, fname = DataExporter.export_hdf5(
                            sim_data, params, results_history, sim_name,
                            params['N'], params['dx'])
                        if buffer:
                            st.download_button("Download HDF5", buffer, fname)
                with col3:
                    st.subheader("Animation Export")
                    anim_field = st.selectbox(
                        "Field for animation",
                        ["phi", "eta1", "sigma_eq", "sigma_h", "h", "eps_p_mag"],
                        key="anim_field")
                    anim_format = st.selectbox("Format", ["gif", "mp4"],
                                               key="anim_format")
                    fps = st.slider("FPS", 1, 30, 5)
                    if st.button("🎬 Generate Animation"):
                        with st.spinner("Creating animation..."):
                            vis = EnhancedTwinVisualizer(
                                params['N'], params['dx'], dt=params.get('dt', 1e-4))
                            anim_buffer = vis.create_animation(
                                results_history, anim_field, anim_format, fps)
                            if anim_buffer:
                                st.download_button(
                                    f"Download {anim_format.upper()}",
                                    anim_buffer,
                                    f"{sim_name}_{anim_field}.{anim_format}")
                st.markdown("---")
                st.subheader("Bulk Export All Simulations")
                if st.button("📦 Export All Simulations"):
                    vis = EnhancedTwinVisualizer(params['N'], params['dx'])
                    bulk_buffer, bulk_fname = DataExporter.bulk_export_all_simulations(
                        params['N'], params['dx'], vis.extent)
                    if bulk_buffer:
                        st.download_button("Download All Simulations ZIP",
                                           bulk_buffer, bulk_fname)
            else:
                st.info("Run a simulation first to export data.")

    elif operation_mode == "Parameter Sweep" and 'sweep_results' in st.session_state:
        st.header("📊 Parameter Sweep Results")
        sweep_results = st.session_state.sweep_results
        sweep_param = st.session_state.sweep_param
        param_vals = []
        avg_stress = []
        max_stress = []
        avg_spacing = []
        plastic_work = []
        for res in sweep_results:
            if res['convergence'] is not None:
                param_vals.append(res['param_value'])
                conv = res['convergence']
                avg_stress.append(conv.get('sigma_eq', 0) / 1e9
                                  if isinstance(conv.get('sigma_eq'), (int, float)) else 0)
                max_stress.append(conv.get('max_stress', 0) / 1e9)
                avg_spacing.append(conv.get('twin_spacing_avg', 0))
                plastic_work.append(conv.get('plastic_work', 0))
        if sweep_param == 'applied_stress':
            param_display = np.array(param_vals) / 1e6
            param_label = "Applied Stress (MPa)"
        else:
            param_display = param_vals
            param_label = sweep_param.replace('_', ' ').title()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].plot(param_display, avg_stress, 'o-', linewidth=2)
        axes[0, 0].set_xlabel(param_label)
        axes[0, 0].set_ylabel("Avg Von Mises Stress (GPa)")
        axes[0, 0].set_title("Stress vs Parameter")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(param_display, avg_spacing, 's-', color='green', linewidth=2)
        axes[0, 1].set_xlabel(param_label)
        axes[0, 1].set_ylabel("Avg Twin Spacing (nm)")
        axes[0, 1].set_title("Twin Spacing vs Parameter")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(param_display, plastic_work, 'd-', color='red', linewidth=2)
        axes[1, 0].set_xlabel(param_label)
        axes[1, 0].set_ylabel("Plastic Work (J)")
        axes[1, 0].set_title("Plastic Work vs Parameter")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(param_display, max_stress, '^-', color='purple',
                        linewidth=2, label='Max')
        axes[1, 1].plot(param_display, avg_stress, 'o-', color='blue',
                        linewidth=2, label='Avg')
        axes[1, 1].set_xlabel(param_label)
        axes[1, 1].set_ylabel("Stress (GPa)")
        axes[1, 1].set_title("Stress Extremes vs Parameter")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        df_sweep = pd.DataFrame({
            param_label: param_display,
            'Avg Stress (GPa)': avg_stress,
            'Max Stress (GPa)': max_stress,
            'Avg Spacing (nm)': avg_spacing,
            'Plastic Work (J)': plastic_work})
        st.dataframe(df_sweep)
        if st.button("Clear Sweep Results"):
            del st.session_state.sweep_results
            del st.session_state.sweep_param
            st.rerun()

    # ========================================================================
    # AI RECOMMENDER VISUALS DASHBOARD (main area, appears when bundle exists)
    # ========================================================================
    render_recommender_visuals_dashboard()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
