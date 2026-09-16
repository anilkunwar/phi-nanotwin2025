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
import pandas as pd
import logging

# Optional HDF5 export
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
# SPECTRAL DERIVATIVE HELPERS (replacement for FDM/Numba kernels)
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

def compute_anisotropic_properties(phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso,
                                   L_CTB, L_ITB, n_mob):
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
    MAX_OVERSTRESS = 3.0
    MAX_PLASTIC_STRAIN = 1.0
    overstress = np.maximum(sigma_eq - sigma_y, 0.0) / np.maximum(sigma_y, 1e-9)
    overstress = np.minimum(overstress, MAX_OVERSTRESS)
    gamma_dot = gamma0_dot * overstress**m
    stress_dev = 2.0 / 3.0 * gamma_dot * dt
    stress_dev = np.minimum(stress_dev, 0.05)
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
        except:
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
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            },
            'science': {
                'figure_width_single': 5.5, 'figure_width_double': 11.4,
                'font_family': 'Helvetica', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 0.75, 'axes_linewidth': 0.75,
                'tick_width': 0.75, 'tick_length': 3, 'grid_alpha': 0.15, 'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
                              '#4DBEEE', '#A2142F', '#FF00FF', '#00FFFF', '#FFA500']
            },
            'advanced_materials': {
                'figure_width_single': 8.6, 'figure_width_double': 17.8,
                'font_family': 'Arial', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0.2, 'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000', '#44AA99',
                              '#882255', '#117733', '#999933', '#AA4499', '#88CCEE']
            },
            'prl': {
                'figure_width_single': 3.4, 'figure_width_double': 7.0,
                'font_family': 'Times New Roman', 'font_size_small': 8,
                'font_size_medium': 10, 'font_size_large': 12,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0, 'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                              '#0072B2', '#D55E00', '#CC79A7', '#999999', '#FFFFFF']
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

# ============================================================================
# POST-PROCESSING CLASSES
# ============================================================================
class EnhancedLineProfiler:
    """Enhanced line profile system with multiple orientations."""
   
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
   
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
            start_idx = (center_x - diag_length//2, center_y - diag_length//2)
            profile = []
            distances = []
            for i in range(diag_length):
                x = start_idx[0] + i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    dist = i * self.dx * np.sqrt(2)
                    distances.append(dist - (diag_length//2) * self.dx * np.sqrt(2))
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = start_idx[0] * self.dx + self.extent[0]
            y_start = start_idx[1] * self.dx + self.extent[2]
            x_end = (start_idx[0] + diag_length - 1) * self.dx + self.extent[0]
            y_end = (start_idx[1] + diag_length - 1) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        elif profile_type == 'anti_diagonal':
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x + diag_length//2, center_y - diag_length//2)
            profile = []
            distances = []
            for i in range(diag_length):
                x = start_idx[0] - i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    dist = i * self.dx * np.sqrt(2)
                    distances.append(dist - (diag_length//2) * self.dx * np.sqrt(2))
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
            dx_line = np.cos(angle_rad) * length//2
            dy_line = np.sin(angle_rad) * length//2
            profile = []
            distances = []
            for t in np.linspace(-length//2, length//2, length):
                x = center_x + t * np.cos(angle_rad) + offset * np.cos(angle_rad + np.pi/2)
                y = center_y + t * np.sin(angle_rad) + offset * np.sin(angle_rad + np.pi/2)
                if 0 <= x < nx-1 and 0 <= y < ny-1:
                    x0, y0 = int(x), int(y)
                    x1, y1 = x0 + 1, y0 + 1   # <-- FIXED: was y1 = y1 + 1
                    if x1 >= nx: x1 = nx - 1
                    if y1 >= ny: y1 = ny - 1
                    wx = x - x0
                    wy = y - y0
                    val = (data[y0, x0] * (1-wx) * (1-wy) +
                           data[y0, x1] * wx * (1-wy) +
                           data[y1, x0] * (1-wx) * wy +
                           data[y1, x1] * wx * wy)
                    profile.append(val)
                    distances.append(t * self.dx)
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = (center_x - dx_line + offset * np.cos(angle_rad + np.pi/2)) * self.dx + self.extent[0]
            y_start = (center_y - dy_line + offset * np.sin(angle_rad + np.pi/2)) * self.dx + self.extent[2]
            x_end = (center_x + dx_line + offset * np.cos(angle_rad + np.pi/2)) * self.dx + self.extent[0]
            y_end = (center_y + dy_line + offset * np.sin(angle_rad + np.pi/2)) * self.dx + self.extent[2]
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
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
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
                        f"σ={params.get('applied_stress', 0)/1e6:.0f}MPa | "
                        f"θ={params.get('applied_stress_angle', 0):.0f}° | "
                        f"W={params.get('W', 0):.1f}")
                simulations.append({
                    'id': sim_id, 'name': name, 'params': params,
                    'metadata': metadata,
                    'results': sim_data['results_history'][-1] if sim_data['results_history'] else None
                })
            except:
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
    except:
        return "unknown"

@handle_errors
def fmt_num_trim(x, ndigits=3):
    try:
        s = f"{x:.{ndigits}f}"
        s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s
    except:
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
        stress_mpa = fmt_num_trim(params.get("applied_stress", 300e6)/1e6, ndigits=0)
        theta = fmt_num_trim(params.get("applied_stress_angle", 0.0), ndigits=0)
        name = f"twin_lambda_{twin_spacing}_W_{W}_stress_{stress_mpa}MPa_theta_{theta}_{geom_token}"
        if sim_id:
            name = f"{name}_{sim_id}"
        return name
    except:
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
                'gamma_tw': 1/np.sqrt(2),
                'n_111': np.array([1, 1, 1])/np.sqrt(3),
                'a_112': np.array([1, 1, -2])/np.sqrt(6),
                'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
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
                'gamma_tw': 1/np.sqrt(2),
                'n_111': np.array([1, 1, 1])/np.sqrt(3),
                'a_112': np.array([1, 1, -2])/np.sqrt(6),
                'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
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
                'gamma_tw': 1/np.sqrt(2),
                'n_111': np.array([1, 1, 1])/np.sqrt(3),
                'a_112': np.array([1, 1, -2])/np.sqrt(6),
                'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
            },
            'plasticity': {'mu': 80e9, 'nu': 0.31, 'b': 0.249e-9,
                          'sigma0': 70e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
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
        self.x = np.linspace(-N*dx/2, N*dx/2, N)
        self.y = np.linspace(-N*dx/2, N*dx/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

    @handle_errors
    def create_twin_grain_geometry(self, twin_spacing=20.0, grain_boundary_pos=0.0,
                                   gb_width=3.0, buffer_width=5.0, left_buffer_width=5.0,
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
                    eta1[i, j] = 1.0; eta2[i, j] = 0.0
                elif dist_from_gb > gb_width:
                    eta1[i, j] = 0.0; eta2[i, j] = 1.0
                else:
                    transition = 0.5 * (1 - np.tanh(dist_from_gb / (gb_width/3)))
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
        C11_2d = (C11 + C12 + 2*C44) / 2
        C12_2d = (C11 + C12 - 2*C44) / 2
        lambda_2d = C12_2d
        mu_2d = (C11_2d - C12_2d) / 2
        self.C11_2d = C11_2d
        self.C12_2d = C12_2d
        self.C44_2d = C44
       
        denom = mu_2d * (lambda_2d + 2*mu_2d) * self.k2 + 1e-15
        self.G11 = (mu_2d*(self.kx**2 + 2*self.ky**2) + lambda_2d*self.ky**2) / denom
        self.G12 = -mu_2d * self.kx * self.ky / denom
        self.G22 = (mu_2d*(self.ky**2 + 2*self.kx**2) + lambda_2d*self.kx**2) / denom

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
        self.dt = dt   # <-- FIXED: store dt so animation() can use it
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
        self.line_profiler = EnhancedLineProfiler(N, dx)
        self.COLORMAPS = COLORMAPS.copy()
        custom = PublicationEnhancer.create_custom_colormaps()
        self.COLORMAPS.update(custom)

    @handle_errors
    def get_colormap(self, cmap_name):
        if cmap_name in self.COLORMAPS:
            entry = self.COLORMAPS[cmap_name]
            if isinstance(entry, str):
                return plt.cm.get_cmap(entry)
            else:
                return entry
        return plt.cm.get_cmap('viridis')

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
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
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
            data = data / 1e9; unit = " (GPa)"
        elif field_name == 'sigma_y':
            data = data / 1e6; unit = " (MPa)"
        elif field_name == 'h':
            unit = " (nm)"
        if field_name == 'phi':
            colorscale = 'RdBu'; zmid = 0
        elif field_name == 'eta1':
            colorscale = 'Reds'; zmid = None
        elif field_name in ['sigma_eq', 'sigma_h']:
            colorscale = 'Viridis' if field_name == 'sigma_eq' else 'RdBu'
            zmid = 0 if field_name == 'sigma_h' else None
        else:
            colorscale = 'Plasma'; zmid = None
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
                profile = profile / 1e9; ylabel = 'Stress (GPa)'
            elif field_name == 'sigma_y':
                profile = profile / 1e6; ylabel = 'Stress (MPa)'
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
            data = data / 1e9; unit = " (GPa)"
        elif field_name == 'sigma_y':
            data = data / 1e6; unit = " (MPa)"
        elif field_name == 'h':
            unit = " (nm)"
        x = np.linspace(self.extent[0], self.extent[1], self.N)
        y = np.linspace(self.extent[2], self.extent[3], self.N)
        X, Y = np.meshgrid(x, y)
        if field_name == 'phi':
            colorscale = 'RdBu'; cmin, cmax = -1.2, 1.2
        elif field_name == 'eta1':
            colorscale = 'Reds'; cmin, cmax = 0, 1
        elif field_name == 'sigma_h':
            colorscale = 'RdBu'
            cmin, cmax = -np.max(np.abs(data)), np.max(np.abs(data))
        else:
            colorscale = 'Viridis'; cmin, cmax = None, None
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
    def create_animation(self, history, field_name, output_format='gif', fps=5, dpi=150):
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
            vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
        im = ax.imshow(first_data, extent=self.extent,
                       cmap=self.get_colormap('viridis'),
                       vmin=vmin, vmax=vmax, origin='lower', interpolation='bilinear')
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
                                      interval=1000/fps, blit=True)
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
        # The anisotropy is handled by treating the isotropic part implicitly
        # (with L_CTB·κ0 as the reference) and the anisotropic correction
        # explicitly. The correction is always damping-additive, so stability
        # is preserved for time steps as large as the implicit baseline allows.
        kappa0 = float(params.get('kappa0', 1.0))
        L_CTB = float(params.get('L_CTB', 0.05))
        self.implicit_diff_phi = L_CTB * kappa0  # scalar, ≥ 0
        kappa_eta = float(params.get('kappa_eta', 2.0))
        L_eta = float(params.get('L_eta', 1.0))
        self.implicit_diff_eta = L_eta * kappa_eta
        
        # Twin confinement toggle
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
        df_deta1 = (2 * A * self.eta1 * (1 - self.eta1) * (1 - 2*self.eta1) +
                    2 * B * self.eta1 * self.eta2**2 +
                    2 * W * (self.phi**2 - 1)**2 * self.eta1)
        df_deta2 = (2 * A * self.eta2 * (1 - self.eta2) * (1 - 2*self.eta2) +
                    2 * B * self.eta2 * self.eta1**2)
        return df_dphi, df_deta1, df_deta2

    @handle_errors
    def compute_elastic_driving_force(self, sxx, syy, sxy):
        try:
            gamma_tw = self.mat_props['twinning']['gamma_tw']
            n = self.mat_props['twinning']['n_2d']
            a = self.mat_props['twinning']['a_2d']
            dh_dphi = 0.25 * (3*self.phi**2 - 2*self.phi - 1)
            nx, ny = n[0], n[1]
            ax, ay = a[0], a[1]
            deps_xx_dphi = gamma_tw * nx * ax * dh_dphi * self.eta1
            deps_yy_dphi = gamma_tw * ny * ay * dh_dphi * self.eta1
            deps_xy_dphi = 0.5 * gamma_tw * (nx * ay + ny * ax) * dh_dphi * self.eta1
            df_el_dphi = -(sxx * deps_xx_dphi + syy * deps_yy_dphi + 2 * sxy * deps_xy_dphi)
            return df_el_dphi
        except Exception as e:
            st.error(f"Error computing elastic driving force: {e}")
            return np.zeros_like(self.phi)

    @handle_errors
    def evolve_twin_field(self, sxx, syy, sxy, eps_p_mag):
        """
        Semi-implicit Fourier spectral update of φ.
        Linear part (L_CTB·κ0·∇²) is implicit; nonlinear + anisotropic
        corrections are explicit and FFT'd once per step.
        """
        kappa0 = float(self.params['kappa0'])
        gamma_aniso = float(self.params['gamma_aniso'])
        L_CTB = float(self.params.get('L_CTB', 0.05))
        L_ITB = float(self.params.get('L_ITB', 5.0))
        n_mob = int(self.params.get('n_mob', 4))
        zeta = float(self.params.get('zeta', 0.3))
       
        n_twin = self.mat_props['twinning']['n_2d']
        nx = float(n_twin[0])
        ny = float(n_twin[1])
       
        # ---- Exact spectral gradients / Laplacian ----
        phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
        lap_phi = spectral_laplacian(self.phi, self.k2)
       
        # Anisotropic κ_φ, L_φ (real space)
        kappa_phi, L_phi = compute_anisotropic_properties(
            phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso,
            L_CTB, L_ITB, n_mob)
       
        # Nonlinear driving forces (real space)
        df_loc_dphi, _, _ = self.compute_local_energy_derivatives()
        df_el_dphi = self.compute_elastic_driving_force(sxx, syy, sxy)
        diss_p = zeta * eps_p_mag * self.phi
       
        NLF = df_loc_dphi + df_el_dphi + diss_p
       
        # Full RHS operator written as R = -L_φ·NLF + (L_φ·κ_φ - L_ref·κ_ref)·∇²φ
        # where L_ref·κ_ref is the implicit baseline (here L_CTB·κ0).
        # The correction (L_φ·κ_φ - L_CTB·κ0) is ≥ 0 (damping-additive).
        correction = (L_phi * kappa_phi - self.implicit_diff_phi) * lap_phi
        R = -L_phi * NLF + correction
       
        # Twin confinement: freeze φ where η₁ ≈ 0
        if self.confine_twin:
            R = R * self.eta1**2
       
        # ---- Fourier-space semi-implicit step ----
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
       
        # RHS = -L_η·df_dη  (the implicit Laplacian is handled by the denominator)
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
                2/3 * (self.eps_p_xx**2 + self.eps_p_yy**2 +
                       2*self.eps_p_xy**2 + 1e-15))
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
        axes[0, 0].set_xlabel('Time (ns)'); axes[0, 0].set_ylabel('||φ||')
        axes[0, 0].set_title('Twin Order Parameter Norm')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(plot_timesteps, history['energy'], 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Time (ns)'); axes[0, 1].set_ylabel('Total Energy (J)')
        axes[0, 1].set_title('System Energy Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 2].plot(plot_timesteps, np.array(history['max_stress'])/1e9,
                        'g-', linewidth=2, alpha=0.8, label='Max')
        axes[0, 2].plot(plot_timesteps, np.array(history['avg_stress'])/1e9,
                        'g--', linewidth=1.5, alpha=0.6, label='Avg')
        axes[0, 2].set_xlabel('Time (ns)'); axes[0, 2].set_ylabel('Stress (GPa)')
        axes[0, 2].set_title('Stress Evolution'); axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[1, 0].plot(plot_timesteps, history['plastic_work'], 'm-',
                        linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Time (ns)'); axes[1, 0].set_ylabel('Plastic Work (J)')
        axes[1, 0].set_title('Plastic Work Evolution'); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(plot_timesteps, history['twin_spacing_avg'], 'c-',
                        linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Time (ns)'); axes[1, 1].set_ylabel('Avg Spacing (nm)')
        axes[1, 1].set_title('Average Twin Spacing'); axes[1, 1].grid(True, alpha=0.3)
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
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
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
        created_at = sim_data.get('metadata', {}).get('created_at', datetime.now().isoformat())
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
                    'sigma_eq_GPa': (frame['sigma_eq']/1e9).flatten(),
                    'sigma_h_GPa': (frame['sigma_h']/1e9).flatten(),
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
            if isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, np.generic): return obj.item()
            elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert(i) for i in obj]
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
                if isinstance(v, (int, float, str)):
                    param_grp.attrs[k] = v
                elif isinstance(v, np.ndarray):
                    param_grp.create_dataset(k, data=v)
            meta_grp = f.create_group('metadata')
            for k, v in sim_data.get('metadata', {}).items():
                if isinstance(v, (int, float, str)):
                    meta_grp.attrs[k] = v
                elif isinstance(v, dict):
                    subgrp = meta_grp.create_group(k)
                    for sk, sv in v.items():
                        subgrp.attrs[sk] = sv
            x = np.linspace(-N*dx/2, N*dx/2, N)
            y = np.linspace(-N*dx/2, N*dx/2, N)
            f.create_dataset('x', data=x)
            f.create_dataset('y', data=y)
            frame_grp = f.create_group('frames')
            for idx, frame in enumerate(history):
                grp = frame_grp.create_group(f'frame_{idx:04d}')
                for field in ['phi', 'eta1', 'eta2', 'sigma_eq', 'sigma_h',
                              'h', 'eps_p_mag', 'sigma_y']:
                    if field in frame:
                        grp.create_dataset(field, data=frame[field])
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
                            'sigma_eq_GPa': (frame['sigma_eq']/1e9).flatten(),
                            'sigma_h_GPa': (frame['sigma_h']/1e9).flatten(),
                            'h_nm': frame['h'].flatten()})
                        zf.writestr(f"{sim_dir}/frame_{idx:04d}.csv",
                                    df.to_csv(index=False))
                    summary += f"\nSimulation {sim_id}:\n"
                    summary += f"  Name: {sim_name}\n"
                    summary += f"  λ = {params.get('twin_spacing', 0):.1f} nm\n"
                    summary += f"  σ_app = {params.get('applied_stress', 0)/1e6:.0f} MPa\n"
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
            status_text.text(f"Running {param_name} = {val:.3f} ({i+1}/{len(values)})")
            params = base_params.copy()
            params[param_name] = val
            if 'history' in params: del params['history']
            if 'geom_viz' in params: del params['geom_viz']
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
    <strong>✅ MAJOR REWRITE — PURE FFT SPECTRAL METHOD:</strong><br>
    • <span style="color: green;">NO FDM/NUMBA:</span> all spatial derivatives are now exact spectral operators.<br>
    • <span style="color: green;">SEMI-IMPLICIT FOURIER:</span> the Laplacian is treated implicitly → unconditional stability for large Δt.<br>
    • <span style="color: green;">ANISOTROPY RETAINED:</span> κ_φ(m̂), L_φ(m̂) enter as explicit damping-additive corrections.<br>
    • <span style="color: green;">TWIN CONFINEMENT:</span> applied directly to the nonlinear driving force before FFT.<br>
    • <span style="color: green;">ALL UI / EXPORT / SWEEP / COMPARISON PRESERVED.</span><br>
    </div>
    """, unsafe_allow_html=True)
   
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
            keys_to_clear = ['N', 'dx', 'dt', 'twin_spacing', 'W', 'A', 'B',
                            'kappa0', 'gamma_aniso', 'kappa_eta', 'L_CTB',
                            'L_ITB', 'n_mob', 'L_eta', 'zeta', 'applied_stress',
                            'geom_type', 'defect_type', 'defect_x', 'defect_y',
                            'defect_radius', 'left_buffer_width', 'buffer_width',
                            'gb_profile', 'gb_curvature', 'grain_boundary_pos',
                            'stability_factor', 'enable_monitoring',
                            'auto_adjust_dt', 'n_steps', 'save_freq', 'confine_twin']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
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
            # With the semi-implicit scheme, dt is no longer CFL-restricted.
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
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal", "Custom"],
                        index=0)
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                    profile_type_mapping = {
                        "Horizontal": "horizontal", "Vertical": "vertical",
                        "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal",
                        "Custom": "custom"}
                    internal_direction = profile_type_mapping.get(profile_direction, "horizontal")
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
                applied_stress_sweep = st.slider("Applied stress (MPa)", 0.0, 600.0, 300.0,
                                                10.0, key="sweep_stress")
            st.subheader("Sweep Parameter")
            sweep_param = st.selectbox(
                "Choose parameter to vary",
                ["twin_spacing", "applied_stress", "applied_stress_angle",
                 "W", "L_CTB", "L_ITB", "kappa0"], index=0)
            col1, col2 = st.columns(2)
            with col1:
                sweep_min = st.number_input(
                    f"Min {sweep_param}",
                    value=10.0 if sweep_param=="twin_spacing" else 0.0, key="sweep_min")
                sweep_max = st.number_input(
                    f"Max {sweep_param}",
                    value=50.0 if sweep_param=="twin_spacing" else 500.0, key="sweep_max")
            with col2:
                sweep_steps = st.number_input("Number of steps", min_value=2, max_value=20,
                                              value=5, step=1, key="sweep_steps")
            if sweep_param in ["applied_stress"]:
                sweep_values = np.linspace(sweep_min*1e6, sweep_max*1e6, sweep_steps)
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
            sim_names = [build_sim_name(sim['params'], sim['id']) for sim in simulations]
            if config['type'] == "Side-by-Side Heatmaps":
                last_frames = []
                for sim in simulations:
                    last_frames.append(sim['results_history'][-1]
                                       if sim['results_history'] else None)
                valid_indices = [i for i, f in enumerate(last_frames) if f is not None]
                if not valid_indices:
                    st.warning("No frame data available.")
                else:
                    n_sims = len(valid_indices)
                    cols = min(3, n_sims)
                    rows = (n_sims + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
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
                            extent = [-sim['params']['N']*sim['params']['dx']/2,
                                      sim['params']['N']*sim['params']['dx']/2] * 2
                            im = ax.imshow(data, extent=extent, cmap='viridis',
                                          origin='lower')
                            ax.set_title(sim_names[sim_idx][:30] + "...", fontsize=8)
                            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    for idx in range(len(valid_indices), len(axes)):
                        axes[idx].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            elif config['type'] == "Overlay Line Profiles":
                fig = go.Figure()
                ref_sim = simulations[0]
                N = ref_sim['params']['N']; dx = ref_sim['params']['dx']
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
                        profile = profile / 1e9; ylabel = 'Stress (GPa)'
                    elif field == 'sigma_y':
                        profile = profile / 1e6; ylabel = 'Stress (MPa)'
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
                    params = sim['params']; hist = sim['results_history']
                    if hist:
                        last = hist[-1].get('convergence', {})
                        row = {
                            'Name': build_sim_name(params, sim['id'])[:40],
                            'λ (nm)': params.get('twin_spacing', 0),
                            'σ_app (MPa)': params.get('applied_stress', 0)/1e6,
                            'θ (deg)': params.get('applied_stress_angle', 0),
                            'W (J/m³)': params.get('W', 0),
                            'Avg σ_eq (GPa)': last.get('avg_stress', 0)/1e9,
                            'Max σ_eq (GPa)': last.get('max_stress', 0)/1e9,
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
                    fig2.add_trace(go.Scatter(x=df['λ (nm)'], y=df['Avg σ_eq (GPa)'],
                                             mode='markers+text',
                                             text=df['Name'], textposition='top center'))
                    fig2.update_layout(title="Twin Spacing vs. Avg Stress",
                                      xaxis_title="λ (nm)",
                                      yaxis_title="Avg Stress (GPa)")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No convergence data available.")
            elif config['type'] == "Correlation Analysis":
                data = []
                for sim in simulations:
                    params = sim['params']; hist = sim['results_history']
                    if hist and 'convergence' in hist[-1]:
                        conv = hist[-1]['convergence']
                        row = {
                            'twin_spacing': params.get('twin_spacing', 0),
                            'applied_stress': params.get('applied_stress', 0)/1e6,
                            'W': params.get('W', 0),
                            'avg_stress': conv.get('avg_stress', 0)/1e9,
                            'max_stress': conv.get('max_stress', 0)/1e9,
                            'avg_spacing': conv.get('avg_spacing', 0),
                            'plastic_work': conv.get('plastic_work', 0)}
                        data.append(row)
                if data:
                    df = pd.DataFrame(data)
                    fig = go.Figure(data=go.Splom(
                        dimensions=[dict(label=k, values=df[k]) for k in df.columns],
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
                chosen_metric = st.selectbox("Metric to track", list(metric_map.keys()))
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
                stress_mag = params.get('applied_stress', 0)/1e6
                angle = params.get('applied_stress_angle', 0)
                st.metric("σ_app / θ", f"{stress_mag:.0f} MPa / {angle:.0f}°")
            with col3:
                st.metric("W (well depth)", f"{params.get('W', 0):.2f} J/m³")
            with col4:
                st.metric("κ₀", f"{params.get('kappa0', 0):.2f}")
            history = sim_data.get('results_history', [])
            if history:
                num_frames = len(history)
                frame_idx = st.slider("Frame", 0, num_frames-1, num_frames-1,
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
        N = params['N']; dx = params['dx']
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
                avg_spacing = np.mean(h[(h>5)&(h<50)]) if np.any((h>5)&(h<50)) else 0
                st.metric("Avg Twin Spacing", f"{avg_spacing:.1f} nm")
            with col2:
                twin_area = np.sum(eta1 > 0.5) * dx**2
                st.metric("Twin Grain Area", f"{twin_area:.0f} nm²")
            with col3:
                num_twins = np.sum(h < 20)
                st.metric("Number of Twins", f"{num_twins:.0f}")
       
        with tabs[1]:
            st.header("Run Simulation (Pure FFT Spectral Method)")
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
                            status_text.text(f"Step {step+1}/{n_steps} | "
                                            f"Time: {(step+1)*dt:.4f} ns")
                            results = solver.step()
                            if step % save_freq == 0:
                                results_history.append(results.copy())
                                timesteps.append(step * dt)
                            progress_bar.progress((step + 1) / n_steps)
                            if step % 10 == 0 and len(results_history) > 0:
                                with monitoring_cols[0]:
                                    st.metric("Avg Stress",
                                              f"{np.mean(results['sigma_eq'])/1e9:.2f} GPa")
                                with monitoring_cols[1]:
                                    valid_h = results['h'][(results['h']>5)&(results['h']<50)]
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
                            run_time=(datetime.now()-start_time).total_seconds())
                        st.balloons()
                    except Exception as e:
                        st.error(f"Simulation failed: {str(e)}")
                        st.exception(e)
       
        with tabs[2]:
            if 'results_history' in st.session_state:
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
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                st.subheader("Convergence Monitoring")
                if hasattr(st.session_state, 'solver') and \
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
                                                  figsize=(10, 4*len(internal_types)))
                if len(internal_types) == 1:
                    axes = [axes]
                for idx, ptype in enumerate(internal_types):
                    ax = axes[idx]
                    distance, profile, _ = profiler.extract_profile(
                        results[field_to_profile], ptype, position_ratio)
                    if field_to_profile in ['sigma_eq', 'sigma_h']:
                        profile = profile / 1e9; ylabel = 'Stress (GPa)'
                    elif field_to_profile == 'sigma_y':
                        profile = profile / 1e6; ylabel = 'Stress (MPa)'
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
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h", "eps_p_mag", "sigma_y"],
                    index=0, key="plotly_2d_field")
                frame_idx_plotly = st.slider("Frame", 0, len(results_history)-1,
                                            len(results_history)-1, key="plotly_2d_frame")
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
                    position_ratio_plotly = st.slider("Position ratio", 0.0, 1.0, 0.5, 0.05,
                                                      key="plotly_2d_pos")
                profile_type_mapping = {
                    "Horizontal": "horizontal", "Vertical": "vertical",
                    "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal"}
                internal_pt = profile_type_mapping.get(profile_type_plotly, "horizontal")
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
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h", "eps_p_mag", "sigma_y"],
                    index=1, key="3d_field")
                frame_idx_3d = st.slider("Frame", 0, len(results_history)-1,
                                        len(results_history)-1, key="3d_frame")
                results = results_history[frame_idx_3d]
                fig_3d = visualizer.create_plotly_3d_surface(results, field_3d, frame_idx_3d)
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
                    anim_format = st.selectbox("Format", ["gif", "mp4"], key="anim_format")
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
        axes[0,0].plot(param_display, avg_stress, 'o-', linewidth=2)
        axes[0,0].set_xlabel(param_label)
        axes[0,0].set_ylabel("Avg Von Mises Stress (GPa)")
        axes[0,0].set_title("Stress vs Parameter")
        axes[0,0].grid(True, alpha=0.3)
        axes[0,1].plot(param_display, avg_spacing, 's-', color='green', linewidth=2)
        axes[0,1].set_xlabel(param_label)
        axes[0,1].set_ylabel("Avg Twin Spacing (nm)")
        axes[0,1].set_title("Twin Spacing vs Parameter")
        axes[0,1].grid(True, alpha=0.3)
        axes[1,0].plot(param_display, plastic_work, 'd-', color='red', linewidth=2)
        axes[1,0].set_xlabel(param_label)
        axes[1,0].set_ylabel("Plastic Work (J)")
        axes[1,0].set_title("Plastic Work vs Parameter")
        axes[1,0].grid(True, alpha=0.3)
        axes[1,1].plot(param_display, max_stress, '^-', color='purple',
                       linewidth=2, label='Max')
        axes[1,1].plot(param_display, avg_stress, 'o-', color='blue',
                       linewidth=2, label='Avg')
        axes[1,1].set_xlabel(param_label)
        axes[1,1].set_ylabel("Stress (GPa)")
        axes[1,1].set_title("Stress Extremes vs Parameter")
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
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

if __name__ == "__main__":
    main()
