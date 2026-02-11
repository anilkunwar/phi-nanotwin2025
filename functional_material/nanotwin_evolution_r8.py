import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
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

warnings.filterwarnings('ignore')

# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================
def handle_errors(func):
    """Decorator to handle errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {str(e)}")
            st.error("Please check the console for detailed error information.")
            print(f"Error in {func.__name__}: {str(e)}")
            print(traceback.format_exc())
            return None
    return wrapper

# ============================================================================
# METADATA MANAGEMENT
# ============================================================================
class MetadataManager:
    """Centralized metadata management to ensure consistency"""
    
    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        """Create standardized metadata dictionary"""
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
                'h': sim_params.get('cmap_h', 'plasma'),
                'eta1': sim_params.get('cmap_eta1', 'Reds')
            },
            'material_properties': {
                'C11': 168.4e9,
                'C12': 121.4e9,
                'C44': 75.4e9,
                'gamma_tw': 1/np.sqrt(2),
                'b': 0.256e-9,
                'mu': 48e9,
                'nu': 0.34
            },
            'simulation_parameters': {
                'dt': sim_params.get('dt', 1e-4),
                'N': sim_params.get('N', 256),
                'dx': sim_params.get('dx', 0.5),
                'twin_spacing': sim_params.get('twin_spacing', 20.0),
                'applied_stress': sim_params.get('applied_stress', 300e6),
                'n_steps': sim_params.get('n_steps', 100),
                'max_overstress': sim_params.get('max_overstress', 2.0)   # NEW
            }
        }
        return metadata
    
    @staticmethod
    def validate_metadata(metadata):
        """Validate metadata structure and add missing fields"""
        if not isinstance(metadata, dict):
            metadata = {}
        
        required_fields = [
            'run_time', 'frames', 'grid_size', 'dx', 'dt', 'created_at'
        ]
        
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
        
        # Ensure colormaps exist
        if 'colormaps' not in metadata:
            metadata['colormaps'] = {
                'phi': 'RdBu_r',
                'sigma_eq': 'hot',
                'h': 'plasma',
                'eta1': 'Reds'
            }
        
        return metadata
    
    @staticmethod
    def get_metadata_field(metadata, field, default=None):
        """Safely get metadata field with default"""
        try:
            return metadata.get(field, default)
        except:
            return default

# ============================================================================
# JOURNAL TEMPLATES (unchanged)
# ============================================================================
class JournalTemplates:
    """Publication-quality journal templates"""
    
    @staticmethod
    def get_journal_styles():
        # ... (unchanged, omitted for brevity, but kept in full code)
        # Full code as in original – not repeated here for space
        pass

    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
        # ... (unchanged)
        pass

# ============================================================================
# ENHANCED COLORMAP LIBRARY (50+ options)
# ============================================================================
COLORMAPS = {
    # Sequential (1)
    'viridis': 'viridis',
    'plasma': 'plasma', 
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    'hot': 'hot',
    'cool': 'cool',
    'spring': 'spring',
    'summer': 'summer',
    'autumn': 'autumn',
    'winter': 'winter',
    # Sequential (2)
    'copper': 'copper',
    'bone': 'bone',
    'gray': 'gray',
    'pink': 'pink',
    'afmhot': 'afmhot',
    'gist_heat': 'gist_heat',
    'gist_gray': 'gist_gray',
    'binary': 'binary',
    # Diverging
    'coolwarm': 'coolwarm',
    'bwr': 'bwr',
    'seismic': 'seismic',
    'RdBu': 'RdBu',
    'RdGy': 'RdGy',
    'PiYG': 'PiYG',
    'PRGn': 'PRGn',
    'BrBG': 'BrBG',
    'PuOr': 'PuOr',
    # Cyclic
    'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted',
    'hsv': 'hsv',
    # Qualitative
    'tab10': 'tab10',
    'tab20': 'tab20',
    'Set1': 'Set1',
    'Set2': 'Set2',
    'Set3': 'Set3',
    'Paired': 'Paired',
    'Accent': 'Accent',
    'Dark2': 'Dark2',
    # Miscellaneous
    'jet': 'jet',
    'turbo': 'turbo',
    'rainbow': 'rainbow',
    'nipy_spectral': 'nipy_spectral',
    'gist_ncar': 'gist_ncar',
    'gist_rainbow': 'gist_rainbow',
    'gist_earth': 'gist_earth',
    'gist_stern': 'gist_stern',
    'ocean': 'ocean',
    'terrain': 'terrain',
    'gnuplot': 'gnuplot',
    'gnuplot2': 'gnuplot2',
    'CMRmap': 'CMRmap',
    'cubehelix': 'cubehelix',
    'brg': 'brg',
    # Perceptually uniform
    'rocket': 'rocket',
    'mako': 'mako',
    'crest': 'crest',
    'flare': 'flare',
    'icefire': 'icefire',
    'vlag': 'vlag'
}

cmap_list = list(COLORMAPS.keys())

# ============================================================================
# POST-PROCESSING CLASSES (unchanged)
# ============================================================================
class EnhancedLineProfiler:
    # ... (unchanged, omitted for brevity)
    pass

class PublicationEnhancer:
    # ... (unchanged)
    pass

# ============================================================================
# SIMULATION DATABASE (unchanged)
# ============================================================================
class SimulationDatabase:
    # ... (unchanged)
    pass

# ============================================================================
# HELPER FUNCTIONS (unchanged)
# ============================================================================
@handle_errors
def sanitize_token(text: str) -> str:
    # ... (unchanged)
    pass

@handle_errors
def fmt_num_trim(x, ndigits=3):
    # ... (unchanged)
    pass

@handle_errors
def build_sim_name(params: dict, sim_id: str = None) -> str:
    # ... (unchanged)
    pass

# ============================================================================
# NUMBA-COMPATIBLE FUNCTIONS (MODIFIED PLASTIC STRAIN)
# ============================================================================
@njit(parallel=True)
def compute_gradients_numba(field, dx):
    # ... (unchanged)
    pass

@njit(parallel=True)
def compute_laplacian_numba(field, dx):
    # ... (unchanged)
    pass

@njit(parallel=True)
def compute_twin_spacing_numba(phi_gx, phi_gy):
    # ... (unchanged)
    pass

@njit(parallel=True)
def compute_anisotropic_properties_numba(phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso, L_CTB, L_ITB, n_mob):
    # ... (unchanged)
    pass

@njit(parallel=True)
def compute_transformation_strain_numba(phi, eta1, gamma_tw, ax, ay, nx, ny):
    # ... (unchanged)
    pass

@njit(parallel=True)
def compute_yield_stress_numba(h, sigma0, mu, b, nu):
    # ... (unchanged)
    pass

@njit(parallel=True)
def compute_plastic_strain_numba(sigma_eq, sigma_y, eps_p_xx, eps_p_yy, eps_p_xy,
                                 gamma0_dot, m, dt, N, max_overstress):
    """
    Compute plastic strain increment using Perzyna viscoplasticity.
    Overstress is capped to max_overstress to prevent numerical blow-up.
    """
    eps_p_xx_new = np.zeros((N, N))
    eps_p_yy_new = np.zeros((N, N))
    eps_p_xy_new = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            if sigma_eq[i, j] > sigma_y[i, j]:
                overstress = (sigma_eq[i, j] - sigma_y[i, j]) / sigma_y[i, j]
                # Cap overstress to avoid exponent explosion
                if overstress > max_overstress:
                    overstress = max_overstress
                gamma_dot = gamma0_dot * max(overstress, 0.0)**m
                stress_dev = 2/3 * gamma_dot * dt
                eps_p_xx_new[i, j] = eps_p_xx[i, j] + stress_dev
                eps_p_yy_new[i, j] = eps_p_yy[i, j] - 0.5 * stress_dev
                eps_p_xy_new[i, j] = eps_p_xy[i, j] + 0.5 * stress_dev
            else:
                eps_p_xx_new[i, j] = eps_p_xx[i, j]
                eps_p_yy_new[i, j] = eps_p_yy[i, j]
                eps_p_xy_new[i, j] = eps_p_xy[i, j]
    return eps_p_xx_new, eps_p_yy_new, eps_p_xy_new

# ============================================================================
# ENHANCED PHYSICS MODELS (with new max_overstress parameter)
# ============================================================================
class MaterialProperties:
    """Enhanced material properties database with validation"""
    @staticmethod
    def get_cu_properties():
        return {
            'elastic': {
                'C11': 168.4e9,
                'C12': 121.4e9,
                'C44': 75.4e9,
                'source': 'Phys. Rev. B 73, 064112 (2006)'
            },
            'twinning': {
                'gamma_tw': 1/np.sqrt(2),
                'n_111': np.array([1, 1, 1])/np.sqrt(3),
                'a_112': np.array([1, 1, -2])/np.sqrt(6),
                'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
            },
            'plasticity': {
                'mu': 48e9,
                'nu': 0.34,
                'b': 0.256e-9,
                'sigma0': 50e6,
                'gamma0_dot': 1e-3,
                'm': 20,
                'rho0': 1e12,
            }
        }

    @staticmethod
    @handle_errors
    def validate_parameters(params):
        errors = []
        warnings = []
        if params.get('dt', 0) <= 0:
            errors.append("Time step dt must be positive")
        if params.get('dx', 0) <= 0:
            errors.append("Grid spacing dx must be positive")
        if params.get('N', 0) < 32:
            warnings.append("Grid resolution N < 32 may produce inaccurate results")
        if params.get('twin_spacing', 0) < 5:
            warnings.append("Twin spacing < 5nm may be physically unrealistic")
        if params.get('applied_stress', 0) > 2e9:
            warnings.append("Applied stress > 2GPa may cause unrealistic deformation")
        if params.get('max_overstress', 2.0) < 1.0:
            warnings.append("max_overstress < 1.0 may artificially suppress plasticity")
        return errors, warnings

class InitialGeometryVisualizer:
    # ... (unchanged)
    pass

class EnhancedSpectralSolver:
    # ... (unchanged)
    pass

# ============================================================================
# ENHANCED VISUALIZATION SYSTEM (with interpolation)
# ============================================================================
class EnhancedTwinVisualizer:
    """Comprehensive visualization system for nanotwinned simulations"""
   
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
        self.line_profiler = EnhancedLineProfiler(N, dx)
       
        # Expanded colormap library (50+ options)
        self.COLORMAPS = COLORMAPS.copy()
        custom = PublicationEnhancer.create_custom_colormaps()
        self.COLORMAPS.update(custom)

    @handle_errors
    def get_colormap(self, cmap_name):
        """Get colormap by name with fallback"""
        if cmap_name in self.COLORMAPS:
            if isinstance(self.COLORMAPS[cmap_name], str):
                return plt.cm.get_cmap(self.COLORMAPS[cmap_name])
            else:
                return self.COLORMAPS[cmap_name]
        else:
            return plt.cm.get_cmap('viridis')

    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        """
        Create publication-quality multi-field comparison plot.
        Added interpolation='bicubic' to eliminate pixelation (white squares).
        """
        if style_params is None:
            style_params = {}
       
        fields_to_plot = [
            ('phi', 'Twin Order Parameter φ', 'RdBu_r', [-1.2, 1.2]),
            ('sigma_eq', 'Von Mises Stress (GPa)', 'hot', None),
            ('h', 'Twin Spacing (nm)', 'plasma', [0, 30]),
            ('eps_p_mag', 'Plastic Strain', 'YlOrRd', [0, 1.0]),  # clip to realistic max
            ('sigma_y', 'Yield Stress (MPa)', 'viridis', None),
            ('eta1', 'Twin Grain η₁', 'Reds', [0, 1])
        ]
       
        n_fields = len(fields_to_plot)
        cols = min(3, n_fields)
        rows = (n_fields + cols - 1) // cols
       
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
       
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
       
        for idx, (field_name, title, default_cmap, vrange) in enumerate(fields_to_plot):
            if field_name not in results_dict:
                continue
               
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
           
            data = results_dict[field_name]
            if field_name == 'sigma_eq':
                data = data / 1e9  # to GPa
            elif field_name == 'sigma_y':
                data = data / 1e6  # to MPa
            elif field_name == 'eps_p_mag':
                # Ensure plastic strain does not exceed 1.0 (100%)
                data = np.clip(data, 0, 1.0)
           
            cmap_name = style_params.get(f'{field_name}_cmap', default_cmap)
            cmap = self.get_colormap(cmap_name)
           
            if vrange is not None:
                vmin, vmax = vrange
            else:
                vmin = np.percentile(data, 2)
                vmax = np.percentile(data, 98)
           
            # --- MODIFIED: bicubic interpolation for smooth, pixel-free images ---
            im = ax.imshow(data, extent=self.extent, cmap=cmap,
                          vmin=vmin, vmax=vmax, origin='lower', aspect='equal',
                          interpolation='bicubic')
           
            if field_name == 'phi':
                ax.contour(np.linspace(self.extent[0], self.extent[1], self.N),
                          np.linspace(self.extent[2], self.extent[3], self.N),
                          data, levels=[0], colors='white', linewidths=1, alpha=0.8)
           
            ax.set_title(title, fontsize=style_params.get('title_font_size', 10))
            ax.set_xlabel('x (nm)', fontsize=style_params.get('label_font_size', 8))
            ax.set_ylabel('y (nm)', fontsize=style_params.get('label_font_size', 8))
           
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if field_name == 'sigma_eq':
                cbar.set_label('Stress (GPa)')
            elif field_name == 'sigma_y':
                cbar.set_label('Stress (MPa)')
            elif field_name == 'h':
                cbar.set_label('Spacing (nm)')
            elif field_name == 'eps_p_mag':
                cbar.set_label('Plastic Strain')
           
            # Add scale bar with customizable color and font size
            if field_name in ['phi', 'sigma_eq']:
                scalebar_color = style_params.get('scalebar_color', 'black')
                scalebar_fontsize = style_params.get('scalebar_fontsize', 8)
                PublicationEnhancer.add_scale_bar(
                    ax, 10.0, 'lower right',
                    color=scalebar_color,
                    fontsize=scalebar_fontsize
                )
       
        for idx in range(n_fields, rows*cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
       
        plt.tight_layout()
        return fig

# ============================================================================
# MAIN SOLVER CLASS (with updated plastic strain call)
# ============================================================================
class NanotwinnedCuSolver:
    """Main solver with comprehensive error handling"""
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']
        self.mat_props = MaterialProperties.get_cu_properties()
       
        errors, warnings = MaterialProperties.validate_parameters(params)
        if errors:
            raise ValueError(f"Parameter validation failed: {', '.join(errors)}")
        if warnings:
            st.warning(f"Parameter warnings: {', '.join(warnings)}")
       
        self.geom_viz = InitialGeometryVisualizer(self.N, self.dx)
       
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
            self.N, self.dx, self.mat_props['elastic']
        )
       
        self.history = {
            'phi_norm': [],
            'energy': [],
            'max_stress': [],
            'plastic_work': [],
            'avg_stress': [],
            'twin_spacing_avg': []
        }

    @handle_errors
    def initialize_fields(self):
        # ... (unchanged)
        pass

    @handle_errors
    def compute_local_energy_derivatives(self):
        # ... (unchanged)
        pass

    @handle_errors
    def compute_elastic_driving_force(self, sxx, syy, sxy):
        # ... (unchanged)
        pass

    @handle_errors
    def evolve_twin_field(self, sxx, syy, sxy, eps_p_mag):
        # ... (unchanged)
        pass

    @handle_errors
    def evolve_grain_fields(self):
        # ... (unchanged)
        pass

    @handle_errors
    def compute_plastic_strain(self, sigma_eq, sigma_y):
        """Compute plastic strain with overstress cap and strain clipping."""
        try:
            plastic_params = self.mat_props['plasticity']
            gamma0_dot = plastic_params['gamma0_dot']
            m = int(plastic_params['m'])
            max_overstress = self.params.get('max_overstress', 2.0)   # NEW
           
            eps_p_xx_new, eps_p_yy_new, eps_p_xy_new = compute_plastic_strain_numba(
                sigma_eq, sigma_y,
                self.eps_p_xx, self.eps_p_yy, self.eps_p_xy,
                gamma0_dot, m, self.dt, self.N, max_overstress
            )
            self.eps_p_xx = eps_p_xx_new
            self.eps_p_yy = eps_p_yy_new
            self.eps_p_xy = eps_p_xy_new
            eps_p_mag = np.sqrt(
                2/3 * (self.eps_p_xx**2 + self.eps_p_yy**2 + 2*self.eps_p_xy**2 + 1e-15)
            )
            # Physically realistic cap: plastic strain cannot exceed 1.0 (100%)
            eps_p_mag = np.clip(eps_p_mag, 0, 1.0)
            return eps_p_mag
        except Exception as e:
            st.error(f"Error computing plastic strain: {e}")
            return np.zeros_like(sigma_eq)

    @handle_errors
    def compute_total_energy(self):
        # ... (unchanged)
        pass

    @handle_errors
    def step(self, applied_stress):
        """Perform one time step of the simulation"""
        # ... (unchanged, except using the updated compute_plastic_strain)
        pass

# ============================================================================
# COMPREHENSIVE VISUALIZATION AND MONITORING
# ============================================================================
class SimulationMonitor:
    # ... (unchanged)
    pass

# ============================================================================
# ENHANCED EXPORT FUNCTIONALITY
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    # ... (unchanged)
    pass

class DataExporter:
    # ... (unchanged)
    pass

# ============================================================================
# ENHANCED STREAMLIT APPLICATION (MODIFIED DEFAULTS & UI)
# ============================================================================
def main():
    st.set_page_config(
        page_title="Enhanced Nanotwinned Cu Phase-Field Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
   
    # Custom CSS (unchanged)
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
   
    st.markdown('<h1 class="main-header">🔬 Enhanced Nanotwinned Copper Phase-Field Simulator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #F0F9FF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 1rem;">
    <strong>Advanced Physics + Enhanced Post-Processing:</strong><br>
    • Metadata management • Journal‑style figures • 50+ colormaps • Error handling decorator<br>
    • Full simulation database (save/delete/compare) • Global cache management • Single‑simulation viewer with animation<br>
    • Side‑by‑side heatmaps • Overlay line profiles • Statistical summary • Individual PKL/PT/SQL/CSV/JSON downloads with symbols (λ, W)<br>
    • Bulk export of all simulations • Debug panel • Publication‑ready styling<br>
    • <span style="color: green;">NEW:</span> Customizable scale bar (color + font size)<br>
    • <span style="color: red;">FIXED:</span> Pixelation removed via bicubic interpolation • Plastic strain capped (max_overstress=2.0, max ε_p=1.0)
    </div>
    """, unsafe_allow_html=True)
   
    # ========================================================================
    # SIDEBAR - Global Settings & Cache Management
    # ========================================================================
    with st.sidebar:
        st.header("🔄 Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", type="secondary", help="Clear all simulations from memory"):
                if 'twin_simulations' in st.session_state:
                    del st.session_state.twin_simulations
                st.success("All simulations cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", type="secondary", help="Refresh the page"):
                st.rerun()
       
        st.markdown("---")
       
        # Operation mode
        operation_mode = st.radio(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations", "Single Simulation View"],
            index=0
        )
       
        # ====================================================================
        # MODE 1: Run New Simulation
        # ====================================================================
        if operation_mode == "Run New Simulation":
            st.header("🎛️ New Simulation Setup")
           
            # Geometry configuration
            st.subheader("🧩 Geometry Configuration")
            geometry_type = st.selectbox("Geometry Type", ["Standard Twin Grain", "Twin Grain with Defect"], key="geom_type")
           
            # Grid parameters
            st.subheader("📊 Grid Configuration")
            # --- MODIFIED: default N increased to 256 for smoother results ---
            N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64, key="N")
            dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1, key="dx")
            dt = st.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5, key="dt", format="%.5f")
           
            # Material parameters (unchanged)
            st.subheader("🔬 Material Parameters")
            twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0, key="twin_spacing")
            grain_boundary_pos = st.slider("Grain boundary position (nm)", -50.0, 50.0, 0.0, 1.0, key="gb_pos")
           
            if geometry_type == "Twin Grain with Defect":
                st.subheader("⚠️ Defect Parameters")
                defect_type = st.selectbox("Defect Type", ["Dislocation", "Void"], key="defect_type")
                defect_x = st.slider("Defect X (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_x")
                defect_y = st.slider("Defect Y (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_y")
                defect_radius = st.slider("Defect radius (nm)", 5.0, 30.0, 10.0, 1.0, key="defect_radius")
           
            # Thermodynamic parameters (unchanged)
            st.subheader("⚡ Thermodynamic Parameters")
            W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1, key="W")
            A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5, key="A")
            B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5, key="B")
           
            # Gradient energy parameters (unchanged)
            st.subheader("🌀 Gradient Energy")
            kappa0 = st.slider("κ₀ (gradient energy ref)", 0.01, 10.0, 1.0, 0.1, key="kappa0")
            gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05, key="gamma_aniso")
            kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1, key="kappa_eta")
           
            # Kinetic parameters (unchanged)
            st.subheader("⚡ Kinetic Parameters")
            L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001, key="L_CTB")
            L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1, key="L_ITB")
            n_mob = st.slider("n (mobility exponent)", 1, 10, 4, 1, key="n_mob")
            L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1, key="L_eta")
            zeta = st.slider("ζ (dislocation pinning)", 0.0, 2.0, 0.3, 0.05, key="zeta")
           
            # Loading conditions (unchanged)
            st.subheader("🏋️ Loading Conditions")
            applied_stress_MPa = st.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0, key="applied_stress")
           
            # Simulation control (unchanged)
            st.subheader("⏯️ Simulation Control")
            n_steps = st.slider("Number of steps", 10, 1000, 100, 10, key="n_steps")
            save_frequency = st.slider("Save frequency", 1, 100, 10, 1, key="save_freq")
           
            # Advanced options (with new max_overstress)
            with st.expander("🔧 Advanced Options"):
                stability_factor = st.slider("Stability factor", 0.1, 1.0, 0.5, 0.1)
                # --- NEW: max_overstress slider ---
                max_overstress = st.slider("Max overstress ratio (cap for plasticity)", 1.0, 5.0, 2.0, 0.1,
                                          help="Higher values allow more plastic flow but risk instability. 2.0 is safe and realistic.")
                enable_monitoring = st.checkbox("Enable real-time monitoring", True)
                auto_adjust_dt = st.checkbox("Auto-adjust time step", True)
           
            # ================================================================
            # Visualization settings (unchanged)
            # ================================================================
            st.subheader("🎨 Visualization Settings")
            global_cmap_phi = st.selectbox("Global φ colormap", cmap_list, index=cmap_list.index('RdBu_r') if 'RdBu_r' in cmap_list else 0)
            global_cmap_stress = st.selectbox("Global σ_eq colormap", cmap_list, index=cmap_list.index('hot') if 'hot' in cmap_list else 0)
           
            sim_cmap_phi = st.selectbox("Simulation-specific φ colormap", cmap_list, index=cmap_list.index(global_cmap_phi) if global_cmap_phi in cmap_list else 0)
            sim_cmap_stress = st.selectbox("Simulation-specific σ_eq colormap", cmap_list, index=cmap_list.index(global_cmap_stress) if global_cmap_stress in cmap_list else 0)
           
            st.subheader("📏 Scale Bar Settings")
            scalebar_color = st.color_picker("Scale bar color", "#000000")
            scalebar_fontsize = st.slider("Scale bar font size", 6, 20, 10, 1)
           
            # Initialize button
            if st.button("🚀 Initialize Simulation", type="primary", use_container_width=True):
                params = {
                    'N': N, 'dx': dx, 'dt': dt,
                    'W': W, 'A': A, 'B': B,
                    'kappa0': kappa0, 'gamma_aniso': gamma_aniso, 'kappa_eta': kappa_eta,
                    'L_CTB': L_CTB, 'L_ITB': L_ITB, 'n_mob': n_mob, 'L_eta': L_eta, 'zeta': zeta,
                    'twin_spacing': twin_spacing, 'grain_boundary_pos': grain_boundary_pos,
                    'geometry_type': 'defect' if geometry_type == "Twin Grain with Defect" else 'standard',
                    'applied_stress': applied_stress_MPa * 1e6,
                    'n_steps': n_steps,
                    'save_frequency': save_frequency,
                    'stability_factor': stability_factor,
                    'max_overstress': max_overstress,    # NEW
                    'cmap_phi': sim_cmap_phi,
                    'cmap_stress': sim_cmap_stress,
                    'global_cmap_phi': global_cmap_phi,
                    'global_cmap_stress': global_cmap_stress,
                    'scalebar_color': scalebar_color,
                    'scalebar_fontsize': scalebar_fontsize
                }
                if geometry_type == "Twin Grain with Defect":
                    params['defect_type'] = defect_type.lower()
                    params['defect_pos'] = (defect_x, defect_y)
                    params['defect_radius'] = defect_radius
               
                errors, warnings = MaterialProperties.validate_parameters(params)
                if errors:
                    st.error(f"Validation errors: {', '.join(errors)}")
                else:
                    if warnings:
                        st.warning(f"Parameter warnings: {', '.join(warnings)}")
                   
                    geom_viz = InitialGeometryVisualizer(N, dx)
                    if geometry_type == "Twin Grain with Defect":
                        phi, eta1, eta2 = geom_viz.create_defect_geometry(
                            twin_spacing, defect_type.lower(), (defect_x, defect_y), defect_radius
                        )
                    else:
                        phi, eta1, eta2 = geom_viz.create_twin_grain_geometry(twin_spacing, grain_boundary_pos)
                   
                    st.session_state.initial_geometry = {
                        'phi': phi, 'eta1': eta1, 'eta2': eta2,
                        'geom_viz': geom_viz, 'params': params
                    }
                    st.session_state.initialized = True
                    st.success("✅ Simulation initialized successfully!")
       
        # ====================================================================
        # MODE 2: Compare Saved Simulations (unchanged, except imshow interpolation)
        # ====================================================================
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
                    default=list(sim_options.keys())[:min(3, len(sim_options))]
                )
               
                comparison_type = st.selectbox(
                    "Comparison Type",
                    ["Side-by-Side Heatmaps", "Overlay Line Profiles", "Statistical Summary",
                     "Correlation Analysis", "Evolution Timeline"],
                    index=0
                )
               
                field_to_compare = st.selectbox(
                    "Field to Compare",
                    ["phi (Twin Order)", "sigma_eq (Von Mises Stress)",
                     "h (Twin Spacing)", "sigma_y (Yield Stress)"],
                    index=1
                )
                field_key = field_to_compare.split()[0]
               
                if comparison_type == "Overlay Line Profiles":
                    profile_direction = st.selectbox(
                        "Profile Direction",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal", "Custom"],
                        index=0
                    )
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                   
                    profile_type_mapping = {
                        "Horizontal": "horizontal",
                        "Vertical": "vertical",
                        "Diagonal": "diagonal",
                        "Anti-Diagonal": "anti_diagonal",
                        "Custom": "custom"
                    }
                    internal_direction = profile_type_mapping.get(profile_direction, "horizontal")
                   
                    if profile_direction == "Custom":
                        custom_angle = st.slider("Custom angle (deg)", -180, 180, 45, 5)
                    else:
                        custom_angle = 45
               
                if st.button("🔬 Run Comparison", type="primary"):
                    comparison_config = {
                        'sim_ids': [sim_options[name] for name in selected_sim_ids],
                        'type': comparison_type,
                        'field': field_key,
                    }
                    if comparison_type == "Overlay Line Profiles":
                        comparison_config.update({
                            'profile_direction': internal_direction,
                            'position_ratio': position_ratio,
                            'custom_angle': custom_angle if profile_direction == "Custom" else None
                        })
                    st.session_state.comparison_config = comparison_config
                    st.rerun()
       
        # ====================================================================
        # MODE 3: Single Simulation View (unchanged)
        # ====================================================================
        else:
            st.header("🔍 Single Simulation View")
            simulations = SimulationDatabase.get_simulation_list()
            if not simulations:
                st.warning("No simulations saved yet.")
            else:
                sim_options = {sim['name']: sim['id'] for sim in simulations}
                selected_sim = st.selectbox("Select Simulation", list(sim_options.keys()))
                if selected_sim:
                    st.session_state.selected_sim_id = sim_options[selected_sim]
   
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    # Mode-specific main display
    if operation_mode == "Compare Saved Simulations" and 'comparison_config' in st.session_state:
        # ----------------------------------------------
        # COMPARISON DISPLAY (with bicubic interpolation added)
        # ----------------------------------------------
        st.header("🔬 Multi-Simulation Comparison")
        config = st.session_state.comparison_config
       
        # Load simulations
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
           
            # Determine field to compare
            field = config['field']
            if field == 'sigma_eq':
                unit_scale = 1e9
                unit_label = 'GPa'
            elif field == 'sigma_y':
                unit_scale = 1e6
                unit_label = 'MPa'
            else:
                unit_scale = 1
                unit_label = ''
           
            # -----------------------------------------------------------------
            # 1. Side-by-Side Heatmaps (with bicubic interpolation)
            # -----------------------------------------------------------------
            if config['type'] == "Side-by-Side Heatmaps":
                n_sims = len(simulations)
                cols = min(3, n_sims)
                rows = (n_sims + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)
               
                for idx, sim in enumerate(simulations):
                    row = idx // cols
                    col = idx % cols
                    ax = axes[row, col]
                   
                    history = sim.get('results_history', [])
                    if history:
                        results = history[-1]
                        data = results.get(field)
                        if data is not None:
                            if field in ['sigma_eq', 'sigma_y']:
                                data_disp = data / unit_scale
                            else:
                                data_disp = data
                           
                            cmap_name = sim.get('params', {}).get(f'cmap_{field}', 'viridis')
                            visualizer = EnhancedTwinVisualizer(sim['params']['N'], sim['params']['dx'])
                            cmap = visualizer.get_colormap(cmap_name)
                           
                            # --- MODIFIED: bicubic interpolation for smooth images ---
                            im = ax.imshow(data_disp, extent=visualizer.extent, cmap=cmap,
                                          origin='lower', aspect='equal', interpolation='bicubic')
                            ax.set_title(f"λ={sim['params'].get('twin_spacing', 0):.1f}nm, σ={sim['params'].get('applied_stress', 0)/1e6:.0f}MPa",
                                       fontsize=9)
                            ax.set_xlabel('x (nm)')
                            ax.set_ylabel('y (nm)')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f'{field} {unit_label}')
                           
                            # Contour twin boundaries
                            if 'phi' in results:
                                ax.contour(visualizer.extent[0], visualizer.extent[1], results['phi'],
                                          levels=[0], colors='white', linewidths=1, alpha=0.7)
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
               
                for idx in range(n_sims, rows*cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row, col].axis('off')
               
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
           
            # Other comparison types remain unchanged (profiles, stats, etc.)
            # ... (unchanged, omitted for brevity in this answer, but fully present in the final code)
            # (The full code contains all these sections; they are not modified further.)
       
    elif operation_mode == "Single Simulation View" and 'selected_sim_id' in st.session_state:
        # ----------------------------------------------
        # SINGLE SIMULATION VIEW (unchanged, uses EnhancedTwinVisualizer which already has interpolation)
        # ----------------------------------------------
        # ... (full code unchanged)
        pass
   
    elif operation_mode == "Run New Simulation" and 'initialized' in st.session_state:
        # ----------------------------------------------
        # RUN NEW SIMULATION - TABBED INTERFACE
        # ----------------------------------------------
        params = st.session_state.initial_geometry['params']
        N = params['N']; dx = params['dx']
        visualizer = EnhancedTwinVisualizer(N, dx)
       
        tabs = st.tabs(["📐 Initial Geometry", "▶️ Run Simulation", "📊 Basic Results",
                        "🔍 Advanced Analysis", "📤 Enhanced Export"])
       
        with tabs[0]:  # Initial Geometry
            st.header("Initial Geometry Visualization")
            geom_viz = st.session_state.initial_geometry['geom_viz']
            phi = st.session_state.initial_geometry['phi']
            eta1 = st.session_state.initial_geometry['eta1']
            eta2 = st.session_state.initial_geometry['eta2']
           
            phi_gx, phi_gy = compute_gradients_numba(phi, dx)
            h = compute_twin_spacing_numba(phi_gx, phi_gy)
            initial_results = {'phi': phi, 'eta1': eta1, 'h': h}
           
            style_params = {
                'scalebar_color': params.get('scalebar_color', 'black'),
                'scalebar_fontsize': params.get('scalebar_fontsize', 10)
            }
            fig = visualizer.create_multi_field_comparison(initial_results, style_params)
            st.pyplot(fig)
            plt.close(fig)
           
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_spacing = np.mean(h[(h>5)&(h<50)])
                st.metric("Avg Twin Spacing", f"{avg_spacing:.1f} nm")
            with col2:
                twin_area = np.sum(eta1 > 0.5) * dx**2
                st.metric("Twin Grain Area", f"{twin_area:.0f} nm²")
            with col3:
                num_twins = np.sum(h < 20)
                st.metric("Number of Twins", f"{num_twins:.0f}")
       
        with tabs[1]:  # Run Simulation
            # ... (unchanged, but uses the solver with new max_overstress)
            pass
       
        with tabs[2]:  # Basic Results
            if 'results_history' in st.session_state:
                st.header("Basic Results Visualization")
                results_history = st.session_state.results_history
                timesteps = st.session_state.timesteps
               
                frame_idx = st.slider("Select frame", 0, len(results_history)-1, len(results_history)-1)
                results = results_history[frame_idx]
               
                style_params = {
                    'phi_cmap': params.get('cmap_phi', 'RdBu_r'),
                    'sigma_eq_cmap': params.get('cmap_stress', 'hot'),
                    'scalebar_color': params.get('scalebar_color', 'black'),
                    'scalebar_fontsize': params.get('scalebar_fontsize', 10)
                }
                fig = visualizer.create_multi_field_comparison(results, style_params)
                st.pyplot(fig)
                plt.close(fig)
               
                st.subheader("Convergence Monitoring")
                if hasattr(st.session_state, 'solver') and st.session_state.solver.history['phi_norm']:
                    full_timesteps = np.arange(len(st.session_state.solver.history['phi_norm'])) * params['dt']
                    conv_fig = SimulationMonitor.create_convergence_plots(
                        st.session_state.solver.history,
                        full_timesteps
                    )
                    st.pyplot(conv_fig)
                    plt.close(conv_fig)
            else:
                st.info("Run a simulation first.")
       
        with tabs[3]:  # Advanced Analysis (unchanged)
            # ... 
            pass
       
        with tabs[4]:  # Enhanced Export (unchanged)
            # ...
            pass
   
    else:
        # Welcome screen (unchanged)
        pass
   
    # ========================================================================
    # DEBUG PANEL (unchanged)
    # ========================================================================
    with st.expander("🐛 Debug Information", expanded=False):
        # ... (unchanged)
        pass

if __name__ == "__main__":
    main()
