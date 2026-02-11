import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from numba import njit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import zipfile
import pickle
import torch
from io import BytesIO, StringIO
import tempfile
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# POST-PROCESSING CLASSES FROM AG NP CODE (ADAPTED)
# ============================================================================
class EnhancedLineProfiler:
    """Enhanced line profile system with multiple orientations and proper scaling"""
   
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
       
    def extract_profile(self, data, profile_type, position_ratio=0.5, angle_deg=45):
        """
        Extract line profiles from 2D data with proper scaling
        """
        # NORMALIZATION FIX: Convert any UI format to internal format
        profile_type = self._normalize_profile_type(profile_type)
       
        ny, nx = data.shape
        center_x, center_y = nx // 2, ny // 2
       
        # Calculate position offset based on ratio
        if profile_type in ['horizontal', 'vertical']:
            offset = int(min(nx, ny) * 0.4 * position_ratio)
        else:
            offset = int(min(nx, ny) * 0.3 * position_ratio)
       
        if profile_type == 'horizontal':
            # Horizontal profile
            row_idx = center_y + offset
            profile = data[row_idx, :]
            distance = np.linspace(self.extent[0], self.extent[1], nx)
            endpoints = (self.extent[0], row_idx * self.dx + self.extent[2],
                        self.extent[1], row_idx * self.dx + self.extent[2])
           
        elif profile_type == 'vertical':
            # Vertical profile
            col_idx = center_x + offset
            profile = data[:, col_idx]
            distance = np.linspace(self.extent[2], self.extent[3], ny)
            endpoints = (col_idx * self.dx + self.extent[0], self.extent[2],
                        col_idx * self.dx + self.extent[0], self.extent[3])
           
        elif profile_type == 'diagonal':
            # Main diagonal (top-left to bottom-right)
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
           
            # Calculate endpoints in physical coordinates
            x_start = start_idx[0] * self.dx + self.extent[0]
            y_start = start_idx[1] * self.dx + self.extent[2]
            x_end = (start_idx[0] + diag_length - 1) * self.dx + self.extent[0]
            y_end = (start_idx[1] + diag_length - 1) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
           
        elif profile_type == 'anti_diagonal':
            # Anti-diagonal (top-right to bottom-left)
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
            # Custom angle line profile
            angle_rad = np.deg2rad(angle_deg)
            length = int(min(nx, ny) * 0.8)
           
            # Calculate line endpoints
            dx_line = np.cos(angle_rad) * length//2
            dy_line = np.sin(angle_rad) * length//2
           
            profile = []
            distances = []
           
            # Interpolate along line
            for t in np.linspace(-length//2, length//2, length):
                x = center_x + t * np.cos(angle_rad) + offset * np.cos(angle_rad + np.pi/2)
                y = center_y + t * np.sin(angle_rad) + offset * np.sin(angle_rad + np.pi/2)
               
                if 0 <= x < nx-1 and 0 <= y < ny-1:
                    # Bilinear interpolation
                    x0, y0 = int(x), int(y)
                    x1, y1 = x0 + 1, y0 + 1
                   
                    # Check bounds
                    if x1 >= nx: x1 = nx - 1
                    if y1 >= ny: y1 = ny - 1
                   
                    # Interpolation weights
                    wx = x - x0
                    wy = y - y0
                   
                    # Bilinear interpolation
                    val = (data[y0, x0] * (1-wx) * (1-wy) +
                          data[y0, x1] * wx * (1-wy) +
                          data[y1, x0] * (1-wx) * wy +
                          data[y1, x1] * wx * wy)
                   
                    profile.append(val)
                    distances.append(t * self.dx)
           
            distance = np.array(distances)
            profile = np.array(profile)
           
            # Calculate endpoints
            x_start = (center_x - dx_line + offset * np.cos(angle_rad + np.pi/2)) * self.dx + self.extent[0]
            y_start = (center_y - dy_line + offset * np.sin(angle_rad + np.pi/2)) * self.dx + self.extent[2]
            x_end = (center_x + dx_line + offset * np.cos(angle_rad + np.pi/2)) * self.dx + self.extent[0]
            y_end = (center_y + dy_line + offset * np.sin(angle_rad + np.pi/2)) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
       
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
       
        return distance, profile, endpoints
   
    def _normalize_profile_type(self, profile_type):
        """Normalize profile type from UI format to internal format"""
        # Convert to lowercase and replace hyphens with underscores
        normalized = str(profile_type).lower().replace('-', '_')
       
        # Mapping for common UI variations
        mapping = {
            'horizontal': 'horizontal',
            'h': 'horizontal',
            'x': 'horizontal',
            'vertical': 'vertical',
            'v': 'vertical',
            'y': 'vertical',
            'diagonal': 'diagonal',
            'd': 'diagonal',
            'diag': 'diagonal',
            'anti_diagonal': 'anti_diagonal',
            'antidiagonal': 'anti_diagonal',
            'anti-diagonal': 'anti_diagonal',
            'ad': 'anti_diagonal',
            'custom': 'custom',
            'c': 'custom',
            'angled': 'custom'
        }
       
        return mapping.get(normalized, normalized)

class PublicationEnhancer:
    """Advanced plotting enhancements for publication-quality figures"""
   
    @staticmethod
    def create_custom_colormaps():
        """Create enhanced scientific colormaps"""
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
       
        # Perceptually uniform sequential
        plasma_enhanced = LinearSegmentedColormap.from_list('plasma_enhanced', [
            (0.0, '#0c0887'),
            (0.1, '#4b03a1'),
            (0.3, '#8b0aa5'),
            (0.5, '#b83289'),
            (0.7, '#db5c68'),
            (0.9, '#f48849'),
            (1.0, '#fec325')
        ])
       
        # Diverging with better contrast
        coolwarm_enhanced = LinearSegmentedColormap.from_list('coolwarm_enhanced', [
            (0.0, '#3a4cc0'),
            (0.25, '#8abcdd'),
            (0.5, '#f7f7f7'),
            (0.75, '#f0b7a4'),
            (1.0, '#b40426')
        ])
       
        # Categorical for twin types
        twin_categorical = ListedColormap([
            '#1f77b4', # CTB - Blue
            '#ff7f0e', # ITB - Orange
            '#2ca02c', # Grain 1 - Green
            '#d62728', # Grain 2 - Red
            '#9467bd', # Plastic zone - Purple
            '#8c564b' # Defect - Brown
        ])
       
        # Stress-specific colormap
        stress_map = LinearSegmentedColormap.from_list('stress_map', [
            (0.0, '#2c7bb6'),
            (0.2, '#abd9e9'),
            (0.4, '#ffffbf'),
            (0.6, '#fdae61'),
            (0.8, '#d7191c'),
            (1.0, '#800026')
        ])
       
        return {
            'plasma_enhanced': plasma_enhanced,
            'coolwarm_enhanced': coolwarm_enhanced,
            'twin_categorical': twin_categorical,
            'stress_map': stress_map
        }
   
    @staticmethod
    def add_error_shading(ax, x, y_mean, y_std, color='blue', alpha=0.3, label=''):
        """Add error shading to line plots"""
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                       color=color, alpha=alpha, label=label + ' ± std')
        return ax
   
    @staticmethod
    def add_scale_bar(ax, length_nm, location='lower right', color='black', linewidth=2):
        """Add scale bar to microscopy-style images"""
        if location == 'lower right':
            x_pos = 0.95
            y_pos = 0.05
            ha = 'right'
            va = 'bottom'
        elif location == 'lower left':
            x_pos = 0.05
            y_pos = 0.05
            ha = 'left'
            va = 'bottom'
        elif location == 'upper right':
            x_pos = 0.95
            y_pos = 0.95
            ha = 'right'
            va = 'top'
        else:
            x_pos = 0.05
            y_pos = 0.95
            ha = 'left'
            va = 'top'
       
        # Convert to axis coordinates
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
       
        # Bar position in data coordinates
        bar_x_start = xlim[1] - x_range * 0.15
        bar_x_end = bar_x_start - length_nm
        bar_y = ylim[0] + y_range * 0.05
       
        # Draw scale bar
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
               color=color, linewidth=linewidth, solid_capstyle='butt')
       
        # Add text
        ax.text((bar_x_start + bar_x_end) / 2, bar_y + y_range * 0.02,
               f'{length_nm} nm', ha='center', va='bottom',
               color=color, fontsize=8, fontweight='bold')
       
        return ax

class SimulationDatabase:
    """Enhanced simulation database for storing and comparing multiple runs"""
   
    @staticmethod
    def generate_id(sim_params):
        """Generate unique ID for simulation"""
        param_str = json.dumps({k: v for k, v in sim_params.items()
                              if k not in ['history', 'results', 'geom_viz']},
                             sort_keys=True)
        import hashlib
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
   
    @staticmethod
    def save_simulation(sim_params, results_history, geometry_data, metadata):
        """Save simulation to database"""
        if 'twin_simulations' not in st.session_state:
            st.session_state.twin_simulations = {}
       
        sim_id = SimulationDatabase.generate_id(sim_params)
       
        # Store simulation data
        st.session_state.twin_simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'results_history': results_history,
            'geometry_data': geometry_data,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
       
        return sim_id
   
    @staticmethod
    def get_simulation_list():
        """Get list of simulations for dropdown"""
        if 'twin_simulations' not in st.session_state:
            return []
       
        simulations = []
        for sim_id, sim_data in st.session_state.twin_simulations.items():
            params = sim_data['params']
            name = f"Twin λ={params.get('twin_spacing', 0):.1f}nm | σ={params.get('applied_stress', 0)/1e6:.0f}MPa"
            simulations.append({
                'id': sim_id,
                'name': name,
                'params': params,
                'results': sim_data['results_history'][-1] if sim_data['results_history'] else None
            })
       
        return simulations

# ============================================================================
# FIXED NUMBA-COMPATIBLE FUNCTIONS
# ============================================================================
@njit(parallel=True)
def compute_gradients_numba(field, dx):
    """Numba-compatible gradient computation"""
    N = field.shape[0]
    gx = np.zeros((N, N))
    gy = np.zeros((N, N))
    for i in prange(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        for j in range(N):
            jp1 = (j + 1) % N
            jm1 = (j - 1) % N
            gx[i, j] = (field[ip1, j] - field[im1, j]) / (2 * dx)
            gy[i, j] = (field[i, jp1] - field[i, jm1]) / (2 * dx)
    return gx, gy

@njit(parallel=True)
def compute_laplacian_numba(field, dx):
    """Numba-compatible Laplacian computation"""
    N = field.shape[0]
    lap = np.zeros((N, N))
    for i in prange(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        for j in range(N):
            jp1 = (j + 1) % N
            jm1 = (j - 1) % N
            lap[i, j] = (field[ip1, j] + field[im1, j] +
                         field[i, jp1] + field[i, jm1] -
                         4 * field[i, j]) / (dx**2)
    return lap

@njit(parallel=True)
def compute_twin_spacing_numba(phi_gx, phi_gy):
    """Numba-compatible twin spacing computation"""
    N = phi_gx.shape[0]
    h = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            grad_mag = np.sqrt(phi_gx[i, j]**2 + phi_gy[i, j]**2)
            if grad_mag > 1e-12:
                h[i, j] = 2.0 / grad_mag
            else:
                h[i, j] = 1e6
    return h

@njit(parallel=True)
def compute_anisotropic_properties_numba(phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso, L_CTB, L_ITB, n_mob):
    """Numba-compatible anisotropic properties computation"""
    N = phi_gx.shape[0]
    kappa_phi = np.zeros((N, N))
    L_phi = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            grad_mag = np.sqrt(phi_gx[i, j]**2 + phi_gy[i, j]**2 + 1e-12)
            if grad_mag > 1e-6:
                mx = phi_gx[i, j] / grad_mag
                my = phi_gy[i, j] / grad_mag
                dot = mx * nx + my * ny
                kappa_phi[i, j] = kappa0 * (1.0 + gamma_aniso * (1.0 - dot**2))
                aniso_factor = (1.0 - dot**2)**n_mob
                L_phi[i, j] = L_CTB + (L_ITB - L_CTB) * aniso_factor
            else:
                kappa_phi[i, j] = kappa0
                L_phi[i, j] = L_CTB
    return kappa_phi, L_phi

@njit(parallel=True)
def compute_transformation_strain_numba(phi, eta1, gamma_tw, ax, ay, nx, ny):
    """Numba-compatible transformation strain computation"""
    N = phi.shape[0]
    exx_star = np.zeros((N, N))
    eyy_star = np.zeros((N, N))
    exy_star = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            if eta1[i, j] > 0.5:
                phi_val = phi[i, j]
                f_phi = 0.25 * (phi_val**3 - phi_val**2 - phi_val + 1)
                exx_star[i, j] = gamma_tw * nx * ax * f_phi
                eyy_star[i, j] = gamma_tw * ny * ay * f_phi
                exy_star[i, j] = 0.5 * gamma_tw * (nx * ay + ny * ax) * f_phi
    return exx_star, eyy_star, exy_star

@njit(parallel=True)
def compute_yield_stress_numba(h, sigma0, mu, b, nu):
    """Numba-compatible yield stress computation"""
    N = h.shape[0]
    sigma_y = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            h_val = h[i, j]
            if h_val > 2 * b:
                log_term = np.log(h_val / b)
                sigma_y[i, j] = sigma0 + (mu * b / (2 * np.pi * h_val * (1 - nu))) * log_term
            else:
                sigma_y[i, j] = sigma0 + mu / (2 * np.pi * (1 - nu))
    return sigma_y

@njit(parallel=True)
def compute_plastic_strain_numba(sigma_eq, sigma_y, eps_p_xx, eps_p_yy, eps_p_xy,
                                 gamma0_dot, m, dt, N):
    """Numba-compatible plastic strain computation - FIXED INDEXING ERROR"""
    eps_p_xx_new = np.zeros((N, N))
    eps_p_yy_new = np.zeros((N, N))
    eps_p_xy_new = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            if sigma_eq[i, j] > sigma_y[i, j]:
                # Overstress ratio
                overstress = (sigma_eq[i, j] - sigma_y[i, j]) / sigma_y[i, j]
                # Strain rate magnitude using Perzyna model (avoid negative values)
                gamma_dot = gamma0_dot * max(overstress, 0.0)**m
                # Plastic strain increment (associated flow rule)
                stress_dev = 2/3 * gamma_dot * dt
                # Update plastic strains (volume preserving)
                eps_p_xx_new[i, j] = eps_p_xx[i, j] + stress_dev
                eps_p_yy_new[i, j] = eps_p_yy[i, j] - 0.5 * stress_dev
                eps_p_xy_new[i, j] = eps_p_xy[i, j] + 0.5 * stress_dev
            else:
                # No plastic strain if not yielding
                eps_p_xx_new[i, j] = eps_p_xx[i, j]
                eps_p_yy_new[i, j] = eps_p_yy[i, j]
                eps_p_xy_new[i, j] = eps_p_xy[i, j]
    return eps_p_xx_new, eps_p_yy_new, eps_p_xy_new

# ============================================================================
# ENHANCED PHYSICS MODELS WITH ERROR HANDLING (FROM CODE 1)
# ============================================================================
class MaterialProperties:
    """Enhanced material properties database with validation"""
    @staticmethod
    def get_cu_properties():
        """Comprehensive Cu properties with references"""
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
    def validate_parameters(params):
        """Validate simulation parameters"""
        errors = []
        warnings = []
        # Check parameter ranges
        if params['dt'] <= 0:
            errors.append("Time step dt must be positive")
        if params['dx'] <= 0:
            errors.append("Grid spacing dx must be positive")
        if params['N'] < 32:
            warnings.append("Grid resolution N < 32 may produce inaccurate results")
        if params['twin_spacing'] < 5:
            warnings.append("Twin spacing < 5nm may be physically unrealistic")
        if params['applied_stress'] > 2e9:
            warnings.append("Applied stress > 2GPa may cause unrealistic deformation")
        return errors, warnings

class InitialGeometryVisualizer:
    """Class to create and visualize initial geometric conditions"""
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.x = np.linspace(-N*dx/2, N*dx/2, N)
        self.y = np.linspace(-N*dx/2, N*dx/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def create_twin_grain_geometry(self, twin_spacing=20.0, grain_boundary_pos=0.0, gb_width=3.0):
        """Create initial twin grain geometry with grain boundary"""
        eta1 = np.zeros((self.N, self.N))
        eta2 = np.zeros((self.N, self.N))
        phi = np.zeros((self.N, self.N))
        # Create grain boundary
        for i in range(self.N):
            for j in range(self.N):
                x_val = self.X[i, j]
                dist_from_gb = x_val - grain_boundary_pos
                if dist_from_gb < -gb_width:
                    eta1[i, j] = 1.0
                    eta2[i, j] = 0.0
                elif dist_from_gb > gb_width:
                    eta1[i, j] = 0.0
                    eta2[i, j] = 1.0
                else:
                    transition = 0.5 * (1 - np.tanh(dist_from_gb / (gb_width/3)))
                    eta1[i, j] = transition
                    eta2[i, j] = 1 - transition
        # Create periodic twin structure
        for i in range(self.N):
            for j in range(self.N):
                if eta1[i, j] > 0.5:
                    phase = 2 * np.pi * self.Y[i, j] / twin_spacing
                    phi[i, j] = np.tanh(np.sin(phase) * 3.0)
        return phi, eta1, eta2

    def create_defect_geometry(self, twin_spacing=20.0, defect_type='dislocation', defect_pos=(0, 0), defect_radius=10.0):
        """Create geometry with defects (dislocations, voids, etc.)"""
        phi, eta1, eta2 = self.create_twin_grain_geometry(twin_spacing, defect_pos[0])
        if defect_type == 'dislocation':
            # Add dislocation by perturbing twin pattern
            center_x, center_y = defect_pos
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i, j] - center_x)**2 + (self.Y[i, j] - center_y)**2)
                    if dist < defect_radius and eta1[i, j] > 0.5:
                        # Create dislocation core by shifting phase
                        phase_shift = np.exp(-dist**2 / (defect_radius**2)) * np.pi
                        phase = 2 * np.pi * self.Y[i, j] / twin_spacing + phase_shift
                        phi[i, j] = np.tanh(np.sin(phase) * 3.0)
        elif defect_type == 'void':
            # Add void by removing material
            center_x, center_y = defect_pos
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i, j] - center_x)**2 + (self.Y[i, j] - center_y)**2)
                    if dist < defect_radius:
                        eta1[i, j] = 0.0
                        eta2[i, j] = 0.0
                        phi[i, j] = 0.0
        return phi, eta1, eta2

class EnhancedSpectralSolver:
    """Enhanced spectral solver with error handling and stability improvements"""
    def __init__(self, N, dx, elastic_params):
        self.N = N
        self.dx = dx
        # Fourier space grid
        self.kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1)
        self.ky = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1)
        self.k2 = self.kx**2 + self.ky**2
        # Avoid division by zero at k=0
        self.k2[0, 0] = 1e-12
       
        # Extract elastic constants
        C11 = elastic_params['C11']
        C12 = elastic_params['C12']
        C44 = elastic_params['C44']
        # 2D plane strain approximation for (111) plane
        C11_2d = (C11 + C12 + 2*C44) / 2
        C12_2d = (C11 + C12 - 2*C44) / 2
        lambda_2d = C12_2d
        mu_2d = (C11_2d - C12_2d) / 2
        # Store stiffness for stress calculation
        self.C11_2d = C11_2d
        self.C12_2d = C12_2d
        self.C44_2d = C44
       
        # Green's function components with stability check
        denom = mu_2d * (lambda_2d + 2*mu_2d) * self.k2 + 1e-15
        self.G11 = (mu_2d*(self.kx**2 + 2*self.ky**2) + lambda_2d*self.ky**2) / denom
        self.G12 = -mu_2d * self.kx * self.ky / denom
        self.G22 = (mu_2d*(self.ky**2 + 2*self.kx**2) + lambda_2d*self.kx**2) / denom

    def solve(self, eigenstrain_xx, eigenstrain_yy, eigenstrain_xy, applied_stress=0):
        """Solve mechanical equilibrium with error handling"""
        try:
            # Check input shapes
            assert eigenstrain_xx.shape == (self.N, self.N), f"Invalid eigenstrain shape: {eigenstrain_xx.shape}"
            # Fourier transforms
            eps_xx_hat = fft2(eigenstrain_xx)
            eps_yy_hat = fft2(eigenstrain_yy)
            eps_xy_hat = fft2(eigenstrain_xy)
            # Solve for displacements in Fourier space
            ux_hat = 1j * (self.G11 * self.kx * eps_xx_hat +
                          self.G12 * self.ky * eps_xx_hat +
                          self.G12 * self.kx * eps_yy_hat +
                          self.G22 * self.ky * eps_yy_hat)
            uy_hat = 1j * (self.G12 * self.kx * eps_xx_hat +
                          self.G22 * self.ky * eps_xx_hat +
                          self.G11 * self.kx * eps_yy_hat +
                          self.G12 * self.ky * eps_yy_hat)
            # Elastic strains
            eps_xx_el = np.real(ifft2(1j * self.kx * ux_hat))
            eps_yy_el = np.real(ifft2(1j * self.ky * uy_hat))
            eps_xy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_hat + self.ky * ux_hat)))
            # Total strains
            eps_xx = eps_xx_el + eigenstrain_xx
            eps_yy = eps_yy_el + eigenstrain_yy
            eps_xy = eps_xy_el + eigenstrain_xy
            # Stresses (plane strain approximation)
            sxx = applied_stress + self.C11_2d * eps_xx + self.C12_2d * eps_yy
            syy = self.C12_2d * eps_xx + self.C11_2d * eps_yy
            sxy = 2 * self.C44_2d * eps_xy
            # von Mises equivalent stress (corrected formula for plane strain)
            sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - 0)**2 + (0 - sxx)**2 + 6 * sxy**2))
            # Clip unrealistic values but preserve physical range
            sigma_eq = np.clip(sigma_eq, 0, 5e9)
            return sigma_eq, sxx, syy, sxy, eps_xx, eps_yy, eps_xy
        except Exception as e:
            st.error(f"Error in spectral solver: {str(e)}")
            st.exception(e) # Show full traceback for debugging
            # Return zeros in case of error
            zeros = np.zeros((self.N, self.N))
            return zeros, zeros, zeros, zeros, zeros, zeros, zeros

# ============================================================================
# ENHANCED VISUALIZATION SYSTEM
# ============================================================================
class EnhancedTwinVisualizer:
    """Comprehensive visualization system for nanotwinned simulations"""
   
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
        self.line_profiler = EnhancedLineProfiler(N, dx)
       
        # Expanded colormap library (50+ options)
        self.COLORMAPS = {
            # Sequential
            'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno', 'magma': 'magma',
            'cividis': 'cividis', 'hot': 'hot', 'cool': 'cool', 'spring': 'spring',
            'summer': 'summer', 'autumn': 'autumn', 'winter': 'winter',
            # Diverging
            'coolwarm': 'coolwarm', 'bwr': 'bwr', 'seismic': 'seismic', 'RdBu': 'RdBu',
            'RdGy': 'RdGy', 'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG', 'PuOr': 'PuOr',
            # Scientific
            'rocket': 'rocket', 'mako': 'mako', 'crest': 'crest', 'icefire': 'icefire',
            'twilight': 'twilight', 'hsv': 'hsv',
            # Custom
            **PublicationEnhancer.create_custom_colormaps()
        }

    def create_multi_field_comparison(self, results_dict, style_params=None):
        """Create publication-quality multi-field comparison plot"""
        if style_params is None:
            style_params = {}
       
        fields_to_plot = [
            ('phi', 'Twin Order Parameter φ', 'RdBu_r', [-1.2, 1.2]),
            ('sigma_eq', 'Von Mises Stress (GPa)', 'hot', None),
            ('h', 'Twin Spacing (nm)', 'plasma', [0, 30]),
            ('eps_p_mag', 'Plastic Strain', 'YlOrRd', None),
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
           
            # Apply colormap from style params or use default
            cmap_name = style_params.get(f'{field_name}_cmap', default_cmap)
            cmap = self.get_colormap(cmap_name)
           
            # Determine vmin/vmax
            if vrange is not None:
                vmin, vmax = vrange
            else:
                # Use percentiles for adaptive scaling
                vmin = np.percentile(data, 2)
                vmax = np.percentile(data, 98)
           
            # Plot heatmap
            im = ax.imshow(data, extent=self.extent, cmap=cmap,
                          vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
           
            # Add contour for twin boundaries (φ = 0)
            if field_name == 'phi':
                ax.contour(np.linspace(self.extent[0], self.extent[1], self.N),
                          np.linspace(self.extent[2], self.extent[3], self.N),
                          data, levels=[0], colors='white', linewidths=1, alpha=0.8)
           
            ax.set_title(title, fontsize=style_params.get('title_font_size', 10))
            ax.set_xlabel('x (nm)', fontsize=style_params.get('label_font_size', 8))
            ax.set_ylabel('y (nm)', fontsize=style_params.get('label_font_size', 8))
           
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
           
            # Add scale bar for reference plots
            if field_name in ['phi', 'sigma_eq']:
                PublicationEnhancer.add_scale_bar(ax, 10.0, 'lower right')
       
        # Hide empty subplots
        for idx in range(n_fields, rows*cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
       
        plt.tight_layout()
        return fig

    def get_colormap(self, cmap_name):
        """Get colormap by name with fallback"""
        if cmap_name in self.COLORMAPS:
            if isinstance(self.COLORMAPS[cmap_name], str):
                return plt.cm.get_cmap(self.COLORMAPS[cmap_name])
            else:
                return self.COLORMAPS[cmap_name] # Custom colormap object
        else:
            return plt.cm.get_cmap('viridis') # Default fallback

# ============================================================================
# MAIN SOLVER CLASS WITH ALL EVOLUTION LOGIC (FROM CODE 1)
# ============================================================================
class NanotwinnedCuSolver:
    """Main solver with comprehensive error handling - UPDATED WITH EVOLUTION LOGIC"""
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']
        # Material properties
        self.mat_props = MaterialProperties.get_cu_properties()
        # Validate parameters
        errors, warnings = MaterialProperties.validate_parameters(params)
        if errors:
            raise ValueError(f"Parameter validation failed: {', '.join(errors)}")
        if warnings:
            st.warning(f"Parameter warnings: {', '.join(warnings)}")
        # Initialize geometry visualizer
        self.geom_viz = InitialGeometryVisualizer(self.N, self.dx)
        # Initialize fields with error handling
        try:
            self.phi, self.eta1, self.eta2 = self.initialize_fields()
        except Exception as e:
            st.error(f"Failed to initialize fields: {e}")
            # Initialize with zeros as fallback
            self.phi = np.zeros((self.N, self.N))
            self.eta1 = np.zeros((self.N, self.N))
            self.eta2 = np.zeros((self.N, self.N))
        # Initialize plastic strain
        self.eps_p_xx = np.zeros((self.N, self.N))
        self.eps_p_yy = np.zeros((self.N, self.N))
        self.eps_p_xy = np.zeros((self.N, self.N))
        # Initialize spectral solver
        self.spectral_solver = EnhancedSpectralSolver(
            self.N, self.dx, self.mat_props['elastic']
        )
        # Initialize history for convergence monitoring (saved EVERY step)
        self.history = {
            'phi_norm': [],
            'energy': [],
            'max_stress': [],
            'plastic_work': [],
            'avg_stress': [], # Added for better monitoring
            'twin_spacing_avg': [] # Added for better monitoring
        }

    def initialize_fields(self):
        """Initialize order parameters based on selected geometry"""
        geom_type = self.params.get('geometry_type', 'standard')
        twin_spacing = self.params['twin_spacing']
        gb_pos = self.params['grain_boundary_pos']
        if geom_type == 'defect':
            defect_type = self.params.get('defect_type', 'dislocation')
            defect_pos = self.params.get('defect_pos', (0, 0))
            defect_radius = self.params.get('defect_radius', 10.0)
            return self.geom_viz.create_defect_geometry(
                twin_spacing, defect_type, defect_pos, defect_radius
            )
        else:
            return self.geom_viz.create_twin_grain_geometry(twin_spacing, gb_pos)

    # ============================================================================
    # CRITICAL EVOLUTION METHODS (MISSING FROM ORIGINAL CODE 2)
    # ============================================================================
    def compute_local_energy_derivatives(self):
        """Compute derivatives of local free energy density"""
        W = self.params['W']
        A = self.params['A']
        B = self.params['B']
        # f_loc = W(φ²-1)²η₁² + ΣAη_i²(1-η_i)² + Bη₁²η₂²
        # Derivative with respect to φ
        df_dphi = 4 * W * self.phi * (self.phi**2 - 1) * self.eta1**2
        # Derivative with respect to η₁
        df_deta1 = (2 * A * self.eta1 * (1 - self.eta1) * (1 - 2*self.eta1) +
                   2 * B * self.eta1 * self.eta2**2 +
                   2 * W * (self.phi**2 - 1)**2 * self.eta1)
        # Derivative with respect to η₂
        df_deta2 = (2 * A * self.eta2 * (1 - self.eta2) * (1 - 2*self.eta2) +
                   2 * B * self.eta2 * self.eta1**2)
        return df_dphi, df_deta1, df_deta2

    def compute_elastic_driving_force(self, sxx, syy, sxy):
        """Compute elastic driving force for twin evolution"""
        try:
            # Get twin parameters
            gamma_tw = self.mat_props['twinning']['gamma_tw']
            n = self.mat_props['twinning']['n_2d']
            a = self.mat_props['twinning']['a_2d']
            # Derivative of interpolation function
            dh_dphi = 0.25 * (3*self.phi**2 - 2*self.phi - 1)
            # Transformation strain derivative
            nx, ny = n[0], n[1]
            ax, ay = a[0], a[1]
            # Only compute where twin grain is active
            active_mask = (self.eta1 > 0.5)
            deps_xx_dphi = np.zeros_like(self.phi)
            deps_yy_dphi = np.zeros_like(self.phi)
            deps_xy_dphi = np.zeros_like(self.phi)
            deps_xx_dphi[active_mask] = gamma_tw * nx * ax * dh_dphi[active_mask]
            deps_yy_dphi[active_mask] = gamma_tw * ny * ay * dh_dphi[active_mask]
            deps_xy_dphi[active_mask] = 0.5 * gamma_tw * (nx * ay + ny * ax) * dh_dphi[active_mask]
            # Elastic driving force: -σ:∂ε*/∂φ
            df_el_dphi = -(sxx * deps_xx_dphi +
                          syy * deps_yy_dphi +
                          2 * sxy * deps_xy_dphi)
            return df_el_dphi
        except Exception as e:
            st.error(f"Error computing elastic driving force: {e}")
            st.exception(e)
            return np.zeros_like(self.phi)

    def evolve_twin_field(self, sxx, syy, sxy, eps_p_mag):
        """Evolve twin order parameter φ with stability checks"""
        try:
            # Extract parameters as floats
            kappa0 = float(self.params['kappa0'])
            gamma_aniso = float(self.params['gamma_aniso'])
            L_CTB = float(self.params.get('L_CTB', 0.05))  # Added default
            L_ITB = float(self.params.get('L_ITB', 5.0))   # Added default
            n_mob = int(self.params.get('n_mob', 4))       # Added default
            zeta = float(self.params.get('zeta', 0.3))     # Added default
            # Get twin parameters as floats
            n_twin = self.mat_props['twinning']['n_2d']
            nx = float(n_twin[0])
            ny = float(n_twin[1])
            # Compute gradients
            phi_gx, phi_gy = compute_gradients_numba(self.phi, self.dx)
            # Compute anisotropic properties
            kappa_phi, L_phi = compute_anisotropic_properties_numba(
                phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso, L_CTB, L_ITB, n_mob
            )
            # Compute driving forces
            df_loc_dphi, _, _ = self.compute_local_energy_derivatives()
            df_el_dphi = self.compute_elastic_driving_force(sxx, syy, sxy)
            # Compute dissipation term
            diss_p = zeta * eps_p_mag * self.phi
            # Compute gradient term
            lap_phi = compute_laplacian_numba(self.phi, self.dx)
            # Evolution equation with stability factor
            stability_factor = self.params.get('stability_factor', 0.5)
            dphi_dt = -L_phi * (df_loc_dphi + df_el_dphi - kappa_phi * lap_phi + diss_p)
            # Apply stability condition (CFL-like condition)
            max_dphi_dt = np.max(np.abs(dphi_dt))
            if max_dphi_dt * self.dt > stability_factor:
                # Scale time step if needed
                scale_factor = stability_factor / (max_dphi_dt * self.dt)
                dphi_dt *= scale_factor
                st.warning(f"Time step scaled by {scale_factor:.3f} for stability")
            # Update twin field
            phi_new = self.phi + self.dt * dphi_dt
            # Clip to physical bounds
            phi_new = np.clip(phi_new, -1.1, 1.1)
            return phi_new
        except Exception as e:
            st.error(f"Error evolving twin field: {e}")
            st.exception(e)
            return self.phi # Return unchanged field on error

    def evolve_grain_fields(self):
        """Evolve grain order parameters with stability checks"""
        try:
            kappa_eta = float(self.params['kappa_eta'])
            L_eta = float(self.params.get('L_eta', 1.0))  # Added default
            # Compute Laplacians
            lap_eta1 = compute_laplacian_numba(self.eta1, self.dx)
            lap_eta2 = compute_laplacian_numba(self.eta2, self.dx)
            # Compute local derivatives
            _, df_deta1, df_deta2 = self.compute_local_energy_derivatives()
            # Evolution equations with stability factor
            stability_factor = self.params.get('stability_factor', 0.5)
            deta1_dt = -L_eta * (df_deta1 - kappa_eta * lap_eta1)
            deta2_dt = -L_eta * (df_deta2 - kappa_eta * lap_eta2)
            # Check stability
            max_change = max(np.max(np.abs(deta1_dt)), np.max(np.abs(deta2_dt)))
            if max_change * self.dt > stability_factor:
                scale = stability_factor / (max_change * self.dt)
                deta1_dt *= scale
                deta2_dt *= scale
                st.warning(f"Grain field time step scaled by {scale:.3f} for stability")
            # Update grain fields
            eta1_new = self.eta1 + self.dt * deta1_dt
            eta2_new = self.eta2 + self.dt * deta2_dt
            # Enforce constraints
            eta1_new = np.clip(eta1_new, 0, 1)
            eta2_new = np.clip(eta2_new, 0, 1)
            # Enforce η₁² + η₂² ≤ 1
            norm = np.sqrt(eta1_new**2 + eta2_new**2 + 1e-12)
            mask = norm > 1
            eta1_new[mask] = eta1_new[mask] / norm[mask]
            eta2_new[mask] = eta2_new[mask] / norm[mask]
            return eta1_new, eta2_new
        except Exception as e:
            st.error(f"Error evolving grain fields: {e}")
            st.exception(e)
            return self.eta1, self.eta2

    def compute_plastic_strain(self, sigma_eq, sigma_y):
        """Compute plastic strain evolution using Perzyna model - FIXED VERSION"""
        try:
            plastic_params = self.mat_props['plasticity']
            gamma0_dot = plastic_params['gamma0_dot']
            m = int(plastic_params['m'])
            # Use Numba function to avoid indexing errors
            eps_p_xx_new, eps_p_yy_new, eps_p_xy_new = compute_plastic_strain_numba(
                sigma_eq, sigma_y,
                self.eps_p_xx, self.eps_p_yy, self.eps_p_xy,
                gamma0_dot, m, self.dt, self.N
            )
            # Update plastic strains
            self.eps_p_xx = eps_p_xx_new
            self.eps_p_yy = eps_p_yy_new
            self.eps_p_xy = eps_p_xy_new
            # Compute plastic strain magnitude (von Mises equivalent)
            eps_p_mag = np.sqrt(
                2/3 * (self.eps_p_xx**2 + self.eps_p_yy**2 + 2*self.eps_p_xy**2 + 1e-15)
            )
            return eps_p_mag
        except Exception as e:
            st.error(f"Error computing plastic strain: {e}")
            st.exception(e)
            # Return zero plastic strain on error
            return np.zeros_like(sigma_eq)

    def compute_total_energy(self):
        """Compute total free energy of the system"""
        try:
            # Local energy
            W = self.params['W']
            A = self.params['A']
            B = self.params['B']
            f_loc = (W * (self.phi**2 - 1)**2 * self.eta1**2 +
                    A * (self.eta1**2 * (1 - self.eta1)**2 + self.eta2**2 * (1 - self.eta2)**2) +
                    B * self.eta1**2 * self.eta2**2)
            # Gradient energy
            phi_gx, phi_gy = compute_gradients_numba(self.phi, self.dx)
            grad_phi_sq = phi_gx**2 + phi_gy**2
            eta1_gx, eta1_gy = compute_gradients_numba(self.eta1, self.dx)
            eta2_gx, eta2_gy = compute_gradients_numba(self.eta2, self.dx)
            grad_eta1_sq = eta1_gx**2 + eta1_gy**2
            grad_eta2_sq = eta2_gx**2 + eta2_gy**2
            kappa0 = self.params['kappa0']
            kappa_eta = self.params['kappa_eta']
            f_grad = 0.5 * kappa0 * grad_phi_sq + 0.5 * kappa_eta * (grad_eta1_sq + grad_eta2_sq)
            # Total energy density
            energy_density = f_loc + f_grad
            # Integrate over domain
            total_energy = np.sum(energy_density) * (self.dx**2)
            return total_energy
        except Exception as e:
            st.warning(f"Error computing energy: {e}")
            st.exception(e)
            return 0.0

    def step(self, applied_stress):
        """Perform one time step of the simulation with comprehensive error handling"""
        try:
            # Get twin parameters
            gamma_tw = self.mat_props['twinning']['gamma_tw']
            n = self.mat_props['twinning']['n_2d']
            a = self.mat_props['twinning']['a_2d']
            # Compute transformation strain
            exx_star, eyy_star, exy_star = compute_transformation_strain_numba(
                self.phi, self.eta1, gamma_tw, a[0], a[1], n[0], n[1]
            )
            # Total eigenstrain (transformation + plastic)
            eigenstrain_xx = exx_star + self.eps_p_xx
            eigenstrain_yy = eyy_star + self.eps_p_yy
            eigenstrain_xy = exy_star + self.eps_p_xy
            # Solve mechanical equilibrium
            sigma_eq, sxx, syy, sxy, eps_xx, eps_yy, eps_xy = self.spectral_solver.solve(
                eigenstrain_xx, eigenstrain_yy, eigenstrain_xy, applied_stress
            )
            # Compute twin spacing
            phi_gx, phi_gy = compute_gradients_numba(self.phi, self.dx)
            h = compute_twin_spacing_numba(phi_gx, phi_gy)
            # Compute yield stress using Ovid'ko model
            plastic_params = self.mat_props['plasticity']
            sigma_y = compute_yield_stress_numba(
                h, plastic_params['sigma0'], plastic_params['mu'],
                plastic_params['b'], plastic_params['nu']
            )
            # ================================================================
            # CRITICAL: UPDATE PLASTIC STRAIN AND EVOLVE FIELDS
            # ================================================================
            # Update plastic strain
            eps_p_mag = self.compute_plastic_strain(sigma_eq, sigma_y)
            # Evolve phase fields
            self.phi = self.evolve_twin_field(sxx, syy, sxy, eps_p_mag)
            self.eta1, self.eta2 = self.evolve_grain_fields()
            # ================================================================
            
            # Update history for monitoring (SAVE EVERY STEP HERE)
            phi_norm = np.linalg.norm(self.phi)
            total_energy = self.compute_total_energy()
            max_stress = np.max(sigma_eq)
            avg_stress = np.mean(sigma_eq)
            avg_spacing = np.mean(h[(h > 5) & (h < 50)])
            plastic_work = np.sum(eps_p_mag) * (self.dx**2)
           
            self.history['phi_norm'].append(phi_norm)
            self.history['energy'].append(total_energy)
            self.history['max_stress'].append(max_stress)
            self.history['avg_stress'].append(avg_stress)
            self.history['plastic_work'].append(plastic_work)
            self.history['twin_spacing_avg'].append(avg_spacing)
           
            # Prepare results
            results = {
                'phi': self.phi.copy(),
                'eta1': self.eta1.copy(),
                'eta2': self.eta2.copy(),
                'sigma_eq': sigma_eq.copy(),
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
            st.exception(e)
            # Return zeros in case of error
            zeros = np.zeros((self.N, self.N))
            return {
                'phi': zeros,
                'eta1': zeros,
                'eta2': zeros,
                'sigma_eq': zeros,
                'sigma_xx': zeros,
                'sigma_yy': zeros,
                'sigma_xy': zeros,
                'h': zeros,
                'sigma_y': zeros,
                'eps_p_mag': zeros,
                'eps_xx': zeros,
                'eps_yy': zeros,
                'eps_xy': zeros,
                'convergence': {
                    'phi_norm': 0,
                    'energy': 0,
                    'max_stress': 0,
                    'avg_stress': 0,
                    'plastic_work': 0,
                    'avg_spacing': 0
                }
            }

# ============================================================================
# COMPREHENSIVE VISUALIZATION AND MONITORING
# ============================================================================
class SimulationMonitor:
    """Monitor simulation progress and convergence"""
    @staticmethod
    def create_convergence_plots(history, timesteps):
        """Create convergence monitoring plots with proper length matching"""
        # CRITICAL FIX: Use history length for timesteps to avoid dimension mismatch
        history_length = len(history['phi_norm'])
        # Create full timestep array matching history length
        if len(timesteps) >= history_length:
            # Use first N timesteps if available
            plot_timesteps = timesteps[:history_length]
        else:
            # Interpolate if needed (shouldn't happen with proper saving)
            plot_timesteps = np.linspace(0, timesteps[-1] if timesteps else 1.0, history_length)
       
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
       
        # Phi norm evolution
        axes[0, 0].plot(plot_timesteps, history['phi_norm'], 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Time (ns)')
        axes[0, 0].set_ylabel('||φ||')
        axes[0, 0].set_title('Twin Order Parameter Norm')
        axes[0, 0].grid(True, alpha=0.3)
       
        # Energy evolution
        axes[0, 1].plot(plot_timesteps, history['energy'], 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Time (ns)')
        axes[0, 1].set_ylabel('Total Energy (J)')
        axes[0, 1].set_title('System Energy Evolution')
        axes[0, 1].grid(True, alpha=0.3)
       
        # Maximum stress evolution
        axes[0, 2].plot(plot_timesteps, np.array(history['max_stress'])/1e9, 'g-', linewidth=2, alpha=0.8, label='Max')
        axes[0, 2].plot(plot_timesteps, np.array(history['avg_stress'])/1e9, 'g--', linewidth=1.5, alpha=0.6, label='Avg')
        axes[0, 2].set_xlabel('Time (ns)')
        axes[0, 2].set_ylabel('Stress (GPa)')
        axes[0, 2].set_title('Stress Evolution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
       
        # Plastic work evolution
        axes[1, 0].plot(plot_timesteps, history['plastic_work'], 'm-', linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Time (ns)')
        axes[1, 0].set_ylabel('Plastic Work (J)')
        axes[1, 0].set_title('Plastic Work Evolution')
        axes[1, 0].grid(True, alpha=0.3)
       
        # Average twin spacing evolution
        axes[1, 1].plot(plot_timesteps, history['twin_spacing_avg'], 'c-', linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Time (ns)')
        axes[1, 1].set_ylabel('Avg Spacing (nm)')
        axes[1, 1].set_title('Average Twin Spacing')
        axes[1, 1].grid(True, alpha=0.3)
       
        # Plastic strain magnitude
        if 'eps_p_mag' in history:
            axes[1, 2].plot(plot_timesteps, history['eps_p_mag'], 'orange', linewidth=2, alpha=0.8)
            axes[1, 2].set_xlabel('Time (ns)')
            axes[1, 2].set_ylabel('Max Plastic Strain')
            axes[1, 2].set_title('Plastic Strain Evolution')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No plastic strain data',
                          ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Plastic Strain Evolution')
       
        plt.tight_layout()
        return fig

# ============================================================================
# ENHANCED EXPORT FUNCTIONALITY
# ============================================================================
class DataExporter:
    """Handle data export in multiple formats"""
    @staticmethod
    def export_simulation_results(results_history, params, geometry, filename_prefix):
        """Export comprehensive simulation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save parameters as JSON
            params_json = json.dumps(params, indent=2, cls=NumpyEncoder)
            zip_file.writestr(f"{filename_prefix}_params_{timestamp}.json", params_json)
            # 2. Save initial and final states
            if results_history:
                initial = results_history[0]
                final = results_history[-1]
                # Save as NPZ
                npz_buffer = BytesIO()
                np.savez_compressed(npz_buffer,
                    phi_initial=initial['phi'],
                    eta1_initial=initial['eta1'],
                    eta2_initial=initial['eta2'],
                    phi_final=final['phi'],
                    eta1_final=final['eta1'],
                    eta2_final=final['eta2'],
                    sigma_eq_final=final['sigma_eq'],
                    h_final=final['h'],
                    sigma_y_final=final['sigma_y'])
                zip_file.writestr(f"{filename_prefix}_states_{timestamp}.npz", npz_buffer.getvalue())
            # 3. Save convergence history
            if hasattr(geometry, 'history') and geometry.history:
                history_data = {
                    'timesteps': np.arange(len(geometry.history['phi_norm'])) * params['dt'],
                    'phi_norm': geometry.history['phi_norm'],
                    'energy': geometry.history['energy'],
                    'max_stress': geometry.history['max_stress'],
                    'avg_stress': geometry.history['avg_stress'],
                    'plastic_work': geometry.history['plastic_work'],
                    'twin_spacing_avg': geometry.history['twin_spacing_avg']
                }
                npz_buffer = BytesIO()
                np.savez_compressed(npz_buffer, **history_data)
                zip_file.writestr(f"{filename_prefix}_history_{timestamp}.npz", npz_buffer.getvalue())
            # 4. Save as CSV for easy analysis
            if results_history:
                csv_data = []
                for i, results in enumerate(results_history):
                    csv_data.append({
                        'step': i,
                        'avg_phi': np.mean(results['phi']),
                        'avg_sigma_eq_gpa': np.mean(results['sigma_eq']) / 1e9,
                        'max_sigma_eq_gpa': np.max(results['sigma_eq']) / 1e9,
                        'avg_h_nm': np.mean(results['h']),
                        'avg_sigma_y_mpa': np.mean(results['sigma_y']) / 1e6,
                        'max_eps_p': np.max(results['eps_p_mag'])
                    })
                df = pd.DataFrame(csv_data)
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                zip_file.writestr(f"{filename_prefix}_summary_{timestamp}.csv", csv_buffer.getvalue())
            # 5. Save as PyTorch tensors
            if results_history:
                torch_data = {
                    'phi': torch.stack([torch.from_numpy(r['phi']) for r in results_history]),
                    'sigma_eq': torch.stack([torch.from_numpy(r['sigma_eq']) for r in results_history]),
                    'h': torch.stack([torch.from_numpy(r['h']) for r in results_history])
                }
                torch_buffer = BytesIO()
                torch.save(torch_data, torch_buffer)
                zip_file.writestr(f"{filename_prefix}_tensors_{timestamp}.pt", torch_buffer.getvalue())
        zip_buffer.seek(0)
        return zip_buffer

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ============================================================================
# ENHANCED STREAMLIT APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="Enhanced Nanotwinned Cu Phase-Field Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
   
    # Custom CSS for enhanced UI
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
    .tab-content {
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
   
    # Header
    st.markdown('<h1 class="main-header">🔬 Enhanced Nanotwinned Copper Phase-Field Simulator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #F0F9FF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 1rem;">
    <strong>Advanced Physics + Enhanced Post-Processing:</strong><br>
    • Phase-field modeling of FCC nanotwins with anisotropic elasticity<br>
    • Ovid'ko confined layer slip + Perzyna viscoplasticity<br>
    • <strong>NEW:</strong> 50+ colormaps, line profiling, statistical analysis, correlation plots<br>
    • <strong>NEW:</strong> Publication-quality styling, multi-simulation comparison<br>
    • <strong>NEW:</strong> Enhanced export with comprehensive data packages
    </div>
    """, unsafe_allow_html=True)
   
    # Sidebar with enhanced controls
    with st.sidebar:
        st.header("⚙️ Simulation Configuration")
       
        # Operation mode
        operation_mode = st.radio(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations"],
            index=0
        )
       
        if operation_mode == "Run New Simulation":
            # Geometry configuration
            st.subheader("🧩 Geometry Configuration")
            geometry_type = st.selectbox(
                "Geometry Type",
                ["Standard Twin Grain", "Twin Grain with Defect"],
                key="geom_type"
            )
            
            # Simulation parameters with validation
            st.subheader("📊 Grid Configuration")
            N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64, key="N",
                         help="Higher resolution = more accurate but slower")
            dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1, key="dx",
                          help="Smaller spacing = finer details")
            dt = st.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5, key="dt",
                          help="Smaller time step = more stable but slower")
            
            # Material parameters
            st.subheader("🔬 Material Parameters")
            twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0, key="twin_spacing",
                                    help="Initial distance between twin boundaries")
            grain_boundary_pos = st.slider("Grain boundary position (nm)", -50.0, 50.0, 0.0, 1.0, key="gb_pos",
                                          help="Location of grain boundary")
            
            # Defect parameters (if selected)
            if geometry_type == "Twin Grain with Defect":
                st.subheader("⚠️ Defect Parameters")
                defect_type = st.selectbox("Defect Type", ["Dislocation", "Void"], key="defect_type")
                defect_x = st.slider("Defect X position (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_x")
                defect_y = st.slider("Defect Y position (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_y")
                defect_radius = st.slider("Defect radius (nm)", 5.0, 30.0, 10.0, 1.0, key="defect_radius")
            
            # Thermodynamic parameters
            st.subheader("⚡ Thermodynamic Parameters")
            W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1, key="W",
                         help="Controls twin boundary energy")
            A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5, key="A",
                         help="Controls grain boundary energy")
            B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5, key="B",
                         help="Prevents grain overlap")
            
            # Gradient energy parameters
            st.subheader("🌀 Gradient Energy")
            kappa0 = st.slider("κ₀ (Gradient energy ref)", 0.01, 10.0, 1.0, 0.1, key="kappa0",
                              help="Baseline gradient energy coefficient")
            gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05, key="gamma_aniso",
                                   help="Controls anisotropy between CTBs and ITBs")
            kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1, key="kappa_eta",
                                 help="Grain boundary gradient energy")
            
            # Kinetic parameters (CRITICAL ADDITION FROM CODE 1)
            st.subheader("⚡ Kinetic Parameters")
            L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001, key="L_CTB",
                             help="Mobility of coherent twin boundaries")
            L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1, key="L_ITB",
                             help="Mobility of incoherent twin boundaries")
            n_mob = st.slider("n (Mobility exponent)", 1, 10, 4, 1, key="n_mob",
                             help="Controls transition sharpness between CTB and ITB")
            L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1, key="L_eta",
                             help="Grain boundary mobility")
            zeta = st.slider("ζ (Dislocation pinning)", 0.0, 2.0, 0.3, 0.05, key="zeta",
                            help="Strength of dislocation pinning")
            
            # Loading conditions
            st.subheader("🏋️ Loading Conditions")
            applied_stress_MPa = st.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0, key="applied_stress",
                                          help="External applied stress")
            
            # Simulation control
            st.subheader("⏯️ Simulation Control")
            n_steps = st.slider("Number of steps", 10, 1000, 100, 10, key="n_steps",
                               help="Total simulation steps")
            save_frequency = st.slider("Save frequency", 1, 100, 10, 1, key="save_freq",
                                      help="How often to save results")
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                stability_factor = st.slider("Stability factor", 0.1, 1.0, 0.5, 0.1,
                                            help="Controls numerical stability")
                enable_monitoring = st.checkbox("Enable real-time monitoring", True,
                                               help="Track convergence during simulation")
                auto_adjust_dt = st.checkbox("Auto-adjust time step", True,
                                            help="Automatically adjust time step for stability")
            
            # Enhanced visualization settings
            st.subheader("🎨 Enhanced Visualization")
            visualizer = EnhancedTwinVisualizer(N, dx)
            colormap_library = visualizer.COLORMAPS
            selected_cmap_phi = st.selectbox("φ Colormap", list(colormap_library.keys()),
                                           index=list(colormap_library.keys()).index('RdBu_r')
                                           if 'RdBu_r' in colormap_library else 0)
            selected_cmap_stress = st.selectbox("σ_eq Colormap", list(colormap_library.keys()),
                                              index=list(colormap_library.keys()).index('hot')
                                              if 'hot' in colormap_library else 0)
            
            # Initialize button
            if st.button("🚀 Initialize Simulation", type="primary", use_container_width=True):
                # Validate parameters
                params = {
                    'N': N,
                    'dx': dx,
                    'dt': dt,
                    'W': W,
                    'A': A,
                    'B': B,
                    'kappa0': kappa0,
                    'gamma_aniso': gamma_aniso,
                    'kappa_eta': kappa_eta,
                    'L_CTB': L_CTB,
                    'L_ITB': L_ITB,
                    'n_mob': n_mob,
                    'L_eta': L_eta,
                    'zeta': zeta,
                    'twin_spacing': twin_spacing,
                    'grain_boundary_pos': grain_boundary_pos,
                    'geometry_type': 'defect' if geometry_type == "Twin Grain with Defect" else 'standard',
                    'applied_stress': applied_stress_MPa * 1e6,
                    'save_frequency': save_frequency,
                    'stability_factor': stability_factor,
                    'cmap_phi': selected_cmap_phi,
                    'cmap_stress': selected_cmap_stress
                }
                
                # Add defect parameters if needed
                if geometry_type == "Twin Grain with Defect":
                    params['defect_type'] = defect_type.lower()
                    params['defect_pos'] = (defect_x, defect_y)
                    params['defect_radius'] = defect_radius
                
                # Validate parameters
                errors, warnings = MaterialProperties.validate_parameters(params)
                if errors:
                    st.error(f"Validation errors: {', '.join(errors)}")
                    st.session_state.validation_failed = True
                else:
                    if warnings:
                        st.warning(f"Parameter warnings: {', '.join(warnings)}")
                    
                    # Initialize geometry visualizer
                    geom_viz = InitialGeometryVisualizer(N, dx)
                    
                    # Create initial geometry
                    if geometry_type == "Twin Grain with Defect":
                        phi, eta1, eta2 = geom_viz.create_defect_geometry(
                            twin_spacing, defect_type.lower(), (defect_x, defect_y), defect_radius
                        )
                    else:
                        phi, eta1, eta2 = geom_viz.create_twin_grain_geometry(twin_spacing, grain_boundary_pos)
                    
                    # Store in session state
                    st.session_state.initial_geometry = {
                        'phi': phi,
                        'eta1': eta1,
                        'eta2': eta2,
                        'geom_viz': geom_viz,
                        'params': params
                    }
                    st.session_state.initialized = True
                    st.session_state.operation_mode = 'new'
                    st.success("✅ Simulation initialized successfully!")
       
        else: # Compare Saved Simulations
            st.subheader("🔍 Comparison Configuration")
            simulations = SimulationDatabase.get_simulation_list()
           
            if not simulations:
                st.warning("No simulations saved yet. Run some simulations first!")
            else:
                # Multi-select for comparison
                sim_options = {sim['name']: sim['id'] for sim in simulations}
                selected_sim_ids = st.multiselect(
                    "Select Simulations to Compare",
                    options=list(sim_options.keys()),
                    default=list(sim_options.keys())[:min(3, len(sim_options))]
                )
               
                # Comparison settings
                comparison_type = st.selectbox(
                    "Comparison Type",
                    ["Side-by-Side Fields", "Overlay Line Profiles", "Statistical Summary",
                     "Correlation Analysis", "Evolution Timeline"],
                    index=0
                )
               
                # FIX: Profile direction normalization for comparison
                if comparison_type == "Overlay Line Profiles":
                    profile_direction = st.selectbox(
                        "Profile Direction",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal", "Custom"],
                        index=0
                    )
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                   
                    # Convert UI profile direction to internal format
                    profile_type_mapping = {
                        "Horizontal": "horizontal",
                        "Vertical": "vertical",
                        "Diagonal": "diagonal",
                        "Anti-Diagonal": "anti_diagonal",
                        "Custom": "custom"
                    }
                    internal_direction = profile_type_mapping.get(profile_direction, "horizontal")
               
                # Field selection for comparison
                field_to_compare = st.selectbox(
                    "Field to Compare",
                    ["phi (Twin Order)", "sigma_eq (Von Mises Stress)",
                     "h (Twin Spacing)", "sigma_y (Yield Stress)"],
                    index=1
                )
               
                if st.button("🔬 Run Comparison", type="primary"):
                    comparison_config = {
                        'sim_ids': [sim_options[name] for name in selected_sim_ids],
                        'type': comparison_type,
                        'field': field_to_compare.split()[0], # Extract field name
                    }
                   
                    # Add profile-specific config
                    if comparison_type == "Overlay Line Profiles":
                        comparison_config['profile_direction'] = internal_direction
                        comparison_config['position_ratio'] = position_ratio
                   
                    st.session_state.comparison_config = comparison_config
                    st.session_state.operation_mode = 'compare'
                    st.rerun()
   
    # Main content area with enhanced tabs
    if 'initialized' in st.session_state and st.session_state.initialized:
        params = st.session_state.initial_geometry['params']
        N = params['N']
        dx = params['dx']
       
        # Initialize enhanced visualizer
        visualizer = EnhancedTwinVisualizer(N, dx)
       
        # Create tabs
        tab_names = ["📐 Initial Geometry", "▶️ Run Simulation", "📊 Basic Results",
                    "🔍 Advanced Analysis", "📈 Comparison Tools", "📤 Enhanced Export"]
       
        tabs = st.tabs(tab_names)
       
        with tabs[0]: # Initial Geometry
            st.header("Initial Geometry Visualization")
            geom_viz = st.session_state.initial_geometry['geom_viz']
            phi = st.session_state.initial_geometry['phi']
            eta1 = st.session_state.initial_geometry['eta1']
            eta2 = st.session_state.initial_geometry['eta2']
           
            # Use enhanced visualizer for initial state
            initial_results = {
                'phi': phi,
                'eta1': eta1,
                'sigma_eq': np.zeros_like(phi), # Placeholder
                'h': compute_twin_spacing_numba(*compute_gradients_numba(phi, dx))
            }
           
            fig = visualizer.create_multi_field_comparison(initial_results)
            st.pyplot(fig)
           
            # Show geometry statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_spacing = np.mean(initial_results['h'][(initial_results['h']>5)&(initial_results['h']<50)])
                st.metric("Avg Twin Spacing", f"{avg_spacing:.1f} nm")
            with col2:
                twin_area = np.sum(eta1 > 0.5) * dx**2
                st.metric("Twin Grain Area", f"{twin_area:.0f} nm²")
            with col3:
                num_twins = np.sum(initial_results['h'] < 20)
                st.metric("Number of Twins", f"{num_twins:.0f}")
       
        with tabs[1]: # Run Simulation
            st.header("Run Simulation")
           
            if st.button("▶️ Start Evolution", type="secondary", use_container_width=True):
                with st.spinner("Running phase-field simulation..."):
                    try:
                        # Initialize solver WITH EVOLUTION CAPABILITIES
                        solver = NanotwinnedCuSolver(params)
                        solver.phi = st.session_state.initial_geometry['phi'].copy()
                        solver.eta1 = st.session_state.initial_geometry['eta1'].copy()
                        solver.eta2 = st.session_state.initial_geometry['eta2'].copy()
                       
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                       
                        # Storage for results (saved at save_frequency intervals)
                        results_history = []
                        timesteps = [] # Only for saved frames
                       
                        # Create placeholders for real-time monitoring
                        monitoring_cols = st.columns(4)
                       
                        # Simulation loop - CRITICAL: history saved EVERY step internally
                        for step in range(n_steps):
                            # Update status
                            status_text.text(f"Step {step+1}/{n_steps} | Time: {(step+1)*dt:.4f} ns")
                           
                            # Perform time step (history saved internally EVERY step)
                            results = solver.step(params['applied_stress'])
                           
                            # Save results at specified frequency for visualization/export
                            if step % save_frequency == 0:
                                results_history.append(results.copy())
                                timesteps.append(step * dt)
                           
                            # Update progress
                            progress_bar.progress((step + 1) / n_steps)
                           
                            # Real-time monitoring (every 10 steps)
                            if step % 10 == 0 and len(results_history) > 0:
                                with monitoring_cols[0]:
                                    st.metric("Avg Stress", f"{np.mean(results['sigma_eq'])/1e9:.2f} GPa")
                                with monitoring_cols[1]:
                                    valid_h = results['h'][(results['h']>5)&(results['h']<50)]
                                    avg_h = np.mean(valid_h) if len(valid_h) > 0 else 0
                                    st.metric("Avg Spacing", f"{avg_h:.1f} nm")
                                with monitoring_cols[2]:
                                    st.metric("Max Plastic Strain", f"{np.max(results['eps_p_mag']):.4f}")
                                with monitoring_cols[3]:
                                    st.metric("Energy", f"{results['convergence']['energy']:.2e} J")
                       
                        st.success(f"✅ Simulation completed! Generated {len(results_history)} frames.")
                        
                        # Store results
                        st.session_state.results_history = results_history
                        st.session_state.timesteps = timesteps # Saved frame timesteps
                        st.session_state.solver = solver # Contains FULL history (every step)
                       
                        # Save to database
                        sim_id = SimulationDatabase.generate_id(params)
                        SimulationDatabase.save_simulation(
                            params, results_history, 
                            st.session_state.initial_geometry, 
                            {'steps': n_steps, 'frames': len(results_history)}
                        )
                       
                        # Show completion message
                        st.balloons()
                    except Exception as e:
                        st.error(f"Simulation failed: {str(e)}")
                        st.exception(e)
                        st.info("Try adjusting parameters or reducing time step")
       
        with tabs[2]: # Basic Results
            if 'results_history' in st.session_state:
                st.header("Basic Results Visualization")
                results_history = st.session_state.results_history
                timesteps = st.session_state.timesteps
               
                # Stress magnitude debug output
                final_results = results_history[-1]
                avg_stress = np.mean(final_results['sigma_eq'])
                max_stress = np.max(final_results['sigma_eq'])
                st.info(f"Stress diagnostics: Avg = {avg_stress/1e6:.1f} MPa, Max = {max_stress/1e6:.1f} MPa")
                if avg_stress < 1e6:
                    st.warning("⚠️ Low stress detected (< 1 MPa). Check elastic parameters and applied stress.")
               
                # Select frame to display
                frame_idx = st.slider("Select frame", 0, len(results_history)-1, len(results_history)-1,
                                    help="Slide to view evolution at different time steps")
                results = results_history[frame_idx]
               
                # Create comprehensive 2D visualization
                st.subheader("2D Field Visualizations")
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
               
                # Row 1: Main fields
                axes[0, 0].imshow(results['phi'], cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                axes[0, 0].set_title('Twin Order Parameter φ')
                axes[0, 1].imshow(results['eta1'], cmap='Reds', vmin=0, vmax=1)
                axes[0, 1].set_title('Twin Grain η₁')
               
                # CRITICAL FIX: Adaptive colormap limits for stress visualization
                stress_gpa = results['sigma_eq'] / 1e9
                vmax_stress = np.percentile(stress_gpa, 95) # 95th percentile for better contrast
                vmin_stress = np.percentile(stress_gpa, 5)
                im2 = axes[0, 2].imshow(stress_gpa, cmap='hot', vmin=vmin_stress, vmax=vmax_stress)
                axes[0, 2].set_title(f'Von Mises Stress (GPa)\n[{vmin_stress:.2f}, {vmax_stress:.2f}]')
                plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
               
                axes[0, 3].imshow(results['h'], cmap='plasma', vmin=0, vmax=30)
                axes[0, 3].set_title('Twin Spacing (nm)')
               
                # Row 2: Secondary fields
                axes[1, 0].imshow(results['sigma_y']/1e6, cmap='viridis', vmin=0, vmax=500)
                axes[1, 0].set_title('Yield Stress (MPa)')
                axes[1, 1].imshow(results['eps_p_mag'], cmap='YlOrRd', vmin=0, vmax=np.percentile(results['eps_p_mag'], 95))
                axes[1, 1].set_title('Plastic Strain')
                axes[1, 2].imshow(results['eps_xx'], cmap='coolwarm', vmin=-0.01, vmax=0.01)
                axes[1, 2].set_title('Strain ε_xx')
                axes[1, 3].imshow(results['eps_xy'], cmap='coolwarm', vmin=-0.005, vmax=0.005)
                axes[1, 3].set_title('Shear Strain ε_xy')
               
                for ax in axes.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.tight_layout()
                st.pyplot(fig)
               
                # Convergence monitoring - FIXED WITH PROPER TIMESTEP MATCHING
                st.subheader("Convergence Monitoring")
                if hasattr(st.session_state.solver, 'history'):
                    # Use FULL history (every step) with properly matched timesteps
                    full_timesteps = np.arange(len(st.session_state.solver.history['phi_norm'])) * params['dt']
                   
                    convergence_fig = SimulationMonitor.create_convergence_plots(
                        st.session_state.solver.history,
                        full_timesteps # NOW MATCHES history length exactly
                    )
                    st.pyplot(convergence_fig)
            else:
                st.info("Run a simulation first to view results")
       
        with tabs[3]: # Advanced Analysis
            if 'results_history' in st.session_state:
                st.header("Advanced Analysis Tools")
               
                # Line Profile Analysis
                st.subheader("Line Profile Analysis")
               
                col1, col2 = st.columns(2)
                with col1:
                    profile_types = st.multiselect(
                        "Profile Directions",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"],
                        default=["Horizontal", "Vertical"]
                    )
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
               
                with col2:
                    field_to_profile = st.selectbox(
                        "Field to Profile",
                        ["phi", "sigma_eq", "h", "sigma_y"],
                        index=1
                    )
               
                results = st.session_state.results_history[-1]
               
                # Convert UI profile type names to internal profile type names
                profile_type_mapping = {
                    "Horizontal": "horizontal",
                    "Vertical": "vertical",
                    "Diagonal": "diagonal",
                    "Anti-Diagonal": "anti_diagonal",
                    "Custom": "custom"
                }
                internal_profile_types = [profile_type_mapping.get(pt, pt.lower().replace('-', '_')) for pt in profile_types]
               
                # Use EnhancedLineProfiler for line profiles
                profiler = EnhancedLineProfiler(N, dx)
               
                fig_profiles, axes = plt.subplots(len(internal_profile_types), 1, 
                                                figsize=(10, 4*len(internal_profile_types)))
                if len(internal_profile_types) == 1:
                    axes = [axes]
                
                for idx, profile_type in enumerate(internal_profile_types):
                    ax = axes[idx]
                    distance, profile, endpoints = profiler.extract_profile(
                        results[field_to_profile], profile_type, position_ratio
                    )
                    ax.plot(distance, profile, 'b-', linewidth=2)
                    ax.set_xlabel('Position (nm)')
                    ax.set_ylabel(field_to_profile.replace('_', ' ').title())
                    ax.set_title(f'{profile_type.replace("_", " ").title()} Profile')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_profiles)
            else:
                st.info("Run a simulation first to use advanced analysis tools")
       
        with tabs[4]: # Comparison Tools
            st.header("Multi-Simulation Comparison")
           
            simulations = SimulationDatabase.get_simulation_list()
           
            if not simulations:
                st.info("Run multiple simulations first to enable comparison")
            else:
                # Display available simulations
                st.subheader("Available Simulations")
               
                sim_data = []
                for sim in simulations:
                    sim_data.append({
                        'ID': sim['id'][:8],
                        'Twin Spacing': f"{sim['params'].get('twin_spacing', 0):.1f} nm",
                        'Applied Stress': f"{sim['params'].get('applied_stress', 0)/1e6:.0f} MPa",
                        'Frames': len(sim['results']) if sim['results'] else 0,
                        'Created': sim.get('created_at', '')[:19]
                    })
               
                df = pd.DataFrame(sim_data)
                st.dataframe(df, use_container_width=True)
               
                st.info("Configure comparison settings in the sidebar and click 'Run Comparison'")
       
        with tabs[5]: # Enhanced Export
            st.header("Enhanced Export Options")
           
            if 'results_history' in st.session_state:
                export_format = st.selectbox(
                    "Export Format",
                    ["Complete ZIP Package", "Field Arrays (NPZ)", "Statistics (CSV)",
                     "Publication Figures (PNG)", "PyTorch Tensors"],
                    index=0
                )
               
                include_options = st.multiselect(
                    "Include in export",
                    ["All Fields", "Convergence History", "Line Profiles",
                     "Statistical Analysis", "Correlation Plots"],
                    default=["All Fields", "Convergence History"]
                )
               
                if st.button("📦 Generate Enhanced Export", type="primary"):
                    with st.spinner("Preparing export package..."):
                        try:
                            # Create exporter
                            exporter = DataExporter()
                            # Generate export based on selected format
                            zip_buffer = exporter.export_simulation_results(
                                st.session_state.results_history,
                                params,
                                st.session_state.solver,
                                "nanotwin_simulation"
                            )
                            st.download_button(
                                label="⬇️ Download ZIP Package",
                                data=zip_buffer,
                                file_name=f"nanotwin_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
                            st.success("✅ Export ready for download!")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                            st.exception(e)
            else:
                st.info("Run a simulation first to export results")
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to the Enhanced Nanotwinned Copper Simulator</h2>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Configure simulation parameters in the sidebar and click "Initialize Simulation" to begin.
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 2rem;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem; border-radius: 10px; color: white;">
        <h4>🎯 Advanced Physics</h4>
        <p>Phase-field modeling of FCC nanotwins with spectral elasticity solver</p>
        </div>
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 1.5rem; border-radius: 10px; color: white;">
        <h4>📊 Enhanced Analysis</h4>
        <p>Line profiling, statistical analysis, correlation plots, multi-simulation comparison</p>
        </div>
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    padding: 1.5rem; border-radius: 10px; color: white;">
        <h4>🎨 Publication Ready</h4>
        <p>50+ colormaps, scale bars, journal templates, high-res export</p>
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
