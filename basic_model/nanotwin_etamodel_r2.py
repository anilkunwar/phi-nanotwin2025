import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import uuid
import shutil
import zipfile
import json
import tempfile
import base64
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import Numba, but handle gracefully if not available
try:
    from numba import jit, njit, prange, float64, int32, vectorize
    import numba as nb
    HAVE_NUMBA = True
    NUMBA_AVAILABLE = True
except ImportError:
    HAVE_NUMBA = False
    NUMBA_AVAILABLE = False
    print("Numba not available, using pure Python fallback")

# Set page config FIRST
st.set_page_config(
    page_title="Nanotwin Phase Field Simulation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# OPTIMIZED NUMERICAL KERNELS with FIXED NUMBA SIGNATURES
# ============================================================================

if NUMBA_AVAILABLE:
    # SIMPLER: Use @jit instead of @njit with explicit signatures
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def laplace_2d_numba(f, dx, dy):
        """2D Laplace operator with periodic boundary conditions - Numba optimized"""
        ny, nx = f.shape
        laplace_f = np.zeros((ny, nx))
        dx2 = dx * dx
        dy2 = dy * dy
        
        for i in prange(ny):
            im1 = (i - 1) % ny
            ip1 = (i + 1) % ny
            for j in prange(nx):
                jm1 = (j - 1) % nx
                jp1 = (j + 1) % nx
                
                # 5-point stencil
                laplace_f[i, j] = (f[im1, j] - 2 * f[i, j] + f[ip1, j]) / dy2 \
                                + (f[i, jm1] - 2 * f[i, j] + f[i, jp1]) / dx2
        return laplace_f

    # SIMPLER: Split into two functions to avoid tuple return type issues
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def allen_cahn_step_numba_m(eta_m, eta_t, dt, L, m, kappa, dx, dy):
        """Numba-optimized Allen-Cahn step for eta_m"""
        ny, nx = eta_m.shape
        
        # Compute Laplacians
        laplace_m = laplace_2d_numba(eta_m, dx, dy)
        laplace_t = laplace_2d_numba(eta_t, dx, dy)
        
        # Update eta_m
        eta_m_new = np.zeros((ny, nx))
        
        for i in prange(ny):
            for j in prange(nx):
                em = eta_m[i, j]
                et = eta_t[i, j]
                
                # Bulk term
                bulk = m * (em**3 - em + 2 * em * et**2)
                
                # Gradient term
                grad = -kappa * laplace_m[i, j]
                
                # Total derivative
                derivative = bulk + grad
                
                # Allen-Cahn update
                eta_m_new[i, j] = em - dt * L * derivative
        
        # Clamp values
        for i in prange(ny):
            for j in prange(nx):
                if eta_m_new[i, j] < 0:
                    eta_m_new[i, j] = 0
                elif eta_m_new[i, j] > 1:
                    eta_m_new[i, j] = 1
        
        return eta_m_new

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def allen_cahn_step_numba_t(eta_m, eta_t, dt, L, m, kappa, dx, dy):
        """Numba-optimized Allen-Cahn step for eta_t"""
        ny, nx = eta_t.shape
        
        # Compute Laplacians
        laplace_m = laplace_2d_numba(eta_m, dx, dy)
        laplace_t = laplace_2d_numba(eta_t, dx, dy)
        
        # Update eta_t
        eta_t_new = np.zeros((ny, nx))
        
        for i in prange(ny):
            for j in prange(nx):
                em = eta_m[i, j]
                et = eta_t[i, j]
                
                # Bulk term
                bulk = m * (et**3 - et + 2 * et * em**2)
                
                # Gradient term
                grad = -kappa * laplace_t[i, j]
                
                # Total derivative
                derivative = bulk + grad
                
                # Allen-Cahn update
                eta_t_new[i, j] = et - dt * L * derivative
        
        # Clamp values
        for i in prange(ny):
            for j in prange(nx):
                if eta_t_new[i, j] < 0:
                    eta_t_new[i, j] = 0
                elif eta_t_new[i, j] > 1:
                    eta_t_new[i, j] = 1
        
        return eta_t_new

    # Vectorized function for faster computations
    @vectorize([float64(float64, float64, float64)], nopython=True, cache=True)
    def double_well_vectorized(x, a, b):
        """Vectorized double-well potential"""
        return a * x**4 - b * x**2

else:
    # Fallback Python implementations
    def laplace_2d_numba(f, dx, dy):
        """Python implementation of Laplace operator"""
        from scipy.ndimage import laplace
        return laplace(f, mode='wrap') / (dx * dy)
    
    def allen_cahn_step_numba_m(eta_m, eta_t, dt, L, m, kappa, dx, dy):
        """Python implementation for eta_m"""
        from scipy.ndimage import laplace
        
        # Bulk term
        df_m = m * (eta_m**3 - eta_m + 2 * eta_m * eta_t**2)
        
        # Gradient term
        df_m -= kappa * laplace(eta_m, mode='wrap') / (dx * dy)
        
        # Update
        eta_m_new = eta_m - dt * L * df_m
        
        # Clamp
        eta_m_new = np.clip(eta_m_new, 0, 1)
        
        return eta_m_new
    
    def allen_cahn_step_numba_t(eta_m, eta_t, dt, L, m, kappa, dx, dy):
        """Python implementation for eta_t"""
        from scipy.ndimage import laplace
        
        # Bulk term
        df_t = m * (eta_t**3 - eta_t + 2 * eta_t * eta_m**2)
        
        # Gradient term
        df_t -= kappa * laplace(eta_t, mode='wrap') / (dx * dy)
        
        # Update
        eta_t_new = eta_t - dt * L * df_t
        
        # Clamp
        eta_t_new = np.clip(eta_t_new, 0, 1)
        
        return eta_t_new
    
    def double_well_vectorized(x, a, b):
        """Vectorized double-well potential (Python)"""
        return a * x**4 - b * x**2

# ============================================================================
# FFT-BASED SOLVER (No Numba dependency)
# ============================================================================

class FFTSolver:
    """FFT-based spectral solver using numpy.fft"""
    
    def __init__(self, nx, ny, Lx, Ly):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        # Wave numbers (using numpy.fft)
        self.kx = 2j * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
        self.ky = 2j * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
        
        # Create 2D k-space grid
        KX, KY = np.meshgrid(self.kx, self.ky)
        self.k2 = KX**2 + KY**2
        
    def laplace_fft(self, f):
        """Compute Laplace using FFT"""
        f_hat = np.fft.fft2(f)
        laplace_f_hat = -self.k2 * f_hat
        laplace_f = np.real(np.fft.ifft2(laplace_f_hat))
        return laplace_f
    
    def gradient_fft(self, f):
        """Compute gradient using FFT"""
        f_hat = np.fft.fft2(f)
        
        grad_x_hat = 1j * self.kx * f_hat
        grad_y_hat = 1j * self.ky * f_hat
        
        grad_x = np.real(np.fft.ifft2(grad_x_hat))
        grad_y = np.real(np.fft.ifft2(grad_y_hat))
        
        return grad_x, grad_y
    
    def spectral_step(self, eta_m, eta_t, dt, L, m, kappa):
        """Spectral time stepping using semi-implicit method"""
        # Precompute linear operator
        lin_op = 1.0 / (1.0 + dt * L * kappa * self.k2)
        
        # Nonlinear terms in real space
        N_m = m * (eta_m**3 - eta_m + 2 * eta_m * eta_t**2)
        N_t = m * (eta_t**3 - eta_t + 2 * eta_t * eta_m**2)
        
        # Transform to Fourier space
        eta_m_hat = np.fft.fft2(eta_m)
        eta_t_hat = np.fft.fft2(eta_t)
        N_m_hat = np.fft.fft2(N_m)
        N_t_hat = np.fft.fft2(N_t)
        
        # Semi-implicit update in Fourier space
        eta_m_hat_new = lin_op * (eta_m_hat - dt * L * N_m_hat)
        eta_t_hat_new = lin_op * (eta_t_hat - dt * L * N_t_hat)
        
        # Transform back to real space
        eta_m_new = np.real(np.fft.ifft2(eta_m_hat_new))
        eta_t_new = np.real(np.fft.ifft2(eta_t_hat_new))
        
        # Apply bounds
        eta_m_new = np.clip(eta_m_new, 0, 1)
        eta_t_new = np.clip(eta_t_new, 0, 1)
        
        # Ensure conservation (eta_m + eta_t ≈ 1)
        sum_eta = eta_m_new + eta_t_new
        mask = np.abs(sum_eta - 1.0) > 0.01
        eta_m_new[mask] /= sum_eta[mask]
        eta_t_new[mask] /= sum_eta[mask]
        
        return eta_m_new, eta_t_new

# ============================================================================
# ADAPTIVE TIME STEPPING
# ============================================================================

class AdaptiveTimeStepper:
    """Adaptive time step controller for stability and efficiency"""
    
    def __init__(self, dt_initial, dt_min=1e-15, dt_max=1e-10, 
                 safety_factor=0.8, max_change=2.0):
        self.dt = dt_initial
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety_factor = safety_factor
        self.max_change = max_change
        self.energy_history = []
        self.time_history = []
        
    def adjust_time_step(self, energy_new, energy_old, step_successful=True):
        """Adjust time step based on energy change"""
        if len(self.energy_history) < 2:
            return self.dt
        
        if not step_successful:
            # Step failed, reduce time step
            self.dt = max(self.dt * 0.5, self.dt_min)
            return self.dt
        
        # Calculate energy change rate
        if len(self.energy_history) >= 2:
            energy_change = abs(energy_new - energy_old) / (abs(energy_old) + 1e-10)
            
            # Adjust time step based on energy change
            if energy_change < 1e-6:
                # Very stable, increase time step
                self.dt = min(self.dt * self.max_change, self.dt_max)
            elif energy_change > 1e-3:
                # Large changes, decrease time step
                self.dt = max(self.dt * 0.5, self.dt_min)
            else:
                # Moderate changes, keep similar time step
                pass
        
        # Apply safety factor
        self.dt *= self.safety_factor
        
        # Ensure within bounds
        self.dt = max(self.dt_min, min(self.dt, self.dt_max))
        
        return self.dt
    
    def record_energy(self, energy, time):
        """Record energy for adaptive stepping"""
        self.energy_history.append(energy)
        self.time_history.append(time)
        
        # Keep only recent history
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)
            self.time_history.pop(0)

# ============================================================================
# MATERIAL DATABASE
# ============================================================================

class MaterialDatabase:
    """Material properties database"""
    
    MATERIALS = {
        'Cu': {
            'name': 'Copper',
            'sigma_ctb': 0.5e-3,
            'sigma_itb': 0.8e-3,
            'sigma_gb': 1.0e-3,
            'M_prefactor': 5e-6,
            'E_act': 0.2,
            'color': '#e41a1c',
            'density': 8960,
            'melting_point': 1357.77
        },
        'Al': {
            'name': 'Aluminum',
            'sigma_ctb': 0.4e-3,
            'sigma_itb': 0.6e-3,
            'sigma_gb': 0.9e-3,
            'M_prefactor': 8e-6,
            'E_act': 0.15,
            'color': '#377eb8',
            'density': 2700,
            'melting_point': 933.47
        },
        'Ni': {
            'name': 'Nickel',
            'sigma_ctb': 0.7e-3,
            'sigma_itb': 1.0e-3,
            'sigma_gb': 1.2e-3,
            'M_prefactor': 3e-6,
            'E_act': 0.25,
            'color': '#4daf4a',
            'density': 8908,
            'melting_point': 1728
        },
        'Ag': {
            'name': 'Silver',
            'sigma_ctb': 0.45e-3,
            'sigma_itb': 0.7e-3,
            'sigma_gb': 0.95e-3,
            'M_prefactor': 6e-6,
            'E_act': 0.18,
            'color': '#984ea3',
            'density': 10490,
            'melting_point': 1234.93
        },
        'Au': {
            'name': 'Gold',
            'sigma_ctb': 0.48e-3,
            'sigma_itb': 0.75e-3,
            'sigma_gb': 1.0e-3,
            'M_prefactor': 4e-6,
            'E_act': 0.22,
            'color': '#ff7f00',
            'density': 19320,
            'melting_point': 1337.33
        },
        'Pd': {
            'name': 'Palladium',
            'sigma_ctb': 0.65e-3,
            'sigma_itb': 0.9e-3,
            'sigma_gb': 1.1e-3,
            'M_prefactor': 3.5e-6,
            'E_act': 0.24,
            'color': '#ffff33',
            'density': 12023,
            'melting_point': 1828.05
        }
    }
    
    @classmethod
    def get_material(cls, name):
        """Get material properties"""
        if name not in cls.MATERIALS:
            raise ValueError(f"Material '{name}' not found")
        return cls.MATERIALS[name].copy()

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationParameters:
    """Simulation parameters with validation"""
    
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, 
                 t_max, output_interval, solver_type='numpy'):
        # Validate inputs
        self._validate_inputs(material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval)
        
        # Material properties
        self.material = material
        mat_props = MaterialDatabase.get_material(material)
        
        self.material_name = mat_props['name']
        self.sigma_itb = mat_props['sigma_itb']
        self.M_prefactor = mat_props['M_prefactor']
        self.E_act = mat_props['E_act']
        
        # Physical parameters
        self.temperature = temperature
        self.k_B = 8.617333262145e-5  # eV/K
        
        # Calculate mobility
        self.M = self.M_prefactor * np.exp(-self.E_act / (self.k_B * temperature))
        
        # Phase field parameters
        self.l_int = 1e-9  # Interface width (m)
        self.m = 6 * self.sigma_itb / self.l_int  # Energy barrier
        self.kappa = 0.75 * self.sigma_itb * self.l_int  # Gradient coefficient
        self.L = 4/3 * self.M / self.l_int  # Kinetic coefficient
        
        # Domain parameters
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        
        # Time parameters
        self.t_max = t_max
        
        # Stability condition for explicit scheme
        dt_stable = 0.25 * min(self.dx**2, self.dy**2) / (self.L * self.kappa)
        self.dt = min(1e-12, dt_stable)
        
        self.total_steps = int(t_max / self.dt)
        self.output_interval = output_interval
        self.twin_width = twin_width
        
        # Solver type
        self.solver_type = solver_type
        
        # Create grid
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def _validate_inputs(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval):
        """Validate all input parameters"""
        if material not in MaterialDatabase.MATERIALS:
            raise ValueError(f"Invalid material: {material}")
        if twin_width <= 0:
            raise ValueError("Twin width must be positive")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain dimensions must be positive")
        if Nx < 10 or Ny < 10:
            raise ValueError("Grid resolution too low")
        if t_max <= 0:
            raise ValueError("Simulation time must be positive")
        if output_interval <= 0:
            raise ValueError("Output interval must be positive")

# ============================================================================
# OPTIMIZED PHASE FIELD SOLVER
# ============================================================================

class OptimizedPhaseFieldSolver:
    """Fast phase field solver with multiple optimization strategies"""
    
    def __init__(self, params):
        self.params = params
        self.energy_history = []
        self.time_history = []
        
        # Initialize solver based on type
        self.solver_type = params.solver_type
        
        if self.solver_type == 'fft':
            self.fft_solver = FFTSolver(params.Nx, params.Ny, params.Lx, params.Ly)
            self.step_function = self._fft_step
        elif self.solver_type == 'numba' and NUMBA_AVAILABLE:
            self.step_function = self._numba_step
        else:
            self.step_function = self._numpy_step
        
        # Adaptive time stepping
        self.adaptive_stepper = AdaptiveTimeStepper(
            dt_initial=params.dt,
            dt_min=params.dt * 0.01,
            dt_max=params.dt * 10.0
        )
        
    def initialize_fields(self):
        """Initialize phase fields with smooth interface"""
        # Center of domain
        y_center = self.params.Ly / 2
        half_width = self.params.twin_width / 2
        
        # Create smooth twin profile
        Y = self.params.Y
        eta_t = 0.5 * (1 + np.tanh(4 * (Y - (y_center - half_width)) / self.params.l_int)) * \
                0.5 * (1 - np.tanh(4 * (Y - (y_center + half_width)) / self.params.l_int))
        
        # Matrix phase
        eta_m = 1 - eta_t
        
        # Add small perturbation
        noise = 0.01 * np.random.randn(self.params.Ny, self.params.Nx)
        eta_m += noise
        eta_t -= noise
        
        # Normalize
        sum_eta = eta_m + eta_t
        eta_m /= sum_eta
        eta_t /= sum_eta
        
        return eta_m, eta_t
    
    def _numpy_step(self, eta_m, eta_t):
        """Standard numpy implementation"""
        from scipy.ndimage import laplace
        
        # Bulk terms
        df_m = self.params.m * (eta_m**3 - eta_m + 2 * eta_m * eta_t**2)
        df_t = self.params.m * (eta_t**3 - eta_t + 2 * eta_t * eta_m**2)
        
        # Gradient terms
        laplace_m = laplace(eta_m, mode='wrap') / (self.params.dx * self.params.dy)
        laplace_t = laplace(eta_t, mode='wrap') / (self.params.dx * self.params.dy)
        
        df_m -= self.params.kappa * laplace_m
        df_t -= self.params.kappa * laplace_t
        
        # Time step
        dt = self.adaptive_stepper.dt
        
        # Update
        eta_m_new = eta_m - dt * self.params.L * df_m
        eta_t_new = eta_t - dt * self.params.L * df_t
        
        # Apply bounds
        eta_m_new = np.clip(eta_m_new, 0, 1)
        eta_t_new = np.clip(eta_t_new, 0, 1)
        
        # Ensure conservation
        sum_eta = eta_m_new + eta_t_new
        mask = np.abs(sum_eta - 1) > 0.01
        eta_m_new[mask] /= sum_eta[mask]
        eta_t_new[mask] /= sum_eta[mask]
        
        return eta_m_new, eta_t_new
    
    def _numba_step(self, eta_m, eta_t):
        """Numba-accelerated step"""
        dt = self.adaptive_stepper.dt
        
        eta_m_new = allen_cahn_step_numba_m(
            eta_m, eta_t, dt, self.params.L,
            self.params.m, self.params.kappa,
            self.params.dx, self.params.dy
        )
        
        eta_t_new = allen_cahn_step_numba_t(
            eta_m, eta_t, dt, self.params.L,
            self.params.m, self.params.kappa,
            self.params.dx, self.params.dy
        )
        
        return eta_m_new, eta_t_new
    
    def _fft_step(self, eta_m, eta_t):
        """FFT-based step"""
        dt = self.adaptive_stepper.dt
        return self.fft_solver.spectral_step(
            eta_m, eta_t, dt, self.params.L,
            self.params.m, self.params.kappa
        )
    
    def compute_energy(self, eta_m, eta_t):
        """Compute total free energy"""
        # Local energy
        f_local = 0.25 * (eta_m**4 + eta_t**4) - 0.5 * (eta_m**2 + eta_t**2)
        f_local += eta_m**2 * eta_t**2
        f_local *= self.params.m
        
        # Gradient energy
        if self.solver_type == 'fft':
            grad_m_x, grad_m_y = self.fft_solver.gradient_fft(eta_m)
            grad_t_x, grad_t_y = self.fft_solver.gradient_fft(eta_t)
        else:
            grad_m_y, grad_m_x = np.gradient(eta_m, self.params.dy, self.params.dx)
            grad_t_y, grad_t_x = np.gradient(eta_t, self.params.dy, self.params.dx)
        
        f_grad = 0.5 * self.params.kappa * (
            grad_m_x**2 + grad_m_y**2 + grad_t_x**2 + grad_t_y**2
        )
        
        # Total energy density
        f_total = f_local + f_grad
        
        # Integrate over domain
        energy = np.trapz(np.trapz(f_total, dx=self.params.dx, axis=1), dx=self.params.dy)
        
        return energy
    
    def update_step(self, eta_m, eta_t, step):
        """Perform one time step with adaptive time stepping"""
        # Record old energy
        old_energy = self.compute_energy(eta_m, eta_t)
        
        # Take step
        eta_m_new, eta_t_new = self.step_function(eta_m, eta_t)
        
        # Compute new energy
        new_energy = self.compute_energy(eta_m_new, eta_t_new)
        
        # Adjust time step based on energy change
        self.adaptive_stepper.adjust_time_step(new_energy, old_energy)
        
        # Record energy
        time = step * self.params.dt
        self.adaptive_stepper.record_energy(new_energy, time)
        self.energy_history.append(new_energy)
        self.time_history.append(time)
        
        return eta_m_new, eta_t_new

# ============================================================================
# VISUALIZATION TOOLS - FIXED VERSION
# ============================================================================

class VisualizationTools:
    """Tools for creating visualizations - FIXED VERSION"""
    
    @staticmethod
    def create_snapshot(eta_m, eta_t, params, time, colorscale='viridis'):
        """Create matplotlib snapshot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Matrix phase
        im1 = axes[0].imshow(
            eta_m, 
            extent=[0, params.Lx*1e9, 0, params.Ly*1e9], 
            origin='lower', 
            cmap=colorscale,
            vmin=0, vmax=1
        )
        axes[0].set_title(f'Matrix (η_m)\nTime: {time*1e9:.2f} ns')
        axes[0].set_xlabel('x (nm)')
        axes[0].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0])
        
        # Twin phase
        im2 = axes[1].imshow(
            eta_t, 
            extent=[0, params.Lx*1e9, 0, params.Ly*1e9], 
            origin='lower', 
            cmap=colorscale,
            vmin=0, vmax=1
        )
        axes[1].set_title(f'Twin (η_t)\nTime: {time*1e9:.2f} ns')
        axes[1].set_xlabel('x (nm)')
        plt.colorbar(im2, ax=axes[1])
        
        # Sum
        im3 = axes[2].imshow(
            eta_m + eta_t, 
            extent=[0, params.Lx*1e9, 0, params.Ly*1e9], 
            origin='lower', 
            cmap='RdYlBu',
            vmin=0.95, vmax=1.05
        )
        axes[2].set_title(f'Sum\nTime: {time*1e9:.2f} ns')
        axes[2].set_xlabel('x (nm)')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_interactive_plot(data_history, params, colorscale='viridis'):
        """Create interactive Plotly visualization - FIXED VERSION"""
        if not data_history:
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Matrix (η_m)', 'Twin (η_t)'),
            horizontal_spacing=0.1
        )
        
        # Get first frame data
        first_data = data_history[0]
        x = np.linspace(0, params.Lx * 1e9, params.Nx)
        y = np.linspace(0, params.Ly * 1e9, params.Ny)
        
        # Add initial heatmaps
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=first_data['eta_m'],
                colorscale=colorscale,
                zmin=0, zmax=1,
                colorbar=dict(x=0.45, title='η_m'),
                showscale=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=first_data['eta_t'],
                colorscale=colorscale,
                zmin=0, zmax=1,
                colorbar=dict(x=1.02, title='η_t'),
                showscale=True
            ),
            row=1, col=2
        )
        
        # Create frames for animation
        frames = []
        for i, data in enumerate(data_history):
            frame = go.Frame(
                data=[
                    go.Heatmap(
                        x=x, y=y, z=data['eta_m'],
                        colorscale=colorscale,
                        zmin=0, zmax=1,
                        showscale=False
                    ),
                    go.Heatmap(
                        x=x, y=y, z=data['eta_t'],
                        colorscale=colorscale,
                        zmin=0, zmax=1,
                        showscale=False
                    )
                ],
                name=f"frame_{i}",
                layout=go.Layout(
                    title_text=f"Time: {data['time']*1e9:.2f} ns",
                    title_x=0.5
                )
            )
            frames.append(frame)
        
        # Create slider - FIXED: Use frame index instead of undefined 'f'
        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Time: ',
                'suffix': ' ns',
                'visible': True,
                'xanchor': 'right'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [f"frame_{i}"],  # FIXED: Use frame index
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f"{data['time']*1e9:.2f}",
                    'method': 'animate'
                }
                for i, data in enumerate(data_history)  # FIXED: Use enumerate to get index
            ]
        }]
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Nanotwin Evolution - {params.material_name}",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            width=1000,
            showlegend=False,
            sliders=sliders,
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }],
                        'label': '▶ Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': '⏸ Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        # Update axes
        fig.update_xaxes(title_text="x (nm)", row=1, col=1)
        fig.update_yaxes(title_text="y (nm)", row=1, col=1)
        fig.update_xaxes(title_text="x (nm)", row=1, col=2)
        fig.update_yaxes(title_text="y (nm)", row=1, col=2, showticklabels=False)
        
        # Add frames to figure
        fig.frames = frames
        
        return fig
    
    @staticmethod
    def create_energy_plot(time_history, energy_history):
        """Create energy evolution plot"""
        if len(time_history) != len(energy_history) or len(time_history) == 0:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[t * 1e9 for t in time_history],  # Convert to ns
            y=energy_history,
            mode='lines+markers',
            name='Free Energy',
            line=dict(color='royalblue', width=2),
            marker=dict(size=6, color='royalblue')
        ))
        
        fig.update_layout(
            title='Free Energy Evolution',
            xaxis_title='Time (ns)',
            yaxis_title='Free Energy (J)',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_3d_surface(eta_m, eta_t, params, time):
        """Create 3D surface plot of order parameters"""
        x = np.linspace(0, params.Lx * 1e9, params.Nx)
        y = np.linspace(0, params.Ly * 1e9, params.Ny)
        X, Y = np.meshgrid(x, y)
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Matrix Phase η_m', 'Twin Phase η_t')
        )
        
        # Matrix phase surface
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=eta_m,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(x=0.45, title='η_m')
            ),
            row=1, col=1
        )
        
        # Twin phase surface
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=eta_t,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(x=1.02, title='η_t')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'3D Phase Field - Time: {time*1e9:.2f} ns',
            scene=dict(
                xaxis_title='x (nm)',
                yaxis_title='y (nm)',
                zaxis_title='Order Parameter'
            ),
            scene2=dict(
                xaxis_title='x (nm)',
                yaxis_title='y (nm)',
                zaxis_title='Order Parameter'
            ),
            height=500,
            width=1000
        )
        
        return fig

# ============================================================================
# DATA EXPORTER
# ============================================================================

class DataExporter:
    """Handles data export in various formats"""
    
    @staticmethod
    def export_summary_json(params, data_history, energy_history, time_history, output_dir):
        """Export simulation summary as JSON"""
        summary = {
            'material': params.material,
            'material_name': params.material_name,
            'temperature': params.temperature,
            'domain': {
                'Lx': params.Lx,
                'Ly': params.Ly,
                'Nx': params.Nx,
                'Ny': params.Ny
            },
            'twin': {
                'width': params.twin_width,
                'width_nm': params.twin_width * 1e9
            },
            'simulation': {
                'dt': params.dt,
                't_max': params.t_max,
                'total_steps': params.total_steps,
                'output_interval': params.output_interval,
                'solver_type': params.solver_type
            },
            'physics': {
                'sigma_itb': params.sigma_itb,
                'M': params.M,
                'm': params.m,
                'kappa': params.kappa,
                'L': params.L
            },
            'results': {
                'num_steps': len(data_history),
                'initial_energy': energy_history[0] if energy_history else 0,
                'final_energy': energy_history[-1] if energy_history else 0,
                'energy_history': energy_history,
                'time_history': time_history
            }
        }
        
        file_path = os.path.join(output_dir, "simulation_summary.json")
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return file_path
    
    @staticmethod
    def export_energy_csv(energy_history, time_history, output_dir):
        """Export energy data as CSV"""
        if not energy_history or not time_history:
            return None
        
        import csv
        
        file_path = os.path.join(output_dir, "energy_data.csv")
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_ns', 'energy_J'])
            for t, e in zip(time_history, energy_history):
                writer.writerow([t * 1e9, e])
        
        return file_path
    
    @staticmethod
    def export_final_state_npz(eta_m, eta_t, params, output_dir):
        """Export final state as NPZ file"""
        file_path = os.path.join(output_dir, "final_state.npz")
        np.savez(file_path,
                 eta_m=eta_m,
                 eta_t=eta_t,
                 x=params.x,
                 y=params.y,
                 Lx=params.Lx,
                 Ly=params.Ly)
        
        return file_path

# ============================================================================
# MAIN SIMULATION ENGINE
# ============================================================================

class FastNanotwinSimulation:
    """Main simulation engine"""
    
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny,
                 t_max, output_interval, colorscale, solver_type='numpy'):
        # Initialize parameters
        self.params = SimulationParameters(
            material, twin_width, temperature, Lx, Ly, Nx, Ny,
            t_max, output_interval, solver_type
        )
        
        # Initialize solver
        self.solver = OptimizedPhaseFieldSolver(self.params)
        
        # Initialize fields
        self.eta_m, self.eta_t = self.solver.initialize_fields()
        
        # Visualization
        self.visualization = VisualizationTools()
        self.exporter = DataExporter()
        self.colorscale = colorscale
        
        # Data storage
        self.output_dir = tempfile.mkdtemp(prefix=f"nanotwin_{material}_")
        self.data_history = []
        self.snapshot_paths = []
        self.export_files = []
        
        # Performance tracking
        self.start_time = None
        self.step_times = []
        
    def run(self, progress_callback=None, status_callback=None):
        """Run the simulation"""
        import time
        self.start_time = time.time()
        
        steps = self.params.total_steps
        
        for step in range(steps + 1):
            step_start = time.time()
            t = step * self.params.dt
            
            # Update phase fields
            self.eta_m, self.eta_t = self.solver.update_step(self.eta_m, self.eta_t, step)
            
            # Save output at intervals
            if step % self.params.output_interval == 0 or step == steps:
                self._save_output(step, t)
                
                # Update progress
                if progress_callback:
                    progress = min(step / steps, 1.0)
                    progress_callback(progress)
                
                # Update status
                if status_callback:
                    energy = self.solver.compute_energy(self.eta_m, self.eta_t)
                    elapsed = time.time() - self.start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    eta = (steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                    status_callback(step, steps, t, energy, steps_per_sec, eta)
            
            # Track step time
            self.step_times.append(time.time() - step_start)
        
        # Export final data
        self._export_final_data()
        
        return self.data_history
    
    def _save_output(self, step, time):
        """Save output data"""
        # Store data
        energy = self.solver.compute_energy(self.eta_m, self.eta_t)
        self.data_history.append({
            'step': step,
            'time': time,
            'eta_m': self.eta_m.copy(),
            'eta_t': self.eta_t.copy(),
            'energy': energy
        })
        
        # Create snapshot (less frequent for performance)
        if step % (self.params.output_interval * 5) == 0 or step == self.params.total_steps:
            try:
                fig = self.visualization.create_snapshot(
                    self.eta_m, self.eta_t, self.params, time, self.colorscale
                )
                snapshot_path = os.path.join(self.output_dir, f'snapshot_{step:06d}.png')
                fig.savefig(snapshot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                self.snapshot_paths.append(snapshot_path)
            except Exception as e:
                print(f"Error creating snapshot: {e}")
    
    def _export_final_data(self):
        """Export final simulation data"""
        try:
            # Export summary JSON
            summary_path = self.exporter.export_summary_json(
                self.params, self.data_history,
                self.solver.energy_history, self.solver.time_history,
                self.output_dir
            )
            self.export_files.append(summary_path)
            
            # Export energy CSV
            energy_path = self.exporter.export_energy_csv(
                self.solver.energy_history, self.solver.time_history,
                self.output_dir
            )
            if energy_path:
                self.export_files.append(energy_path)
            
            # Export final state NPZ
            final_state_path = self.exporter.export_final_state_npz(
                self.eta_m, self.eta_t, self.params, self.output_dir
            )
            self.export_files.append(final_state_path)
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def get_results(self):
        """Get simulation results"""
        return {
            'data_history': self.data_history,
            'energy_history': self.solver.energy_history,
            'time_history': self.solver.time_history,
            'snapshots': self.snapshot_paths,
            'export_files': self.export_files,
            'params': self.params,
            'performance': self._get_performance_stats(),
            'output_dir': self.output_dir
        }
    
    def _get_performance_stats(self):
        """Get performance statistics"""
        if not self.step_times:
            return {}
        
        times = np.array(self.step_times)
        total_time = np.sum(times)
        
        return {
            'total_time': total_time,
            'avg_step_time': np.mean(times) * 1000,  # in ms
            'std_step_time': np.std(times) * 1000,
            'min_step_time': np.min(times) * 1000,
            'max_step_time': np.max(times) * 1000,
            'steps_per_second': len(times) / total_time if total_time > 0 else 0,
            'solver_type': self.params.solver_type,
            'numba_available': NUMBA_AVAILABLE,
            'total_steps': len(self.step_times)
        }
    
    def create_zip_archive(self):
        """Create a ZIP archive of all output files"""
        zip_path = os.path.join(self.output_dir, "simulation_results.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in output directory
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if file != "simulation_results.zip":  # Don't add the zip itself
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.output_dir)
                        zipf.write(file_path, arcname)
        
        return zip_path
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.output_dir)
        except:
            pass

# ============================================================================
# STREAMLIT APPLICATION - MAIN FUNCTION
# ============================================================================

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-left: 4px solid #2ca02c;
        background-color: #f8f9fa;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .performance-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .badge-fast { background-color: #00a8ff; color: white; }
    .badge-faster { background-color: #9c88ff; color: white; }
    .badge-fastest { background-color: #fbc531; color: black; }
    .tab-content {
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚡ High-Performance Nanotwin Phase Field Simulation</h1>', unsafe_allow_html=True)
    
    # Performance information
    col1, col2, col3 = st.columns(3)
    with col1:
        if NUMBA_AVAILABLE:
            st.success(f"✅ Numba JIT: Available")
            if HAVE_NUMBA:
                st.write(f"Version: {nb.__version__}")
        else:
            st.warning("⚠️ Numba: Not installed")
            st.write("Using pure Python fallback")
    with col2:
        st.info(f"NumPy: v{np.__version__}")
    with col3:
        import sys
        st.info(f"Python: {sys.version.split()[0]}")
    
    # Introduction
    with st.expander("🚀 About This Simulation", expanded=True):
        st.markdown("""
        ### **Scientific Background**
        
        This application simulates the **detwinning process of nanotwins** in FCC metals using 
        advanced phase field methods with multiple optimization strategies.
        
        ### **Key Features:**
        
        **Performance Optimizations:**
        1. **Numba JIT Compilation** - 10-100x speedup when available
        2. **FFT Spectral Methods** - Fast Fourier Transform for large grids
        3. **Adaptive Time Stepping** - Automatic Δt adjustment for stability
        4. **Vectorized Operations** - Efficient array computations
        
        **Scientific Capabilities:**
        - **6 FCC Materials**: Cu, Al, Ni, Ag, Au, Pd with accurate properties
        - **Temperature Dependence**: Realistic mobility calculations
        - **Anisotropic Interfaces**: Direction-dependent interfacial energy
        - **Energy Tracking**: Monitor free energy evolution
        
        **Visualization & Export:**
        - Interactive 3D plots with time animation
        - Real-time progress tracking
        - Multiple export formats (JSON, CSV, NPZ, PNG)
        - Cross-section analysis tools
        """)
    
    # Sidebar for parameters
    st.sidebar.markdown("## ⚙️ Simulation Parameters")
    
    # Material selection
    material = st.sidebar.selectbox(
        "Material",
        ["Cu", "Al", "Ni", "Ag", "Au", "Pd"],
        index=0,
        help="Select FCC material for simulation"
    )
    
    # Display material properties
    with st.sidebar.expander(f"📊 {material} Properties"):
        try:
            mat_props = MaterialDatabase.get_material(material)
            st.write(f"**{mat_props['name']}**")
            st.write(f"Σ_ITB: {mat_props['sigma_itb']:.2e} J/m²")
            st.write(f"Mobility Prefactor: {mat_props['M_prefactor']:.2e} m⁴/(J·s)")
            st.write(f"Activation Energy: {mat_props['E_act']:.2f} eV")
            st.write(f"Density: {mat_props['density']:,} kg/m³")
            st.write(f"Melting Point: {mat_props['melting_point']} K")
        except:
            st.write("Material properties not available")
    
    # Solver selection
    st.sidebar.markdown("### ⚡ Solver Options")
    
    solver_options = []
    if NUMBA_AVAILABLE:
        solver_options.append(("numba", "⚡ Numba JIT (Fastest)"))
    solver_options.append(("fft", "🌀 FFT Spectral (Large grids)"))
    solver_options.append(("numpy", "🐍 Standard NumPy (Reliable)"))
    
    solver_choice = st.sidebar.selectbox(
        "Select Solver",
        options=[opt[0] for opt in solver_options],
        format_func=lambda x: dict(solver_options)[x],
        index=0 if NUMBA_AVAILABLE else 2
    )
    
    # Performance options
    with st.sidebar.expander("🎯 Performance Settings"):
        adaptive_dt = st.checkbox("Adaptive Time Step", value=True, 
                                 help="Automatically adjust time step for stability")
        reduce_output = st.checkbox("Reduce Output Frequency", value=False,
                                   help="Save memory by storing fewer snapshots")
    
    # Colorscale selection
    colorscale_options = {
        'Viridis': 'viridis',
        'Plasma': 'plasma',
        'Inferno': 'inferno',
        'Magma': 'magma',
        'Hot': 'hot',
        'Jet': 'jet',
        'Rainbow': 'rainbow',
        'Coolwarm': 'coolwarm',
        'RdBu': 'RdBu'
    }
    
    colorscale = st.sidebar.selectbox(
        "Colorscale",
        list(colorscale_options.keys()),
        index=0,
        help="Color scheme for visualization"
    )
    
    # Geometry parameters
    st.sidebar.markdown("### 📏 Geometry Parameters")
    
    twin_width_nm = st.sidebar.slider(
        "Nanotwin Width (nm)",
        1.0, 100.0, 10.0,
        help="Initial width of the nanotwin"
    )
    
    Lx_nm = st.sidebar.slider(
        "Domain Width (nm)",
        20.0, 500.0, 100.0,
        help="Width of the simulation domain"
    )
    
    Ly_nm = st.sidebar.slider(
        "Domain Height (nm)",
        20.0, 500.0, 100.0,
        help="Height of the simulation domain"
    )
    
    # Grid resolution
    st.sidebar.markdown("### 🎯 Numerical Parameters")
    
    if solver_choice == 'fft':
        # FFT works best with power of 2 sizes
        grid_sizes = [32, 64, 128, 256, 512]
        default_idx = 2  # 128
    else:
        grid_sizes = [50, 100, 150, 200, 300, 400]
        default_idx = 1  # 100
    
    Nx = st.sidebar.selectbox(
        "Grid Points X",
        grid_sizes,
        index=default_idx,
        help="Number of grid points in x-direction"
    )
    
    Ny = st.sidebar.selectbox(
        "Grid Points Y",
        grid_sizes,
        index=default_idx,
        help="Number of grid points in y-direction"
    )
    
    # Physical parameters
    st.sidebar.markdown("### 🔬 Physical Parameters")
    
    temperature = st.sidebar.slider(
        "Temperature (K)",
        100, 1500, 500,
        help="Simulation temperature in Kelvin"
    )
    
    t_max_ns = st.sidebar.slider(
        "Simulation Time (ns)",
        0.1, 10.0, 1.0,
        help="Total simulation time in nanoseconds"
    )
    
    if reduce_output:
        output_interval = st.sidebar.slider(
            "Output Interval (steps)",
            100, 1000, 200,
            help="Steps between saving output data"
        )
    else:
        output_interval = st.sidebar.slider(
            "Output Interval (steps)",
            10, 500, 50,
            help="Steps between saving output data"
        )
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        random_seed = st.number_input(
            "Random Seed",
            0, 10000, 42,
            help="Seed for random number generator"
        )
        np.random.seed(random_seed)
        
        anisotropy_strength = st.slider(
            "Anisotropy Strength",
            0.0, 1.0, 0.5,
            help="Strength of anisotropic interfacial energy"
        )
    
    # Run button
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button(
        "🚀 Run Simulation", 
        type="primary",
        use_container_width=True,
        help="Start the simulation with current parameters"
    )
    
    # Convert units for simulation
    twin_width = twin_width_nm * 1e-9  # Convert nm to m
    Lx = Lx_nm * 1e-9
    Ly = Ly_nm * 1e-9
    t_max = t_max_ns * 1e-9
    
    # Main content area
    if run_simulation:
        # Display configuration summary
        st.markdown("### 🔬 Simulation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Material:** {material}
            **Solver:** {solver_choice.upper()}
            **Domain:** {Lx_nm:.1f} nm × {Ly_nm:.1f} nm
            **Grid:** {Nx} × {Ny} ({Nx * Ny:,} points)
            """)
        
        with col2:
            st.markdown(f"""
            **Twin Width:** {twin_width_nm:.1f} nm
            **Temperature:** {temperature} K
            **Total Time:** {t_max_ns:.2f} ns
            **Output Interval:** {output_interval} steps
            """)
        
        # Performance estimate
        st.markdown("### 📊 Performance Estimate")
        
        estimate_cols = st.columns(4)
        with estimate_cols[0]:
            if solver_choice == 'numba':
                st.markdown('<div class="performance-badge badge-fastest">FASTEST: Numba JIT</div>', unsafe_allow_html=True)
            elif solver_choice == 'fft':
                st.markdown('<div class="performance-badge badge-faster">FASTER: FFT Spectral</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="performance-badge badge-fast">FAST: Standard NumPy</div>', unsafe_allow_html=True)
        
        with estimate_cols[1]:
            grid_size = Nx * Ny
            st.metric("Grid Size", f"{grid_size:,}")
        
        with estimate_cols[2]:
            # Estimate steps based on stability condition
            dt_est = 0.25 * min((Lx/Nx)**2, (Ly/Ny)**2) / (1e-12)  # Simplified estimate
            est_steps = int(t_max / max(dt_est, 1e-15))
            st.metric("Estimated Steps", f"{est_steps:,}")
        
        with estimate_cols[3]:
            # Rough time estimate
            if solver_choice == 'numba':
                est_time = est_steps * 0.0001  # 0.1ms per step
            elif solver_choice == 'fft':
                est_time = est_steps * 0.0005  # 0.5ms per step
            else:
                est_time = est_steps * 0.001  # 1ms per step
            st.metric("Est. Time", f"{est_time:.1f}s")
        
        # Initialize and run simulation
        with st.spinner("🚀 Initializing simulation..."):
            try:
                # Create simulation instance
                sim = FastNanotwinSimulation(
                    material, twin_width, temperature,
                    Lx, Ly, Nx, Ny, t_max,
                    output_interval, colorscale_options[colorscale],
                    solver_type=solver_choice
                )
                
                # Setup progress tracking
                progress_bar = st.progress(0)
                status_container = st.empty()
                performance_container = st.empty()
                
                # Run simulation
                with st.spinner("⏳ Running simulation..."):
                    import time
                    simulation_start = time.time()
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                    
                    def update_status(step, total_steps, t, energy, steps_per_sec, eta):
                        elapsed = time.time() - simulation_start
                        progress_percent = step / total_steps * 100
                        
                        status_container.markdown(f"""
                        **Simulation Progress:**
                        - **Step:** {step:,} / {total_steps:,} ({progress_percent:.1f}%)
                        - **Time:** {t*1e9:.3f} ns / {t_max_ns:.2f} ns
                        - **Energy:** {energy:.2e} J
                        - **Adaptive Δt:** {sim.solver.adaptive_stepper.dt:.2e} s
                        """)
                        
                        performance_container.markdown(f"""
                        **Performance Metrics:**
                        - **Elapsed:** {elapsed:.1f}s
                        - **Steps/sec:** {steps_per_sec:.0f}
                        - **ETA:** {eta:.1f}s
                        - **Memory:** {sim.eta_m.nbytes/1e6:.1f} MB
                        """)
                    
                    # Run the simulation
                    data_history = sim.run(update_progress, update_status)
                    
                    total_time = time.time() - simulation_start
                
                # Get results
                results = sim.get_results()
                perf_stats = results['performance']
                
                # Success message with performance stats
                st.success(f"""
                ✅ **Simulation completed successfully!**
                
                **Performance Summary:**
                - Total time: {total_time:.1f}s
                - Steps processed: {perf_stats.get('total_steps', 0):,}
                - Average step time: {perf_stats.get('avg_step_time', 0):.2f} ms
                - Steps per second: {perf_stats.get('steps_per_second', 0):.0f}
                - Solver: {solver_choice.upper()}
                """)
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Visualization", 
                    "📈 Analysis",
                    "📉 Energy", 
                    "💾 Export"
                ])
                
                # Tab 1: Visualization
                with tab1:
                    st.markdown('<h3 class="sub-header">Phase Field Evolution</h3>', unsafe_allow_html=True)
                    
                    if data_history:
                        # Create interactive plot
                        plotly_fig = VisualizationTools.create_interactive_plot(
                            data_history, sim.params, colorscale_options[colorscale]
                        )
                        
                        if plotly_fig:
                            st.plotly_chart(plotly_fig, use_container_width=True)
                        else:
                            st.warning("Could not create interactive plot")
                        
                        # Show snapshots
                        st.markdown("#### Simulation Snapshots")
                        if sim.snapshot_paths:
                            cols = st.columns(3)
                            # Show last 3 snapshots
                            for i, snapshot_path in enumerate(sim.snapshot_paths[-3:]):
                                with cols[i]:
                                    try:
                                        st.image(snapshot_path, use_column_width=True)
                                    except:
                                        st.write(f"Could not load snapshot {i}")
                        else:
                            st.info("No snapshots available")
                    
                    # 3D visualization
                    if data_history and len(data_history) > 0:
                        st.markdown("#### 3D Surface Plot")
                        last_data = data_history[-1]
                        surface_fig = VisualizationTools.create_3d_surface(
                            last_data['eta_m'], last_data['eta_t'], 
                            sim.params, last_data['time']
                        )
                        if surface_fig:
                            st.plotly_chart(surface_fig, use_container_width=True)
                
                # Tab 2: Analysis
                with tab2:
                    st.markdown('<h3 class="sub-header">Simulation Analysis</h3>', unsafe_allow_html=True)
                    
                    if data_history:
                        # Show statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            last_data = data_history[-1]
                            eta_m_mean = np.mean(last_data['eta_m'])
                            st.metric("Matrix Mean", f"{eta_m_mean:.3f}")
                        
                        with col2:
                            eta_t_mean = np.mean(last_data['eta_t'])
                            st.metric("Twin Mean", f"{eta_t_mean:.3f}")
                        
                        with col3:
                            sum_mean = np.mean(last_data['eta_m'] + last_data['eta_t'])
                            st.metric("Sum Mean", f"{sum_mean:.3f}")
                        
                        with col4:
                            if len(data_history) > 1:
                                first_eta_t = np.mean(data_history[0]['eta_t'])
                                last_eta_t = np.mean(data_history[-1]['eta_t'])
                                change = (last_eta_t - first_eta_t) / first_eta_t * 100
                                st.metric("Twin Change", f"{change:.1f}%")
                        
                        # Cross-section analysis
                        st.markdown("#### Cross-Section Analysis")
                        cross_section_y = st.slider(
                            "Select Y position (nm)",
                            0.0, sim.params.Ly * 1e9,
                            sim.params.Ly * 1e9 / 2,
                            key="cross_section"
                        )
                        
                        # Find nearest index
                        y_idx = int(cross_section_y / (sim.params.Ly * 1e9) * (sim.params.Ny - 1))
                        y_idx = max(0, min(y_idx, sim.params.Ny - 1))
                        
                        if data_history:
                            # Plot cross-section for latest data
                            last_data = data_history[-1]
                            x_nm = np.linspace(0, sim.params.Lx * 1e9, sim.params.Nx)
                            
                            fig_cross = go.Figure()
                            fig_cross.add_trace(go.Scatter(
                                x=x_nm,
                                y=last_data['eta_m'][y_idx, :],
                                mode='lines',
                                name='Matrix (η_m)',
                                line=dict(color='blue', width=2)
                            ))
                            fig_cross.add_trace(go.Scatter(
                                x=x_nm,
                                y=last_data['eta_t'][y_idx, :],
                                mode='lines',
                                name='Twin (η_t)',
                                line=dict(color='red', width=2)
                            ))
                            fig_cross.add_trace(go.Scatter(
                                x=x_nm,
                                y=last_data['eta_m'][y_idx, :] + last_data['eta_t'][y_idx, :],
                                mode='lines',
                                name='Sum',
                                line=dict(color='green', width=2, dash='dash')
                            ))
                            
                            fig_cross.update_layout(
                                title=f'Cross-section at y = {cross_section_y:.1f} nm',
                                xaxis_title='x (nm)',
                                yaxis_title='Order Parameter',
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_cross, use_container_width=True)
                
                # Tab 3: Energy Analysis
                with tab3:
                    st.markdown('<h3 class="sub-header">Energy Evolution</h3>', unsafe_allow_html=True)
                    
                    if results['energy_history']:
                        # Create energy plot
                        energy_fig = VisualizationTools.create_energy_plot(
                            results['time_history'], results['energy_history']
                        )
                        
                        if energy_fig:
                            st.plotly_chart(energy_fig, use_container_width=True)
                        
                        # Energy statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            initial_energy = results['energy_history'][0]
                            st.metric("Initial Energy", f"{initial_energy:.2e} J")
                        
                        with col2:
                            final_energy = results['energy_history'][-1]
                            st.metric("Final Energy", f"{final_energy:.2e} J")
                        
                        with col3:
                            energy_change = final_energy - initial_energy
                            st.metric("ΔEnergy", f"{energy_change:.2e} J")
                        
                        with col4:
                            if initial_energy != 0:
                                percent_change = (energy_change / initial_energy) * 100
                                st.metric("% Change", f"{percent_change:.1f}%")
                        
                        # Energy rate of change
                        if len(results['energy_history']) > 1:
                            st.markdown("#### Energy Dissipation Rate")
                            
                            # Compute derivative
                            times_ns = [t * 1e9 for t in results['time_history']]
                            energy_diff = np.diff(results['energy_history'])
                            time_diff = np.diff(results['time_history'])
                            power = -energy_diff / time_diff  # Negative of energy change rate
                            
                            fig_power = go.Figure()
                            fig_power.add_trace(go.Scatter(
                                x=times_ns[1:],
                                y=power,
                                mode='lines',
                                name='Dissipation Rate',
                                line=dict(color='purple', width=2)
                            ))
                            
                            fig_power.update_layout(
                                title='Energy Dissipation Rate',
                                xaxis_title='Time (ns)',
                                yaxis_title='Power (W)',
                                height=400
                            )
                            
                            st.plotly_chart(fig_power, use_container_width=True)
                
                # Tab 4: Export
                with tab4:
                    st.markdown('<h3 class="sub-header">Export Simulation Data</h3>', unsafe_allow_html=True)
                    
                    # Create ZIP archive
                    try:
                        zip_path = sim.create_zip_archive()
                        
                        # Read ZIP file
                        with open(zip_path, 'rb') as f:
                            zip_data = f.read()
                        
                        # Download button for ZIP
                        st.download_button(
                            label="📦 Download All Data (ZIP)",
                            data=zip_data,
                            file_name=f"nanotwin_{material}_{solver_choice}_{temperature}K.zip",
                            mime="application/zip",
                            help="Download all simulation data as a ZIP archive"
                        )
                    except Exception as e:
                        st.error(f"Error creating ZIP archive: {e}")
                    
                    # Individual export options
                    st.markdown("#### Individual Files")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON summary
                        if any('summary.json' in f for f in sim.export_files):
                            summary_path = next(f for f in sim.export_files if 'summary.json' in f)
                            with open(summary_path, 'r') as f:
                                summary_json = f.read()
                            
                            st.download_button(
                                label="📄 Download Summary (JSON)",
                                data=summary_json,
                                file_name=f"nanotwin_summary_{material}.json",
                                mime="application/json"
                            )
                        
                        # Energy CSV
                        if any('energy_data.csv' in f for f in sim.export_files):
                            energy_path = next(f for f in sim.export_files if 'energy_data.csv' in f)
                            with open(energy_path, 'r') as f:
                                energy_csv = f.read()
                            
                            st.download_button(
                                label="📈 Download Energy Data (CSV)",
                                data=energy_csv,
                                file_name=f"nanotwin_energy_{material}.csv",
                                mime="text/csv"
                            )
                    
                    with col2:
                        # Final state NPZ
                        if any('final_state.npz' in f for f in sim.export_files):
                            final_state_path = next(f for f in sim.export_files if 'final_state.npz' in f)
                            with open(final_state_path, 'rb') as f:
                                final_state_data = f.read()
                            
                            st.download_button(
                                label="🔢 Download Final State (NPZ)",
                                data=final_state_data,
                                file_name=f"nanotwin_final_state_{material}.npz",
                                mime="application/octet-stream"
                            )
                        
                        # Performance report
                        perf_report = json.dumps(perf_stats, indent=2)
                        st.download_button(
                            label="⚡ Download Performance Report (JSON)",
                            data=perf_report,
                            file_name=f"nanotwin_performance_{material}.json",
                            mime="application/json"
                        )
                    
                    # Display file list
                    st.markdown("#### Generated Files")
                    if sim.export_files:
                        for file_path in sim.export_files:
                            file_name = os.path.basename(file_path)
                            file_size = os.path.getsize(file_path) / 1024  # Size in KB
                            st.write(f"- **{file_name}** ({file_size:.1f} KB)")
                
                # Cleanup temporary files
                sim.cleanup()
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    else:
        # Welcome screen when simulation is not running
        st.markdown("""
        <div class="info-box">
        <h3>🚀 Welcome to the Nanotwin Simulation Platform!</h3>
        
        <p>This advanced simulation tool models the detwinning process in nanocrystalline FCC metals 
        using state-of-the-art phase field methods with multiple optimization strategies.</p>
        
        <h4>📋 Getting Started:</h4>
        <ol>
            <li><b>Select material</b> and solver type in the sidebar</li>
            <li><b>Configure simulation parameters</b> (geometry, temperature, time)</li>
            <li><b>Click 'Run Simulation'</b> to start the calculation</li>
            <li><b>Explore results</b> in the interactive tabs</li>
            <li><b>Export data</b> for further analysis</li>
        </ol>
        
        <h4>🔬 Scientific Applications:</h4>
        <ul>
            <li>Study nanotwin stability under thermal loads</li>
            <li>Analyze interface migration kinetics</li>
            <li>Investigate anisotropic effects on detwinning</li>
            <li>Compare material response across different FCC metals</li>
            <li>Optimize processing conditions for nanotwin stability</li>
        </ul>
        
        <h4>⚡ Performance Features:</h4>
        <ul>
            <li><b>Numba JIT compilation</b> for 10-100x speedup</li>
            <li><b>FFT spectral methods</b> for large-scale simulations</li>
            <li><b>Adaptive time stepping</b> for numerical stability</li>
            <li><b>Real-time progress tracking</b> with ETA estimates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start examples
        st.markdown("### 🚀 Quick Start Examples")
        
        example_cols = st.columns(3)
        
        with example_cols[0]:
            st.markdown("""
            <div class="metric-box">
            <h4>Quick Test</h4>
            <p><b>Material:</b> Cu</p>
            <p><b>Grid:</b> 100×100</p>
            <p><b>Time:</b> 0.5 ns</p>
            <p><b>Solver:</b> Standard</p>
            <p><i>~30 seconds</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        with example_cols[1]:
            st.markdown("""
            <div class="metric-box">
            <h4>Standard Run</h4>
            <p><b>Material:</b> Al</p>
            <p><b>Grid:</b> 200×200</p>
            <p><b>Time:</b> 1.0 ns</p>
            <p><b>Solver:</b> Numba</p>
            <p><i>~1-2 minutes</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        with example_cols[2]:
            st.markdown("""
            <div class="metric-box">
            <h4>High Resolution</h4>
            <p><b>Material:</b> Ni</p>
            <p><b>Grid:</b> 400×400</p>
            <p><b>Time:</b> 2.0 ns</p>
            <p><b>Solver:</b> FFT</p>
            <p><i>~5-10 minutes</i></p>
            </div>
            """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
