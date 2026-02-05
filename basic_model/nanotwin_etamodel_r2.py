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
    # Use simpler signature without complex tuple returns
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
# VISUALIZATION TOOLS
# ============================================================================

class VisualizationTools:
    """Tools for creating visualizations"""
    
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
        """Create interactive Plotly visualization"""
        if not data_history:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Matrix (η_m)', 'Twin (η_t)'),
            horizontal_spacing=0.1
        )
        
        # First frame
        first_data = data_history[0]
        x = np.linspace(0, params.Lx * 1e9, params.Nx)
        y = np.linspace(0, params.Ly * 1e9, params.Ny)
        
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=first_data['eta_m'],
                colorscale=colorscale,
                zmin=0, zmax=1,
                colorbar=dict(x=0.45, title='η_m')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=first_data['eta_t'],
                colorscale=colorscale,
                zmin=0, zmax=1,
                colorbar=dict(x=1.02, title='η_t')
            ),
            row=1, col=2
        )
        
        # Create frames
        frames = []
        for i, data in enumerate(data_history):
            frame = go.Frame(
                data=[
                    go.Heatmap(z=data['eta_m'], colorscale=colorscale),
                    go.Heatmap(z=data['eta_t'], colorscale=colorscale)
                ],
                name=f"frame_{i}",
                layout=go.Layout(
                    title_text=f"Time: {data['time']*1e9:.2f} ns"
                )
            )
            frames.append(frame)
        
        # Create slider
        sliders = [{
            'active': 0,
            'currentvalue': {'prefix': 'Time: ', 'suffix': ' ns'},
            'pad': {'t': 50},
            'steps': [
                {
                    'args': [[f.name], {'frame': {'duration': 300, 'redraw': True}}],
                    'label': f"{data['time']*1e9:.2f}",
                    'method': 'animate'
                }
                for data in data_history
            ]
        }]
        
        # Update layout
        fig.update_layout(
            title=f"Nanotwin Evolution - {params.material_name}",
            height=500,
            width=1000,
            sliders=sliders,
            updatemenus=[{
                'buttons': [
                    {'args': [None, {'frame': {'duration': 500}}], 'label': '▶ Play', 'method': 'animate'},
                    {'args': [[None], {'frame': {'duration': 0}}], 'label': '⏸ Pause', 'method': 'animate'}
                ]
            }]
        )
        
        fig.update_xaxes(title_text="x (nm)", row=1, col=1)
        fig.update_yaxes(title_text="y (nm)", row=1, col=1)
        fig.update_xaxes(title_text="x (nm)", row=1, col=2)
        
        fig.frames = frames
        return fig

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
        self.colorscale = colorscale
        
        # Data storage
        self.output_dir = tempfile.mkdtemp(prefix=f"nanotwin_{material}_")
        self.data_history = []
        self.snapshot_paths = []
        
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
                    status_callback(step, steps, t, energy, steps_per_sec)
            
            # Track step time
            self.step_times.append(time.time() - step_start)
        
        return self.data_history
    
    def _save_output(self, step, time):
        """Save output data"""
        # Store data
        self.data_history.append({
            'step': step,
            'time': time,
            'eta_m': self.eta_m.copy(),
            'eta_t': self.eta_t.copy(),
            'energy': self.solver.compute_energy(self.eta_m, self.eta_t)
        })
        
        # Create snapshot (less frequent for performance)
        if step % (self.params.output_interval * 5) == 0 or step == self.params.total_steps:
            fig = self.visualization.create_snapshot(
                self.eta_m, self.eta_t, self.params, time, self.colorscale
            )
            snapshot_path = os.path.join(self.output_dir, f'snapshot_{step:06d}.png')
            fig.savefig(snapshot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.snapshot_paths.append(snapshot_path)
    
    def get_results(self):
        """Get simulation results"""
        return {
            'data_history': self.data_history,
            'energy_history': self.solver.energy_history,
            'time_history': self.solver.time_history,
            'snapshots': self.snapshot_paths,
            'params': self.params,
            'performance': self._get_performance_stats()
        }
    
    def _get_performance_stats(self):
        """Get performance statistics"""
        if not self.step_times:
            return {}
        
        times = np.array(self.step_times)
        total_time = np.sum(times)
        
        return {
            'total_time': total_time,
            'avg_step_time': np.mean(times),
            'steps_per_second': len(times) / total_time if total_time > 0 else 0,
            'solver_type': self.params.solver_type,
            'numba_available': NUMBA_AVAILABLE
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.output_dir)
        except:
            pass

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    # Custom CSS
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Fast Nanotwin Phase Field Simulation</h1>', unsafe_allow_html=True)
    
    # Performance info
    col1, col2, col3 = st.columns(3)
    with col1:
        if NUMBA_AVAILABLE:
            st.success(f"✅ Numba JIT: Available")
        else:
            st.warning("⚠️ Numba: Not installed")
    with col2:
        st.info(f"NumPy: v{np.__version__}")
    with col3:
        import sys
        st.info(f"Python: {sys.version.split()[0]}")
    
    # Introduction
    with st.expander("🚀 About This Simulation", expanded=True):
        st.markdown("""
        ### **High-Performance Phase Field Simulation**
        
        This application simulates nanotwin detwinning in FCC metals using optimized algorithms:
        
        **Available Solvers:**
        1. **Standard NumPy** - Reliable, works everywhere
        2. **Numba JIT** - 10-100x faster (if Numba is installed)
        3. **FFT Spectral** - Fast for large grids
        
        **Performance Features:**
        - Adaptive time stepping for stability
        - Optimized memory usage
        - Real-time progress tracking
        - Multiple visualization options
        """)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Simulation Parameters")
    
    # Material selection
    material = st.sidebar.selectbox(
        "Material",
        ["Cu", "Al", "Ni", "Ag", "Au", "Pd"],
        index=0
    )
    
    # Solver selection
    st.sidebar.markdown("### ⚡ Solver Type")
    
    solver_options = []
    if NUMBA_AVAILABLE:
        solver_options.append(("numba", "Numba JIT (Fastest)"))
    solver_options.append(("fft", "FFT Spectral (Large grids)"))
    solver_options.append(("numpy", "Standard NumPy (Reliable)"))
    
    solver_type = st.sidebar.selectbox(
        "Select Solver",
        options=[opt[0] for opt in solver_options],
        format_func=lambda x: dict(solver_options)[x],
        index=0
    )
    
    # Performance options
    with st.sidebar.expander("🎯 Performance"):
        adaptive_dt = st.checkbox("Adaptive Time Step", value=True)
        vectorized = st.checkbox("Vectorized Operations", value=True)
    
    # Colorscale
    colorscale_options = {
        'Viridis': 'viridis',
        'Plasma': 'plasma',
        'Inferno': 'inferno',
        'Magma': 'magma',
        'Hot': 'hot',
        'Jet': 'jet',
        'Rainbow': 'rainbow'
    }
    
    colorscale = st.sidebar.selectbox("Colorscale", list(colorscale_options.keys()), index=0)
    
    # Geometry
    st.sidebar.markdown("### 📏 Geometry")
    
    twin_width_nm = st.sidebar.slider("Twin Width (nm)", 1.0, 50.0, 10.0)
    Lx_nm = st.sidebar.slider("Domain Width (nm)", 20.0, 200.0, 100.0)
    Ly_nm = st.sidebar.slider("Domain Height (nm)", 20.0, 200.0, 100.0)
    
    # Grid resolution
    if solver_type == 'fft':
        grid_sizes = [64, 128, 256]
        default_idx = 1
    else:
        grid_sizes = [64, 128, 256, 512]
        default_idx = 1
    
    Nx = st.sidebar.selectbox("Grid Points X", grid_sizes, index=default_idx)
    Ny = st.sidebar.selectbox("Grid Points Y", grid_sizes, index=default_idx)
    
    # Physics
    st.sidebar.markdown("### 🔬 Physics")
    
    temperature = st.sidebar.slider("Temperature (K)", 100, 1000, 500)
    t_max_ns = st.sidebar.slider("Simulation Time (ns)", 0.1, 5.0, 1.0)
    output_interval = st.sidebar.slider("Output Interval", 10, 200, 50)
    
    # Advanced
    with st.sidebar.expander("🔧 Advanced"):
        random_seed = st.number_input("Random Seed", 0, 1000, 42)
        np.random.seed(random_seed)
    
    # Run button
    st.sidebar.markdown("---")
    run_button = st.sidebar.button(
        "🚀 Run Simulation",
        type="primary",
        use_container_width=True
    )
    
    # Convert units
    twin_width = twin_width_nm * 1e-9
    Lx, Ly = Lx_nm * 1e-9, Ly_nm * 1e-9
    t_max = t_max_ns * 1e-9
    
    # Main content
    if run_button:
        # Display configuration
        st.markdown("### 🔬 Simulation Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Material:** {material}")
            st.write(f"**Solver:** {solver_type}")
            st.write(f"**Domain:** {Lx_nm:.1f} nm × {Ly_nm:.1f} nm")
            st.write(f"**Grid:** {Nx} × {Ny}")
        
        with col2:
            st.write(f"**Twin Width:** {twin_width_nm:.1f} nm")
            st.write(f"**Temperature:** {temperature} K")
            st.write(f"**Total Time:** {t_max_ns:.2f} ns")
            st.write(f"**Output Interval:** {output_interval}")
        
        # Performance estimate
        st.markdown("### 📊 Performance Estimate")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if solver_type == 'numba':
                st.markdown('<div class="performance-badge badge-fastest">FASTEST: Numba JIT</div>', unsafe_allow_html=True)
            elif solver_type == 'fft':
                st.markdown('<div class="performance-badge badge-faster">FASTER: FFT Spectral</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="performance-badge badge-fast">FAST: Standard NumPy</div>', unsafe_allow_html=True)
        
        with col2:
            grid_size = Nx * Ny
            st.metric("Grid Size", f"{grid_size:,}")
        
        with col3:
            est_steps = int(t_max / 1e-12)
            st.metric("Est. Steps", f"{est_steps:,}")
        
        # Initialize simulation
        with st.spinner("🚀 Initializing simulation..."):
            try:
                sim = FastNanotwinSimulation(
                    material, twin_width, temperature,
                    Lx, Ly, Nx, Ny, t_max,
                    output_interval, colorscale_options[colorscale],
                    solver_type=solver_type
                )
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                perf_text = st.empty()
                
                # Run simulation
                with st.spinner("⏳ Running simulation..."):
                    import time
                    total_start = time.time()
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                    
                    def update_status(step, total_steps, t, energy, steps_per_sec):
                        elapsed = time.time() - total_start
                        eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                        
                        status_text.markdown(f"""
                        **Progress:** {step:,} / {total_steps:,} steps ({step/total_steps*100:.1f}%)
                        **Time:** {t*1e9:.3f} ns
                        **Energy:** {energy:.2e} J
                        """)
                        
                        perf_text.markdown(f"""
                        **Performance:**
                        - Elapsed: {elapsed:.1f}s
                        - Steps/sec: {steps_per_sec:.0f}
                        - ETA: {eta:.1f}s
                        """)
                    
                    # Run simulation
                    data_history = sim.run(update_progress, update_status)
                    
                    total_time = time.time() - total_start
                
                # Success
                results = sim.get_results()
                perf_stats = results['performance']
                
                st.success(f"""
                ✅ Simulation completed in {total_time:.1f}s!
                - Processed {len(data_history)} time steps
                - Average step time: {perf_stats.get('avg_step_time', 0)*1000:.2f}ms
                - Steps per second: {perf_stats.get('steps_per_second', 0):.0f}
                """)
                
                # Results tabs
                tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Analysis", "💾 Export"])
                
                with tab1:
                    # Interactive plot
                    if data_history:
                        st.markdown("#### Phase Field Evolution")
                        plotly_fig = VisualizationTools.create_interactive_plot(
                            data_history, sim.params, colorscale_options[colorscale]
                        )
                        if plotly_fig:
                            st.plotly_chart(plotly_fig, use_container_width=True)
                        
                        # Snapshots
                        st.markdown("#### Snapshots")
                        if sim.snapshot_paths:
                            cols = st.columns(3)
                            for i, path in enumerate(sim.snapshot_paths[-3:]):  # Last 3 snapshots
                                with cols[i % 3]:
                                    st.image(path, use_column_width=True)
                
                with tab2:
                    # Energy analysis
                    if results['energy_history']:
                        st.markdown("#### Energy Evolution")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[t*1e9 for t in results['time_history']],
                            y=results['energy_history'],
                            mode='lines',
                            name='Free Energy'
                        ))
                        
                        fig.update_layout(
                            title='Free Energy vs Time',
                            xaxis_title='Time (ns)',
                            yaxis_title='Energy (J)',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Energy", f"{results['energy_history'][0]:.2e} J")
                        with col2:
                            st.metric("Final Energy", f"{results['energy_history'][-1]:.2e} J")
                        with col3:
                            if results['energy_history'][0] != 0:
                                change = (results['energy_history'][-1] - results['energy_history'][0]) / results['energy_history'][0] * 100
                                st.metric("% Change", f"{change:.1f}%")
                
                with tab3:
                    # Export
                    st.markdown("#### Export Data")
                    
                    # Create summary
                    summary = {
                        'material': material,
                        'solver': solver_type,
                        'parameters': {
                            'twin_width_nm': twin_width_nm,
                            'temperature': temperature,
                            'domain_nm': [Lx_nm, Ly_nm],
                            'grid': [Nx, Ny],
                            'total_time_ns': t_max_ns
                        },
                        'performance': perf_stats,
                        'energy_history': results['energy_history'],
                        'time_history': results['time_history']
                    }
                    
                    # JSON export
                    json_str = json.dumps(summary, indent=2)
                    st.download_button(
                        label="📄 Download Summary (JSON)",
                        data=json_str,
                        file_name=f"nanotwin_{material}_{solver_type}.json",
                        mime="application/json"
                    )
                    
                    # CSV export for energy data
                    if results['energy_history']:
                        import io
                        csv_buffer = io.StringIO()
                        csv_buffer.write("time_ns,energy_J\n")
                        for t, e in zip(results['time_history'], results['energy_history']):
                            csv_buffer.write(f"{t*1e9},{e}\n")
                        
                        st.download_button(
                            label="📈 Download Energy Data (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"nanotwin_{material}_energy.csv",
                            mime="text/csv"
                        )
                
                # Cleanup
                sim.cleanup()
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Fast Nanotwin Simulator!
        
        This tool simulates the detwinning of nanotwins in FCC metals using optimized phase field methods.
        
        **To get started:**
        1. Select a material and solver type in the sidebar
        2. Configure the simulation parameters
        3. Click "Run Simulation"
        
        **Available Solvers:**
        - **Numba JIT**: Fastest option (requires Numba installation)
        - **FFT Spectral**: Best for large grids
        - **Standard NumPy**: Works everywhere
        
        **Performance Tips:**
        - Use smaller grids for faster simulations
        - Enable adaptive time stepping for stability
        - Reduce output frequency for speed
        """)
        
        # Quick examples
        st.markdown("#### Quick Examples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Copper Nanotwin (Fast)**
            - Material: Cu
            - Grid: 128×128
            - Time: 1 ns
            - Solver: Numba
            """)
        
        with col2:
            st.markdown("""
            **Aluminum Study (Accurate)**
            - Material: Al
            - Grid: 256×256
            - Time: 2 ns
            - Solver: FFT
            """)

if __name__ == "__main__":
    main()
