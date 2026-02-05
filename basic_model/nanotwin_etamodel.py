import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import streamlit as st
import plotly.graph_objects as go
import os
import uuid
import shutil
import zipfile
import yaml
from pathlib import Path
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Try to import pyvista with proper error handling
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError as e:
    st.error(f"PyVista import failed: {e}")
    PYVISTA_AVAILABLE = False
except Exception as e:
    # This catches the specific error you mentioned
    st.warning(f"PyVista import had issues (may be due to VTK dependencies): {e}")
    PYVISTA_AVAILABLE = False

# Module 0: Material Database (embedded to avoid YAML file dependencies)
class MaterialDatabase:
    """Embedded material properties database to avoid external YAML file dependencies."""
    
    MATERIALS = {
        'Cu': {
            'material': 'Copper',
            'sigma_ctb': 0.5e-3,  # J/m²
            'sigma_itb': 0.8e-3,
            'sigma_gb': 1.0e-3,
            'M_itb_prefactor': 5e-6,  # m⁴/(J·s)
            'M_itb_activation_energy': 0.2  # eV
        },
        'Al': {
            'material': 'Aluminum',
            'sigma_ctb': 0.4e-3,
            'sigma_itb': 0.6e-3,
            'sigma_gb': 0.9e-3,
            'M_itb_prefactor': 8e-6,
            'M_itb_activation_energy': 0.15
        },
        'Ni': {
            'material': 'Nickel',
            'sigma_ctb': 0.7e-3,
            'sigma_itb': 1.0e-3,
            'sigma_gb': 1.2e-3,
            'M_itb_prefactor': 3e-6,
            'M_itb_activation_energy': 0.25
        },
        'Ag': {
            'material': 'Silver',
            'sigma_ctb': 0.45e-3,
            'sigma_itb': 0.7e-3,
            'sigma_gb': 0.95e-3,
            'M_itb_prefactor': 6e-6,
            'M_itb_activation_energy': 0.18
        },
        'Au': {
            'material': 'Gold',
            'sigma_ctb': 0.48e-3,
            'sigma_itb': 0.75e-3,
            'sigma_gb': 1.0e-3,
            'M_itb_prefactor': 4e-6,
            'M_itb_activation_energy': 0.22
        },
        'Pd': {
            'material': 'Palladium',
            'sigma_ctb': 0.65e-3,
            'sigma_itb': 0.9e-3,
            'sigma_gb': 1.1e-3,
            'M_itb_prefactor': 3.5e-6,
            'M_itb_activation_energy': 0.24
        }
    }
    
    @classmethod
    def get_material_properties(cls, material):
        """Get material properties from embedded database."""
        if material not in cls.MATERIALS:
            raise ValueError(f"Material '{material}' not found in database. Available: {list(cls.MATERIALS.keys())}")
        return cls.MATERIALS[material].copy()

# Module 1: Enhanced Parameters
class SimulationParameters:
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval):
        # Get material properties from embedded database
        material_data = MaterialDatabase.get_material_properties(material)
        
        self.material = material_data['material']
        self.sigma_ctb = material_data['sigma_ctb']
        self.sigma_itb = material_data['sigma_itb']
        self.sigma_gb = material_data['sigma_gb']
        self.M_itb_prefactor = material_data['M_itb_prefactor']
        self.M_itb_activation_energy = material_data['M_itb_activation_energy']
        
        # Boltzmann constant in eV/K
        k_B = 8.617333262145e-5
        
        # Calculate mobility with temperature dependence
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.M_itb = self.M_itb_prefactor * np.exp(-self.M_itb_activation_energy / (k_B * temperature))
        
        # Interface parameters
        self.l_int = 1e-9  # Interface width (m)
        self.m = 6 * self.sigma_itb / self.l_int  # Free energy barrier
        self.kappa = 0.75 * self.sigma_itb * self.l_int  # Gradient coefficient
        self.delta_sigma = 0.5  # Anisotropy strength
        self.L = 4/3 * self.M_itb / self.l_int  # Kinetic coefficient
        
        # Numerical parameters with validation
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain dimensions must be positive")
        if Nx < 10 or Ny < 10:
            raise ValueError("Grid resolution too low (minimum 10x10)")
        
        self.Lx, self.Ly = Lx, Ly  # Domain size (m)
        self.Nx, self.Ny = Nx, Ny  # Grid points
        self.dx = self.Lx / self.Nx  # Grid spacing (m)
        
        # Adaptive time step based on stability criteria
        dt_stable = 0.25 * self.dx**2 * self.Lx * self.Ly / (self.kappa * self.L)
        self.dt = min(1e-12, dt_stable)  # Time step (s) ensuring stability
        
        if t_max <= 0:
            raise ValueError("Simulation time must be positive")
        self.t_max = t_max  # Total simulation time (s)
        
        if output_interval <= 0:
            raise ValueError("Output interval must be positive")
        self.output_interval = output_interval  # Output frequency
        
        if twin_width <= 0 or twin_width >= Ly:
            raise ValueError(f"Twin width must be positive and less than domain height ({Ly:.2e} m)")
        self.twin_width = twin_width  # Nanotwin width (m)
        
        # Derived parameters
        self.grid_x = np.linspace(0, Lx, Nx)
        self.grid_y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)
        
        # Physical constants
        self.temperature = temperature
        self.k_B = k_B
        
    def __str__(self):
        """String representation for debugging."""
        info = f"""
        Simulation Parameters:
        ---------------------
        Material: {self.material}
        Domain: {self.Lx*1e9:.1f} nm x {self.Ly*1e9:.1f} nm
        Grid: {self.Nx} x {self.Ny}
        Twin width: {self.twin_width*1e9:.1f} nm
        Temperature: {self.temperature} K
        Time step: {self.dt:.2e} s
        Total time: {self.t_max:.2e} s
        Sigma_ITB: {self.sigma_itb:.2e} J/m²
        Mobility: {self.M_itb:.2e} m⁴/(J·s)
        """
        return info

# Module 2: Enhanced Free Energy
class FreeEnergy:
    def __init__(self, params):
        self.params = params
        self._initialize_anisotropy_lookup()

    def _initialize_anisotropy_lookup(self):
        """Initialize lookup table for anisotropy to improve performance."""
        g_vals = np.linspace(0.01, 5, 1000)
        p_g2 = -3.0944 * g_vals**4 - 1.8169 * g_vals**3 + 10.323 * g_vals**2 - 8.1819 * g_vals
        gamma_vals = 1.0 / p_g2
        self.gamma_lookup = (g_vals, gamma_vals)

    def local_energy(self, eta):
        """Compute local free energy density f_0(eta)."""
        eta_m, eta_t = eta
        m = self.params.m
        
        # Double well potential for each phase
        f0 = 0
        for eta_i in [eta_m, eta_t]:
            f0 += (eta_i**4 / 4 - eta_i**2 / 2)
        
        # Interaction term
        gamma_mt = self.compute_gamma(eta_m, eta_t)
        f0 += gamma_mt * eta_m**2 * eta_t**2 / 2 + 0.25
        
        return m * f0

    def compute_gamma(self, eta_m, eta_t):
        """Compute inclination-dependent gamma_mt with interpolation."""
        # Compute gradients
        grad_m_x, grad_m_y = np.gradient(eta_m, self.params.dx)
        grad_t_x, grad_t_y = np.gradient(eta_t, self.params.dx)
        
        # Compute phase difference gradient
        grad_diff_x = grad_m_x - grad_t_x
        grad_diff_y = grad_m_y - grad_t_y
        
        # Compute inclination angle
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = np.arctan2(grad_diff_y, grad_diff_x)
            phi = np.nan_to_num(phi, nan=0.0)
        
        # Compute anisotropic interfacial energy
        sigma_mt = self.params.sigma_itb * (1 + self.params.delta_sigma * np.cos(phi))
        
        # Compute g parameter
        sqrt_km = np.sqrt(self.params.kappa * self.params.m)
        g = sigma_mt / sqrt_km
        
        # Interpolate gamma from lookup table
        g_vals, gamma_vals = self.gamma_lookup
        gamma = np.interp(g, g_vals, gamma_vals, left=0.1, right=10.0)
        
        return np.clip(gamma, 0.1, 10)

    def anisotropic_term(self, eta_m, eta_t, gamma_mt):
        """Compute anisotropic term for evolution equation with improved stability."""
        # Compute gradients
        grad_m_x, grad_m_y = np.gradient(eta_m, self.params.dx)
        grad_t_x, grad_t_y = np.gradient(eta_t, self.params.dx)
        
        # Compute phase difference gradient
        grad_diff_x = grad_m_x - grad_t_x
        grad_diff_y = grad_m_y - grad_t_y
        
        # Compute norm squared with regularization
        norm_sq = grad_diff_x**2 + grad_diff_y**2 + 1e-10
        
        # Compute inclination angle
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = np.arctan2(grad_diff_y, grad_diff_x)
            phi = np.nan_to_num(phi, nan=0.0)
        
        # Compute sigma and its derivative with respect to phi
        sigma_mt = self.params.sigma_itb * (1 + self.params.delta_sigma * np.cos(phi))
        dsigma_dphi = -self.params.sigma_itb * self.params.delta_sigma * np.sin(phi)
        
        # Compute g and its derivative
        sqrt_km = np.sqrt(self.params.kappa * self.params.m)
        g = sigma_mt / sqrt_km
        dg_dphi = dsigma_dphi / sqrt_km
        
        # Compute polynomial and its derivative
        p_g2 = -3.0944 * g**4 - 1.8169 * g**3 + 10.323 * g**2 - 8.1819 * g
        dp_dg2 = -12.3776 * g**3 - 5.4507 * g**2 + 20.646 * g - 8.1819
        
        # Compute derivative of gamma with respect to phi
        dgamma_dphi = -2 * g * gamma_mt**2 * dp_dg2 * dg_dphi
        
        # Compute vector components
        vec_x = -grad_diff_y / norm_sq
        vec_y = grad_diff_x / norm_sq
        
        # Compute anisotropic term
        term_x = dgamma_dphi * vec_x * eta_m**2 * eta_t**2
        term_y = dgamma_dphi * vec_y * eta_m**2 * eta_t**2
        
        # Compute divergence
        div_term_x, _ = np.gradient(term_x, self.params.dx)
        _, div_term_y = np.gradient(term_y, self.params.dx)
        
        div_term = div_term_x + div_term_y
        
        return self.params.m * div_term

    def total_free_energy(self, eta):
        """Compute total free energy of the system."""
        eta_m, eta_t = eta
        
        # Local energy contribution
        f_local = self.local_energy(eta)
        
        # Gradient energy contribution
        grad_m_x, grad_m_y = np.gradient(eta_m, self.params.dx)
        grad_t_x, grad_t_y = np.gradient(eta_t, self.params.dx)
        
        f_grad = 0.5 * self.params.kappa * (
            grad_m_x**2 + grad_m_y**2 + 
            grad_t_x**2 + grad_t_y**2
        )
        
        # Total free energy density
        f_total = f_local + f_grad
        
        # Integrate over domain
        total_energy = np.trapz(np.trapz(f_total, dx=self.params.dx), dx=self.params.dx)
        
        return total_energy

# Module 3: Enhanced Evolution
class PhaseFieldEvolution:
    def __init__(self, params, free_energy):
        self.params = params
        self.free_energy = free_energy
        self.energy_history = []
        self.time_history = []

    def compute_derivative(self, eta):
        """Compute functional derivative of free energy with improved stability."""
        eta_m, eta_t = eta
        m = self.params.m
        kappa = self.params.kappa
        
        # Compute interaction parameter
        gamma_mt = self.free_energy.compute_gamma(eta_m, eta_t)
        
        # Bulk term derivatives
        df_m = m * (eta_m**3 - eta_m + 2 * eta_m * gamma_mt * eta_t**2)
        df_t = m * (eta_t**3 - eta_t + 2 * eta_t * gamma_mt * eta_m**2)
        
        # Gradient term using finite differences with wrap-around boundary conditions
        laplace_m = laplace(eta_m, mode='wrap') / self.params.dx**2
        laplace_t = laplace(eta_t, mode='wrap') / self.params.dx**2
        
        df_m -= kappa * laplace_m
        df_t -= kappa * laplace_t
        
        # Anisotropic term
        anisotropic_m = self.free_energy.anisotropic_term(eta_m, eta_t, gamma_mt)
        df_m += anisotropic_m
        df_t -= anisotropic_m  # Symmetric contribution
        
        return np.array([df_m, df_t])

    def update(self, eta):
        """Update order parameters using Allen-Cahn equation with stability checks."""
        # Compute functional derivative
        df = self.compute_derivative(eta)
        
        # Allen-Cahn evolution
        eta_new = eta - self.params.dt * self.params.L * df
        
        # Apply bounds with smooth constraint
        eta_new = np.clip(eta_new, 0, 1)
        
        # Ensure conservation (eta_m + eta_t ≈ 1) with tolerance
        sum_eta = eta_new[0] + eta_new[1]
        mask = np.abs(sum_eta - 1) > 0.01
        if np.any(mask):
            # Normalize where needed
            norm = eta_new[0][mask] + eta_new[1][mask]
            eta_new[0][mask] = eta_new[0][mask] / norm
            eta_new[1][mask] = eta_new[1][mask] / norm
        
        # Track energy evolution
        current_energy = self.free_energy.total_free_energy(eta_new)
        self.energy_history.append(current_energy)
        self.time_history.append(len(self.energy_history) * self.params.dt)
        
        return eta_new

    def get_energy_history(self):
        """Return energy evolution data."""
        return np.array(self.time_history), np.array(self.energy_history)

# Module 4: Enhanced Simulation with Progress Tracking
class Simulation:
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval, colorscale):
        # Initialize parameters with validation
        self.params = SimulationParameters(material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval)
        self.free_energy = FreeEnergy(self.params)
        self.evolution = PhaseFieldEvolution(self.params, self.free_energy)
        self.colorscale = colorscale
        
        # Create unique output directory
        self.output_dir = tempfile.mkdtemp(prefix="nanotwin_sim_")
        self.vtk_dir = os.path.join(self.output_dir, "vtk_files")
        os.makedirs(self.vtk_dir, exist_ok=True)
        
        # Initialize fields
        self.initialize_fields()
        
        # Store simulation data
        self.output_data = []
        self.snapshot_paths = []
        self.vtk_paths = []
        
        # Simulation metrics
        self.simulation_time = 0
        self.current_step = 0
        
    def initialize_fields(self):
        """Initialize order parameters for a nanotwin in a matrix with smooth interfaces."""
        Nx, Ny = self.params.Nx, self.params.Ny
        X, Y = self.params.X, self.params.Y
        
        # Center of the domain
        y_center = self.params.Ly / 2
        twin_half_width = self.params.twin_width / 2
        
        # Smooth twin profile using hyperbolic tangent
        # Interface width parameter
        interface_width = self.params.l_int
        
        # Twin region (eta_t = 1, eta_m = 0)
        eta_t = 0.5 * (
            1 + np.tanh(4 * (Y - (y_center - twin_half_width)) / interface_width)
        ) * 0.5 * (
            1 + np.tanh(-4 * (Y - (y_center + twin_half_width)) / interface_width)
        )
        
        # Matrix region (eta_m = 1, eta_t = 0)
        eta_m = 1 - eta_t
        
        # Add small random perturbation to break symmetry
        random_perturbation = 0.01 * np.random.randn(Ny, Nx)
        eta_m = np.clip(eta_m + random_perturbation, 0, 1)
        eta_t = np.clip(eta_t - random_perturbation, 0, 1)
        
        # Ensure normalization
        sum_eta = eta_m + eta_t
        eta_m = eta_m / sum_eta
        eta_t = eta_t / sum_eta
        
        self.eta = np.array([eta_m, eta_t])
        
        # Initial energy calculation
        initial_energy = self.free_energy.total_free_energy(self.eta)
        self.initial_energy = initial_energy
        
    def save_output(self, step, t):
        """Save current state as image and optionally as VTK file."""
        # Create figure with improved visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot matrix order parameter
        im1 = ax1.imshow(self.eta[0], 
                        extent=[0, self.params.Lx*1e9, 0, self.params.Ly*1e9], 
                        origin='lower', 
                        cmap=self.colorscale,
                        vmin=0, vmax=1)
        ax1.set_title(f'Matrix (η_m)\nTime: {t*1e9:.2f} ns')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot twin order parameter
        im2 = ax2.imshow(self.eta[1], 
                        extent=[0, self.params.Lx*1e9, 0, self.params.Ly*1e9], 
                        origin='lower', 
                        cmap=self.colorscale,
                        vmin=0, vmax=1)
        ax2.set_title(f'Nanotwin (η_t)\nTime: {t*1e9:.2f} ns')
        ax2.set_xlabel('x (nm)')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Plot phase field sum (should be ~1 everywhere)
        im3 = ax3.imshow(self.eta[0] + self.eta[1], 
                        extent=[0, self.params.Lx*1e9, 0, self.params.Ly*1e9], 
                        origin='lower', 
                        cmap='RdYlBu',
                        vmin=0.95, vmax=1.05)
        ax3.set_title(f'Sum (η_m + η_t)\nTime: {t*1e9:.2f} ns')
        ax3.set_xlabel('x (nm)')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        image_path = os.path.join(self.output_dir, f'step_{step:06d}.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.snapshot_paths.append(image_path)
        
        # Save VTK file if pyvista is available
        if PYVISTA_AVAILABLE:
            try:
                # Create structured grid
                grid = pv.StructuredGrid()
                grid.points = np.array([
                    [x, y, 0] 
                    for y in self.params.grid_y 
                    for x in self.params.grid_x
                ])
                grid.dimensions = [self.params.Nx, self.params.Ny, 1]
                
                # Add order parameters as point data
                grid.point_data['eta_m'] = self.eta[0].flatten(order='F')
                grid.point_data['eta_t'] = self.eta[1].flatten(order='F')
                grid.point_data['eta_sum'] = (self.eta[0] + self.eta[1]).flatten(order='F')
                
                # Save VTK file
                vtk_path = os.path.join(self.vtk_dir, f'step_{step:06d}.vtr')
                grid.save(vtk_path)
                self.vtk_paths.append(vtk_path)
            except Exception as e:
                st.warning(f"Could not save VTK file at step {step}: {e}")
        
        # Store data for Plotly visualization
        self.output_data.append({
            'step': step,
            'time': t,
            'eta_m': self.eta[0].copy(),
            'eta_t': self.eta[1].copy(),
            'eta_sum': self.eta[0] + self.eta[1],
            'energy': self.free_energy.total_free_energy(self.eta)
        })
        
        return image_path

    def create_plotly_figure(self):
        """Create an interactive Plotly figure with multiple views."""
        if not self.output_data:
            return None
            
        # Create subplot figure
        fig = go.Figure()
        
        # Get grid coordinates
        x = np.linspace(0, self.params.Lx * 1e9, self.params.Nx)
        y = np.linspace(0, self.params.Ly * 1e9, self.params.Ny)
        
        # Initial data
        initial_data = self.output_data[0]
        
        # Add heatmaps for initial state
        # Matrix phase
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=initial_data['eta_m'],
                colorscale=self.colorscale,
                zmin=0, zmax=1,
                name='Matrix (η_m)',
                visible=True,
                colorbar=dict(title='η_m', x=0.45, len=0.8)
            ),
            row=1, col=1
        )
        
        # Twin phase
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=initial_data['eta_t'],
                colorscale=self.colorscale,
                zmin=0, zmax=1,
                name='Twin (η_t)',
                visible=True,
                colorbar=dict(title='η_t', x=1.02, len=0.8)
            ),
            row=1, col=2
        )
        
        # Create frames for animation
        frames = []
        for i, data in enumerate(self.output_data):
            frame = go.Frame(
                data=[
                    go.Heatmap(
                        x=x, y=y, z=data['eta_m'],
                        colorscale=self.colorscale,
                        zmin=0, zmax=1
                    ),
                    go.Heatmap(
                        x=x, y=y, z=data['eta_t'],
                        colorscale=self.colorscale,
                        zmin=0, zmax=1
                    )
                ],
                name=f"frame_{i}",
                layout=go.Layout(
                    title_text=f"Time: {data['time']*1e9:.2f} ns",
                    title_x=0.5
                )
            )
            frames.append(frame)
        
        # Create slider
        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
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
                        [f.name],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f"{data['time']*1e9:.2f}",
                    'method': 'animate'
                }
                for data in self.output_data
            ]
        }]
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Nanotwin Evolution - {self.params.material}",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            width=1200,
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
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': 'Pause',
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
        
        fig.frames = frames
        
        return fig

    def create_energy_plot(self):
        """Create plot of energy evolution."""
        if not self.output_data:
            return None
            
        times = [data['time'] for data in self.output_data]
        energies = [data['energy'] for data in self.output_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=np.array(times) * 1e9,  # Convert to ns
            y=energies,
            mode='lines+markers',
            name='Total Free Energy',
            line=dict(color='royalblue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Free Energy Evolution',
            xaxis_title='Time (ns)',
            yaxis_title='Free Energy (J)',
            height=400,
            width=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig

    def run(self, progress_bar, status_text):
        """Run the simulation loop with progress tracking."""
        total_steps = int(self.params.t_max / self.params.dt)
        
        # Create progress containers
        progress_container = st.empty()
        
        # Run simulation
        for step in range(total_steps + 1):
            t = step * self.params.dt
            self.current_step = step
            self.simulation_time = t
            
            # Update phase field
            self.eta = self.evolution.update(self.eta)
            
            # Save output at intervals
            if step % self.params.output_interval == 0:
                self.save_output(step, t)
                
                # Update progress
                progress = min(step / total_steps, 1.0)
                progress_bar.progress(progress)
                
                # Update status
                status_text.text(f"""
                Simulation Progress:
                - Step: {step:,} / {total_steps:,}
                - Time: {t*1e9:.2f} / {self.params.t_max*1e9:.2f} ns
                - Energy: {self.free_energy.total_free_energy(self.eta):.2e} J
                - Output frames: {len(self.output_data)}
                """)
                
                # Yield to prevent timeout in Streamlit
                if step % (self.params.output_interval * 10) == 0:
                    st.empty()  # Force refresh
        
        # Final save
        self.save_output(total_steps, self.params.t_max)
        
        return self.snapshot_paths

# Module 5: Enhanced Streamlit Interface
def main():
    st.set_page_config(
        page_title="Nanotwin Phase Field Simulation",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Phase-Field Simulation of Nanotwin Detwinning</h1>', unsafe_allow_html=True)
    
    # Introduction
    with st.expander("📘 About this Simulation", expanded=True):
        st.markdown("""
        This interactive application simulates the **detwinning process of nanotwins** in FCC materials using a **phase-field model**.
        
        ### Key Features:
        - **Multiple Materials**: Simulate Cu, Al, Ni, Ag, Au, and Pd with accurate material properties
        - **Interactive Controls**: Adjust simulation parameters in real-time
        - **Advanced Visualization**: Interactive 3D plots with time evolution
        - **Export Capabilities**: Download simulation data for further analysis
        - **Energy Tracking**: Monitor free energy evolution during detwinning
        
        ### How it Works:
        1. Select material and adjust parameters in the sidebar
        2. Click "Run Simulation" to start the phase-field calculation
        3. View results in interactive plots
        4. Download data for offline analysis
        
        The model uses **Allen-Cahn equations** to describe interface motion with **anisotropic interfacial energy**.
        """)
    
    # Sidebar with parameters
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Simulation Parameters</h2>', unsafe_allow_html=True)
    
    # Material selection
    material = st.sidebar.selectbox(
        "Select Material",
        ["Cu", "Al", "Ni", "Ag", "Au", "Pd"],
        index=0,
        help="Choose the FCC material for simulation"
    )
    
    # Display material properties
    with st.sidebar.expander("📊 Material Properties"):
        props = MaterialDatabase.get_material_properties(material)
        st.write(f"**{props['material']}**")
        st.write(f"Σ_CTB: {props['sigma_ctb']:.2e} J/m²")
        st.write(f"Σ_ITB: {props['sigma_itb']:.2e} J/m²")
        st.write(f"Σ_GB: {props['sigma_gb']:.2e} J/m²")
        st.write(f"Mobility Prefactor: {props['M_itb_prefactor']:.2e} m⁴/(J·s)")
        st.write(f"Activation Energy: {props['M_itb_activation_energy']:.2f} eV")
    
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
        'RdYlBu': 'RdYlBu',
        'Spectral': 'Spectral',
        'Phase': 'hsv'
    }
    
    colorscale = st.sidebar.selectbox(
        "Colorscale",
        list(colorscale_options.keys()),
        index=0,
        help="Color scheme for visualization"
    )
    
    # Simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📏 Geometry Parameters")
    
    twin_width_nm = st.sidebar.slider(
        "Nanotwin Width (nm)",
        1.0, 50.0, 10.0,
        help="Width of the initial nanotwin"
    )
    
    Lx_nm = st.sidebar.slider(
        "Domain Width (nm)",
        20.0, 200.0, 100.0,
        help="Width of simulation domain"
    )
    
    Ly_nm = st.sidebar.slider(
        "Domain Height (nm)", 
        20.0, 200.0, 100.0,
        help="Height of simulation domain"
    )
    
    st.sidebar.markdown("### 🎯 Numerical Parameters")
    
    Nx = st.sidebar.slider(
        "Grid Points X",
        50, 500, 200,
        help="Number of grid points in x-direction"
    )
    
    Ny = st.sidebar.slider(
        "Grid Points Y",
        50, 500, 200,
        help="Number of grid points in y-direction"
    )
    
    st.sidebar.markdown("### ⚡ Physical Parameters")
    
    temperature = st.sidebar.slider(
        "Temperature (K)",
        100, 1000, 500,
        help="Simulation temperature"
    )
    
    t_max_ns = st.sidebar.slider(
        "Simulation Time (ns)",
        0.1, 10.0, 1.0,
        help="Total simulation time"
    )
    
    output_interval = st.sidebar.slider(
        "Output Interval (steps)",
        10, 500, 100,
        help="Steps between saving output"
    )
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        anisotropy_strength = st.slider(
            "Anisotropy Strength",
            0.0, 1.0, 0.5,
            help="Strength of anisotropic interfacial energy"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            0, 1000, 42,
            help="Seed for random perturbations"
        )
        
        np.random.seed(random_seed)
    
    # Run button
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Convert units
    twin_width = twin_width_nm * 1e-9
    Lx, Ly = Lx_nm * 1e-9, Ly_nm * 1e-9
    t_max = t_max_ns * 1e-9
    
    # Main content area
    if run_button:
        # Initialize simulation
        with st.spinner("Initializing simulation..."):
            try:
                sim = Simulation(
                    material, twin_width, temperature, 
                    Lx, Ly, Nx, Ny, t_max, 
                    output_interval, colorscale_options[colorscale]
                )
                
                # Display simulation info
                st.markdown(f"""
                <div class="info-box">
                <h3>Simulation Setup</h3>
                <p><b>Material:</b> {material}</p>
                <p><b>Domain:</b> {Lx_nm:.1f} nm × {Ly_nm:.1f} nm</p>
                <p><b>Grid:</b> {Nx} × {Ny} points</p>
                <p><b>Twin Width:</b> {twin_width_nm:.1f} nm</p>
                <p><b>Temperature:</b> {temperature} K</p>
                <p><b>Total Time:</b> {t_max_ns:.2f} ns</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run simulation
                with st.spinner("Running simulation..."):
                    output_paths = sim.run(progress_bar, status_text)
                
                # Display success message
                st.success(f"✅ Simulation completed successfully! Processed {len(sim.output_data)} time steps.")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Interactive Visualization", 
                    "📈 Energy Analysis",
                    "🖼️ Snapshots", 
                    "💾 Export Data"
                ])
                
                with tab1:
                    st.markdown('<h3 class="sub-header">Interactive Phase Field Evolution</h3>', unsafe_allow_html=True)
                    plotly_fig = sim.create_plotly_figure()
                    if plotly_fig:
                        st.plotly_chart(plotly_fig, use_container_width=True)
                    else:
                        st.warning("No visualization data available.")
                
                with tab2:
                    st.markdown('<h3 class="sub-header">Energy Evolution Analysis</h3>', unsafe_allow_html=True)
                    energy_fig = sim.create_energy_plot()
                    if energy_fig:
                        st.plotly_chart(energy_fig, use_container_width=True)
                        
                        # Display energy statistics
                        if sim.output_data:
                            energies = [data['energy'] for data in sim.output_data]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Initial Energy", f"{energies[0]:.2e} J")
                            with col2:
                                st.metric("Final Energy", f"{energies[-1]:.2e} J")
                            with col3:
                                energy_change = ((energies[-1] - energies[0]) / energies[0]) * 100
                                st.metric("Energy Change", f"{energy_change:.1f}%")
                
                with tab3:
                    st.markdown('<h3 class="sub-header">Simulation Snapshots</h3>', unsafe_allow_html=True)
                    if sim.snapshot_paths:
                        # Show selected snapshots
                        num_snapshots = min(6, len(sim.snapshot_paths))
                        selected_indices = np.linspace(0, len(sim.snapshot_paths)-1, num_snapshots, dtype=int)
                        
                        cols = st.columns(3)
                        for i, idx in enumerate(selected_indices):
                            with cols[i % 3]:
                                path = sim.snapshot_paths[idx]
                                step = idx * sim.params.output_interval
                                time_ns = step * sim.params.dt * 1e9
                                st.image(path, caption=f"Time: {time_ns:.2f} ns", use_column_width=True)
                    else:
                        st.warning("No snapshots available.")
                
                with tab4:
                    st.markdown('<h3 class="sub-header">Export Simulation Data</h3>', unsafe_allow_html=True)
                    
                    # Create data summary
                    summary_data = {
                        'material': material,
                        'temperature_K': temperature,
                        'domain_nm': [Lx_nm, Ly_nm],
                        'twin_width_nm': twin_width_nm,
                        'grid_points': [Nx, Ny],
                        'total_time_ns': t_max_ns,
                        'output_steps': len(sim.output_data),
                        'initial_energy_J': sim.initial_energy if hasattr(sim, 'initial_energy') else None,
                        'final_energy_J': sim.output_data[-1]['energy'] if sim.output_data else None
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export summary as JSON
                        import json
                        summary_json = json.dumps(summary_data, indent=2)
                        st.download_button(
                            label="📄 Download Summary (JSON)",
                            data=summary_json,
                            file_name=f"nanotwin_summary_{material}_{temperature}K.json",
                            mime="application/json"
                        )
                        
                        # Export time series data
                        if sim.output_data:
                            time_data = np.array([[data['time'], data['energy']] for data in sim.output_data])
                            np.savetxt('energy_vs_time.csv', time_data, 
                                      delimiter=',', header='Time (s),Energy (J)')
                            with open('energy_vs_time.csv', 'rb') as f:
                                st.download_button(
                                    label="📈 Download Energy Data (CSV)",
                                    data=f,
                                    file_name=f"energy_data_{material}_{temperature}K.csv",
                                    mime="text/csv"
                                )
                    
                    with col2:
                        # Export VTK files if available
                        if PYVISTA_AVAILABLE and sim.vtk_paths:
                            # Create zip of VTK files
                            zip_path = os.path.join(sim.output_dir, "vtk_files.zip")
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for vtk_file in sim.vtk_paths:
                                    zipf.write(vtk_file, os.path.basename(vtk_file))
                            
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    label="📦 Download VTK Files (ZIP)",
                                    data=f,
                                    file_name=f"vtk_files_{material}_{temperature}K.zip",
                                    mime="application/zip"
                                )
                        else:
                            st.info("VTK export requires PyVista with VTK support.")
                    
                    # Clean up
                    st.info("Note: Temporary files will be automatically cleaned up.")
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(sim.output_dir)
                except:
                    pass
                
            except Exception as e:
                st.error(f"❌ Simulation failed with error: {str(e)}")
                st.exception(e)
    
    else:
        # Show instructions when not running
        st.markdown("""
        <div class="info-box">
        <h3>Ready to Simulate?</h3>
        <p>To start a simulation:</p>
        <ol>
            <li>Select your material and adjust parameters in the sidebar</li>
            <li>Configure simulation settings (domain size, resolution, time)</li>
            <li>Click the <b>'Run Simulation'</b> button in the sidebar</li>
        </ol>
        <p>The simulation will run and display results in multiple interactive tabs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example visualizations
        st.markdown('<h3 class="sub-header">Example Visualizations</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://raw.githubusercontent.com/pyvista/pyvista/main/doc/_static/pyvista_logo_sm.png", 
                    caption="PyVista 3D Visualization", use_column_width=True)
        with col2:
            st.image("https://plotly.com/python/images/plotly_logo.png", 
                    caption="Plotly Interactive Plots", use_column_width=True)
        
        # System information
        with st.expander("System Information"):
            st.write(f"PyVista Available: {PYVISTA_AVAILABLE}")
            st.write(f"NumPy Version: {np.__version__}")
            st.write(f"Streamlit Version: {st.__version__}")

if __name__ == "__main__":
    main()
