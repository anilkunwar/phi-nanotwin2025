import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
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

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Nanotwin Phase Field Simulation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Module 1: Material Database
class MaterialDatabase:
    """Embedded material properties database."""
    
    MATERIALS = {
        'Cu': {
            'material': 'Copper',
            'sigma_ctb': 0.5e-3,  # J/m²
            'sigma_itb': 0.8e-3,
            'sigma_gb': 1.0e-3,
            'M_itb_prefactor': 5e-6,  # m⁴/(J·s)
            'M_itb_activation_energy': 0.2,  # eV
            'color': '#e41a1c',  # Red
            'density': 8960,  # kg/m³
            'melting_point': 1357.77  # K
        },
        'Al': {
            'material': 'Aluminum',
            'sigma_ctb': 0.4e-3,
            'sigma_itb': 0.6e-3,
            'sigma_gb': 0.9e-3,
            'M_itb_prefactor': 8e-6,
            'M_itb_activation_energy': 0.15,
            'color': '#377eb8',  # Blue
            'density': 2700,
            'melting_point': 933.47
        },
        'Ni': {
            'material': 'Nickel',
            'sigma_ctb': 0.7e-3,
            'sigma_itb': 1.0e-3,
            'sigma_gb': 1.2e-3,
            'M_itb_prefactor': 3e-6,
            'M_itb_activation_energy': 0.25,
            'color': '#4daf4a',  # Green
            'density': 8908,
            'melting_point': 1728
        },
        'Ag': {
            'material': 'Silver',
            'sigma_ctb': 0.45e-3,
            'sigma_itb': 0.7e-3,
            'sigma_gb': 0.95e-3,
            'M_itb_prefactor': 6e-6,
            'M_itb_activation_energy': 0.18,
            'color': '#984ea3',  # Purple
            'density': 10490,
            'melting_point': 1234.93
        },
        'Au': {
            'material': 'Gold',
            'sigma_ctb': 0.48e-3,
            'sigma_itb': 0.75e-3,
            'sigma_gb': 1.0e-3,
            'M_itb_prefactor': 4e-6,
            'M_itb_activation_energy': 0.22,
            'color': '#ff7f00',  # Orange
            'density': 19320,
            'melting_point': 1337.33
        },
        'Pd': {
            'material': 'Palladium',
            'sigma_ctb': 0.65e-3,
            'sigma_itb': 0.9e-3,
            'sigma_gb': 1.1e-3,
            'M_itb_prefactor': 3.5e-6,
            'M_itb_activation_energy': 0.24,
            'color': '#ffff33',  # Yellow
            'density': 12023,
            'melting_point': 1828.05
        }
    }
    
    @classmethod
    def get_material_properties(cls, material):
        """Get material properties from embedded database."""
        if material not in cls.MATERIALS:
            raise ValueError(f"Material '{material}' not found. Available: {list(cls.MATERIALS.keys())}")
        return cls.MATERIALS[material].copy()

# Module 2: Enhanced Parameters with Validation
class SimulationParameters:
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval):
        # Validate inputs
        self.validate_inputs(material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval)
        
        # Get material properties
        material_data = MaterialDatabase.get_material_properties(material)
        
        self.material = material
        self.material_name = material_data['material']
        self.material_color = material_data['color']
        self.sigma_ctb = material_data['sigma_ctb']
        self.sigma_itb = material_data['sigma_itb']
        self.sigma_gb = material_data['sigma_gb']
        self.M_itb_prefactor = material_data['M_itb_prefactor']
        self.M_itb_activation_energy = material_data['M_itb_activation_energy']
        
        # Physical constants
        self.k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
        
        # Calculate temperature-dependent mobility
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature
        self.M_itb = self.M_itb_prefactor * np.exp(-self.M_itb_activation_energy / (self.k_B * temperature))
        
        # Interface parameters
        self.l_int = 1e-9  # Interface width (m)
        self.m = 6 * self.sigma_itb / self.l_int  # Free energy barrier
        self.kappa = 0.75 * self.sigma_itb * self.l_int  # Gradient coefficient
        self.delta_sigma = 0.5  # Anisotropy strength
        self.L = 4/3 * self.M_itb / self.l_int  # Kinetic coefficient
        
        # Numerical parameters
        self.Lx, self.Ly = Lx, Ly  # Domain size (m)
        self.Nx, self.Ny = Nx, Ny  # Grid points
        self.dx = self.Lx / self.Nx  # Grid spacing (m)
        self.dy = self.Ly / self.Ny
        
        # Adaptive time step for stability
        dt_stable = min(self.dx**2, self.dy**2) / (4 * self.L * self.kappa)
        self.dt = min(1e-12, dt_stable * 0.1)  # Conservative time step
        
        self.t_max = t_max  # Total simulation time (s)
        self.output_interval = output_interval  # Output frequency
        self.twin_width = twin_width  # Nanotwin width (m)
        
        # Create grid
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Simulation metrics
        self.total_steps = int(self.t_max / self.dt)
        
    def validate_inputs(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval):
        """Validate all input parameters."""
        if material not in MaterialDatabase.MATERIALS:
            raise ValueError(f"Invalid material: {material}")
        if twin_width <= 0:
            raise ValueError("Twin width must be positive")
        if twin_width >= Ly:
            raise ValueError(f"Twin width ({twin_width}m) must be less than domain height ({Ly}m)")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain dimensions must be positive")
        if Nx < 10 or Ny < 10:
            raise ValueError("Grid resolution too low (minimum 10x10)")
        if t_max <= 0:
            raise ValueError("Simulation time must be positive")
        if output_interval <= 0:
            raise ValueError("Output interval must be positive")
            
    def get_info_string(self):
        """Get formatted information string."""
        return f"""
        Material: {self.material_name} ({self.material})
        Domain: {self.Lx*1e9:.1f} nm × {self.Ly*1e9:.1f} nm
        Grid: {self.Nx} × {self.Ny} points
        Resolution: {self.dx*1e9:.2f} nm
        Twin Width: {self.twin_width*1e9:.1f} nm
        Temperature: {self.temperature} K
        Time Step: {self.dt:.2e} s
        Total Time: {self.t_max:.2e} s ({self.t_max*1e9:.2f} ns)
        Total Steps: {self.total_steps:,}
        Sigma_ITB: {self.sigma_itb:.2e} J/m²
        Mobility: {self.M_itb:.2e} m⁴/(J·s)
        """

# Module 3: Free Energy Model
class FreeEnergyModel:
    def __init__(self, params):
        self.params = params
        self.initialize_lookup_tables()
        
    def initialize_lookup_tables(self):
        """Initialize lookup tables for performance optimization."""
        # Create lookup for anisotropy function
        self.g_values = np.linspace(0.1, 5, 1000)
        p_values = -3.0944 * self.g_values**4 - 1.8169 * self.g_values**3 + 10.323 * self.g_values**2 - 8.1819 * self.g_values
        self.gamma_values = 1.0 / p_values
        
    def local_free_energy(self, eta_m, eta_t):
        """Compute local free energy density."""
        # Double-well potentials
        f_m = eta_m**4 / 4 - eta_m**2 / 2
        f_t = eta_t**4 / 4 - eta_t**2 / 2
        
        # Interaction term
        gamma = self.compute_gamma(eta_m, eta_t)
        f_interaction = gamma * eta_m**2 * eta_t**2 / 2
        
        return self.params.m * (f_m + f_t + f_interaction + 0.25)
    
    def compute_gamma(self, eta_m, eta_t):
        """Compute inclination-dependent gamma parameter."""
        # Compute gradients
        grad_m_x, grad_m_y = np.gradient(eta_m, self.params.dx, self.params.dy)
        grad_t_x, grad_t_y = np.gradient(eta_t, self.params.dx, self.params.dy)
        
        # Difference in gradients
        grad_diff_x = grad_m_x - grad_t_x
        grad_diff_y = grad_m_y - grad_t_y
        
        # Compute inclination angle
        phi = np.arctan2(grad_diff_y, grad_diff_x)
        phi = np.nan_to_num(phi, nan=0.0)
        
        # Anisotropic interfacial energy
        sigma_mt = self.params.sigma_itb * (1 + self.params.delta_sigma * np.cos(phi))
        
        # Compute g parameter
        sqrt_km = np.sqrt(self.params.kappa * self.params.m)
        g = sigma_mt / sqrt_km
        
        # Interpolate gamma from lookup table
        gamma = np.interp(g, self.g_values, self.gamma_values, left=0.1, right=10.0)
        
        return np.clip(gamma, 0.1, 10.0)
    
    def gradient_energy(self, eta_m, eta_t):
        """Compute gradient energy contribution."""
        grad_m_x, grad_m_y = np.gradient(eta_m, self.params.dx, self.params.dy)
        grad_t_x, grad_t_y = np.gradient(eta_t, self.params.dx, self.params.dy)
        
        f_grad = 0.5 * self.params.kappa * (
            grad_m_x**2 + grad_m_y**2 + 
            grad_t_x**2 + grad_t_y**2
        )
        
        return f_grad
    
    def total_free_energy(self, eta_m, eta_t):
        """Compute total free energy of the system."""
        # Local energy
        f_local = self.local_free_energy(eta_m, eta_t)
        
        # Gradient energy
        f_grad = self.gradient_energy(eta_m, eta_t)
        
        # Total energy density
        f_total = f_local + f_grad
        
        # Integrate over domain
        energy = np.trapz(np.trapz(f_total, dx=self.params.dx, axis=0), dx=self.params.dy)
        
        return energy
    
    def functional_derivative(self, eta_m, eta_t):
        """Compute functional derivative for Allen-Cahn equation."""
        gamma = self.compute_gamma(eta_m, eta_t)
        
        # Bulk term
        df_m_bulk = self.params.m * (eta_m**3 - eta_m + 2 * gamma * eta_m * eta_t**2)
        df_t_bulk = self.params.m * (eta_t**3 - eta_t + 2 * gamma * eta_t * eta_m**2)
        
        # Gradient term
        df_m_grad = -self.params.kappa * laplace(eta_m, mode='wrap')
        df_t_grad = -self.params.kappa * laplace(eta_t, mode='wrap')
        
        # Anisotropic term (simplified for stability)
        grad_m_x, grad_m_y = np.gradient(eta_m, self.params.dx, self.params.dy)
        grad_t_x, grad_t_y = np.gradient(eta_t, self.params.dx, self.params.dy)
        
        # Compute anisotropy contribution
        phi = np.arctan2(grad_m_y - grad_t_y, grad_m_x - grad_t_x)
        phi = np.nan_to_num(phi, nan=0.0)
        
        dsigma_dphi = -self.params.sigma_itb * self.params.delta_sigma * np.sin(phi)
        anisotropy_term = dsigma_dphi * eta_m**2 * eta_t**2
        
        df_m = df_m_bulk + df_m_grad + anisotropy_term
        df_t = df_t_bulk + df_t_grad - anisotropy_term
        
        return df_m, df_t

# Module 4: Phase Field Evolution
class PhaseFieldEvolution:
    def __init__(self, params, free_energy):
        self.params = params
        self.free_energy = free_energy
        self.energy_history = []
        self.time_history = []
        
    def initialize_fields(self):
        """Initialize order parameters with smooth interface."""
        # Center of domain
        y_center = self.params.Ly / 2
        half_width = self.params.twin_width / 2
        
        # Create smooth twin profile using hyperbolic tangent
        eta_t = 0.5 * (
            1 + np.tanh(4 * (self.params.Y - (y_center - half_width)) / self.params.l_int)
        ) * 0.5 * (
            1 - np.tanh(4 * (self.params.Y - (y_center + half_width)) / self.params.l_int)
        )
        
        # Matrix phase
        eta_m = 1 - eta_t
        
        # Add small perturbation to break symmetry
        noise = 0.01 * np.random.randn(self.params.Ny, self.params.Nx)
        eta_m += noise
        eta_t -= noise
        
        # Normalize
        sum_eta = eta_m + eta_t
        eta_m = eta_m / sum_eta
        eta_t = eta_t / sum_eta
        
        return eta_m, eta_t
    
    def update_step(self, eta_m, eta_t):
        """Perform one time step of Allen-Cahn evolution."""
        # Compute functional derivatives
        df_m, df_t = self.free_energy.functional_derivative(eta_m, eta_t)
        
        # Allen-Cahn evolution
        eta_m_new = eta_m - self.params.dt * self.params.L * df_m
        eta_t_new = eta_t - self.params.dt * self.params.L * df_t
        
        # Apply bounds
        eta_m_new = np.clip(eta_m_new, 0, 1)
        eta_t_new = np.clip(eta_t_new, 0, 1)
        
        # Ensure conservation (eta_m + eta_t ≈ 1)
        sum_eta = eta_m_new + eta_t_new
        mask = np.abs(sum_eta - 1) > 0.01
        if np.any(mask):
            norm = eta_m_new[mask] + eta_t_new[mask]
            eta_m_new[mask] = eta_m_new[mask] / norm
            eta_t_new[mask] = eta_t_new[mask] / norm
        
        # Track energy
        current_energy = self.free_energy.total_free_energy(eta_m_new, eta_t_new)
        self.energy_history.append(current_energy)
        
        return eta_m_new, eta_t_new
    
    def get_energy_history(self):
        """Return energy evolution data."""
        times = np.arange(len(self.energy_history)) * self.params.dt
        return times, np.array(self.energy_history)

# Module 5: Data Export (No VTK dependency)
class DataExporter:
    """Handles data export in various formats without VTK dependency."""
    
    @staticmethod
    def export_numpy_data(eta_m, eta_t, x, y, time, filename):
        """Export data as numpy binary files."""
        data_dict = {
            'eta_m': eta_m,
            'eta_t': eta_t,
            'x': x,
            'y': y,
            'time': time,
            'metadata': {
                'description': 'Nanotwin phase field simulation data',
                'units': {
                    'x': 'm',
                    'y': 'm',
                    'time': 's',
                    'eta': 'dimensionless'
                }
            }
        }
        
        np.savez(filename, **data_dict)
        return filename
    
    @staticmethod
    def export_csv_data(eta_m, eta_t, x, y, time, filename):
        """Export data as CSV files."""
        # Flatten arrays for CSV
        X_flat, Y_flat = np.meshgrid(x, y)
        data = np.column_stack([
            X_flat.flatten(),
            Y_flat.flatten(),
            eta_m.flatten(),
            eta_t.flatten()
        ])
        
        header = "x (m),y (m),eta_m,eta_t"
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        return filename
    
    @staticmethod
    def export_json_summary(params, energy_history, filename):
        """Export simulation summary as JSON."""
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
                'output_interval': params.output_interval
            },
            'physics': {
                'sigma_itb': params.sigma_itb,
                'M_itb': params.M_itb,
                'm': params.m,
                'kappa': params.kappa
            },
            'energy': {
                'initial': energy_history[0] if energy_history else 0,
                'final': energy_history[-1] if energy_history else 0,
                'history': energy_history.tolist() if hasattr(energy_history, 'tolist') else energy_history
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filename
    
    @staticmethod
    def create_zip_archive(files, zip_filename):
        """Create a ZIP archive of multiple files."""
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                if os.path.exists(file):
                    zipf.write(file, os.path.basename(file))
        return zip_filename

# Module 6: Visualization
class Visualization:
    """Handles all visualization tasks."""
    
    @staticmethod
    def create_snapshot(eta_m, eta_t, params, time, colorscale='viridis'):
        """Create a matplotlib figure snapshot."""
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
        
        # Sum (should be ~1)
        im3 = axes[2].imshow(
            eta_m + eta_t, 
            extent=[0, params.Lx*1e9, 0, params.Ly*1e9], 
            origin='lower', 
            cmap='RdYlBu',
            vmin=0.95, vmax=1.05
        )
        axes[2].set_title(f'Sum (η_m + η_t)\nTime: {time*1e9:.2f} ns')
        axes[2].set_xlabel('x (nm)')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_interactive_plot(data_history, params, colorscale='viridis'):
        """Create interactive Plotly visualization."""
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
                        zmin=0, zmax=1
                    ),
                    go.Heatmap(
                        x=x, y=y, z=data['eta_t'],
                        colorscale=colorscale,
                        zmin=0, zmax=1
                    )
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
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Time: ',
                'suffix': ' ns',
                'visible': True
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [f"frame_{i}"],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f"{data['time']*1e9:.2f}",
                    'method': 'animate'
                }
                for i, data in enumerate(data_history)
            ]
        }]
        
        # Update layout
        fig.update_layout(
            title=f"Nanotwin Evolution - {params.material_name}",
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
        
        fig.frames = frames
        
        return fig
    
    @staticmethod
    def create_energy_plot(time_history, energy_history):
        """Create energy evolution plot."""
        if len(time_history) != len(energy_history) or len(time_history) == 0:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_history * 1e9,  # Convert to ns
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
        """Create 3D surface plot of order parameters."""
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

# Module 7: Main Simulation Engine
class NanotwinSimulation:
    """Main simulation engine."""
    
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval, colorscale):
        # Initialize parameters
        self.params = SimulationParameters(material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval)
        self.free_energy = FreeEnergyModel(self.params)
        self.evolution = PhaseFieldEvolution(self.params, self.free_energy)
        self.visualization = Visualization()
        self.exporter = DataExporter()
        
        self.colorscale = colorscale
        self.output_dir = tempfile.mkdtemp(prefix=f"nanotwin_{material}_")
        
        # Initialize fields
        self.eta_m, self.eta_t = self.evolution.initialize_fields()
        
        # Data storage
        self.data_history = []
        self.snapshot_paths = []
        self.export_files = []
        
    def run(self, progress_callback=None, status_callback=None):
        """Run the simulation."""
        steps = self.params.total_steps
        
        for step in range(steps + 1):
            t = step * self.params.dt
            
            # Update phase fields
            self.eta_m, self.eta_t = self.evolution.update_step(self.eta_m, self.eta_t)
            
            # Save output at intervals
            if step % self.params.output_interval == 0 or step == steps:
                self._save_output(step, t)
                
                # Call progress callback if provided
                if progress_callback:
                    progress = min(step / steps, 1.0)
                    progress_callback(progress)
                
                # Call status callback if provided
                if status_callback:
                    energy = self.free_energy.total_free_energy(self.eta_m, self.eta_t)
                    status_callback(step, steps, t, energy)
        
        return self.data_history
    
    def _save_output(self, step, time):
        """Save output data for current step."""
        # Store data for visualization
        self.data_history.append({
            'step': step,
            'time': time,
            'eta_m': self.eta_m.copy(),
            'eta_t': self.eta_t.copy(),
            'energy': self.free_energy.total_free_energy(self.eta_m, self.eta_t)
        })
        
        # Create snapshot
        fig = self.visualization.create_snapshot(
            self.eta_m, self.eta_t, 
            self.params, time, 
            self.colorscale
        )
        
        snapshot_path = os.path.join(self.output_dir, f'snapshot_{step:06d}.png')
        fig.savefig(snapshot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.snapshot_paths.append(snapshot_path)
        
        # Export data files
        if step % (self.params.output_interval * 10) == 0 or step == self.params.total_steps:
            # Export numpy data
            npz_path = os.path.join(self.output_dir, f'data_{step:06d}.npz')
            self.exporter.export_numpy_data(
                self.eta_m, self.eta_t,
                self.params.x, self.params.y,
                time, npz_path
            )
            self.export_files.append(npz_path)
            
            # Export CSV data
            csv_path = os.path.join(self.output_dir, f'data_{step:06d}.csv')
            self.exporter.export_csv_data(
                self.eta_m, self.eta_t,
                self.params.x, self.params.y,
                time, csv_path
            )
            self.export_files.append(csv_path)
    
    def get_results(self):
        """Get simulation results."""
        times, energies = self.evolution.get_energy_history()
        
        return {
            'data_history': self.data_history,
            'energy_history': energies,
            'time_history': times,
            'snapshots': self.snapshot_paths,
            'export_files': self.export_files,
            'params': self.params,
            'output_dir': self.output_dir
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.output_dir)
        except:
            pass

# Module 8: Streamlit Application
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
    .parameter-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Phase-Field Simulation of Nanotwin Detwinning</h1>', unsafe_allow_html=True)
    
    # Introduction
    with st.expander("📘 About This Simulation", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Scientific Background
            This application simulates the **detwinning process of nanotwins** in FCC metals using a **multi-phase field model**. 
            
            ### Key Features:
            - **6 FCC Materials**: Cu, Al, Ni, Ag, Au, Pd with accurate properties
            - **Interactive 3D Visualization**: Real-time phase field evolution
            - **Energy Tracking**: Monitor free energy changes during detwinning
            - **Multiple Export Formats**: NPZ, CSV, JSON for further analysis
            - **No External Dependencies**: Runs entirely in the browser
            
            ### Physics Model:
            The simulation uses **Allen-Cahn equations** with:
            - Double-well potential for phase stability
            - Gradient energy for interface description
            - Anisotropic interfacial energy
            - Temperature-dependent mobility
            """)
        
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Phase_field_simulation.png/400px-Phase_field_simulation.png", 
                    caption="Phase Field Simulation Concept")
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Simulation Parameters")
    
    # Material selection
    material = st.sidebar.selectbox(
        "Material",
        ["Cu", "Al", "Ni", "Ag", "Au", "Pd"],
        index=0,
        help="Select FCC material for simulation"
    )
    
    # Display material info
    material_props = MaterialDatabase.get_material_properties(material)
    with st.sidebar.expander(f"📊 {material} Properties"):
        st.write(f"**{material_props['material']}**")
        st.write(f"Color: {material_props['color']}")
        st.write(f"Density: {material_props['density']:,} kg/m³")
        st.write(f"Melting Point: {material_props['melting_point']} K")
        st.write(f"Σ_ITB: {material_props['sigma_itb']:.2e} J/m²")
    
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
        'Spectral': 'Spectral'
    }
    
    colorscale = st.sidebar.selectbox(
        "Colorscale",
        list(colorscale_options.keys()),
        index=0
    )
    
    # Geometry parameters
    st.sidebar.markdown("### 📏 Geometry")
    
    twin_width_nm = st.sidebar.slider(
        "Twin Width (nm)",
        1.0, 50.0, 10.0,
        help="Initial width of the nanotwin"
    )
    
    Lx_nm = st.sidebar.slider(
        "Domain Width (nm)",
        20.0, 200.0, 100.0
    )
    
    Ly_nm = st.sidebar.slider(
        "Domain Height (nm)",
        20.0, 200.0, 100.0
    )
    
    # Numerical parameters
    st.sidebar.markdown("### 🎯 Numerical")
    
    Nx = st.sidebar.slider(
        "Grid Points X",
        50, 300, 150
    )
    
    Ny = st.sidebar.slider(
        "Grid Points Y",
        50, 300, 150
    )
    
    # Physical parameters
    st.sidebar.markdown("### ⚡ Physics")
    
    temperature = st.sidebar.slider(
        "Temperature (K)",
        100, 1000, 500
    )
    
    t_max_ns = st.sidebar.slider(
        "Simulation Time (ns)",
        0.1, 5.0, 1.0
    )
    
    output_interval = st.sidebar.slider(
        "Output Interval (steps)",
        10, 200, 50
    )
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced"):
        random_seed = st.number_input("Random Seed", 0, 1000, 42)
        np.random.seed(random_seed)
        
        anisotropy = st.slider("Anisotropy Strength", 0.0, 1.0, 0.5)
    
    # Run button
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button(
        "🚀 Run Simulation", 
        type="primary",
        use_container_width=True
    )
    
    # Convert units
    twin_width = twin_width_nm * 1e-9
    Lx, Ly = Lx_nm * 1e-9, Ly_nm * 1e-9
    t_max = t_max_ns * 1e-9
    
    # Main content
    if run_simulation:
        # Initialize simulation
        with st.spinner("🚀 Initializing simulation..."):
            try:
                sim = NanotwinSimulation(
                    material, twin_width, temperature,
                    Lx, Ly, Nx, Ny, t_max,
                    output_interval, colorscale_options[colorscale]
                )
                
                # Display simulation info
                st.markdown(f"""
                <div class="info-box">
                <h3>Simulation Configuration</h3>
                <p><b>Material:</b> {material_props['material']} ({material})</p>
                <p><b>Domain:</b> {Lx_nm:.1f} nm × {Ly_nm:.1f} nm</p>
                <p><b>Grid:</b> {Nx} × {Ny} points (Δx = {sim.params.dx*1e9:.2f} nm)</p>
                <p><b>Twin Width:</b> {twin_width_nm:.1f} nm</p>
                <p><b>Temperature:</b> {temperature} K</p>
                <p><b>Total Time:</b> {t_max_ns:.2f} ns ({sim.params.total_steps:,} steps)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run simulation
                with st.spinner("⏳ Running simulation... This may take a moment."):
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                    
                    def update_status(step, total_steps, time, energy):
                        status_text.text(f"""
                        **Simulation Progress:**
                        - Step: {step:,} / {total_steps:,}
                        - Time: {time*1e9:.3f} ns
                        - Energy: {energy:.2e} J
                        - Progress: {step/total_steps*100:.1f}%
                        """)
                    
                    # Run the simulation
                    data_history = sim.run(update_progress, update_status)
                    
                    # Get results
                    results = sim.get_results()
                
                # Success message
                st.success(f"✅ Simulation completed! Processed {len(data_history)} time steps.")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Interactive", 
                    "📈 Analysis",
                    "🖼️ Snapshots", 
                    "🔬 3D View",
                    "💾 Export"
                ])
                
                # Tab 1: Interactive visualization
                with tab1:
                    st.markdown('<h3 class="sub-header">Interactive Phase Field Evolution</h3>', unsafe_allow_html=True)
                    plotly_fig = Visualization.create_interactive_plot(
                        data_history, sim.params, colorscale_options[colorscale]
                    )
                    if plotly_fig:
                        st.plotly_chart(plotly_fig, use_container_width=True)
                    
                    # Add statistics
                    if data_history:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Time Steps", f"{len(data_history):,}")
                        with col2:
                            st.metric("Final Time", f"{data_history[-1]['time']*1e9:.2f} ns")
                        with col3:
                            eta_m_mean = np.mean(data_history[-1]['eta_m'])
                            st.metric("Matrix Fraction", f"{eta_m_mean:.3f}")
                        with col4:
                            eta_t_mean = np.mean(data_history[-1]['eta_t'])
                            st.metric("Twin Fraction", f"{eta_t_mean:.3f}")
                
                # Tab 2: Analysis
                with tab2:
                    st.markdown('<h3 class="sub-header">Energy Analysis</h3>', unsafe_allow_html=True)
                    
                    # Energy plot
                    time_history, energy_history = sim.evolution.get_energy_history()
                    energy_fig = Visualization.create_energy_plot(time_history, energy_history)
                    if energy_fig:
                        st.plotly_chart(energy_fig, use_container_width=True)
                    
                    # Energy statistics
                    if len(energy_history) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Initial Energy", f"{energy_history[0]:.2e} J")
                        with col2:
                            st.metric("Final Energy", f"{energy_history[-1]:.2e} J")
                        with col3:
                            energy_change = energy_history[-1] - energy_history[0]
                            st.metric("ΔEnergy", f"{energy_change:.2e} J")
                        with col4:
                            if energy_history[0] != 0:
                                percent_change = (energy_change / energy_history[0]) * 100
                                st.metric("% Change", f"{percent_change:.1f}%")
                        
                        # Energy derivatives
                        if len(energy_history) > 1:
                            st.markdown("#### Energy Derivatives")
                            energy_diff = np.diff(energy_history)
                            time_diff = np.diff(time_history)
                            power = -energy_diff / time_diff  # Negative sign for dissipation rate
                            
                            fig_power = go.Figure()
                            fig_power.add_trace(go.Scatter(
                                x=time_history[1:] * 1e9,
                                y=power,
                                mode='lines',
                                name='Dissipation Rate',
                                line=dict(color='red', width=2)
                            ))
                            fig_power.update_layout(
                                title='Energy Dissipation Rate',
                                xaxis_title='Time (ns)',
                                yaxis_title='Power (W)',
                                height=400
                            )
                            st.plotly_chart(fig_power, use_container_width=True)
                
                # Tab 3: Snapshots
                with tab3:
                    st.markdown('<h3 class="sub-header">Simulation Snapshots</h3>', unsafe_allow_html=True)
                    
                    if sim.snapshot_paths:
                        # Show selected snapshots
                        num_to_show = min(6, len(sim.snapshot_paths))
                        indices = np.linspace(0, len(sim.snapshot_paths)-1, num_to_show, dtype=int)
                        
                        cols = st.columns(3)
                        for i, idx in enumerate(indices):
                            with cols[i % 3]:
                                path = sim.snapshot_paths[idx]
                                time_ns = data_history[idx]['time'] * 1e9
                                st.image(
                                    path, 
                                    caption=f"Time: {time_ns:.2f} ns",
                                    use_column_width=True
                                )
                    else:
                        st.info("No snapshots available.")
                
                # Tab 4: 3D View
                with tab4:
                    st.markdown('<h3 class="sub-header">3D Phase Field Visualization</h3>', unsafe_allow_html=True)
                    
                    if data_history:
                        # Select time step for 3D view
                        selected_idx = st.slider(
                            "Select Time Step",
                            0, len(data_history)-1, 
                            len(data_history)-1,
                            key="3d_slider"
                        )
                        
                        selected_data = data_history[selected_idx]
                        surface_fig = Visualization.create_3d_surface(
                            selected_data['eta_m'],
                            selected_data['eta_t'],
                            sim.params,
                            selected_data['time']
                        )
                        
                        if surface_fig:
                            st.plotly_chart(surface_fig, use_container_width=True)
                        
                        # Cross-section plots
                        st.markdown("#### Cross-Section Analysis")
                        cross_section_y = st.slider(
                            "Y Position (nm)",
                            0.0, sim.params.Ly*1e9,
                            sim.params.Ly*1e9/2,
                            key="cross_section"
                        )
                        
                        # Find nearest y index
                        y_idx = int(cross_section_y / (sim.params.Ly*1e9) * sim.params.Ny)
                        y_idx = np.clip(y_idx, 0, sim.params.Ny-1)
                        
                        fig_cross = go.Figure()
                        fig_cross.add_trace(go.Scatter(
                            x=np.linspace(0, sim.params.Lx*1e9, sim.params.Nx),
                            y=selected_data['eta_m'][y_idx, :],
                            mode='lines',
                            name='Matrix (η_m)',
                            line=dict(color='blue', width=2)
                        ))
                        fig_cross.add_trace(go.Scatter(
                            x=np.linspace(0, sim.params.Lx*1e9, sim.params.Nx),
                            y=selected_data['eta_t'][y_idx, :],
                            mode='lines',
                            name='Twin (η_t)',
                            line=dict(color='red', width=2)
                        ))
                        fig_cross.update_layout(
                            title=f'Cross-Section at y = {cross_section_y:.1f} nm',
                            xaxis_title='x (nm)',
                            yaxis_title='Order Parameter',
                            height=400
                        )
                        st.plotly_chart(fig_cross, use_container_width=True)
                
                # Tab 5: Export
                with tab5:
                    st.markdown('<h3 class="sub-header">Export Simulation Data</h3>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export summary
                        st.markdown("##### Simulation Summary")
                        summary_path = os.path.join(sim.output_dir, "simulation_summary.json")
                        DataExporter.export_json_summary(
                            sim.params, 
                            results['energy_history'], 
                            summary_path
                        )
                        
                        with open(summary_path, 'r') as f:
                            summary_data = json.load(f)
                        
                        st.download_button(
                            label="📄 Download Summary (JSON)",
                            data=json.dumps(summary_data, indent=2),
                            file_name=f"nanotwin_{material}_{temperature}K_summary.json",
                            mime="application/json"
                        )
                        
                        # Export energy data
                        st.markdown("##### Energy Data")
                        energy_csv = os.path.join(sim.output_dir, "energy_data.csv")
                        energy_array = np.column_stack([time_history, energy_history])
                        np.savetxt(energy_csv, energy_array, delimiter=',', 
                                  header='time(s),energy(J)', comments='')
                        
                        with open(energy_csv, 'rb') as f:
                            st.download_button(
                                label="📈 Download Energy (CSV)",
                                data=f,
                                file_name=f"nanotwin_{material}_energy.csv",
                                mime="text/csv"
                            )
                    
                    with col2:
                        # Export final state
                        st.markdown("##### Final State Data")
                        if data_history:
                            final_data = data_history[-1]
                            
                            # Export as numpy
                            npz_path = os.path.join(sim.output_dir, "final_state.npz")
                            np.savez(npz_path,
                                    eta_m=final_data['eta_m'],
                                    eta_t=final_data['eta_t'],
                                    x=sim.params.x,
                                    y=sim.params.y,
                                    time=final_data['time'])
                            
                            with open(npz_path, 'rb') as f:
                                st.download_button(
                                    label="🔢 Download Final State (NPZ)",
                                    data=f,
                                    file_name=f"nanotwin_{material}_final.npz",
                                    mime="application/octet-stream"
                                )
                            
                            # Export as CSV
                            csv_path = os.path.join(sim.output_dir, "final_state.csv")
                            X, Y = np.meshgrid(sim.params.x, sim.params.y)
                            csv_data = np.column_stack([
                                X.flatten(),
                                Y.flatten(),
                                final_data['eta_m'].flatten(),
                                final_data['eta_t'].flatten()
                            ])
                            np.savetxt(csv_path, csv_data, delimiter=',',
                                      header='x(m),y(m),eta_m,eta_t', comments='')
                            
                            with open(csv_path, 'rb') as f:
                                st.download_button(
                                    label="📋 Download Final State (CSV)",
                                    data=f,
                                    file_name=f"nanotwin_{material}_final.csv",
                                    mime="text/csv"
                                )
                    
                    # Export all data as ZIP
                    st.markdown("---")
                    st.markdown("##### Complete Dataset")
                    
                    all_files = [summary_path, energy_csv]
                    if 'npz_path' in locals():
                        all_files.append(npz_path)
                    if 'csv_path' in locals():
                        all_files.append(csv_path)
                    
                    zip_path = os.path.join(sim.output_dir, "complete_dataset.zip")
                    DataExporter.create_zip_archive(all_files, zip_path)
                    
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="📦 Download All Data (ZIP)",
                            data=f,
                            file_name=f"nanotwin_{material}_complete.zip",
                            mime="application/zip"
                        )
                
                # Cleanup
                sim.cleanup()
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                st.exception(e)
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="info-box">
        <h3>Welcome to the Nanotwin Simulation Platform!</h3>
        <p>This tool simulates the detwinning process in nanocrystalline FCC metals using advanced phase field methods.</p>
        
        <h4>📋 Getting Started:</h4>
        <ol>
            <li>Select your material from the sidebar</li>
            <li>Configure the simulation parameters</li>
            <li>Click <b>'Run Simulation'</b> to start</li>
            <li>Explore results in the interactive tabs</li>
        </ol>
        
        <h4>🔬 Scientific Applications:</h4>
        <ul>
            <li>Study nanotwin stability under thermal loads</li>
            <li>Analyze interface migration kinetics</li>
            <li>Investigate anisotropic effects on detwinning</li>
            <li>Compare material response across FCC metals</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start examples
        st.markdown("### 🚀 Quick Start Examples")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="parameter-box">
            <h4>Copper Nanotwin</h4>
            <p><b>Material:</b> Cu</p>
            <p><b>Twin Width:</b> 10 nm</p>
            <p><b>Temperature:</b> 500 K</p>
            <p><b>Domain:</b> 100×100 nm</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="parameter-box">
            <h4>Aluminum Study</h4>
            <p><b>Material:</b> Al</p>
            <p><b>Twin Width:</b> 15 nm</p>
            <p><b>Temperature:</b> 400 K</p>
            <p><b>Domain:</b> 80×80 nm</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="parameter-box">
            <h4>Nickel Analysis</h4>
            <p><b>Material:</b> Ni</p>
            <p><b>Twin Width:</b> 8 nm</p>
            <p><b>Temperature:</b> 600 K</p>
            <p><b>Domain:</b> 120×120 nm</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
