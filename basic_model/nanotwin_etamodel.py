import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import streamlit as st
import plotly.graph_objects as go
import pyvista as pv
import os
import uuid
import shutil
import zipfile
import yaml
from pathlib import Path

# Module 1: Parameters
class SimulationParameters:
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval):
        # Load material properties from YAML
        #yaml_file = f"{material.lower()}.yaml"
        # Modified line with straightforward  definition of path:
        yaml_file = os.path.join(os.path.dirname(__file__), f"{material.lower()}.yaml")
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"YAML file for material '{material}' not found: {yaml_file}")
        with open(yaml_file, 'r') as file:
            material_data = yaml.safe_load(file)
        
        self.material = material_data['material']
        self.sigma_ctb = material_data['sigma_ctb']
        self.sigma_itb = material_data['sigma_itb']
        self.sigma_gb = material_data['sigma_gb']
        self.M_itb_prefactor = material_data['M_itb_prefactor']
        self.M_itb_activation_energy = material_data['M_itb_activation_energy']
        self.M_itb = self.M_itb_prefactor * np.exp(-self.M_itb_activation_energy / (8.617e-5 * temperature))
        
        self.l_int = 1e-9      # Interface width (m)
        self.m = 6 * self.sigma_itb / self.l_int  # Free energy barrier
        self.kappa = 0.75 * self.sigma_itb * self.l_int  # Gradient coefficient
        self.delta_sigma = 0.5  # Anisotropy strength
        self.L = 4/3 * self.M_itb / self.l_int  # Kinetic coefficient

        # Numerical parameters
        self.Lx, self.Ly = Lx, Ly  # Domain size (m)
        self.Nx, self.Ny = Nx, Ny  # Grid points
        self.dx = self.Lx / self.Nx  # Grid spacing (m)
        self.dt = 1e-12  # Time step (s)
        self.t_max = t_max  # Total simulation time (s)
        self.output_interval = output_interval  # Output frequency
        self.twin_width = twin_width  # Nanotwin width (m)

# Module 2: Free Energy
class FreeEnergy:
    def __init__(self, params):
        self.params = params

    def local_energy(self, eta):
        """Compute local free energy density f_0(eta)."""
        eta_m, eta_t = eta
        m = self.params.m
        f0 = 0
        for eta_i in [eta_m, eta_t]:
            f0 += (eta_i**4 / 4 - eta_i**2 / 2)
        gamma_mt = self.compute_gamma(eta_m, eta_t)
        f0 += gamma_mt * eta_m**2 * eta_t**2 / 2 + 0.25
        return m * f0

    def compute_gamma(self, eta_m, eta_t):
        """Compute inclination-dependent gamma_mt."""
        grad_m = np.gradient(eta_m, self.params.dx, axis=(0, 1))
        grad_t = np.gradient(eta_t, self.params.dx, axis=(0, 1))
        grad_diff = np.array([grad_m[0] - grad_t[0], grad_m[1] - grad_t[1]])
        norm = np.sqrt(grad_diff[0]**2 + grad_diff[1]**2 + 1e-10)
        phi = np.arctan2(grad_diff[1], grad_diff[0])
        sigma_mt = self.params.sigma_itb * (1 + self.params.delta_sigma * np.cos(phi))
        g = sigma_mt / np.sqrt(self.params.kappa * self.params.m)
        gamma = 1.0 / (-3.0944 * g**4 - 1.8169 * g**3 + 10.323 * g**2 - 8.1819 * g)
        return np.clip(gamma, 0.1, 10)

    def anisotropic_term(self, eta_m, eta_t, gamma_mt):
        """Compute anisotropic term for evolution equation."""
        grad_m = np.gradient(eta_m, self.params.dx, axis=(0, 1))
        grad_t = np.gradient(eta_t, self.params.dx, axis=(0, 1))
        grad_diff = np.array([grad_m[0] - grad_t[0], grad_m[1] - grad_t[1]])
        norm_sq = grad_diff[0]**2 + grad_diff[1]**2 + 1e-10
        phi = np.arctan2(grad_diff[1], grad_diff[0])
        sigma_mt = self.params.sigma_itb * (1 + self.params.delta_sigma * np.cos(phi))
        g = sigma_mt / np.sqrt(self.params.kappa * self.params.m)
        dg_dphi = -self.params.sigma_itb * self.params.delta_sigma * np.sin(phi) / np.sqrt(self.params.kappa * self.params.m)
        p_g2 = -3.0944 * g**4 - 1.8169 * g**3 + 10.323 * g**2 - 8.1819 * g
        dp_dg2 = -12.3776 * g**3 - 5.4507 * g**2 + 20.646 * g - 8.1819
        dgamma_dphi = -2 * g * gamma_mt**2 * dp_dg2 * dg_dphi
        vec = np.array([-grad_diff[1], grad_diff[0]]) / norm_sq
        term = dgamma_dphi * vec * eta_m**2 * eta_t**2
        div_term = np.gradient(term[0], self.params.dx, axis=0) + np.gradient(term[1], self.params.dx, axis=1)
        return self.params.m * div_term

# Module 3: Evolution
class PhaseFieldEvolution:
    def __init__(self, params, free_energy):
        self.params = params
        self.free_energy = free_energy

    def compute_derivative(self, eta):
        """Compute functional derivative of free energy."""
        eta_m, eta_t = eta
        m = self.params.m
        kappa = self.params.kappa
        gamma_mt = self.free_energy.compute_gamma(eta_m, eta_t)

        # Bulk term
        df_m = m * (eta_m**3 - eta_m + 2 * eta_m * gamma_mt * eta_t**2)
        df_t = m * (eta_t**3 - eta_t + 2 * eta_t * gamma_mt * eta_m**2)

        # Gradient term
        df_m -= kappa * laplace(eta_m, mode='wrap')
        df_t -= kappa * laplace(eta_t, mode='wrap')

        # Anisotropic term
        df_m += self.free_energy.anisotropic_term(eta_m, eta_t, gamma_mt)
        df_t -= self.free_energy.anisotropic_term(eta_m, eta_t, gamma_mt)

        return np.array([df_m, df_t])

    def update(self, eta):
        """Update order parameters using Allen–Cahn equation."""
        df = self.compute_derivative(eta)
        eta_new = eta - self.params.dt * self.params.L * df
        return np.clip(eta_new, 0, 1)

# Module 4: Simulation
class Simulation:
    def __init__(self, material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval, colorscale):
        self.params = SimulationParameters(material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval)
        self.free_energy = FreeEnergy(self.params)
        self.evolution = PhaseFieldEvolution(self.params, self.free_energy)
        self.colorscale = colorscale
        self.output_dir = f"output_{uuid.uuid4().hex[:8]}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.initialize_fields()
        self.output_data = []  # Store data for Plotly visualization

    def initialize_fields(self):
        """Initialize order parameters for a nanotwin in a matrix."""
        Nx, Ny = self.params.Nx, self.params.Ny
        x = np.linspace(0, self.params.Lx, Nx)
        y = np.linspace(0, self.params.Ly, Ny)
        X, Y = np.meshgrid(x, y)
        eta_m = np.ones((Ny, Nx))
        eta_t = np.zeros((Ny, Nx))
        mask = (Y >= self.params.Ly/2 - self.params.twin_width/2) & (Y <= self.params.Ly/2 + self.params.twin_width/2)
        eta_t[mask] = 1.0
        eta_m[mask] = 0.0
        eta_t = 0.5 * (1 + np.tanh(4 * (Y - (self.params.Ly/2 - self.params.twin_width/2)) / self.params.l_int)) * \
                0.5 * (1 + np.tanh(-4 * (Y - (self.params.Ly/2 + self.params.twin_width/2)) / self.params.l_int))
        eta_m = 1 - eta_t
        self.eta = np.array([eta_m, eta_t])

    def save_output(self, step, t):
        """Save current state as image and VTK file, store data for Plotly."""
        # Save image for snapshots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        #im1 = ax1.imshow(self.eta[0], extent=[0, self.params.Lx*1e9, 0, self.params.Ly*1e9], origin='lower', cmap='viridis')
        im1 = ax1.imshow(self.eta[0], extent=[0, self.params.Lx*1e9, 0, self.params.Ly*1e9], origin='lower', cmap=self.colorscale)
        ax1.set_title('Matrix (η_m)')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')
        plt.colorbar(im1, ax=ax1)
        #im2 = ax2.imshow(self.eta[1], extent=[0, self.params.Lx*1e9, 0, self.params.Ly*1e9], origin='lower', cmap='viridis')
        im2 = ax2.imshow(self.eta[1], extent=[0, self.params.Lx*1e9, 0, self.params.Ly*1e9], origin='lower',  cmap=self.colorscale)
        ax2.set_title('Nanotwin (η_t)')
        ax2.set_xlabel('x (nm)')
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        image_path = os.path.join(self.output_dir, f'step_{step:06d}.png')
        plt.savefig(image_path)
        plt.close()

        # Save VTK file
        #grid = pv.RectilinearGrid(
        #    np.linspace(0, self.params.Lx, self.params.Nx),
        #    np.linspace(0, self.params.Ly, self.params.Ny),
        #    [0.0]  # Z-direction (required for 3D structure)
        #)
        #grid.dimensions = (self.params.Nx, self.params.Ny, 1)  # 3D dimensions is required for 2D visualization
        # Corrected PyVista grid initialization
        x_coords = np.linspace(0, self.params.Lx, self.params.Nx)
        y_coords = np.linspace(0, self.params.Ly, self.params.Ny)
        z_coords = np.array([0.0])  # Single layer in z-direction
    
        grid = pv.RectilinearGrid(x_coords, y_coords, z_coords)
        grid.point_data['eta_m'] = self.eta[0].flatten(order='F')
        grid.point_data['eta_t'] = self.eta[1].flatten(order='F')
        vtr_path = os.path.join(self.output_dir, f'step_{step:06d}.vtr')
        grid.save(vtr_path)

        # Store data for Plotly
        self.output_data.append({
            'step': step,
            'time': t,
            'eta_m': self.eta[0].copy(),
            'eta_t': self.eta[1].copy()
        })

        return image_path

    def create_plotly_figure(self):
        """Create a Plotly figure with time slider and single colorbar."""
        fig = go.Figure()

        # Initialize with first step
        initial_data = self.output_data[0]
        x = np.linspace(0, self.params.Lx * 1e9, self.params.Nx)
        y = np.linspace(0, self.params.Ly * 1e9, self.params.Ny)

        # Add heatmap for eta_m (left subplot)
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=initial_data['eta_m'],
                colorscale=self.colorscale, zmin=0, zmax=1,
                showscale=False  # Disable individual colorbar
            )
        )

        # Add heatmap for eta_t (right subplot)
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=initial_data['eta_t'],
                colorscale=self.colorscale, zmin=0, zmax=1,
                showscale=True,  # Enable single colorbar
                colorbar=dict(
                    x=1.02,  # Position to the right
                    len=0.8,
                    y=0.5,
                    yanchor='middle',
                    title='η'
                )
            )
        )

        # Create frames for each step
        frames = []
        for data in self.output_data:
            frame = go.Frame(
                data=[
                    go.Heatmap(x=x, y=y, z=data['eta_m'], colorscale=self.colorscale, zmin=0, zmax=1),
                    go.Heatmap(x=x, y=y, z=data['eta_t'], colorscale=self.colorscale, zmin=0, zmax=1)
                ],
                name=f"step_{data['step']}"
            )
            frames.append(frame)

        # Add slider
        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Time (ns): ',
                'visible': True,
                'xanchor': 'right'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{
                'args': [[f"step_{data['step']}"], {
                    'frame': {'duration': 300, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 300}
                }],
                'label': f"{data['time']*1e9:.2f}",
                'method': 'animate'
            } for data in self.output_data]
        }]

        # Update layout
        fig.update_layout(
            title=f"Nanotwin Evolution ({self.params.material})",
            grid={'rows': 1, 'columns': 2, 'pattern': 'independent'},
            xaxis=dict(title="x (nm)", range=[0, self.params.Lx * 1e9]),
            yaxis=dict(title="y (nm)", range=[0, self.params.Ly * 1e9]),
            xaxis2=dict(title="x (nm)", range=[0, self.params.Lx * 1e9]),
            yaxis2=dict(title="y (nm)", range=[0, self.params.Ly * 1e9]),
            annotations=[
                dict(text="Matrix (η_m)", x=0.2, y=1.1, xref="paper", yref="paper", showarrow=False),
                dict(text="Nanotwin (η_t)", x=0.8, y=1.1, xref="paper", yref="paper", showarrow=False)
            ],
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
            }],
            width=1000,
            height=500
        )

        fig.frames = frames
        return fig

    def run(self, progress_bar):
        """Run the simulation loop with progress updates."""
        steps = int(self.params.t_max / self.params.dt)
        output_paths = []
        for step in range(steps + 1):
            t = step * self.params.dt
            self.eta = self.evolution.update(self.eta)
            if step % self.params.output_interval == 0:
                output_paths.append(self.save_output(step, t))
                progress_bar.progress(min(step / steps, 1.0), text=f"Step {step}/{steps}, Time {t:.2e} s")
        return output_paths

# Module 5: Streamlit Interface
def main():
    st.title("Phase-Field Simulation of Nanotwin Detwinning")
    st.markdown("""
    This app simulates the detwinning process of nanotwins in FCC materials (Cu, Al, Ni, Ag, Au, Pd) using a phase-field model.
    Select a material, colorscale, and adjust parameters below, then click 'Run Simulation' to start. Results are visualized with
    Plotly (time slider, single colorbar) and saved as VTK (.vtr) files for external analysis (e.g., in ParaView).
    """)

    # Sidebar for user inputs
    st.sidebar.header("Simulation Parameters")
    material = st.sidebar.selectbox("Material", ["Cu", "Al", "Ni", "Ag", "Au", "Pd"])
    colormap_map = {
        'Viridis': 'viridis', 'Plasma': 'plasma', 'Inferno': 'inferno', 
        'Magma': 'magma', 'Hot': 'hot', 'Cividis': 'cividis', 'Blues': 'Blues', 
        'Greens': 'Greens', 'Reds': 'Reds', 'Oranges': 'Oranges', 'Purples': 'Purples', 
        'Greys': 'Greys', 'YlOrRd': 'YlOrRd', 'YlOrBr': 'YlOrBr', 'YlGnBu': 'YlGnBu', 
        'BuGn': 'BuGn', 'BuPu': 'BuPu', 'GnBu': 'GnBu', 'PuBu': 'PuBu', 'PuRd': 'PuRd', 
        'RdPu': 'RdPu', 'Jet': 'jet', 'Rainbow': 'rainbow', 'RdBu': 'RdBu', 
        'Spectral': 'Spectral', 'PiYG': 'PiYG', 'PRGn': 'PRGn', 'RdYlBu': 'RdYlBu', 
        'RdGy': 'RdGy', 'RdYlGn': 'RdYlGn', 'BrBG': 'BrBG', 'Twilight': 'twilight', 
        'Hsv': 'hsv', 'Phase': 'hsv'  # Phase uses hsv as fallback
    }
    
    colorscale = st.sidebar.selectbox(
        "Colorscale",
        list(colormap_map.keys()),  # Use all dictionary keys in order
        index=0  # Default to Viridis
    )
    #colorscale = st.sidebar.selectbox(
    #    "Colorscale",
    #    ["Viridis", "Plasma", "Inferno", "Magma", "Hot", "Blues", "Greens", "Reds", "Cividis", "RdBu", "Spectral", "Twilight"],
    #    index=0  # Default to Viridis
    #)
    twin_width_nm = st.sidebar.slider("Nanotwin Width (nm)", 1.0, 15.0, 5.0) # Variable for machine learning 
    temperature = st.sidebar.slider("Temperature (K)", 300, 700, 500)
    Lx_nm = st.sidebar.slider("Domain Size X (nm)", 20.0, 100.0, 50.0)
    Ly_nm = st.sidebar.slider("Domain Size Y (nm)", 20.0, 100.0, 50.0)
    Nx = st.sidebar.slider("Grid Points X", 100, 300, 200)
    Ny = st.sidebar.slider("Grid Points Y", 100, 300, 200)
    t_max_ns = st.sidebar.slider("Simulation Time (ns)", 0.1, 1.0, 0.5)
    output_interval = st.sidebar.slider("Output Interval (steps)", 50, 200, 100)

    # Convert units
    twin_width = twin_width_nm * 1e-9  # m
    Lx, Ly = Lx_nm * 1e-9, Ly_nm * 1e-9  # m
    t_max = t_max_ns * 1e-9  # s

    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            progress_bar = st.progress(0.0)
            sim = Simulation(material, twin_width, temperature, Lx, Ly, Nx, Ny, t_max, output_interval,  colormap_map[colorscale])
            output_paths = sim.run(progress_bar)
            plotly_fig = sim.create_plotly_figure()

        st.success("Simulation completed! VTK files saved in output directory.")

        # Display Plotly figure
        st.subheader("Simulation Visualization")
        st.plotly_chart(plotly_fig, use_container_width=True)

        # Display individual frames (snapshots)
        st.subheader("Simulation Snapshots")
        cols = st.columns(3)
        for i, path in enumerate(output_paths[::len(output_paths)//3]):
            with cols[i % 3]:
                st.image(path, caption=f"Step {i*output_interval}", use_column_width=True)

        # Provide download link for output directory (VTK files)
        st.subheader("Download VTK Files")
        zip_path = os.path.join(sim.output_dir, "vtk_files.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for vtr_file in Path(sim.output_dir).glob("*.vtr"):
                zipf.write(vtr_file, os.path.basename(vtr_file))
        with open(zip_path, "rb") as file:
            st.download_button(
                label="Download VTK Files",
                data=file,
                file_name="vtk_files.zip",
                mime="application/zip"
            )

        # Clean up output directory
        shutil.rmtree(sim.output_dir)

if __name__ == "__main__":
    main()
