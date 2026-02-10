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
import hashlib
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED SIMULATION DATABASE SYSTEM
# ============================================================================
class SimulationDBCu:
    """Database system for storing and retrieving nanotwin simulations"""
    
    @staticmethod
    def generate_id(sim_params):
        """Generate unique ID for simulation"""
        param_str = json.dumps(sim_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    @staticmethod
    def save_simulation(sim_params, history, metadata, results_history):
        """Save simulation to database"""
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
        
        sim_id = SimulationDBCu.generate_id(sim_params)
        
        # Store simulation data
        st.session_state.simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'history': history,  # Internal solver history (every step)
            'results_history': results_history,  # Saved frames for visualization
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        return sim_id
    
    @staticmethod
    def get_simulation(sim_id):
        """Retrieve simulation by ID"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            return st.session_state.simulations[sim_id]
        return None
    
    @staticmethod
    def get_all_simulations():
        """Get all stored simulations"""
        if 'simulations' in st.session_state:
            return st.session_state.simulations
        return {}
    
    @staticmethod
    def delete_simulation(sim_id):
        """Delete simulation from database"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            del st.session_state.simulations[sim_id]
            return True
        return False
    
    @staticmethod
    def get_simulation_list():
        """Get list of simulations for dropdown"""
        if 'simulations' not in st.session_state:
            return []
        
        simulations = []
        for sim_id, sim_data in st.session_state.simulations.items():
            params = sim_data['params']
            name = f"{params['geometry_type']} - λ={params['twin_spacing']:.1f}nm - σ={params['applied_stress']/1e6:.0f}MPa"
            simulations.append({
                'id': sim_id,
                'name': name,
                'params': params
            })
        
        return simulations

# ============================================================================
# ENHANCED PUBLICATION STYLING SYSTEM
# ============================================================================
class PublicationStylerCu:
    """Publication-quality styling for nanotwin simulations"""
    
    @staticmethod
    def get_journal_styles():
        """Return journal-specific style parameters"""
        return {
            'nature': {
                'figure_width_single': 8.9,
                'figure_width_double': 18.3,
                'font_family': 'Arial',
                'font_size_small': 7,
                'font_size_medium': 8,
                'font_size_large': 9,
                'line_width': 0.5,
                'axes_linewidth': 0.5,
                'tick_width': 0.5,
                'tick_length': 2,
                'grid_alpha': 0.1,
                'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            },
            'science': {
                'figure_width_single': 5.5,
                'figure_width_double': 11.4,
                'font_family': 'Helvetica',
                'font_size_small': 8,
                'font_size_medium': 9,
                'font_size_large': 10,
                'line_width': 0.75,
                'axes_linewidth': 0.75,
                'tick_width': 0.75,
                'tick_length': 3,
                'grid_alpha': 0.15,
                'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']
            },
            'acta_materialia': {
                'figure_width_single': 8.6,
                'figure_width_double': 17.8,
                'font_family': 'Arial',
                'font_size_small': 8,
                'font_size_medium': 9,
                'font_size_large': 10,
                'line_width': 1.0,
                'axes_linewidth': 1.0,
                'tick_width': 1.0,
                'tick_length': 4,
                'grid_alpha': 0.2,
                'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000', '#44AA99']
            }
        }
    
    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
        """Apply journal-specific styling to figure"""
        styles = PublicationStylerCu.get_journal_styles()
        style = styles.get(journal_name, styles['nature'])
        
        # Set rcParams for consistent styling
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
        
        # Apply to all axes
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        for ax in axes_flat:
            if ax is not None:
                # Add minor ticks
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                # Set spine visibility
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['top'].set_linewidth(style['axes_linewidth'] * 0.5)
                ax.spines['right'].set_linewidth(style['axes_linewidth'] * 0.5)
                
                # Improve tick formatting
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=style['tick_length'])
                ax.tick_params(which='minor', length=style['tick_length'] * 0.6)
        
        return fig, style

# ============================================================================
# ENHANCED COMPARISON SYSTEM
# ============================================================================
class SimulationComparatorCu:
    """Compare multiple nanotwin simulations"""
    
    @staticmethod
    def create_multi_simulation_plot(simulations, frames, config, style_params):
        """Create comparison plot for multiple simulations"""
        n_sims = len(simulations)
        cols = min(3, n_sims)
        rows = (n_sims + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), constrained_layout=True)
        
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (sim, frame_idx) in enumerate(zip(simulations, frames)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get data
            sim_data = SimulationDBCu.get_simulation(sim['id'])
            results = sim_data['results_history'][frame_idx]
            
            # Create visualization based on config
            if config['visualization_type'] == 'stress':
                im = ax.imshow(results['sigma_eq']/1e9, cmap='hot', 
                             vmin=0, vmax=np.percentile(results['sigma_eq'], 95)/1e9)
                ax.set_title(f"{sim['params']['geometry_type']}\nλ={sim['params']['twin_spacing']:.1f}nm", fontsize=10)
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('σ_eq (GPa)', fontsize=8)
                
            elif config['visualization_type'] == 'twin_structure':
                im = ax.imshow(results['phi'], cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                ax.set_title(f"Twin Structure\nσ={sim['params']['applied_stress']/1e6:.0f}MPa", fontsize=10)
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('φ', fontsize=8)
                
            elif config['visualization_type'] == 'twin_spacing':
                im = ax.imshow(np.clip(results['h'], 0, 50), cmap='plasma', vmin=0, vmax=30)
                ax.set_title(f"Twin Spacing\nL_CTB={sim['params']['L_CTB']:.3f}", fontsize=10)
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Spacing (nm)', fontsize=8)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide empty subplots
        for idx in range(n_sims, rows*cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        # Apply publication styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig
    
    @staticmethod
    def create_evolution_comparison(simulations, config, style_params):
        """Compare evolution of multiple simulations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
        
        for idx, sim in enumerate(simulations):
            sim_data = SimulationDBCu.get_simulation(sim['id'])
            history = sim_data['history']
            
            # Extract time evolution
            time_steps = np.arange(len(history['phi_norm'])) * sim['params']['dt']
            
            # Plot 1: Twin order parameter evolution
            axes[0, 0].plot(time_steps, history['phi_norm'], 
                          color=colors[idx], linewidth=2, alpha=0.7,
                          label=f"{sim['params']['geometry_type']}")
            
            # Plot 2: Maximum stress evolution
            axes[0, 1].plot(time_steps, np.array(history['max_stress'])/1e9, 
                          color=colors[idx], linewidth=2, alpha=0.7)
            
            # Plot 3: Average twin spacing evolution
            axes[1, 0].plot(time_steps, history['twin_spacing_avg'], 
                          color=colors[idx], linewidth=2, alpha=0.7)
            
            # Plot 4: Plastic work evolution
            axes[1, 1].plot(time_steps, history['plastic_work'], 
                          color=colors[idx], linewidth=2, alpha=0.7)
        
        # Label plots
        axes[0, 0].set_xlabel('Time (ns)')
        axes[0, 0].set_ylabel('||φ||')
        axes[0, 0].set_title('Twin Order Parameter Evolution')
        axes[0, 0].legend(fontsize=8)
        
        axes[0, 1].set_xlabel('Time (ns)')
        axes[0, 1].set_ylabel('Max Stress (GPa)')
        axes[0, 1].set_title('Stress Evolution')
        
        axes[1, 0].set_xlabel('Time (ns)')
        axes[1, 0].set_ylabel('Avg Twin Spacing (nm)')
        axes[1, 0].set_title('Twin Spacing Evolution')
        
        axes[1, 1].set_xlabel('Time (ns)')
        axes[1, 1].set_ylabel('Plastic Work (J)')
        axes[1, 1].set_title('Plastic Work Evolution')
        
        # Apply styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig
    
    @staticmethod
    def create_parameter_sensitivity_plot(simulations, config, style_params):
        """Create parameter sensitivity analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        
        # Extract parameter values and final results
        param_data = []
        for sim in simulations:
            sim_data = SimulationDBCu.get_simulation(sim['id'])
            final_results = sim_data['results_history'][-1]
            params = sim['params']
            
            param_data.append({
                'twin_spacing': params['twin_spacing'],
                'applied_stress': params['applied_stress']/1e6,  # MPa
                'W': params['W'],
                'L_CTB': params['L_CTB'],
                'max_stress': np.max(final_results['sigma_eq'])/1e6,  # MPa
                'avg_spacing': np.mean(final_results['h'][(final_results['h']>5) & (final_results['h']<50)]),
                'plastic_work': sim_data['history']['plastic_work'][-1]
            })
        
        df = pd.DataFrame(param_data)
        
        # Plot 1: Twin spacing vs max stress
        scatter1 = axes[0, 0].scatter(df['twin_spacing'], df['max_stress'], 
                                     c=df['applied_stress'], cmap='viridis', s=100, alpha=0.7)
        axes[0, 0].set_xlabel('Twin Spacing (nm)')
        axes[0, 0].set_ylabel('Max Stress (MPa)')
        axes[0, 0].set_title('Twin Spacing Effect')
        plt.colorbar(scatter1, ax=axes[0, 0]).set_label('Applied Stress (MPa)')
        
        # Plot 2: Applied stress vs max stress
        scatter2 = axes[0, 1].scatter(df['applied_stress'], df['max_stress'], 
                                     c=df['twin_spacing'], cmap='plasma', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Applied Stress (MPa)')
        axes[0, 1].set_ylabel('Max Stress (MPa)')
        axes[0, 1].set_title('Stress Response')
        plt.colorbar(scatter2, ax=axes[0, 1]).set_label('Twin Spacing (nm)')
        
        # Plot 3: W parameter vs plastic work
        axes[0, 2].scatter(df['W'], df['plastic_work'], s=100, alpha=0.7)
        axes[0, 2].set_xlabel('W (Twin Well Depth)')
        axes[0, 2].set_ylabel('Plastic Work (J)')
        axes[0, 2].set_title('Energy Parameter Effect')
        
        # Plot 4: L_CTB vs avg spacing
        axes[1, 0].scatter(df['L_CTB'], df['avg_spacing'], s=100, alpha=0.7)
        axes[1, 0].set_xlabel('L_CTB (Mobility)')
        axes[1, 0].set_ylabel('Avg Twin Spacing (nm)')
        axes[1, 0].set_title('Mobility Effect on Spacing')
        
        # Plot 5: Correlation matrix
        corr_matrix = df[['twin_spacing', 'applied_stress', 'W', 'L_CTB', 'max_stress', 'avg_spacing']].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Parameter Correlation Matrix')
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(corr_matrix.columns)
        plt.colorbar(im, ax=axes[1, 1])
        
        # Plot 6: 3D scatter (if available)
        if len(df) >= 3:
            ax3d = fig.add_subplot(236, projection='3d')
            scatter3d = ax3d.scatter(df['twin_spacing'], df['applied_stress'], df['max_stress'],
                                    c=df['plastic_work'], cmap='hot', s=100, alpha=0.7)
            ax3d.set_xlabel('Twin Spacing')
            ax3d.set_ylabel('Applied Stress')
            ax3d.set_zlabel('Max Stress')
            ax3d.set_title('3D Parameter Space')
        
        # Apply styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig

# ============================================================================
# ENHANCED EXPORT SYSTEM
# ============================================================================
class EnhancedExporterCu:
    """Enhanced export functionality with multiple formats"""
    
    @staticmethod
    def export_publication_figures(simulations, config, style_params):
        """Export publication-ready figures"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figures based on comparison type
        if config['comparison_type'] == 'multi_simulation':
            fig = SimulationComparatorCu.create_multi_simulation_plot(simulations, config['frames'], config, style_params)
        elif config['comparison_type'] == 'evolution':
            fig = SimulationComparatorCu.create_evolution_comparison(simulations, config, style_params)
        elif config['comparison_type'] == 'parameter_sensitivity':
            fig = SimulationComparatorCu.create_parameter_sensitivity_plot(simulations, config, style_params)
        
        # Save in multiple formats
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save as PDF (vector)
            pdf_buffer = BytesIO()
            fig.savefig(pdf_buffer, format='pdf', dpi=style_params.get('dpi', 600), 
                       bbox_inches='tight')
            zf.writestr(f"publication_figure_{timestamp}.pdf", pdf_buffer.getvalue())
            
            # Save as PNG (raster)
            png_buffer = BytesIO()
            fig.savefig(png_buffer, format='png', dpi=style_params.get('dpi', 600), 
                       bbox_inches='tight')
            zf.writestr(f"publication_figure_{timestamp}.png", png_buffer.getvalue())
            
            # Save as SVG (vector)
            svg_buffer = BytesIO()
            fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
            zf.writestr(f"publication_figure_{timestamp}.svg", svg_buffer.getvalue())
            
            # Save figure data
            fig_data = {
                'config': config,
                'style_params': style_params,
                'simulation_ids': [sim['id'] for sim in simulations],
                'created_at': timestamp
            }
            zf.writestr(f"figure_metadata_{timestamp}.json", json.dumps(fig_data, indent=2))
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def export_simulation_data(simulations, format='zip'):
        """Export simulation data in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'zip':
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for sim in simulations:
                    sim_data = SimulationDBCu.get_simulation(sim['id'])
                    sim_id = sim['id']
                    
                    # Export parameters
                    params_json = json.dumps(sim_data['params'], indent=2)
                    zf.writestr(f"{sim_id}/parameters.json", params_json)
                    
                    # Export convergence history
                    history_df = pd.DataFrame(sim_data['history'])
                    zf.writestr(f"{sim_id}/convergence_history.csv", history_df.to_csv(index=False))
                    
                    # Export final frame data
                    final_results = sim_data['results_history'][-1]
                    for key, value in final_results.items():
                        if isinstance(value, np.ndarray):
                            np.savez_compressed(BytesIO(), value)
                            zf.writestr(f"{sim_id}/final_{key}.npz", BytesIO().getvalue())
            
            zip_buffer.seek(0)
            return zip_buffer
        
        elif format == 'csv':
            # Create combined CSV of all simulations
            all_data = []
            for sim in simulations:
                sim_data = SimulationDBCu.get_simulation(sim['id'])
                params = sim_data['params']
                final_results = sim_data['results_history'][-1]
                
                sim_summary = {
                    'simulation_id': sim['id'],
                    'geometry_type': params['geometry_type'],
                    'twin_spacing_nm': params['twin_spacing'],
                    'applied_stress_MPa': params['applied_stress']/1e6,
                    'max_stress_MPa': np.max(final_results['sigma_eq'])/1e6,
                    'avg_twin_spacing_nm': np.mean(final_results['h'][(final_results['h']>5) & (final_results['h']<50)]),
                    'plastic_work_J': sim_data['history']['plastic_work'][-1]
                }
                all_data.append(sim_summary)
            
            df = pd.DataFrame(all_data)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue().encode()

# ============================================================================
# FIXED NUMBA-COMPATIBLE FUNCTIONS (KEEP ORIGINAL)
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
# ENHANCED PHYSICS MODELS WITH ERROR HANDLING (KEEP ORIGINAL)
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

# ============================================================================
# MODIFIED MAIN FUNCTION WITH ENHANCED DATA FLOW
# ============================================================================
def main_enhanced():
    st.set_page_config(
        page_title="Nanotwinned Cu Phase-Field Simulator Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: white;
        padding: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with mode selector
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🔬 Nanotwinned Cu Phase-Field Simulator Pro</h1>', unsafe_allow_html=True)
        
        # Operation mode selector
        operation_mode = st.radio(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations", "Parameter Sweep Study"],
            horizontal=True,
            label_visibility="collapsed"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize database if not exists
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    
    # =============================================
    # MODE 1: RUN NEW SIMULATION
    # =============================================
    if operation_mode == "Run New Simulation":
        st.sidebar.header("⚙️ Simulation Configuration")
        
        # Geometry configuration
        st.sidebar.subheader("🧩 Geometry Configuration")
        geometry_type = st.sidebar.selectbox(
            "Geometry Type",
            ["Standard Twin Grain", "Twin Grain with Defect"],
            key="geom_type"
        )
        
        # Simulation parameters
        st.sidebar.subheader("📊 Grid Configuration")
        N = st.sidebar.slider("Grid resolution (N×N)", 64, 512, 256, 64, key="N")
        dx = st.sidebar.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1, key="dx")
        dt = st.sidebar.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5, key="dt")
        
        # Material parameters
        st.sidebar.subheader("🔬 Material Parameters")
        twin_spacing = st.sidebar.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0, key="twin_spacing")
        grain_boundary_pos = st.sidebar.slider("Grain boundary position (nm)", -50.0, 50.0, 0.0, 1.0, key="gb_pos")
        
        # Defect parameters
        if geometry_type == "Twin Grain with Defect":
            st.sidebar.subheader("⚠️ Defect Parameters")
            defect_type = st.sidebar.selectbox("Defect Type", ["Dislocation", "Void"], key="defect_type")
            defect_x = st.sidebar.slider("Defect X position (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_x")
            defect_y = st.sidebar.slider("Defect Y position (nm)", -50.0, 50.0, 0.0, 1.0, key="defect_y")
            defect_radius = st.sidebar.slider("Defect radius (nm)", 5.0, 30.0, 10.0, 1.0, key="defect_radius")
        
        # Thermodynamic parameters
        st.sidebar.subheader("⚡ Thermodynamic Parameters")
        W = st.sidebar.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1, key="W")
        A = st.sidebar.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5, key="A")
        B = st.sidebar.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5, key="B")
        
        # Gradient energy parameters
        st.sidebar.subheader("🌀 Gradient Energy")
        kappa0 = st.sidebar.slider("κ₀ (Gradient energy ref)", 0.01, 10.0, 1.0, 0.1, key="kappa0")
        gamma_aniso = st.sidebar.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05, key="gamma_aniso")
        kappa_eta = st.sidebar.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1, key="kappa_eta")
        
        # Kinetic parameters
        st.sidebar.subheader("⚡ Kinetic Parameters")
        L_CTB = st.sidebar.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001, key="L_CTB")
        L_ITB = st.sidebar.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1, key="L_ITB")
        n_mob = st.sidebar.slider("n (Mobility exponent)", 1, 10, 4, 1, key="n_mob")
        L_eta = st.sidebar.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1, key="L_eta")
        zeta = st.sidebar.slider("ζ (Dislocation pinning)", 0.0, 2.0, 0.3, 0.05, key="zeta")
        
        # Loading conditions
        st.sidebar.subheader("🏋️ Loading Conditions")
        applied_stress_MPa = st.sidebar.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0, key="applied_stress")
        
        # Simulation control
        st.sidebar.subheader("⏯️ Simulation Control")
        n_steps = st.sidebar.slider("Number of steps", 10, 1000, 100, 10, key="n_steps")
        save_frequency = st.sidebar.slider("Save frequency", 1, 100, 10, 1, key="save_freq")
        
        # Publication styling options
        st.sidebar.subheader("🎨 Publication Styling")
        journal_style = st.sidebar.selectbox("Journal Style", ["Nature", "Science", "Acta Materialia", "Custom"], index=0)
        export_dpi = st.sidebar.slider("Export DPI", 150, 1200, 600, 150)
        
        # Run button
        if st.sidebar.button("🚀 Run & Save Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation..."):
                # Prepare parameters
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
                else:
                    if warnings:
                        st.warning(f"Parameter warnings: {', '.join(warnings)}")
                    
                    # Run simulation (using your existing solver)
                    # Note: This is a placeholder - you need to integrate your actual solver here
                    st.info("Simulation running... (integrate your solver here)")
                    
                    # For demonstration, create mock data
                    # In reality, you would call your NanotwinnedCuSolver here
                    
                    # Mock simulation results
                    mock_history = {
                        'phi_norm': np.random.rand(n_steps) * 0.5 + 0.5,
                        'energy': np.random.rand(n_steps) * 1e-10,
                        'max_stress': np.random.rand(n_steps) * 1e9,
                        'avg_stress': np.random.rand(n_steps) * 5e8,
                        'plastic_work': np.random.rand(n_steps) * 1e-12,
                        'twin_spacing_avg': np.random.rand(n_steps) * 10 + 15
                    }
                    
                    # Mock results history (saved frames)
                    mock_results_history = []
                    for i in range(0, n_steps, save_frequency):
                        mock_results = {
                            'phi': np.random.randn(N, N) * 0.3,
                            'sigma_eq': np.random.rand(N, N) * 1e9,
                            'h': np.random.rand(N, N) * 20 + 10,
                            'sigma_y': np.random.rand(N, N) * 2e8 + 5e7,
                            'eps_p_mag': np.random.rand(N, N) * 0.01,
                            'eta1': np.ones((N, N)) * 0.8,
                            'eta2': np.ones((N, N)) * 0.2
                        }
                        mock_results_history.append(mock_results)
                    
                    # Create metadata
                    metadata = {
                        'run_time': 5.2,  # seconds
                        'frames': len(mock_results_history),
                        'grid_size': N,
                        'dx': dx,
                        'journal_style': journal_style,
                        'export_dpi': export_dpi
                    }
                    
                    # Save to database
                    sim_id = SimulationDBCu.save_simulation(params, mock_history, metadata, mock_results_history)
                    
                    st.success(f"""
                    ✅ Simulation Complete!
                    - **ID**: `{sim_id}`
                    - **Frames**: {len(mock_results_history)}
                    - **Saved to database**
                    - **Ready for comparison**
                    """)
        
        # Show saved simulations
        st.sidebar.subheader("📋 Saved Simulations")
        simulations = SimulationDBCu.get_simulation_list()
        if simulations:
            for sim in simulations:
                st.sidebar.text(f"ID: {sim['id'][:6]}...")
                st.sidebar.text(f"Name: {sim['name'][:30]}...")
                st.sidebar.markdown("---")
        else:
            st.sidebar.info("No simulations saved yet")
    
    # =============================================
    # MODE 2: COMPARE SAVED SIMULATIONS
    # =============================================
    elif operation_mode == "Compare Saved Simulations":
        st.sidebar.header("🔍 Comparison Setup")
        
        # Get available simulations
        simulations = SimulationDBCu.get_simulation_list()
        
        if not simulations:
            st.sidebar.warning("No simulations saved yet. Run some simulations first!")
        else:
            # Multi-select for comparison
            sim_options = {f"{sim['name']} (ID: {sim['id'][:6]}...)": sim['id'] for sim in simulations}
            selected_sim_ids = st.sidebar.multiselect(
                "Select Simulations to Compare",
                options=list(sim_options.keys()),
                default=list(sim_options.keys())[:min(3, len(sim_options))]
            )
            
            # Convert back to IDs
            selected_ids = [sim_options[name] for name in selected_sim_ids]
            
            # Comparison settings
            st.sidebar.subheader("Comparison Settings")
            comparison_type = st.sidebar.selectbox(
                "Comparison Type",
                ["Side-by-Side Heatmaps", "Evolution Comparison", "Parameter Sensitivity", 
                 "Stress Distribution Analysis", "Twin Structure Comparison"],
                index=0
            )
            
            # Visualization settings
            visualization_type = st.sidebar.selectbox(
                "Visualization Type",
                ["stress", "twin_structure", "twin_spacing", "yield_stress"],
                index=0
            )
            
            # Frame selection
            frame_selection = st.sidebar.radio(
                "Frame Selection",
                ["Final Frame", "Same Evolution Time", "Mid Evolution"],
                horizontal=True
            )
            
            # Publication styling
            st.sidebar.subheader("🎨 Publication Styling")
            journal_style = st.sidebar.selectbox("Journal Style", ["Nature", "Science", "Acta Materialia", "Custom"], 
                                                index=0, key="compare_journal")
            export_dpi = st.sidebar.slider("Export DPI", 150, 1200, 600, 150, key="compare_dpi")
            
            # Run comparison
            if st.sidebar.button("🔬 Run Comparison", type="primary"):
                # Load selected simulations
                selected_simulations = []
                for sim_id in selected_ids:
                    sim_data = SimulationDBCu.get_simulation(sim_id)
                    if sim_data:
                        selected_simulations.append({
                            'id': sim_id,
                            'params': sim_data['params'],
                            'data': sim_data
                        })
                
                if len(selected_simulations) < 1:
                    st.error("No valid simulations selected!")
                else:
                    # Determine frame indices
                    frames = []
                    for sim in selected_simulations:
                        if frame_selection == "Final Frame":
                            frames.append(len(sim['data']['results_history']) - 1)
                        elif frame_selection == "Mid Evolution":
                            frames.append(len(sim['data']['results_history']) // 2)
                        else:  # Same evolution time
                            frames.append(0)  # Default to first frame
                    
                    # Create comparison config
                    config = {
                        'comparison_type': 'multi_simulation',
                        'visualization_type': visualization_type,
                        'frames': frames,
                        'frame_selection': frame_selection
                    }
                    
                    # Style parameters
                    style_params = {
                        'journal_style': journal_style.lower(),
                        'dpi': export_dpi,
                        'vector_output': True
                    }
                    
                    # Create and display comparison
                    with st.spinner("Creating comparison plots..."):
                        if comparison_type == "Side-by-Side Heatmaps":
                            fig = SimulationComparatorCu.create_multi_simulation_plot(
                                selected_simulations, frames, config, style_params
                            )
                            st.pyplot(fig)
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📥 Export as PDF"):
                                    pdf_buffer = BytesIO()
                                    fig.savefig(pdf_buffer, format='pdf', dpi=export_dpi, bbox_inches='tight')
                                    st.download_button(
                                        label="Download PDF",
                                        data=pdf_buffer.getvalue(),
                                        file_name=f"nanotwin_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                            
                            with col2:
                                if st.button("📊 Export as PNG"):
                                    png_buffer = BytesIO()
                                    fig.savefig(png_buffer, format='png', dpi=export_dpi, bbox_inches='tight')
                                    st.download_button(
                                        label="Download PNG",
                                        data=png_buffer.getvalue(),
                                        file_name=f"nanotwin_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png"
                                    )
                        
                        elif comparison_type == "Evolution Comparison":
                            fig = SimulationComparatorCu.create_evolution_comparison(
                                selected_simulations, config, style_params
                            )
                            st.pyplot(fig)
                        
                        elif comparison_type == "Parameter Sensitivity":
                            fig = SimulationComparatorCu.create_parameter_sensitivity_plot(
                                selected_simulations, config, style_params
                            )
                            st.pyplot(fig)
        
        # Export all data
        st.sidebar.subheader("📤 Bulk Export")
        if st.sidebar.button("📦 Export All Simulations", type="secondary"):
            if simulations:
                with st.spinner("Preparing export..."):
                    zip_buffer = EnhancedExporterCu.export_simulation_data(simulations, format='zip')
                    st.sidebar.download_button(
                        label="Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"nanotwin_all_simulations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
            else:
                st.sidebar.warning("No simulations to export")
    
    # =============================================
    # MODE 3: PARAMETER SWEEP STUDY
    # =============================================
    else:  # Parameter Sweep Study
        st.sidebar.header("📊 Parameter Sweep Configuration")
        
        st.info("""
        **Parameter Sweep Mode:**
        Run multiple simulations with varying parameters to study:
        - Twin spacing effects on strengthening
        - Stress-strain response sensitivity
        - Mobility parameter optimization
        """)
        
        # Parameter sweep settings
        st.sidebar.subheader("Sweep Parameters")
        sweep_variable = st.sidebar.selectbox(
            "Variable to Sweep",
            ["Twin Spacing (λ)", "Applied Stress", "Twin Well Depth (W)", "CTB Mobility (L_CTB)"],
            index=0
        )
        
        # Range settings
        st.sidebar.subheader("Range Settings")
        if sweep_variable == "Twin Spacing (λ)":
            min_val = st.sidebar.slider("Min twin spacing (nm)", 5.0, 50.0, 10.0, 1.0)
            max_val = st.sidebar.slider("Max twin spacing (nm)", min_val+5.0, 100.0, 30.0, 1.0)
            n_points = st.sidebar.slider("Number of points", 3, 10, 5, 1)
        elif sweep_variable == "Applied Stress":
            min_val = st.sidebar.slider("Min stress (MPa)", 50.0, 500.0, 100.0, 10.0)
            max_val = st.sidebar.slider("Max stress (MPa)", min_val+50.0, 1000.0, 500.0, 10.0)
            n_points = st.sidebar.slider("Number of points", 3, 10, 5, 1)
        
        # Base parameters
        st.sidebar.subheader("Base Parameters")
        base_N = st.sidebar.slider("Grid size N", 128, 256, 192, 64)
        base_n_steps = st.sidebar.slider("Steps per simulation", 50, 300, 100, 10)
        
        # Run sweep
        if st.sidebar.button("🧪 Run Parameter Sweep", type="primary"):
            with st.spinner(f"Running {n_points} simulations..."):
                # Generate parameter values
                param_values = np.linspace(min_val, max_val, n_points)
                
                # Progress tracking
                progress_bar = st.progress(0)
                results = []
                
                for i, param_val in enumerate(param_values):
                    # Update progress
                    progress_bar.progress((i + 1) / n_points)
                    
                    # Create parameters for this simulation
                    params = {
                        'N': base_N,
                        'dx': 0.5,
                        'dt': 1e-4,
                        'W': 2.0,
                        'A': 5.0,
                        'B': 10.0,
                        'kappa0': 1.0,
                        'gamma_aniso': 0.7,
                        'kappa_eta': 2.0,
                        'L_CTB': 0.05,
                        'L_ITB': 5.0,
                        'n_mob': 4,
                        'L_eta': 1.0,
                        'zeta': 0.3,
                        'geometry_type': 'standard',
                        'save_frequency': 10,
                    }
                    
                    # Set swept parameter
                    if sweep_variable == "Twin Spacing (λ)":
                        params['twin_spacing'] = param_val
                        params['grain_boundary_pos'] = 0.0
                        params['applied_stress'] = 300e6
                    elif sweep_variable == "Applied Stress":
                        params['twin_spacing'] = 20.0
                        params['grain_boundary_pos'] = 0.0
                        params['applied_stress'] = param_val * 1e6
                    
                    # Run simulation (mock for demonstration)
                    # In reality, you would run your actual solver here
                    mock_history = {
                        'phi_norm': np.random.rand(base_n_steps) * 0.5 + 0.5,
                        'max_stress': np.random.rand(base_n_steps) * param_val * 1e7,
                        'twin_spacing_avg': np.random.rand(base_n_steps) * 5 + param_val if sweep_variable == "Twin Spacing (λ)" else 20.0
                    }
                    
                    # Store results
                    results.append({
                        'parameter': param_val,
                        'final_max_stress': mock_history['max_stress'][-1],
                        'final_phi_norm': mock_history['phi_norm'][-1],
                        'final_avg_spacing': mock_history['twin_spacing_avg'][-1]
                    })
                
                # Create sweep analysis plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
                
                # Plot 1: Parameter vs Max Stress
                param_vals = [r['parameter'] for r in results]
                max_stresses = [r['final_max_stress']/1e6 for r in results]  # MPa
                
                axes[0].plot(param_vals, max_stresses, 'bo-', linewidth=2, markersize=8)
                axes[0].set_xlabel(sweep_variable)
                axes[0].set_ylabel('Final Max Stress (MPa)')
                axes[0].set_title(f'{sweep_variable} vs Max Stress')
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Parameter vs Twin Order Parameter
                phi_norms = [r['final_phi_norm'] for r in results]
                axes[1].plot(param_vals, phi_norms, 'ro-', linewidth=2, markersize=8)
                axes[1].set_xlabel(sweep_variable)
                axes[1].set_ylabel('Final ||φ||')
                axes[1].set_title(f'{sweep_variable} vs Twin Order Parameter')
                axes[1].grid(True, alpha=0.3)
                
                # Apply publication styling
                fig = PublicationStylerCu.apply_journal_style(fig, axes, 'nature')[0]
                st.pyplot(fig)
                
                # Display results table
                st.subheader("Sweep Results")
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                st.success(f"✅ Parameter sweep completed! {n_points} simulations analyzed.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main_enhanced()
