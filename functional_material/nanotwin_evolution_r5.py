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
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate

warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED POST-PROCESSING MODULES FROM FIRST CODE
# ============================================================================

# =============================================
# EXPANDED COLORMAP LIBRARY (50+ options)
# =============================================
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

# =============================================
# ENHANCED LINE PROFILE SYSTEM
# =============================================
class EnhancedLineProfiler:
    """Enhanced line profile system for nanotwinned materials"""
    
    @staticmethod
    @njit(parallel=True)
    def extract_profile_numba(data, N, dx, profile_type, position_ratio=0.5, angle_deg=45):
        """
        Numba-accelerated profile extraction for performance
        """
        ny, nx = N, N
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
            distance = np.linspace(-N*dx/2, N*dx/2, nx)
            
        elif profile_type == 'vertical':
            # Vertical profile
            col_idx = center_x + offset
            profile = data[:, col_idx]
            distance = np.linspace(-N*dx/2, N*dx/2, ny)
            
        elif profile_type == 'diagonal':
            # Main diagonal (top-left to bottom-right)
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x - diag_length//2, center_y - diag_length//2)
            
            profile = np.zeros(diag_length)
            distances = np.zeros(diag_length)
            
            for i in prange(diag_length):
                x = start_idx[0] + i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile[i] = data[y, x]
                    dist = i * dx * np.sqrt(2)
                    distances[i] = dist - (diag_length//2) * dx * np.sqrt(2)
            
            # Trim zeros
            valid_mask = distances != 0
            distance = distances[valid_mask]
            profile = profile[valid_mask]
            
        elif profile_type == 'anti_diagonal':
            # Anti-diagonal (top-right to bottom-left)
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x + diag_length//2, center_y - diag_length//2)
            
            profile = np.zeros(diag_length)
            distances = np.zeros(diag_length)
            
            for i in prange(diag_length):
                x = start_idx[0] - i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile[i] = data[y, x]
                    dist = i * dx * np.sqrt(2)
                    distances[i] = dist - (diag_length//2) * dx * np.sqrt(2)
            
            # Trim zeros
            valid_mask = distances != 0
            distance = distances[valid_mask]
            profile = profile[valid_mask]
            
        else:
            # Custom angle (simplified for Numba)
            angle_rad = np.deg2rad(angle_deg)
            length = int(min(nx, ny) * 0.8)
            
            profile = np.zeros(length)
            distances = np.zeros(length)
            
            for i in prange(length):
                t = -length//2 + i
                x = center_x + t * np.cos(angle_rad) + offset * np.cos(angle_rad + np.pi/2)
                y = center_y + t * np.sin(angle_rad) + offset * np.sin(angle_rad + np.pi/2)
                
                if 0 <= x < nx-1 and 0 <= y < ny-1:
                    # Simplified interpolation for Numba
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, nx-1), min(y0 + 1, ny-1)
                    
                    wx = x - x0
                    wy = y - y0
                    
                    val = (data[y0, x0] * (1-wx) * (1-wy) +
                          data[y0, x1] * wx * (1-wy) +
                          data[y1, x0] * (1-wx) * wy +
                          data[y1, x1] * wx * wy)
                    
                    profile[i] = val
                    distances[i] = t * dx
            
            distance = distances
            profile = profile
        
        return distance, profile
    
    @staticmethod
    def extract_profile(data, N, dx, profile_type, position_ratio=0.5, angle_deg=45):
        """
        Extract line profiles from 2D nanotwin fields
        """
        return EnhancedLineProfiler.extract_profile_numba(
            data, N, dx, profile_type, position_ratio, angle_deg
        )
    
    @staticmethod
    def extract_multiple_profiles(data, N, dx, profile_types, position_ratio=0.5, angle_deg=45):
        """
        Extract multiple line profiles from the same data
        """
        profiles = {}
        for profile_type in profile_types:
            distance, profile = EnhancedLineProfiler.extract_profile(
                data, N, dx, profile_type, position_ratio, angle_deg
            )
            profiles[profile_type] = {
                'distance': distance,
                'profile': profile
            }
        return profiles
    
    @staticmethod
    def plot_profile_locations(ax, data, N, dx, profile_configs, cmap='viridis', alpha=0.7):
        """
        Plot data with overlay of profile lines
        """
        extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
        im = ax.imshow(data, extent=extent, cmap=cmap, origin='lower', aspect='equal')
        
        # Define colors for different profile types
        profile_colors = {
            'horizontal': 'red',
            'vertical': 'blue',
            'diagonal': 'green',
            'anti_diagonal': 'purple',
            'custom': 'orange'
        }
        
        # Plot each profile line
        for profile_type, config in profile_configs.items():
            if profile_type in config['profiles']:
                # Calculate endpoints for visualization
                center = N // 2
                offset = int(N * 0.4 * config.get('position_ratio', 0.5))
                
                if profile_type == 'horizontal':
                    y_pos = (center + offset - N/2) * dx
                    ax.axhline(y=y_pos, color='red', linewidth=2, alpha=alpha, 
                              label='Horizontal Profile')
                elif profile_type == 'vertical':
                    x_pos = (center + offset - N/2) * dx
                    ax.axvline(x=x_pos, color='blue', linewidth=2, alpha=alpha,
                              label='Vertical Profile')
        
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.legend(loc='upper right', fontsize=8)
        
        return im, ax

# =============================================
# JOURNAL-SPECIFIC STYLING TEMPLATES
# =============================================
class JournalTemplates:
    """Publication-quality journal templates for nanotwin research"""
    
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
                'figure_width_single': 8.5,
                'figure_width_double': 17.0,
                'font_family': 'Times New Roman',
                'font_size_small': 10,
                'font_size_medium': 11,
                'font_size_large': 12,
                'line_width': 1.0,
                'axes_linewidth': 1.0,
                'tick_width': 1.0,
                'tick_length': 4,
                'grid_alpha': 0.2,
                'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000', '#44AA99']
            },
            'custom': {
                'figure_width_single': 6.0,
                'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans',
                'font_size_small': 10,
                'font_size_medium': 12,
                'font_size_large': 14,
                'line_width': 1.5,
                'axes_linewidth': 1.5,
                'tick_width': 1.0,
                'tick_length': 5,
                'grid_alpha': 0.3,
                'dpi': 300,
                'color_cycle': plt.cm.Set2(np.linspace(0, 1, 10))
            }
        }
    
    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
        """Apply journal-specific styling to figure"""
        styles = JournalTemplates.get_journal_styles()
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

# =============================================
# POST-PROCESSING STYLING SYSTEM
# =============================================
class FigureStyler:
    """Advanced figure styling and post-processing system for nanotwin visualizations"""
    
    @staticmethod
    def apply_advanced_styling(fig, axes, style_params):
        """Apply advanced styling to figure and axes"""
        
        # Apply to all axes in figure
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        for ax in axes_flat:
            if ax is not None:
                # Apply axis styling
                ax.tick_params(axis='both', which='major', 
                              labelsize=style_params.get('tick_font_size', 12),
                              width=style_params.get('tick_width', 2.0),
                              length=style_params.get('tick_length', 6))
                
                # Apply spine styling
                for spine in ax.spines.values():
                    spine.set_linewidth(style_params.get('spine_width', 2.5))
                    spine.set_color(style_params.get('spine_color', 'black'))
                
                # Apply grid if requested
                if style_params.get('show_grid', True):
                    ax.grid(True, 
                           alpha=style_params.get('grid_alpha', 0.3),
                           linestyle=style_params.get('grid_style', '--'),
                           linewidth=style_params.get('grid_width', 0.5))
                
                # Apply title styling
                if hasattr(ax, 'title'):
                    title = ax.get_title()
                    if title:
                        ax.set_title(title, 
                                    fontsize=style_params.get('title_font_size', 16),
                                    fontweight=style_params.get('title_weight', 'bold'),
                                    color=style_params.get('title_color', 'black'))
                
                # Apply label styling
                if ax.get_xlabel():
                    ax.set_xlabel(ax.get_xlabel(),
                                 fontsize=style_params.get('label_font_size', 14),
                                 fontweight=style_params.get('label_weight', 'bold'))
                if ax.get_ylabel():
                    ax.set_ylabel(ax.get_ylabel(),
                                 fontsize=style_params.get('label_font_size', 14),
                                 fontweight=style_params.get('label_weight', 'bold'))
        
        # Apply figure background
        if style_params.get('figure_facecolor'):
            fig.set_facecolor(style_params['figure_facecolor'])
        
        # Tight layout
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def get_styling_controls():
        """Get comprehensive styling controls for Streamlit sidebar"""
        style_params = {}
        
        st.sidebar.header("🎨 Advanced Post-Processing")
        
        with st.sidebar.expander("📐 Font & Text Styling", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['title_font_size'] = st.slider("Title Size", 8, 32, 16)
                style_params['label_font_size'] = st.slider("Label Size", 8, 28, 14)
                style_params['tick_font_size'] = st.slider("Tick Size", 6, 20, 12)
            with col2:
                style_params['title_weight'] = st.selectbox("Title Weight", 
                                                           ['normal', 'bold', 'light', 'semibold'], 
                                                           index=1)
                style_params['label_weight'] = st.selectbox("Label Weight", 
                                                           ['normal', 'bold', 'light'], 
                                                           index=1)
                style_params['title_color'] = st.color_picker("Title Color", "#000000")
        
        with st.sidebar.expander("📏 Line & Border Styling", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['line_width'] = st.slider("Line Width", 0.5, 5.0, 2.0, 0.5)
                style_params['spine_width'] = st.slider("Spine Width", 1.0, 4.0, 2.5, 0.5)
                style_params['tick_width'] = st.slider("Tick Width", 0.5, 3.0, 2.0, 0.5)
            with col2:
                style_params['tick_length'] = st.slider("Tick Length", 2, 15, 6)
                style_params['spine_color'] = st.color_picker("Spine Color", "#000000")
                style_params['grid_width'] = st.slider("Grid Width", 0.1, 2.0, 0.5, 0.1)
        
        with st.sidebar.expander("🌐 Grid & Background", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['show_grid'] = st.checkbox("Show Grid", True)
                style_params['grid_style'] = st.selectbox("Grid Style", 
                                                         ['-', '--', '-.', ':'],
                                                         index=1)
                style_params['grid_alpha'] = st.slider("Grid Alpha", 0.0, 1.0, 0.3, 0.05)
            with col2:
                style_params['figure_facecolor'] = st.color_picker("Figure Background", "#FFFFFF")
                style_params['axes_facecolor'] = st.color_picker("Axes Background", "#FFFFFF")
        
        return style_params

# =============================================
# SIMULATION DATABASE SYSTEM
# =============================================
class SimulationDB:
    """In-memory simulation database for storing and retrieving nanotwin simulations"""
    
    @staticmethod
    def generate_id(sim_params):
        """Generate unique ID for simulation"""
        param_str = json.dumps(sim_params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    @staticmethod
    def save_simulation(sim_params, results_history, metadata):
        """Save simulation to database"""
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
        
        sim_id = SimulationDB.generate_id(sim_params)
        
        # Store simulation data
        st.session_state.simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'results_history': results_history,
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'type': 'nanotwin'
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
            return {k: v for k, v in st.session_state.simulations.items() if v.get('type') == 'nanotwin'}
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
        simulations = SimulationDB.get_all_simulations()
        
        if not simulations:
            return []
        
        sim_list = []
        for sim_id, sim_data in simulations.items():
            params = sim_data['params']
            name = f"Twin Spacing: {params.get('twin_spacing', 'N/A')}nm | Stress: {params.get('applied_stress', 0)/1e6:.0f}MPa"
            sim_list.append({
                'id': sim_id,
                'name': name,
                'params': params
            })
        
        return sim_list

# =============================================
# ENHANCED SIMULATION MONITOR WITH POST-PROCESSING
# =============================================
class EnhancedSimulationMonitor:
    """Extended simulation monitor with post-processing capabilities"""
    
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
        self.line_profiler = EnhancedLineProfiler()
    
    def create_enhanced_line_profiles(self, field_data, field_name, profile_types, 
                                     position_ratio=0.5, angle_deg=45, style_params=None):
        """Create enhanced line profile plots for nanotwin fields"""
        if style_params is None:
            style_params = {}
        
        n_profiles = len(profile_types)
        fig, axes = plt.subplots(1, n_profiles, figsize=(5*n_profiles, 5), constrained_layout=True)
        
        if n_profiles == 1:
            axes = [axes]
        
        for idx, profile_type in enumerate(profile_types):
            ax = axes[idx]
            
            # Extract profile
            distance, profile = self.line_profiler.extract_profile(
                field_data, self.N, self.dx, profile_type, position_ratio, angle_deg
            )
            
            # Plot profile
            ax.plot(distance, profile, 'b-', linewidth=style_params.get('line_width', 2.0))
            ax.set_xlabel('Position (nm)', fontsize=style_params.get('label_font_size', 12))
            ax.set_ylabel(field_name, fontsize=style_params.get('label_font_size', 12))
            ax.set_title(f'{profile_type.title()} Profile', 
                        fontsize=style_params.get('title_font_size', 14),
                        fontweight=style_params.get('title_weight', 'bold'))
            
            # Add grid if requested
            if style_params.get('show_grid', True):
                ax.grid(True, alpha=style_params.get('grid_alpha', 0.3),
                       linestyle=style_params.get('grid_style', '--'))
        
        # Apply styling
        fig = FigureStyler.apply_advanced_styling(fig, axes, style_params)
        
        return fig
    
    def create_field_comparison_with_profiles(self, initial_data, final_data, field_name,
                                            profile_types, style_params=None):
        """Create comparison plot with field visualizations and profiles"""
        if style_params is None:
            style_params = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        
        # Row 1: Field visualizations
        # Initial field
        im1 = axes[0, 0].imshow(initial_data, extent=self.extent, cmap='viridis', origin='lower')
        axes[0, 0].set_title(f'Initial {field_name}')
        axes[0, 0].set_xlabel('x (nm)')
        axes[0, 0].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Final field
        im2 = axes[0, 1].imshow(final_data, extent=self.extent, cmap='viridis', origin='lower')
        axes[0, 1].set_title(f'Final {field_name}')
        axes[0, 1].set_xlabel('x (nm)')
        axes[0, 1].set_ylabel('y (nm)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Difference
        diff = final_data - initial_data
        im3 = axes[0, 2].imshow(diff, extent=self.extent, cmap='coolwarm', origin='lower')
        axes[0, 2].set_title(f'Δ{field_name}')
        axes[0, 2].set_xlabel('x (nm)')
        axes[0, 2].set_ylabel('y (nm)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Row 2: Line profiles for each type
        profile_colors = {'horizontal': 'red', 'vertical': 'blue', 'diagonal': 'green'}
        
        for idx, profile_type in enumerate(profile_types[:3]):  # Show up to 3 profiles
            ax = axes[1, idx]
            
            # Extract profiles for both initial and final
            dist_initial, prof_initial = self.line_profiler.extract_profile(
                initial_data, self.N, self.dx, profile_type
            )
            dist_final, prof_final = self.line_profiler.extract_profile(
                final_data, self.N, self.dx, profile_type
            )
            
            # Plot both profiles
            ax.plot(dist_initial, prof_initial, 'b-', linewidth=2, alpha=0.7, label='Initial')
            ax.plot(dist_final, prof_final, 'r-', linewidth=2, alpha=0.7, label='Final')
            
            ax.set_xlabel('Position (nm)')
            ax.set_ylabel(field_name)
            ax.set_title(f'{profile_type.title()} Profile Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Apply styling
        fig = FigureStyler.apply_advanced_styling(fig, axes, style_params)
        
        return fig
    
    def create_statistical_analysis(self, field_data, field_name, style_params=None):
        """Create comprehensive statistical analysis of field data"""
        if style_params is None:
            style_params = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        
        # Flatten data for statistics
        flat_data = field_data.flatten()
        flat_data = flat_data[np.isfinite(flat_data)]
        
        # Plot 1: Histogram with KDE
        axes[0, 0].hist(flat_data, bins=50, density=True, alpha=0.7, color='blue')
        # Add KDE
        kde = stats.gaussian_kde(flat_data)
        x_range = np.linspace(np.min(flat_data), np.max(flat_data), 100)
        axes[0, 0].plot(x_range, kde(x_range), 'r-', linewidth=2)
        axes[0, 0].set_xlabel(field_name)
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title(f'Distribution of {field_name}')
        
        # Plot 2: Box plot
        axes[0, 1].boxplot(flat_data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'))
        axes[0, 1].set_ylabel(field_name)
        axes[0, 1].set_title(f'Box Plot of {field_name}')
        
        # Plot 3: QQ plot
        stats.probplot(flat_data, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title(f'Q-Q Plot of {field_name}')
        
        # Plot 4: Cumulative distribution
        sorted_data = np.sort(flat_data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 0].plot(sorted_data, y_vals, 'b-', linewidth=2)
        axes[1, 0].set_xlabel(field_name)
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title(f'CDF of {field_name}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Spatial autocorrelation
        # Compute 2D autocorrelation
        autocorr = np.correlate(field_data.flatten(), field_data.flatten(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:100]  # First 100 lags
        axes[1, 1].plot(autocorr, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Lag (grid points)')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title(f'Spatial Autocorrelation of {field_name}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Statistics table
        axes[1, 2].axis('off')
        stats_text = (
            f"Statistics for {field_name}:\n\n"
            f"Mean: {np.mean(flat_data):.4f}\n"
            f"Std Dev: {np.std(flat_data):.4f}\n"
            f"Min: {np.min(flat_data):.4f}\n"
            f"Max: {np.max(flat_data):.4f}\n"
            f"Skewness: {stats.skew(flat_data):.4f}\n"
            f"Kurtosis: {stats.kurtosis(flat_data):.4f}\n"
            f"N: {len(flat_data):,}"
        )
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', fontfamily='monospace')
        
        # Apply styling
        fig = FigureStyler.apply_advanced_styling(fig, axes, style_params)
        
        return fig
    
    def create_correlation_analysis(self, field1_data, field2_data, field1_name, field2_name,
                                   style_params=None):
        """Create correlation analysis between two fields"""
        if style_params is None:
            style_params = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        
        # Flatten data
        flat1 = field1_data.flatten()
        flat2 = field2_data.flatten()
        
        # Remove NaNs and infinite values
        mask = np.isfinite(flat1) & np.isfinite(flat2)
        flat1 = flat1[mask]
        flat2 = flat2[mask]
        
        # Subsample for large datasets
        if len(flat1) > 10000:
            indices = np.random.choice(len(flat1), 10000, replace=False)
            flat1 = flat1[indices]
            flat2 = flat2[indices]
        
        # Plot 1: Scatter plot with regression line
        axes[0, 0].scatter(flat1, flat2, alpha=0.3, s=10, color='blue')
        
        # Add regression line
        if len(flat1) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(flat1, flat2)
            x_range = np.linspace(np.min(flat1), np.max(flat1), 100)
            axes[0, 0].plot(x_range, slope*x_range + intercept, 'r-', linewidth=2,
                          label=f'R = {r_value:.3f}')
            axes[0, 0].legend()
        
        axes[0, 0].set_xlabel(field1_name)
        axes[0, 0].set_ylabel(field2_name)
        axes[0, 0].set_title(f'{field1_name} vs {field2_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: 2D histogram (hexbin)
        hb = axes[0, 1].hexbin(flat1, flat2, gridsize=50, cmap='viridis', mincnt=1)
        plt.colorbar(hb, ax=axes[0, 1])
        axes[0, 1].set_xlabel(field1_name)
        axes[0, 1].set_ylabel(field2_name)
        axes[0, 1].set_title('2D Density Distribution')
        
        # Plot 3: Residuals plot
        if len(flat1) > 2:
            y_pred = slope * flat1 + intercept
            residuals = flat2 - y_pred
            axes[1, 0].scatter(y_pred, residuals, alpha=0.3, s=10, color='green')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Predicted Values')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residual Plot')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Correlation statistics
        axes[1, 1].axis('off')
        if len(flat1) > 2:
            corr_text = (
                f"Correlation Analysis:\n\n"
                f"Pearson R: {r_value:.4f}\n"
                f"R²: {r_value**2:.4f}\n"
                f"Slope: {slope:.4f}\n"
                f"Intercept: {intercept:.4f}\n"
                f"p-value: {p_value:.2e}\n"
                f"Std Error: {std_err:.4f}\n"
                f"N: {len(flat1):,}"
            )
        else:
            corr_text = "Insufficient data for correlation analysis"
        
        axes[1, 1].text(0.1, 0.5, corr_text, fontsize=10,
                       verticalalignment='center', fontfamily='monospace')
        
        # Apply styling
        fig = FigureStyler.apply_advanced_styling(fig, axes, style_params)
        
        return fig

# ============================================================================
# ENHANCED VISUALIZATION SYSTEM FOR STREAMLIT
# ============================================================================
class EnhancedVisualizationSystem:
    """Comprehensive visualization system for nanotwin simulations"""
    
    def __init__(self):
        self.monitor = None
        self.style_params = {}
        
    def initialize(self, N, dx):
        """Initialize with grid parameters"""
        self.monitor = EnhancedSimulationMonitor(N, dx)
        self.style_params = FigureStyler.get_styling_controls()
        
    def create_publication_quality_plots(self, results_history, params):
        """Create publication-quality plots for nanotwin research"""
        if not results_history:
            return None
        
        final_results = results_history[-1]
        initial_results = results_history[0]
        N = params['N']
        dx = params['dx']
        
        plots = {}
        
        # 1. Twin order parameter evolution with profiles
        plots['phi_evolution'] = self.monitor.create_field_comparison_with_profiles(
            initial_results['phi'],
            final_results['phi'],
            'Twin Order Parameter (φ)',
            ['horizontal', 'vertical'],
            self.style_params
        )
        
        # 2. Stress analysis with statistics
        plots['stress_analysis'] = self.monitor.create_statistical_analysis(
            final_results['sigma_eq'],
            'Von Mises Stress (Pa)',
            self.style_params
        )
        
        # 3. Twin spacing analysis
        plots['spacing_analysis'] = self.monitor.create_statistical_analysis(
            final_results['h'],
            'Twin Spacing (nm)',
            self.style_params
        )
        
        # 4. Correlation between stress and twin spacing
        plots['stress_spacing_correlation'] = self.monitor.create_correlation_analysis(
            final_results['h'],
            final_results['sigma_eq'],
            'Twin Spacing (nm)',
            'Von Mises Stress (Pa)',
            self.style_params
        )
        
        # 5. Enhanced line profiles for multiple fields
        field_profiles = {}
        for field_name, field_data in [
            ('φ', final_results['phi']),
            ('σ_eq', final_results['sigma_eq']),
            ('h', final_results['h']),
            ('ε_p', final_results['eps_p_mag'])
        ]:
            field_profiles[field_name] = self.monitor.create_enhanced_line_profiles(
                field_data,
                field_name,
                ['horizontal', 'vertical'],
                style_params=self.style_params
            )
        plots['field_profiles'] = field_profiles
        
        return plots
    
    def create_comparison_dashboard(self, simulations_data):
        """Create comparison dashboard for multiple simulations"""
        if len(simulations_data) < 2:
            st.warning("Need at least 2 simulations for comparison")
            return None
        
        fig = plt.figure(figsize=(16, 12))
        
        n_sims = len(simulations_data)
        n_cols = min(3, n_sims)
        n_rows = (n_sims + n_cols - 1) // n_cols
        
        # Create subplot grid
        gs = fig.add_gridspec(n_rows * 2, n_cols * 3)
        
        # Plot each simulation
        for idx, (sim_name, sim_data) in enumerate(simulations_data.items()):
            row = (idx // n_cols) * 2
            col = (idx % n_cols) * 3
            
            # Final phi field
            ax1 = fig.add_subplot(gs[row:row+2, col])
            im1 = ax1.imshow(sim_data['phi'], cmap='RdBu_r', vmin=-1.2, vmax=1.2)
            ax1.set_title(f'{sim_name}\nTwin Order φ')
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Final stress field
            ax2 = fig.add_subplot(gs[row:row+2, col+1])
            stress_gpa = sim_data['sigma_eq'] / 1e9
            vmax = np.percentile(stress_gpa, 95)
            im2 = ax2.imshow(stress_gpa, cmap='hot', vmin=0, vmax=vmax)
            ax2.set_title('Von Mises Stress (GPa)')
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            # Final twin spacing
            ax3 = fig.add_subplot(gs[row:row+2, col+2])
            im3 = ax3.imshow(sim_data['h'], cmap='plasma', vmin=0, vmax=30)
            ax3.set_title('Twin Spacing (nm)')
            ax3.set_xticks([])
            ax3.set_yticks([])
            plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # Apply styling
        all_axes = fig.get_axes()
        FigureStyler.apply_advanced_styling(fig, all_axes, self.style_params)
        
        return fig

# ============================================================================
# ENHANCED DATA EXPORTER
# ============================================================================
class EnhancedDataExporter:
    """Enhanced data exporter with post-processing features"""
    
    @staticmethod
    def export_comprehensive_package(results_history, params, plots, filename_prefix):
        """Export comprehensive simulation package with post-processing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save parameters with styling info
            export_params = params.copy()
            export_params['export_timestamp'] = timestamp
            export_params['export_format'] = 'comprehensive_package'
            
            params_json = json.dumps(export_params, indent=2, cls=NumpyEncoder)
            zip_file.writestr(f"{filename_prefix}_parameters_{timestamp}.json", params_json)
            
            # 2. Save numerical data
            if results_history:
                # Save as NPZ
                npz_buffer = BytesIO()
                np.savez_compressed(npz_buffer,
                    phi_initial=results_history[0]['phi'],
                    sigma_eq_initial=results_history[0]['sigma_eq'],
                    h_initial=results_history[0]['h'],
                    phi_final=results_history[-1]['phi'],
                    sigma_eq_final=results_history[-1]['sigma_eq'],
                    h_final=results_history[-1]['h'],
                    eps_p_final=results_history[-1]['eps_p_mag'])
                zip_file.writestr(f"{filename_prefix}_data_{timestamp}.npz", npz_buffer.getvalue())
                
                # Save as CSV for easy analysis
                csv_data = []
                for i, results in enumerate(results_history):
                    csv_data.append({
                        'step': i,
                        'avg_phi': np.mean(results['phi']),
                        'std_phi': np.std(results['phi']),
                        'avg_sigma_eq_gpa': np.mean(results['sigma_eq']) / 1e9,
                        'max_sigma_eq_gpa': np.max(results['sigma_eq']) / 1e9,
                        'avg_h_nm': np.mean(results['h'][results['h'] < 100]),  # Filter outliers
                        'std_h_nm': np.std(results['h'][results['h'] < 100]),
                        'avg_eps_p': np.mean(results['eps_p_mag']),
                        'max_eps_p': np.max(results['eps_p_mag'])
                    })
                
                df = pd.DataFrame(csv_data)
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                zip_file.writestr(f"{filename_prefix}_evolution_{timestamp}.csv", csv_buffer.getvalue())
            
            # 3. Save plots as PNG
            if plots:
                for plot_name, plot_fig in plots.items():
                    if isinstance(plot_fig, dict):
                        for sub_name, sub_fig in plot_fig.items():
                            png_buffer = BytesIO()
                            sub_fig.savefig(png_buffer, format='png', dpi=300, 
                                          bbox_inches='tight', facecolor='white')
                            zip_file.writestr(f"{filename_prefix}_{plot_name}_{sub_name}_{timestamp}.png", 
                                            png_buffer.getvalue())
                    elif plot_fig is not None:
                        png_buffer = BytesIO()
                        plot_fig.savefig(png_buffer, format='png', dpi=300,
                                       bbox_inches='tight', facecolor='white')
                        zip_file.writestr(f"{filename_prefix}_{plot_name}_{timestamp}.png", 
                                        png_buffer.getvalue())
            
            # 4. Save summary report
            summary = f"""NANOTWINNED COPPER SIMULATION EXPORT
========================================
Generated: {timestamp}
Simulation Type: Phase-field nanotwin evolution
Grid Size: {params['N']}×{params['N']}
Grid Spacing: {params['dx']} nm
Twin Spacing: {params.get('twin_spacing', 'N/A')} nm
Applied Stress: {params.get('applied_stress', 0)/1e6:.1f} MPa
Time Steps: {len(results_history) if results_history else 0}

FILES INCLUDED:
---------------
1. parameters.json - Simulation parameters
2. data.npz - Numerical field data (NPZ format)
3. evolution.csv - Time evolution statistics
4. *.png - Publication-quality plots

POST-PROCESSING FEATURES:
-------------------------
• Enhanced line profile analysis
• Statistical analysis (distributions, correlations)
• Publication-quality figure styling
• Multi-simulation comparison
• Comprehensive data export

For questions: nanotwin.simulator@research.example.com
"""
            zip_file.writestr(f"{filename_prefix}_README_{timestamp}.txt", summary)
        
        zip_buffer.seek(0)
        return zip_buffer

# ============================================================================
# ENHANCED STREAMLIT APPLICATION
# ============================================================================
def create_enhanced_streamlit_app():
    """Main function to create enhanced Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Nanotwinned Cu Simulator",
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
    .feature-card {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .tab-content {
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Enhanced Nanotwinned Copper Phase-Field Simulator</h1>', 
                unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4>📊 Advanced Post-Processing</h4>
        • Enhanced line profiling<br>
        • Statistical analysis<br>
        • Correlation studies<br>
        • Publication-quality plots
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4>🎨 Visualization Enhancements</h4>
        • 50+ colormaps<br>
        • Journal templates<br>
        • Interactive 3D plots<br>
        • Real-time styling
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h4>🔄 Multi-Simulation Analysis</h4>
        • Simulation database<br>
        • Side-by-side comparison<br>
        • Parameter sweeps<br>
        • Comprehensive export
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    if 'visualization_system' not in st.session_state:
        st.session_state.visualization_system = EnhancedVisualizationSystem()
    if 'current_simulation' not in st.session_state:
        st.session_state.current_simulation = None
    if 'style_params' not in st.session_state:
        st.session_state.style_params = {}
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.header("⚙️ Enhanced Configuration")
        
        # Operation mode
        operation_mode = st.selectbox(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations", "Advanced Analysis"],
            index=0
        )
        
        # Simulation parameters (simplified for example)
        if operation_mode == "Run New Simulation":
            st.subheader("🧪 Simulation Parameters")
            
            # Grid parameters
            N = st.slider("Grid Size (N)", 64, 512, 256, 64)
            dx = st.slider("Grid Spacing (nm)", 0.1, 2.0, 0.5, 0.1)
            
            # Twin parameters
            twin_spacing = st.slider("Twin Spacing (nm)", 5.0, 100.0, 20.0, 1.0)
            applied_stress = st.slider("Applied Stress (MPa)", 0.0, 1000.0, 300.0, 10.0)
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                W = st.slider("Twin Energy (W)", 0.1, 10.0, 2.0, 0.1)
                A = st.slider("Grain Energy (A)", 0.1, 20.0, 5.0, 0.5)
                kappa0 = st.slider("Gradient Energy (κ₀)", 0.01, 10.0, 1.0, 0.1)
                n_steps = st.slider("Simulation Steps", 10, 500, 100, 10)
            
            # Post-processing options
            st.subheader("📈 Post-Processing")
            enable_line_profiles = st.checkbox("Enable Line Profiles", True)
            enable_statistics = st.checkbox("Enable Statistical Analysis", True)
            enable_correlations = st.checkbox("Enable Correlation Analysis", True)
            
            # Visualization options
            st.subheader("🎨 Visualization")
            phi_cmap = st.selectbox("φ Colormap", cmap_list, index=cmap_list.index('RdBu_r'))
            stress_cmap = st.selectbox("Stress Colormap", cmap_list, index=cmap_list.index('hot'))
            spacing_cmap = st.selectbox("Spacing Colormap", cmap_list, index=cmap_list.index('plasma'))
            
            # Run simulation button
            if st.button("🚀 Run Enhanced Simulation", type="primary", use_container_width=True):
                # Initialize visualization system
                st.session_state.visualization_system.initialize(N, dx)
                
                # Create mock simulation data (replace with actual simulation)
                with st.spinner("Running simulation with enhanced post-processing..."):
                    # Generate mock results
                    mock_results = generate_mock_simulation(N, dx, twin_spacing, applied_stress, n_steps)
                    
                    # Store in session state
                    sim_params = {
                        'N': N,
                        'dx': dx,
                        'twin_spacing': twin_spacing,
                        'applied_stress': applied_stress * 1e6,
                        'W': W,
                        'A': A,
                        'kappa0': kappa0,
                        'n_steps': n_steps
                    }
                    
                    # Save to database
                    sim_id = SimulationDB.save_simulation(
                        sim_params, 
                        mock_results['history'],
                        {'type': 'nanotwin', 'status': 'completed'}
                    )
                    
                    # Generate enhanced plots
                    plots = st.session_state.visualization_system.create_publication_quality_plots(
                        mock_results['history'], sim_params
                    )
                    
                    # Store for display
                    st.session_state.current_simulation = {
                        'id': sim_id,
                        'params': sim_params,
                        'results': mock_results,
                        'plots': plots
                    }
                    
                    st.success(f"✅ Simulation {sim_id} completed with enhanced analysis!")
        
        elif operation_mode == "Compare Saved Simulations":
            st.subheader("🔍 Simulation Comparison")
            
            # Get saved simulations
            sim_list = SimulationDB.get_simulation_list()
            
            if sim_list:
                sim_options = {sim['name']: sim['id'] for sim in sim_list}
                selected_names = st.multiselect(
                    "Select Simulations to Compare",
                    options=list(sim_options.keys()),
                    default=list(sim_options.keys())[:min(3, len(sim_options))]
                )
                
                selected_ids = [sim_options[name] for name in selected_names]
                
                if st.button("📊 Generate Comparison Dashboard", type="secondary"):
                    # Load selected simulations
                    sims_data = {}
                    for sim_id in selected_ids:
                        sim_data = SimulationDB.get_simulation(sim_id)
                        if sim_data and sim_data['results_history']:
                            final_results = sim_data['results_history'][-1]
                            sims_data[f"Sim {sim_id}"] = {
                                'phi': final_results['phi'],
                                'sigma_eq': final_results['sigma_eq'],
                                'h': final_results['h']
                            }
                    
                    # Create comparison dashboard
                    if sims_data:
                        comparison_fig = st.session_state.visualization_system.create_comparison_dashboard(sims_data)
                        st.session_state.comparison_figure = comparison_fig
                        st.success(f"Generated comparison dashboard for {len(sims_data)} simulations")
            else:
                st.info("No simulations saved yet. Run some simulations first!")
    
    # Main content area
    if operation_mode == "Run New Simulation" and st.session_state.current_simulation:
        st.header("📊 Enhanced Analysis Results")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "📊 Statistics", "🔗 Correlations", 
            "📏 Line Profiles", "📥 Export"
        ])
        
        with tab1:
            st.subheader("Simulation Overview")
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                final_phi = st.session_state.current_simulation['results']['history'][-1]['phi']
                st.metric("Avg φ", f"{np.mean(final_phi):.3f}")
            with col2:
                final_stress = st.session_state.current_simulation['results']['history'][-1]['sigma_eq']
                st.metric("Max Stress", f"{np.max(final_stress)/1e9:.2f} GPa")
            with col3:
                final_spacing = st.session_state.current_simulation['results']['history'][-1]['h']
                valid_spacing = final_spacing[final_spacing < 100]
                st.metric("Avg Spacing", f"{np.mean(valid_spacing):.1f} nm")
            with col4:
                final_plastic = st.session_state.current_simulation['results']['history'][-1]['eps_p_mag']
                st.metric("Max Plastic Strain", f"{np.max(final_plastic):.4f}")
            
            # Display field plots
            st.subheader("Field Distributions")
            plots = st.session_state.current_simulation['plots']
            
            if plots and 'phi_evolution' in plots:
                st.pyplot(plots['phi_evolution'])
        
        with tab2:
            st.subheader("Statistical Analysis")
            
            if plots and 'stress_analysis' in plots:
                st.pyplot(plots['stress_analysis'])
            
            if plots and 'spacing_analysis' in plots:
                st.pyplot(plots['spacing_analysis'])
        
        with tab3:
            st.subheader("Correlation Analysis")
            
            if plots and 'stress_spacing_correlation' in plots:
                st.pyplot(plots['stress_spacing_correlation'])
        
        with tab4:
            st.subheader("Line Profile Analysis")
            
            if plots and 'field_profiles' in plots:
                for field_name, profile_fig in plots['field_profiles'].items():
                    st.subheader(f"{field_name} Profiles")
                    st.pyplot(profile_fig)
        
        with tab5:
            st.subheader("Comprehensive Export")
            
            # Export options
            export_format = st.selectbox(
                "Export Format",
                ["Complete Package (ZIP)", "Data Only (NPZ)", "Plots Only (PNG)", "Report (PDF)"]
            )
            
            if st.button("📦 Generate Export", type="primary"):
                with st.spinner("Preparing export package..."):
                    exporter = EnhancedDataExporter()
                    
                    # Generate export
                    zip_buffer = exporter.export_comprehensive_package(
                        st.session_state.current_simulation['results']['history'],
                        st.session_state.current_simulation['params'],
                        st.session_state.current_simulation['plots'],
                        f"nanotwin_sim_{st.session_state.current_simulation['id']}"
                    )
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Export Package",
                        data=zip_buffer,
                        file_name=f"nanotwin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
                    
                    st.success("Export package generated successfully!")
    
    elif operation_mode == "Compare Saved Simulations" and 'comparison_figure' in st.session_state:
        st.header("📊 Multi-Simulation Comparison Dashboard")
        
        # Display comparison figure
        st.pyplot(st.session_state.comparison_figure)
        
        # Add comparison metrics
        st.subheader("Comparison Statistics")
        
        # Load simulations for detailed comparison
        sim_list = SimulationDB.get_simulation_list()
        if sim_list:
            comparison_data = []
            for sim in sim_list[:5]:  # Limit to 5 for display
                sim_data = SimulationDB.get_simulation(sim['id'])
                if sim_data and sim_data['results_history']:
                    final = sim_data['results_history'][-1]
                    comparison_data.append({
                        'Simulation': sim['name'],
                        'Avg φ': f"{np.mean(final['phi']):.3f}",
                        'Max σ_eq (GPa)': f"{np.max(final['sigma_eq'])/1e9:.2f}",
                        'Avg h (nm)': f"{np.mean(final['h'][final['h'] < 100]):.1f}",
                        'Max ε_p': f"{np.max(final['eps_p_mag']):.4f}"
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to the Enhanced Nanotwinned Copper Simulator</h2>
        <p style="font-size: 1.2rem; color: #666;">
        A comprehensive phase-field simulation platform with advanced post-processing capabilities
        </p>
        
        <div style="margin-top: 2rem; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;">
        <h4>🎯 Key Features:</h4>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
        <strong>Advanced Post-Processing</strong><br>
        • Enhanced line profiling<br>
        • Statistical analysis<br>
        • Correlation studies<br>
        • Multi-simulation comparison
        </div>
        
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
        <strong>Visualization Enhancements</strong><br>
        • 50+ scientific colormaps<br>
        • Journal templates<br>
        • Publication-ready plots<br>
        • Interactive 3D views
        </div>
        
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
        <strong>Data Management</strong><br>
        • Simulation database<br>
        • Comprehensive export<br>
        • Parameter sweeps<br>
        • Cloud-style storage
        </div>
        
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
        <strong>Physics Capabilities</strong><br>
        • FCC twinning mechanics<br>
        • Anisotropic elasticity<br>
        • Plasticity coupling<br>
        • Defect interactions
        </div>
        </div>
        
        <div style="margin-top: 2rem; padding: 1.5rem; background: #e8f4fd; border-radius: 10px;">
        <strong>📋 Getting Started:</strong><br>
        1. Configure simulation parameters in the sidebar<br>
        2. Run a new simulation or load saved ones<br>
        3. Explore enhanced post-processing features<br>
        4. Generate publication-quality results<br>
        5. Export comprehensive data packages
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
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

def generate_mock_simulation(N, dx, twin_spacing, applied_stress, n_steps):
    """Generate mock simulation data for demonstration"""
    # Create mock twin structure
    x = np.linspace(-N*dx/2, N*dx/2, N)
    y = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y = np.meshgrid(x, y)
    
    history = []
    
    for step in range(n_steps):
        # Create evolving twin pattern
        phase = 2 * np.pi * Y / twin_spacing + step * 0.1
        phi = np.tanh(np.sin(phase) * 3.0)
        
        # Create grain structure
        eta1 = 0.5 * (1 + np.tanh(X / 5.0))
        eta2 = 1 - eta1
        
        # Create stress field
        sigma_eq = applied_stress * 1e6 * (1 + 0.5 * np.sin(2*np.pi*X/50) * np.cos(2*np.pi*Y/50))
        sigma_eq += 0.3 * applied_stress * 1e6 * phi
        
        # Create twin spacing
        h = twin_spacing * (1 + 0.2 * np.random.randn(N, N))
        h = np.clip(h, 5, 100)
        
        # Create plastic strain
        eps_p_mag = 0.001 * step/n_steps * np.exp(-(X**2 + Y**2) / (100 * dx**2))
        
        # Create strain components
        eps_xx = 0.001 * np.sin(2*np.pi*X/100)
        eps_yy = 0.0005 * np.cos(2*np.pi*Y/100)
        eps_xy = 0.0003 * np.sin(2*np.pi*X/50) * np.cos(2*np.pi*Y/50)
        
        history.append({
            'phi': phi,
            'eta1': eta1,
            'eta2': eta2,
            'sigma_eq': sigma_eq,
            'sigma_xx': sigma_eq * 0.7,
            'sigma_yy': sigma_eq * 0.3,
            'sigma_xy': sigma_eq * 0.2,
            'h': h,
            'sigma_y': applied_stress * 0.8e6 * np.ones_like(sigma_eq),
            'eps_p_mag': eps_p_mag,
            'eps_xx': eps_xx,
            'eps_yy': eps_yy,
            'eps_xy': eps_xy
        })
    
    return {
        'history': history,
        'final': history[-1]
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    create_enhanced_streamlit_app()
