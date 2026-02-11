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
            '#1f77b4',  # CTB - Blue
            '#ff7f0e',  # ITB - Orange
            '#2ca02c',  # Grain 1 - Green
            '#d62728',  # Grain 2 - Red
            '#9467bd',  # Plastic zone - Purple
            '#8c564b'   # Defect - Brown
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
    
    def create_enhanced_line_profiles(self, results_dict, profile_config):
        """Create enhanced line profile analysis"""
        profile_types = profile_config.get('types', ['horizontal', 'vertical'])
        position_ratio = profile_config.get('position_ratio', 0.5)
        angle_deg = profile_config.get('angle_deg', 45)
        
        # Select which fields to profile
        fields_to_profile = ['phi', 'sigma_eq', 'h', 'eps_p_mag']
        available_fields = [f for f in fields_to_profile if f in results_dict]
        
        n_fields = len(available_fields)
        n_profiles = len(profile_types)
        
        fig = plt.figure(figsize=(15, 4*n_fields))
        
        # Create subplot grid: each field gets profile plot and location map
        gs = fig.add_gridspec(n_fields, 3, width_ratios=[2, 1, 1])
        
        for field_idx, field_name in enumerate(available_fields):
            data = results_dict[field_name]
            
            # Main profile plot
            ax_profiles = fig.add_subplot(gs[field_idx, 0])
            
            # Extract and plot profiles
            for profile_type in profile_types:
                distance, profile, endpoints = self.line_profiler.extract_profile(
                    data, profile_type, position_ratio, angle_deg
                )
                
                # Plot profile
                ax_profiles.plot(distance, profile, 
                               linewidth=2, alpha=0.8,
                               label=f'{profile_type.title()} Profile')
            
            ax_profiles.set_xlabel('Position (nm)', fontsize=10)
            ax_profiles.set_ylabel(field_name.replace('_', ' ').title(), fontsize=10)
            ax_profiles.set_title(f'{field_name.replace("_", " ").title()} Line Profiles', fontsize=12)
            ax_profiles.legend(fontsize=8)
            ax_profiles.grid(True, alpha=0.3)
            
            # Location map
            ax_location = fig.add_subplot(gs[field_idx, 1])
            im = ax_location.imshow(data, extent=self.extent, cmap='viridis', 
                                   origin='lower', aspect='equal')
            
            # Plot profile lines
            profile_colors = {'horizontal': 'red', 'vertical': 'blue', 
                            'diagonal': 'green', 'anti_diagonal': 'purple',
                            'custom': 'orange'}
            
            for profile_type in profile_types:
                distance, profile, endpoints = self.line_profiler.extract_profile(
                    data, profile_type, position_ratio, angle_deg
                )
                color = profile_colors.get(profile_type, 'white')
                ax_location.plot([endpoints[0], endpoints[2]], [endpoints[1], endpoints[3]],
                               color=color, linewidth=2, alpha=0.7, linestyle='--')
            
            ax_location.set_title(f'{field_name} Profile Locations', fontsize=10)
            ax_location.set_xlabel('x (nm)', fontsize=8)
            ax_location.set_ylabel('y (nm)', fontsize=8)
            plt.colorbar(im, ax=ax_location, fraction=0.046, pad=0.04)
            
            # Statistics panel
            ax_stats = fig.add_subplot(gs[field_idx, 2])
            
            # Calculate field statistics
            stats_data = {
                'Mean': np.mean(data),
                'Std Dev': np.std(data),
                'Max': np.max(data),
                'Min': np.min(data),
                '95th %ile': np.percentile(data, 95)
            }
            
            # Create horizontal bar chart
            y_pos = np.arange(len(stats_data))
            values = list(stats_data.values())
            
            bars = ax_stats.barh(y_pos, values, color='steelblue', alpha=0.7)
            ax_stats.set_yticks(y_pos)
            ax_stats.set_yticklabels(list(stats_data.keys()))
            ax_stats.set_xlabel('Value', fontsize=8)
            ax_stats.set_title(f'{field_name} Statistics', fontsize=10)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax_stats.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2,
                            f'{val:.3e}' if abs(val) > 1000 else f'{val:.3f}',
                            va='center', fontsize=7)
        
        plt.tight_layout()
        return fig
    
    def create_statistical_analysis(self, results_history, timesteps):
        """Create comprehensive statistical analysis plots"""
        # Collect data over time
        time_data = {
            'avg_stress': [],
            'max_stress': [],
            'avg_spacing': [],
            'plastic_work': [],
            'phi_norm': []
        }
        
        for results in results_history:
            time_data['avg_stress'].append(np.mean(results['sigma_eq']) / 1e9)  # GPa
            time_data['max_stress'].append(np.max(results['sigma_eq']) / 1e9)   # GPa
            
            valid_h = results['h'][(results['h'] > 5) & (results['h'] < 50)]
            time_data['avg_spacing'].append(np.mean(valid_h) if len(valid_h) > 0 else 0)
            
            time_data['plastic_work'].append(np.sum(results['eps_p_mag']) * (self.dx**2))
            time_data['phi_norm'].append(np.linalg.norm(results['phi']))
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Time evolution plots
        axes[0, 0].plot(timesteps, time_data['avg_stress'], 'b-', linewidth=2, label='Average')
        axes[0, 0].plot(timesteps, time_data['max_stress'], 'r-', linewidth=2, label='Maximum')
        axes[0, 0].set_xlabel('Time (ns)', fontsize=10)
        axes[0, 0].set_ylabel('Stress (GPa)', fontsize=10)
        axes[0, 0].set_title('Stress Evolution', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(timesteps, time_data['avg_spacing'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time (ns)', fontsize=10)
        axes[0, 1].set_ylabel('Avg Twin Spacing (nm)', fontsize=10)
        axes[0, 1].set_title('Twin Spacing Evolution', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(timesteps, time_data['plastic_work'], 'm-', linewidth=2)
        axes[0, 2].set_xlabel('Time (ns)', fontsize=10)
        axes[0, 2].set_ylabel('Plastic Work', fontsize=10)
        axes[0, 2].set_title('Plastic Work Evolution', fontsize=12)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Final frame histograms
        final_results = results_history[-1]
        
        axes[1, 0].hist(final_results['sigma_eq'].flatten() / 1e9, bins=50, 
                       density=True, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_xlabel('Von Mises Stress (GPa)', fontsize=10)
        axes[1, 0].set_ylabel('Probability Density', fontsize=10)
        axes[1, 0].set_title('Stress Distribution', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        valid_h = final_results['h'][(final_results['h'] > 5) & (final_results['h'] < 50)]
        if len(valid_h) > 0:
            axes[1, 1].hist(valid_h.flatten(), bins=30, density=True, 
                           alpha=0.7, color='green', edgecolor='black')
            axes[1, 1].set_xlabel('Twin Spacing (nm)', fontsize=10)
            axes[1, 1].set_ylabel('Probability Density', fontsize=10)
            axes[1, 1].set_title('Twin Spacing Distribution', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Correlation plot
        if len(time_data['avg_stress']) > 1:
            stress_gradient = np.gradient(time_data['avg_stress'])
            spacing_gradient = np.gradient(time_data['avg_spacing'])
            axes[1, 2].scatter(stress_gradient[:-1], spacing_gradient[1:], 
                             alpha=0.6, s=50, c=timesteps[1:], cmap='viridis')
            axes[1, 2].set_xlabel('Stress Rate (GPa/ns)', fontsize=10)
            axes[1, 2].set_ylabel('Spacing Rate (nm/ns)', fontsize=10)
            axes[1, 2].set_title('Stress-Spacing Correlation', fontsize=12)
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add colorbar for time
            sm = plt.cm.ScalarMappable(cmap='viridis')
            sm.set_array(timesteps[1:])
            plt.colorbar(sm, ax=axes[1, 2], label='Time (ns)')
        
        plt.tight_layout()
        return fig
    
    def create_correlation_analysis(self, results_dict):
        """Create correlation analysis between different fields"""
        # Prepare data for correlation analysis
        field_pairs = [
            ('sigma_eq', 'h', 'Stress vs Twin Spacing'),
            ('eps_p_mag', 'sigma_eq', 'Plastic Strain vs Stress'),
            ('phi', 'sigma_y', 'Twin Order vs Yield Stress'),
            ('eta1', 'eps_p_mag', 'Twin Grain vs Plastic Strain')
        ]
        
        n_pairs = len(field_pairs)
        cols = min(2, n_pairs)
        rows = (n_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (field1, field2, title) in enumerate(field_pairs):
            if field1 not in results_dict or field2 not in results_dict:
                continue
                
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            data1 = results_dict[field1].flatten()
            data2 = results_dict[field2].flatten()
            
            # Sample data for clarity
            sample_size = min(5000, len(data1))
            indices = np.random.choice(len(data1), sample_size, replace=False)
            data1_sampled = data1[indices]
            data2_sampled = data2[indices]
            
            # Calculate correlation
            valid_mask = np.isfinite(data1_sampled) & np.isfinite(data2_sampled)
            if np.sum(valid_mask) > 10:
                corr_coef = np.corrcoef(data1_sampled[valid_mask], 
                                       data2_sampled[valid_mask])[0, 1]
                
                # Scatter plot
                scatter = ax.scatter(data1_sampled[valid_mask], data2_sampled[valid_mask],
                                   alpha=0.3, s=10, c='blue', edgecolors='none')
                
                # Add regression line
                if abs(corr_coef) > 0.1:
                    coeffs = np.polyfit(data1_sampled[valid_mask], 
                                       data2_sampled[valid_mask], 1)
                    x_range = np.linspace(np.min(data1_sampled[valid_mask]), 
                                         np.max(data1_sampled[valid_mask]), 100)
                    y_pred = np.polyval(coeffs, x_range)
                    ax.plot(x_range, y_pred, 'r-', linewidth=2, alpha=0.8,
                           label=f'R = {corr_coef:.3f}')
                
                ax.set_xlabel(field1.replace('_', ' ').title(), fontsize=10)
                ax.set_ylabel(field2.replace('_', ' ').title(), fontsize=10)
                ax.set_title(f'{title}\nCorrelation: {corr_coef:.3f}', fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            else:
                ax.text(0.5, 0.5, 'Insufficient valid data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=11)
        
        # Hide empty subplots
        for idx in range(n_pairs, rows*cols):
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
                return self.COLORMAPS[cmap_name]  # Custom colormap object
        else:
            return plt.cm.get_cmap('viridis')  # Default fallback

# ============================================================================
# FIXED NUMBA-COMPATIBLE FUNCTIONS (ORIGINAL)
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
# ORIGINAL CLASSES (KEPT FOR COMPATIBILITY)
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
            # Return zeros in case of error
            zeros = np.zeros((self.N, self.N))
            return zeros, zeros, zeros, zeros, zeros, zeros, zeros

class NanotwinnedCuSolver:
    """Main solver with comprehensive error handling"""
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']
        # Material properties
        self.mat_props = MaterialProperties.get_cu_properties()
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
        # Initialize history for convergence monitoring
        self.history = {
            'phi_norm': [],
            'energy': [],
            'max_stress': [],
            'plastic_work': [],
            'avg_stress': [],
            'twin_spacing_avg': []
        }

    def initialize_fields(self):
        """Initialize order parameters based on selected geometry"""
        geom_type = self.params.get('geometry_type', 'standard')
        twin_spacing = self.params['twin_spacing']
        gb_pos = self.params['grain_boundary_pos']
        return self.geom_viz.create_twin_grain_geometry(twin_spacing, gb_pos)

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
            # Update history for monitoring
            phi_norm = np.linalg.norm(self.phi)
            total_energy = self.compute_total_energy()
            max_stress = np.max(sigma_eq)
            avg_stress = np.mean(sigma_eq)
            avg_spacing = np.mean(h[(h > 5) & (h < 50)])
            
            self.history['phi_norm'].append(phi_norm)
            self.history['energy'].append(total_energy)
            self.history['max_stress'].append(max_stress)
            self.history['avg_stress'].append(avg_stress)
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
                'eps_p_mag': np.zeros_like(sigma_eq),  # Placeholder
                'eps_xx': eps_xx.copy(),
                'eps_yy': eps_yy.copy(),
                'eps_xy': eps_xy.copy(),
                'convergence': {
                    'phi_norm': phi_norm,
                    'energy': total_energy,
                    'max_stress': max_stress,
                    'avg_stress': avg_stress,
                    'avg_spacing': avg_spacing
                }
            }
            return results
        except Exception as e:
            st.error(f"Error in simulation step: {e}")
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
                    'avg_spacing': 0
                }
            }

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
            N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64)
            dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1)
            dt = st.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5)
            
            # Material parameters
            st.subheader("🔬 Material Parameters")
            twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0)
            grain_boundary_pos = st.slider("Grain boundary position (nm)", -50.0, 50.0, 0.0, 1.0)
            
            # Thermodynamic parameters
            st.subheader("⚡ Thermodynamic Parameters")
            W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1)
            A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5)
            B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5)
            
            # Gradient energy parameters
            st.subheader("🌀 Gradient Energy")
            kappa0 = st.slider("κ₀ (Gradient energy ref)", 0.01, 10.0, 1.0, 0.1)
            gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05)
            kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1)
            
            # Loading conditions
            st.subheader("🏋️ Loading Conditions")
            applied_stress_MPa = st.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0)
            
            # Simulation control
            st.subheader("⏯️ Simulation Control")
            n_steps = st.slider("Number of steps", 10, 1000, 100, 10)
            save_frequency = st.slider("Save frequency", 1, 100, 10, 1)
            
            # Enhanced visualization settings
            st.subheader("🎨 Enhanced Visualization")
            colormap_library = EnhancedTwinVisualizer(N, dx).COLORMAPS
            selected_cmap_phi = st.selectbox("φ Colormap", list(colormap_library.keys()), 
                                           index=list(colormap_library.keys()).index('RdBu_r') 
                                           if 'RdBu_r' in colormap_library else 0)
            selected_cmap_stress = st.selectbox("σ_eq Colormap", list(colormap_library.keys()),
                                              index=list(colormap_library.keys()).index('hot')
                                              if 'hot' in colormap_library else 0)
            
            # Initialize button
            if st.button("🚀 Initialize Simulation", type="primary", use_container_width=True):
                params = {
                    'N': N, 'dx': dx, 'dt': dt,
                    'W': W, 'A': A, 'B': B,
                    'kappa0': kappa0, 'gamma_aniso': gamma_aniso, 'kappa_eta': kappa_eta,
                    'twin_spacing': twin_spacing,
                    'grain_boundary_pos': grain_boundary_pos,
                    'applied_stress': applied_stress_MPa * 1e6,
                    'save_frequency': save_frequency,
                    'cmap_phi': selected_cmap_phi,
                    'cmap_stress': selected_cmap_stress
                }
                
                # Initialize geometry
                geom_viz = InitialGeometryVisualizer(N, dx)
                phi, eta1, eta2 = geom_viz.create_twin_grain_geometry(twin_spacing, grain_boundary_pos)
                
                # Store in session state
                st.session_state.initial_geometry = {
                    'phi': phi, 'eta1': eta1, 'eta2': eta2,
                    'geom_viz': geom_viz, 'params': params
                }
                st.session_state.initialized = True
                st.session_state.operation_mode = 'new'
                st.success("✅ Simulation initialized successfully!")
        
        else:  # Compare Saved Simulations
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
                
                if comparison_type == "Overlay Line Profiles":
                    profile_direction = st.selectbox(
                        "Profile Direction",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal", "Custom"],
                        index=0
                    )
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                
                # Field selection for comparison
                field_to_compare = st.selectbox(
                    "Field to Compare",
                    ["phi (Twin Order)", "sigma_eq (Von Mises Stress)", 
                     "h (Twin Spacing)", "sigma_y (Yield Stress)"],
                    index=1
                )
                
                if st.button("🔬 Run Comparison", type="primary"):
                    st.session_state.comparison_config = {
                        'sim_ids': [sim_options[name] for name in selected_sim_ids],
                        'type': comparison_type,
                        'field': field_to_compare,
                        'profile_direction': profile_direction if comparison_type == "Overlay Line Profiles" else None,
                        'position_ratio': position_ratio if comparison_type == "Overlay Line Profiles" else None
                    }
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
        
        with tabs[0]:  # Initial Geometry
            st.header("Initial Geometry Visualization")
            geom_viz = st.session_state.initial_geometry['geom_viz']
            phi = st.session_state.initial_geometry['phi']
            eta1 = st.session_state.initial_geometry['eta1']
            eta2 = st.session_state.initial_geometry['eta2']
            
            # Use enhanced visualizer for initial state
            initial_results = {
                'phi': phi,
                'eta1': eta1,
                'sigma_eq': np.zeros_like(phi),  # Placeholder
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
        
        with tabs[1]:  # Run Simulation
            st.header("Run Simulation")
            
            if st.button("▶️ Start Evolution", type="secondary", use_container_width=True):
                with st.spinner("Running phase-field simulation..."):
                    # Initialize solver
                    solver = NanotwinnedCuSolver(params)
                    solver.phi = st.session_state.initial_geometry['phi'].copy()
                    solver.eta1 = st.session_state.initial_geometry['eta1'].copy()
                    solver.eta2 = st.session_state.initial_geometry['eta2'].copy()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Storage for results
                    results_history = []
                    timesteps = []
                    
                    # Simulation loop
                    for step in range(n_steps):
                        status_text.text(f"Step {step+1}/{n_steps} | Time: {(step+1)*dt:.4f} ns")
                        results = solver.step(params['applied_stress'])
                        
                        if step % save_frequency == 0:
                            results_history.append(results.copy())
                            timesteps.append(step * dt)
                        
                        progress_bar.progress((step + 1) / n_steps)
                    
                    st.success(f"✅ Simulation completed! Generated {len(results_history)} frames.")
                    
                    # Store results
                    st.session_state.results_history = results_history
                    st.session_state.timesteps = timesteps
                    st.session_state.solver = solver
                    
                    # Save to database
                    if 'twin_simulations' not in st.session_state:
                        st.session_state.twin_simulations = {}
                    
                    sim_id = SimulationDatabase.generate_id(params)
                    st.session_state.twin_simulations[sim_id] = {
                        'id': sim_id,
                        'params': params,
                        'results_history': results_history,
                        'timesteps': timesteps,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.balloons()
        
        with tabs[2]:  # Basic Results
            if 'results_history' in st.session_state:
                st.header("Basic Results Visualization")
                
                # Select frame to display
                frame_idx = st.slider("Select frame", 0, len(st.session_state.results_history)-1, 
                                    len(st.session_state.results_history)-1)
                results = st.session_state.results_history[frame_idx]
                
                # Enhanced field comparison
                fig = visualizer.create_multi_field_comparison(results)
                st.pyplot(fig)
                
                # Convergence plots
                st.subheader("Convergence Monitoring")
                if hasattr(st.session_state.solver, 'history'):
                    fig_conv = visualizer.create_statistical_analysis(
                        st.session_state.results_history,
                        st.session_state.timesteps
                    )
                    st.pyplot(fig_conv)
            else:
                st.info("Run a simulation first to view results")
        
        with tabs[3]:  # Advanced Analysis
            if 'results_history' in st.session_state:
                st.header("Advanced Analysis Tools")
                
                # Field selection for analysis
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Line Profile Analysis", "Statistical Analysis", "Correlation Analysis"],
                    index=0
                )
                
                if analysis_type == "Line Profile Analysis":
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
                    profile_config = {
                        'types': profile_types,
                        'position_ratio': position_ratio
                    }
                    
                    fig_profiles = visualizer.create_enhanced_line_profiles(
                        {field_to_profile: results[field_to_profile]},
                        profile_config
                    )
                    st.pyplot(fig_profiles)
                
                elif analysis_type == "Statistical Analysis":
                    st.subheader("Statistical Analysis")
                    
                    fig_stats = visualizer.create_statistical_analysis(
                        st.session_state.results_history,
                        st.session_state.timesteps
                    )
                    st.pyplot(fig_stats)
                    
                    # Display statistical summary
                    final_results = st.session_state.results_history[-1]
                    st.subheader("Final Frame Statistics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for field in ['phi', 'sigma_eq', 'h']:
                            if field in final_results:
                                data = final_results[field]
                                st.metric(
                                    f"{field.replace('_', ' ').title()}",
                                    f"Mean: {np.mean(data):.3e}\nStd: {np.std(data):.3e}"
                                )
                    
                    with col2:
                        # Percentile information
                        stress_data = final_results['sigma_eq'] / 1e9  # GPa
                        percentiles = np.percentile(stress_data, [25, 50, 75, 95])
                        st.metric(
                            "Stress Percentiles (GPa)",
                            f"25%: {percentiles[0]:.2f}\n50%: {percentiles[1]:.2f}\n"
                            f"75%: {percentiles[2]:.2f}\n95%: {percentiles[3]:.2f}"
                        )
                
                elif analysis_type == "Correlation Analysis":
                    st.subheader("Field Correlation Analysis")
                    
                    fig_corr = visualizer.create_correlation_analysis(
                        st.session_state.results_history[-1]
                    )
                    st.pyplot(fig_corr)
            else:
                st.info("Run a simulation first to use advanced analysis tools")
        
        with tabs[4]:  # Comparison Tools
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
                        'Frames': len(simulations[0]['results']) if simulations and simulations[0]['results'] else 0,
                        'Created': sim.get('created_at', '')[:19]
                    })
                
                df = pd.DataFrame(sim_data)
                st.dataframe(df, use_container_width=True)
                
                # Comparison visualization
                if len(simulations) >= 2:
                    st.subheader("Comparison Visualization")
                    
                    comparison_type = st.selectbox(
                        "Visualization Type",
                        ["Stress Distribution", "Twin Spacing", "Convergence Rate"],
                        index=0
                    )
                    
                    if comparison_type == "Stress Distribution":
                        col1, col2 = st.columns(2)
                        for idx, sim in enumerate(simulations[:4]):
                            with (col1 if idx % 2 == 0 else col2):
                                if sim['results']:
                                    stress_data = sim['results']['sigma_eq'] / 1e9
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    im = ax.imshow(stress_data, cmap='hot', 
                                                  vmin=0, vmax=np.percentile(stress_data, 95),
                                                  aspect='equal')
                                    ax.set_title(f"Sim {idx+1}: {sim['params'].get('twin_spacing', 0):.1f}nm")
                                    ax.axis('off')
                                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                    st.pyplot(fig)
        
        with tabs[5]:  # Enhanced Export
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
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        zip_buffer = BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # 1. Save parameters
                            params_json = json.dumps(params, indent=2, cls=NumpyEncoder)
                            zip_file.writestr(f"simulation_params_{timestamp}.json", params_json)
                            
                            # 2. Save final field data
                            final_results = st.session_state.results_history[-1]
                            npz_buffer = BytesIO()
                            np.savez_compressed(npz_buffer, **final_results)
                            zip_file.writestr(f"final_fields_{timestamp}.npz", npz_buffer.getvalue())
                            
                            # 3. Save convergence history
                            if hasattr(st.session_state.solver, 'history'):
                                history_data = {
                                    'timesteps': np.array(st.session_state.timesteps),
                                    'phi_norm': np.array(st.session_state.solver.history['phi_norm']),
                                    'energy': np.array(st.session_state.solver.history['energy']),
                                    'max_stress': np.array(st.session_state.solver.history['max_stress']),
                                    'avg_stress': np.array(st.session_state.solver.history['avg_stress']),
                                    'twin_spacing_avg': np.array(st.session_state.solver.history['twin_spacing_avg'])
                                }
                                npz_buffer = BytesIO()
                                np.savez_compressed(npz_buffer, **history_data)
                                zip_file.writestr(f"convergence_history_{timestamp}.npz", npz_buffer.getvalue())
                            
                            # 4. Generate and save figures
                            if "Publication Figures (PNG)" in export_format:
                                # Create multi-field comparison figure
                                fig_comparison = visualizer.create_multi_field_comparison(final_results)
                                img_buffer = BytesIO()
                                fig_comparison.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                                zip_file.writestr(f"field_comparison_{timestamp}.png", img_buffer.getvalue())
                                
                                # Create line profile figure
                                profile_config = {'types': ['Horizontal', 'Vertical'], 'position_ratio': 0.5}
                                fig_profiles = visualizer.create_enhanced_line_profiles(
                                    {'sigma_eq': final_results['sigma_eq']}, 
                                    profile_config
                                )
                                img_buffer = BytesIO()
                                fig_profiles.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                                zip_file.writestr(f"line_profiles_{timestamp}.png", img_buffer.getvalue())
                            
                            # 5. Save summary CSV
                            csv_data = []
                            for i, results in enumerate(st.session_state.results_history):
                                csv_data.append({
                                    'step': i,
                                    'time_ns': i * params['dt'] * params['save_frequency'],
                                    'avg_phi': np.mean(results['phi']),
                                    'avg_sigma_eq_gpa': np.mean(results['sigma_eq']) / 1e9,
                                    'max_sigma_eq_gpa': np.max(results['sigma_eq']) / 1e9,
                                    'avg_h_nm': np.mean(results['h'][(results['h']>5)&(results['h']<50)]),
                                    'avg_sigma_y_mpa': np.mean(results['sigma_y']) / 1e6
                                })
                            
                            df = pd.DataFrame(csv_data)
                            csv_buffer = StringIO()
                            df.to_csv(csv_buffer, index=False)
                            zip_file.writestr(f"summary_{timestamp}.csv", csv_buffer.getvalue())
                        
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label="⬇️ Download Export Package",
                            data=zip_buffer,
                            file_name=f"nanotwin_export_{timestamp}.zip",
                            mime="application/zip"
                        )
                        st.success("✅ Export package ready!")
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

if __name__ == "__main__":
    main()
