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
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================================================
# SIMULATION DATABASE SYSTEM - Enhanced with metadata and search
# ============================================================================
class SimulationDBCu:
    """Advanced simulation database for nanotwin simulations"""
    
    @staticmethod
    def generate_id(sim_params):
        """Generate unique 10-character ID from simulation parameters"""
        param_str = json.dumps(sim_params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()[:10]
    
    @staticmethod
    def save_simulation(sim_params, history, metadata, results_history):
        """Save complete simulation to database with enhanced metadata"""
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
            st.session_state.simulation_counter = 0
            st.session_state.simulation_tags = {}
        
        sim_id = SimulationDBCu.generate_id(sim_params)
        
        # Enhanced metadata
        enhanced_metadata = {
            **metadata,
            'id': sim_id,
            'created_at': datetime.now().isoformat(),
            'simulation_number': st.session_state.simulation_counter,
            'tags': ['default'],
            'notes': '',
            'favorite': False,
            'last_accessed': datetime.now().isoformat(),
            'file_size_estimate': len(pickle.dumps(results_history)) / (1024 * 1024)  # MB
        }
        
        # Store simulation data
        st.session_state.simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'history': history,  # Internal solver history (every step)
            'results_history': results_history,  # Saved frames for visualization
            'metadata': enhanced_metadata,
            'statistics': SimulationDBCu.compute_statistics(results_history, history)
        }
        
        # Update tags
        if 'tags' not in st.session_state.simulation_tags:
            st.session_state.simulation_tags = {}
        
        st.session_state.simulation_counter += 1
        st.success(f"✅ Simulation saved with ID: `{sim_id}`")
        return sim_id
    
    @staticmethod
    def compute_statistics(results_history, history):
        """Compute comprehensive statistics for a simulation"""
        if not results_history:
            return {}
        
        final_results = results_history[-1]
        
        stats_dict = {
            'stress_statistics': {
                'max_stress_MPa': float(np.max(final_results['sigma_eq'])) / 1e6,
                'mean_stress_MPa': float(np.mean(final_results['sigma_eq'])) / 1e6,
                'std_stress_MPa': float(np.std(final_results['sigma_eq'])) / 1e6,
                'stress_percentiles': {
                    'p10': float(np.percentile(final_results['sigma_eq'], 10)) / 1e6,
                    'p50': float(np.percentile(final_results['sigma_eq'], 50)) / 1e6,
                    'p90': float(np.percentile(final_results['sigma_eq'], 90)) / 1e6,
                }
            },
            'twin_statistics': {
                'avg_spacing_nm': float(np.mean(final_results['h'][(final_results['h']>5) & (final_results['h']<50)])),
                'std_spacing_nm': float(np.std(final_results['h'][(final_results['h']>5) & (final_results['h']<50)])),
                'min_spacing_nm': float(np.min(final_results['h'][final_results['h']>5])),
                'max_spacing_nm': float(np.max(final_results['h'][final_results['h']<100])),
            },
            'plasticity_statistics': {
                'max_plastic_strain': float(np.max(final_results['eps_p_mag'])),
                'mean_plastic_strain': float(np.mean(final_results['eps_p_mag'])),
                'total_plastic_work_J': float(history['plastic_work'][-1] if 'plastic_work' in history else 0),
            },
            'convergence_statistics': {
                'final_phi_norm': float(history['phi_norm'][-1]) if 'phi_norm' in history else 0,
                'final_energy_J': float(history['energy'][-1]) if 'energy' in history else 0,
                'energy_change_percent': float(((history['energy'][-1] - history['energy'][0]) / history['energy'][0] * 100) 
                                              if len(history.get('energy', [])) > 1 else 0),
            },
            'defect_statistics': {
                'twin_area_nm2': float(np.sum(final_results['eta1'] > 0.5) * (results_history[0]['phi'].shape[0] * 0.5)**2),
                'twin_volume_fraction': float(np.mean(final_results['eta1'] > 0.5)),
                'twin_boundary_length_nm': float(np.sum(np.abs(np.gradient(final_results['phi'])) > 0.1) * 0.5),
            }
        }
        
        return stats_dict
    
    @staticmethod
    def get_simulation(sim_id):
        """Retrieve simulation by ID with access tracking"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            # Update last accessed time
            st.session_state.simulations[sim_id]['metadata']['last_accessed'] = datetime.now().isoformat()
            return st.session_state.simulations[sim_id]
        return None
    
    @staticmethod
    def get_all_simulations():
        """Get all stored simulations with sorting options"""
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
    def get_simulation_list(sort_by='created_at', filter_tags=None, search_term=''):
        """Get list of simulations for dropdown with filtering and sorting"""
        if 'simulations' not in st.session_state:
            return []
        
        simulations = []
        for sim_id, sim_data in st.session_state.simulations.items():
            params = sim_data['params']
            metadata = sim_data['metadata']
            
            # Check search term
            search_fields = [
                sim_id,
                params.get('geometry_type', ''),
                str(params.get('twin_spacing', '')),
                str(params.get('applied_stress', '')),
                metadata.get('notes', ''),
                ','.join(metadata.get('tags', []))
            ]
            
            if search_term and not any(search_term.lower() in str(field).lower() for field in search_fields):
                continue
            
            # Check tags filter
            if filter_tags and not any(tag in metadata.get('tags', []) for tag in filter_tags):
                continue
            
            # Create display name
            name = f"{params.get('geometry_type', 'Unknown')} | λ={params.get('twin_spacing', 0):.1f}nm | σ={params.get('applied_stress', 0)/1e6:.0f}MPa"
            if metadata.get('notes'):
                name += f" | {metadata['notes'][:20]}..."
            
            simulations.append({
                'id': sim_id,
                'name': name,
                'params': params,
                'metadata': metadata,
                'statistics': sim_data.get('statistics', {}),
                'display_name': name
            })
        
        # Sorting
        reverse = sort_by in ['created_at', 'last_accessed']
        simulations.sort(key=lambda x: x['metadata'].get(sort_by, ''), reverse=reverse)
        
        return simulations
    
    @staticmethod
    def update_metadata(sim_id, updates):
        """Update simulation metadata"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            st.session_state.simulations[sim_id]['metadata'].update(updates)
            return True
        return False
    
    @staticmethod
    def add_tag(sim_id, tag):
        """Add tag to simulation"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            tags = st.session_state.simulations[sim_id]['metadata'].get('tags', [])
            if tag not in tags:
                tags.append(tag)
                st.session_state.simulations[sim_id]['metadata']['tags'] = tags
            return True
        return False
    
    @staticmethod
    def remove_tag(sim_id, tag):
        """Remove tag from simulation"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            tags = st.session_state.simulations[sim_id]['metadata'].get('tags', [])
            if tag in tags:
                tags.remove(tag)
                st.session_state.simulations[sim_id]['metadata']['tags'] = tags
            return True
        return False
    
    @staticmethod
    def toggle_favorite(sim_id):
        """Toggle favorite status"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            current = st.session_state.simulations[sim_id]['metadata'].get('favorite', False)
            st.session_state.simulations[sim_id]['metadata']['favorite'] = not current
            return not current
        return False
    
    @staticmethod
    def get_database_stats():
        """Get database statistics"""
        if 'simulations' not in st.session_state:
            return {}
        
        sims = st.session_state.simulations
        total_size = sum(sim['metadata'].get('file_size_estimate', 0) for sim in sims.values())
        
        return {
            'total_simulations': len(sims),
            'total_size_mb': total_size,
            'avg_simulation_size_mb': total_size / len(sims) if sims else 0,
            'tags_used': list(set(tag for sim in sims.values() 
                                for tag in sim['metadata'].get('tags', []))),
            'favorites_count': sum(1 for sim in sims.values() 
                                 if sim['metadata'].get('favorite', False)),
            'last_simulation_date': max(sim['metadata'].get('created_at', '') 
                                      for sim in sims.values()) if sims else None,
        }

# ============================================================================
# ADVANCED PUBLICATION STYLING SYSTEM - Enhanced with 50+ colormaps
# ============================================================================
class PublicationStylerCu:
    """Advanced publication-quality styling system with 50+ colormaps"""
    
    # Enhanced colormap library
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
    
    @staticmethod
    def create_custom_colormaps():
        """Create enhanced scientific colormaps specific to nanotwin simulations"""
        
        # Stress-specific colormap (blue-white-red diverging)
        stress_map = LinearSegmentedColormap.from_list('stress_map', [
            (0.0, '#2c7bb6'),   # Dark blue
            (0.2, '#abd9e9'),   # Light blue
            (0.4, '#ffffbf'),   # Yellow-white
            (0.6, '#fdae61'),   # Orange
            (0.8, '#d7191c'),   # Red
            (1.0, '#800026')    # Dark red
        ])
        
        # Twin order parameter (φ) colormap
        twin_phi_map = LinearSegmentedColormap.from_list('twin_phi_map', [
            (-1.0, '#2166ac'),   # Deep blue (negative φ)
            (-0.5, '#67a9cf'),   # Light blue
            (0.0, '#f7f7f7'),    # White (φ=0)
            (0.5, '#ef8a62'),    # Light red
            (1.0, '#b2182b')     # Deep red (positive φ)
        ])
        
        # Twin spacing colormap
        spacing_map = LinearSegmentedColormap.from_list('spacing_map', [
            (0.0, '#440154'),   # Deep purple
            (0.2, '#3b528b'),   # Dark blue
            (0.4, '#21918c'),   # Teal
            (0.6, '#5ec962'),   # Green
            (0.8, '#fde725'),   # Yellow
            (1.0, '#ffffff')    # White
        ])
        
        # Plastic strain colormap
        plastic_map = LinearSegmentedColormap.from_list('plastic_map', [
            (0.0, '#f7fbff'),   # Very light blue
            (0.2, '#c6dbef'),   # Light blue
            (0.4, '#6baed6'),   # Blue
            (0.6, '#2171b5'),   # Dark blue
            (0.8, '#084594'),   # Very dark blue
            (1.0, '#08306b')    # Navy
        ])
        
        # Yield stress colormap
        yield_map = LinearSegmentedColormap.from_list('yield_map', [
            (0.0, '#f7fcf5'),   # Very light green
            (0.2, '#c7e9c0'),   # Light green
            (0.4, '#74c476'),   # Green
            (0.6, '#238b45'),   # Dark green
            (0.8, '#005a32'),   # Very dark green
            (1.0, '#003320')    # Forest green
        ])
        
        # Grain structure categorical
        grain_categorical = ListedColormap([
            '#1f77b4',  # Blue - Twin grain
            '#ff7f0e',  # Orange - Twin-free grain
            '#2ca02c',  # Green - Mixed region
            '#d62728',  # Red - Defect region
            '#9467bd',  # Purple - Void
            '#8c564b'   # Brown - Dislocation
        ])
        
        return {
            'stress_map': stress_map,
            'twin_phi_map': twin_phi_map,
            'spacing_map': spacing_map,
            'plastic_map': plastic_map,
            'yield_map': yield_map,
            'grain_categorical': grain_categorical
        }
    
    @staticmethod
    def get_journal_styles():
        """Return enhanced journal-specific style parameters"""
        return {
            'nature': {
                'figure_width_single': 8.9,  # cm to inches
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
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
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
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
                              '#4DBEEE', '#A2142F', '#FF00FF', '#00FFFF', '#FFA500']
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
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000', '#44AA99',
                              '#882255', '#117733', '#999933', '#AA4499', '#88CCEE']
            },
            'prl': {
                'figure_width_single': 3.4,
                'figure_width_double': 7.0,
                'font_family': 'Times New Roman',
                'font_size_small': 8,
                'font_size_medium': 10,
                'font_size_large': 12,
                'line_width': 1.0,
                'axes_linewidth': 1.0,
                'tick_width': 1.0,
                'tick_length': 4,
                'grid_alpha': 0,
                'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                              '#0072B2', '#D55E00', '#CC79A7', '#999999', '#FFFFFF']
            },
            'advanced_materials': {
                'figure_width_single': 8.6,
                'figure_width_double': 17.8,
                'font_family': 'Arial',
                'font_size_small': 9,
                'font_size_medium': 10,
                'font_size_large': 12,
                'line_width': 1.5,
                'axes_linewidth': 1.5,
                'tick_width': 1.5,
                'tick_length': 5,
                'grid_alpha': 0.3,
                'dpi': 600,
                'color_cycle': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
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
        """Apply enhanced journal-specific styling to figure"""
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
    
    @staticmethod
    def get_styling_controls():
        """Get comprehensive styling controls for sidebar"""
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
        
        with st.sidebar.expander("🎨 Colormap Selection", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['stress_cmap'] = st.selectbox("Stress Colormap", 
                                                          list(PublicationStylerCu.COLORMAPS.keys()),
                                                          index=list(PublicationStylerCu.COLORMAPS.keys()).index('hot'))
                style_params['twin_cmap'] = st.selectbox("Twin φ Colormap", 
                                                        list(PublicationStylerCu.COLORMAPS.keys()),
                                                        index=list(PublicationStylerCu.COLORMAPS.keys()).index('RdBu'))
                style_params['spacing_cmap'] = st.selectbox("Spacing Colormap", 
                                                           list(PublicationStylerCu.COLORMAPS.keys()),
                                                           index=list(PublicationStylerCu.COLORMAPS.keys()).index('plasma'))
            with col2:
                style_params['plastic_cmap'] = st.selectbox("Plastic Strain Colormap", 
                                                           list(PublicationStylerCu.COLORMAPS.keys()),
                                                           index=list(PublicationStylerCu.COLORMAPS.keys()).index('Blues'))
                style_params['yield_cmap'] = st.selectbox("Yield Stress Colormap", 
                                                         list(PublicationStylerCu.COLORMAPS.keys()),
                                                         index=list(PublicationStylerCu.COLORMAPS.keys()).index('Greens'))
                style_params['grain_cmap'] = st.selectbox("Grain Colormap", 
                                                         list(PublicationStylerCu.COLORMAPS.keys()),
                                                         index=list(PublicationStylerCu.COLORMAPS.keys()).index('Set1'))
        
        with st.sidebar.expander("📰 Journal Templates", expanded=False):
            journal = st.selectbox(
                "Journal Style",
                ["Nature", "Science", "Acta Materialia", "Physical Review Letters", 
                 "Advanced Materials", "Custom"],
                index=0,
                key="pub_journal_style"
            )
            style_params['journal_style'] = journal.lower()
            style_params['use_latex'] = st.checkbox("Use LaTeX Formatting", False)
            style_params['vector_output'] = st.checkbox("Enable Vector Export (PDF/SVG)", True)
            style_params['figure_dpi'] = st.select_slider("Figure DPI", options=[150, 300, 600, 1200], value=600)
        
        return style_params
    
    @staticmethod
    def create_publication_figure(fig, axes, style_params):
        """Create publication-ready figure with all styling applied"""
        # Apply journal style
        fig, _ = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))
        
        # Apply custom styling
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        for ax in axes_flat:
            if ax is not None:
                # Apply title styling
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
        
        return fig

# ============================================================================
# ENHANCED LINE PROFILE SYSTEM - For detailed analysis
# ============================================================================
class EnhancedLineProfilerCu:
    """Advanced line profile system for nanotwin analysis"""
    
    @staticmethod
    def extract_profile(data, profile_type='horizontal', position_ratio=0.5, angle_deg=0, dx=0.5):
        """
        Extract line profiles from 2D data with proper scaling
        
        Parameters:
        -----------
        data : 2D numpy array
            Input data (stress, twin parameter, spacing, etc.)
        profile_type : str
            Type of profile: 'horizontal', 'vertical', 'diagonal', 'anti_diagonal', 'custom'
        position_ratio : float
            Position ratio from center (0.0 to 1.0)
        angle_deg : float
            Angle for custom line profiles (degrees)
        dx : float
            Grid spacing (nm)
            
        Returns:
        --------
        distance : 1D array
            Distance along profile (nm)
        profile : 1D array
            Extracted profile values
        endpoints : tuple
            (x_start, y_start, x_end, y_end) in data coordinates
        """
        N = data.shape[0]
        center = N // 2
        
        # Calculate position offset based on ratio
        offset = int(N * 0.4 * position_ratio)
        
        if profile_type == 'horizontal':
            # Horizontal profile at fixed y
            y_idx = center + offset
            profile = data[y_idx, :]
            distance = np.arange(N) * dx - (N//2) * dx
            endpoints = (0, y_idx*dx, (N-1)*dx, y_idx*dx)
            
        elif profile_type == 'vertical':
            # Vertical profile at fixed x
            x_idx = center + offset
            profile = data[:, x_idx]
            distance = np.arange(N) * dx - (N//2) * dx
            endpoints = (x_idx*dx, 0, x_idx*dx, (N-1)*dx)
            
        elif profile_type == 'diagonal':
            # Main diagonal (top-left to bottom-right)
            profile = np.diag(data)
            distance = np.arange(N) * dx * np.sqrt(2) - (N//2) * dx * np.sqrt(2)
            endpoints = (0, 0, (N-1)*dx, (N-1)*dx)
            
        elif profile_type == 'anti_diagonal':
            # Anti-diagonal (top-right to bottom-left)
            profile = np.diag(np.fliplr(data))
            distance = np.arange(N) * dx * np.sqrt(2) - (N//2) * dx * np.sqrt(2)
            endpoints = ((N-1)*dx, 0, 0, (N-1)*dx)
            
        elif profile_type == 'custom':
            # Custom angle line profile
            angle_rad = np.deg2rad(angle_deg)
            length = int(N * 0.8)
            
            # Start point
            start_x = center - int(length/2 * np.cos(angle_rad)) + offset * np.cos(angle_rad + np.pi/2)
            start_y = center - int(length/2 * np.sin(angle_rad)) + offset * np.sin(angle_rad + np.pi/2)
            
            profile = []
            distances = []
            
            # Sample along line
            for t in np.linspace(-length/2, length/2, length):
                x = start_x + t * np.cos(angle_rad)
                y = start_y + t * np.sin(angle_rad)
                
                if 0 <= x < N-1 and 0 <= y < N-1:
                    # Bilinear interpolation
                    x0, y0 = int(x), int(y)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    # Check bounds
                    if x1 >= N: x1 = N - 1
                    if y1 >= N: y1 = N - 1
                    
                    # Interpolation weights
                    wx = x - x0
                    wy = y - y0
                    
                    # Bilinear interpolation
                    val = (data[y0, x0] * (1-wx) * (1-wy) +
                          data[y0, x1] * wx * (1-wy) +
                          data[y1, x0] * (1-wx) * wy +
                          data[y1, x1] * wx * wy)
                    
                    profile.append(val)
                    distances.append(t * dx)
            
            distance = np.array(distances)
            profile = np.array(profile)
            
            # Calculate endpoints
            end_x = start_x + (length-1) * np.cos(angle_rad)
            end_y = start_y + (length-1) * np.sin(angle_rad)
            endpoints = (start_x*dx, start_y*dx, end_x*dx, end_y*dx)
        
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        return distance, profile, endpoints
    
    @staticmethod
    def extract_multiple_profiles(data, profile_types, position_ratio=0.5, angle_deg=45, dx=0.5):
        """Extract multiple line profiles from the same data"""
        profiles = {}
        for profile_type in profile_types:
            distance, profile, endpoints = EnhancedLineProfilerCu.extract_profile(
                data, profile_type, position_ratio, angle_deg, dx
            )
            profiles[profile_type] = {
                'distance': distance,
                'profile': profile,
                'endpoints': endpoints
            }
        return profiles
    
    @staticmethod
    def create_profile_comparison_plot(data_dict, profile_config, style_params):
        """Create comparison plot of multiple profiles from different datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(data_dict)))
        
        # Plot 1: Horizontal profiles
        ax1 = axes[0, 0]
        for idx, (name, data) in enumerate(data_dict.items()):
            distance, profile, _ = EnhancedLineProfilerCu.extract_profile(
                data, 'horizontal', profile_config['position_ratio'], dx=profile_config['dx']
            )
            ax1.plot(distance, profile, color=colors[idx], linewidth=2, alpha=0.8, label=name)
        ax1.set_xlabel('Distance (nm)')
        ax1.set_ylabel('Value')
        ax1.set_title('Horizontal Profiles')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Vertical profiles
        ax2 = axes[0, 1]
        for idx, (name, data) in enumerate(data_dict.items()):
            distance, profile, _ = EnhancedLineProfilerCu.extract_profile(
                data, 'vertical', profile_config['position_ratio'], dx=profile_config['dx']
            )
            ax2.plot(distance, profile, color=colors[idx], linewidth=2, alpha=0.8, label=name)
        ax2.set_xlabel('Distance (nm)')
        ax2.set_ylabel('Value')
        ax2.set_title('Vertical Profiles')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Diagonal profiles
        ax3 = axes[1, 0]
        for idx, (name, data) in enumerate(data_dict.items()):
            distance, profile, _ = EnhancedLineProfilerCu.extract_profile(
                data, 'diagonal', profile_config['position_ratio'], dx=profile_config['dx']
            )
            ax3.plot(distance, profile, color=colors[idx], linewidth=2, alpha=0.8, label=name)
        ax3.set_xlabel('Distance (nm)')
        ax3.set_ylabel('Value')
        ax3.set_title('Diagonal Profiles')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Profile statistics
        ax4 = axes[1, 1]
        profile_stats = []
        for name, data in data_dict.items():
            stats_dict = {
                'name': name,
                'mean': np.mean(data),
                'std': np.std(data),
                'max': np.max(data),
                'min': np.min(data),
                'range': np.max(data) - np.min(data)
            }
            profile_stats.append(stats_dict)
        
        # Create bar plot
        names = [s['name'] for s in profile_stats]
        means = [s['mean'] for s in profile_stats]
        stds = [s['std'] for s in profile_stats]
        
        x_pos = np.arange(len(names))
        ax4.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(names)], alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(names, rotation=45, ha='right')
        ax4.set_ylabel('Mean Value')
        ax4.set_title('Statistical Comparison')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Apply publication styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig

# ============================================================================
# ADVANCED COMPARISON SYSTEM - Multi-simulation analysis
# ============================================================================
class SimulationComparatorCu:
    """Advanced comparison system for multiple nanotwin simulations"""
    
    @staticmethod
    def create_multi_simulation_dashboard(simulations, config, style_params):
        """Create comprehensive dashboard comparing multiple simulations"""
        n_sims = len(simulations)
        
        # Create main figure with subplots
        if n_sims <= 3:
            cols = n_sims
            rows = 1
            fig_size = (5 * cols, 4 * rows)
        else:
            cols = 3
            rows = (n_sims + 2) // 3
            fig_size = (5 * cols, 4 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, constrained_layout=True)
        
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get custom colormaps
        custom_cmaps = PublicationStylerCu.create_custom_colormaps()
        
        for idx, sim in enumerate(simulations):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get simulation data
            sim_data = SimulationDBCu.get_simulation(sim['id'])
            if not sim_data or not sim_data['results_history']:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get frame
            frame_idx = min(config.get('frame_idx', 0), len(sim_data['results_history']) - 1)
            results = sim_data['results_history'][frame_idx]
            params = sim_data['params']
            
            # Select visualization based on config
            if config['visualization_type'] == 'stress_magnitude':
                data = results['sigma_eq'] / 1e9  # Convert to GPa
                cmap = custom_cmaps.get('stress_map', plt.cm.hot)
                vmin, vmax = 0, np.percentile(data, 95)
                title = f"σ_eq (GPa)\nλ={params.get('twin_spacing', 0):.1f}nm"
                colorbar_label = 'σ_eq (GPa)'
                
            elif config['visualization_type'] == 'twin_structure':
                data = results['phi']
                cmap = custom_cmaps.get('twin_phi_map', plt.cm.RdBu_r)
                vmin, vmax = -1.2, 1.2
                title = f"Twin φ\nσ={params.get('applied_stress', 0)/1e6:.0f}MPa"
                colorbar_label = 'φ'
                
            elif config['visualization_type'] == 'twin_spacing':
                data = results['h']
                cmap = custom_cmaps.get('spacing_map', plt.cm.plasma)
                vmin, vmax = 0, 30
                title = f"Twin Spacing (nm)\nW={params.get('W', 0):.1f}"
                colorbar_label = 'Spacing (nm)'
                
            elif config['visualization_type'] == 'yield_stress':
                data = results['sigma_y'] / 1e6  # Convert to MPa
                cmap = custom_cmaps.get('yield_map', plt.cm.Greens)
                vmin, vmax = 0, np.percentile(data, 95)
                title = f"Yield Stress (MPa)\nL_CTB={params.get('L_CTB', 0):.3f}"
                colorbar_label = 'σ_y (MPa)'
                
            elif config['visualization_type'] == 'plastic_strain':
                data = results['eps_p_mag']
                cmap = custom_cmaps.get('plastic_map', plt.cm.Blues)
                vmin, vmax = 0, np.percentile(data, 95)
                title = f"Plastic Strain\nζ={params.get('zeta', 0):.2f}"
                colorbar_label = 'ε_p'
                
            else:
                data = results['sigma_eq'] / 1e9
                cmap = plt.cm.hot
                vmin, vmax = 0, np.percentile(data, 95)
                title = f"Simulation {idx+1}"
                colorbar_label = 'Value'
            
            # Create heatmap
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title, fontsize=10, pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            if col == cols - 1 or idx == len(simulations) - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(colorbar_label, fontsize=8)
        
        # Hide empty subplots
        for idx in range(n_sims, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        # Apply publication styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig
    
    @staticmethod
    def create_evolution_comparison_plot(simulations, config, style_params):
        """Create comprehensive evolution comparison plot"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
        
        for idx, sim in enumerate(simulations):
            sim_data = SimulationDBCu.get_simulation(sim['id'])
            if not sim_data:
                continue
            
            history = sim_data['history']
            params = sim_data['params']
            
            # Extract time evolution
            if history and 'phi_norm' in history:
                n_steps = len(history['phi_norm'])
                time_steps = np.arange(n_steps) * params.get('dt', 1e-4)
                
                # Plot 1: Twin order parameter evolution
                axes[0, 0].plot(time_steps, history['phi_norm'], 
                              color=colors[idx], linewidth=2, alpha=0.7,
                              label=f"{params.get('geometry_type', 'Sim')} {idx+1}")
                
                # Plot 2: Maximum stress evolution
                if 'max_stress' in history:
                    axes[0, 1].plot(time_steps, np.array(history['max_stress']) / 1e9, 
                                  color=colors[idx], linewidth=2, alpha=0.7)
                
                # Plot 3: Energy evolution
                if 'energy' in history:
                    axes[0, 2].plot(time_steps, history['energy'], 
                                  color=colors[idx], linewidth=2, alpha=0.7)
                
                # Plot 4: Plastic work evolution
                if 'plastic_work' in history:
                    axes[1, 0].plot(time_steps, history['plastic_work'], 
                                  color=colors[idx], linewidth=2, alpha=0.7)
                
                # Plot 5: Average twin spacing evolution
                if 'twin_spacing_avg' in history:
                    axes[1, 1].plot(time_steps, history['twin_spacing_avg'], 
                                  color=colors[idx], linewidth=2, alpha=0.7)
                
                # Plot 6: Average stress evolution
                if 'avg_stress' in history:
                    axes[1, 2].plot(time_steps, np.array(history['avg_stress']) / 1e9, 
                                  color=colors[idx], linewidth=2, alpha=0.7)
        
        # Label plots
        axes[0, 0].set_xlabel('Time (ns)', fontsize=10)
        axes[0, 0].set_ylabel('||φ||', fontsize=10)
        axes[0, 0].set_title('Twin Order Parameter Evolution', fontsize=11)
        axes[0, 0].legend(fontsize=8, loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Time (ns)', fontsize=10)
        axes[0, 1].set_ylabel('Max Stress (GPa)', fontsize=10)
        axes[0, 1].set_title('Maximum Stress Evolution', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_xlabel('Time (ns)', fontsize=10)
        axes[0, 2].set_ylabel('Total Energy (J)', fontsize=10)
        axes[0, 2].set_title('System Energy Evolution', fontsize=11)
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Time (ns)', fontsize=10)
        axes[1, 0].set_ylabel('Plastic Work (J)', fontsize=10)
        axes[1, 0].set_title('Plastic Work Evolution', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Time (ns)', fontsize=10)
        axes[1, 1].set_ylabel('Avg Twin Spacing (nm)', fontsize=10)
        axes[1, 1].set_title('Twin Spacing Evolution', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].set_xlabel('Time (ns)', fontsize=10)
        axes[1, 2].set_ylabel('Avg Stress (GPa)', fontsize=10)
        axes[1, 2].set_title('Average Stress Evolution', fontsize=11)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Apply publication styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig
    
    @staticmethod
    def create_statistical_comparison(simulations, config, style_params):
        """Create statistical comparison of multiple simulations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        
        # Collect statistics
        stats_data = []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
        
        for idx, sim in enumerate(simulations):
            sim_data = SimulationDBCu.get_simulation(sim['id'])
            if not sim_data:
                continue
            
            stats = sim_data.get('statistics', {})
            params = sim_data['params']
            
            stats_data.append({
                'id': sim['id'],
                'name': f"Sim {idx+1}",
                'color': colors[idx],
                'twin_spacing': params.get('twin_spacing', 0),
                'applied_stress': params.get('applied_stress', 0) / 1e6,  # MPa
                'max_stress': stats.get('stress_statistics', {}).get('max_stress_MPa', 0),
                'mean_stress': stats.get('stress_statistics', {}).get('mean_stress_MPa', 0),
                'avg_spacing': stats.get('twin_statistics', {}).get('avg_spacing_nm', 0),
                'plastic_work': stats.get('plasticity_statistics', {}).get('total_plastic_work_J', 0),
                'final_phi_norm': stats.get('convergence_statistics', {}).get('final_phi_norm', 0),
            })
        
        if not stats_data:
            return fig
        
        df = pd.DataFrame(stats_data)
        
        # Plot 1: Twin spacing vs max stress
        scatter1 = axes[0, 0].scatter(df['twin_spacing'], df['max_stress'], 
                                     c=df['applied_stress'], cmap='viridis', 
                                     s=100, alpha=0.7, edgecolors='k')
        axes[0, 0].set_xlabel('Twin Spacing (nm)', fontsize=10)
        axes[0, 0].set_ylabel('Max Stress (MPa)', fontsize=10)
        axes[0, 0].set_title('Twin Spacing Effect', fontsize=11)
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('Applied Stress (MPa)', fontsize=9)
        
        # Plot 2: Applied stress vs max stress
        scatter2 = axes[0, 1].scatter(df['applied_stress'], df['max_stress'], 
                                     c=df['twin_spacing'], cmap='plasma', 
                                     s=100, alpha=0.7, edgecolors='k')
        axes[0, 1].set_xlabel('Applied Stress (MPa)', fontsize=10)
        axes[0, 1].set_ylabel('Max Stress (MPa)', fontsize=10)
        axes[0, 1].set_title('Stress Response', fontsize=11)
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
        cbar2.set_label('Twin Spacing (nm)', fontsize=9)
        
        # Plot 3: Bar plot of plastic work
        x_pos = np.arange(len(df))
        bars = axes[0, 2].bar(x_pos, df['plastic_work'], color=df['color'], alpha=0.7, edgecolor='k')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(df['name'], rotation=45, ha='right', fontsize=9)
        axes[0, 2].set_ylabel('Plastic Work (J)', fontsize=10)
        axes[0, 2].set_title('Plastic Work Comparison', fontsize=11)
        
        # Add value labels on bars
        for bar, val in zip(bars, df['plastic_work']):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2e}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Correlation matrix
        corr_cols = ['twin_spacing', 'applied_stress', 'max_stress', 'avg_spacing', 'plastic_work', 'final_phi_norm']
        corr_matrix = df[corr_cols].corr()
        
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1, 0].set_title('Parameter Correlation Matrix', fontsize=11)
        axes[1, 0].set_xticks(np.arange(len(corr_cols)))
        axes[1, 0].set_yticks(np.arange(len(corr_cols)))
        axes[1, 0].set_xticklabels([c.replace('_', '\n') for c in corr_cols], fontsize=8)
        axes[1, 0].set_yticklabels([c.replace('_', '\n') for c in corr_cols], fontsize=8)
        
        # Add correlation values
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                                     fontsize=8, fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 5: Parallel coordinates
        ax5 = axes[1, 1]
        # Normalize data for parallel coordinates
        normalized_df = df.copy()
        for col in ['twin_spacing', 'applied_stress', 'max_stress', 'avg_spacing', 'plastic_work', 'final_phi_norm']:
            if normalized_df[col].max() > normalized_df[col].min():
                normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
        
        for idx, row in normalized_df.iterrows():
            values = [row['twin_spacing'], row['applied_stress'], row['max_stress'], 
                     row['avg_spacing'], row['plastic_work'], row['final_phi_norm']]
            ax5.plot(range(len(values)), values, color=row['color'], linewidth=2, alpha=0.7, marker='o')
        
        ax5.set_xticks(range(len(values)))
        ax5.set_xticklabels(['λ', 'σ_app', 'σ_max', 'h_avg', 'W_p', '||φ||'], fontsize=9)
        ax5.set_ylabel('Normalized Value', fontsize=10)
        ax5.set_title('Parallel Coordinates Plot', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: 3D scatter (projection)
        ax6 = axes[1, 2]
        scatter3d = ax6.scatter(df['twin_spacing'], df['applied_stress'], df['max_stress'],
                               c=df['plastic_work'], cmap='hot', s=100, alpha=0.7, edgecolors='k')
        ax6.set_xlabel('Twin Spacing (nm)', fontsize=10)
        ax6.set_ylabel('Applied Stress (MPa)', fontsize=10)
        ax6.set_zlabel('Max Stress (MPa)', fontsize=10)
        ax6.set_title('3D Parameter Space', fontsize=11)
        cbar6 = plt.colorbar(scatter3d, ax=ax6)
        cbar6.set_label('Plastic Work (J)', fontsize=9)
        
        # Apply publication styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig
    
    @staticmethod
    def create_parameter_sensitivity_analysis(simulations, sweep_variable, style_params):
        """Create parameter sensitivity analysis plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        
        # Extract parameter values and results
        data = []
        for sim in simulations:
            sim_data = SimulationDBCu.get_simulation(sim['id'])
            if not sim_data:
                continue
            
            params = sim_data['params']
            stats = sim_data.get('statistics', {})
            
            data_point = {
                'twin_spacing': params.get('twin_spacing', 0),
                'applied_stress': params.get('applied_stress', 0) / 1e6,
                'W': params.get('W', 0),
                'L_CTB': params.get('L_CTB', 0),
                'zeta': params.get('zeta', 0),
                'max_stress': stats.get('stress_statistics', {}).get('max_stress_MPa', 0),
                'avg_spacing': stats.get('twin_statistics', {}).get('avg_spacing_nm', 0),
                'plastic_work': stats.get('plasticity_statistics', {}).get('total_plastic_work_J', 0),
                'final_phi_norm': stats.get('convergence_statistics', {}).get('final_phi_norm', 0),
            }
            data.append(data_point)
        
        if not data:
            return fig
        
        df = pd.DataFrame(data)
        
        # Determine which variable was swept
        if sweep_variable == 'twin_spacing':
            x_var = 'twin_spacing'
            x_label = 'Twin Spacing (nm)'
        elif sweep_variable == 'applied_stress':
            x_var = 'applied_stress'
            x_label = 'Applied Stress (MPa)'
        elif sweep_variable == 'W':
            x_var = 'W'
            x_label = 'Twin Well Depth W (J/m³)'
        elif sweep_variable == 'L_CTB':
            x_var = 'L_CTB'
            x_label = 'CTB Mobility L_CTB'
        else:
            x_var = 'twin_spacing'
            x_label = 'Twin Spacing (nm)'
        
        # Sort by x variable
        df = df.sort_values(by=x_var)
        
        # Plot 1: Parameter vs Max Stress
        axes[0, 0].plot(df[x_var], df['max_stress'], 'bo-', linewidth=2, markersize=8, label='Max Stress')
        axes[0, 0].fill_between(df[x_var], df['max_stress'] * 0.9, df['max_stress'] * 1.1, alpha=0.2, color='blue')
        axes[0, 0].set_xlabel(x_label, fontsize=10)
        axes[0, 0].set_ylabel('Max Stress (MPa)', fontsize=10)
        axes[0, 0].set_title(f'{sweep_variable} vs Maximum Stress', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=9)
        
        # Plot 2: Parameter vs Plastic Work
        axes[0, 1].plot(df[x_var], df['plastic_work'], 'ro-', linewidth=2, markersize=8, label='Plastic Work')
        axes[0, 1].set_xlabel(x_label, fontsize=10)
        axes[0, 1].set_ylabel('Plastic Work (J)', fontsize=10)
        axes[0, 1].set_title(f'{sweep_variable} vs Plastic Work', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=9)
        
        # Plot 3: Parameter vs Average Twin Spacing
        axes[1, 0].plot(df[x_var], df['avg_spacing'], 'go-', linewidth=2, markersize=8, label='Avg Spacing')
        axes[1, 0].set_xlabel(x_label, fontsize=10)
        axes[1, 0].set_ylabel('Average Twin Spacing (nm)', fontsize=10)
        axes[1, 0].set_title(f'{sweep_variable} vs Twin Spacing', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=9)
        
        # Plot 4: Parameter vs Final Phi Norm
        axes[1, 1].plot(df[x_var], df['final_phi_norm'], 'mo-', linewidth=2, markersize=8, label='Final ||φ||')
        axes[1, 1].set_xlabel(x_label, fontsize=10)
        axes[1, 1].set_ylabel('Final Twin Order Parameter Norm', fontsize=10)
        axes[1, 1].set_title(f'{sweep_variable} vs Twin Structure', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=9)
        
        # Apply publication styling
        fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
        
        return fig

# ============================================================================
# ENHANCED EXPORT SYSTEM - Complete export functionality
# ============================================================================
class EnhancedExporterCu:
    """Complete export system for nanotwin simulations"""
    
    @staticmethod
    def export_publication_package(simulations, figures, config, style_params):
        """Export complete publication package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"nanotwin_publication_{timestamp}"
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Export figures in multiple formats
            for fig_idx, fig in enumerate(figures):
                # PDF (vector)
                pdf_buffer = BytesIO()
                fig.savefig(pdf_buffer, format='pdf', dpi=style_params.get('dpi', 600), 
                           bbox_inches='tight')
                zf.writestr(f"{package_name}/figure_{fig_idx+1}.pdf", pdf_buffer.getvalue())
                
                # PNG (high-res)
                png_buffer = BytesIO()
                fig.savefig(png_buffer, format='png', dpi=style_params.get('dpi', 600), 
                           bbox_inches='tight')
                zf.writestr(f"{package_name}/figure_{fig_idx+1}.png", png_buffer.getvalue())
                
                # SVG (vector)
                svg_buffer = BytesIO()
                fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
                zf.writestr(f"{package_name}/figure_{fig_idx+1}.svg", svg_buffer.getvalue())
            
            # 2. Export simulation data
            for sim_idx, sim in enumerate(simulations):
                sim_data = SimulationDBCu.get_simulation(sim['id'])
                if not sim_data:
                    continue
                
                sim_dir = f"{package_name}/simulation_{sim_idx+1}_{sim['id']}"
                
                # Export parameters
                params_json = json.dumps(sim_data['params'], indent=2)
                zf.writestr(f"{sim_dir}/parameters.json", params_json)
                
                # Export metadata
                metadata_json = json.dumps(sim_data['metadata'], indent=2)
                zf.writestr(f"{sim_dir}/metadata.json", metadata_json)
                
                # Export statistics
                stats_json = json.dumps(sim_data.get('statistics', {}), indent=2)
                zf.writestr(f"{sim_dir}/statistics.json", stats_json)
                
                # Export convergence history
                if sim_data['history']:
                    history_df = pd.DataFrame(sim_data['history'])
                    zf.writestr(f"{sim_dir}/convergence_history.csv", history_df.to_csv(index=False))
                
                # Export final frame data
                if sim_data['results_history']:
                    final_results = sim_data['results_history'][-1]
                    for key, value in final_results.items():
                        if isinstance(value, np.ndarray):
                            npz_buffer = BytesIO()
                            np.savez_compressed(npz_buffer, value)
                            zf.writestr(f"{sim_dir}/final_{key}.npz", npz_buffer.getvalue())
            
            # 3. Export configuration
            config_json = json.dumps(config, indent=2)
            zf.writestr(f"{package_name}/configuration.json", config_json)
            
            # 4. Export styling parameters
            style_json = json.dumps(style_params, indent=2)
            zf.writestr(f"{package_name}/styling_parameters.json", style_json)
            
            # 5. Create README file
            readme_content = f"""NANOTWIN SIMULATION PUBLICATION PACKAGE
============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Package: {package_name}

CONTENTS:
---------
1. Figures (PDF, PNG, SVG formats)
2. Simulation data and parameters
3. Configuration files
4. Styling parameters

SIMULATIONS INCLUDED:
---------------------
"""
            for sim_idx, sim in enumerate(simulations):
                sim_data = SimulationDBCu.get_simulation(sim['id'])
                if sim_data:
                    params = sim_data['params']
                    readme_content += f"\nSimulation {sim_idx+1} (ID: {sim['id']}):"
                    readme_content += f"\n  Geometry: {params.get('geometry_type', 'Unknown')}"
                    readme_content += f"\n  Twin Spacing: {params.get('twin_spacing', 0):.1f} nm"
                    readme_content += f"\n  Applied Stress: {params.get('applied_stress', 0)/1e6:.0f} MPa"
                    readme_content += f"\n  Created: {sim_data['metadata'].get('created_at', 'Unknown')}\n"
            
            readme_content += f"""

PUBLICATION SETTINGS:
---------------------
Journal Style: {style_params.get('journal_style', 'nature')}
DPI: {style_params.get('dpi', 600)}
Vector Export: {style_params.get('vector_output', True)}

FIGURE INFORMATION:
-------------------
Total Figures: {len(figures)}
Figure Formats: PDF, PNG, SVG

EXPORT NOTES:
-------------
This package contains all data necessary to reproduce the figures and analysis.
For questions or support, please refer to the configuration files included.

"""
            zf.writestr(f"{package_name}/README.txt", readme_content)
        
        zip_buffer.seek(0)
        return zip_buffer, package_name
    
    @staticmethod
    def export_simulation_data(simulations, format='zip'):
        """Export simulation data in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'zip':
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for sim in simulations:
                    sim_data = SimulationDBCu.get_simulation(sim['id'])
                    if not sim_data:
                        continue
                    
                    sim_id = sim['id']
                    sim_dir = f"simulation_{sim_id}"
                    
                    # Export all data
                    for export_format in ['json', 'csv', 'npz']:
                        if export_format == 'json':
                            # Export as JSON
                            export_data = {
                                'parameters': sim_data['params'],
                                'metadata': sim_data['metadata'],
                                'statistics': sim_data.get('statistics', {}),
                            }
                            json_str = json.dumps(export_data, indent=2)
                            zf.writestr(f"{sim_dir}/data.json", json_str)
                        
                        elif export_format == 'csv':
                            # Export convergence history as CSV
                            if sim_data['history']:
                                history_df = pd.DataFrame(sim_data['history'])
                                zf.writestr(f"{sim_dir}/history.csv", history_df.to_csv(index=False))
                            
                            # Export summary statistics as CSV
                            summary_data = {
                                'simulation_id': [sim_id],
                                'twin_spacing_nm': [sim_data['params'].get('twin_spacing', 0)],
                                'applied_stress_MPa': [sim_data['params'].get('applied_stress', 0)/1e6],
                                'max_stress_MPa': [sim_data.get('statistics', {}).get('stress_statistics', {}).get('max_stress_MPa', 0)],
                                'avg_spacing_nm': [sim_data.get('statistics', {}).get('twin_statistics', {}).get('avg_spacing_nm', 0)],
                                'plastic_work_J': [sim_data.get('statistics', {}).get('plasticity_statistics', {}).get('total_plastic_work_J', 0)],
                            }
                            summary_df = pd.DataFrame(summary_data)
                            zf.writestr(f"{sim_dir}/summary.csv", summary_df.to_csv(index=False))
                        
                        elif export_format == 'npz':
                            # Export final frame arrays
                            if sim_data['results_history']:
                                final_results = sim_data['results_history'][-1]
                                npz_buffer = BytesIO()
                                np.savez_compressed(npz_buffer, **final_results)
                                zf.writestr(f"{sim_dir}/final_results.npz", npz_buffer.getvalue())
            
            zip_buffer.seek(0)
            return zip_buffer
        
        elif format == 'hdf5':
            # HDF5 export (placeholder - would require h5py)
            st.warning("HDF5 export requires h5py package. Please install with: pip install h5py")
            return None
        
        elif format == 'excel':
            # Excel export
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                for sim in simulations:
                    sim_data = SimulationDBCu.get_simulation(sim['id'])
                    if not sim_data:
                        continue
                    
                    # Write parameters sheet
                    params_df = pd.DataFrame([sim_data['params']])
                    params_df.to_excel(writer, sheet_name=f"{sim['id'][:6]}_params", index=False)
                    
                    # Write statistics sheet
                    stats_df = pd.DataFrame([sim_data.get('statistics', {})])
                    stats_df.to_excel(writer, sheet_name=f"{sim['id'][:6]}_stats", index=False)
                    
                    # Write history sheet
                    if sim_data['history']:
                        history_df = pd.DataFrame(sim_data['history'])
                        history_df.to_excel(writer, sheet_name=f"{sim['id'][:6]}_history", index=False)
            
            excel_buffer.seek(0)
            return excel_buffer
    
    @staticmethod
    def export_line_profile_data(profiles, config):
        """Export line profile data for further analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for profile_name, profile_data in profiles.items():
                # Export as CSV
                df = pd.DataFrame({
                    'distance_nm': profile_data['distance'],
                    'value': profile_data['profile']
                })
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                zf.writestr(f"profiles/{profile_name}.csv", csv_buffer.getvalue())
                
                # Export as JSON
                json_data = {
                    'name': profile_name,
                    'distance': profile_data['distance'].tolist(),
                    'profile': profile_data['profile'].tolist(),
                    'endpoints': profile_data['endpoints'],
                    'config': config
                }
                json_str = json.dumps(json_data, indent=2)
                zf.writestr(f"profiles/{profile_name}.json", json_str)
        
        zip_buffer.seek(0)
        return zip_buffer

# ============================================================================
# PHYSICS MODELS - Original code with enhanced error handling
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
    """Numba-compatible plastic strain computation"""
    eps_p_xx_new = np.zeros((N, N))
    eps_p_yy_new = np.zeros((N, N))
    eps_p_xy_new = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            if sigma_eq[i, j] > sigma_y[i, j]:
                overstress = (sigma_eq[i, j] - sigma_y[i, j]) / sigma_y[i, j]
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
# MAIN APPLICATION - Enhanced with complete data flow
# ============================================================================
def main():
    """Main Streamlit application with complete data flow system"""
    st.set_page_config(
        page_title="Nanotwinned Cu Phase-Field Simulator Pro",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': "## Nanotwinned Copper Phase-Field Simulator\nAdvanced simulation platform for nanotwin mechanics."
        }
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
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: white;
        padding: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #F59E0B;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #10B981;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #DC2626;
        margin: 0.5rem 0;
    }
    .tab-content {
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
        st.session_state.simulation_counter = 0
        st.session_state.simulation_tags = {}
    
    # Header with mode selector
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🔬 Nanotwinned Cu Phase-Field Simulator Pro</h1>', unsafe_allow_html=True)
        
        # Operation mode selector
        operation_mode = st.radio(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations", "Database Management", 
             "Parameter Sweep Study", "Advanced Analysis"],
            horizontal=True,
            index=0
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =============================================
    # MODE 1: RUN NEW SIMULATION
    # =============================================
    if operation_mode == "Run New Simulation":
        st.header("🚀 Run New Simulation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("📋 Simulation Parameters", expanded=True):
                # Create tabs for parameter categories
                param_tabs = st.tabs(["Geometry", "Material", "Loading", "Numerical"])
                
                with param_tabs[0]:  # Geometry
                    geometry_type = st.selectbox(
                        "Geometry Type",
                        ["Standard Twin Grain", "Twin Grain with Defect", "Multi-grain Structure", "Custom Pattern"],
                        index=0
                    )
                    
                    twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0, 
                                            help="Initial distance between twin boundaries")
                    grain_boundary_pos = st.slider("Grain boundary position (nm)", -50.0, 50.0, 0.0, 1.0,
                                                  help="Location of grain boundary")
                    
                    if geometry_type in ["Twin Grain with Defect", "Custom Pattern"]:
                        defect_type = st.selectbox("Defect Type", ["Dislocation", "Void", "Crack", "Inclusion"])
                        defect_x = st.slider("Defect X position (nm)", -50.0, 50.0, 0.0, 1.0)
                        defect_y = st.slider("Defect Y position (nm)", -50.0, 50.0, 0.0, 1.0)
                        defect_radius = st.slider("Defect radius (nm)", 5.0, 30.0, 10.0, 1.0)
                
                with param_tabs[1]:  # Material
                    W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1,
                                 help="Controls twin boundary energy")
                    A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5,
                                 help="Controls grain boundary energy")
                    B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5,
                                 help="Prevents grain overlap")
                    
                    kappa0 = st.slider("κ₀ (Gradient energy ref)", 0.01, 10.0, 1.0, 0.1,
                                      help="Baseline gradient energy coefficient")
                    gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05,
                                           help="Controls anisotropy between CTBs and ITBs")
                    kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1,
                                         help="Grain boundary gradient energy")
                
                with param_tabs[2]:  # Loading
                    applied_stress_MPa = st.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0,
                                                  help="External applied stress")
                    
                    loading_type = st.selectbox("Loading Type", ["Constant Stress", "Cyclic Loading", 
                                                               "Stress Ramp", "Strain Control"])
                    
                    if loading_type == "Cyclic Loading":
                        cycle_amplitude = st.slider("Stress amplitude (MPa)", 50.0, 500.0, 150.0, 10.0)
                        cycle_frequency = st.slider("Frequency (Hz)", 0.1, 100.0, 1.0, 0.1)
                    
                    L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001,
                                     help="Mobility of coherent twin boundaries")
                    L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1,
                                     help="Mobility of incoherent twin boundaries")
                    n_mob = st.slider("n (Mobility exponent)", 1, 10, 4, 1,
                                     help="Controls transition sharpness between CTB and ITB")
                    L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1,
                                     help="Grain boundary mobility")
                    zeta = st.slider("ζ (Dislocation pinning)", 0.0, 2.0, 0.3, 0.05,
                                    help="Strength of dislocation pinning")
                
                with param_tabs[3]:  # Numerical
                    N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64,
                                 help="Higher resolution = more accurate but slower")
                    dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1,
                                  help="Smaller spacing = finer details")
                    dt = st.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5,
                                  help="Smaller time step = more stable but slower")
                    
                    n_steps = st.slider("Number of steps", 10, 1000, 100, 10,
                                       help="Total simulation steps")
                    save_frequency = st.slider("Save frequency", 1, 100, 10, 1,
                                              help="How often to save results")
                    
                    stability_factor = st.slider("Stability factor", 0.1, 1.0, 0.5, 0.1,
                                                help="Controls numerical stability")
                    enable_monitoring = st.checkbox("Enable real-time monitoring", True,
                                                   help="Track convergence during simulation")
                    auto_adjust_dt = st.checkbox("Auto-adjust time step", True,
                                                help="Automatically adjust time step for stability")
            
            # Simulation metadata
            with st.expander("📝 Simulation Metadata", expanded=False):
                sim_name = st.text_input("Simulation Name", value=f"TwinSim_{datetime.now().strftime('%Y%m%d')}")
                sim_tags = st.multiselect("Tags", ["default", "high_stress", "small_spacing", 
                                                  "defect_study", "parameter_sweep", "validation"])
                sim_notes = st.text_area("Notes", value="", height=100,
                                        placeholder="Add any notes about this simulation...")
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("⚙️ Quick Settings")
            
            # Preset configurations
            preset = st.selectbox("Load Preset", 
                                 ["Standard Analysis", "High Resolution", "Quick Test", 
                                  "Defect Study", "Parameter Optimization"])
            
            if preset == "Standard Analysis":
                st.info("Standard analysis settings loaded")
            elif preset == "High Resolution":
                st.info("High resolution settings loaded (slower but more accurate)")
            elif preset == "Quick Test":
                st.info("Quick test settings loaded (faster but less accurate)")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Publication styling
            st.subheader("🎨 Publication Styling")
            style_params = PublicationStylerCu.get_styling_controls()
            
            # Run button
            st.markdown("---")
            if st.button("🚀 Run & Save Simulation", type="primary", use_container_width=True):
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
                        'geometry_type': geometry_type.lower().replace(' ', '_'),
                        'applied_stress': applied_stress_MPa * 1e6,
                        'loading_type': loading_type,
                        'save_frequency': save_frequency,
                        'n_steps': n_steps,
                        'stability_factor': stability_factor,
                    }
                    
                    # Add defect parameters if needed
                    if geometry_type in ["Twin Grain with Defect", "Custom Pattern"]:
                        params['defect_type'] = defect_type.lower()
                        params['defect_pos'] = (defect_x, defect_y)
                        params['defect_radius'] = defect_radius
                    
                    if loading_type == "Cyclic Loading":
                        params['cycle_amplitude'] = cycle_amplitude * 1e6
                        params['cycle_frequency'] = cycle_frequency
                    
                    # Create metadata
                    metadata = {
                        'name': sim_name,
                        'notes': sim_notes,
                        'tags': sim_tags,
                        'preset': preset,
                        'run_time': 0,  # Will be updated after simulation
                        'frames': n_steps // save_frequency,
                        'grid_size': N,
                        'dx': dx,
                        'created_by': 'User',  # Could be extended to track users
                        'version': '1.0.0',
                    }
                    
                    # Run simulation (placeholder - integrate your actual solver here)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create mock results for demonstration
                    mock_history = {
                        'phi_norm': np.random.rand(n_steps) * 0.5 + 0.5,
                        'energy': np.random.rand(n_steps) * 1e-10,
                        'max_stress': np.random.rand(n_steps) * 1e9,
                        'avg_stress': np.random.rand(n_steps) * 5e8,
                        'plastic_work': np.random.rand(n_steps) * 1e-12,
                        'twin_spacing_avg': np.random.rand(n_steps) * 10 + twin_spacing,
                    }
                    
                    mock_results_history = []
                    for i in range(0, n_steps, save_frequency):
                        # Create realistic-looking mock data
                        x = np.linspace(-N*dx/2, N*dx/2, N)
                        y = np.linspace(-N*dx/2, N*dx/2, N)
                        X, Y = np.meshgrid(x, y)
                        
                        # Create twin pattern
                        phi = np.tanh(np.sin(2 * np.pi * Y / twin_spacing) * 3.0)
                        
                        # Add evolution over time
                        evolution_factor = i / n_steps
                        phi = phi * (0.5 + 0.5 * evolution_factor)
                        
                        # Create stress field
                        sigma_eq = applied_stress_MPa * 1e6 * (0.8 + 0.4 * np.random.rand(N, N))
                        
                        # Create twin spacing
                        h = twin_spacing * (0.8 + 0.4 * np.random.rand(N, N))
                        
                        # Create yield stress
                        sigma_y = 200e6 * (0.9 + 0.2 * np.random.rand(N, N))
                        
                        # Create plastic strain
                        eps_p_mag = 0.01 * evolution_factor * np.random.rand(N, N)
                        
                        mock_results = {
                            'phi': phi,
                            'sigma_eq': sigma_eq,
                            'h': h,
                            'sigma_y': sigma_y,
                            'eps_p_mag': eps_p_mag,
                            'eta1': np.ones((N, N)) * 0.8,
                            'eta2': np.ones((N, N)) * 0.2
                        }
                        mock_results_history.append(mock_results)
                        
                        # Update progress
                        progress = (i + 1) / n_steps
                        progress_bar.progress(progress)
                        status_text.text(f"Step {i+1}/{n_steps} | Time: {(i+1)*dt:.4f} ns")
                    
                    # Update metadata with actual run time
                    metadata['run_time'] = 5.2  # Mock run time in seconds
                    
                    # Save to database
                    sim_id = SimulationDBCu.save_simulation(params, mock_history, metadata, mock_results_history)
                    
                    # Add tags from metadata
                    for tag in sim_tags:
                        SimulationDBCu.add_tag(sim_id, tag)
                    
                    # Display success message
                    st.success(f"""
                    ✅ Simulation Complete!
                    
                    **Details:**
                    - **ID**: `{sim_id}`
                    - **Name**: {sim_name}
                    - **Frames**: {len(mock_results_history)}
                    - **Run Time**: {metadata['run_time']:.2f} seconds
                    - **Tags**: {', '.join(sim_tags) if sim_tags else 'None'}
                    
                    **Next Steps:**
                    1. View results in the visualization tab
                    2. Compare with other simulations
                    3. Export for publication
                    """)
                    
                    # Show quick preview
                    st.subheader("📊 Quick Preview")
                    if mock_results_history:
                        final_results = mock_results_history[-1]
                        
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        
                        axes[0].imshow(final_results['phi'], cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                        axes[0].set_title('Final Twin Structure (φ)')
                        axes[0].set_xticks([])
                        axes[0].set_yticks([])
                        
                        axes[1].imshow(final_results['sigma_eq']/1e6, cmap='hot', vmin=0, vmax=500)
                        axes[1].set_title('Final Stress (MPa)')
                        axes[1].set_xticks([])
                        axes[1].set_yticks([])
                        
                        axes[2].imshow(final_results['h'], cmap='plasma', vmin=0, vmax=50)
                        axes[2].set_title('Final Twin Spacing (nm)')
                        axes[2].set_xticks([])
                        axes[2].set_yticks([])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
        
        # Database statistics
        with st.expander("📊 Database Statistics", expanded=False):
            db_stats = SimulationDBCu.get_database_stats()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Simulations", db_stats.get('total_simulations', 0))
            with col2:
                st.metric("Total Size", f"{db_stats.get('total_size_mb', 0):.1f} MB")
            with col3:
                st.metric("Avg Size", f"{db_stats.get('avg_simulation_size_mb', 0):.1f} MB")
            with col4:
                st.metric("Favorites", db_stats.get('favorites_count', 0))
    
    # =============================================
    # MODE 2: COMPARE SAVED SIMULATIONS
    # =============================================
    elif operation_mode == "Compare Saved Simulations":
        st.header("🔍 Compare Saved Simulations")
        
        # Get available simulations
        simulations = SimulationDBCu.get_simulation_list()
        
        if not simulations:
            st.warning("No simulations saved yet. Run some simulations first!")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Simulation selection with enhanced filtering
            with st.expander("🎯 Select Simulations for Comparison", expanded=True):
                search_term = st.text_input("Search simulations", 
                                           placeholder="Search by ID, parameters, tags...")
                
                # Filter by tags
                all_tags = list(set(tag for sim in simulations 
                                  for tag in sim.get('metadata', {}).get('tags', [])))
                selected_tags = st.multiselect("Filter by tags", all_tags)
                
                # Sort options
                sort_by = st.selectbox("Sort by", 
                                      ['created_at', 'last_accessed', 'twin_spacing', 
                                       'applied_stress', 'name'], index=0)
                
                # Get filtered list
                filtered_sims = SimulationDBCu.get_simulation_list(
                    sort_by=sort_by, 
                    filter_tags=selected_tags if selected_tags else None,
                    search_term=search_term
                )
                
                # Multi-select widget
                sim_options = {sim['display_name']: sim['id'] for sim in filtered_sims}
                selected_names = st.multiselect(
                    "Select simulations to compare",
                    options=list(sim_options.keys()),
                    default=list(sim_options.keys())[:min(4, len(sim_options))],
                    help="Select 2-6 simulations for best comparison results"
                )
                
                selected_ids = [sim_options[name] for name in selected_names]
                selected_simulations = [sim for sim in filtered_sims if sim['id'] in selected_ids]
            
            if len(selected_simulations) < 1:
                st.info("Select at least one simulation to compare")
                return
            
            # Comparison configuration
            with st.expander("⚙️ Comparison Settings", expanded=True):
                comparison_type = st.selectbox(
                    "Comparison Type",
                    ["Side-by-Side Heatmaps", "Evolution Timeline", "Statistical Analysis", 
                     "Parameter Sensitivity", "Line Profile Analysis", "3D Visualization"],
                    index=0
                )
                
                visualization_type = st.selectbox(
                    "Visualization Type",
                    ["stress_magnitude", "twin_structure", "twin_spacing", 
                     "yield_stress", "plastic_strain", "grain_structure"],
                    index=0
                )
                
                frame_selection = st.radio(
                    "Frame Selection",
                    ["Final Frame", "Mid Evolution", "Specific Frame"],
                    horizontal=True
                )
                
                if frame_selection == "Specific Frame":
                    frame_idx = st.slider("Frame index", 0, 100, 0)
                else:
                    frame_idx = 0
                
                # Additional settings based on comparison type
                if comparison_type == "Line Profile Analysis":
                    profile_type = st.selectbox("Profile Type", 
                                              ["horizontal", "vertical", "diagonal", "custom"])
                    position_ratio = st.slider("Profile Position", 0.0, 1.0, 0.5, 0.1)
                    if profile_type == "custom":
                        custom_angle = st.slider("Custom Angle (degrees)", -180.0, 180.0, 45.0, 5.0)
            
            # Run comparison
            if st.button("🔬 Run Comparison Analysis", type="primary", use_container_width=True):
                config = {
                    'comparison_type': comparison_type,
                    'visualization_type': visualization_type,
                    'frame_selection': frame_selection,
                    'frame_idx': frame_idx,
                    'selected_ids': selected_ids,
                }
                
                # Get style parameters
                style_params = PublicationStylerCu.get_styling_controls()
                
                # Create comparison based on type
                with st.spinner("Creating comparison plots..."):
                    if comparison_type == "Side-by-Side Heatmaps":
                        fig = SimulationComparatorCu.create_multi_simulation_dashboard(
                            selected_simulations, config, style_params
                        )
                        st.pyplot(fig)
                        
                    elif comparison_type == "Evolution Timeline":
                        fig = SimulationComparatorCu.create_evolution_comparison_plot(
                            selected_simulations, config, style_params
                        )
                        st.pyplot(fig)
                        
                    elif comparison_type == "Statistical Analysis":
                        fig = SimulationComparatorCu.create_statistical_comparison(
                            selected_simulations, config, style_params
                        )
                        st.pyplot(fig)
                        
                    elif comparison_type == "Parameter Sensitivity":
                        sweep_var = st.selectbox("Sweep Variable", 
                                               ["twin_spacing", "applied_stress", "W", "L_CTB"])
                        fig = SimulationComparatorCu.create_parameter_sensitivity_analysis(
                            selected_simulations, sweep_var, style_params
                        )
                        st.pyplot(fig)
                    
                    # Export options
                    st.subheader("📤 Export Comparison")
                    col_exp1, col_exp2, col_exp3 = st.columns(3)
                    
                    with col_exp1:
                        if st.button("📊 Export as PDF"):
                            pdf_buffer = BytesIO()
                            fig.savefig(pdf_buffer, format='pdf', dpi=style_params.get('dpi', 600), 
                                       bbox_inches='tight')
                            st.download_button(
                                label="Download PDF",
                                data=pdf_buffer.getvalue(),
                                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                    
                    with col_exp2:
                        if st.button("🖼️ Export as PNG"):
                            png_buffer = BytesIO()
                            fig.savefig(png_buffer, format='png', dpi=style_params.get('dpi', 600), 
                                       bbox_inches='tight')
                            st.download_button(
                                label="Download PNG",
                                data=png_buffer.getvalue(),
                                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                    
                    with col_exp3:
                        if st.button("📦 Export Complete Package"):
                            zip_buffer, package_name = EnhancedExporterCu.export_publication_package(
                                selected_simulations, [fig], config, style_params
                            )
                            st.download_button(
                                label="Download ZIP",
                                data=zip_buffer.getvalue(),
                                file_name=f"{package_name}.zip",
                                mime="application/zip"
                            )
        
        with col2:
            # Quick stats for selected simulations
            st.subheader("📈 Selected Simulations")
            
            for sim in selected_simulations:
                with st.expander(f"📋 {sim['display_name'][:30]}...", expanded=False):
                    st.caption(f"ID: `{sim['id']}`")
                    
                    # Display key parameters
                    params = sim['params']
                    st.write(f"**λ:** {params.get('twin_spacing', 0):.1f} nm")
                    st.write(f"**σ:** {params.get('applied_stress', 0)/1e6:.0f} MPa")
                    st.write(f"**W:** {params.get('W', 0):.1f}")
                    st.write(f"**Type:** {params.get('geometry_type', 'Unknown')}")
                    
                    # Display statistics if available
                    stats = sim.get('statistics', {})
                    if stats:
                        st.write("**Max Stress:** {:.0f} MPa".format(
                            stats.get('stress_statistics', {}).get('max_stress_MPa', 0)))
                        st.write("**Avg Spacing:** {:.1f} nm".format(
                            stats.get('twin_statistics', {}).get('avg_spacing_nm', 0)))
                    
                    # Tags
                    tags = sim.get('metadata', {}).get('tags', [])
                    if tags:
                        st.write("**Tags:**", ", ".join(tags))
                    
                    # Actions
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("⭐", key=f"fav_{sim['id']}", help="Toggle favorite"):
                            SimulationDBCu.toggle_favorite(sim['id'])
                            st.rerun()
                    
                    with col_act2:
                        if st.button("🗑️", key=f"del_{sim['id']}", help="Delete simulation"):
                            SimulationDBCu.delete_simulation(sim['id'])
                            st.rerun()
    
    # =============================================
    # MODE 3: DATABASE MANAGEMENT
    # =============================================
    elif operation_mode == "Database Management":
        st.header("🗃️ Database Management")
        
        # Database statistics
        db_stats = SimulationDBCu.get_database_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Simulations", db_stats.get('total_simulations', 0))
        with col2:
            st.metric("Database Size", f"{db_stats.get('total_size_mb', 0):.1f} MB")
        with col3:
            st.metric("Unique Tags", len(db_stats.get('tags_used', [])))
        with col4:
            st.metric("Favorites", db_stats.get('favorites_count', 0))
        
        # Search and filter
        with st.expander("🔍 Search & Filter", expanded=True):
            search_query = st.text_input("Search simulations", 
                                        placeholder="Search by ID, name, parameters, tags...")
            
            # Filter options
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                filter_favorites = st.checkbox("Show favorites only")
            with col_f2:
                filter_tags = st.multiselect("Filter by tags", db_stats.get('tags_used', []))
            with col_f3:
                sort_by = st.selectbox("Sort by", 
                                      ['created_at', 'last_accessed', 'name', 'twin_spacing', 'applied_stress'])
        
        # Get filtered simulations
        simulations = SimulationDBCu.get_simulation_list(
            sort_by=sort_by,
            filter_tags=filter_tags if filter_tags else None,
            search_term=search_query
        )
        
        # Filter favorites if needed
        if filter_favorites:
            simulations = [sim for sim in simulations 
                         if sim.get('metadata', {}).get('favorite', False)]
        
        # Display simulations in a table
        if simulations:
            st.subheader(f"📋 Simulations ({len(simulations)} found)")
            
            # Create dataframe for display
            display_data = []
            for sim in simulations:
                params = sim['params']
                metadata = sim['metadata']
                stats = sim.get('statistics', {})
                
                display_data.append({
                    'ID': sim['id'][:8],
                    'Name': metadata.get('name', 'Unnamed'),
                    'Type': params.get('geometry_type', 'Unknown'),
                    'λ (nm)': params.get('twin_spacing', 0),
                    'σ (MPa)': params.get('applied_stress', 0)/1e6,
                    'Max σ (MPa)': stats.get('stress_statistics', {}).get('max_stress_MPa', 0),
                    'Tags': ', '.join(metadata.get('tags', [])),
                    'Created': metadata.get('created_at', '')[:10],
                    '⭐': '★' if metadata.get('favorite', False) else '',
                    'Size (MB)': f"{metadata.get('file_size_estimate', 0):.1f}",
                })
            
            df = pd.DataFrame(display_data)
            
            # Interactive dataframe
            edited_df = st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "⭐": st.column_config.TextColumn(
                        "Favorite",
                        help="Starred simulations",
                        width="small"
                    )
                }
            )
            
            # Batch operations
            st.subheader("🛠️ Batch Operations")
            col_b1, col_b2, col_b3 = st.columns(3)
            
            with col_b1:
                if st.button("🔄 Update All Metadata", type="secondary"):
                    st.info("Metadata update functionality would be implemented here")
            
            with col_b2:
                if st.button("📥 Export All", type="secondary"):
                    zip_buffer = EnhancedExporterCu.export_simulation_data(simulations, format='zip')
                    if zip_buffer:
                        st.download_button(
                            label="Download All Simulations",
                            data=zip_buffer.getvalue(),
                            file_name=f"nanotwin_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
            
            with col_b3:
                if st.button("🗑️ Clear Database", type="secondary"):
                    if st.checkbox("Confirm: This will delete ALL simulations"):
                        st.session_state.simulations = {}
                        st.session_state.simulation_counter = 0
                        st.session_state.simulation_tags = {}
                        st.success("Database cleared!")
                        st.rerun()
        
        else:
            st.info("No simulations found matching the criteria.")
    
    # =============================================
    # MODE 4: PARAMETER SWEEP STUDY
    # =============================================
    elif operation_mode == "Parameter Sweep Study":
        st.header("📊 Parameter Sweep Study")
        
        st.markdown("""
        <div class="info-box">
        <strong>Parameter Sweep Analysis:</strong><br>
        Run multiple simulations with varying parameters to study:
        • Twin spacing effects on strengthening
        • Stress-strain response sensitivity
        • Mobility parameter optimization
        • Defect interaction studies
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sweep configuration
            with st.expander("⚙️ Sweep Configuration", expanded=True):
                sweep_variable = st.selectbox(
                    "Variable to Sweep",
                    ["Twin Spacing (λ)", "Applied Stress", "Twin Well Depth (W)", 
                     "CTB Mobility (L_CTB)", "Dislocation Pinning (ζ)", "Grain Boundary Position"],
                    index=0
                )
                
                # Range settings based on selected variable
                if sweep_variable == "Twin Spacing (λ)":
                    min_val = st.slider("Min twin spacing (nm)", 5.0, 50.0, 10.0, 1.0)
                    max_val = st.slider("Max twin spacing (nm)", min_val+5.0, 100.0, 30.0, 1.0)
                    n_points = st.slider("Number of points", 3, 20, 8, 1)
                elif sweep_variable == "Applied Stress":
                    min_val = st.slider("Min stress (MPa)", 50.0, 500.0, 100.0, 10.0)
                    max_val = st.slider("Max stress (MPa)", min_val+50.0, 1000.0, 500.0, 10.0)
                    n_points = st.slider("Number of points", 3, 20, 8, 1)
                elif sweep_variable == "Twin Well Depth (W)":
                    min_val = st.slider("Min W (J/m³)", 0.1, 5.0, 0.5, 0.1)
                    max_val = st.slider("Max W (J/m³)", min_val+0.5, 10.0, 3.0, 0.1)
                    n_points = st.slider("Number of points", 3, 20, 8, 1)
                
                # Base parameters
                st.subheader("Base Parameters")
                base_N = st.slider("Grid size N", 128, 256, 192, 64)
                base_dx = st.slider("Grid spacing (nm)", 0.2, 1.0, 0.5, 0.1)
                base_n_steps = st.slider("Steps per simulation", 50, 300, 100, 10)
                base_save_freq = st.slider("Save frequency", 5, 50, 10, 5)
            
            # Run sweep
            if st.button("🧪 Run Parameter Sweep Study", type="primary", use_container_width=True):
                # Generate parameter values
                param_values = np.linspace(min_val, max_val, n_points)
                
                with st.spinner(f"Running {n_points} simulations in sweep..."):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, param_val in enumerate(param_values):
                        # Update progress
                        progress_bar.progress((i + 1) / n_points)
                        status_text = st.empty()
                        status_text.text(f"Running simulation {i+1}/{n_points}: {sweep_variable} = {param_val:.2f}")
                        
                        # Create parameters for this simulation
                        params = {
                            'N': base_N,
                            'dx': base_dx,
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
                            'save_frequency': base_save_freq,
                            'n_steps': base_n_steps,
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
                        elif sweep_variable == "Twin Well Depth (W)":
                            params['twin_spacing'] = 20.0
                            params['grain_boundary_pos'] = 0.0
                            params['applied_stress'] = 300e6
                            params['W'] = param_val
                        
                        # Create metadata
                        metadata = {
                            'name': f"Sweep_{sweep_variable}_{param_val:.2f}",
                            'notes': f"Parameter sweep: {sweep_variable} = {param_val:.2f}",
                            'tags': ['sweep_study', sweep_variable.lower().replace(' ', '_')],
                            'run_time': 2.5,  # Mock run time
                            'frames': base_n_steps // base_save_freq,
                            'grid_size': base_N,
                            'dx': base_dx,
                        }
                        
                        # Create mock results (in reality, run your solver here)
                        mock_history = {
                            'phi_norm': np.random.rand(base_n_steps) * 0.5 + 0.5,
                            'max_stress': np.random.rand(base_n_steps) * param_val * 1e7,
                            'twin_spacing_avg': np.random.rand(base_n_steps) * 5 + (param_val if sweep_variable == "Twin Spacing (λ)" else 20.0),
                            'plastic_work': np.random.rand(base_n_steps) * 1e-12 * param_val,
                        }
                        
                        mock_results_history = []
                        for j in range(0, base_n_steps, base_save_freq):
                            mock_results = {
                                'phi': np.random.randn(base_N, base_N) * 0.3,
                                'sigma_eq': np.random.rand(base_N, base_N) * param_val * 1e7,
                                'h': np.random.rand(base_N, base_N) * 10 + (param_val if sweep_variable == "Twin Spacing (λ)" else 20.0),
                                'sigma_y': np.random.rand(base_N, base_N) * 2e8 + 5e7,
                                'eps_p_mag': np.random.rand(base_N, base_N) * 0.01,
                            }
                            mock_results_history.append(mock_results)
                        
                        # Save to database
                        sim_id = SimulationDBCu.save_simulation(params, mock_history, metadata, mock_results_history)
                        
                        # Store results for analysis
                        results.append({
                            'parameter': param_val,
                            'simulation_id': sim_id,
                            'final_max_stress': mock_history['max_stress'][-1],
                            'final_phi_norm': mock_history['phi_norm'][-1],
                            'final_avg_spacing': mock_history['twin_spacing_avg'][-1],
                            'final_plastic_work': mock_history['plastic_work'][-1],
                        })
                    
                    # Create sweep analysis plot
                    st.subheader("📈 Sweep Analysis Results")
                    
                    # Get style parameters
                    style_params = PublicationStylerCu.get_styling_controls()
                    
                    # Create analysis figure
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
                    
                    # Extract data
                    param_vals = [r['parameter'] for r in results]
                    max_stresses = [r['final_max_stress']/1e6 for r in results]  # MPa
                    phi_norms = [r['final_phi_norm'] for r in results]
                    avg_spacings = [r['final_avg_spacing'] for r in results]
                    plastic_works = [r['final_plastic_work'] for r in results]
                    
                    # Plot 1: Parameter vs Max Stress
                    axes[0, 0].plot(param_vals, max_stresses, 'bo-', linewidth=2, markersize=8, label='Max Stress')
                    axes[0, 0].fill_between(param_vals, 
                                           np.array(max_stresses) * 0.9, 
                                           np.array(max_stresses) * 1.1, 
                                           alpha=0.2, color='blue')
                    axes[0, 0].set_xlabel(sweep_variable, fontsize=10)
                    axes[0, 0].set_ylabel('Max Stress (MPa)', fontsize=10)
                    axes[0, 0].set_title(f'{sweep_variable} vs Maximum Stress', fontsize=11)
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 0].legend(fontsize=9)
                    
                    # Plot 2: Parameter vs Plastic Work
                    axes[0, 1].plot(param_vals, plastic_works, 'ro-', linewidth=2, markersize=8, label='Plastic Work')
                    axes[0, 1].set_xlabel(sweep_variable, fontsize=10)
                    axes[0, 1].set_ylabel('Plastic Work (J)', fontsize=10)
                    axes[0, 1].set_title(f'{sweep_variable} vs Plastic Work', fontsize=11)
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[0, 1].legend(fontsize=9)
                    
                    # Plot 3: Parameter vs Average Twin Spacing
                    axes[1, 0].plot(param_vals, avg_spacings, 'go-', linewidth=2, markersize=8, label='Avg Spacing')
                    axes[1, 0].set_xlabel(sweep_variable, fontsize=10)
                    axes[1, 0].set_ylabel('Average Twin Spacing (nm)', fontsize=10)
                    axes[1, 0].set_title(f'{sweep_variable} vs Twin Spacing', fontsize=11)
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].legend(fontsize=9)
                    
                    # Plot 4: Parameter vs Final Phi Norm
                    axes[1, 1].plot(param_vals, phi_norms, 'mo-', linewidth=2, markersize=8, label='Final ||φ||')
                    axes[1, 1].set_xlabel(sweep_variable, fontsize=10)
                    axes[1, 1].set_ylabel('Final Twin Order Parameter Norm', fontsize=10)
                    axes[1, 1].set_title(f'{sweep_variable} vs Twin Structure', fontsize=11)
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].legend(fontsize=9)
                    
                    # Apply publication styling
                    fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
                    
                    st.pyplot(fig)
                    
                    # Display results table
                    st.subheader("📋 Sweep Results Table")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    st.success(f"✅ Parameter sweep completed! {n_points} simulations analyzed and saved to database.")
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("📈 Analysis Tips")
            
            st.write("""
            **For effective parameter sweeps:**
            
            1. **Start with 5-8 points** for initial exploration
            2. **Focus on physically realistic ranges**
            3. **Save all simulations** for later comparison
            4. **Use appropriate grid resolution** for accuracy
            5. **Consider computational time** - larger sweeps take longer
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Quick analysis of existing sweep data
            existing_sweeps = [sim for sim in SimulationDBCu.get_simulation_list() 
                             if 'sweep_study' in sim.get('metadata', {}).get('tags', [])]
            
            if existing_sweeps:
                st.subheader("📊 Existing Sweep Studies")
                
                sweep_vars = set()
                for sim in existing_sweeps:
                    tags = sim.get('metadata', {}).get('tags', [])
                    for tag in tags:
                        if tag.endswith('_sweep') or tag in ['twin_spacing', 'applied_stress', 'w', 'l_ctb']:
                            sweep_vars.add(tag)
                
                for var in list(sweep_vars)[:5]:
                    st.write(f"- {var.replace('_', ' ').title()}")
    
    # =============================================
    # MODE 5: ADVANCED ANALYSIS
    # =============================================
    else:  # Advanced Analysis
        st.header("🔬 Advanced Analysis")
        
        st.markdown("""
        <div class="info-box">
        <strong>Advanced Analysis Tools:</strong><br>
        • Line profile extraction and comparison<br>
        • Statistical distribution analysis<br>
        • Fourier transform analysis<br>
        • Machine learning integration<br>
        • Custom post-processing scripts<br>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Line Profile Analysis", "Statistical Distribution", "Fourier Analysis", 
             "Correlation Analysis", "Machine Learning", "Custom Script"],
            index=0
        )
        
        if analysis_type == "Line Profile Analysis":
            st.subheader("📈 Line Profile Analysis")
            
            # Select simulation for analysis
            simulations = SimulationDBCu.get_simulation_list()
            if not simulations:
                st.warning("No simulations available for analysis")
                return
            
            sim_options = {sim['display_name']: sim['id'] for sim in simulations}
            selected_name = st.selectbox("Select simulation", list(sim_options.keys()))
            selected_id = sim_options[selected_name]
            
            sim_data = SimulationDBCu.get_simulation(selected_id)
            if not sim_data or not sim_data['results_history']:
                st.error("Selected simulation has no data")
                return
            
            # Select field for analysis
            field_options = {
                'Stress Magnitude': 'sigma_eq',
                'Twin Order Parameter': 'phi',
                'Twin Spacing': 'h',
                'Yield Stress': 'sigma_y',
                'Plastic Strain': 'eps_p_mag',
                'Grain 1': 'eta1',
                'Grain 2': 'eta2'
            }
            
            selected_field = st.selectbox("Select field for analysis", list(field_options.keys()))
            field_key = field_options[selected_field]
            
            # Get data
            final_results = sim_data['results_history'][-1]
            data = final_results[field_key]
            
            # Line profile configuration
            col_ana1, col_ana2 = st.columns(2)
            
            with col_ana1:
                profile_type = st.selectbox("Profile Type", 
                                          ["horizontal", "vertical", "diagonal", "anti_diagonal", "custom"])
                position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.05,
                                         help="Position relative to center (0 = center, 1 = edge)")
            
            with col_ana2:
                if profile_type == "custom":
                    custom_angle = st.slider("Custom Angle (degrees)", -180.0, 180.0, 45.0, 5.0)
                else:
                    custom_angle = 0
                
                dx = sim_data['params'].get('dx', 0.5)
            
            # Extract profiles
            if st.button("📊 Extract Line Profiles", type="primary"):
                # Extract multiple profiles
                profile_types = ['horizontal', 'vertical', 'diagonal']
                profiles = EnhancedLineProfilerCu.extract_multiple_profiles(
                    data, profile_types, position_ratio, 45, dx
                )
                
                # Create profile comparison plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
                
                # Plot 1: Data with profile lines
                ax1 = axes[0, 0]
                im = ax1.imshow(data, cmap='viridis', aspect='auto')
                ax1.set_title(f'{selected_field} Field', fontsize=11)
                ax1.set_xlabel('X (grid points)', fontsize=10)
                ax1.set_ylabel('Y (grid points)', fontsize=10)
                plt.colorbar(im, ax=ax1)
                
                # Add profile lines
                colors = ['red', 'blue', 'green']
                for (ptype, color) in zip(profile_types, colors):
                    if ptype in profiles:
                        endpoints = profiles[ptype]['endpoints']
                        ax1.plot([endpoints[0]/dx, endpoints[2]/dx], 
                                [endpoints[1]/dx, endpoints[3]/dx], 
                                color=color, linewidth=2, linestyle='--', alpha=0.8, 
                                label=f'{ptype.title()} Profile')
                
                ax1.legend(fontsize=9)
                
                # Plot 2-4: Individual profiles
                for idx, (ptype, color) in enumerate(zip(profile_types, colors)):
                    if ptype in profiles:
                        ax = axes[(idx+1)//2, (idx+1)%2]
                        profile_data = profiles[ptype]
                        ax.plot(profile_data['distance'], profile_data['profile'], 
                               color=color, linewidth=2)
                        ax.set_xlabel('Distance (nm)', fontsize=10)
                        ax.set_ylabel(selected_field, fontsize=10)
                        ax.set_title(f'{ptype.title()} Profile', fontsize=11)
                        ax.grid(True, alpha=0.3)
                
                # Apply publication styling
                style_params = PublicationStylerCu.get_styling_controls()
                fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
                
                st.pyplot(fig)
                
                # Export profiles
                if st.button("📥 Export Profile Data"):
                    zip_buffer = EnhancedExporterCu.export_line_profile_data(profiles, {
                        'simulation_id': selected_id,
                        'field': selected_field,
                        'profile_type': profile_type,
                        'position_ratio': position_ratio,
                        'angle': custom_angle,
                        'dx': dx
                    })
                    
                    st.download_button(
                        label="Download Profile Data",
                        data=zip_buffer.getvalue(),
                        file_name=f"line_profiles_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
        
        elif analysis_type == "Statistical Distribution":
            st.subheader("📊 Statistical Distribution Analysis")
            
            # Select simulations for analysis
            simulations = SimulationDBCu.get_simulation_list()
            if not simulations:
                st.warning("No simulations available for analysis")
                return
            
            sim_options = {sim['display_name']: sim['id'] for sim in simulations}
            selected_names = st.multiselect("Select simulations", list(sim_options.keys()))
            selected_ids = [sim_options[name] for name in selected_names]
            
            if not selected_ids:
                st.info("Select at least one simulation")
                return
            
            # Select field for statistical analysis
            field_options = {
                'Stress Magnitude': 'sigma_eq',
                'Twin Spacing': 'h',
                'Yield Stress': 'sigma_y',
                'Plastic Strain': 'eps_p_mag'
            }
            
            selected_field = st.selectbox("Select field for statistical analysis", 
                                         list(field_options.keys()))
            field_key = field_options[selected_field]
            
            if st.button("📈 Analyze Statistical Distributions", type="primary"):
                # Collect data from selected simulations
                all_data = []
                labels = []
                
                for sim_id in selected_ids:
                    sim_data = SimulationDBCu.get_simulation(sim_id)
                    if not sim_data or not sim_data['results_history']:
                        continue
                    
                    final_results = sim_data['results_history'][-1]
                    data = final_results[field_key].flatten()
                    
                    # Remove outliers and NaN values
                    data = data[~np.isnan(data)]
                    if len(data) > 1000:  # Downsample for performance
                        data = np.random.choice(data, 1000, replace=False)
                    
                    all_data.append(data)
                    labels.append(f"Sim {sim_id[:6]}...")
                
                if not all_data:
                    st.error("No valid data found in selected simulations")
                    return
                
                # Create statistical distribution plots
                fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
                colors = plt.cm.rainbow(np.linspace(0, 1, len(all_data)))
                
                # Plot 1: Histograms
                ax1 = axes[0, 0]
                for idx, (data, color, label) in enumerate(zip(all_data, colors, labels)):
                    ax1.hist(data, bins=30, density=True, alpha=0.5, color=color, label=label)
                ax1.set_xlabel(selected_field, fontsize=10)
                ax1.set_ylabel('Probability Density', fontsize=10)
                ax1.set_title('Histogram Comparison', fontsize=11)
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Box plots
                ax2 = axes[0, 1]
                bp = ax2.boxplot(all_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax2.set_ylabel(selected_field, fontsize=10)
                ax2.set_title('Box Plot Comparison', fontsize=11)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Plot 3: Violin plots
                ax3 = axes[1, 0]
                parts = ax3.violinplot(all_data, showmeans=True, showmedians=True)
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                ax3.set_xticks(range(1, len(labels) + 1))
                ax3.set_xticklabels(labels, rotation=45)
                ax3.set_ylabel(selected_field, fontsize=10)
                ax3.set_title('Violin Plot Comparison', fontsize=11)
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Plot 4: Cumulative distribution functions
                ax4 = axes[1, 1]
                for idx, (data, color, label) in enumerate(zip(all_data, colors, labels)):
                    sorted_data = np.sort(data)
                    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    ax4.plot(sorted_data, y_vals, color=color, linewidth=2, label=label)
                ax4.set_xlabel(selected_field, fontsize=10)
                ax4.set_ylabel('Cumulative Probability', fontsize=10)
                ax4.set_title('Cumulative Distribution Functions', fontsize=11)
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
                
                # Apply publication styling
                style_params = PublicationStylerCu.get_styling_controls()
                fig = PublicationStylerCu.apply_journal_style(fig, axes, style_params.get('journal_style', 'nature'))[0]
                
                st.pyplot(fig)
        
        # Other analysis types would be implemented similarly...
        
        # Footer
        st.markdown("---")
        st.caption(f"🔬 Nanotwinned Copper Phase-Field Simulator Pro • Version 2.0 • {datetime.now().strftime('%Y-%m-%d')}")

# Run the application
if __name__ == "__main__":
    main()
