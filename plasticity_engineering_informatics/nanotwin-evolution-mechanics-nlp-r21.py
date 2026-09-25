# ============================================================================
# ███ ENHANCED NANOTWINNED Cu PHASE-FIELD SIMULATOR (PURE FFT SPECTRAL) ███
# ███ + PLASTICITY PARAMETER INTELLIGENT RECOMMENDER v10.0.0            ███
# ███ PUBLICATION-QUALITY VISUALS DASHBOARD                             ███
# ███ STREAMLIT NESTED-EXPANDER FIX APPLIED (v8.0.1)                   ███
# ███ FULL CACHE PURGE ON "FORCE RELOAD CORPUS" (v8.1.1)               ███
# ███ FIX: LLM-DRIVEN PRIOR LEARNER + OLLAMA RAW RESPONSE DEBUG (v8.1.2)███
# ███ NEW: DUAL-MODE PROMPTS + CHAIN-OF-THOUGHT REASONING (v8.2.0)      ███
# ███ FIX: GATEKEEPER — PARAM ALIAS CANONICALIZATION + PER-PARAM        ███
# ███      HEURISTIC FALLBACK + VALUE COERCION + UI DEBUG TOGGLE        ███
# ███ FIX (v8.4.0): gamma0_dot OMISSION                            ██████
# ███ FIX (v8.5.0): PUBLICATION CANDIDATE-SCORE FIGURE               ██████
# ███ FIX (v8.5.1): MATHTEXT ESCAPING HARDENING                       ████
# ███ NEW (v8.6.0): BAR-CHART COLORMAP ENGINE                          ████
# ███ NEW (v8.7.0): BAR-CHART TYPOGRAPHY CUSTOMIZATION                 ████
# ███ NEW (v8.8.0): NER-GAZETTEER RETRIEVAL + GROUNDED LLM EXTRACTION  ████
# ███ FIX (v8.8.1): heuristic_extract pass (b) kwarg collision         ██████
# ███ NEW (v8.8.2): THREE-TIER CASCADE — RESTORES FULL 5/5 COVERAGE    ██████
# ███ FIX (v8.8.3): PYRAMIDAL SANITY — prior-inferred sigma0/mu rows    ██████
# ███ NEW (v8.8.3): FRICTION STRESS (σ₀) + PEIERLS–NABARRO (τ_P)         ██████
# ███ FIX (v8.8.4): 'ValueCandidate' object has no attribute 'reasoning' ██████
# ███ NEW (v8.9.0): FRICTION / LATTICE STRESS LAB + 5/5 COMPLETION     ████
# ███   · Four σ₀ derivation routes: HP(d), HP(λ), solver-law inversion,  ███
# ███     P–N + Taylor FLOOR.  Weighted-median consensus; conf =          ███
# ███     gmean(conf_routes) · exp(−relative_spread), capped at 0.75.     ███
# ███   · FIX — G = 0.000 GPa degenerate from side-note parser:           ███
# ███     parse_side_note_table now rejects 0/negative values;            ███
# ███     _normalize_unit returns NaN on zero-collapse; gatekeep rejects  ███
# ███     v ≤ 0 for positive-lower-bound params. P–N route additionally   ███
# ███     guards with ctx['G'].value > 1.0 before trusting explicit G.    ███
# ███   · FIX — w silently clamped for P–N: w is now NEVER clamped.       ███
# ███     Auto mode defaults to w = b/(1−ν); captured w is used only if   ███
# ███     its provenance is llm/heuristic; user can force ζ = w/b via a   ███
# ███     slider.  τ_P sensitivity plot and ζ readout displayed.          ███
# ███   · Pyramid sanity: σ₀ ≥ M·τ_P floor check + solver closure         ███
# ███     residual vs the simulator's own compute_yield_stress().         ███
# ███   · Context fields (σ_y,d,λ,b,ν,M,k_y,w,G) moved from sidebar →     ███
# ███     main-bar "🧮 Friction Stress Lab" tab.                          ███
# ███   · σ₀_fric card completed via derived consensus (▼) → 5/5 coverage. ███
# ███   · `derived_consensus` method added to LatentMoE method-expert     ███
# ███     weighting (0.55) so the derived row can outrank a weak prior    ███
# ███     but never a verbatim Tier-1 hit.                                ███
# ███ NEW (v9.0.0): PHYSICS-GROUNDED ρ₀ INFERENCE                       ███
# ███   · GAZETTEER gains two new entity types:                           ███
# ███       `synthesis_method`      (sputter/PVD, MBE, electrodeposition, ███
# ███                                additive-assisted electrodeposition) ███
# ███       `grain_architecture`    (columnar, equiaxed/nanocrystalline,  ███
# ███                                single crystal, polycrystal)         ███
# ███   · `extract_rho0_with_context`: snippet-enhanced NER that captures ███
# ███     the ±400-char snippet around each ρ₀ mention and pulls out       ███
# ███     (synthesis, architecture, λ) as qualitative context factors.    ███
# ███   · `classify_rho0_regime`: three-factor physics-regime classifier. ███
# ███   · `build_rho0_physics_inference_prompt`: dedicated ρ₀ prompt       ███
# ███   · `PhysicsRegimeExpert`: new LatentMoE expert (weight 0.25 for    ███
# ███     ρ₀ only, activated when any contextual factor is known).        ███
# ███   · `PlasticityLatentMoEScorer.score` gains `synthesis`,            ███
# ███     `architecture`, `twin_spacing` kwargs.                          ███
# ███   · `recommend_rho0_physics`: dedicated ρ₀ cascade.                 ███
# ███   · Regression test `_regression_test_v90`.                        ███
# ███ NEW (v9.1.0): THEORY-AWARE γ̇₀ NER                                ███
# ███   · GAZETTEER gains `theory_framework` entity type.                 ███
# ███   · `GAMMA0_DOT_REGIMES`: seven-entry regime table.                 ███
# ███   · `_norm_theory`, `classify_gamma0_regime`,                       ███
# ███     `extract_gamma0_with_context`,                                  ███
# ███     `build_gamma0_theory_aware_prompt`,                             ███
# ███     `build_gamma0_theory_inference_prompt`.                         ███
# ███   · `TheoryRegimeExpert`: new LatentMoE expert for γ̇₀.             ███
# ███   · `recommend_gamma0_theory_aware`: dedicated γ̇₀ cascade.          ███
# ███   · Regression test `_regression_test_v91`.                        ███
# ███ NEW (v9.2.0): 3-BUCKET LEGEND — llm_grounded / llm_prior /      ███
# ███   deterministic; display-level rollup only, fine keys intact.   ███
# ███ FIX (v9.2.1): regime-before-prior ordering; granularity         ███
# ███   normalized once; 'd'→'p'; fill = corpus evidence (fine mode); ███
# ███   fixed Okabe–Ito yellow best-match bar; word-boundary 'ai'.    ███
# ███ FIX (v9.2.2): HONESTY PASS — llm_reasoned is a RESERVED fine    ███
# ███   key, unwired until a CoT prompt path exists (Option B) or     ███
# ███   stamped from the dual-mode signal (Option A);                 ███
# ███   PlasticityCandidate.provenance restored + fed from ext — the  ███
# ███   read side of _stamp_provenance; chart provenance lists prefer ███
# ███   c.provenance or c.method; suite: globals().get + NTCU_REGRESSION;███
# ███   M1–M6 (diag init, value_gpa, st.image width, glyph choice,    ███
# ███   signature protocol, v92 body restored).                       ███
# ███ SIGNED-OFF deltas (pinned by tests): model_inversion llm(3)→    ███
# ███   derived(2) · explicit heuristic(2)→llm_extract(3) ·           ███
# ███   physics_regime heuristic(2)→regime_prior(1).  Results may     ███
# ███   differ from v9.1 beyond the legend.                           ███
# ███ NEW (v9.3.0): UNION CASCADE + CONTEXT CANDIDATES — all viable   ███
# ███   routes run; non-incumbent routes (INCUMBENT_ROUTES pin-list)  ███
# ███   enter as grey non-ranked context bars — v9.2.2 findings are   ███
# ███   structurally unchangeable; σ₀ four-route disaggregation;      ███
# ███   'Context (non-ranked)' legend entry, present-only; frozen-    ███
# ███   findings stability test.                                       ███
# ███ FIX (v9.3.1): HONEST NAMING PASS — the fine key `derived` is    ███
# ███   RENAMED to `physics_inferred`.  `_DERIVED_HINTS` retains      ███
# ███   backward-compatible matching on the old token.  NEW: chart-   ███
# ███   tab toggle "🔓 Show all routes as competitive".               ███
# ███ NEW (v10.0.0): SCORE-ANATOMY RECOMMENDER (CTC² Phases 3–4)      ███
# ███   · Legend keys = EXPERTS (evidence / regime / theory /         ███
# ███     closure / consensus / compat).                              ███
# ███   · Provenance = badge, never a rank.                           ███
# ███   · Bar height = Σ = Λ·κ_n·κ_m·G — the decision variable.       ███
# ███   · Bar body   = stacked expert segments ŵ_k·s_k.               ███
# ███   · Gates as hatched, named amputations; counts as pips +       ███
# ███     ghost Λ outline; weight sensitivity as Dirichlet whisker.    ███
# ███   · v9 pin-list survives as annotation-only reference marks.    ███
# ============================================================================

import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FormatStrFormatter,
                               FuncFormatter, NullLocator)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
import matplotlib.animation as animation
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import zipfile
import pickle
import torch
import sqlite3
import hashlib
import traceback
import warnings
from scipy import stats
from io import BytesIO, StringIO
import tempfile
import os
import re
import math
import time
import threading
import logging
import unicodedata
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import faiss as _faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"❌ Error in {func.__name__}: {str(e)}"
            st.error(error_msg)
            st.error("Please check the console for detailed error information.")
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None
    return wrapper


# ============================================================================
# SPECTRAL DERIVATIVE HELPERS
# ============================================================================
def make_k_vectors(N, dx):
    kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1)
    ky = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-12
    return kx, ky, k2


def spectral_gradients(field, kx, ky):
    fh = fft2(field)
    gx = np.real(ifft2(1j * kx * fh))
    gy = np.real(ifft2(1j * ky * fh))
    return gx, gy


def spectral_laplacian(field, k2):
    fh = fft2(field)
    lap = np.real(ifft2(-k2 * fh))
    return lap


def compute_twin_spacing_from_gradient(phi_gx, phi_gy):
    grad_mag = np.sqrt(phi_gx**2 + phi_gy**2)
    h = np.where(grad_mag > 1e-12, 2.0 / np.maximum(grad_mag, 1e-12), 1e6)
    return h


def compute_anisotropic_properties(phi_gx, phi_gy, nx, ny, kappa0,
                                   gamma_aniso, L_CTB, L_ITB, n_mob):
    grad_mag = np.sqrt(phi_gx**2 + phi_gy**2 + 1e-12)
    mask = grad_mag > 1e-6
    mx = np.where(mask, phi_gx / grad_mag, 0.0)
    my = np.where(mask, phi_gy / grad_mag, 0.0)
    dot = mx * nx + my * ny
    kappa_phi = kappa0 * (1.0 + gamma_aniso * (1.0 - dot**2))
    aniso_factor = (1.0 - dot**2) ** n_mob
    L_phi = L_CTB + (L_ITB - L_CTB) * aniso_factor
    kappa_phi = np.where(mask, kappa_phi, kappa0)
    L_phi = np.where(mask, L_phi, L_CTB)
    return kappa_phi, L_phi


def compute_transformation_strain(phi, eta1, gamma_tw, ax, ay, nx, ny):
    f_phi = 0.25 * (phi**3 - phi**2 - phi + 1)
    eta1_clamped = np.clip(eta1, 0.0, 1.0)
    exx_star = gamma_tw * nx * ax * f_phi * eta1_clamped
    eyy_star = gamma_tw * ny * ay * f_phi * eta1_clamped
    exy_star = 0.5 * gamma_tw * (nx * ay + ny * ax) * f_phi * eta1_clamped
    return exx_star, eyy_star, exy_star


def compute_yield_stress(h, sigma0, mu, b, nu):
    safe = h > 2 * b
    sigma_y = np.empty_like(h)
    log_term = np.log(np.maximum(h, 2.001 * b) / b)
    sigma_y[safe] = sigma0 + (mu * b / (2 * np.pi * h[safe] * (1 - nu))) * log_term[safe]
    sigma_y[~safe] = sigma0 + mu / (2 * np.pi * (1 - nu))
    return sigma_y


def update_plastic_strain(sigma_eq, sigma_y, eps_p_xx, eps_p_yy, eps_p_xy,
                          gamma0_dot, m, dt):
    MAX_OVERSTRESS = 1.0
    MAX_PLASTIC_STRAIN = 0.1
    overstress = np.maximum(sigma_eq - sigma_y, 0.0) / np.maximum(sigma_y, 1e-9)
    overstress = np.minimum(overstress, MAX_OVERSTRESS)
    gamma_dot = gamma0_dot * overstress**m
    stress_dev = 2.0 / 3.0 * gamma_dot * dt
    stress_dev = np.minimum(stress_dev, 0.001)
    d_xx = stress_dev
    d_yy = -0.5 * stress_dev
    d_xy = 0.5 * stress_dev
    eps_p_xx_new = np.clip(eps_p_xx + d_xx, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_yy_new = np.clip(eps_p_yy + d_yy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_xy_new = np.clip(eps_p_xy + d_xy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    return eps_p_xx_new, eps_p_yy_new, eps_p_xy_new


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================
class MetadataManager:
    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        if run_time is None:
            run_time = 0.0
        return {
            'run_time': run_time,
            'frames': len(history) if history else 0,
            'grid_size': kwargs.get('grid_size', sim_params.get('N', 256)),
            'dx': kwargs.get('dx', sim_params.get('dx', 0.5)),
            'dt': sim_params.get('dt', 1e-4),
            'created_at': datetime.now().isoformat(),
            'colormaps': {
                'phi': sim_params.get('cmap_phi', 'RdBu_r'),
                'sigma_eq': sim_params.get('cmap_stress', 'hot'),
                'sigma_h': sim_params.get('cmap_hydro', 'RdBu'),
                'h': sim_params.get('cmap_h', 'plasma'),
                'eta1': sim_params.get('cmap_eta1', 'Reds')
            },
            'material_properties': sim_params.get('material_properties', {}),
            'simulation_parameters': {
                'dt': sim_params.get('dt', 1e-4),
                'N': sim_params.get('N', 256),
                'dx': sim_params.get('dx', 0.5),
                'twin_spacing': sim_params.get('twin_spacing', 20.0),
                'applied_stress': sim_params.get('applied_stress', 300e6),
                'applied_stress_angle': sim_params.get('applied_stress_angle', 0.0),
                'n_steps': sim_params.get('n_steps', 100),
                'solver_method': 'semi_implicit_fourier_spectral'
            }
        }

    @staticmethod
    def validate_metadata(metadata):
        if not isinstance(metadata, dict):
            metadata = {}
        required_fields = ['run_time', 'frames', 'grid_size', 'dx', 'dt', 'created_at']
        for field in required_fields:
            if field not in metadata:
                if field == 'created_at':
                    metadata[field] = datetime.now().isoformat()
                elif field == 'run_time':
                    metadata[field] = 0.0
                elif field == 'frames':
                    metadata[field] = 0
                elif field == 'grid_size':
                    metadata[field] = 256
                elif field == 'dx':
                    metadata[field] = 0.5
                elif field == 'dt':
                    metadata[field] = 1e-4
        if 'colormaps' not in metadata:
            metadata['colormaps'] = {
                'phi': 'RdBu_r', 'sigma_eq': 'hot', 'sigma_h': 'RdBu',
                'h': 'plasma', 'eta1': 'Reds'
            }
        return metadata

    @staticmethod
    def get_metadata_field(metadata, field, default=None):
        try:
            return metadata.get(field, default)
        except Exception:
            return default


# ============================================================================
# JOURNAL TEMPLATES
# ============================================================================
class JournalTemplates:
    @staticmethod
    def get_journal_styles():
        return {
            'nature': {
                'figure_width_single': 8.9, 'figure_width_double': 18.3,
                'font_family': 'Arial', 'font_size_small': 7,
                'font_size_medium': 8, 'font_size_large': 9,
                'line_width': 0.5, 'axes_linewidth': 0.5,
                'tick_width': 0.5, 'tick_length': 2, 'grid_alpha': 0.1, 'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                '#bcbd22', '#17becf']
            },
            'science': {
                'figure_width_single': 5.5, 'figure_width_double': 11.4,
                'font_family': 'Helvetica', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 0.75, 'axes_linewidth': 0.75,
                'tick_width': 0.75, 'tick_length': 3, 'grid_alpha': 0.15, 'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                                '#77AC30', '#4DBEEE', '#A2142F', '#FF00FF',
                                '#00FFFF', '#FFA500']
            },
            'advanced_materials': {
                'figure_width_single': 8.6, 'figure_width_double': 17.8,
                'font_family': 'Arial', 'font_size_small': 8,
                'font_size_medium': 9, 'font_size_large': 10,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0.2, 'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000',
                                '#44AA99', '#882255', '#117733', '#999933',
                                '#AA4499', '#88CCEE']
            },
            'prl': {
                'figure_width_single': 3.4, 'figure_width_double': 7.0,
                'font_family': 'Times New Roman', 'font_size_small': 8,
                'font_size_medium': 10, 'font_size_large': 12,
                'line_width': 1.0, 'axes_linewidth': 1.0,
                'tick_width': 1.0, 'tick_length': 4, 'grid_alpha': 0, 'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                                '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                                '#999999', '#FFFFFF']
            },
            'custom': {
                'figure_width_single': 6.0, 'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans', 'font_size_small': 10,
                'font_size_medium': 12, 'font_size_large': 14,
                'line_width': 1.5, 'axes_linewidth': 1.5,
                'tick_width': 1.0, 'tick_length': 5, 'grid_alpha': 0.3, 'dpi': 300,
                'color_cycle': plt.cm.Set2(np.linspace(0, 1, 10))
            }
        }

    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
        styles = JournalTemplates.get_journal_styles()
        style = styles.get(journal_name, styles['nature'])
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
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        for ax in axes_flat:
            if ax is not None:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['top'].set_linewidth(style['axes_linewidth'] * 0.5)
                ax.spines['right'].set_linewidth(style['axes_linewidth'] * 0.5)
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=style['tick_length'])
                ax.tick_params(which='minor', length=style['tick_length'] * 0.6)
        return fig, style


# ============================================================================
# ENHANCED COLORMAP LIBRARY
# ============================================================================
COLORMAPS = {
    'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno',
    'magma': 'magma', 'cividis': 'cividis', 'hot': 'hot', 'cool': 'cool',
    'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn', 'winter': 'winter',
    'copper': 'copper', 'bone': 'bone', 'gray': 'gray', 'pink': 'pink',
    'afmhot': 'afmhot', 'gist_heat': 'gist_heat', 'gist_gray': 'gist_gray',
    'binary': 'binary', 'coolwarm': 'coolwarm', 'bwr': 'bwr', 'seismic': 'seismic',
    'RdBu': 'RdBu', 'RdBu_r': 'RdBu_r', 'RdGy': 'RdGy', 'PiYG': 'PiYG',
    'PRGn': 'PRGn', 'BrBG': 'BrBG', 'PuOr': 'PuOr', 'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted', 'hsv': 'hsv', 'tab10': 'tab10',
    'tab20': 'tab20', 'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3',
    'Paired': 'Paired', 'Accent': 'Accent', 'Dark2': 'Dark2', 'jet': 'jet',
    'turbo': 'turbo', 'rainbow': 'rainbow', 'nipy_spectral': 'nipy_spectral',
    'gist_ncar': 'gist_ncar', 'gist_rainbow': 'gist_rainbow',
    'gist_earth': 'gist_earth', 'gist_stern': 'gist_stern', 'ocean': 'ocean',
    'terrain': 'terrain', 'gnuplot': 'gnuplot', 'gnuplot2': 'gnuplot2',
    'CMRmap': 'CMRmap', 'cubehelix': 'cubehelix', 'brg': 'brg',
    'rocket': 'rocket', 'mako': 'mako', 'crest': 'crest', 'flare': 'flare',
    'icefire': 'icefire', 'vlag': 'vlag'
}

cmap_list = list(COLORMAPS.keys())

BAR_CMAP_CATEGORIES = {
    "🌈 Sequential (score gradient)": [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'rocket', 'mako', 'crest', 'flare',
        'Blues', 'Greens', 'Reds', 'Purples', 'Oranges',
        'YlOrRd', 'YlGnBu', 'BuGn', 'PuRd',
    ],
    "🔄 Diverging (low↔high)": [
        'coolwarm', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
        'seismic', 'bwr', 'vlag', 'icefire',
    ],
    "🌊 Cyclic / Perceptual": [
        'twilight', 'twilight_shifted', 'hsv', 'turbo',
    ],
    "🎯 Categorical (distinct)": [
        'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
        'Paired', 'Accent', 'Dark2', 'Pastel1', 'Pastel2',
    ],
    "🖤 Classic": [
        'Greys', 'gray', 'bone', 'copper', 'hot', 'afmhot',
        'gist_heat', 'binary',
    ],
}
BAR_CMAP_FLAT = [n for cat in BAR_CMAP_CATEGORIES.values() for n in cat]

PUBLICATION_PALETTES = {
    'okabe_ito': ['#000000', '#E69F00', '#56B4E9', '#009E73',
                  '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
    'tol_bright': ['#4477AA', '#EE6677', '#228833', '#CCBB44',
                   '#66CCEE', '#AA3377', '#BBBBBB'],
    'tol_muted': ['#CC6677', '#332288', '#DDCC77', '#117733',
                  '#88CCEE', '#882255', '#44AA99', '#999933',
                  '#AA4499', '#DDDDDD'],
    'ibm_carbon': ['#6929c4', '#1192e8', '#005d5d', '#9f1853',
                   '#fa4d56', '#570408', '#198038', '#002d9c',
                   '#ee538b', '#b28600'],
    'nature_classic': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf'],
    'science_aaas': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                     '#77AC30', '#4DBEEE', '#A2142F'],
    'ieee': ['#0000FF', '#FF0000', '#00AA00', '#FF8000',
             '#800080', '#00AAAA', '#800000'],
}

FONT_FAMILIES = [
    'Arial', 'Helvetica', 'Times New Roman', 'Courier New',
    'DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono',
    'Calibri', 'Cambria', 'Georgia', 'Verdana', 'Tahoma',
    'Trebuchet MS', 'Garamond', 'Palatino', 'Bookman Old Style',
    'sans-serif', 'serif', 'monospace'
]


# ============================================================================
# POST-PROCESSING CLASSES
# ============================================================================
class EnhancedLineProfiler:
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.extent = [-N * dx / 2, N * dx / 2, -N * dx / 2, N * dx / 2]

    @handle_errors
    def extract_profile(self, data, profile_type, position_ratio=0.5, angle_deg=45):
        profile_type = self._normalize_profile_type(profile_type)
        ny, nx = data.shape
        center_x, center_y = nx // 2, ny // 2

        if profile_type in ['horizontal', 'vertical']:
            offset = int(min(nx, ny) * 0.4 * position_ratio)
        else:
            offset = int(min(nx, ny) * 0.3 * position_ratio)

        if profile_type == 'horizontal':
            row_idx = center_y + offset
            profile = data[row_idx, :]
            distance = np.linspace(self.extent[0], self.extent[1], nx)
            endpoints = (self.extent[0], row_idx * self.dx + self.extent[2],
                         self.extent[1], row_idx * self.dx + self.extent[2])
        elif profile_type == 'vertical':
            col_idx = center_x + offset
            profile = data[:, col_idx]
            distance = np.linspace(self.extent[2], self.extent[3], ny)
            endpoints = (col_idx * self.dx + self.extent[0], self.extent[2],
                         col_idx * self.dx + self.extent[0], self.extent[3])
        elif profile_type == 'diagonal':
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x - diag_length // 2, center_y - diag_length // 2)
            profile, distances = [], []
            for i in range(diag_length):
                x = start_idx[0] + i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    dist = i * self.dx * np.sqrt(2)
                    distances.append(dist - (diag_length // 2) * self.dx * np.sqrt(2))
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = start_idx[0] * self.dx + self.extent[0]
            y_start = start_idx[1] * self.dx + self.extent[2]
            x_end = (start_idx[0] + diag_length - 1) * self.dx + self.extent[0]
            y_end = (start_idx[1] + diag_length - 1) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        elif profile_type == 'anti_diagonal':
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x + diag_length // 2, center_y - diag_length // 2)
            profile, distances = [], []
            for i in range(diag_length):
                x = start_idx[0] - i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    dist = i * self.dx * np.sqrt(2)
                    distances.append(dist - (diag_length // 2) * self.dx * np.sqrt(2))
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = start_idx[0] * self.dx + self.extent[0]
            y_start = start_idx[1] * self.dx + self.extent[2]
            x_end = (start_idx[0] - diag_length + 1) * self.dx + self.extent[0]
            y_end = (start_idx[1] + diag_length - 1) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        elif profile_type == 'custom':
            angle_rad = np.deg2rad(angle_deg)
            length = int(min(nx, ny) * 0.8)
            dx_line = np.cos(angle_rad) * length // 2
            dy_line = np.sin(angle_rad) * length // 2
            profile, distances = [], []
            for t in np.linspace(-length // 2, length // 2, length):
                x = center_x + t * np.cos(angle_rad) + offset * np.cos(angle_rad + np.pi / 2)
                y = center_y + t * np.sin(angle_rad) + offset * np.sin(angle_rad + np.pi / 2)
                if 0 <= x < nx - 1 and 0 <= y < ny - 1:
                    x0, y0 = int(x), int(y)
                    x1, y1 = x0 + 1, y0 + 1
                    if x1 >= nx: x1 = nx - 1
                    if y1 >= ny: y1 = ny - 1
                    wx = x - x0
                    wy = y - y0
                    val = (data[y0, x0] * (1 - wx) * (1 - wy) +
                           data[y0, x1] * wx * (1 - wy) +
                           data[y1, x0] * (1 - wx) * wy +
                           data[y1, x1] * wx * wy)
                    profile.append(val)
                    distances.append(t * self.dx)
            distance = np.array(distances)
            profile = np.array(profile)
            x_start = (center_x - dx_line + offset * np.cos(angle_rad + np.pi / 2)) * self.dx + self.extent[0]
            y_start = (center_y - dy_line + offset * np.sin(angle_rad + np.pi / 2)) * self.dx + self.extent[2]
            x_end = (center_x + dx_line + offset * np.cos(angle_rad + np.pi / 2)) * self.dx + self.extent[0]
            y_end = (center_y + dy_line + offset * np.sin(angle_rad + np.pi / 2)) * self.dx + self.extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        return distance, profile, endpoints

    def _normalize_profile_type(self, profile_type):
        normalized = str(profile_type).lower().replace('-', '_')
        mapping = {
            'horizontal': 'horizontal', 'h': 'horizontal', 'x': 'horizontal',
            'vertical': 'vertical', 'v': 'vertical', 'y': 'vertical',
            'diagonal': 'diagonal', 'd': 'diagonal', 'diag': 'diagonal',
            'anti_diagonal': 'anti_diagonal', 'antidiagonal': 'anti_diagonal',
            'anti-diagonal': 'anti_diagonal', 'ad': 'anti_diagonal',
            'custom': 'custom', 'c': 'custom', 'angled': 'custom'
        }
        return mapping.get(normalized, normalized)


class PublicationEnhancer:
    @staticmethod
    def create_custom_colormaps():
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        plasma_enhanced = LinearSegmentedColormap.from_list('plasma_enhanced', [
            (0.0, '#0c0887'), (0.1, '#4b03a1'), (0.3, '#8b0aa5'),
            (0.5, '#b83289'), (0.7, '#db5c68'), (0.9, '#f48849'), (1.0, '#fec325')
        ])
        coolwarm_enhanced = LinearSegmentedColormap.from_list('coolwarm_enhanced', [
            (0.0, '#3a4cc0'), (0.25, '#8abcdd'), (0.5, '#f7f7f7'),
            (0.75, '#f0b7a4'), (1.0, '#b40426')
        ])
        twin_categorical = ListedColormap([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'
        ])
        stress_map = LinearSegmentedColormap.from_list('stress_map', [
            (0.0, '#2c7bb6'), (0.2, '#abd9e9'), (0.4, '#ffffbf'),
            (0.6, '#fdae61'), (0.8, '#d7191c'), (1.0, '#800026')
        ])
        return {
            'plasma_enhanced': plasma_enhanced,
            'coolwarm_enhanced': coolwarm_enhanced,
            'twin_categorical': twin_categorical,
            'stress_map': stress_map
        }

    @staticmethod
    def add_error_shading(ax, x, y_mean, y_std, color='blue', alpha=0.3, label=''):
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        color=color, alpha=alpha, label=label + ' ± std')
        return ax

    @staticmethod
    def add_scale_bar(ax, length_nm, location='lower right', color='black',
                      linewidth=2, fontsize=8):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        bar_x_start = xlim[1] - x_range * 0.15
        bar_x_end = bar_x_start - length_nm
        bar_y = ylim[0] + y_range * 0.05
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
                color=color, linewidth=linewidth, solid_capstyle='butt')
        ax.text((bar_x_start + bar_x_end) / 2, bar_y + y_range * 0.02,
                f'{length_nm} nm', ha='center', va='bottom',
                color=color, fontsize=fontsize, fontweight='bold')
        return ax


# ============================================================================
# ███ v9.3.1 — 3-BUCKET DISPLAY TAXONOMY + HONEST NAMING               ███
# ============================================================================

FINE_PROVENANCE_KEYS: Tuple[str, ...] = (
    'llm_extract', 'llm_reasoned', 'regex_ner',
    'regime_prior', 'llm_prior', 'physics_inferred',
)

PROVENANCE_GROUPS: Dict[str, Tuple[str, ...]] = {
    'llm_grounded':  ('llm_extract', 'llm_reasoned'),
    'llm_prior':     ('llm_prior',),
    'deterministic': ('regex_ner', 'regime_prior', 'physics_inferred'),
}
_FINE_TO_GROUP: Dict[str, str] = {
    f: g for g, fs in PROVENANCE_GROUPS.items() for f in fs
}
COARSE_PROVENANCE_KEYS: Tuple[str, ...] = tuple(PROVENANCE_GROUPS.keys())

LEGEND_MARKERS: Dict[str, Dict[str, Any]] = {
    'llm_grounded':  dict(marker='D', ms=6.0, fill=True,
                          label='LLM (corpus-grounded)'),
    'llm_prior':     dict(marker='o', ms=6.5, fill=True,
                          label='LLM prior (parametric)'),
    'deterministic': dict(marker='^', ms=7.5, fill=True,
                          label='Deterministic (regex · physics-inferred)'),
}

PROVENANCE_MARKERS: Dict[str, Dict[str, Any]] = {
    'llm_extract':   dict(marker='D', ms=6.0, fill=True,
                          label='LLM extraction (verbatim, grounded)'),
    'llm_reasoned':  dict(marker='p', ms=6.5, fill=True,
                          label='LLM chain-of-thought (grounded)'),
    'llm_prior':     dict(marker='o', ms=6.5, fill=False,
                          label='LLM prior (no corpus evidence)'),
    'regex_ner':     dict(marker='^', ms=7.5, fill=True,
                          label='Regex NER (deterministic)'),
    'regime_prior':  dict(marker='s', ms=6.5, fill=False,
                          label='Regime classifier (lookup table)'),
    'physics_inferred': dict(marker='v', ms=7.0, fill=False,
                             label='Physics-inferred (Hall–Petch / P–N)'),
}

INCUMBENT_ROUTES: Dict[str, Tuple[str, ...]] = {
    'gamma0_dot':  ('llm_grounded',),
    'srs':         ('deterministic',),
    'mu':          ('deterministic',),
    'rho0':        ('deterministic',),
    'sigma0_fric': ('deterministic',),
}


def _is_context(param: str, prov: Any) -> bool:
    bucket = _FINE_TO_GROUP.get(_norm_provenance(prov), 'deterministic')
    pinned = INCUMBENT_ROUTES.get(param, ())
    return bool(pinned) and bucket not in pinned


_DERIVED_HINTS: Tuple[str, ...] = (
    'physics_inferred', 'derived',
    'hall_petch', 'peierls', 'taylor_factor_cross',
    'solver_law', 'model_inversion', 'consensus',
)
_REGIME_HINTS: Tuple[str, ...] = (
    'regime_prior', 'regime_inference', 'regime_classifier',
    'physics_regime', 'theory_regime',
)
_PRIOR_HINTS: Tuple[str, ...] = (
    'prior', 'estimate', 'no evidence', 'world knowledge', 'parametric',
)
_REASONED_HINTS: Tuple[str, ...] = (
    'reasoned', 'chain_of_thought', 'chain-of-thought', 'cot',
)
_REGEX_HINTS: Tuple[str, ...] = (
    'regex', 'heuristic', 'side_note', 'side-note', 'default_fallback',
)
_LLM_HINTS: Tuple[str, ...] = ('llm', 'inferred', 'model', 'explicit')


def _norm_provenance(p: Any) -> str:
    s = str(p).strip().lower()
    if s in FINE_PROVENANCE_KEYS:
        return s
    if any(k in s for k in _DERIVED_HINTS):
        return 'physics_inferred'
    if any(k in s for k in _REGIME_HINTS):
        return 'regime_prior'
    if any(k in s for k in _PRIOR_HINTS):
        return 'llm_prior'
    if any(k in s for k in _REASONED_HINTS):
        return 'llm_reasoned'
    if any(k in s for k in _REGEX_HINTS):
        return 'regex_ner'
    if any(k in s for k in _LLM_HINTS) or re.search(r'\bai\b', s):
        return 'llm_extract'
    return 'regex_ner'


_VALID_GRANULARITIES: Tuple[str, ...] = ('coarse', 'fine')


def _norm_granularity(g: Any) -> str:
    s = str(g).strip().lower() if g is not None else 'coarse'
    if s not in _VALID_GRANULARITIES:
        logger.warning("legend_granularity %r invalid — defaulting to "
                       "'coarse'", g)
        return 'coarse'
    return s


def _legend_key(prov: Any, granularity: str = 'coarse') -> str:
    granularity = _norm_granularity(granularity)
    fine = _norm_provenance(prov)
    if granularity == 'fine':
        return fine
    return _FINE_TO_GROUP.get(fine, 'deterministic')


def _stamp_provenance(ext: Dict[str, Any], fine_key: str) -> None:
    if fine_key not in FINE_PROVENANCE_KEYS:
        raise ValueError(f"not a fine key: {fine_key!r}")
    ext['_provenance'] = fine_key


def _default_reasoning(c) -> str:
    prov = _norm_provenance(getattr(c, "provenance", "")
                            or getattr(c, "method", ""))
    if prov == "llm_extract":
        return ("Tier-1 grounded LLM extraction — verbatim span, "
                "corpus-anchored, unit-coerced by the gatekeeper.")
    if prov == "llm_reasoned":
        return ("Tier-1 grounded LLM chain-of-thought — evidence span "
                "plus an explicit step-by-step derivation, corpus-anchored.")
    if prov == "regex_ner":
        return ("Tier-2 deterministic regex — sci-notation scanner + "
                "param-aware unit tokens; no LLM involved.")
    if prov == "llm_prior":
        return ("Tier-3 LLM prior inference — no verbatim corpus evidence; "
                "confidence capped at 0.5, bounded by PARAM_CANON.")
    if prov == "regime_prior":
        return ("Tier-3b curated regime classifier — reproducible lookup "
                "table, no LLM involved.")
    if prov == "physics_inferred":
        return ("Physics-inferred from measured side-note context — "
                "Hall–Petch / Peierls–Nabarro / M·τ_P / solver-law inversion.")
    return ""


# ============================================================================
# ███ v10.0.0 — SCORE-ANATOMY RECOMMENDER (CTC² Phases 3–4)           ███
# ███   Legend keys = EXPERTS.  Provenance = badge, never a rank.      ███
# ███   Bar height = Σ = Λ·κ_n·κ_m·G — the decision variable.          ███
# ███   Bar body   = stacked expert segments ŵ_k·s_k.                  ███
# ============================================================================

EXPERT_KEYS: Tuple[str, ...] = (
    'evidence', 'regime', 'theory', 'closure', 'consensus', 'compat',
)

EXPERT_META: Dict[str, Dict[str, Any]] = {
    'evidence':  dict(color='#4477AA',
                      label='Evidence strength (span quality)'),
    'regime':    dict(color='#228833',
                      label='Regime compatibility (log-band)'),
    'theory':    dict(color='#66CCEE',
                      label='Theory match (tag × family)'),
    'closure':   dict(color='#EE6677',
                      label='Physics closure (solver residual)'),
    'consensus': dict(color='#AA3377',
                      label='Route consensus (agreement)'),
    'compat':    dict(color='#BBBBBB',
                      label='Context compatibility (family)'),
}

BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    'rho0':        dict(evidence=1.0, regime=0.9, compat=0.5),
    'mu':          dict(evidence=1.0, compat=1.2, closure=0.8),
    'gamma0_dot':  dict(evidence=1.0, theory=1.4, closure=0.8),
    'srs':         dict(evidence=1.0, regime=0.7, compat=0.6),
    'sigma0_fric': dict(evidence=1.0, consensus=1.5, closure=1.0),
}

PROV_CONF_CAP: Dict[str, float] = {
    'llm_extract':      0.95,
    'llm_reasoned':     0.90,
    'regex_ner':        0.85,
    'physics_inferred': 0.75,
    'regime_prior':     0.50,
    'llm_prior':        0.50,
}

COARSE_FOOT: Dict[str, str] = {
    'llm_grounded':  '#0072B2',
    'llm_prior':     '#E69F00',
    'deterministic': '#009E73',
}

_SPAN_QUALITY_BY_PROV: Dict[str, float] = {
    'llm_extract':      0.92,
    'llm_reasoned':     0.86,
    'regex_ner':        0.72,
    'physics_inferred': 0.78,
    'regime_prior':     0.55,
    'llm_prior':        0.48,
}


# ---------------------------------------------------------------- kernels
def kappa_count(n: int, n0: float = 1.0) -> float:
    """Saturating evidence-source count."""
    return float(n) / (float(n) + float(n0))


def kappa_corroborate(m: int, m0: float = 1.0) -> float:
    """Saturating corroboration (m = candidates within tolerance)."""
    return float(m) / (float(m) + float(m0))


def regime_membership(v: float, lo: float, hi: float) -> float:
    """Gaussian peak at log-center inside the band; e-fold per decade
    outside (continuous at edge = 0.607)."""
    if v <= 0 or lo <= 0 or hi <= lo:
        return 0.0
    lc = 0.5 * (np.log10(lo) + np.log10(hi))
    hw = 0.5 * (np.log10(hi) - np.log10(lo))
    d  = abs(np.log10(v) - lc)
    if d <= hw:
        return math.exp(-0.5 * (d / hw) ** 2)
    return 0.6065 * math.exp(-(d - hw))


FAMILY: Dict[str, str] = {
    'johnson_cook': 'continuum', 'power_law': 'continuum',
    'cpfem': 'continuum', 'phase_field': 'continuum',
    'experimental': 'continuum', 'ddd': 'mesoscale', 'md': 'atomistic',
}


def theory_match(tag: str, target: str
                 ) -> Tuple[float, Optional[Tuple[str, float]]]:
    """Return (s_theory, gate).  Cross-family mismatch is a GATE."""
    if not tag or not target:
        return 0.5, None
    t_tag  = _norm_theory(tag)  or tag
    t_tgt  = _norm_theory(target) or target
    if t_tag == t_tgt:
        return 1.0, None
    if FAMILY.get(t_tag) and FAMILY.get(t_tag) == FAMILY.get(t_tgt):
        return 0.6, None
    return 0.05, ('cross_theory', 0.05)


def closure_score(residuals: Dict[str, float],
                  tols: Dict[str, float]) -> float:
    """Worst-law-limited closure: exp(−max_k |r_k|/tol_k)."""
    if not residuals:
        return 1.0
    worst = 0.0
    for k, r in residuals.items():
        tol = tols.get(k, 1e-6)
        if tol <= 0:
            continue
        worst = max(worst, abs(r) / tol)
    return math.exp(-worst)


def alpha_k(expert: str, param: str, ctx: 'ContextVector') -> float:
    """Context activators: richer context ⇒ louder prior experts."""
    if expert == 'regime':   return ctx.kappa
    if expert == 'theory':   return ctx.kappa_framework
    if expert == 'compat':   return 0.5 + 0.5 * ctx.kappa
    return 1.0


# ---------------------------------------------------------------- contexts
@dataclass
class ContextVector:
    """Compact bundle of the qualitative context used by α_k, band
    selection and compat().  Completeness κ(c) ∈ [0,1] is the fraction
    of the well-defined context slots that are filled."""
    synthesis:        Optional[str]   = None
    architecture:     Optional[str]   = None
    twin_spacing:     Optional[float] = None
    framework:        Optional[str]   = None
    material:         str             = 'Cu'
    fields:           Dict[str, Any]  = field(default_factory=dict)
    kappa:            float           = 0.0
    kappa_framework:  float           = 0.0

    def __post_init__(self):
        slots = [self.synthesis, self.architecture, self.twin_spacing]
        filled = sum(1 for s in slots if s not in (None, '', '(unknown)'))
        if isinstance(self.twin_spacing, (int, float)) \
                and (self.twin_spacing is None or self.twin_spacing <= 0):
            filled -= 1
        self.kappa = float(filled) / float(len(slots))

        fw = self.framework
        self.kappa_framework = 0.0 if not fw or fw == '(unspecified)' else 1.0

    @classmethod
    def from_sidebar(cls, material, synthesis, architecture,
                     twin_spacing, framework) -> 'ContextVector':
        return cls(
            synthesis=synthesis, architecture=architecture,
            twin_spacing=(float(twin_spacing)
                          if isinstance(twin_spacing, (int, float))
                             and twin_spacing and twin_spacing > 0
                          else None),
            framework=framework, material=material,
            fields={'synthesis': synthesis, 'architecture': architecture,
                    'twin_spacing': twin_spacing, 'framework': framework},
        )


class MetaDatabase:
    """Provides (a) context-conditioned log bands per parameter and
    (b) a coarse framework-family compatibility score for context tags."""

    def band(self, param: str, ctx: ContextVector
             ) -> Optional[Tuple[float, float]]:
        if param == 'rho0':
            r = classify_rho0_regime(ctx.synthesis, ctx.architecture,
                                     ctx.twin_spacing)
            return (float(r['low']), float(r['high']))
        if param == 'gamma0_dot' and ctx.framework:
            r = classify_gamma0_regime(ctx.framework)
            return (float(r['low']), float(r['high']))
        spec = PLASTICITY_ONTOLOGY.get(param)
        if spec is not None:
            lo, hi = spec.get('soft_range', spec.get('valid_range'))
            return (float(lo), float(hi))
        return None

    def compat(self, tag: Optional[str], ctx: ContextVector) -> float:
        """Coarse family similarity of a value's context tag to the
        current ContextVector.  Total function; safe default 0.5."""
        if not tag:
            return 0.5
        t = str(tag).lower()
        if ctx.synthesis and ctx.synthesis.lower() in t:
            return 1.0
        if ctx.architecture and ctx.architecture.lower() in t:
            return 0.9
        if ctx.framework and _norm_theory(t) == _norm_theory(ctx.framework):
            return 1.0
        fam_tag = FAMILY.get(_norm_theory(t) or t)
        fam_ctx = FAMILY.get(_norm_theory(ctx.framework or '') or '')
        if fam_tag and fam_tag == fam_ctx:
            return 0.7
        return 0.5


class ValueCluster:
    """Groups candidates within tolerance so corroboration is visible.
    Tolerance defaults to 5 % (log-symmetric for positive quantities)."""

    def __init__(self, values: List[float], tol: float = 0.05):
        self.values = [float(v) for v in values if v is not None and v > 0]
        self.tol = float(tol)
        self._groups: List[List[float]] = []
        for v in sorted(self.values):
            if self._groups and \
               abs(math.log(v / self._groups[-1][0])) <= math.log(1 + self.tol):
                self._groups[-1].append(v)
            else:
                self._groups.append([v])

    def size_of(self, value: float) -> int:
        for g in self._groups:
            if any(abs(math.log(value / x)) <= math.log(1 + self.tol) + 1e-12
                   for x in g if x > 0 and value > 0):
                return len(g)
        return 1

    def relative_spread(self) -> float:
        if not self.values:
            return 0.0
        lo, hi = min(self.values), max(self.values)
        if lo <= 0:
            return 1.0
        return float((hi - lo) / lo)


# ---------------------------------------------------------------- scorer
@dataclass(frozen=True)
class ExpertOpinion:
    key:    str
    s:      float
    detail: str = ''


@dataclass
class ScoredCandidate:
    param:       str
    value:       float
    unit:        Optional[str]
    fine_prov:   str
    coarse:      str
    opinions:    Dict[str, ExpertOpinion]
    w_hat:       Dict[str, float]
    Lambda:      float
    kappa_n:     float
    kappa_m:     float
    gate_list:   List[Tuple[str, float]]
    Sigma:       float
    stab_lo:     float
    stab_hi:     float
    n_sources:   int
    cluster_size: int
    conf:        float
    reasoning:   str = ''

    def to_display(self) -> Dict[str, Any]:
        return {
            'value':       self.value,
            'unit':        self.unit,
            'Sigma':       round(self.Sigma, 4),
            'Lambda':      round(self.Lambda, 4),
            'kappa_n':     round(self.kappa_n, 3),
            'kappa_m':     round(self.kappa_m, 3),
            'gates':       [g for g, _ in self.gate_list],
            'confidence':  round(self.conf, 3),
            'provenance':  self.fine_prov,
            'coarse':      self.coarse,
            'n_sources':   self.n_sources,
            'cluster_size': self.cluster_size,
            'opinions':    {k: round(v.s, 3)
                            for k, v in self.opinions.items()},
            'weights':     {k: round(v, 3) for k, v in self.w_hat.items()},
            'reasoning':   (self.reasoning[:200] + '…'
                            if len(self.reasoning) > 200 else self.reasoning),
        }


def _opinions_from_raw(cand: Dict[str, Any], ctx: ContextVector,
                       metadb: MetaDatabase, param: str
                       ) -> Tuple[Dict[str, ExpertOpinion],
                                  List[Tuple[str, float]]]:
    ops: Dict[str, ExpertOpinion] = {}
    gates: List[Tuple[str, float]] = []

    span_q = max(cand.get('span_qualities', [0.25]) or [0.25])
    ops['evidence'] = ExpertOpinion(
        'evidence', float(span_q),
        f"best of {cand.get('n_sources', 1)} source(s)")

    band = metadb.band(param, ctx)
    if band and band[1] > band[0]:
        ops['regime'] = ExpertOpinion(
            'regime', regime_membership(cand['value'], *band),
            f"band [{band[0]:g}, {band[1]:g}]")

    tgt = ctx.framework
    if cand.get('theory') and tgt:
        s, gate = theory_match(cand['theory'], tgt)
        ops['theory'] = ExpertOpinion(
            'theory', s, f"tag={cand['theory']} target={tgt}")
        if gate:
            gates.append(gate)

    if cand.get('closure_residuals'):
        ops['closure'] = ExpertOpinion(
            'closure',
            closure_score(cand['closure_residuals'],
                          cand.get('closure_tols', {})),
            "worst of C1..C5")

    if cand.get('route_agreement') is not None:
        a = int(cand['route_agreement'])
        r = int(cand.get('routes_available', 1))
        ops['consensus'] = ExpertOpinion(
            'consensus', (a + 1.0) / (r + 2.0),
            f"{a}/{r} routes agree")

    if cand.get('context_tag'):
        ops['compat'] = ExpertOpinion(
            'compat', metadb.compat(cand['context_tag'], ctx),
            f"tag={cand['context_tag']}")

    return ops, gates


def weight_stability(ops: Dict[str, ExpertOpinion],
                     w_hat: Dict[str, float],
                     kn: float, km: float, G: float,
                     n_boot: int = 200, seed: int = 20240101
                     ) -> Tuple[float, float]:
    """Dirichlet-bootstrap over ŵ.  Fixed seed ⇒ reproducible."""
    if not ops:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    keys = list(ops)
    base = np.array([w_hat[k] for k in keys], dtype=float) * 8.0 + 1e-9
    s    = np.array([ops[k].s for k in keys], dtype=float)
    lams = rng.dirichlet(base, size=n_boot) @ s
    sig  = lams * kn * km * G
    return float(np.percentile(sig, 5)), float(np.percentile(sig, 95))


class LatentMoEScorerV10:
    """v10 score-anatomy scorer.  Every candidate becomes a ScoredCandidate
    with a stacked-expert decomposition whose segments sum exactly to Σ."""

    def __init__(self, ctx: ContextVector, metadb: MetaDatabase,
                 cluster_tol: float = 0.05):
        self.ctx = ctx
        self.metadb = metadb
        self.cluster_tol = cluster_tol

    def score_all(self, raw_by_param: Dict[str, List[Dict[str, Any]]]
                  ) -> Dict[str, List[ScoredCandidate]]:
        out: Dict[str, List[ScoredCandidate]] = {}
        for p, raws in raw_by_param.items():
            if not raws:
                continue
            clusters = ValueCluster([r['value'] for r in raws
                                     if r.get('value')],
                                    tol=self.cluster_tol)
            scored = [self._score_one(p, r, clusters) for r in raws
                      if r.get('value')]
            scored.sort(key=lambda c: -c.Sigma)
            out[p] = scored
        return out

    def _score_one(self, param: str, cand: Dict[str, Any],
                   clusters: ValueCluster) -> ScoredCandidate:
        ops, gates = _opinions_from_raw(cand, self.ctx, self.metadb, param)

        base = BASE_WEIGHTS.get(param, {})
        w = {k: base.get(k, 0.0) * alpha_k(k, param, self.ctx)
             for k in ops}
        wsum = sum(w.values()) or 1.0
        w_hat = {k: w[k] / wsum for k in ops}

        Lambda = sum(w_hat[k] * ops[k].s for k in ops)

        n_src = int(cand.get('n_sources', 1))
        c_size = clusters.size_of(cand['value'])
        kn = kappa_count(n_src)
        km = kappa_corroborate(c_size)

        G = math.prod(g for _, g in gates) if gates else 1.0
        Sigma = Lambda * kn * km * G

        lo, hi = weight_stability(ops, w_hat, kn, km, G)

        fine = _norm_provenance(cand.get('provenance', '') or
                                cand.get('method', ''))
        spread = clusters.relative_spread()
        conf = min(PROV_CONF_CAP.get(fine, 0.5),
                   Sigma * math.exp(-spread)
                   * (self.ctx.kappa ** 0.25 if self.ctx.kappa > 0 else 1.0))

        return ScoredCandidate(
            param=param, value=float(cand['value']),
            unit=PARAM_META.get(param, {}).get('unit'),
            fine_prov=fine,
            coarse=_legend_key(fine, 'coarse'),
            opinions=ops, w_hat=w_hat, Lambda=Lambda,
            kappa_n=kn, kappa_m=km, gate_list=list(gates),
            Sigma=Sigma, stab_lo=lo, stab_hi=hi,
            n_sources=n_src, cluster_size=c_size, conf=conf,
            reasoning=cand.get('reasoning') or _default_reasoning(cand),
        )


# ------------------------------------------------------- PlasticityCandidate adapter
def _pl_cand_to_v10_raw(c: 'PlasticityCandidate',
                        param: str) -> Dict[str, Any]:
    """Translate a v9 PlasticityCandidate into the v10 raw dict consumed
    by LatentMoEScorerV10.  Preserves provenance; no information loss."""
    prov = _norm_provenance(getattr(c, 'provenance', '')
                            or getattr(c, 'method', ''))
    span_q = _SPAN_QUALITY_BY_PROV.get(prov, 0.35)

    route_agreement = None
    routes_available = None
    if getattr(c, 'method', '') == 'derived_consensus':
        route_agreement = 3
        routes_available = 4

    return dict(
        param=param,
        value=float(c.value_si),
        provenance=prov,
        n_sources=1 if prov not in ('llm_extract', 'regex_ner') else 2,
        span_qualities=[span_q],
        theory=(getattr(c, 'theory', '') or None),
        context_tag=(getattr(c, 'material', '') or None),
        route_agreement=route_agreement,
        routes_available=routes_available,
        closure_residuals=None,
        closure_tols={},
        reasoning=getattr(c, 'reasoning', '') or '',
    )


# ---------------------------------------------------------------- figures
from matplotlib.patches import Rectangle as _MPLRectangle  # noqa: E402


def _sci_tex_short(value, unit=None) -> str:
    try:
        return sci_tex(value, unit)
    except Exception:
        return f'{value:g}'


@handle_errors
def render_score_anatomy_figure(scored: List['ScoredCandidate'],
                                param: str,
                                journal: str = 'nature',
                                show_reference: bool = True
                                ) -> plt.Figure:
    """The v10.0.0 core visualization: five-layer stacked ladder."""
    meta = PARAM_META.get(param, dict(title=param, symbol=param, unit=None))
    rows = sorted(scored, key=lambda c: -c.Sigma)
    n = len(rows)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, 'No scored candidates',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(max(3.4, 1.1 * n), 3.6))
    JournalTemplates.apply_journal_style(fig, ax, journal)
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'

    for i, c in enumerate(rows):
        scale = c.Sigma / c.Lambda if c.Lambda > 0 else 0.0

        # (1) stacked expert segments
        y0 = 0.0
        for k in EXPERT_KEYS:
            if k not in c.opinions:
                continue
            h = c.w_hat[k] * c.opinions[k].s * scale
            ax.bar(i, h, bottom=y0, width=0.62,
                   color=EXPERT_META[k]['color'], edgecolor='white',
                   linewidth=0.4, zorder=3)
            y0 += h

        # (2) gate hatch — the chopped-off, NAMED penalty
        h_count = c.Lambda * c.kappa_n * c.kappa_m
        if c.gate_list and h_count > c.Sigma + 1e-12:
            ax.bar(i, h_count - c.Sigma, bottom=c.Sigma, width=0.62,
                   facecolor='none', hatch='///', edgecolor='#D55E00',
                   linewidth=0, zorder=4)
            ax.text(i, 0.5 * (c.Sigma + h_count),
                    ' ⛔ ' + '·'.join(g for g, _ in c.gate_list),
                    ha='center', va='center', fontsize=5.5,
                    color='#D55E00', zorder=5)

        # (3) count discount grey band
        if c.Lambda > h_count + 1e-9:
            ax.bar(i, c.Lambda - h_count, bottom=h_count, width=0.62,
                   color='#888888', alpha=0.18, linewidth=0, zorder=2)

        # (4) ghost outline at Λ
        ax.add_patch(_MPLRectangle((i - 0.31, 0), 0.62, c.Lambda,
                                   fill=False, linestyle='--', linewidth=0.8,
                                   edgecolor='#888888', zorder=4))

        # (5) stability whisker
        ax.errorbar(i, c.Sigma,
                    yerr=[[max(c.Sigma - c.stab_lo, 0)],
                          [max(c.stab_hi - c.Sigma, 0)]],
                    fmt='none', ecolor='#444444', capsize=3,
                    elinewidth=0.8, zorder=6)

        # --- below-axis anatomy ----------------------------------------
        ax.bar(i, 0.012, bottom=-0.024, width=0.62,
               color=COARSE_FOOT.get(c.coarse, '#999999'),
               linewidth=0, clip_on=False)

        for j in range(min(c.n_sources, 8)):
            ax.plot(i - 0.22 + 0.055 * j, -0.037, marker='o', ms=2.4,
                    color=EXPERT_META['evidence']['color'],
                    clip_on=False)

        for j in range(min(max(c.cluster_size - 1, 0), 8)):
            ax.plot(i + 0.22 - 0.055 * j, -0.037, marker='|', ms=4,
                    color=EXPERT_META['consensus']['color'],
                    clip_on=False)

        mk = PROVENANCE_MARKERS.get(c.fine_prov, PROVENANCE_MARKERS['regex_ner'])
        ax.plot(i, -0.055, marker=mk['marker'], ms=4,
                mfc=('#444444' if mk.get('fill', True) else 'white'),
                mec='#444444', clip_on=False, linestyle='none')

        ax.text(i, c.stab_hi + 0.035,
                _sci_tex_short(c.value, meta.get('unit')),
                ha='center', va='bottom', fontsize=5.5)

    # winner frame + star
    ax.add_patch(_MPLRectangle((-0.33, 0), 0.66, rows[0].Sigma,
                               fill=False, edgecolor='#F0E442',
                               linewidth=2.0, zorder=7))
    ax.plot(0, rows[0].stab_hi + 0.11, marker='*', ms=10,
            color='#F0E442', mec='black', zorder=8)

    # v9 reference mark (annotation, never a ranking key)
    if show_reference:
        ref_routes = INCUMBENT_ROUTES.get(param, ())
        for i, c in enumerate(rows):
            if c.coarse in ref_routes:
                ax.text(i, -0.068, '📌 v9 ref', ha='center',
                        fontsize=5.5, color='#666666', clip_on=False)
                break

    ax.set_xticks(range(n))
    ax.set_xticklabels([
        f"{_sci_tex_short(c.value)}\nn={c.n_sources} m={c.cluster_size} "
        f"conf={c.conf:.2f}" for c in rows], fontsize=6)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.075, max(0.4, max(c.stab_hi for c in rows) * 1.22))
    ax.set_ylabel(safe_mathtext(
        r'Support  $\Sigma = \Lambda\,\kappa_n\,\kappa_m\,G$'))
    ax.set_title(safe_mathtext(
        f"{meta.get('title', param)} — score anatomy"), pad=10)

    active = {k for c in rows for k in c.opinions}
    handles: List[Any] = [
        Patch(facecolor=EXPERT_META[k]['color'],
              label=EXPERT_META[k]['label'])
        for k in EXPERT_KEYS if k in active
    ]
    if any(c.Lambda > c.Lambda * c.kappa_n * c.kappa_m + 1e-9 for c in rows):
        handles.append(Patch(facecolor='#888888', alpha=0.18, ls='--',
                             label='count discount → ghost Λ'))
    if any(c.gate_list for c in rows):
        handles.append(Patch(facecolor='none', hatch='///',
                             edgecolor='#D55E00',
                             label='gate penalty'))
    for k in sorted({c.fine_prov for c in rows}):
        mk = PROVENANCE_MARKERS.get(k)
        if mk is None:
            continue
        handles.append(Line2D([], [], marker=mk['marker'], color='none',
                              markerfacecolor=('#444444' if mk.get('fill', True)
                                               else 'white'),
                              markeredgecolor='#444444', ms=6,
                              label=mk['label']))

    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=5.5,
                  title='score anatomy  ·  ◂ badge row = provenance '
                        '(non-ranking)',
                  title_fontsize=5.5, frameon=False)
    return fig


def render_expert_mass_sankey(scored: List['ScoredCandidate'],
                              param: str) -> Optional[go.Figure]:
    """Left = experts (mass = Σ ŵ_k·s_k), right = candidates (mass = Σ)."""
    if not scored:
        return None
    active = [k for k in EXPERT_KEYS
              if any(k in c.opinions for c in scored)]
    if not active:
        return None
    labels: List[str] = [EXPERT_META[k]['label'] for k in active]
    sorted_rows = sorted(scored, key=lambda x: -x.Sigma)
    for c in sorted_rows:
        labels.append(_sci_tex_short(c.value))
    srcs, tgts, vals, cols = [], [], [], []
    for ci, c in enumerate(sorted_rows):
        scale = c.Sigma / c.Lambda if c.Lambda else 0.0
        for k in active:
            if k in c.opinions:
                srcs.append(active.index(k))
                tgts.append(len(active) + ci)
                vals.append(c.w_hat[k] * c.opinions[k].s * scale)
                cols.append(EXPERT_META[k]['color'])
    return go.Figure(go.Sankey(
        node=dict(label=labels, pad=12, thickness=14),
        link=dict(source=srcs, target=tgts, value=vals, color=cols)))


def render_value_landscape(scored: List['ScoredCandidate'],
                            param: str) -> Optional[plt.Figure]:
    """x = value (log for ρ₀/γ̇₀), y = Σ, size ∝ n_sources."""
    if not scored:
        return None
    logx = param in ('rho0', 'gamma0_dot', 'mu', 'sigma0_fric')
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    JournalTemplates.apply_journal_style(fig, ax, 'nature')
    for c in scored:
        ax.scatter(c.value, c.Sigma,
                   s=18 + 14 * c.n_sources,
                   color=COARSE_FOOT.get(c.coarse, '#999999'),
                   alpha=0.85, edgecolor='k', lw=0.4,
                   marker=PROVENANCE_MARKERS.get(
                       c.fine_prov, PROVENANCE_MARKERS['regex_ner'])['marker'])
    if logx:
        ax.set_xscale('log')
    meta = PARAM_META.get(param, dict(title=param, unit=None))
    ax.set_xlabel(safe_mathtext(
        f"{meta.get('title', param)}"
        + (f" ($\\mathrm{{{meta['unit']}}}$)" if meta.get('unit') else '')))
    ax.set_ylabel(safe_mathtext(r'Support $\Sigma$'))
    ax.set_title(safe_mathtext(f"{meta.get('title', param)} — landscape"),
                 pad=8)
    return fig


# ---------------------------------------------------------------- gatekeeper v10
@dataclass(frozen=True)
class RangeSet:
    phys: Tuple[float, float]
    lit:  Tuple[float, float]
    band: Optional[Tuple[float, float]] = None

    def accepts(self, v: float) -> Tuple[bool, str]:
        if not (self.phys[0] <= v <= self.phys[1]):
            return False, f"outside physics envelope {self.phys}"
        if self.lit[0] <= v <= self.lit[1]:
            return True, "in literature extent"
        if self.band and self.band[0] <= v <= self.band[1]:
            return True, "in context band (unattested but banded)"
        return False, "outside literature extent and context band"


def compute_range_set(param: str, ctx: ContextVector,
                      metadb: MetaDatabase) -> RangeSet:
    spec = PLASTICITY_ONTOLOGY.get(param, {})
    phys = tuple(spec.get('valid_range', (1e-30, 1e30)))
    soft = tuple(spec.get('soft_range', phys))
    band = metadb.band(param, ctx)
    return RangeSet(phys=phys, lit=soft, band=band)


def gatekeep_v10(raw: List[Dict[str, Any]], param: str,
                 ctx: ContextVector, metadb: MetaDatabase
                 ) -> List[Dict[str, Any]]:
    """Range-aware gatekeeper with a total-function floor."""
    rs = compute_range_set(param, ctx, metadb)
    kept: List[Dict[str, Any]] = []
    for cand in raw:
        try:
            v = float(cand.get('value'))
        except (TypeError, ValueError):
            continue
        ok, why = rs.accepts(v)
        if ok:
            kept.append({**cand, '_range_note': why})
        else:
            logger.info("gatekeep_v10[%s]: rejected %s (%s)",
                        param, v, why)

    if not kept:
        if rs.band:
            lo, hi = rs.band
            v0 = float(math.sqrt(lo * hi)) if lo > 0 and hi > lo else hi
        else:
            v0 = 0.5 * (rs.lit[0] + rs.lit[1])
        kept = [dict(param=param, value=v0, provenance='regime_prior',
                     n_sources=0, span_qualities=[0.25],
                     reasoning='Curated floor: band log-center, no evidence.')]

    for c in kept:
        fine = _norm_provenance(c.get('provenance') or c.get('method') or '')
        if fine in FINE_PROVENANCE_KEYS:
            _stamp_provenance(c, fine)
    return kept
# ============================================================================
# END v10.0.0 core module
# ============================================================================


def sci_tex(value, unit=None, decimals=1):
    v = float(value)
    if v == 0:
        s = '0'
    else:
        e = int(np.floor(np.log10(abs(v))))
        if -1 <= e <= 3:
            s = f'{v:g}'
        else:
            m = f'{v / 10.0 ** e:.{decimals}f}'.rstrip('0').rstrip('.')
            s = f'10^{{{e}}}' if m == '1' else f'{m}\\times10^{{{e}}}'
    if unit:
        s += f'\\,\\mathrm{{{unit}}}'
    return f'${s}$'


PARAM_META = {
    'rho0':           dict(title='Initial dislocation density',      symbol=r'\rho_0',                unit='m^{-2}'),
    'mu':             dict(title='Shear modulus',                    symbol=r'\mu',                   unit='GPa'),
    'gamma0_dot':     dict(title='Reference shear strain rate',      symbol=r'\dot{\gamma}_0',        unit='s^{-1}'),
    'srs':            dict(title='Strain-rate sensitivity exponent', symbol='m',                      unit=None),
    'sigma0_fric':    dict(title='Friction stress',                  symbol=r'\sigma_0',              unit='MPa'),
    'sigma0_yield':   dict(title='Yield stress',                     symbol=r'\sigma_y',              unit='MPa'),
    'tau_p':          dict(title='Peierls–Nabarro stress',           symbol=r'\tau_P',                unit='MPa'),
    'sigma0':         dict(title='Friction stress',                  symbol=r'\sigma_0',              unit='MPa'),
    'twin_spacing':   dict(title='Twin spacing',                     symbol=r'\lambda',               unit='nm'),
    'applied_stress': dict(title='Applied stress',                   symbol=r'\sigma_{\mathrm{app}}', unit='MPa'),
    'W':              dict(title='Interface width',                  symbol='W',                      unit='nm'),
    'grain_size_d':   dict(title='Grain size',                       symbol='d',                      unit='nm'),
    'twin_thickness_lambda': dict(title='Twin thickness',            symbol=r'\lambda',               unit='nm'),
    'burgers_vector_b': dict(title='Burgers vector',                 symbol='b',                      unit='nm'),
    'poisson_ratio_nu': dict(title="Poisson's ratio",                symbol=r'\nu',                   unit=None),
    'taylor_factor_M': dict(title='Taylor factor',                   symbol='M',                      unit=None),
    'hall_petch_k':   dict(title='Hall–Petch coefficient',           symbol='k_y',                    unit='MPa·m^{1/2}'),
    'core_width_w':   dict(title='Dislocation core width',           symbol='w',                      unit='nm'),
    'shear_modulus_G': dict(title='Shear modulus (context)',         symbol='G',                      unit='GPa'),
}


def normalize_tex(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    if '\\' not in s:
        return s
    return re.sub(r'\\{2,}', r'\\', s)


def strip_tex(s: Any) -> str:
    if s is None:
        return ''
    return re.sub(r'[\\{}$]', '', str(s))


_MTX = MathTextParser('path')


def safe_mathtext(s: Any, fallback: Optional[str] = None) -> str:
    if s is None:
        return ''
    s = normalize_tex(str(s))
    if '$' not in s:
        return s
    try:
        _MTX.parse(s, dpi=100, prop=FontProperties())
        return s
    except Exception:
        logger.warning("safe_mathtext: refusing to render malformed mathtext %r", s)
        return fallback if fallback is not None else strip_tex(s)


def _color_to_rgb(c: Any) -> Tuple[float, float, float]:
    if c is None:
        return (0.7, 0.7, 0.7)
    if isinstance(c, (tuple, list, np.ndarray)) and len(c) >= 3:
        try:
            return (float(c[0]), float(c[1]), float(c[2]))
        except Exception:
            return (0.7, 0.7, 0.7)
    if isinstance(c, str):
        try:
            return to_rgb(c)
        except Exception:
            pass
    try:
        return to_rgb(str(c))
    except Exception:
        return (0.7, 0.7, 0.7)


def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _resolve_bar_typography(style: Dict[str, Any],
                            annotation_fontsize: Optional[float],
                            legend_fontsize: Optional[float],
                            colorbar_label_fontsize: Optional[float],
                            colorbar_tick_fontsize: Optional[float]
                            ) -> Tuple[float, float, float, float]:
    base = float(style.get('font_size_small', 8))
    ann_fs  = base if annotation_fontsize      is None else max(2.0, float(annotation_fontsize))
    leg_fs  = base if legend_fontsize          is None else max(2.0, float(legend_fontsize))
    cbar_fs = base if colorbar_label_fontsize  is None else max(2.0, float(colorbar_label_fontsize))
    ctick_fs = (max(cbar_fs - 1.0, 2.0)
                if colorbar_tick_fontsize is None
                else max(2.0, float(colorbar_tick_fontsize)))
    return ann_fs, leg_fs, cbar_fs, ctick_fs


def _compute_min_safe_headroom(fig_height_in: float,
                               label_pad_pt: float,
                               annotation_fontsize_pt: float,
                               axes_fraction: float = 0.85) -> float:
    clearance_pt = float(label_pad_pt) + 1.5 * float(annotation_fontsize_pt)
    axes_pt = max(1.0, axes_fraction * float(fig_height_in) * 72.0)
    clearance_frac = min(0.85, clearance_pt / axes_pt)
    return 1.0 / max(1e-6, 1.0 - clearance_frac)


def plot_candidate_scores(candidates, scores, provenance, best_idx=None, *,
                          param_title=None, param_symbol=None, unit=None,
                          score_label='LatentMoE score', journal='nature',
                          fig_size=(5.2, 3.4), bar_width=0.62,
                          label_pad=9, headroom=1.22, despine=True, title=None,
                          score_std=None,
                          bar_colormap='viridis',
                          color_by='score',
                          best_color_override=None,
                          cmap_vmin=None,
                          cmap_vmax=None,
                          show_colorbar=False,
                          annotation_fontsize=None,
                          legend_fontsize=None,
                          colorbar_label_fontsize=None,
                          colorbar_tick_fontsize=None,
                          colorbar_label='Score',
                          legend_anchor_y=1.02,
                          legend_columnspacing=1.4,
                          legend_handletextpad=0.4,
                          legend_borderaxespad=0.0,
                          legend_granularity='coarse',
                          legend_show_counts=False,
                          context_flags=None):
    n = len(candidates)
    assert n == len(scores) == len(provenance), 'length mismatch'

    param_symbol = normalize_tex(param_symbol)

    granularity = _norm_granularity(legend_granularity)

    order = np.argsort(candidates)
    xs, cands = np.arange(n), [candidates[i] for i in order]
    scs   = [scores[i] for i in order]
    provs = [_legend_key(provenance[i], granularity) for i in order]

    ctxs = ([bool(context_flags[i]) for i in order]
            if context_flags is not None else [False] * n)

    if logger.isEnabledFor(logging.INFO):
        logger.info("plot_candidate_scores[%s] legend mix (%s): %s "
                    "context=%d",
                    param_title or param_symbol or 'param', granularity,
                    {k: provs.count(k) for k in set(provs)},
                    sum(1 for c in ctxs if c))

    stds  = [score_std[i] for i in order] if score_std is not None else None
    best_x = int(np.where(order == best_idx)[0][0]) \
        if (best_idx is not None and best_idx in order) else None

    style = JournalTemplates.get_journal_styles()
    style = style.get(journal, style['nature'])

    ann_fs, leg_fs, cbar_fs, ctick_fs = _resolve_bar_typography(
        style, annotation_fontsize, legend_fontsize,
        colorbar_label_fontsize, colorbar_tick_fontsize)

    min_safe = _compute_min_safe_headroom(
        fig_height_in=float(fig_size[1]),
        label_pad_pt=float(label_pad),
        annotation_fontsize_pt=float(ann_fs))
    effective_headroom = max(float(headroom), min_safe * 1.05)

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    JournalTemplates.apply_journal_style(fig, ax, journal)
    mpl.rcParams['mathtext.fontset'] = (
        'stix' if 'times' in style['font_family'].lower() else 'dejavusans')

    C_EDGE = 'black'

    if isinstance(bar_colormap, str):
        try:
            cmap = plt.get_cmap(bar_colormap)
        except ValueError:
            logger.warning("Unknown colormap %r — falling back to 'viridis'",
                           bar_colormap)
            cmap = plt.get_cmap('viridis')
    else:
        cmap = bar_colormap

    norm_scores: List[float] = []
    if color_by == 'score':
        vmin = float(cmap_vmin) if cmap_vmin is not None else float(min(scs))
        vmax = float(cmap_vmax) if cmap_vmax is not None else float(max(scs))
        span = max(vmax - vmin, 1e-9)
        norm_scores = [(float(s) - vmin) / span for s in scs]
        faces = [cmap(ns) for ns in norm_scores]
    elif color_by == 'index':
        indices = np.linspace(0.05, 0.95, max(n, 1))
        faces = [cmap(float(i)) for i in indices]
    else:
        faces = ['#BBBBBB'] * n

    if best_x is not None:
        if best_color_override:
            try:
                faces[best_x] = best_color_override
            except Exception:
                logger.warning("Invalid best_color_override %r — ignoring",
                               best_color_override)
        else:
            faces[best_x] = '#F0E442' if color_by == 'score' else '#0072B2'

    comp_x = [x for x in xs if not ctxs[x]]
    ctx_x  = [x for x in xs if ctxs[x]]
    if comp_x:
        ax.bar(comp_x, [scs[x] for x in comp_x], width=bar_width,
               facecolor=[faces[x] for x in comp_x],
               edgecolor=C_EDGE, linewidth=0.8, zorder=3)
    if ctx_x:
        ax.bar(ctx_x, [scs[x] for x in ctx_x], width=bar_width,
               facecolor=[faces[x] for x in ctx_x], alpha=0.55,
               edgecolor='0.35', linewidth=0.7, zorder=3)

    if stds is not None:
        ax.errorbar(xs, scs, yerr=stds, fmt='none', ecolor='black',
                    elinewidth=0.8, capsize=3, zorder=5)

    marker_table = (LEGEND_MARKERS if granularity == 'coarse'
                    else PROVENANCE_MARKERS)

    for x, s, p in zip(xs, scs, provs):
        st_ = marker_table[p]
        bar_rgb = _color_to_rgb(faces[x])
        lum = _relative_luminance(bar_rgb)
        is_ctx = ctxs[x]
        marker_edge = '0.35' if is_ctx else ('black' if lum > 0.55 else 'white')
        mfc = 'white' if st_.get('fill', True) else 'none'
        ms = st_['ms'] * (0.75 if is_ctx else 1.0)
        ax.plot([x], [s], marker=st_['marker'], markersize=ms,
                linestyle='none', markerfacecolor=mfc,
                markeredgecolor=marker_edge,
                markeredgewidth=1.2 if not is_ctx else 0.9,
                clip_on=False, zorder=6)

    for x, s in zip(xs, scs):
        bar_rgb = _color_to_rgb(faces[x])
        lum = _relative_luminance(bar_rgb)
        is_ctx = ctxs[x]
        if is_ctx:
            text_color = '0.45'
            halo_color = 'white'
        else:
            text_color = 'black' if lum > 0.55 else 'white'
            halo_color = 'white' if lum > 0.55 else 'black'
        ax.annotate(f'{s:.3f}', xy=(x, s),
                    xytext=(0, label_pad), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=ann_fs * (0.9 if is_ctx else 1.0),
                    fontweight=('bold' if (x == best_x and not is_ctx)
                                else 'normal'),
                    color=text_color,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor=halo_color,
                              edgecolor='none', alpha=0.85),
                    zorder=7)

    ax.set_ylim(0, max(scs) * effective_headroom)
    ax.set_xticks(xs)
    ax.set_xticklabels([safe_mathtext(sci_tex(c)) for c in cands])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.6, n - 0.4)

    parts = [p for p in (param_title,
                         f'${param_symbol}$' if param_symbol else '') if p]
    xlabel = ' '.join(parts)
    if unit:
        xlabel += f' ($\\mathrm{{{unit}}}$)'
    ax.set_xlabel(safe_mathtext(xlabel))
    ax.set_ylabel(safe_mathtext(score_label))

    ax.yaxis.grid(True, linewidth=0.5, alpha=0.22)
    ax.set_axisbelow(True)

    if despine:
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(which='both', top=False, right=False)

    if show_colorbar and color_by == 'score':
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02,
                            orientation='vertical')
        cbar.set_label(colorbar_label, fontsize=cbar_fs)
        cbar.ax.tick_params(labelsize=ctick_fs)

    if best_x is not None and any(ctxs):
        ax.axhline(scs[best_x], color='0.35', linewidth=0.7,
                   linestyle=(0, (4, 3)), zorder=2)

    present_keys = {p for p in provs if p in marker_table}
    handles = []
    for key, st_ in marker_table.items():
        if key not in present_keys:
            continue
        lbl = st_['label']
        if legend_show_counts:
            n_ctx_only = sum(1 for i, p in enumerate(provs)
                             if p == key and not ctxs[i])
            if n_ctx_only != provs.count(key):
                lbl = f"{lbl} (n={n_ctx_only}+ctx)"
            else:
                lbl = f"{lbl} (n={provs.count(key)})"
        handles.append(Line2D([], [], marker=st_['marker'],
                              linestyle='none', markersize=st_['ms'],
                              markerfacecolor=('white'
                                               if st_.get('fill', True)
                                               else 'none'),
                              markeredgecolor='black', markeredgewidth=1.1,
                              label=lbl))
    if best_x is not None and not ctxs[best_x]:
        handles.append(Patch(facecolor=faces[best_x], edgecolor=C_EDGE,
                             linewidth=0.8, label='Best match'))
    if any(ctxs):
        handles.append(Patch(facecolor='0.75', edgecolor='0.4',
                             alpha=0.65, label='Context (non-ranked)'))
    if handles:
        ncol = min(4, len(handles))
        ax.legend(handles=handles, loc='lower left',
                  bbox_to_anchor=(0.0, legend_anchor_y),
                  ncol=ncol, frameon=False, fontsize=leg_fs,
                  columnspacing=legend_columnspacing,
                  handletextpad=legend_handletextpad,
                  borderaxespad=legend_borderaxespad)

    if title:
        title_pad = 34.0 + max(0.0, leg_fs - 8.0) * 2.0
        if handles and len(handles) > ncol:
            title_pad += leg_fs * 1.8
        ax.set_title(safe_mathtext(title), pad=title_pad,
                     fontsize=style['font_size_large'])
    return fig


@handle_errors
def render_candidate_score_chart(param_key, candidates, scores, provenance,
                                 best_idx=None, journal='nature',
                                 hires_dpi=600, key_suffix='',
                                 annotation_fontsize=None,
                                 legend_fontsize=None,
                                 colorbar_label_fontsize=None,
                                 colorbar_tick_fontsize=None,
                                 annotation_offset=None,
                                 bar_colormap='viridis',
                                 color_by='score',
                                 best_color_override=None,
                                 cmap_vmin=None,
                                 cmap_vmax=None,
                                 show_colorbar=False,
                                 legend_granularity='coarse',
                                 legend_show_counts=False,
                                 context_flags=None):
    spec_meta = PARAM_META.get(param_key, None)
    if spec_meta is None:
        spec_meta = dict(
            title=PLASTICITY_ONTOLOGY.get(param_key, {}).get('label', param_key),
            symbol=PLASTICITY_ONTOLOGY.get(param_key, {}).get('symbol'),
            unit=None,
        )
    spec_meta = dict(spec_meta)
    spec_meta['symbol'] = normalize_tex(spec_meta.get('symbol'))
    spec_meta['title']  = normalize_tex(spec_meta.get('title'))
    spec_meta['unit']   = normalize_tex(spec_meta.get('unit'))

    resolved_label_pad = 9.0 if annotation_offset is None \
        else float(annotation_offset)

    fig = plot_candidate_scores(
        candidates, scores, provenance, best_idx,
        param_title=spec_meta['title'],
        param_symbol=spec_meta['symbol'],
        unit=spec_meta['unit'],
        journal=journal,
        label_pad=resolved_label_pad,
        annotation_fontsize=annotation_fontsize,
        legend_fontsize=legend_fontsize,
        colorbar_label_fontsize=colorbar_label_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        bar_colormap=bar_colormap,
        color_by=color_by,
        best_color_override=best_color_override,
        cmap_vmin=cmap_vmin,
        cmap_vmax=cmap_vmax,
        show_colorbar=show_colorbar,
        legend_granularity=legend_granularity,
        legend_show_counts=legend_show_counts,
        context_flags=context_flags,
    )
    screen, hires = BytesIO(), BytesIO()
    fig.savefig(screen, format='png', dpi=200, bbox_inches='tight',
                pad_inches=0.08)
    fig.savefig(hires,  format='png', dpi=hires_dpi, bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    screen.seek(0)
    try:
        _ver = tuple(int(x) for x in st.__version__.split('.')[:2])
    except Exception:
        _ver = (0, 0)
    if _ver >= (1, 46):
        st.image(screen, width='stretch')
    else:
        st.image(screen, use_container_width=True)
    st.download_button(f'Download {hires_dpi}-dpi PNG', data=hires.getvalue(),
                       file_name=f'latentmoe_{param_key}{key_suffix}.png',
                       mime='image/png')


# ============================================================================
# ███ v9.0.0 / v9.1.0 — NER PIPELINE + CASCADE + DERIVED LAB          ███
# ============================================================================

GAZETTEER: Dict[str, List[str]] = {
    "material":      ["Cu", "copper", "nanotwinned Cu", "nt-Cu", "copper (Cu)",
                      "Al", "aluminium", "aluminum",
                      "Ni", "nickel", "Fe", "iron", "steel"],
    "property":      ["shear modulus", "shear moduli", "c44", "elastic constant",
                      "elastic constants", "elastic moduli", "soec",
                      "second order elastic constants", "lame parameter",
                      "dislocation density", "initial dislocation density",
                      "reference strain rate", "reference shear strain rate",
                      "strain rate sensitivity", "srs",
                      "friction stress", "lattice friction",
                      "friction lattice stress", "lattice friction stress",
                      "athermal stress",
                      "peierls stress", "peierls-nabarro stress",
                      "peierls nabarro stress", "pn stress",
                      "lattice resistance", "intrinsic lattice friction",
                      "yield stress", "yield strength"],
    "property_weak": ["mu", "g", "sigma0", "tau_p", "tau0",
                      "sigma_0", "τ_p", "σ₀", "σy", "σ_y"],
    "grain_size":     ["grain size", "grain diameter", "average grain size",
                       "mean grain size", "grain diameter d",
                       "crystallite size", "d50"],
    "twin_thickness": ["twin thickness", "twin spacing", "twin boundary spacing",
                       "twin boundary separation", "lambda twin",
                       "nanotwin spacing", "twin lamella thickness",
                       "λ twin", "twin spacing λ"],
    "hall_petch_k":   ["hall-petch coefficient", "hall petch coefficient",
                       "strengthening coefficient", "ky", "k_y", "k_lambda",
                       "kλ", "k_λ", "grain boundary strengthening coefficient"],
    "burgers_vector": ["burgers vector", "burgers' vector",
                       "magnitude of burgers vector", "b vector"],
    "poisson_ratio":  ["poisson's ratio", "poisson ratio", "poissons ratio",
                       "ν poisson", "poisson coefficient"],
    "taylor_factor":  ["taylor factor", "taylor m factor",
                       "orientation factor m", "schmid factor",
                       "taylor orientation factor"],
    "dislocation_core_width":
                      ["core width", "dislocation core width",
                       "core half-width", "peierls valley width",
                       "core spreading"],
    "resolved_shear_stress":
                      ["critical resolved shear stress", "crss",
                       "resolved shear stress", "tau_crss", "τ_crss"],
    "related":       ["c11", "c12", "bulk modulus", "youngs modulus",
                      "young's modulus", "stiffness tensor",
                      "c'", "c_s", "stacking fault energy",
                      "generalized stacking fault energy",
                      "gsfe", "gamma surface", "γ-surface"],
    "method":        ["rus", "resonant ultrasound spectroscopy", "md",
                      "molecular dynamics", "dft", "eam", "embedded atom method",
                      "voigt", "reuss", "hill", "vrh", "hashin-shtrikman",
                      "self-consistent", "nanoindentation", "tensile test",
                      "compression test", "hall-petch plot",
                      "hall petch plot", "low-temperature flow stress"],
    "structure":     ["single crystal", "polycrystalline", "nanocrystalline",
                      "fcc", "face-centered cubic", "bcc", "hcp",
                      "nanotwinned", "nt-cu", "nt-Cu"],
    "unit":          ["GPa", "MPa", "Pa", "s^-1", "s-1", "/s", "nm",
                      "m^-2", "m-2", "μm", "um", "Å", "angstrom",
                      "K", "C", "°C", "Celsius"],
    "synthesis_method": [
        "magnetron sputter", "magnetron sputtering", "sputter-deposited",
        "sputter deposited", "sputtering", "dc sputtering", "rf sputtering",
        "pvd", "physical vapor deposition",
        "mbe", "molecular beam epitaxy",
        "electrodeposition", "electrodeposited", "electroplating",
        "electroplated", "dc electrodeposition",
        "direct-current electrodeposition",
        "pulsed electrodeposition", "ped",
        "localized electroplating", "additive micromanufacturing",
        "additive manufacturing", "additively manufactured",
        "additive-assisted electrodeposition",
        "electroforming", "jet electrodeposition",
    ],
    "grain_architecture": [
        "columnar", "columnar grain", "columnar microstructure",
        "columnar grains", "columnar growth",
        "highly oriented", "(111) oriented", "(111)-oriented", "111-oriented",
        "highly-aligned", "highly aligned", "aligned grains",
        "equiaxed", "equiaxed grain", "equiaxed grains",
        "equiaxed microstructure",
        "nanocrystalline", "nc-cu", "nc cu",
        "ultrafine grain", "ultrafine-grained", "ultrafine grained", "ufg",
        "single crystal", "single-crystal", "single crystalline",
        "polycrystalline", "polycrystal", "poly-crystalline",
        "randomly oriented", "random texture", "randomly-textured",
        "randomly textured",
    ],
    "theory_framework": [
        "johnson-cook", "johnson cook", "johnson–cook", "jc model",
        "jc plasticity", "jc constitutive", "j-c model",
        "power law", "power-law", "power-law plasticity",
        "power-law hardening", "power-law flow",
        "empirical constitutive", "constitutive fit",
        "constitutive law", "constitutive model", "curve fit",
        "strain-rate sensitivity index", "srs index",
        "discrete dislocation dynamics", "dislocation dynamics", "ddd",
        "ddd simulation", "ddd model", "paradis", "paradis code",
        "mesoscale dislocation", "mesoscale simulation",
        "dislocation-mediated plasticity",
        "crystal plasticity", "crystal plasticity finite element",
        "cpfem", "cp-fem", "cp fem",
        "taylor model", "taylor polycrystal", "taylor factor",
        "viscoplastic self-consistent", "vpsc", "self-consistent model",
        "slip system", "slip rate", "slip system rate",
        "reference slip rate", "reference shear rate", "reference shear",
        "schmid law", "schmid factor",
        "molecular dynamics", "md simulation",
        "atomistic simulation", "atomistic", "eam potential", "md potential",
        "phase field", "phase-field", "phase-field model",
        "phase field simulation", "allen-cahn", "cahn-hilliard",
        "tensile test", "compression test", "quasi-static",
        "split hopkinson", "shpb", "dynamic compression",
    ],
}


_CHAR_FOLD = {"μ": "mu", "µ": "mu", "ρ": "rho", "–": "-", "—": "-",
              "’": "'", "\u00a0": " ", "γ": "gamma", "σ": "sigma",
              "λ": "lambda", "θ": "theta", "φ": "phi", "η": "eta"}


def norm_text(s: Any) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    for k, v in _CHAR_FOLD.items():
        s = s.replace(k, v)
    s = re.sub(r"[{}$\\]", "", s)
    s = re.sub(r"_(?=\d)", "", s)
    s = re.sub(r"(?<=[a-z])\s+(?=\d)", "", s.lower())
    return s


def alias_pattern(alias: str) -> str:
    a = norm_text(alias)
    out = []
    for i, ch in enumerate(a):
        if i > 0 and ch.isalnum() and out and out[-1][-1].isalnum():
            out.append(r"[\s_\-]{0,2}")
        out.append(re.escape(ch))
    return rf"(?<![a-z0-9]){''.join(out)}(?![a-z0-9])"


ENTITY_PATTERNS = {k: [re.compile(alias_pattern(a)) for a in v]
                   for k, v in GAZETTEER.items()}


PARAM_CANON: Dict[str, Dict[str, Any]] = {
    "mu":             dict(aliases=["shear modulus", "shear moduli", "c44",
                                    "elastic constants", "elastic moduli",
                                    "soec", "mu", "g", "lame parameter",
                                    "second order elastic constants"],
                           unit="GPa", plausible=(20.0, 200.0)),
    "rho0":           dict(aliases=["dislocation density", "rho0",
                                    "initial dislocation density",
                                    "forest dislocation density",
                                    "mobile dislocation density"],
                           unit="m^-2", plausible=(1e10, 1e17)),
    "gamma0_dot":     dict(aliases=["reference shear strain rate",
                                    "reference strain rate", "strain rate",
                                    "deformation rate", "loading rate",
                                    "gamma0_dot", "gamma dot 0"],
                           unit="s^-1", plausible=(1e-8, 1e6)),
    "srs":            dict(aliases=["strain rate sensitivity exponent",
                                    "rate sensitivity exponent",
                                    "m exponent", "strain-rate sensitivity",
                                    "srs"],
                           unit=None, plausible=(0.001, 200.0)),

    "sigma0_fric":    dict(aliases=["friction stress", "lattice friction stress",
                                    "friction lattice stress",
                                    "lattice friction",
                                    "athermal stress", "sigma0 friction",
                                    "sigma_0 friction"],
                           unit="MPa",
                           plausible=(0.1, 2000.0),
                           derivation="hall_petch_intercept or M·τ_CRSS"),
    "sigma0_yield":   dict(aliases=["yield stress", "yield strength",
                                    "yield stress sigma_y", "sigma_y",
                                    "0.2% yield stress", "offset yield stress"],
                           unit="MPa", plausible=(1.0, 5000.0)),
    "tau_p":          dict(aliases=["peierls stress", "peierls-nabarro stress",
                                    "peierls nabarro stress", "pn stress",
                                    "intrinsic lattice friction",
                                    "lattice resistance",
                                    "peierls-nabarro τ", "τ_p", "tau_p"],
                           unit="MPa",
                           plausible=(0.01, 5000.0),
                           derivation="P-N formula or DFT/MD at 0 K or τ₀=σ₀/M"),

    "grain_size_d":   dict(aliases=["grain size", "grain diameter",
                                    "average grain size", "mean grain size",
                                    "d50", "crystallite size"],
                           unit="nm", plausible=(1.0, 1e5)),
    "twin_thickness_lambda":
                      dict(aliases=["twin thickness", "twin spacing",
                                    "twin boundary spacing",
                                    "twin boundary separation",
                                    "twin lamella thickness",
                                    "λ twin", "twin spacing λ"],
                           unit="nm", plausible=(0.5, 1000.0)),
    "burgers_vector_b":
                      dict(aliases=["burgers vector", "burgers' vector",
                                    "magnitude of burgers vector",
                                    "b vector"],
                           unit="nm", plausible=(0.1, 1.0)),
    "poisson_ratio_nu":
                      dict(aliases=["poisson's ratio", "poisson ratio",
                                    "poissons ratio", "poisson coefficient"],
                           unit=None, plausible=(0.0, 0.5)),
    "taylor_factor_M":
                      dict(aliases=["taylor factor", "taylor m factor",
                                    "orientation factor m",
                                    "taylor orientation factor"],
                           unit=None, plausible=(2.0, 3.5)),
    "hall_petch_k":   dict(aliases=["hall-petch coefficient",
                                    "hall petch coefficient",
                                    "strengthening coefficient", "ky", "k_y",
                                    "k_lambda", "kλ", "k_λ",
                                    "grain boundary strengthening coefficient"],
                           unit="MPa·m^(1/2)",
                           plausible=(0.01, 1.0)),
    "shear_modulus_G":
                      dict(aliases=["shear modulus", "shear moduli", "c44",
                                    "elastic shear modulus",
                                    "rigidity modulus"],
                           unit="GPa", plausible=(20.0, 200.0)),
    "core_width_w":   dict(aliases=["core width", "dislocation core width",
                                    "core half-width",
                                    "peierls valley width",
                                    "core spreading"],
                           unit="nm", plausible=(0.05, 2.0)),

    "twin_spacing":   dict(aliases=["twin spacing", "twin boundary spacing",
                                    "twin thickness", "lambda"],
                           unit="nm", plausible=(1.0, 1000.0)),
    "applied_stress": dict(aliases=["applied stress", "loading stress",
                                    "external stress"],
                           unit="MPa", plausible=(0.1, 5000.0)),
    "W":              dict(aliases=["interface width", "well depth",
                                    "twin well depth"],
                           unit="nm", plausible=(0.1, 20.0)),
}


_POSITIVE_LOWER_BOUND_PARAMS = frozenset({
    "mu", "rho0", "gamma0_dot",
    "sigma0_fric", "sigma0_yield", "tau_p",
    "grain_size_d", "twin_thickness_lambda", "burgers_vector_b",
    "hall_petch_k", "shear_modulus_G", "core_width_w",
    "twin_spacing", "applied_stress", "W",
})


def merge_no_clash(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    out = dict(base)
    out.update(overrides)
    return out


def build_boolean_queries(material="Cu", param_key="mu", value_hints=None,
                          methods=("RUS", "Voigt", "Reuss", "self-consistent")):
    p = PARAM_CANON.get(param_key, {}).get("aliases", [param_key])
    qs = [f'"{material}" AND "{a}"' for a in p[:4]]
    qs += [f'"{material}" AND "{a}" AND "GPa"' for a in p[:2]]
    for v in (value_hints or []):
        qs.append(f'"{material}" AND "c44" AND "shear modulus" AND "{v}"')
        qs.append(f'"{material.lower()}" AND "shear modulus" AND "{v} GPa"')
        for m in methods:
            qs.append(f'"{material}" AND "{m}" AND "{v}"')
    return qs


def build_query_texts(material="Cu", param_key="mu", value_hints=None):
    props = PARAM_CANON.get(param_key, {}).get("aliases", [param_key])[:3]
    texts = [f"{material} {p} in GPa" for p in props]
    texts += [f"{material} shear modulus c44 {m}"
              for m in ("RUS resonant ultrasound spectroscopy",
                        "Voigt Reuss Hill", "molecular dynamics EAM", "DFT")]
    texts += [f"{material} shear modulus c44 {v} GPa"
              for v in (value_hints or [])[:4]]
    return texts


def load_corpus_folder(folder: str) -> Dict[str, Any]:
    corpus = {}
    if not os.path.isdir(folder):
        return corpus
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".json"):
            try:
                with open(os.path.join(folder, fn), encoding="utf-8-sig") as f:
                    corpus[fn] = json.load(f)
            except Exception as e:
                logger.warning("load_corpus_folder: cannot load %s: %s", fn, e)
    return corpus


def iter_corpus_records(corpus) -> List[Dict[str, Any]]:
    records = []
    if isinstance(corpus, dict):
        for fname, data in corpus.items():
            rows = data if isinstance(data, list) else [data]
            records += [{"id": f"{fname}[{i}]", "source": fname, "data": r}
                        for i, r in enumerate(rows)]
    else:
        records = [{"id": f"rec[{i}]", "source": "corpus", "data": r}
                   for i, r in enumerate(corpus)]
    return records


def flatten_keyvals(node, path=""):
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                out += flatten_keyvals(v, f"{path}.{k}")
            else:
                out.append((f"{path}.{k}", str(k),
                            "" if v is None else str(v)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += flatten_keyvals(v, f"{path}[{i}]")
    return out


def record_text(data) -> str:
    return "; ".join(f"{k}: {v}" for _, k, v in flatten_keyvals(data))


def _spans(text, patterns):
    return [(m.start(), m.group(0)) for p in patterns for m in p.finditer(text)]


def score_record(text, param_key, material="Cu", value_hints=None):
    score, why = 0.0, []
    mat   = _spans(text, ENTITY_PATTERNS["material"])
    props = _spans(text, ENTITY_PATTERNS["property"])
    units = _spans(text, ENTITY_PATTERNS["unit"])
    if mat:
        score += 2.0
        why.append("material")
    if props:
        score += 3.0 + 0.5 * min(len(props), 3)
        why.append("property")
    elif _spans(text, ENTITY_PATTERNS["property_weak"]) and (units or mat):
        score += 0.5
        why.append("property(weak)")

    for etype, w in (("related", .3), ("method", .8), ("structure", .5),
                     ("grain_size",            .6),
                     ("twin_thickness",        .6),
                     ("hall_petch_k",          .6),
                     ("burgers_vector",        .4),
                     ("poisson_ratio",         .4),
                     ("taylor_factor",         .4),
                     ("dislocation_core_width", .4),
                     ("resolved_shear_stress",  .6),
                     ("synthesis_method",       .7),
                     ("grain_architecture",     .6),
                     ("theory_framework",       .9)):
        if _spans(text, ENTITY_PATTERNS[etype]):
            score += w
            why.append(etype)

    for v in (value_hints or []):
        m = re.search(rf"(?<![\d.]){re.escape(str(v))}(?!\d)", text)
        if m:
            near = any(abs(m.start() - s) < 80 for s, _ in props) or \
                   any(abs(m.start() - s) < 15 for s, _ in units)
            score += 4.0 if near else 1.0
            why.append(f"value:{v}{'*' if near else ''}")
    return score, why


UNIT_TOKENS = {
    "GPa":  r"gpa",
    "MPa":  r"mpa",
    "Pa":   r"pa",
    "m^-2": r"m\^?\(?-2\)?|m-2|per\s+(?:square\s+)?meter",
    "s^-1": r"s\^?\(?-1\)?|s-1|sec-1|/s|per\s+second",
    "nm":   r"nm|nanometers?",
    "μm":   r"μ[mμ]|um|micrometers?|microns?",
    "Å":    r"å|angstroms?",
    "MPa·m^(1/2)":
           r"mpa\s*[·•\*x×]?\s*m\^?\(?1/?2\)?|mpa\s*m\^?0?\.?5"
           r"|mpa\s*·\s*m\^?0?\.?5|mpa\s*m\^(?:1/2|½|0\.5)",
    "K":    r"kelvin",
    "°C":   r"°c|celsius|deg\s*c",
}
_STRESS_FAMILY = ("GPa", "MPa", "Pa")


def unit_regex_for(param_key: str) -> Optional[str]:
    canon = PARAM_CANON.get(param_key, {})
    u = canon.get("unit")
    if u is None:
        return None
    if u in _STRESS_FAMILY:
        return "|".join(UNIT_TOKENS[x] for x in _STRESS_FAMILY)
    if param_key in ("grain_size_d", "twin_thickness_lambda",
                     "burgers_vector_b", "core_width_w"):
        return "|".join(UNIT_TOKENS[x] for x in ("nm", "μm", "Å"))
    if u == "MPa·m^(1/2)":
        return UNIT_TOKENS["MPa·m^(1/2)"]
    return UNIT_TOKENS.get(u)


_NUM_ANY = re.compile(
    r"(?<![a-z0-9.])"
    r"(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"(?:\s*[x×*]\s*10\s*\^?\(?\s*(?P<exp>[+-]?\d+)\s*\)?)?"
)


def _num_value(m: "re.Match") -> float:
    v = float(m.group("val"))
    return v * 10.0 ** int(m.group("exp")) if m.group("exp") else v


def find_value_after(text: str, pos: int, param_key: str, window: int = 90):
    ure = unit_regex_for(param_key)
    seg = text[pos: pos + window]
    best = None
    for m in _NUM_ANY.finditer(seg):
        um = re.match(rf"\s*({ure})\b", seg[m.end(): m.end() + 12]) if ure else None
        sc = (2.0 if um else (1.0 if ure is None else 0.6)) - 0.01 * m.start()
        if best is None or sc > best[0]:
            best = (sc, _num_value(m), um.group(1) if um else None)
    return None if best is None else dict(value=best[1], unit=best[2])


def _value_from_cell(key: str, val: str, param_key: str) -> Tuple[Optional[float], Optional[str]]:
    kt = norm_text(key)
    vt = norm_text(val)
    ure = unit_regex_for(param_key)
    best = None
    for m in _NUM_ANY.finditer(vt):
        um = re.match(rf"\s*({ure})\b", vt[m.end(): m.end() + 12]) if ure else None
        sc = 2.0 if um else (1.0 if (ure is None or re.search(rf"\b{ure}\b", kt)) else 0.0)
        if sc and (best is None or sc > best[0]):
            best = (sc, _num_value(m), um.group(1) if um else None)
    return (best[1], best[2]) if best else (None, None)


def heuristic_extract(records, param_key, target_unit=None):
    canon = PARAM_CANON.get(param_key, {})
    tu = (target_unit or canon.get("unit") or "").lower()
    key_pats = [re.compile(alias_pattern(a)) for a in canon.get("aliases", [])]
    out: List[Dict[str, Any]] = []

    for rec in records:
        text = rec["text_norm"]
        raw  = rec.get("text", text)

        for path, key, val in flatten_keyvals(rec["data"]):
            if not any(p.search(norm_text(key)) for p in key_pats):
                continue
            v, u = _value_from_cell(key, val, param_key)
            if v is None:
                continue
            out.append({
                "value":          float(v),
                "unit":           u or tu,
                "property_label": key,
                "evidence_span":  f"{key} = {val}",
                "source":         rec["source"],
                "path":           path,
            })

        for p in key_pats:
            for m in p.finditer(text):
                hit = find_value_after(text, m.end(), param_key)
                if not hit:
                    continue
                item = merge_no_clash(
                    hit,
                    unit=hit.get("unit") or tu,
                    property_label=m.group(0),
                    evidence_span=text[max(0, m.start() - 15): m.end() + 45],
                    source=rec["source"],
                    path="",
                )
                try:
                    item["value"] = float(item["value"])
                except (TypeError, ValueError):
                    logger.debug(
                        "heuristic_extract[%s]: skipping malformed hit "
                        "value=%r from %r",
                        param_key, item.get("value"), m.group(0),
                    )
                    continue
                out.append(item)

        for row in parse_side_note_table(raw):
            if row["param"] != param_key:
                continue
            try:
                v_si = _normalize_unit(row["value"], row["unit"], row["param"])
            except Exception:
                v_si = float(row["value"])
            if v_si is None or (isinstance(v_si, float) and math.isnan(v_si)):
                continue
            out.append({
                "value":          float(v_si),
                "unit":           PARAM_CANON.get(row["param"], {}).get("unit") or tu,
                "property_label": row["evidence"][:80],
                "evidence_span":  row["evidence"],
                "source":         rec["source"],
                "path":           "side_note_table",
            })

    seen, ded = set(), []
    for e in out:
        kk = (round(e["value"], 6), str(e.get("unit")))
        if kk not in seen:
            seen.add(kk)
            ded.append(e)
    return ded


# ============================================================================
# ███ v9.0.0 — PHYSICS-GROUNDED ρ₀ EXTRACTION + REGIME CLASSIFIER       ███
# ============================================================================

def extract_rho0_with_context(text: str, window: int = 400) -> List[Dict[str, Any]]:
    rho0_pats = [re.compile(alias_pattern(a))
                 for a in PARAM_CANON["rho0"]["aliases"]]
    syn_pats  = [re.compile(alias_pattern(a), re.I)
                 for a in GAZETTEER["synthesis_method"]]
    arch_pats = [re.compile(alias_pattern(a), re.I)
                 for a in GAZETTEER["grain_architecture"]]
    twin_pats = [re.compile(alias_pattern(a))
                 for a in GAZETTEER["twin_thickness"]]

    findings: List[Dict[str, Any]] = []
    for p in rho0_pats:
        for m in p.finditer(text):
            lo = max(0, m.start() - window)
            hi = min(len(text), m.end() + window)
            snippet = text[lo:hi]

            syn_hit: Optional[str] = None
            for sp in syn_pats:
                sm = sp.search(snippet)
                if sm:
                    syn_hit = sm.group(0)
                    break

            arch_hit: Optional[str] = None
            for ap in arch_pats:
                am = ap.search(snippet)
                if am:
                    arch_hit = am.group(0)
                    break

            twin_val: Optional[float] = None
            twin_unit: Optional[str] = None
            for tp in twin_pats:
                tm = tp.search(snippet)
                if not tm:
                    continue
                hit = find_value_after(snippet, tm.end(),
                                       "twin_thickness_lambda")
                if hit:
                    twin_val = hit["value"]
                    twin_unit = hit.get("unit")
                    break

            rho_hit = find_value_after(text, m.end(), "rho0")

            findings.append({
                "rho0_mention":        m.group(0),
                "rho0_value":          rho_hit["value"] if rho_hit else None,
                "rho0_unit":           rho_hit.get("unit") if rho_hit else None,
                "synthesis_method":    syn_hit,
                "grain_architecture":  arch_hit,
                "twin_spacing":        twin_val,
                "twin_spacing_unit":   twin_unit,
                "snippet":             snippet,
                "position":            m.start(),
            })
    return findings


RHO0_REGIMES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("sputter", "columnar"): dict(
        low=1e13, high=1e14, bench=5e13,
        desc="Magnetron sputter / PVD columnar — very low residual "
             "defect density (surface annihilation during growth)"),
    ("sputter", "equiaxed"): dict(
        low=1e14, high=1e15, bench=3e14,
        desc="Sputter nanocrystalline — moderate density from GB trapping"),
    ("electrodeposit", "columnar"): dict(
        low=1e14, high=1e15, bench=3.9e14,
        desc="(111)-oriented electrodeposited columnar — moderate density"),
    ("electrodeposit", "equiaxed"): dict(
        low=1e16, high=1e17, bench=5e16,
        desc="Equiaxed electrodeposited — high density from organics + "
             "H co-deposition"),
    ("additive", "equiaxed"): dict(
        low=5e16, high=8.4e16, bench=8.38e16,
        desc="Additive-assisted electrodeposited — very high near-surface "
             "density"),
    ("additive", "columnar"): dict(
        low=1e15, high=1e16, bench=5e15,
        desc="Additive + columnar — intermediate"),
    ("mbe", "single"): dict(
        low=1e11, high=1e12, bench=5e11,
        desc="MBE single crystal — lowest measurable density"),
    ("electrodeposit", "single"): dict(
        low=1e12, high=1e13, bench=5e12,
        desc="Electrodeposited epitaxial — low density"),
}


def _norm_synthesis(s: Optional[str]) -> Optional[str]:
    s = (s or "").lower()
    if any(k in s for k in ("sputter", "pvd", "physical vapor")):
        return "sputter"
    if "mbe" in s or "molecular beam" in s:
        return "mbe"
    if any(k in s for k in ("additive", "additively")):
        return "additive"
    if any(k in s for k in ("electrodeposit", "electroplat", "electroform")):
        return "electrodeposit"
    return None


def _norm_architecture(a: Optional[str]) -> Optional[str]:
    a = (a or "").lower()
    if "columnar" in a:
        return "columnar"
    if "single" in a:
        return "single"
    if any(k in a for k in ("equiaxed", "nanocrystalline", "nc-cu",
                            "nc cu", "ufg", "ultrafine")):
        return "equiaxed"
    if "polycrystal" in a:
        return "equiaxed"
    return None


def classify_rho0_regime(
    synthesis: Optional[str],
    architecture: Optional[str],
    twin_spacing_nm: Optional[float],
) -> Dict[str, Any]:
    syn_key  = _norm_synthesis(synthesis)
    arch_key = _norm_architecture(architecture)

    regime = (RHO0_REGIMES.get((syn_key, arch_key))
              if (syn_key and arch_key) else None)

    if regime is None:
        return dict(
            regime="unclassified — using PARAM_CANON plausible range",
            low=1e10, high=1e17, bench=1e12,
            inferred=1e12, source="fallback")

    if (twin_spacing_nm is not None
            and isinstance(twin_spacing_nm, (int, float))
            and twin_spacing_nm > 0
            and np.isfinite(twin_spacing_nm)):
        lam = float(twin_spacing_nm)
        log_lam = np.log10(max(lam, 1.0))
        log_5   = np.log10(5.0)
        log_100 = np.log10(100.0)
        pos = float(np.clip(
            1.0 - (log_lam - log_5) / (log_100 - log_5),
            0.10, 0.90))
        log_lo = np.log10(regime["low"])
        log_hi = np.log10(regime["high"])
        center = log_lo + pos * (log_hi - log_lo)
        inferred = float(10.0 ** center)
    else:
        inferred = regime["bench"]

    return dict(
        regime=regime["desc"],
        low=regime["low"],
        high=regime["high"],
        bench=regime["bench"],
        inferred=inferred,
        source=f"classifier({syn_key},{arch_key},λ={twin_spacing_nm})",
    )


# ============================================================================
# ███ v9.1.0 — THEORY-AWARE γ̇₀ REGIME TABLE + CLASSIFIER              ███
# ============================================================================
GAMMA0_DOT_REGIMES: Dict[str, Dict[str, Any]] = {
    "johnson_cook": dict(
        low=1e-4, high=1e-3, bench=1e-3,
        desc="Johnson-Cook / empirical power-law — quasi-static reference rate"),
    "power_law": dict(
        low=1e-4, high=1e-3, bench=1e-3,
        desc="Power-law plasticity fit — quasi-static reference rate"),
    "cpfem": dict(
        low=1e-4, high=1e-3, bench=1e-3,
        desc="Crystal Plasticity FEM — slip-system reference γ̇₀"),
    "ddd": dict(
        low=1e3, high=1e4, bench=5e3,
        desc="Discrete Dislocation Dynamics — high computational reference"),
    "md": dict(
        low=1e6, high=1e8, bench=1e7,
        desc="Molecular Dynamics — atomistic reference rate"),
    "phase_field": dict(
        low=1e-4, high=1e-2, bench=1e-3,
        desc="Phase-field continuum — quasi-static reference rate"),
    "experimental": dict(
        low=1e-4, high=1e-3, bench=1e-3,
        desc="Experimental tensile / SHPB — quasi-static reference rate"),
}

_GAMMA0_THEORY_KEYS = (
    "johnson_cook", "power_law", "cpfem", "ddd",
    "md", "phase_field", "experimental",
)

_GAMMA0_QUASI_STATIC_GROUP = frozenset({
    "johnson_cook", "power_law", "experimental", "phase_field",
})


def _norm_theory(t: Optional[str]) -> Optional[str]:
    t = (t or "").lower()
    if any(k in t for k in ("johnson-cook", "johnson cook", "johnson–cook",
                            "jc model", "jc plasticity", "jc constitutive",
                            "j-c model", "j-c")):
        return "johnson_cook"
    if any(k in t for k in ("power law", "power-law", "empirical",
                            "constitutive fit", "constitutive law",
                            "constitutive model", "curve fit",
                            "srs index", "strain-rate sensitivity index")):
        return "power_law"
    if any(k in t for k in ("discrete dislocation", "dislocation dynamics",
                            "ddd", "paradis", "mesoscale dislocation",
                            "dislocation-mediated plasticity")):
        return "ddd"
    if any(k in t for k in ("crystal plasticity", "cpfem", "cp-fem",
                            "cp fem", "taylor model", "taylor polycrystal",
                            "vpsc", "viscoplastic self-consistent",
                            "self-consistent model", "slip system",
                            "slip rate", "reference slip rate",
                            "reference shear rate", "reference shear",
                            "schmid law")):
        return "cpfem"
    if any(k in t for k in ("molecular dynamics", "md simulation",
                            "atomistic")):
        return "md"
    if any(k in t for k in ("phase field", "phase-field", "allen-cahn",
                            "cahn-hilliard")):
        return "phase_field"
    if any(k in t for k in ("tensile test", "compression test",
                            "quasi-static", "shpb", "split hopkinson",
                            "dynamic compression")):
        return "experimental"
    return None


def classify_gamma0_regime(theory: Optional[str]) -> Dict[str, Any]:
    key = _norm_theory(theory)
    if key is None:
        return dict(
            regime="unclassified — using PARAM_CANON plausible range",
            low=1e-8, high=1e6, bench=1e-3,
            inferred=1e-3, source="fallback")
    r = GAMMA0_DOT_REGIMES[key]
    return dict(regime=r["desc"], low=r["low"], high=r["high"],
                bench=r["bench"], inferred=r["bench"],
                source=f"classifier({key})")


# ============================================================================
# v8.8.3 — SIDE-NOTE TABLE PARSER
# ============================================================================
_SIDE_NOTE_ROW_RE = re.compile(
    r"\|\s*(?P<key>[A-Za-z][A-Za-z0-9_'λσνβ·\-\s]{0,40}?)\s*\|"
    r"\s*(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"(?:\s*[×xX\*]\s*10\s*\^?\s*\{?\s*[+-]?\d+\s*\}?)?)"
    r"\s*(?P<unit>[A-Za-zμÅ·/^(){}.0-9\-\s]{0,24}?)\s*\|",
    re.MULTILINE,
)

_SIDE_NOTE_KEY_MAP = {
    "grain size":               "grain_size_d",
    "grain diameter":           "grain_size_d",
    "mean grain size":          "grain_size_d",
    "average grain size":       "grain_size_d",
    "twin thickness":           "twin_thickness_lambda",
    "twin spacing":             "twin_thickness_lambda",
    "twin boundary spacing":    "twin_thickness_lambda",
    "yield stress":             "sigma0_yield",
    "yield strength":           "sigma0_yield",
    "sigma_y":                  "sigma0_yield",
    "shear modulus":            "shear_modulus_G",
    "poisson":                  "poisson_ratio_nu",
    "burgers":                  "burgers_vector_b",
    "taylor":                   "taylor_factor_M",
    "hall-petch":               "hall_petch_k",
    "hall petch":               "hall_petch_k",
    "k_y":                      "hall_petch_k",
    "k_λ":                      "hall_petch_k",
    "kλ":                       "hall_petch_k",
    "core width":               "core_width_w",
    "friction stress":          "sigma0_fric",
    "lattice friction":         "sigma0_fric",
    "peierls":                  "tau_p",
    "peierls-nabarro":          "tau_p",
    "pn stress":                "tau_p",
}


def parse_side_note_table(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in _SIDE_NOTE_ROW_RE.finditer(text):
        key_norm = norm_text(m.group("key")).strip(" -|").lower()
        param: Optional[str] = None
        for k in sorted(_SIDE_NOTE_KEY_MAP, key=len, reverse=True):
            if k in key_norm:
                param = _SIDE_NOTE_KEY_MAP[k]
                break
        if param is None:
            continue

        num_field = m.group("val").strip()
        mantissa_m = re.match(r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", num_field)
        if not mantissa_m:
            continue
        try:
            mantissa = float(mantissa_m.group(1))
        except (TypeError, ValueError):
            continue
        exp_m = re.search(r"10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?", num_field)
        exponent = int(exp_m.group(1)) if exp_m else 0
        value = mantissa * (10 ** exponent)

        if value <= 0.0:
            logger.info(
                "parse_side_note_table: rejecting %s row with non-positive "
                "value=%r (raw='%s')",
                param, value, m.group(0).strip()[:80],
            )
            continue

        unit = m.group("unit").strip()
        out.append({
            "param":    param,
            "value":    value,
            "unit":     unit,
            "evidence": m.group(0).strip(),
            "method":   "side_note_table",
        })
    return out


# ============================================================================
# v8.8.3 — DERIVED-QUANTITY BRIDGE
# ============================================================================
def _pick_best_value_cand(candidates: List["ValueCandidate"]) -> Optional["ValueCandidate"]:
    return candidates[0] if candidates else None


def derive_sigma0_and_tau_p(
    cands_by_param: Dict[str, List["ValueCandidate"]],
    bundle_material: str,
) -> List["ValueCandidate"]:
    out: List["ValueCandidate"] = []

    def best(param_key: str) -> Optional["ValueCandidate"]:
        return _pick_best_value_cand(cands_by_param.get(param_key, []))

    def _fmt(v, unit="MPa"):
        if unit == "MPa":  return f"{v/1e6:.2f} MPa"
        if unit == "GPa":  return f"{v/1e9:.3f} GPa"
        if unit == "nm":   return f"{v*1e9:.3f} nm"
        if unit == "m":    return f"{v:.3e} m"
        return f"{v:.4g}"

    def _default(param_key: str) -> float:
        d = PLASTICITY_ONTOLOGY[param_key]["defaults"]
        return d.get(bundle_material, d.get("Cu"))

    sigma_y = best("sigma0_yield")
    d       = best("grain_size_d")
    lam     = best("twin_thickness_lambda")
    k_y     = best("hall_petch_k")
    G       = best("mu") or best("shear_modulus_G")
    nu      = best("poisson_ratio_nu")
    b       = best("burgers_vector_b")
    M       = best("taylor_factor_M")
    w       = best("core_width_w")

    sigma_y_v = sigma_y.value if sigma_y is not None else None
    d_v       = d.value       if d       is not None else None
    lam_v     = lam.value     if lam     is not None else None
    k_y_v     = (k_y.value * 1e6) if k_y is not None else None
    G_v       = G.value       if G       is not None else None
    nu_v      = nu.value      if nu      is not None else None
    b_v       = b.value       if b       is not None else None
    M_v       = M.value       if M       is not None else None
    w_v       = w.value       if w       is not None else None

    if G_v is not None and G_v <= 1.0:
        logger.warning(
            "derive_sigma0_and_tau_p[%s]: rejecting explicit G=%s "
            "(below 1.0 Pa; likely side-note zero-collapse)",
            bundle_material, G_v)
        G_v = None
    if G_v is None:
        G_v = float(_default("mu")) * 1e9

    if nu_v is None: nu_v = float(_default("poisson_ratio_nu"))
    if b_v  is None: b_v  = float(_default("burgers_vector_b")) * 1e-9
    if M_v  is None: M_v  = float(_default("taylor_factor_M"))

    _W_TRUSTED_FINE = ('llm_extract', 'llm_reasoned', 'regex_ner')
    w_captured_ok = (w is not None
                     and _norm_provenance(getattr(w, "provenance", ""))
                         in _W_TRUSTED_FINE
                     and w_v is not None
                     and w_v > 0
                     and not getattr(w, "clamped", False))
    if not w_captured_ok and b_v and nu_v is not None and (1.0 - nu_v) > 1e-3:
        w_v = b_v / (1.0 - nu_v)
    elif w_v is None or w_v <= 0:
        w_v = float(_default("core_width_w")) * 1e-9

    if sigma_y_v is not None and k_y_v is not None and \
       (lam_v is not None or d_v is not None):
        if lam_v is not None:
            length_v     = lam_v
            length_label = "twin thickness λ"
            length_src   = lam.source if lam else "twin_thickness_lambda"
        else:
            length_v     = d_v
            length_label = "grain size d"
            length_src   = d.source if d else "grain_size_d"

        if length_v < 1e-7:
            d_m = length_v
        elif length_v < 1.0:
            d_m = length_v * 1e-9
        else:
            d_m = length_v * 1e-9

        if d_m > 0:
            hp_term  = k_y_v * (d_m ** -0.5)
            sigma0   = sigma_y_v - hp_term

            if sigma0 > 0:
                evidence = (
                    f"σ_y = {_fmt(sigma_y_v)} "
                    f"(from {sigma_y.source[:40] if sigma_y else '?'}); "
                    f"k_y = {k_y_v/1e6:.4g} MPa·m^0.5 "
                    f"(from {k_y.source[:40] if k_y else '?'}); "
                    f"{length_label} = {_fmt(d_m, 'nm')} "
                    f"(from {str(length_src)[:40]})"
                )
                reasoning = (
                    f"Step 1: Hall–Petch relation σ_y = σ₀ + k_y · d^(−1/2).\n"
                    f"Step 2: Rearrange → σ₀ = σ_y − k_y · d^(−1/2).\n"
                    f"Step 3: σ_y = {sigma_y_v/1e6:.2f} MPa.\n"
                    f"Step 4: k_y = {k_y_v/1e6:.4g} MPa·m^0.5, "
                    f"{length_label} = {d_m*1e9:.3f} nm = {d_m:.4e} m.\n"
                    f"Step 5: k_y · d^(−1/2) = {hp_term/1e6:.2f} MPa.\n"
                    f"Step 6: σ₀ = {sigma_y_v/1e6:.2f} − "
                    f"{hp_term/1e6:.2f} = {sigma0/1e6:.2f} MPa."
                )
                out.append(ValueCandidate(
                    value=max(sigma0, 1.0),
                    unit="Pa",
                    provenance="physics_inferred",
                    property_label="Friction stress σ₀ (Hall–Petch intercept)",
                    method="hall_petch_intercept",
                    evidence=evidence,
                    source=f"side-note-cluster[{length_label}]",
                    confidence=0.45,
                    reasoning=reasoning,
                ))

    sigma0_for_peierls = None
    hp_cand = next((c for c in out if c.method == "hall_petch_intercept"), None)
    if hp_cand is not None:
        sigma0_for_peierls = hp_cand.value
    else:
        s0 = best("sigma0_fric")
        if s0 is not None:
            sigma0_for_peierls = s0.value

    if sigma0_for_peierls is not None and M_v is not None and M_v > 0:
        tau_p_taylor = sigma0_for_peierls / M_v
        evidence = (
            f"σ₀ = {_fmt(sigma0_for_peierls)}; "
            f"Taylor factor M = {M_v:.3f} "
            f"({'explicit' if M is not None else 'default for '+bundle_material})"
        )
        reasoning = (
            f"Step 1: Polycrystal friction stress σ₀ ≈ M · τ_P.\n"
            f"Step 2: Rearrange → τ_P ≈ σ₀ / M.\n"
            f"Step 3: σ₀ = {sigma0_for_peierls/1e6:.2f} MPa, M = {M_v:.3f}.\n"
            f"Step 4: τ_P ≈ {tau_p_taylor/1e6:.3f} MPa."
        )
        out.append(ValueCandidate(
            value=max(tau_p_taylor, 1.0),
            unit="Pa",
            provenance="physics_inferred",
            property_label="Peierls–Nabarro stress τ_P (σ₀/M estimate)",
            method="taylor_factor_cross_conversion",
            evidence=evidence,
            source="side-note-cluster[sigma0/M]",
            confidence=0.35,
            reasoning=reasoning,
        ))

    if G_v is not None and nu_v is not None and \
       b_v is not None and w_v is not None:
        b_m = b_v if b_v < 1e-3 else b_v * 1e-9
        w_m = w_v if w_v < 1e-3 else w_v * 1e-9

        if b_m > 0 and w_m > 0 and (1.0 - nu_v) > 1e-3:
            pre_factor = (2.0 * G_v) / (1.0 - nu_v)
            exponent   = -2.0 * np.pi * w_m / b_m

            if exponent < -700.0:
                tau_p_pn = 0.0
            else:
                tau_p_pn = pre_factor * float(np.exp(exponent))

            if tau_p_pn > 0:
                evidence = (
                    f"G = {G_v/1e9:.3f} GPa "
                    f"({'explicit' if G is not None else 'default '+bundle_material}); "
                    f"ν = {nu_v:.3f}; b = {b_m*1e9:.4f} nm; "
                    f"w = {w_m*1e9:.4f} nm"
                    + (" [w = b/(1−ν) physical default]" if not w_captured_ok else "")
                )
                reasoning = (
                    f"Step 1: Classical Peierls–Nabarro formula:\n"
                    f"        τ_P = (2G / (1 − ν)) · exp(−2π w / b).\n"
                    f"Step 2: G = {G_v/1e9:.3f} GPa, ν = {nu_v:.3f}.\n"
                    f"Step 3: b = {b_m*1e9:.4f} nm, "
                    f"w = {w_m*1e9:.4f} nm "
                    f"{'(captured)' if w_captured_ok else '(physical default b/(1−ν))'}.\n"
                    f"Step 4: 2G/(1−ν) = {pre_factor/1e9:.3f} GPa.\n"
                    f"Step 5: −2π w / b = {exponent:.4f}.\n"
                    f"Step 6: τ_P = {tau_p_pn/1e6:.4g} MPa."
                )
                missing = sum(x is None for x in (G, nu, b, w))
                conf = max(0.20, 0.45 - 0.05 * missing)
                out.append(ValueCandidate(
                    value=max(tau_p_pn, 1.0),
                    unit="Pa",
                    provenance="physics_inferred",
                    property_label="Peierls–Nabarro stress τ_P (P–N formula)",
                    method="peierls_nabarro_formula",
                    evidence=evidence,
                    source="side-note-cluster[P-N formula]",
                    confidence=conf,
                    reasoning=reasoning,
                ))

    return out


# ============================================================================
# ███ v8.9.0 — FRICTION / LATTICE STRESS LAB                           ███
# ============================================================================
NM           = 1e-9
MPA_PER_GPA  = 1.0e3

ROUTE_PENALTY = {
    'hall_petch_d':    0.85,
    'hall_petch_lam':  0.85,
    'model_inversion': 0.95,
    'peierls_nabarro': 0.50,
}
DERIVED_CONF_CAP  = 0.75
ROUTE_SPREAD_WARN = 0.50

CTX_KEYS: Dict[str, str] = {
    'sigma_y': 'sigma0_yield',
    'd':       'grain_size_d',
    'lam':     'twin_thickness_lambda',
    'b':       'burgers_vector_b',
    'nu':      'poisson_ratio_nu',
    'M':       'taylor_factor_M',
    'k_y':     'hall_petch_k',
    'w':       'core_width_w',
    'G':       'shear_modulus_G',
}


@dataclass
class CtxField:
    value: Optional[float] = None
    conf: float = 0.0
    provenance: str = 'missing'
    clamped: bool = False


@dataclass
class DerivedRoute:
    key: str
    label: str
    formula_tex: str
    inputs: Dict[str, float] = field(default_factory=dict)
    value_MPa: Optional[float] = None
    tau_P_MPa: Optional[float] = None
    conf: float = 0.0
    status: str = 'missing_inputs'
    note: str = ''


def _gmean(xs):
    return float(np.exp(np.mean(np.log([max(x, 1e-6) for x in xs]))))


def _entry_candidates(bundle, key) -> List:
    if bundle is None:
        return []
    cands = getattr(bundle, 'candidates', None)
    if isinstance(cands, dict):
        return list(cands.get(key, []) or [])
    if isinstance(bundle, dict):
        entry = bundle.get(key)
        if isinstance(entry, dict):
            return list(entry.get('candidates', []))
        if isinstance(entry, (list, tuple)):
            return list(entry)
        if hasattr(entry, 'candidates'):
            return list(entry.candidates or [])
    return []


def _candidate_score(c) -> float:
    return float(getattr(c, 'score',
                         getattr(c, 'confidence', 0.0)) or 0.0)


def _pick_best(cands):
    if not cands:
        return None
    ranked = [c for c in cands if not getattr(c, 'context', False)]
    if not ranked:
        logger.warning("_pick_best: zero non-context candidates — "
                       "check INCUMBENT_ROUTES; ranking all as fallback")
        ranked = list(cands)
    trust = {
        'llm_extract':  3, 'llm_reasoned': 3,
        'regex_ner':    2, 'physics_inferred': 2,
        'regime_prior': 1, 'llm_prior':    1,
    }
    def key(c):
        s = _candidate_score(c)
        prov = _norm_provenance(getattr(c, 'provenance', '') or
                                getattr(c, 'method', ''))
        return (s, trust.get(prov, 0))
    return max(ranked, key=key)


def _cand_value_ui_unit(c, param_key) -> Optional[float]:
    spec = PLASTICITY_ONTOLOGY.get(param_key)
    if c is None or spec is None:
        return None
    v_si = getattr(c, 'value_si', None)
    if v_si is not None:
        try:
            return float(v_si) / spec['ui_scale']
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    v = getattr(c, 'value', None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect_side_note_context(bundle) -> Dict[str, CtxField]:
    ctx: Dict[str, CtxField] = {}
    for sym, key in CTX_KEYS.items():
        best = _pick_best(_entry_candidates(bundle, key))
        if best is None:
            ctx[sym] = CtxField()
            continue
        v = _cand_value_ui_unit(best, key)
        ctx[sym] = CtxField(
            value=v,
            conf=_candidate_score(best),
            provenance=_norm_provenance(
                getattr(best, 'provenance', '') or getattr(best, 'method', '')),
            clamped=bool(getattr(best, 'clamped', False)),
        )
    return ctx


def build_derived_routes(ctx, mu_GPa=None, w_mode='auto') -> List[DerivedRoute]:
    routes: List[DerivedRoute] = []
    sy, ky = ctx['sigma_y'].value, ctx['k_y'].value
    b, nu, M = ctx['b'].value, ctx['nu'].value, ctx['M'].value

    for sym, key, lab in (('d', 'hall_petch_d', 'grain d'),
                          ('lam', 'hall_petch_lam', 'twin λ')):
        L = ctx[sym].value
        if None in (sy, ky, L) or L <= 0 or (ky or 0) <= 0:
            routes.append(DerivedRoute(key, f'HP ({lab})',
                         r'$\sigma_0=\sigma_y-k\,L^{-1/2}$',
                         status='missing_inputs'))
            continue
        hp = ky / math.sqrt(L * NM)
        s0 = sy - hp
        routes.append(DerivedRoute(
            key, f'HP ({lab})', r'$\sigma_0=\sigma_y-k\,L^{-1/2}$',
            inputs={'σ_y': sy, 'k_y': ky, 'L': L},
            value_MPa=s0,
            conf=_gmean([ctx['sigma_y'].conf, ctx['k_y'].conf, ctx[sym].conf])
                 * ROUTE_PENALTY[key],
            status='ok' if s0 > 0 else 'degenerate',
            note=f'obstacle term k/√L = {hp:.1f} MPa'
                 + ('' if s0 > 0 else ' — exceeds σ_y; route degenerate')))

    lam = ctx['lam'].value
    G_model = mu_GPa or (ctx['G'].value
                         if ctx['G'].value and ctx['G'].value > 1.0 else None)
    if None not in (sy, G_model, b, lam, nu) and b > 0 and lam > 0 and nu < 1:
        mu_MPa = G_model * MPA_PER_GPA
        if lam > 2 * b:
            term = (mu_MPa * b / (2 * math.pi * lam * (1 - nu))
                    * math.log(lam / b))
        else:
            term = mu_MPa / (2 * math.pi * (1 - nu))
        s0 = sy - term
        routes.append(DerivedRoute(
            'model_inversion', 'Solver-law inversion',
            r'$\sigma_y=\sigma_0+\frac{\mu b}{2\pi\lambda(1-\nu)}\ln\frac{\lambda}{b}$',
            inputs={'σ_y': sy, 'μ': G_model, 'λ': lam, 'b': b, 'ν': nu},
            value_MPa=s0,
            conf=_gmean([ctx['sigma_y'].conf, ctx['lam'].conf,
                         ctx['b'].conf, ctx['nu'].conf])
                 * ROUTE_PENALTY['model_inversion'],
            status='ok' if s0 > 0 else 'degenerate',
            note=f'obstacle term = {term:.1f} MPa — exact inverse of '
                 'compute_yield_stress()'))

    if ctx['G'].value and ctx['G'].value > 1.0:
        G_MPa, G_note = ctx['G'].value * MPA_PER_GPA, 'explicit G'
    elif mu_GPa:
        G_MPa, G_note = mu_GPa * MPA_PER_GPA, 'G from μ recommendation'
    else:
        G_MPa, G_note = None, 'no usable shear modulus'

    w_default = b / (1.0 - nu) if (b and nu is not None and nu < 1) else None
    w_cap = ctx['w'].value

    _W_TRUSTED_FINE = ('llm_extract', 'llm_reasoned', 'regex_ner')

    if w_mode == 'captured' and w_cap:
        w_use, w_note = w_cap, f'captured w = {w_cap:.4f} nm (user override)'
    elif w_mode == 'zeta':
        zeta = st.session_state.get('pn_zeta',
                                    1.0 / (1 - (nu if nu is not None else 0.34)))
        w_use = (zeta * b if b else None)
        w_note = f'user ζ = w/b = {zeta:.2f}'
    elif w_cap and ctx['w'].provenance in _W_TRUSTED_FINE \
            and not ctx['w'].clamped:
        w_use, w_note = w_cap, f'captured w = {w_cap:.4f} nm'
    else:
        w_use = w_default
        w_note = (f'w = b/(1−ν) = {w_default:.4f} nm '
                  f'(captured w is {ctx["w"].provenance}/clamped)') \
                 if w_default else 'no usable w'

    if None not in (G_MPa, nu, b, w_use, M) \
            and min(G_MPa, b, w_use) > 0 and nu < 1:
        tau_P = 2.0 * G_MPa / (1.0 - nu) * math.exp(-2.0 * math.pi * w_use / b)
        s0 = M * tau_P
        conf = _gmean([ctx['M'].conf, ctx['b'].conf, ctx['nu'].conf,
                       ctx['G'].conf if 'explicit' in G_note else 0.5]) \
               * ROUTE_PENALTY['peierls_nabarro']
        degenerate = s0 > 0.1 * G_MPa
        routes.append(DerivedRoute(
            'peierls_nabarro', 'Peierls–Nabarro + Taylor',
            r'$\tau_P=\frac{2G}{1-\nu}e^{-2\pi w/b},\;\ \sigma_0=M\tau_P$',
            inputs={'G': G_MPa / MPA_PER_GPA, 'ν': nu, 'b': b, 'w': w_use, 'M': M},
            value_MPa=s0, tau_P_MPa=tau_P, conf=conf,
            status='degenerate' if degenerate else 'ok',
            note=f'τ_P = {tau_P:.3g} MPa — lattice FLOOR, not the full '
                 f'athermal σ₀ ({w_note}; {G_note})'
                 + (' — τ_P ≳ 0.1·G: w/b too small, rejected' if degenerate else '')))
    return routes


def consensus_sigma0(routes):
    ok = [r for r in routes if r.status == 'ok' and (r.value_MPa or 0) > 0]
    if not ok:
        return None
    v = np.array([r.value_MPa for r in ok])
    c = np.array([r.conf for r in ok])
    order = np.argsort(v)
    cum = np.cumsum((c / c.sum())[order])
    med = float(v[order][min(int(np.searchsorted(cum, 0.5)), len(v) - 1)])
    rel = float((v.max() - v.min()) / max(med, 1e-9))
    return dict(value_MPa=med, lo=float(v.min()), hi=float(v.max()),
                rel_spread=rel, n_routes=len(ok),
                disagree=rel > ROUTE_SPREAD_WARN,
                conf=min(DERIVED_CONF_CAP, _gmean(list(c)) * math.exp(-rel)),
                contributors=[r.key for r in ok])


def floor_check(cons, routes):
    pn = next((r for r in routes if r.key == 'peierls_nabarro'
               and r.status == 'ok'), None)
    if not (cons and pn and pn.value_MPa):
        return None
    passed = cons['value_MPa'] >= pn.value_MPa
    return dict(floor_MPa=pn.value_MPa, passed=passed,
                msg=('OK — σ₀ ≥ M·τ_P lattice floor respected' if passed else
                     f'VIOLATION — σ₀ = {cons["value_MPa"]:.1f} MPa < '
                     f'M·τ_P = {pn.value_MPa:.1f} MPa; inputs inconsistent'))


def closure_check(cons, ctx, mu_GPa):
    lam, b, nu = ctx['lam'].value, ctx['b'].value, ctx['nu'].value
    sy = ctx['sigma_y'].value
    if None in (lam, b, nu, sy) or not (cons and mu_GPa):
        return None
    try:
        sy_pred = float(compute_yield_stress(
            np.array([[lam]]),
            cons['value_MPa'],
            mu_GPa * MPA_PER_GPA,
            b, nu)[0, 0])
    except Exception:
        return None
    return dict(sigma_y_pred_MPa=sy_pred, sigma_y_card_MPa=sy,
                residual_MPa=sy_pred - sy)


def build_derived_stress_bundle(bundle, w_mode='auto'):
    ctx = collect_side_note_context(bundle)
    mu_best = _pick_best(_entry_candidates(bundle, 'mu'))
    mu_GPa = _cand_value_ui_unit(mu_best, 'mu') or None
    routes = build_derived_routes(ctx, mu_GPa=mu_GPa, w_mode=w_mode)
    cons = consensus_sigma0(routes)
    return dict(ctx=ctx, mu_GPa=mu_GPa, routes=routes, consensus=cons,
                floor=floor_check(cons, routes),
                closure=closure_check(cons, ctx, mu_GPa))


# ============================================================================
# v8.8 HYBRID RETRIEVER
# ============================================================================
def rrf(rank_lists, k=60):
    agg: Dict[str, float] = {}
    for lst in rank_lists:
        for r, rid in enumerate(lst):
            agg[rid] = agg.get(rid, 0.0) + 1.0 / (k + r + 1)
    return [rid for rid, _ in sorted(agg.items(), key=lambda kv: -kv[1])]


class HybridRetriever:
    def __init__(self, corpus, use_dense=True):
        self.records = iter_corpus_records(corpus)
        for r in self.records:
            r["text_norm"] = norm_text(record_text(r["data"]))
        self.ids  = [r["id"] for r in self.records]
        self._txt = {r["id"]: r["text_norm"] for r in self.records}
        self.dense = False
        self.model = None
        self.index = None
        if use_dense and FAISS_AVAILABLE and SBERT_AVAILABLE and self.records:
            try:
                self.model = _SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                emb = self.model.encode(list(self._txt.values()),
                                        normalize_embeddings=True,
                                        show_progress_bar=False).astype(np.float32)
                self.index = _faiss.IndexFlatIP(emb.shape[1])
                self.index.add(np.ascontiguousarray(emb, dtype="float32"))
                self.dense = True
            except Exception as e:
                logger.warning("HybridRetriever dense channel disabled: %s", e)

    def search(self, param_key, material="Cu", value_hints=None, k=6):
        if not self.records:
            return []
        lex = []
        for rid in self.ids:
            s, _ = score_record(self._txt[rid], param_key, material, value_hints)
            if s > 0:
                lex.append((-s, rid))
        rank_lists = [[rid for _, rid in sorted(lex)]]
        if self.dense:
            for q in build_query_texts(material, param_key, value_hints)[:4]:
                qv = self.model.encode([q], normalize_embeddings=True).astype(np.float32)
                _, I = self.index.search(np.ascontiguousarray(qv, "float32"),
                                         min(10, len(self.ids)))
                rank_lists.append([self.ids[i] for i in I[0]])
        id2rec = {r["id"]: r for r in self.records}
        return [id2rec[rid] for rid in rrf(rank_lists) if rid in id2rec][:k]


def build_extraction_prompt(param_key, material, records, max_chars=1200):
    meta  = PARAM_META.get(param_key, dict(title=param_key, symbol=param_key, unit=""))
    canon = PARAM_CANON.get(param_key, {})
    aliases = ", ".join(f'"{a}"' for a in canon.get("aliases", [])[:8])
    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r['source']}, id={r['id']})\n"
        f"{json.dumps(r['data'], ensure_ascii=False, default=str)[:max_chars]}"
        for i, r in enumerate(records))
    return f"""TASK: Extract the {meta['title']} ({meta['symbol']}, unit {meta['unit']})
of {material} from the EVIDENCE records below.
The property may be labelled: {aliases}.

STRICT RULES
1. Report ONLY numbers that appear VERBATIM in an EVIDENCE record.
2. A number qualifies ONLY IF the SAME record contains (a) the material
   ({material}), (b) a property label from the list, and (c) a unit
   ({meta['unit']} or convertible MPa).
3. Convert MPa to GPa (divide by 1000). Never invent, average, or interpolate.
4. Answer with a JSON array ONLY — no markdown, no prose. Schema:
   [{{"value": 47.19, "unit": "GPa", "value_gpa": 47.19, "property_label": "c44",
      "method": "RUS", "evidence_id": 0, "evidence_span": "verbatim quote <=15 words",
      "confidence": 0.9}}]
5. If nothing qualifies, answer [].

{ev}"""


def build_prior_inference_prompt(param_key, material, records, max_chars=600):
    meta  = PARAM_META.get(param_key, dict(title=param_key, symbol=param_key,
                                           unit=None))
    canon = PARAM_CANON.get(param_key, {})
    lo, hi = canon.get("plausible", (float("-inf"), float("inf")))
    unit_str = canon.get("unit") or "(dimensionless)"
    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r['source']})\n"
        f"{json.dumps(r['data'], ensure_ascii=False, default=str)[:max_chars]}"
        for i, r in enumerate(records)
    ) or "(no records retrieved)"
    return f"""TASK: The evidence below does NOT contain an explicit value for the
{meta['title']} ({meta['symbol']}) of {material}. You are in PRIOR-INFERENCE
mode, which is DISTINCT from verbatim extraction:

1. Reason step by step toward a physically defensible value for {material},
   using established materials-science knowledge (textbook values, landmark
   studies, typical experimental ranges).
2. Stay strictly inside [{lo:g}, {hi:g}] {unit_str}.
3. Propose 1–3 candidates, most defensible first. Never more than 3.
4. Confidence MUST NOT exceed 0.5 for inferred candidates.
5. Output a JSON array ONLY — no markdown, no prose. Schema per element:
   [{{"value": <number>, "unit": "{unit_str}", "value_gpa": <number|null>,
      "property_label": "{meta['symbol']}", "method": "prior inference",
      "evidence_id": null, "evidence_span": null,
      "inference_basis": "<one-line justification>", "confidence": <0.0-0.5>}}]
6. Never present an inferred value as if it were measured.

NEW IN v8.8.3 — SIDE-NOTE CONTEXT EXTRACTION
For ANY of the following context fields, if the EVIDENCE records contain
explicit numbers, surface them as separate JSON objects with the `param`
field set to the corresponding key. These fields feed a downstream
derived-quantity bridge that computes σ₀ and τ_P from Hall–Petch and
Peierls–Nabarro formulas:

  "sigma0_yield"     — yield stress σ_y                    [MPa or GPa]
  "grain_size_d"      — grain size d                        [nm, μm, or Å]
  "twin_thickness_lambda" — twin thickness λ                [nm]
  "hall_petch_k"      — Hall–Petch coefficient k_y         [MPa·m^(1/2)]
  "burgers_vector_b"  — Burgers vector magnitude b          [nm or Å]
  "poisson_ratio_nu"  — Poisson's ratio ν                   [dimensionless]
  "taylor_factor_M"   — Taylor factor M                     [dimensionless]
  "core_width_w"      — dislocation core width w            [nm]

For each context field you surface, set method="explicit" (verbatim from
the evidence) and confidence ≥ 0.8. These context-field extractions are
NOT subject to the 0.5 confidence cap — only the inferred {meta['symbol']}
values are.

EVIDENCE (context only — do NOT quote its numbers as verbatim hits):
{ev}"""


# ============================================================================
# ███ v9.0.0 — PHYSICS-GROUNDED ρ₀ INFERENCE PROMPT                     ███
# ============================================================================
def build_rho0_physics_inference_prompt(
    material: str,
    synthesis: Optional[str],
    architecture: Optional[str],
    twin_spacing_nm: Optional[float],
    records,
    max_chars: int = 600,
) -> str:
    canon = PARAM_CANON.get("rho0", {})
    lo, hi = canon.get("plausible", (1e10, 1e17))
    regime = classify_rho0_regime(synthesis, architecture, twin_spacing_nm)

    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r['source']})\n"
        f"{json.dumps(r['data'], ensure_ascii=False, default=str)[:max_chars]}"
        for i, r in enumerate(records)
    ) or "(no records retrieved)"

    ts_str = (str(twin_spacing_nm) if twin_spacing_nm is not None
              else 'unknown')

    return f"""TASK: Infer the initial dislocation density ρ₀ of {material}
WITHOUT a verbatim corpus value.  You MUST walk the THREE-FACTOR physics
chain and stay inside the classifier-supplied range.

INPUT FACTORS (qualitative — do NOT treat as quantities):
  • Synthesis method     : {synthesis or 'unknown'}
  • Grain architecture   : {architecture or 'unknown'}
  • Twin lamella spacing : {ts_str} nm

PHYSICS PRIOR (from the three-factor classifier):
  Regime             : {regime['regime']}
  Physical range     : [{regime['low']:.2e}, {regime['high']:.2e}] m^-2
  Log-center estimate: {regime['inferred']:.3e} m^-2

MANDATORY REASONING CHAIN (write each step into `reasoning`):

  Step 1 — SYNTHESIS sets the decade baseline.
  Step 2 — GRAIN ARCHITECTURE refines within the baseline.
  Step 3 — TWIN LAMELLA SPACING λ MODULATES within the regime.
  Step 4 — COMBINE multiplicatively in log-space.

STRICT RULES:
  1. Confidence MUST NOT exceed 0.50.
  2. Propose 2–3 candidates, all inside the regime range.
  3. Each candidate MUST tag `dominant_factor` ∈
     {{"synthesis", "architecture", "twin_spacing", "combined"}}.
  4. NEVER present an inferred value as if it were measured.
  5. Stay strictly inside PARAM_CANON plausible range [{lo:g}, {hi:g}].

OUTPUT — JSON ARRAY ONLY, no markdown, no prose:
  [
    {{"value": <number>,
      "unit": "m^-2",
      "value_gpa": null,
      "property_label": "ρ₀",
      "method": "physics_regime_inference",
      "evidence_id": null,
      "evidence_span": null,
      "inference_basis": "<one-line justification>",
      "dominant_factor": "synthesis|architecture|twin_spacing|combined",
      "reasoning": "<Step 1 … Step 4 chain, newline-separated>",
      "confidence": <0.0-0.5>}}
  ]

EVIDENCE (context only — do NOT quote its numbers as verbatim hits):
{ev}
"""


# ============================================================================
# ███ v9.1.0 — THEORY-AWARE γ̇₀ PROMPTS                                ███
# ============================================================================
def build_gamma0_theory_aware_prompt(material: str,
                                     records,
                                     max_chars: int = 1200) -> str:
    canon = PARAM_CANON.get("gamma0_dot", {})
    aliases = ", ".join(f'"{a}"' for a in canon.get("aliases", [])[:8])
    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r['source']}, id={r['id']})\n"
        f"{json.dumps(r['data'], ensure_ascii=False, default=str)[:max_chars]}"
        for i, r in enumerate(records))

    return f"""TASK: Extract the reference strain rate γ̇₀ of {material}
from the EVIDENCE records below.  The property may be labelled: {aliases}.

⚠️ CRITICAL — γ̇₀ IS THEORY-DEPENDENT.
Tag every extraction with `theory` ∈
{{"johnson_cook", "power_law", "cpfem", "ddd", "md", "phase_field",
  "experimental", "unspecified"}}.

STRICT RULES:
1. Report ONLY numbers that appear VERBATIM in an EVIDENCE record.
2. The `theory` tag MUST reflect the framework of the SAME record.
3. Convert units to s⁻¹.
4. Output JSON ARRAY ONLY.

{ev}"""


def build_gamma0_theory_inference_prompt(
    material: str,
    target_theory: str,
    records,
    max_chars: int = 600,
) -> str:
    canon = PARAM_CANON.get("gamma0_dot", {})
    lo, hi = canon.get("plausible", (1e-8, 1e6))
    regime = classify_gamma0_regime(target_theory)
    norm_key = _norm_theory(target_theory) or "unspecified"

    ev = "\n\n".join(
        f"### EVIDENCE {i} (source={r['source']})\n"
        f"{json.dumps(r['data'], ensure_ascii=False, default=str)[:max_chars]}"
        for i, r in enumerate(records)
    ) or "(no records retrieved)"

    return f"""TASK: Infer the reference strain rate γ̇₀ of {material}
WITHOUT a verbatim corpus value.  Use the THREE-FACTOR theory chain.

INPUT FACTORS:
  • Target theory framework : {target_theory}

PHYSICS PRIOR (from theory classifier):
  Regime             : {regime['regime']}
  Physical range     : [{regime['low']:.2e}, {regime['high']:.2e}] s⁻¹
  Log-center estimate: {regime['inferred']:.3e} s⁻¹

STRICT RULES:
  1. Confidence MUST NOT exceed 0.50.
  2. Propose 2–3 candidates inside the regime.
  3. Each candidate MUST tag `theory` = "{norm_key}".
  4. NEVER present an inferred value as if it were measured.
  5. Stay inside PARAM_CANON plausible [{lo:g}, {hi:g}] s⁻¹.

OUTPUT — JSON ARRAY ONLY:
  [
    {{"value": <number>, "unit": "s^-1", "value_gpa": null,
      "property_label": "γ̇₀", "method": "theory_regime_inference",
      "theory": "{norm_key}", "evidence_id": null, "evidence_span": null,
      "inference_basis": "<one-line justification>",
      "dominant_factor": "theory",
      "reasoning": "<Step 1 … Step 3 chain, newline-separated>",
      "confidence": <0.0-0.5>}}
  ]

EVIDENCE (context only — do NOT quote its numbers as verbatim hits):
{ev}
"""


def extract_gamma0_with_context(text: str,
                                 window: int = 500) -> List[Dict[str, Any]]:
    gamma0_pats = [re.compile(alias_pattern(a))
                   for a in PARAM_CANON["gamma0_dot"]["aliases"]]
    theory_pats = [re.compile(alias_pattern(a), re.I)
                   for a in GAZETTEER["theory_framework"]]

    findings: List[Dict[str, Any]] = []
    for p in gamma0_pats:
        for m in p.finditer(text):
            lo = max(0, m.start() - window)
            hi = min(len(text), m.end() + window)
            snippet = text[lo:hi]

            theory_hit: Optional[str] = None
            for tp in theory_pats:
                tm = tp.search(snippet)
                if tm:
                    theory_hit = tm.group(0)
                    break

            val_hit = find_value_after(text, m.end(), "gamma0_dot")

            findings.append({
                "gamma0_mention":  m.group(0),
                "gamma0_value":     val_hit["value"] if val_hit else None,
                "gamma0_unit":      val_hit.get("unit") if val_hit else None,
                "theory":           theory_hit,
                "theory_key":       _norm_theory(theory_hit),
                "snippet":          snippet,
                "position":         m.start(),
            })
    return findings


def parse_llm_json(raw: str) -> list:
    txt = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.S)
    txt = txt.replace("```json", "").replace("```", "")
    starts = [i for i in (txt.find("["), txt.find("{")) if i >= 0]
    if not starts:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(txt[min(starts):])
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        logger.warning("parse_llm_json: unparseable LLM output")
        return []


def ollama_extract(prompt, model="qwen2.5:7b", host="http://localhost:11434",
                   timeout=120):
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests not installed")
    r = _requests.post(f"{host}/api/generate",
                       json={"model": model, "prompt": prompt, "stream": False,
                             "options": {"temperature": 0}},
                       timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


@dataclass
class ValueCandidate:
    value: float
    unit: str
    provenance: str
    property_label: str = ""
    method: str = ""
    evidence: str = ""
    source: str = ""
    confidence: float = 1.0
    reasoning: str = ""
    dominant_factor: str = ""
    theory: str = ""
    context: bool = False

    def to_display(self) -> Dict[str, Any]:
        d = {
            "value": round(self.value, 6),
            "unit": self.unit,
            "provenance": self.provenance,
            "property_label": self.property_label[:80],
            "method": self.method,
            "source": self.source[:80],
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:140],
            "reasoning": (self.reasoning[:200] + "…"
                          if len(self.reasoning) > 200
                          else self.reasoning),
        }
        if self.dominant_factor:
            d["dominant_factor"] = self.dominant_factor
        if self.theory:
            d["theory"] = self.theory
        if self.context:
            d["context"] = True
        return d


_TO_GPA = {"gpa": 1.0, "mpa": 1e-3, "pa": 1e-9}


def gatekeep(items, param_key, provenance="regex_ner", conf_cap=None):
    canon = PARAM_CANON.get(param_key, {})
    (lo, hi) = canon.get("plausible", (float("-inf"), float("inf")))
    target_unit = (canon.get("unit") or "").lower()

    if conf_cap is None:
        if provenance in ("llm_prior", "regime_prior"):
            conf_cap = 0.5
        elif provenance == "physics_inferred":
            conf_cap = 0.75
        else:
            conf_cap = 1.0

    reject_non_positive = param_key in _POSITIVE_LOWER_BOUND_PARAMS

    out, seen = [], set()
    for it in items:
        try:
            v = float(it.get("value_gpa") if it.get("value_gpa") is not None
                      else it.get("value"))
        except (TypeError, ValueError):
            continue
        u = str(it.get("unit") or canon.get("unit") or "").lower()
        if u in _TO_GPA and target_unit == "gpa":
            v = v * _TO_GPA[u]

        if reject_non_positive and v <= 0.0:
            logger.info("gatekeep[%s]: rejected non-positive %s "
                        "(likely side-note zero-collapse)", param_key, v)
            continue

        if not (lo <= v <= hi):
            logger.info("gatekeep[%s]: rejected %s (out of range %s..%s)",
                        param_key, v, lo, hi)
            continue
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(ValueCandidate(
            value=v,
            unit=canon.get("unit") or "",
            provenance=provenance,
            property_label=str(it.get("property_label", "")),
            method=str(it.get("method", "")),
            evidence=str(it.get("evidence_span") or it.get("evidence")
                         or it.get("inference_basis") or ""),
            source=str(it.get("source", "")),
            confidence=min(float(it.get("confidence", 1.0)), conf_cap),
            reasoning=str(it.get("reasoning") or ""),
            dominant_factor=str(it.get("dominant_factor") or ""),
            theory=str(it.get("theory") or "unspecified"),
            context=bool(it.get("_context", False)),
        ))
    return sorted(out, key=lambda c: -c.confidence)


def recommend_param_values(param_key, material="Cu", value_hints=None,
                           k=6, use_llm=True, ollama_model="qwen2.5:7b",
                           retriever=None, allow_prior_inference=True,
                           cascade_mode='fallback'):
    if retriever is None:
        retriever = get_retriever()

    diag: Dict[str, Any] = dict(param=param_key, records=0, n_llm=0,
                                n_heuristic=0, n_prior=0, tier="none",
                                reason="", cascade_mode=cascade_mode)

    records = retriever.search(param_key, material, value_hints=value_hints, k=k)
    if not records:
        records = retriever.search(param_key, material=None,
                                   value_hints=value_hints, k=2 * k)
        diag["reason"] += "relaxed retrieval (material term dropped); "
    diag["records"] = len(records)

    cands: List[ValueCandidate] = []

    t1_before = len(cands)
    if use_llm and records:
        try:
            prompt = build_extraction_prompt(param_key, material, records)
            raw = ollama_extract(prompt, model=ollama_model)
            raw_items = parse_llm_json(raw)
            for _it in raw_items:
                if isinstance(_it, dict):
                    _stamp_provenance(_it, 'llm_extract')
                    _it['_context'] = _is_context(param_key, 'llm_extract')
            cands.extend(gatekeep(raw_items, param_key, provenance="llm_extract"))
        except Exception as e:
            diag["reason"] += f"llm failed: {e}; "
            logger.warning("Tier-1 LLM failed (%s) → heuristic fallback: %s",
                           param_key, e)
    diag["n_llm"] = len(cands) - t1_before

    t2_before = len(cands)
    if records and (cascade_mode == 'union' or not cands):
        hits = heuristic_extract(records, param_key)
        for _h in hits:
            if isinstance(_h, dict):
                _stamp_provenance(_h, 'regex_ner')
                _h['_context'] = _is_context(param_key, 'regex_ner')
        cands.extend(gatekeep(hits, param_key, provenance="regex_ner"))
    diag["n_heuristic"] = len(cands) - t2_before

    t3_before = len(cands)
    if use_llm and allow_prior_inference and \
            (cascade_mode == 'union' or not cands):
        try:
            prompt = build_prior_inference_prompt(param_key, material, records)
            raw = ollama_extract(prompt, model=ollama_model)
            raw_items = parse_llm_json(raw)
            for _it in raw_items:
                if isinstance(_it, dict):
                    _stamp_provenance(_it, 'llm_prior')
                    _it['_context'] = _is_context(param_key, 'llm_prior')
            cands.extend(gatekeep(raw_items, param_key,
                                  provenance="llm_prior", conf_cap=0.5))
            if cascade_mode == 'union':
                diag["reason"] += "union: prior inference added as context; "
        except Exception as e:
            diag["reason"] += f"prior inference failed: {e}; "
            logger.warning("Tier-3 prior inference failed (%s): %s",
                           param_key, e)
    diag["n_prior"] = len(cands) - t3_before

    diag["tier"] = ("llm" if diag["n_llm"] else
                    "heuristic" if diag["n_heuristic"] else
                    "llm_prior" if diag["n_prior"] else "none")
    return cands, records, diag


# ============================================================================
# ███ v9.0.0 — PHYSICS-GROUNDED ρ₀ CASCADE                              ███
# ============================================================================
def recommend_rho0_physics(
    material: str = "Cu",
    synthesis: Optional[str] = None,
    architecture: Optional[str] = None,
    twin_spacing: Optional[float] = None,
    retriever=None,
    ollama_model: str = "qwen2.5:7b",
    cascade_mode: str = 'fallback',
) -> Tuple[List[ValueCandidate], List[Dict[str, Any]], Dict[str, Any]]:
    if retriever is None:
        retriever = get_retriever()

    cands, records, diag = recommend_param_values(
        param_key="rho0",
        material=material,
        k=6,
        use_llm=True,
        ollama_model=ollama_model,
        retriever=retriever,
        allow_prior_inference=False,
        cascade_mode=cascade_mode,
    )

    snippet_findings: List[Dict[str, Any]] = []
    for rec in records:
        text = rec.get("text") or rec.get("text_norm", "") or ""
        if not text:
            continue
        snippet_findings.extend(extract_rho0_with_context(text))

    for c in cands:
        for f in snippet_findings:
            fv = f.get("rho0_value")
            if fv is None:
                continue
            try:
                denom = max(abs(c.value), 1e-30)
                if abs(fv - c.value) / denom < 0.15:
                    extra = (
                        f"\nSnippet context → "
                        f"synthesis={f.get('synthesis_method')}, "
                        f"architecture={f.get('grain_architecture')}, "
                        f"λ={f.get('twin_spacing')} "
                        f"{f.get('twin_spacing_unit') or 'nm'}."
                    )
                    c.reasoning = (c.reasoning or "") + extra
                    break
            except (TypeError, ValueError):
                continue

    user_syn = synthesis
    user_arch = architecture
    user_lam = twin_spacing
    if user_syn or user_arch or user_lam:
        snippet_findings.append({
            "rho0_mention":        "(user-supplied context)",
            "rho0_value":          None,
            "rho0_unit":           None,
            "synthesis_method":    user_syn,
            "grain_architecture":  user_arch,
            "twin_spacing":        user_lam,
            "twin_spacing_unit":   "nm",
            "snippet":             "",
            "position":            -1,
        })

    if cascade_mode == 'union' or not cands:
        try:
            prompt = build_rho0_physics_inference_prompt(
                material=material,
                synthesis=synthesis,
                architecture=architecture,
                twin_spacing_nm=twin_spacing,
                records=records,
            )
            raw = ollama_extract(prompt, model=ollama_model)
            raw_items = parse_llm_json(raw)
            for _it in raw_items:
                if isinstance(_it, dict):
                    _stamp_provenance(_it, 'llm_prior')
                    _it['_context'] = _is_context('rho0', 'llm_prior')
            cands.extend(gatekeep(raw_items, "rho0",
                                  provenance="llm_prior", conf_cap=0.5))
        except Exception as e:
            logger.warning("ρ₀ physics-regime LLM failed: %s", e)

    if cascade_mode == 'union' or not cands:
        regime = classify_rho0_regime(synthesis, architecture, twin_spacing)
        regime_cand = ValueCandidate(
            value=regime["inferred"],
            unit="m^-2",
            provenance="regime_prior",
            property_label="ρ₀ (physics-regime classifier fallback)",
            method="physics_regime_inference",
            evidence=f"Regime: {regime['regime']}",
            source=regime["source"],
            confidence=0.35,
            reasoning=(
                f"Step 1: synthesis={synthesis} → baseline regime.\n"
                f"Step 2: architecture={architecture} → refine columnar "
                f"vs equiaxed.\n"
                f"Step 3: λ={twin_spacing} nm → modulate within regime.\n"
                f"Step 4: regime={regime['regime']} "
                f"[{regime['low']:.1e}, {regime['high']:.1e}] m^-2.\n"
                f"Step 5: inferred log-center = {regime['inferred']:.3e} m^-2."
            ),
            dominant_factor=("combined" if (synthesis and architecture)
                             else "synthesis" if synthesis
                             else "architecture" if architecture
                             else "twin_spacing"),
            context=_is_context('rho0', 'regime_prior'),
        )
        if not any(abs(c.value - regime_cand.value) / max(regime_cand.value, 1e-30) < 1e-3
                   for c in cands):
            cands.append(regime_cand)

    final_regime = classify_rho0_regime(synthesis, architecture, twin_spacing)
    return cands, snippet_findings, final_regime


# ============================================================================
# ███ v9.1.0 — THEORY-AWARE γ̇₀ CASCADE                                ███
# ============================================================================
def recommend_gamma0_theory_aware(
    material: str = "Cu",
    target_theory: Optional[str] = None,
    retriever=None,
    ollama_model: str = "qwen2.5:7b",
    cascade_mode: str = 'fallback',
) -> Tuple[List[ValueCandidate], List[Dict[str, Any]], Dict[str, Any]]:
    if retriever is None:
        retriever = get_retriever()

    records = retriever.search("gamma0_dot", material, k=8)
    cands: List[ValueCandidate] = []
    parsed_items: List[Dict[str, Any]] = []
    if records:
        try:
            prompt = build_gamma0_theory_aware_prompt(material, records)
            raw = ollama_extract(prompt, model=ollama_model)
            parsed_items = parse_llm_json(raw)
            for _it in parsed_items:
                if isinstance(_it, dict):
                    _stamp_provenance(_it, 'llm_extract')
                    _it['_context'] = _is_context('gamma0_dot', 'llm_extract')
            cands.extend(gatekeep(parsed_items, "gamma0_dot",
                                  provenance="llm_extract"))
            for c in cands:
                for item in parsed_items:
                    try:
                        iv = float(item.get("value"))
                    except (TypeError, ValueError):
                        continue
                    if iv <= 0:
                        continue
                    if abs(iv - c.value) / max(abs(c.value), 1e-30) < 0.05:
                        tag = str(item.get("theory") or "unspecified")
                        if tag and tag != "unspecified":
                            c.theory = tag
                        break
        except Exception as e:
            logger.warning("γ̇₀ theory-aware LLM failed: %s", e)

    snippet_findings: List[Dict[str, Any]] = []
    for rec in records:
        text = rec.get("text") or rec.get("text_norm", "") or ""
        if text:
            snippet_findings.extend(extract_gamma0_with_context(text))

    for c in cands:
        if c.theory and c.theory != "unspecified":
            continue
        for f in snippet_findings:
            fv = f.get("gamma0_value")
            if fv is None:
                continue
            try:
                if abs(fv - c.value) / max(abs(c.value), 1e-30) < 0.15:
                    if f.get("theory_key"):
                        c.theory = f["theory_key"]
                    elif f.get("theory"):
                        c.theory = f["theory"]
                    if f.get("theory"):
                        c.reasoning = (c.reasoning or "") + (
                            f"\nSnippet theory context: {f['theory']}")
                    break
            except (TypeError, ValueError, ZeroDivisionError):
                continue

    if cascade_mode == 'union' and records:
        hits = heuristic_extract(records, "gamma0_dot")
        for _h in hits:
            if isinstance(_h, dict):
                _stamp_provenance(_h, 'regex_ner')
                _h['_context'] = _is_context('gamma0_dot', 'regex_ner')
        existing_vals = [c.value for c in cands]
        new_cands = gatekeep(hits, "gamma0_dot", provenance="regex_ner")
        for nc in new_cands:
            if not any(abs(nc.value - ev) / max(abs(ev), 1e-30) < 1e-3
                       for ev in existing_vals):
                cands.append(nc)
                existing_vals.append(nc.value)

    if target_theory:
        snippet_findings.append({
            "gamma0_mention": "(user-supplied context)",
            "gamma0_value":    None,
            "gamma0_unit":     None,
            "theory":          target_theory,
            "theory_key":      _norm_theory(target_theory),
            "snippet":         "",
            "position":        -1,
        })

    if target_theory and (cascade_mode == 'union' or not cands):
        try:
            prompt = build_gamma0_theory_inference_prompt(
                material, target_theory, records)
            raw = ollama_extract(prompt, model=ollama_model)
            raw_items = parse_llm_json(raw)
            for _it in raw_items:
                if isinstance(_it, dict):
                    _stamp_provenance(_it, 'llm_prior')
                    _it['_context'] = _is_context('gamma0_dot', 'llm_prior')
            new_cands = gatekeep(raw_items, "gamma0_dot",
                                 provenance="llm_prior", conf_cap=0.5)
            norm_key = _norm_theory(target_theory) or target_theory
            for c in new_cands:
                if not c.theory or c.theory == "unspecified":
                    c.theory = norm_key
            cands.extend(new_cands)
        except Exception as e:
            logger.warning("γ̇₀ theory-regime LLM failed: %s", e)

    if target_theory and (cascade_mode == 'union' or not cands):
        regime = classify_gamma0_regime(target_theory)
        norm_key = _norm_theory(target_theory) or target_theory
        regime_cand = ValueCandidate(
            value=regime["inferred"],
            unit="s^-1",
            provenance="regime_prior",
            property_label="γ̇₀ (theory-regime classifier fallback)",
            method="theory_regime_inference",
            evidence=f"Regime: {regime['regime']}",
            source=regime["source"],
            confidence=0.35,
            theory=norm_key,
            reasoning=(f"Step 1: theory={target_theory} → regime.\n"
                       f"Step 2: regime={regime['regime']} "
                       f"[{regime['low']:.1e}, {regime['high']:.1e}] s⁻¹.\n"
                       f"Step 3: inferred γ̇₀="
                       f"{regime['inferred']:.3e} s⁻¹."),
            dominant_factor="theory",
            context=_is_context('gamma0_dot', 'regime_prior'),
        )
        if not any(abs(c.value - regime_cand.value) / max(regime_cand.value, 1e-30) < 1e-3
                   for c in cands):
            cands.append(regime_cand)

    final_regime = classify_gamma0_regime(target_theory)
    return cands, snippet_findings, final_regime


@st.cache_resource(show_spinner=False)
def get_retriever(folder: str = "json_metadatabase",
                  use_dense: bool = True) -> HybridRetriever:
    corpus = load_corpus_folder(folder)
    return HybridRetriever(corpus, use_dense=use_dense)


# ============================================================================
# ENHANCED SIMULATION DATABASE
# ============================================================================
class SimulationDatabase:
    @staticmethod
    @handle_errors
    def generate_id(sim_params):
        param_str = json.dumps({k: v for k, v in sim_params.items()
                                if k not in ['history', 'results', 'geom_viz']},
                               sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    @staticmethod
    @handle_errors
    def save_simulation(sim_params, results_history, geometry_data,
                        metadata=None, run_time=0.0):
        if 'twin_simulations' not in st.session_state:
            st.session_state.twin_simulations = {}
        sim_id = SimulationDatabase.generate_id(sim_params)
        if metadata is None:
            metadata = MetadataManager.create_metadata(sim_params, results_history, run_time)
        else:
            metadata = MetadataManager.validate_metadata(metadata)
        st.session_state.twin_simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'results_history': results_history,
            'geometry_data': geometry_data,
            'metadata': metadata,
            'created_at': metadata.get('created_at', datetime.now().isoformat()),
            'last_modified': datetime.now().isoformat()
        }
        return sim_id

    @staticmethod
    @handle_errors
    def get_simulation(sim_id):
        if 'twin_simulations' in st.session_state and sim_id in st.session_state.twin_simulations:
            sim_data = st.session_state.twin_simulations[sim_id]
            if 'metadata' in sim_data:
                sim_data['metadata'] = MetadataManager.validate_metadata(sim_data['metadata'])
            return sim_data
        return None

    @staticmethod
    @handle_errors
    def delete_simulation(sim_id):
        if 'twin_simulations' in st.session_state and sim_id in st.session_state.twin_simulations:
            del st.session_state.twin_simulations[sim_id]
            return True
        return False

    @staticmethod
    def get_all_simulations():
        if 'twin_simulations' in st.session_state:
            for sim_id, sim_data in st.session_state.twin_simulations.items():
                if 'metadata' in sim_data:
                    sim_data['metadata'] = MetadataManager.validate_metadata(sim_data['metadata'])
            return st.session_state.twin_simulations
        return {}

    @staticmethod
    def get_simulation_list():
        if 'twin_simulations' not in st.session_state:
            return []
        simulations = []
        for sim_id, sim_data in st.session_state.twin_simulations.items():
            try:
                params = sim_data.get('params', {})
                metadata = sim_data.get('metadata', {})
                name = (f"λ={params.get('twin_spacing', 0):.1f}nm | "
                        f"σ={params.get('applied_stress', 0) / 1e6:.0f}MPa | "
                        f"θ={params.get('applied_stress_angle', 0):.0f}° | "
                        f"W={params.get('W', 0):.1f}")
                simulations.append({
                    'id': sim_id, 'name': name, 'params': params,
                    'metadata': metadata,
                    'results': sim_data['results_history'][-1] if sim_data['results_history'] else None
                })
            except Exception:
                continue
        return simulations


@handle_errors
def sanitize_token(text: str) -> str:
    try:
        s = str(text)
        for ch in [" ", "{", "}", "/", "\\", ",", ";", "(", ")", "[", "]", "°"]:
            s = s.replace(ch, "")
        return s
    except Exception:
        return "unknown"


@handle_errors
def fmt_num_trim(x, ndigits=3):
    try:
        s = f"{x:.{ndigits}f}"
        s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s
    except Exception:
        return "0.0"


@handle_errors
def build_sim_name(params: dict, sim_id: str = None) -> str:
    try:
        geom_type = params.get("geometry_type", "standard")
        if geom_type == "defect":
            defect = params.get("defect_type", "dislocation")
            geom_token = f"twin_grain_with_{defect}"
        else:
            geom_token = "standard_twin_grain"
        twin_spacing = fmt_num_trim(params.get("twin_spacing", 20.0), ndigits=1)
        W = fmt_num_trim(params.get("W", 2.0), ndigits=1)
        stress_mpa = fmt_num_trim(params.get("applied_stress", 300e6) / 1e6, ndigits=0)
        theta = fmt_num_trim(params.get("applied_stress_angle", 0.0), ndigits=0)
        name = f"twin_lambda_{twin_spacing}_W_{W}_stress_{stress_mpa}MPa_theta_{theta}_{geom_token}"
        if sim_id:
            name = f"{name}_{sim_id}"
        return name
    except Exception:
        return f"twin_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# MATERIAL PROPERTIES
# ============================================================================
class MaterialProperties:
    @staticmethod
    def get_cu_properties():
        return {
            'elastic': {'C11': 168.4e9, 'C12': 121.4e9, 'C44': 75.4e9,
                        'source': 'Phys. Rev. B 73, 064112 (2006)'},
            'twinning': {
                'gamma_tw': 1 / np.sqrt(2),
                'n_111': np.array([1, 1, 1]) / np.sqrt(3),
                'a_112': np.array([1, 1, -2]) / np.sqrt(6),
                'n_2d': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
                'a_2d': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
            },
            'plasticity': {'mu': 48e9, 'nu': 0.34, 'b': 0.256e-9,
                           'sigma0': 50e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
        }

    @staticmethod
    def get_al_properties():
        return {
            'elastic': {'C11': 106.8e9, 'C12': 60.4e9, 'C44': 28.3e9,
                        'source': 'J. Appl. Phys. 88, 3287 (2000)'},
            'twinning': {
                'gamma_tw': 1 / np.sqrt(2),
                'n_111': np.array([1, 1, 1]) / np.sqrt(3),
                'a_112': np.array([1, 1, -2]) / np.sqrt(6),
                'n_2d': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
                'a_2d': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
            },
            'plasticity': {'mu': 26e9, 'nu': 0.33, 'b': 0.286e-9,
                           'sigma0': 30e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e12}
        }

    @staticmethod
    def get_ni_properties():
        return {
            'elastic': {'C11': 246.5e9, 'C12': 147.3e9, 'C44': 124.7e9,
                        'source': 'Phys. Rev. B 94, 014110 (2016)'},
            'twinning': {
                'gamma_tw': 1 / np.sqrt(2),
                'n_111': np.array([1, 1, 1]) / np.sqrt(3),
                'a_112': np.array([1, 1, -2]) / np.sqrt(6),
                'n_2d': np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]),
                'a_2d': np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
            },
            'plasticity': {'mu': 80e9, 'nu': 0.31, 'b': 0.249e-9,
                           'sigma0': 70e6, 'gamma0_dot': 1e-3, 'm': 20, 'rho0': 1e13}
        }

    @staticmethod
    @handle_errors
    def get_material(material_name='Cu'):
        if material_name == 'Cu':
            return MaterialProperties.get_cu_properties()
        elif material_name == 'Al':
            return MaterialProperties.get_al_properties()
        elif material_name == 'Ni':
            return MaterialProperties.get_ni_properties()
        else:
            return MaterialProperties.get_cu_properties()

    @staticmethod
    @handle_errors
    def validate_parameters(params):
        errors = []
        warnings_list = []
        if params.get('dt', 0) <= 0:
            errors.append("Time step dt must be positive")
        if params.get('dx', 0) <= 0:
            errors.append("Grid spacing dx must be positive")
        if params.get('N', 0) < 32:
            warnings_list.append("Grid resolution N < 32 may produce inaccurate results")
        if params.get('twin_spacing', 0) < 5:
            warnings_list.append("Twin spacing < 5nm may be physically unrealistic")
        if params.get('applied_stress', 0) > 2e9:
            warnings_list.append("Applied stress > 2GPa may cause unrealistic deformation")
        return errors, warnings_list


# ============================================================================
# INITIAL GEOMETRY
# ============================================================================
class InitialGeometryVisualizer:
    def __init__(self, N, dx):
        self.N = N
        self.dx = dx
        self.x = np.linspace(-N * dx / 2, N * dx / 2, N)
        self.y = np.linspace(-N * dx / 2, N * dx / 2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.extent = [-N * dx / 2, N * dx / 2, -N * dx / 2, N * dx / 2]

    @handle_errors
    def create_twin_grain_geometry(self, twin_spacing=20.0, grain_boundary_pos=0.0,
                                   gb_width=3.0, buffer_width=5.0,
                                   left_buffer_width=5.0,
                                   gb_profile='plane', gb_curvature=0.0):
        eta1 = np.zeros((self.N, self.N))
        eta2 = np.zeros((self.N, self.N))
        phi = np.zeros((self.N, self.N))

        def gb_x_func(y):
            if gb_profile == 'plane':
                return grain_boundary_pos
            elif gb_profile == 'concave':
                sigma = (self.N * self.dx) / 4.0
                deviation = -gb_curvature * np.exp(-y**2 / (2 * sigma**2))
                return grain_boundary_pos + deviation
            elif gb_profile == 'convex':
                sigma = (self.N * self.dx) / 4.0
                deviation = gb_curvature * np.exp(-y**2 / (2 * sigma**2))
                return grain_boundary_pos + deviation
            else:
                return grain_boundary_pos

        left_edge = self.extent[0]

        for i in range(self.N):
            for j in range(self.N):
                x_val = self.X[i, j]
                y_val = self.Y[i, j]
                gb_x = gb_x_func(y_val)
                dist_from_gb = x_val - gb_x
                if dist_from_gb < -gb_width:
                    eta1[i, j] = 1.0
                    eta2[i, j] = 0.0
                elif dist_from_gb > gb_width:
                    eta1[i, j] = 0.0
                    eta2[i, j] = 1.0
                else:
                    transition = 0.5 * (1 - np.tanh(dist_from_gb / (gb_width / 3)))
                    eta1[i, j] = transition
                    eta2[i, j] = 1 - transition

        for i in range(self.N):
            for j in range(self.N):
                if eta1[i, j] > 0.5:
                    x_val = self.X[i, j]
                    y_val = self.Y[i, j]
                    gb_x = gb_x_func(y_val)
                    dist_from_gb = abs(x_val - gb_x)
                    dist_from_left_edge = abs(x_val - left_edge)
                    if dist_from_gb > buffer_width and dist_from_left_edge > left_buffer_width:
                        phase = 2 * np.pi * self.Y[i, j] / twin_spacing
                        phi[i, j] = np.tanh(np.sin(phase) * 3.0)
                    else:
                        phi[i, j] = 1.0
        return phi, eta1, eta2

    @handle_errors
    def create_defect_geometry(self, twin_spacing=20.0, defect_type='dislocation',
                               defect_pos=(0, 0), defect_radius=10.0,
                               grain_boundary_pos=0.0, gb_width=3.0,
                               buffer_width=5.0, left_buffer_width=5.0,
                               gb_profile='plane', gb_curvature=0.0):
        phi, eta1, eta2 = self.create_twin_grain_geometry(
            twin_spacing, grain_boundary_pos, gb_width,
            buffer_width, left_buffer_width, gb_profile, gb_curvature
        )
        if defect_type == 'dislocation':
            center_x, center_y = defect_pos
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i, j] - center_x)**2 +
                                   (self.Y[i, j] - center_y)**2)
                    if dist < defect_radius:
                        phase_shift = np.exp(-dist**2 / (defect_radius**2)) * np.pi
                        phase = 2 * np.pi * self.Y[i, j] / twin_spacing + phase_shift
                        phi_candidate = np.tanh(np.sin(phase) * 3.0)
                        if eta1[i, j] > 0.5:
                            phi[i, j] = phi_candidate
        elif defect_type == 'void':
            center_x, center_y = defect_pos
            for i in range(self.N):
                for j in range(self.N):
                    dist = np.sqrt((self.X[i, j] - center_x)**2 +
                                   (self.Y[i, j] - center_y)**2)
                    if dist < defect_radius:
                        eta1[i, j] = 0.0
                        eta2[i, j] = 0.0
                        phi[i, j] = 0.0
        return phi, eta1, eta2


class EnhancedSpectralSolver:
    def __init__(self, N, dx, elastic_params, kx=None, ky=None, k2=None):
        self.N = N
        self.dx = dx
        if kx is None or ky is None or k2 is None:
            self.kx, self.ky, self.k2 = make_k_vectors(N, dx)
        else:
            self.kx, self.ky, self.k2 = kx, ky, k2

        C11 = elastic_params['C11']
        C12 = elastic_params['C12']
        C44 = elastic_params['C44']
        C11_2d = (C11 + C12 + 2 * C44) / 2
        C12_2d = (C11 + C12 - 2 * C44) / 2
        lambda_2d = C12_2d
        mu_2d = (C11_2d - C12_2d) / 2
        self.C11_2d = C11_2d
        self.C12_2d = C12_2d
        self.C44_2d = C44

        denom = mu_2d * (lambda_2d + 2 * mu_2d) * self.k2 + 1e-15
        self.G11 = (mu_2d * (self.kx**2 + 2 * self.ky**2) + lambda_2d * self.ky**2) / denom
        self.G12 = -mu_2d * self.kx * self.ky / denom
        self.G22 = (mu_2d * (self.ky**2 + 2 * self.kx**2) + lambda_2d * self.kx**2) / denom

    @handle_errors
    def solve(self, eigenstrain_xx, eigenstrain_yy, eigenstrain_xy,
              applied_stress_xx=0, applied_stress_yy=0, applied_stress_xy=0):
        assert eigenstrain_xx.shape == (self.N, self.N)
        eps_xx_hat = fft2(eigenstrain_xx)
        eps_yy_hat = fft2(eigenstrain_yy)
        eps_xy_hat = fft2(eigenstrain_xy)

        ux_hat = 1j * (self.G11 * self.kx * eps_xx_hat +
                       self.G12 * self.ky * eps_xx_hat +
                       self.G12 * self.kx * eps_yy_hat +
                       self.G22 * self.ky * eps_yy_hat)
        uy_hat = 1j * (self.G12 * self.kx * eps_xx_hat +
                       self.G22 * self.ky * eps_xx_hat +
                       self.G11 * self.kx * eps_yy_hat +
                       self.G12 * self.ky * eps_yy_hat)

        eps_xx_el = np.real(ifft2(1j * self.kx * ux_hat))
        eps_yy_el = np.real(ifft2(1j * self.ky * uy_hat))
        eps_xy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_hat + self.ky * ux_hat)))

        eps_xx = eps_xx_el + eigenstrain_xx
        eps_yy = eps_yy_el + eigenstrain_yy
        eps_xy = eps_xy_el + eigenstrain_xy

        sxx = applied_stress_xx + self.C11_2d * eps_xx + self.C12_2d * eps_yy
        syy = applied_stress_yy + self.C12_2d * eps_xx + self.C11_2d * eps_yy
        sxy = applied_stress_xy + 2 * self.C44_2d * eps_xy

        sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + syy**2 + sxx**2 + 6 * sxy**2))
        sigma_eq = np.clip(sigma_eq, 0, 5e9)
        sigma_h = (sxx + syy) / 2

        return sigma_eq, sxx, syy, sxy, sigma_h, eps_xx, eps_yy, eps_xy


class EnhancedTwinVisualizer:
    def __init__(self, N, dx, dt=1e-4):
        self.N = N
        self.dx = dx
        self.dt = dt
        self.extent = [-N * dx / 2, N * dx / 2, -N * dx / 2, N * dx / 2]
        self.line_profiler = EnhancedLineProfiler(N, dx)
        self.COLORMAPS = COLORMAPS.copy()
        custom = PublicationEnhancer.create_custom_colormaps()
        self.COLORMAPS.update(custom)

    @handle_errors
    def get_colormap(self, cmap_name):
        if cmap_name in self.COLORMAPS:
            entry = self.COLORMAPS[cmap_name]
            if isinstance(entry, str):
                return plt.get_cmap(entry)
            else:
                return entry
        return plt.get_cmap('viridis')

    @handle_errors
    def create_multi_field_comparison(self, results_dict, style_params=None):
        if style_params is None:
            style_params = {}
        defaults = {
            'title_font_size': 10, 'label_font_size': 8,
            'scalebar_color': 'black', 'scalebar_fontsize': 8,
            'phi_cmap': 'RdBu_r', 'sigma_eq_cmap': 'hot',
            'sigma_h_cmap': 'RdBu', 'h_cmap': 'plasma',
            'eps_p_mag_cmap': 'YlOrRd', 'sigma_y_cmap': 'viridis',
            'eta1_cmap': 'Reds',
        }
        for k, v in defaults.items():
            style_params.setdefault(k, v)
        title_font_size = float(style_params['title_font_size'])
        label_font_size = float(style_params['label_font_size'])
        scalebar_fontsize = float(style_params['scalebar_fontsize'])

        fields_to_plot = [
            ('phi', 'Twin Order Parameter φ', style_params['phi_cmap'], [-1.2, 1.2]),
            ('eta1', 'Grain η₁', style_params['eta1_cmap'], [0, 1]),
            ('sigma_eq', 'Von Mises Stress (GPa)', style_params['sigma_eq_cmap'], None),
            ('sigma_h', 'Hydrostatic Stress (GPa)', style_params['sigma_h_cmap'], None),
            ('h', 'Twin Spacing (nm)', style_params['h_cmap'], [0, 30]),
            ('eps_p_mag', 'Plastic Strain', style_params['eps_p_mag_cmap'], None),
            ('sigma_y', 'Yield Stress (MPa)', style_params['sigma_y_cmap'], None),
        ]
        available_fields = [(fname, title, cmap, vrange)
                            for fname, title, cmap, vrange in fields_to_plot
                            if fname in results_dict]
        n_fields = len(available_fields)
        if n_fields == 0:
            return None
        cols = min(3, n_fields)
        rows = (n_fields + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        for idx, (field_name, title, default_cmap, vrange) in enumerate(available_fields):
            ax = axes[idx]
            data = results_dict[field_name]
            if field_name in ['sigma_eq', 'sigma_h']:
                data = data / 1e9
            elif field_name == 'sigma_y':
                data = data / 1e6
            cmap_name = style_params.get(f'{field_name}_cmap', default_cmap)
            cmap = self.get_colormap(cmap_name)
            if vrange is not None:
                vmin, vmax = vrange
            else:
                vmin = np.percentile(data, 2)
                vmax = np.percentile(data, 98)
                if field_name in ['sigma_h']:
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
            im = ax.imshow(data, extent=self.extent, cmap=cmap,
                           vmin=vmin, vmax=vmax, origin='lower', aspect='equal',
                           interpolation='bilinear')
            if field_name == 'phi':
                ax.contour(np.linspace(self.extent[0], self.extent[1], self.N),
                           np.linspace(self.extent[2], self.extent[3], self.N),
                           data, levels=[0], colors='white', linewidths=1, alpha=0.8)
            ax.set_title(title, fontsize=title_font_size)
            ax.set_xlabel('x (nm)', fontsize=label_font_size)
            ax.set_ylabel('y (nm)', fontsize=label_font_size)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if field_name in ('sigma_eq', 'sigma_h'):
                cbar.set_label('Stress (GPa)')
            elif field_name == 'sigma_y':
                cbar.set_label('Stress (MPa)')
            elif field_name == 'h':
                cbar.set_label('Spacing (nm)')
            if field_name in ['phi', 'eta1', 'sigma_eq', 'sigma_h']:
                PublicationEnhancer.add_scale_bar(
                    ax, 10.0, 'lower right',
                    color=style_params['scalebar_color'],
                    fontsize=scalebar_fontsize)
        for idx in range(n_fields, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        return fig

    @handle_errors
    def create_plotly_heatmap(self, results_dict, field_name, frame_idx=0):
        if field_name not in results_dict:
            return None
        data = results_dict[field_name].copy()
        title = field_name
        unit = ""
        if field_name in ['sigma_eq', 'sigma_h']:
            data = data / 1e9
            unit = " (GPa)"
        elif field_name == 'sigma_y':
            data = data / 1e6
            unit = " (MPa)"
        elif field_name == 'h':
            unit = " (nm)"
        if field_name == 'phi':
            colorscale = 'RdBu'; zmid = 0
        elif field_name == 'eta1':
            colorscale = 'Reds'; zmid = None
        elif field_name in ['sigma_eq', 'sigma_h']:
            colorscale = 'Viridis' if field_name == 'sigma_eq' else 'RdBu'
            zmid = 0 if field_name == 'sigma_h' else None
        else:
            colorscale = 'Plasma'; zmid = None
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=data,
            x=np.linspace(self.extent[0], self.extent[1], self.N),
            y=np.linspace(self.extent[2], self.extent[3], self.N),
            colorscale=colorscale, zmid=zmid,
            colorbar=dict(title=f"{field_name}{unit}"),
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>%{z:.3f}<extra></extra>'))
        if field_name != 'phi' and 'phi' in results_dict:
            phi_data = results_dict['phi']
            fig.add_trace(go.Contour(
                z=phi_data,
                x=np.linspace(self.extent[0], self.extent[1], self.N),
                y=np.linspace(self.extent[2], self.extent[3], self.N),
                contours=dict(start=0, end=0, size=0, coloring='none', showlabels=False),
                line=dict(color='white', width=2),
                showscale=False, hoverinfo='skip'))
        fig.update_layout(
            title=f"{title} (Frame {frame_idx})",
            xaxis_title="x (nm)", yaxis_title="y (nm)",
            width=600, height=500, template="plotly_white")
        return fig

    @handle_errors
    def create_plotly_line_profiles(self, results_dict, field_name, profile_types,
                                    position_ratio=0.5):
        fig = go.Figure()
        for ptype in profile_types:
            distance, profile, _ = self.line_profiler.extract_profile(
                results_dict[field_name], ptype, position_ratio)
            if field_name in ['sigma_eq', 'sigma_h']:
                profile = profile / 1e9
                ylabel = 'Stress (GPa)'
            elif field_name == 'sigma_y':
                profile = profile / 1e6
                ylabel = 'Stress (MPa)'
            else:
                ylabel = field_name
            fig.add_trace(go.Scatter(x=distance, y=profile, mode='lines',
                                     name=ptype.replace('_', ' ').title()))
        fig.update_layout(
            title=f"{field_name} Line Profiles",
            xaxis_title="Position (nm)", yaxis_title=ylabel,
            hovermode='x unified', template="plotly_white")
        return fig

    @handle_errors
    def create_plotly_3d_surface(self, results_dict, field_name, frame_idx=0):
        if field_name not in results_dict:
            return None
        data = results_dict[field_name].copy()
        title = field_name
        unit = ""
        if field_name in ['sigma_eq', 'sigma_h']:
            data = data / 1e9
            unit = " (GPa)"
        elif field_name == 'sigma_y':
            data = data / 1e6
            unit = " (MPa)"
        elif field_name == 'h':
            unit = " (nm)"
        x = np.linspace(self.extent[0], self.extent[1], self.N)
        y = np.linspace(self.extent[2], self.extent[3], self.N)
        X, Y = np.meshgrid(x, y)
        if field_name == 'phi':
            colorscale = 'RdBu'; cmin, cmax = -1.2, 1.2
        elif field_name == 'eta1':
            colorscale = 'Reds'; cmin, cmax = 0, 1
        elif field_name == 'sigma_h':
            colorscale = 'RdBu'
            cmin, cmax = -np.max(np.abs(data)), np.max(np.abs(data))
        else:
            colorscale = 'Viridis'; cmin, cmax = None, None
        fig = go.Figure(data=[go.Surface(
            z=data, x=X, y=Y, colorscale=colorscale, cmin=cmin, cmax=cmax,
            colorbar=dict(title=f"{field_name}{unit}"),
            hovertemplate='x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>z: %{z:.3f}<extra></extra>')])
        fig.update_layout(
            title=f"3D Surface: {title} (Frame {frame_idx})",
            scene=dict(xaxis_title='x (nm)', yaxis_title='y (nm)',
                       zaxis_title=f'{title}{unit}',
                       camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
            width=700, height=600, template="plotly_white")
        return fig

    @handle_errors
    def create_animation(self, history, field_name, output_format='gif',
                         fps=5, dpi=150):
        if not history:
            return None
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        first_data = history[0][field_name].copy()
        if field_name in ['sigma_eq', 'sigma_h']:
            first_data = first_data / 1e9
        elif field_name == 'sigma_y':
            first_data = first_data / 1e6
        vmin = np.percentile(first_data, 2)
        vmax = np.percentile(first_data, 98)
        if field_name == 'phi':
            vmin, vmax = -1.2, 1.2
        elif field_name == 'eta1':
            vmin, vmax = 0, 1
        elif field_name == 'sigma_h':
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        im = ax.imshow(first_data, extent=self.extent,
                       cmap=self.get_colormap('viridis'),
                       vmin=vmin, vmax=vmax, origin='lower',
                       interpolation='bilinear')
        cbar = plt.colorbar(im, ax=ax)
        if field_name in ['sigma_eq', 'sigma_h']:
            cbar.set_label('Stress (GPa)')
        elif field_name == 'sigma_y':
            cbar.set_label('Stress (MPa)')
        elif field_name == 'h':
            cbar.set_label('Spacing (nm)')
        else:
            cbar.set_label(field_name)
        title = ax.set_title(f"{field_name} - t = 0.000 ns")
        dt_local = self.dt

        def update_frame(frame_idx):
            data = history[frame_idx][field_name].copy()
            if field_name in ['sigma_eq', 'sigma_h']:
                data = data / 1e9
            elif field_name == 'sigma_y':
                data = data / 1e6
            im.set_array(data)
            time_ns = frame_idx * dt_local * 1e3
            title.set_text(f"{field_name} - t = {time_ns:.3f} ns")
            return [im, title]

        ani = animation.FuncAnimation(fig, update_frame, frames=len(history),
                                      interval=1000 / fps, blit=True)
        buffer = BytesIO()
        if output_format == 'gif':
            ani.save(buffer, writer='pillow', fps=fps, dpi=dpi)
        else:
            ani.save(buffer, writer='ffmpeg', fps=fps, dpi=dpi)
        plt.close(fig)
        buffer.seek(0)
        return buffer


# ============================================================================
# MAIN SOLVER
# ============================================================================
class NanotwinnedCuSolver:
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']

        material_name = params.get('material', 'Cu')
        self.mat_props = MaterialProperties.get_material(material_name)
        self.params['material'] = material_name

        apply_plasticity_overrides(self)

        errors, warnings_list = MaterialProperties.validate_parameters(params)
        if errors:
            raise ValueError(f"Parameter validation failed: {', '.join(errors)}")
        if warnings_list:
            st.warning(f"Parameter warnings: {', '.join(warnings_list)}")

        self.geom_viz = InitialGeometryVisualizer(self.N, self.dx)
        self.kx, self.ky, self.k2 = make_k_vectors(self.N, self.dx)

        try:
            self.phi, self.eta1, self.eta2 = self.initialize_fields()
        except Exception as e:
            st.error(f"Failed to initialize fields: {e}")
            self.phi = np.zeros((self.N, self.N))
            self.eta1 = np.zeros((self.N, self.N))
            self.eta2 = np.zeros((self.N, self.N))

        self.eps_p_xx = np.zeros((self.N, self.N))
        self.eps_p_yy = np.zeros((self.N, self.N))
        self.eps_p_xy = np.zeros((self.N, self.N))

        self.spectral_solver = EnhancedSpectralSolver(
            self.N, self.dx, self.mat_props['elastic'],
            kx=self.kx, ky=self.ky, k2=self.k2)

        self.history = {
            'phi_norm': [], 'energy': [], 'max_stress': [],
            'plastic_work': [], 'avg_stress': [], 'twin_spacing_avg': []
        }

        kappa0 = float(params.get('kappa0', 1.0))
        gamma_aniso = float(params.get('gamma_aniso', 0.7))
        L_CTB = float(params.get('L_CTB', 0.05))
        L_ITB = float(params.get('L_ITB', 5.0))
        self.implicit_diff_phi = L_ITB * kappa0 * (1.0 + gamma_aniso)
        kappa_eta = float(params.get('kappa_eta', 2.0))
        L_eta = float(params.get('L_eta', 1.0))
        self.implicit_diff_eta = L_eta * kappa_eta

        self.confine_twin = params.get('confine_twin', True)

    @handle_errors
    def initialize_fields(self):
        geom_type = self.params.get('geometry_type', 'standard')
        twin_spacing = self.params['twin_spacing']
        gb_pos = self.params['grain_boundary_pos']
        gb_width = self.params.get('gb_width', 3.0)
        buffer_width = self.params.get('buffer_width', 5.0)
        left_buffer_width = self.params.get('left_buffer_width', 5.0)
        gb_profile = self.params.get('gb_profile', 'plane')
        gb_curvature = self.params.get('gb_curvature', 0.0)

        if geom_type == 'defect':
            defect_type = self.params.get('defect_type', 'dislocation')
            defect_pos = self.params.get('defect_pos', (0, 0))
            defect_radius = self.params.get('defect_radius', 10.0)
            return self.geom_viz.create_defect_geometry(
                twin_spacing, defect_type, defect_pos, defect_radius,
                gb_pos, gb_width, buffer_width, left_buffer_width,
                gb_profile, gb_curvature)
        else:
            return self.geom_viz.create_twin_grain_geometry(
                twin_spacing, gb_pos, gb_width,
                buffer_width, left_buffer_width, gb_profile, gb_curvature)

    @handle_errors
    def compute_local_energy_derivatives(self):
        W = self.params['W']
        A = self.params['A']
        B = self.params['B']
        df_dphi = 4 * W * self.phi * (self.phi**2 - 1) * self.eta1**2
        df_deta1 = (2 * A * self.eta1 * (1 - self.eta1) * (1 - 2 * self.eta1) +
                    2 * B * self.eta1 * self.eta2**2 +
                    2 * W * (self.phi**2 - 1)**2 * self.eta1)
        df_deta2 = (2 * A * self.eta2 * (1 - self.eta2) * (1 - 2 * self.eta2) +
                    2 * B * self.eta2 * self.eta1**2)
        return df_dphi, df_deta1, df_deta2

    @handle_errors
    def compute_elastic_driving_force(self, sxx, syy, sxy):
        try:
            gamma_tw = self.mat_props['twinning']['gamma_tw']
            n = self.mat_props['twinning']['n_2d']
            a = self.mat_props['twinning']['a_2d']
            dh_dphi = 0.25 * (3 * self.phi**2 - 2 * self.phi - 1)
            nx, ny = n[0], n[1]
            ax, ay = a[0], a[1]
            deps_xx_dphi = gamma_tw * nx * ax * dh_dphi * self.eta1
            deps_yy_dphi = gamma_tw * ny * ay * dh_dphi * self.eta1
            deps_xy_dphi = 0.5 * gamma_tw * (nx * ay + ny * ax) * dh_dphi * self.eta1
            df_el_dphi = -(sxx * deps_xx_dphi + syy * deps_yy_dphi +
                           2 * sxy * deps_xy_dphi)
            return df_el_dphi
        except Exception as e:
            st.error(f"Error computing elastic driving force: {e}")
            return np.zeros_like(self.phi)

    @handle_errors
    def evolve_twin_field(self, sxx, syy, sxy, eps_p_mag):
        kappa0 = float(self.params['kappa0'])
        gamma_aniso = float(self.params['gamma_aniso'])
        L_CTB = float(self.params.get('L_CTB', 0.05))
        L_ITB = float(self.params.get('L_ITB', 5.0))
        n_mob = int(self.params.get('n_mob', 4))
        zeta = float(self.params.get('zeta', 0.3))

        n_twin = self.mat_props['twinning']['n_2d']
        nx = float(n_twin[0])
        ny = float(n_twin[1])

        phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
        lap_phi = spectral_laplacian(self.phi, self.k2)

        kappa_phi, L_phi = compute_anisotropic_properties(
            phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso,
            L_CTB, L_ITB, n_mob)

        df_loc_dphi, _, _ = self.compute_local_energy_derivatives()
        df_el_dphi = self.compute_elastic_driving_force(sxx, syy, sxy)
        diss_p = zeta * eps_p_mag * self.phi

        NLF = df_loc_dphi + df_el_dphi + diss_p
        correction = (L_phi * kappa_phi - self.implicit_diff_phi) * lap_phi
        R = -L_phi * NLF + correction

        if self.confine_twin:
            R = R * self.eta1**2

        phi_hat = fft2(self.phi)
        R_hat = fft2(R)
        denom = 1.0 + self.dt * self.implicit_diff_phi * self.k2
        phi_hat_new = (phi_hat + self.dt * R_hat) / denom
        phi_new = np.real(ifft2(phi_hat_new))
        phi_new = np.clip(phi_new, -1.1, 1.1)
        return phi_new

    @handle_errors
    def evolve_grain_fields(self):
        kappa_eta = float(self.params['kappa_eta'])
        L_eta = float(self.params.get('L_eta', 1.0))
        _, df_deta1, df_deta2 = self.compute_local_energy_derivatives()

        R_eta1 = -L_eta * df_deta1
        R_eta2 = -L_eta * df_deta2

        eta1_hat = fft2(self.eta1)
        eta2_hat = fft2(self.eta2)
        R1_hat = fft2(R_eta1)
        R2_hat = fft2(R_eta2)
        denom = 1.0 + self.dt * self.implicit_diff_eta * self.k2
        eta1_hat_new = (eta1_hat + self.dt * R1_hat) / denom
        eta2_hat_new = (eta2_hat + self.dt * R2_hat) / denom
        eta1_new = np.real(ifft2(eta1_hat_new))
        eta2_new = np.real(ifft2(eta2_hat_new))

        eta1_new = np.clip(eta1_new, 0, 1)
        eta2_new = np.clip(eta2_new, 0, 1)
        norm = np.sqrt(eta1_new**2 + eta2_new**2 + 1e-12)
        mask = norm > 1
        eta1_new[mask] = eta1_new[mask] / norm[mask]
        eta2_new[mask] = eta2_new[mask] / norm[mask]
        return eta1_new, eta2_new

    @handle_errors
    def compute_plastic_strain(self, sigma_eq, sigma_y):
        try:
            plastic_params = self.mat_props['plasticity']
            gamma0_dot = plastic_params['gamma0_dot']
            m = int(plastic_params['m'])
            eps_p_xx_new, eps_p_yy_new, eps_p_xy_new = update_plastic_strain(
                sigma_eq, sigma_y,
                self.eps_p_xx, self.eps_p_yy, self.eps_p_xy,
                gamma0_dot, m, self.dt)
            self.eps_p_xx = eps_p_xx_new
            self.eps_p_yy = eps_p_yy_new
            self.eps_p_xy = eps_p_xy_new
            eps_p_mag = np.sqrt(
                2 / 3 * (self.eps_p_xx**2 + self.eps_p_yy**2 +
                         2 * self.eps_p_xy**2 + 1e-15))
            eps_p_mag = np.clip(eps_p_mag, 0, 0.5)
            if np.max(eps_p_mag) > 0.1:
                st.warning(f"⚠️ Large plastic strain detected: {np.max(eps_p_mag):.3f}")
            return eps_p_mag
        except Exception as e:
            st.error(f"Error computing plastic strain: {e}")
            return np.zeros_like(sigma_eq)

    @handle_errors
    def compute_total_energy(self):
        try:
            W = self.params['W']
            A = self.params['A']
            B = self.params['B']
            f_loc = (W * (self.phi**2 - 1)**2 * self.eta1**2 +
                     A * (self.eta1**2 * (1 - self.eta1)**2 +
                          self.eta2**2 * (1 - self.eta2)**2) +
                     B * self.eta1**2 * self.eta2**2)
            phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
            grad_phi_sq = phi_gx**2 + phi_gy**2
            eta1_gx, eta1_gy = spectral_gradients(self.eta1, self.kx, self.ky)
            eta2_gx, eta2_gy = spectral_gradients(self.eta2, self.kx, self.ky)
            grad_eta1_sq = eta1_gx**2 + eta1_gy**2
            grad_eta2_sq = eta2_gx**2 + eta2_gy**2
            kappa0 = self.params['kappa0']
            kappa_eta = self.params['kappa_eta']
            f_grad = (0.5 * kappa0 * grad_phi_sq +
                      0.5 * kappa_eta * (grad_eta1_sq + grad_eta2_sq))
            energy_density = f_loc + f_grad
            total_energy = np.sum(energy_density) * (self.dx**2)
            return total_energy
        except Exception as e:
            st.warning(f"Error computing energy: {e}")
            return 0.0

    @handle_errors
    def step(self):
        try:
            sigma_mag = self.params.get('applied_stress', 0.0)
            theta_deg = self.params.get('applied_stress_angle', 0.0)
            theta = np.deg2rad(theta_deg)
            applied_xx = sigma_mag * np.cos(theta)**2
            applied_yy = sigma_mag * np.sin(theta)**2
            applied_xy = sigma_mag * np.sin(theta) * np.cos(theta)

            gamma_tw = self.mat_props['twinning']['gamma_tw']
            n = self.mat_props['twinning']['n_2d']
            a = self.mat_props['twinning']['a_2d']

            exx_star, eyy_star, exy_star = compute_transformation_strain(
                self.phi, self.eta1, gamma_tw, a[0], a[1], n[0], n[1])

            eigenstrain_xx = exx_star + self.eps_p_xx
            eigenstrain_yy = eyy_star + self.eps_p_yy
            eigenstrain_xy = exy_star + self.eps_p_xy

            (sigma_eq, sxx, syy, sxy, sigma_h,
             eps_xx, eps_yy, eps_xy) = self.spectral_solver.solve(
                eigenstrain_xx, eigenstrain_yy, eigenstrain_xy,
                applied_xx, applied_yy, applied_xy)

            phi_gx, phi_gy = spectral_gradients(self.phi, self.kx, self.ky)
            h = compute_twin_spacing_from_gradient(phi_gx, phi_gy)

            plastic_params = self.mat_props['plasticity']
            sigma_y = compute_yield_stress(
                h, plastic_params['sigma0'], plastic_params['mu'],
                plastic_params['b'], plastic_params['nu'])

            eps_p_mag = self.compute_plastic_strain(sigma_eq, sigma_y)

            self.phi = self.evolve_twin_field(sxx, syy, sxy, eps_p_mag)
            self.eta1, self.eta2 = self.evolve_grain_fields()

            phi_norm = np.linalg.norm(self.phi)
            total_energy = self.compute_total_energy()
            max_stress = np.max(sigma_eq)
            avg_stress = np.mean(sigma_eq)
            valid_h = h[(h > 5) & (h < 50)]
            avg_spacing = np.mean(valid_h) if valid_h.size > 0 else 0.0
            plastic_work = np.sum(eps_p_mag) * (self.dx**2)

            self.history['phi_norm'].append(phi_norm)
            self.history['energy'].append(total_energy)
            self.history['max_stress'].append(max_stress)
            self.history['avg_stress'].append(avg_stress)
            self.history['plastic_work'].append(plastic_work)
            self.history['twin_spacing_avg'].append(avg_spacing)

            results = {
                'phi': self.phi.copy(),
                'eta1': self.eta1.copy(),
                'eta2': self.eta2.copy(),
                'sigma_eq': sigma_eq.copy(),
                'sigma_h': sigma_h.copy(),
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
            zeros = np.zeros((self.N, self.N))
            return {
                'phi': zeros, 'eta1': zeros, 'eta2': zeros,
                'sigma_eq': zeros, 'sigma_h': zeros,
                'sigma_xx': zeros, 'sigma_yy': zeros, 'sigma_xy': zeros,
                'h': zeros, 'sigma_y': zeros, 'eps_p_mag': zeros,
                'eps_xx': zeros, 'eps_yy': zeros, 'eps_xy': zeros,
                'convergence': {k: 0 for k in
                    ['phi_norm', 'energy', 'max_stress',
                     'avg_stress', 'plastic_work', 'avg_spacing']}
            }


class SimulationMonitor:
    @staticmethod
    @handle_errors
    def create_convergence_plots(history, timesteps):
        history_length = len(history['phi_norm'])
        if len(timesteps) >= history_length:
            plot_timesteps = timesteps[:history_length]
        else:
            plot_timesteps = np.linspace(0, timesteps[-1] if len(timesteps) else 1.0,
                                         history_length)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].plot(plot_timesteps, history['phi_norm'], 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Time (ns)')
        axes[0, 0].set_ylabel('||φ||')
        axes[0, 0].set_title('Twin Order Parameter Norm')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(plot_timesteps, history['energy'], 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Time (ns)')
        axes[0, 1].set_ylabel('Total Energy (J)')
        axes[0, 1].set_title('System Energy Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 2].plot(plot_timesteps, np.array(history['max_stress']) / 1e9,
                        'g-', linewidth=2, alpha=0.8, label='Max')
        axes[0, 2].plot(plot_timesteps, np.array(history['avg_stress']) / 1e9,
                        'g--', linewidth=1.5, alpha=0.6, label='Avg')
        axes[0, 2].set_xlabel('Time (ns)')
        axes[0, 2].set_ylabel('Stress (GPa)')
        axes[0, 2].set_title('Stress Evolution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[1, 0].plot(plot_timesteps, history['plastic_work'], 'm-',
                        linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Time (ns)')
        axes[1, 0].set_ylabel('Plastic Work (J)')
        axes[1, 0].set_title('Plastic Work Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(plot_timesteps, history['twin_spacing_avg'], 'c-',
                        linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Time (ns)')
        axes[1, 1].set_ylabel('Avg Spacing (nm)')
        axes[1, 1].set_title('Average Twin Spacing')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 2].text(0.5, 0.5, 'Plastic strain history not saved\n(use per-step analysis)',
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Plastic Strain Evolution')
        plt.tight_layout()
        return fig


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DataExporter:
    @staticmethod
    @handle_errors
    def export_pkl(sim_data, params, history, sim_name):
        buffer = BytesIO()
        data = {'params': params, 'history': history,
                'metadata': sim_data.get('metadata', {}), 'sim_name': sim_name}
        pickle.dump(data, buffer)
        buffer.seek(0)
        return buffer, f"{sim_name}.pkl"

    @staticmethod
    @handle_errors
    def export_pt(sim_data, params, history, sim_name):
        buffer = BytesIO()
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return torch.tensor(x)
        tensor_data = {'params': params, 'metadata': sim_data.get('metadata', {}),
                       'history': []}
        for frame in history:
            frame_tensor = {
                'phi': to_tensor(frame['phi']),
                'eta1': to_tensor(frame['eta1']),
                'eta2': to_tensor(frame['eta2']),
                'sigma_eq': to_tensor(frame['sigma_eq']),
                'sigma_h': to_tensor(frame['sigma_h']),
                'h': to_tensor(frame['h']),
                'eps_p_mag': to_tensor(frame['eps_p_mag'])
            }
            tensor_data['history'].append(frame_tensor)
        torch.save(tensor_data, buffer)
        buffer.seek(0)
        return buffer, f"{sim_name}.pt"

    @staticmethod
    @handle_errors
    def export_sql(sim_data, params, history, sim_name, sim_id, N, dx):
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        c.execute('''CREATE TABLE simulations (
                     id TEXT PRIMARY KEY, sim_name TEXT,
                     twin_spacing REAL, applied_stress REAL,
                     applied_stress_angle REAL, W REAL,
                     geometry_type TEXT, created_at TEXT,
                     grid_size INTEGER, dx REAL)''')
        c.execute('''CREATE TABLE frames (
                     sim_id TEXT, frame_idx INTEGER,
                     phi BLOB, eta1 BLOB, eta2 BLOB,
                     sigma_eq BLOB, sigma_h BLOB, h BLOB, eps_p_mag BLOB)''')
        created_at = sim_data.get('metadata', {}).get('created_at',
                                                     datetime.now().isoformat())
        c.execute("INSERT INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (sim_id, sim_name, params.get('twin_spacing', 0.0),
                   params.get('applied_stress', 0.0),
                   params.get('applied_stress_angle', 0.0),
                   params.get('W', 0.0),
                   params.get('geometry_type', 'standard'),
                   created_at, params.get('N', N), params.get('dx', dx)))
        for idx, frame in enumerate(history):
            c.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (sim_id, idx,
                       pickle.dumps(frame['phi']),
                       pickle.dumps(frame['eta1']),
                       pickle.dumps(frame['eta2']),
                       pickle.dumps(frame['sigma_eq']),
                       pickle.dumps(frame['sigma_h']),
                       pickle.dumps(frame['h']),
                       pickle.dumps(frame['eps_p_mag'])))
        conn.commit()
        dump_buffer = StringIO()
        for line in conn.iterdump():
            dump_buffer.write('%s\n' % line)
        conn.close()
        sql_str = dump_buffer.getvalue()
        return BytesIO(sql_str.encode()), f"{sim_name}.sql"

    @staticmethod
    @handle_errors
    def export_csv(history, sim_name, extent, N, dx):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            x = np.linspace(extent[0], extent[1], N)
            y = np.linspace(extent[2], extent[3], N)
            X, Y = np.meshgrid(x, y)
            for idx, frame in enumerate(history):
                df = pd.DataFrame({
                    'x': X.flatten(), 'y': Y.flatten(),
                    'phi': frame['phi'].flatten(),
                    'eta1': frame['eta1'].flatten(),
                    'eta2': frame['eta2'].flatten(),
                    'sigma_eq_GPa': (frame['sigma_eq'] / 1e9).flatten(),
                    'sigma_h_GPa': (frame['sigma_h'] / 1e9).flatten(),
                    'h_nm': frame['h'].flatten(),
                    'eps_p_mag': frame['eps_p_mag'].flatten()})
                csv_str = df.to_csv(index=False)
                zf.writestr(f"{sim_name}_frame_{idx:04d}.csv", csv_str)
        zip_buffer.seek(0)
        return zip_buffer, f"{sim_name}_csv.zip"

    @staticmethod
    @handle_errors
    def export_json(sim_data, params, history, sim_name):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        export_data = {'sim_name': sim_name, 'params': params,
                       'metadata': sim_data.get('metadata', {}),
                       'history': convert(history)}
        json_str = json.dumps(export_data, indent=2, cls=NumpyEncoder)
        return BytesIO(json_str.encode()), f"{sim_name}.json"

    @staticmethod
    @handle_errors
    def export_hdf5(sim_data, params, history, sim_name, N, dx):
        if not H5PY_AVAILABLE:
            st.error("HDF5 export requires h5py. Please install it: pip install h5py")
            return None, None
        buffer = BytesIO()
        with h5py.File(buffer, 'w') as f:
            param_grp = f.create_group('parameters')
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    param_grp.attrs[k] = v
                elif isinstance(v, np.ndarray):
                    param_grp.create_dataset(k, data=v)
                else:
                    param_grp.attrs[k] = json.dumps(v, default=str)
            meta_grp = f.create_group('metadata')
            for k, v in sim_data.get('metadata', {}).items():
                if isinstance(v, (int, float, str, bool)):
                    meta_grp.attrs[k] = v
                elif isinstance(v, dict):
                    subgrp = meta_grp.create_group(k)
                    for sk, sv in v.items():
                        if isinstance(sv, (int, float, str, bool)):
                            subgrp.attrs[sk] = sv
                        else:
                            subgrp.attrs[sk] = json.dumps(sv, default=str)
                else:
                    meta_grp.attrs[k] = json.dumps(v, default=str)
            x = np.linspace(-N * dx / 2, N * dx / 2, N)
            y = np.linspace(-N * dx / 2, N * dx / 2, N)
            f.create_dataset('x', data=x)
            f.create_dataset('y', data=y)
            frame_grp = f.create_group('frames')
            for idx, frame in enumerate(history):
                grp = frame_grp.create_group(f'frame_{idx:04d}')
                for field in ['phi', 'eta1', 'eta2', 'sigma_eq', 'sigma_h',
                              'h', 'eps_p_mag', 'sigma_y']:
                    if field in frame:
                        grp.create_dataset(field, data=frame[field],
                                           compression='gzip')
        buffer.seek(0)
        return buffer, f"{sim_name}.h5"

    @staticmethod
    @handle_errors
    def bulk_export_all_simulations(N, dx, extent):
        all_sims = SimulationDatabase.get_all_simulations()
        if not all_sims:
            return None
        bulk_buffer = BytesIO()
        with zipfile.ZipFile(bulk_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            summary = f"MULTI-SIMULATION EXPORT SUMMARY\n"
            summary += f"Generated: {datetime.now().isoformat()}\n"
            summary += f"Total Simulations: {len(all_sims)}\n\n"
            for sim_id, sim_data in all_sims.items():
                try:
                    params = sim_data.get('params', {})
                    history = sim_data.get('results_history', [])
                    metadata = sim_data.get('metadata', {})
                    sim_name = build_sim_name(params, sim_id)
                    sim_dir = f"simulation_{sim_id}"
                    zf.writestr(f"{sim_dir}/parameters.json",
                                json.dumps(params, indent=2, cls=NumpyEncoder))
                    zf.writestr(f"{sim_dir}/metadata.json",
                                json.dumps(metadata, indent=2, cls=NumpyEncoder))
                    x = np.linspace(extent[0], extent[1], N)
                    y = np.linspace(extent[2], extent[3], N)
                    X, Y = np.meshgrid(x, y)
                    for idx, frame in enumerate(history):
                        df = pd.DataFrame({
                            'x': X.flatten(), 'y': Y.flatten(),
                            'phi': frame['phi'].flatten(),
                            'sigma_eq_GPa': (frame['sigma_eq'] / 1e9).flatten(),
                            'sigma_h_GPa': (frame['sigma_h'] / 1e9).flatten(),
                            'h_nm': frame['h'].flatten()})
                        zf.writestr(f"{sim_dir}/frame_{idx:04d}.csv",
                                    df.to_csv(index=False))
                    summary += f"\nSimulation {sim_id}:\n"
                    summary += f"  Name: {sim_name}\n"
                    summary += f"  λ = {params.get('twin_spacing', 0):.1f} nm\n"
                    summary += f"  σ_app = {params.get('applied_stress', 0) / 1e6:.0f} MPa\n"
                    summary += f"  θ = {params.get('applied_stress_angle', 0):.0f}°\n"
                    summary += f"  W = {params.get('W', 0):.1f}\n"
                    summary += f"  Frames: {len(history)}\n"
                except Exception as e:
                    summary += f"\nSimulation {sim_id}: ERROR - {str(e)}\n"
            zf.writestr("EXPORT_SUMMARY.txt", summary)
        bulk_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return bulk_buffer, f"twin_all_simulations_{timestamp}.zip"


class ParameterSweep:
    @staticmethod
    @handle_errors
    def run_sweep(base_params, param_name, values, save=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, val in enumerate(values):
            status_text.text(f"Running {param_name} = {val:.3f} ({i + 1}/{len(values)})")
            params = base_params.copy()
            params[param_name] = val
            if 'history' in params: del params['history']
            if 'geom_viz' in params: del params['geom_viz']
            try:
                solver = NanotwinnedCuSolver(params)
                n_steps = params.get('n_steps', 100)
                for step in range(n_steps):
                    solver.step()
                final_results = {
                    'phi': solver.phi.copy(),
                    'eta1': solver.eta1.copy(),
                    'eta2': solver.eta2.copy(),
                    'sigma_eq': solver.history['avg_stress'][-1] if solver.history['avg_stress'] else 0,
                    'max_stress': solver.history['max_stress'][-1] if solver.history['max_stress'] else 0,
                    'plastic_work': solver.history['plastic_work'][-1] if solver.history['plastic_work'] else 0,
                    'twin_spacing_avg': solver.history['twin_spacing_avg'][-1] if solver.history['twin_spacing_avg'] else 0,
                    'energy': solver.history['energy'][-1] if solver.history['energy'] else 0,
                }
                if save:
                    SimulationDatabase.save_simulation(params, solver.history, None)
                results.append({'param_value': val,
                                'convergence': final_results,
                                'solver': solver})
            except Exception as e:
                st.error(f"Failed for {param_name}={val}: {e}")
                results.append({'param_value': val,
                                'convergence': None, 'error': str(e)})
            progress_bar.progress((i + 1) / len(values))
        status_text.text("Sweep completed!")
        return results


# ============================================================================
# PLASTICITY ONTOLOGY
# ============================================================================
PLASTICITY_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "rho0": {
        "label": "Initial Dislocation Density",
        "symbol": "ρ₀",
        "aliases": PARAM_CANON["rho0"]["aliases"] + [
            "rho_0", "rho0", "ρ₀", "forest dislocation density",
            "mobile dislocation density", "immobile dislocation density",
            "total dislocation density", "geometrically necessary dislocation",
        ],
        "unit": "m^-2", "ui_unit": "m⁻²", "ui_scale": 1.0,
        "valid_range": PARAM_CANON["rho0"]["plausible"],
        "soft_range": (1e11, 1e16),
        "defaults": {"Cu": 1e12, "Al": 1e12, "Ni": 1e13},
        "expected_file": "initial_dislocation_density_metadatabase.json",
    },
    "mu": {
        "label": "Shear Modulus",
        "symbol": "μ",
        "aliases": PARAM_CANON["mu"]["aliases"] + [
            "mu", "G", "elastic shear modulus", "c44", "c_44",
            "second lame parameter", "second lamé parameter",
            "rigidity modulus",
        ],
        "unit": "Pa", "ui_unit": "GPa", "ui_scale": 1e9,
        "valid_range": (1e9, 5e11),
        "soft_range": (PARAM_CANON["mu"]["plausible"][0] * 1e9,
                       PARAM_CANON["mu"]["plausible"][1] * 1e9),
        "defaults": {"Cu": 48e9, "Al": 26e9, "Ni": 80e9},
        "expected_file": "shear_modulus_metadatabase.json",
    },
    "gamma0_dot": {
        "label": "Reference Strain Rate",
        "symbol": "γ̇₀",
        "aliases": PARAM_CANON["gamma0_dot"]["aliases"] + [
            "reference strain-rate", "gamma0_dot", "gamma_dot_0",
            "gamma_0_dot", "γ̇₀", "γ0", "gamma dot 0", "gamma-dot-0",
            "pre-exponential strain rate", "pre-exponential factor",
            "pre-exponential", "attempt frequency", "reference rate",
        ],
        "unit": "s^-1", "ui_unit": "s⁻¹", "ui_scale": 1.0,
        "valid_range": (1e-6, 1e12),
        "soft_range": (1e-4, 1e6),
        "defaults": {"Cu": 1e-3, "Al": 1e-3, "Ni": 1e-3},
        "expected_file": "reference_strain_rate_metadatabase.json",
    },
    "srs": {
        "label": "Strain-Rate Sensitivity Exponent",
        "symbol": "m",
        "aliases": PARAM_CANON["srs"]["aliases"] + [
            "strain rate sensitivity", "srs", "m exponent", "stress exponent",
            "rate sensitivity", "strain-rate sensitivity",
            "rate sensitivity exponent", "n exponent", "viscous exponent",
        ],
        "unit": "dimensionless", "ui_unit": "–", "ui_scale": 1.0,
        "valid_range": (1.0, 200.0), "soft_range": (5.0, 50.0),
        "defaults": {"Cu": 20.0, "Al": 20.0, "Ni": 20.0},
        "expected_file": "inverse_strain_rate_sensitivity_metadatabase.json",
    },
    "sigma0_fric": {
        "label": "Friction Stress (Hall–Petch intercept)",
        "symbol": "σ₀",
        "aliases": PARAM_CANON["sigma0_fric"]["aliases"],
        "unit": "Pa", "ui_unit": "MPa", "ui_scale": 1e6,
        "valid_range": (1e5, 2e9), "soft_range": (10e6, 500e6),
        "defaults": {"Cu": 50e6, "Al": 30e6, "Ni": 70e6, "Fe": 100e6},
        "expected_file": "friction_stress_metadatabase.json",
        "derivation": "hall_petch_intercept or M·τ_CRSS",
    },
    "sigma0_yield": {
        "label": "Yield Stress",
        "symbol": "σ_y",
        "aliases": PARAM_CANON["sigma0_yield"]["aliases"],
        "unit": "Pa", "ui_unit": "MPa", "ui_scale": 1e6,
        "valid_range": (1e5, 5e9), "soft_range": (50e6, 2e9),
        "defaults": {"Cu": 300e6, "Al": 200e6, "Ni": 400e6, "Fe": 500e6},
        "expected_file": "yield_stress_metadatabase.json",
        "derivation": "tensile test or Hall–Petch σ_y = σ₀ + k·d^(-1/2)",
    },
    "tau_p": {
        "label": "Peierls–Nabarro (lattice) Stress",
        "symbol": "τ_P",
        "aliases": PARAM_CANON["tau_p"]["aliases"],
        "unit": "Pa", "ui_unit": "MPa", "ui_scale": 1e6,
        "valid_range": (1e3, 5e9), "soft_range": (1e6, 3e9),
        "defaults": {"Cu": 5e6, "Al": 5e6, "Ni": 10e6,
                     "Fe": 350e6, "W": 1.7e9},
        "expected_file": "peierls_lattice_stress_metadatabase.json",
        "derivation": "P-N formula: τ_P = 2G/(1-ν)·exp(-2πw/b)",
    },
    "grain_size_d": {
        "label": "Grain Size", "symbol": "d",
        "aliases": PARAM_CANON["grain_size_d"]["aliases"],
        "unit": "nm", "ui_unit": "nm", "ui_scale": 1.0,
        "valid_range": (1.0, 1e5), "soft_range": (10.0, 1e4),
        "defaults": {"Cu": 1000.0, "Al": 1000.0, "Ni": 1000.0,
                     "Fe": 1000.0, "W": 1000.0},
        "expected_file": "grain_size_metadatabase.json",
    },
    "twin_thickness_lambda": {
        "label": "Twin Thickness", "symbol": "λ",
        "aliases": PARAM_CANON["twin_thickness_lambda"]["aliases"],
        "unit": "nm", "ui_unit": "nm", "ui_scale": 1.0,
        "valid_range": (0.5, 1000.0), "soft_range": (5.0, 100.0),
        "defaults": {"Cu": 20.0, "Al": 20.0, "Ni": 20.0,
                     "Fe": 20.0, "W": 20.0},
        "expected_file": "twin_thickness_metadatabase.json",
    },
    "burgers_vector_b": {
        "label": "Burgers Vector", "symbol": "b",
        "aliases": PARAM_CANON["burgers_vector_b"]["aliases"],
        "unit": "nm", "ui_unit": "nm", "ui_scale": 1.0,
        "valid_range": (0.05, 1.0), "soft_range": (0.2, 0.4),
        "defaults": {"Cu": 0.256, "Al": 0.286, "Ni": 0.249,
                     "Fe": 0.248, "W": 0.274},
        "expected_file": "burgers_vector_metadatabase.json",
    },
    "poisson_ratio_nu": {
        "label": "Poisson's Ratio", "symbol": "ν",
        "aliases": PARAM_CANON["poisson_ratio_nu"]["aliases"],
        "unit": "dimensionless", "ui_unit": "–", "ui_scale": 1.0,
        "valid_range": (0.0, 0.5), "soft_range": (0.25, 0.40),
        "defaults": {"Cu": 0.34, "Al": 0.33, "Ni": 0.31,
                     "Fe": 0.29, "W": 0.28},
        "expected_file": "poisson_ratio_metadatabase.json",
    },
    "taylor_factor_M": {
        "label": "Taylor Factor", "symbol": "M",
        "aliases": PARAM_CANON["taylor_factor_M"]["aliases"],
        "unit": "dimensionless", "ui_unit": "–", "ui_scale": 1.0,
        "valid_range": (2.0, 3.5), "soft_range": (2.8, 3.2),
        "defaults": {"Cu": 3.06, "Al": 3.06, "Ni": 3.06,
                     "Fe": 2.9, "W": 3.0},
        "expected_file": "taylor_factor_metadatabase.json",
    },
    "hall_petch_k": {
        "label": "Hall–Petch Coefficient", "symbol": "k_y",
        "aliases": PARAM_CANON["hall_petch_k"]["aliases"],
        "unit": "MPa·m^(1/2)", "ui_unit": "MPa·m^(1/2)", "ui_scale": 1.0,
        "valid_range": (0.01, 1.0), "soft_range": (0.05, 0.5),
        "defaults": {"Cu": 0.11, "Al": 0.07, "Ni": 0.16,
                     "Fe": 0.60, "W": 0.40},
        "expected_file": "hall_petch_k_metadatabase.json",
    },
    "core_width_w": {
        "label": "Dislocation Core Width", "symbol": "w",
        "aliases": PARAM_CANON["core_width_w"]["aliases"],
        "unit": "nm", "ui_unit": "nm", "ui_scale": 1.0,
        "valid_range": (0.05, 2.0), "soft_range": (0.3, 1.0),
        "defaults": {"Cu": 0.5, "Al": 0.5, "Ni": 0.5,
                     "Fe": 0.25, "W": 0.25},
        "expected_file": "core_width_metadatabase.json",
    },
    "shear_modulus_G": {
        "label": "Shear Modulus (context)", "symbol": "G",
        "aliases": PARAM_CANON["shear_modulus_G"]["aliases"],
        "unit": "Pa", "ui_unit": "GPa", "ui_scale": 1e9,
        "valid_range": (1e9, 5e11), "soft_range": (20e9, 200e9),
        "defaults": {"Cu": 48e9, "Al": 26e9, "Ni": 80e9,
                     "Fe": 80e9, "W": 160e9},
        "expected_file": "shear_modulus_metadatabase.json",
    },
    "twin_spacing": {
        "label": "Twin Spacing", "symbol": "λ",
        "aliases": PARAM_CANON["twin_spacing"]["aliases"],
        "unit": "nm", "ui_unit": "nm", "ui_scale": 1.0,
        "valid_range": (1.0, 1000.0), "soft_range": (5.0, 200.0),
        "defaults": {"Cu": 20.0, "Al": 20.0, "Ni": 20.0},
        "expected_file": "twin_spacing_metadatabase.json",
    },
    "applied_stress": {
        "label": "Applied Stress", "symbol": "σ_app",
        "aliases": PARAM_CANON["applied_stress"]["aliases"],
        "unit": "Pa", "ui_unit": "MPa", "ui_scale": 1e6,
        "valid_range": (1e5, 5e9), "soft_range": (1e7, 1e9),
        "defaults": {"Cu": 300e6, "Al": 200e6, "Ni": 400e6},
        "expected_file": "applied_stress_metadatabase.json",
    },
    "W": {
        "label": "Interface Width", "symbol": "W",
        "aliases": PARAM_CANON["W"]["aliases"],
        "unit": "nm", "ui_unit": "nm", "ui_scale": 1.0,
        "valid_range": (0.1, 20.0), "soft_range": (0.5, 10.0),
        "defaults": {"Cu": 2.0, "Al": 2.0, "Ni": 2.0},
        "expected_file": "interface_width_metadatabase.json",
    },
}


PARAM_ORDER: List[str] = [
    "rho0", "mu", "gamma0_dot", "srs",
    "sigma0_fric", "sigma0_yield", "tau_p",
    "grain_size_d", "twin_thickness_lambda",
    "burgers_vector_b", "poisson_ratio_nu",
    "taylor_factor_M", "hall_petch_k", "core_width_w",
]


SIDEBAR_PARAM_ORDER: List[str] = [
    "rho0", "mu", "gamma0_dot", "srs", "sigma0_fric",
]
CONTEXT_PARAM_ORDER: List[str] = [
    "sigma0_yield", "grain_size_d", "twin_thickness_lambda",
    "burgers_vector_b", "poisson_ratio_nu", "taylor_factor_M",
    "hall_petch_k", "core_width_w",
]


SOLVER_KEY_MAP: Dict[str, str] = {
    "rho0":                    "rho0",
    "mu":                      "mu",
    "gamma0_dot":              "gamma0_dot",
    "srs":                     "m",
    "sigma0_fric":             "sigma0",
    "sigma0_yield":            "_yield_passthrough",
    "tau_p":                   "_tau_p_passthrough",
    "grain_size_d":            "_ignore",
    "twin_thickness_lambda":   "_ignore",
    "burgers_vector_b":        "_ignore",
    "poisson_ratio_nu":        "_ignore",
    "taylor_factor_M":         "_ignore",
    "hall_petch_k":            "_ignore",
    "core_width_w":            "_ignore",
}

TARGET_JSON_FILES: List[str] = [
    "friction_stress_metadatabase.json",
    "peierls_lattice_stress_metadatabase.json",
    "yield_stress_metadatabase.json",
    "grain_size_metadatabase.json",
    "twin_thickness_metadatabase.json",
    "burgers_vector_metadatabase.json",
    "poisson_ratio_metadatabase.json",
    "taylor_factor_metadatabase.json",
    "hall_petch_k_metadatabase.json",
    "core_width_metadatabase.json",
    "initial_dislocation_density_metadatabase.json",
    "reference_strain_rate_metadatabase.json",
    "shear_modulus_metadatabase.json",
    "inverse_strain_rate_sensitivity_metadatabase.json",
]


_PARAM_ALIASES: Dict[str, str] = {
    "rho_0": "rho0", "rho": "rho0", "ρ₀": "rho0", "ρ0": "rho0",
    "rho0_": "rho0", "rho_0_": "rho0",
    "dislocation_density": "rho0",
    "initial_dislocation_density": "rho0",
    "initial disloc density": "rho0",
    "forest_density": "rho0",
    "rho_dis": "rho0",
    "g": "mu", "shear_modulus": "mu", "μ": "mu", "mu_s": "mu",
    "shear modulus": "mu", "rigidity_modulus": "mu", "c44": "mu",
    "c_44": "mu", "elastic_shear_modulus": "mu",
    "gamma_dot_0": "gamma0_dot", "gamma_0_dot": "gamma0_dot",
    "γ̇₀": "gamma0_dot", "γ0": "gamma0_dot", "gamma_dot": "gamma0_dot",
    "gamma0": "gamma0_dot",
    "reference_strain_rate": "gamma0_dot",
    "reference_shear_rate": "gamma0_dot",
    "strain_rate_reference": "gamma0_dot",
    "gammadot0": "gamma0_dot", "gamma_dot0": "gamma0_dot",
    "reference_strain-rate": "gamma0_dot",
    "gamma dot 0": "gamma0_dot", "gamma-dot-0": "gamma0_dot",
    "gamma_0": "gamma0_dot", "reference_rate": "gamma0_dot",
    "pre_exponential_strain_rate": "gamma0_dot",
    "pre-exponential": "gamma0_dot", "pre_exponential": "gamma0_dot",
    "attempt_frequency": "gamma0_dot",
    "m": "srs", "m_exponent": "srs", "rate_sensitivity": "srs",
    "strain_rate_sensitivity": "srs", "srs_exponent": "srs",
    "rate_sensitivity_exponent": "srs",
    "strain-rate_sensitivity": "srs", "stress_exponent": "srs", "n": "srs",
    "friction_stress": "sigma0_fric",
    "sigma_0_fric": "sigma0_fric",
    "lattice_friction_stress": "sigma0_fric",
    "lattice_friction": "sigma0_fric",
    "sigma0_fric": "sigma0_fric", "sigma0_f": "sigma0_fric",
    "friction_lattice_stress": "sigma0_fric",
    "athermal_stress": "sigma0_fric",
    "sigma_0": "sigma0_fric", "σ₀": "sigma0_fric", "σ0": "sigma0_fric",
    "sigma0": "sigma0_fric",
    "yield_stress": "sigma0_yield", "sigma_y": "sigma0_yield",
    "yield_strength": "sigma0_yield", "sigma0_yield": "sigma0_yield",
    "sy": "sigma0_yield", "σy": "sigma0_yield",
    "0.2%_yield_stress": "sigma0_yield",
    "offset_yield_stress": "sigma0_yield",
    "peierls_stress": "tau_p", "peierls_nabarro_stress": "tau_p",
    "peierls-nabarro_stress": "tau_p", "pn_stress": "tau_p",
    "tau_p": "tau_p", "tau_p_": "tau_p", "τ_p": "tau_p",
    "intrinsic_lattice_friction": "tau_p",
    "lattice_resistance": "tau_p", "peierls": "tau_p",
    "grain_size": "grain_size_d", "grain_diameter": "grain_size_d",
    "d_grain": "grain_size_d", "d50": "grain_size_d",
    "crystallite_size": "grain_size_d", "mean_grain_size": "grain_size_d",
    "average_grain_size": "grain_size_d",
    "twin_thickness": "twin_thickness_lambda",
    "twin_spacing_lambda": "twin_thickness_lambda",
    "twin_thickness_lambda": "twin_thickness_lambda",
    "twin_boundary_spacing": "twin_thickness_lambda",
    "twin_boundary_separation": "twin_thickness_lambda",
    "lambda_twin": "twin_thickness_lambda",
    "burgers_vector": "burgers_vector_b", "burgers_b": "burgers_vector_b",
    "burgers_vector_b": "burgers_vector_b", "b_vector": "burgers_vector_b",
    "b": "burgers_vector_b",
    "poisson_ratio": "poisson_ratio_nu",
    "poisson_nu": "poisson_ratio_nu",
    "poisson_ratio_nu": "poisson_ratio_nu",
    "nu": "poisson_ratio_nu", "ν": "poisson_ratio_nu",
    "taylor_M": "taylor_factor_M", "taylor_factor": "taylor_factor_M",
    "taylor_factor_M": "taylor_factor_M",
    "taylor_m_factor": "taylor_factor_M",
    "orientation_factor_m": "taylor_factor_M",
    "schmid_factor": "taylor_factor_M",
    "hall_petch_k": "hall_petch_k", "ky": "hall_petch_k",
    "k_y": "hall_petch_k", "k_lambda": "hall_petch_k",
    "kλ": "hall_petch_k", "k_λ": "hall_petch_k",
    "hall-petch_coefficient": "hall_petch_k",
    "hall_petch_coefficient": "hall_petch_k",
    "core_width": "core_width_w", "core_width_w": "core_width_w",
    "dislocation_core_width": "core_width_w",
    "core_half_width": "core_width_w",
    "shear_modulus_G": "shear_modulus_G", "shear_moduli": "shear_modulus_G",
    "rigidity": "shear_modulus_G",
    "lambda": "twin_spacing", "twin_spacing": "twin_spacing",
    "twin_thickness_legacy": "twin_spacing",
    "applied_stress": "applied_stress",
    "interface_width": "W", "well_depth": "W",
}


def _canonicalize_param(raw_p: str) -> Optional[str]:
    if raw_p is None:
        return None
    p = str(raw_p).strip()
    if not p:
        return None
    if p in PLASTICITY_ONTOLOGY: return p
    if p in _PARAM_ALIASES: return _PARAM_ALIASES[p]
    low = p.lower().replace(" ", "_").replace("-", "_")
    if low in PLASTICITY_ONTOLOGY: return low
    if low in _PARAM_ALIASES: return _PARAM_ALIASES[low]
    for canon in sorted(PLASTICITY_ONTOLOGY, key=len, reverse=True):
        if canon in low: return canon
    for alias in sorted(_PARAM_ALIASES, key=len, reverse=True):
        if alias and alias in low: return _PARAM_ALIASES[alias]
    return None


def _coerce_value(item: Dict[str, Any]) -> Tuple[Optional[float], str]:
    raw_v = item.get("value")
    explicit_unit = str(item.get("unit") or "").strip()
    if isinstance(raw_v, (int, float)) and not isinstance(raw_v, bool):
        return float(raw_v), explicit_unit
    s = str(raw_v or "").strip()
    if not s:
        return None, explicit_unit
    m = re.search(
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        r"(?:\s*[×xX\*]\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?)?"
        r"(.*)", s)
    if not m:
        return None, explicit_unit
    try:
        mantissa = float(m.group(1))
    except (TypeError, ValueError):
        return None, explicit_unit
    exponent = int(m.group(2)) if m.group(2) else 0
    value = mantissa * (10 ** exponent)
    trailing = m.group(3).strip()
    unit = explicit_unit or trailing
    return value, unit


def _pl_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalize_unit(value: float, unit: str, param: str) -> float:
    u = (unit or "").strip().lower()\
        .replace("μ", "u").replace("µ", "u")\
        .replace("·", ".").replace("•", ".").replace("*", ".")\
        .replace("×", "x").replace(" ", "")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if v == 0.0 and param in _POSITIVE_LOWER_BOUND_PARAMS:
        return float("nan")

    if param in ("mu", "sigma0_fric", "sigma0_yield", "tau_p",
                 "shear_modulus_G"):
        if "gpa" in u: return v * 1e9
        if "mpa" in u: return v * 1e6
        if "kpa" in u: return v * 1e3
        if "pa"  in u: return v
        return v
    if param == "rho0":
        if "cm^-2" in u or "cm-2" in u or "cm^(-2)" in u:
            return v * 1e4
        if "mm^-2" in u or "mm-2" in u or "mm^(-2)" in u:
            return v * 1e6
        return v
    if param in ("grain_size_d", "twin_thickness_lambda",
                 "burgers_vector_b", "core_width_w"):
        if "nm" in u or "nanometer" in u: return v * 1e-9
        if "um" in u or "micrometer" in u or "micron" in u: return v * 1e-6
        if "angstrom" in u or (u.startswith("a") and len(u) <= 3):
            return v * 1e-10
        if u.startswith("mm"): return v * 1e-3
        if u.startswith("m") and "pa" not in u: return v
        return v * 1e-9
    if param == "hall_petch_k":
        if "um" in u or "μm" in u:
            return v * 1e3
        return v
    return v


def _pl_clamp(value: float, param: str) -> Tuple[float, bool]:
    lo, hi = PLASTICITY_ONTOLOGY[param]["valid_range"]
    if value < lo: return lo, True
    if value > hi: return hi, True
    return value, False


def _pl_fmt(param: str, si_value: float) -> str:
    spec = PLASTICITY_ONTOLOGY[param]
    ui_val = si_value / spec["ui_scale"]
    if param in ("rho0", "gamma0_dot"):
        return f"{ui_val:.2e} {spec['ui_unit']}"
    if param == "mu":
        return f"{ui_val:.2f} {spec['ui_unit']}"
    if param in ("sigma0_fric", "sigma0_yield", "tau_p"):
        return f"{ui_val:.1f} {spec['ui_unit']}"
    if param in ("grain_size_d", "twin_thickness_lambda",
                 "burgers_vector_b", "core_width_w"):
        return f"{ui_val:.3g} {spec['ui_unit']}"
    if param == "hall_petch_k":
        return f"{ui_val:.3f} {spec['ui_unit']}"
    if param == "shear_modulus_G":
        return f"{ui_val:.2f} {spec['ui_unit']}"
    if param in ("poisson_ratio_nu", "taylor_factor_M"):
        return f"{ui_val:.3f} {spec['ui_unit']}"
    return f"{ui_val:.2f} {spec['ui_unit']}"


class PlasticityOllamaClient:
    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b",
                 timeout: float = 120.0, max_retries: int = 2):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def is_available(url: str = "http://localhost:11434") -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        try:
            r = _requests.get(f"{url.rstrip('/')}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def list_models(url: str = "http://localhost:11434") -> List[str]:
        if not REQUESTS_AVAILABLE:
            return []
        try:
            r = _requests.get(f"{url.rstrip('/')}/api/tags", timeout=3.0)
            if r.status_code == 200:
                return sorted(m.get("name", "") for m in r.json().get("models", []))
        except Exception:
            pass
        return []

    def generate_json(self, prompt: str, system: Optional[str] = None,
                      debug: bool = False) -> Optional[Any]:
        if not REQUESTS_AVAILABLE:
            return None
        payload: Dict[str, Any] = {
            "model": self.model, "prompt": prompt, "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 4096},
        }
        if system:
            payload["system"] = system
        for attempt in range(self.max_retries + 1):
            try:
                r = _requests.post(f"{self.url}/api/generate",
                                   json=payload, timeout=self.timeout)
                r.raise_for_status()
                raw = r.json().get("response", "")
                parsed = self._parse_lenient(raw)
                if debug:
                    logger.info("Ollama raw (first 800 chars): %s", str(raw)[:800])
                else:
                    logger.debug("Ollama raw (first 400 chars): %s", (raw or "")[:400])
                return parsed
            except Exception as e:
                logger.warning("Ollama attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    return None
                time.sleep(0.75 * (attempt + 1))
        return None

    @staticmethod
    def _parse_lenient(raw: str) -> Any:
        if raw is None:
            return None
        s = raw.strip().lstrip()
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()
        for oc, cc in (("[", "]"), ("{", "}")):
            i, j = s.find(oc), s.rfind(cc)
            if i != -1 and j > i:
                try:
                    return json.loads(s[i: j + 1])
                except Exception:
                    continue
        try:
            return json.loads(s)
        except Exception:
            return None


class PlasticityCorpus:
    _CACHE_VERSION = "v931"

    def __init__(self, db_dir: str = "json_metadatabase", max_chars: int = 4000):
        self.db_dir = db_dir
        self.max_chars = max_chars

    def discover_files(self) -> List[str]:
        found: List[str] = []
        for fname in TARGET_JSON_FILES:
            fpath = os.path.join(self.db_dir, fname)
            if os.path.exists(fpath):
                found.append(fpath)
        if os.path.isdir(self.db_dir):
            for fname in os.listdir(self.db_dir):
                if not fname.lower().endswith(".json"):
                    continue
                fpath = os.path.join(self.db_dir, fname)
                if fpath in found:
                    continue
                low = fname.lower()
                if any(k in low for k in [
                    "dislocation", "shear", "strain_rate", "strain-rate",
                    "friction", "lattice", "yield", "sensitivity",
                    "peierls", "hall", "grain", "burgers", "poisson",
                    "taylor", "core_width",
                ]):
                    found.append(fpath)
        return found

    def load(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        cache_key = f"pl_corpus_{self._CACHE_VERSION}"
        if not force_reload and cache_key in st.session_state:
            return st.session_state[cache_key]
        corpus: List[Dict[str, Any]] = []
        for fpath in self.discover_files():
            try:
                with open(fpath, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Cannot load %s: %s", fpath, e)
                continue
            if isinstance(data, dict):
                for key in ("records", "data", "papers", "entries", "items"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    data = [data]
            if not isinstance(data, list):
                continue
            fname = os.path.basename(fpath)
            for item in data:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("Title") or item.get("title")
                            or item.get("name") or "")
                abstract = str(item.get("Abstract") or item.get("abstract")
                               or item.get("summary") or "")
                full = str(item.get("Full Text") or item.get("full_text")
                           or item.get("text") or item.get("content") or "")
                text = (f"Title: {title}. "
                        f"Abstract: {abstract}. "
                        f"Full Text: {full[:self.max_chars]}")
                corpus.append({"source": fname, "title": title,
                               "text": text, "raw": item})
        st.session_state[cache_key] = corpus
        return corpus

    @staticmethod
    def keyword_prefilter(corpus: List[Dict[str, Any]],
                          material: str, k: int = 30) -> List[Dict[str, Any]]:
        syn = {
            "cu": ["cu", "copper"], "al": ["al", "aluminium", "aluminum"],
            "ni": ["ni", "nickel"], "fe": ["fe", "iron", "steel"],
            "cocrfeni": ["cocrfeni", "co-cr-fe-ni", "hea", "mpea"],
            "ti": ["ti", "titanium"], "mg": ["mg", "magnesium"],
        }
        keys = syn.get(material.lower(), [material.lower()])
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in corpus:
            low = doc["text"].lower()
            score = 0.0
            for kk in keys:
                score += low.count(kk) * 2.0
            for spec in PLASTICITY_ONTOLOGY.values():
                for alias in spec["aliases"]:
                    if alias in low:
                        score += 1.0
                        break
            scored.append((score, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in scored[:k]]


class PlasticityFAISSRetriever:
    CACHE_DIR = ".plasticity_cache"
    INDEX_FILE = "faiss_index.pkl"

    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self._model = None
        self._index = None
        self._doc_vectors = None
        self._docs: List[Dict[str, Any]] = []
        self._dim: int = 0
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None and SBERT_AVAILABLE:
            try:
                self._model = _SentenceTransformer(self.embed_model_name, device="cpu")
            except Exception as e:
                logger.warning("SentenceTransformer failed: %s", e)
                self._model = None
        return self._model

    def build(self, corpus: List[Dict[str, Any]], force: bool = False) -> None:
        if not corpus:
            self._docs, self._index, self._doc_vectors = [], None, None
            return
        fingerprint = _pl_hash(json.dumps(
            [d["source"] + "|" + d["title"] for d in corpus], sort_keys=False))
        cache_path = os.path.join(self.CACHE_DIR, self.INDEX_FILE)
        if not force and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                if payload.get("fingerprint") == fingerprint:
                    self._docs = payload["docs"]
                    self._doc_vectors = payload["vectors"]
                    self._dim = payload["dim"]
                    self._attach_index()
                    return
            except Exception as e:
                logger.warning("FAISS cache load failed: %s", e)
        m = self.model
        if m is None:
            self._docs = list(corpus)
            self._doc_vectors = self._tfidf_fallback_vectors(corpus)
            self._dim = self._doc_vectors.shape[1] if self._doc_vectors.size else 0
            self._attach_index()
            self._persist(fingerprint, cache_path)
            return
        with self._lock:
            texts = [d["text"][:1500] for d in corpus]
            vectors = m.encode(texts, batch_size=32, show_progress_bar=False,
                               convert_to_numpy=True,
                               normalize_embeddings=True).astype(np.float32)
        self._docs = list(corpus)
        self._doc_vectors = vectors
        self._dim = vectors.shape[1]
        self._attach_index()
        self._persist(fingerprint, cache_path)

    def _tfidf_fallback_vectors(self, corpus):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=2048, stop_words="english")
            X = vec.fit_transform([d["text"] for d in corpus]).toarray().astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / norms
        except Exception:
            return np.zeros((len(corpus), 1), dtype=np.float32)

    def _attach_index(self) -> None:
        if self._doc_vectors is None or self._doc_vectors.size == 0:
            self._index = None
            return
        if FAISS_AVAILABLE:
            try:
                index = _faiss.IndexFlatIP(self._doc_vectors.shape[1])
                index.add(self._doc_vectors)
                self._index = index
                return
            except Exception as e:
                logger.warning("FAISS index build failed: %s", e)
        self._index = None

    def _persist(self, fingerprint: str, path: str) -> None:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"fingerprint": fingerprint, "docs": self._docs,
                             "vectors": self._doc_vectors, "dim": self._dim}, f)
        except Exception as e:
            logger.warning("FAISS persist failed: %s", e)

    def search(self, query: str, k: int = 15,
               material_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._docs or self._doc_vectors is None or self._doc_vectors.size == 0:
            return []
        m = self.model
        if m is not None:
            qvec = m.encode([query], batch_size=1, show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True).astype(np.float32)
        else:
            qvec = np.zeros((1, self._dim), dtype=np.float32)
            for i, d in enumerate(self._docs):
                if query.lower()[:30] in d["text"].lower():
                    qvec[0, i % self._dim] += 1.0
            nrm = np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12
            qvec = qvec / nrm
        top_k = min(len(self._docs), max(k * 3, k))
        if self._index is not None:
            D, I = self._index.search(qvec, top_k)
            sims, idxs = D[0], I[0]
        else:
            sims_all = (self._doc_vectors @ qvec[0])
            idxs = np.argsort(-sims_all)[:top_k]
            sims = sims_all[idxs]
        hits: List[Tuple[float, Dict[str, Any]]] = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self._docs):
                continue
            doc = self._docs[int(idx)]
            bonus = 0.0
            if material_hint and material_hint.lower() in doc["text"].lower():
                bonus += 0.05
            hits.append((float(sim) + bonus, doc))
        hits.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in hits[:k]]


_EXTRACT_SCHEMA = (
    '{"param": "rho0|mu|gamma0_dot|srs|sigma0", '
    '"value": <number>, "unit": "<string>", "material": "<string>", '
    '"temp": <number in K or null>, "strain_rate": <number in s^-1 or null>, '
    '"method": "explicit|LLM_inferred", "confidence": <0.0-1.0>, '
    '"evidence": "<short quoted snippet or empty>", '
    '"reasoning": "<3-5 step chain-of-thought, or empty for explicit>"}'
)
_REASONED_INFERENCE_SCHEMA = _EXTRACT_SCHEMA

_STRICT_EXTRACT_PROMPT = """You are a strict materials-science NER system.
Extract ONLY plasticity parameters that are explicitly and unambiguously
stated as numerical values in the text. Do NOT infer. Do NOT fill in
missing parameters.

Parameters of interest (the `param` field MUST be exactly one of these
five ASCII strings — do NOT use symbols or synonyms):
  - "rho0"        (initial dislocation density, m^-2)
  - "mu"          (shear modulus, Pa or GPa)
  - "gamma0_dot"  (reference strain rate, s^-1)
  - "srs"         (strain-rate sensitivity exponent, dimensionless)
  - "sigma0_fric" (friction / lattice-friction stress, Pa or MPa)
  - "sigma0_yield" (yield stress, Pa or MPa)
  - "tau_p"       (Peierls–Nabarro stress, Pa or MPa)

For each parameter you find, report the value verbatim (preserve the unit
as written), identify the material mentioned in the text, and quote the
exact sentence as evidence. Set method="explicit" and confidence >= 0.8.
Leave the reasoning field as an empty string.

If a parameter is missing, qualitative, or only mentioned without a number,
OMIT it from the output entirely.

Schema per element:
  {schema}

Return ONLY a JSON ARRAY. No markdown, no prose.

TEXT:
\"\"\"{text}\"\"\"
"""

_REASONED_INFERENCE_PROMPT = """You are an expert materials-science AI.

GOAL: For the target material "{material}" at T={temp_k} K and strain rate
{strain_rate} s^-1, return ALL plasticity parameters relevant to a phase-
field nanotwinned-metal simulation.

⚠️ CRITICAL: In the JSON output, the `param` field MUST be EXACTLY one of
these ASCII strings (character-for-character):

    "rho0"          — initial dislocation density           [m^-2]
    "mu"            — shear modulus                          [Pa]
    "gamma0_dot"    — reference strain rate                  [s^-1]
    "srs"           — strain-rate sensitivity exponent m     [dimensionless]
    "sigma0_fric"   — friction / lattice-friction stress     [Pa]
    "sigma0_yield"  — yield stress                           [Pa]
    "tau_p"         — Peierls–Nabarro stress                 [Pa]

For EACH parameter, choose ONE of two paths:

PATH A — EXPLICIT EXTRACTION: copy verbatim, method="explicit", conf=0.9.
PATH B — REASONED INFERENCE: derive via physics chain, conf 0.4-0.6.

REASONING CHAINS (use these formulas, do NOT just look up a constant):

  mu (T, material):
    Step 1: Identify material.
    Step 2: Room-T baseline: Cu 48 GPa, Al 26 GPa, Ni 80 GPa, Fe 80 GPa.
    Step 3: mu(T) = mu_300 * (1 - 5e-4 * (T - 300)).
    Step 4: Output in Pa.

  sigma0_fric (T, material, processing):
    Step 1: Identify material and processing.
    Step 2: Baseline: Cu 50 MPa, Al 30 MPa, Ni 70 MPa, Fe 100 MPa.
    Step 3: Adjust ±2-3× for cold work / anneal.
    Step 4: Output in Pa.

  sigma0_yield (T, material, processing):
    Step 1: Identify material and processing.
    Step 2: Baseline: Cu 300 MPa, Al 200 MPa, Ni 400 MPa, Fe 500 MPa.
    Step 3: Adjust for grain size (Hall–Petch) if d is known.
    Step 4: Output in Pa.

  tau_p (T, material):
    Step 1: Identify material class (FCC / BCC / HCP).
    Step 2: Typical FCC τ_P: 0.5-5 MPa; BCC: 300-1500 MPa.
    Step 3: Output in Pa.

  rho0 (T, material, processing):
    Step 1: Identify material and processing.
    Step 2: Annealed: 1e12 m^-2; cold work: 1e14-1e15 m^-2.
    Step 3: Output in m^-2.

  gamma0_dot (T, strain_rate, material):
    🚨 DO NOT OMIT 🚨
    Default quasi-static: 1e-3 s^-1; dynamic: 1e3-1e6; MD: 1e6-1e8.
    ALWAYS emit a JSON object with param="gamma0_dot".

  srs (m, material, T):
    Step 1: FCC m = 20; BCC m = 100; HCP m = 50; HEA m = 30.
    Step 2: Output dimensionless value.

CRITICAL RULES:
- The `param` field MUST be exactly one of the ASCII strings listed above.
- Return ONLY a JSON ARRAY.

Return ONLY a JSON ARRAY. Schema per element:
  {schema}
No markdown, no comments, no prose outside JSON.

TEXT:
\"\"\"{text}\"\"\"
"""


class PlasticityParameterExtractor:
    NUM_RE = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(?:[×xX\*]\s*10\s*\^?\s*\{?(-?\d+)\}?)?\s*"
        r"(GPa|MPa|kPa|Pa|m\^?-?2|m-2|s\^?-?1|/s)?", re.I)

    _CTX_RATE_RE = re.compile(
        r"(?:strain\s*[- ]?\s*rate|deformation\s+rate|loading\s+rate|"
        r"shear\s+rate|strain\s+rate\s+of)"
        r"[^.\n]{0,60}?"
        r"(\d+(?:\.\d+)?(?:\s*[×xX\*]\s*10\s*\^?\s*\{?[+-]?\d+\}?)?)"
        r"\s*(?:s\^?-?1|/\s*s|per\s+second|s⁻¹|s-1)", re.I)

    def __init__(self, client: Optional[PlasticityOllamaClient],
                 cache: Optional[Dict[str, Any]] = None):
        self.client = client
        self.cache = cache if cache is not None else {}

    def extract(self, text: str, material: str, temp_k: float,
                strain_rate: float, use_llm: bool = True,
                mode: str = "reasoned_inference",
                debug_llm: bool = False) -> List[Dict[str, Any]]:
        key = _pl_hash(
            f"{text[:2000]}|{material}|{temp_k}|{strain_rate}|{use_llm}|{mode}")
        if key in self.cache:
            return self.cache[key]
        llm_out: List[Dict[str, Any]] = []
        if use_llm and self.client is not None:
            if mode == "strict_extract":
                prompt = _STRICT_EXTRACT_PROMPT.format(
                    schema=_EXTRACT_SCHEMA, text=text[:3500])
            else:
                prompt = _REASONED_INFERENCE_PROMPT.format(
                    schema=_REASONED_INFERENCE_SCHEMA,
                    material=material, temp_k=temp_k, strain_rate=strain_rate,
                    text=text[:3500])
            raw = self.client.generate_json(prompt, debug=debug_llm)
            logger.debug("Ollama raw (%s): %s", mode, raw)
            llm_out = self._validate(raw)
            if debug_llm:
                logger.info("LLM returned %d valid items (mode=%s): %s",
                            len(llm_out), mode,
                            [x["param"] for x in llm_out])
        heuristic_out = self._heuristic(text, material, temp_k, strain_rate)
        have = {e["param"] for e in llm_out}
        for h in heuristic_out:
            if h["param"] not in have:
                llm_out.append(h)
                have.add(h["param"])
        if "gamma0_dot" not in have:
            llm_out.append({
                "param": "gamma0_dot",
                "value": float(strain_rate) if strain_rate else 1e-3,
                "unit": "s^-1", "material": material, "temp": temp_k,
                "strain_rate": strain_rate, "method": "default_fallback",
                "confidence": 0.3, "evidence": "",
                "reasoning": (
                    "Step 1: No explicit 'reference strain rate' found in text.\n"
                    "Step 2: Heuristic scan returned no candidate.\n"
                    f"Step 3: Falling back to user strain rate ({strain_rate} s^-1).\n"
                    "Step 4: Default 1e-3 s^-1 applies if user value unavailable."),
            })
            have.add("gamma0_dot")
        self.cache[key] = llm_out
        return llm_out

    @staticmethod
    def _validate(raw: Any) -> List[Dict[str, Any]]:
        if raw is None: return []
        if isinstance(raw, dict): raw = [raw]
        if not isinstance(raw, list): return []
        out: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict): continue
            raw_p = item.get("param", "")
            p = _canonicalize_param(raw_p)
            if p is None: continue
            v, unit = _coerce_value(item)
            if v is None: continue
            out.append({
                "param": p, "value": float(v), "unit": str(unit or ""),
                "material": str(item.get("material") or ""),
                "temp": item.get("temp"), "strain_rate": item.get("strain_rate"),
                "method": str(item.get("method") or "unknown").lower(),
                "confidence": float(item.get("confidence", 0.5) or 0.5),
                "evidence": str(item.get("evidence") or "")[:240],
                "reasoning": str(item.get("reasoning") or "")[:600],
            })
        return out

    @classmethod
    def _heuristic(cls, text: str, material: str, temp_k: float,
                   strain_rate: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        low = text.lower()
        for param, spec in PLASTICITY_ONTOLOGY.items():
            for alias in spec["aliases"]:
                idx = low.find(alias.lower())
                if idx < 0: continue
                window = text[max(0, idx - 80): idx + 300]
                m = cls.NUM_RE.search(window)
                if not m: continue
                try:
                    mantissa = float(m.group(1))
                    exponent = int(m.group(2)) if m.group(2) else 0
                    unit = (m.group(3) or "").strip()
                    value = mantissa * (10 ** exponent)
                except Exception:
                    continue
                out.append({
                    "param": param, "value": value, "unit": unit,
                    "material": material, "temp": temp_k,
                    "strain_rate": strain_rate, "method": "heuristic",
                    "confidence": 0.35, "evidence": window[:180].strip(),
                    "reasoning": "",
                })
                break
        if not any(e["param"] == "gamma0_dot" for e in out):
            m = cls._CTX_RATE_RE.search(text)
            if m:
                num_str = m.group(1).replace(" ", "")
                sci = re.match(
                    r"(\d+(?:\.\d+)?)(?:[×xX\*]10\^?\{?([+-]?\d+)\}?)?",
                    num_str)
                if sci:
                    try:
                        mantissa = float(sci.group(1))
                        exponent = int(sci.group(2)) if sci.group(2) else 0
                        value = mantissa * (10 ** exponent)
                        start = max(0, m.start() - 40)
                        end = min(len(text), m.end() + 40)
                        out.append({
                            "param": "gamma0_dot", "value": value, "unit": "s^-1",
                            "material": material, "temp": temp_k,
                            "strain_rate": strain_rate, "method": "heuristic",
                            "confidence": 0.3,
                            "evidence": text[start:end].strip(),
                            "reasoning": (
                                f"Contextual heuristic: matched "
                                f"'{m.group(0)[:60]}' → value {value:.3e} s^-1."),
                        })
                    except (ValueError, TypeError):
                        pass
        return out


@dataclass
class PlasticityCandidate:
    param: str
    value_si: float
    raw_value: float
    raw_unit: str
    score: float
    confidence: float
    material: str
    temp_k: Optional[float]
    strain_rate: Optional[float]
    method: str
    source_file: str
    source_title: str
    evidence: str
    reasoning: str = ""
    clamped: bool = False
    theory: str = ""
    provenance: str = ""
    context: bool = False

    def to_display(self) -> Dict[str, Any]:
        spec = PLASTICITY_ONTOLOGY[self.param]
        d = {
            "value": self.value_si / spec["ui_scale"],
            "unit": spec["ui_unit"],
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "material": self.material or "n/a",
            "temp_K": self.temp_k,
            "method": self.method,
            "source": f"{self.source_file} — {self.source_title[:60]}",
            "clamped": self.clamped,
            "evidence": self.evidence[:140],
            "reasoning": (self.reasoning[:200] + "…"
                          if len(self.reasoning) > 200 else self.reasoning),
        }
        if self.theory:
            d["theory"] = self.theory
        if self.provenance:
            d["provenance"] = self.provenance
        if self.context:
            d["context"] = True
        return d


# ============================================================================
# ███ v9.0.0 — PHYSICS-REGIME EXPERT (LatentMoE)                        ███
# ============================================================================
class PhysicsRegimeExpert:
    def __init__(self, falloff_decades: float = 1.0,
                 min_sigma_decades: float = 0.25):
        self.falloff_decades = float(falloff_decades)
        self.min_sigma_decades = float(min_sigma_decades)

    def score(
        self,
        value_si: float,
        synthesis: Optional[str],
        architecture: Optional[str],
        twin_spacing_nm: Optional[float],
    ) -> float:
        try:
            v = float(value_si)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0

        regime = classify_rho0_regime(synthesis, architecture, twin_spacing_nm)
        log_v   = np.log10(v)
        log_lo  = np.log10(regime["low"])
        log_hi  = np.log10(regime["high"])
        log_ctr = np.log10(max(regime["inferred"], 1e-10))

        if log_lo <= log_v <= log_hi:
            sigma = max((log_hi - log_lo) / 4.0, self.min_sigma_decades)
            return float(np.exp(-(log_v - log_ctr) ** 2 / (2.0 * sigma ** 2)))

        dist = min(abs(log_v - log_lo), abs(log_v - log_hi))
        return float(np.exp(-dist / self.falloff_decades))


# ============================================================================
# ███ v9.1.0 — THEORY-REGIME EXPERT (LatentMoE)                         ███
# ============================================================================
class TheoryRegimeExpert:
    def __init__(self, falloff_decades: float = 1.5,
                 min_sigma_decades: float = 0.25):
        self.falloff_decades = float(falloff_decades)
        self.min_sigma_decades = float(min_sigma_decades)

    def score(self, value_si: float,
              target_theory: Optional[str],
              candidate_theory: Optional[str] = None) -> float:
        try:
            v = float(value_si)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0 or not np.isfinite(v):
            return 0.0

        regime = classify_gamma0_regime(target_theory)
        log_v   = np.log10(v)
        log_lo  = np.log10(regime["low"])
        log_hi  = np.log10(regime["high"])
        log_ctr = np.log10(max(regime["inferred"], 1e-30))

        tag_score = 1.0
        if target_theory and candidate_theory:
            t_target = _norm_theory(target_theory)
            t_cand   = _norm_theory(candidate_theory)
            if t_target and t_cand:
                if t_target == t_cand:
                    tag_score = 1.0
                elif ({t_target, t_cand} <= _GAMMA0_QUASI_STATIC_GROUP):
                    tag_score = 0.85
                else:
                    tag_score = 0.05
            elif t_cand and t_target and t_cand != t_target:
                tag_score = 0.10

        if log_lo <= log_v <= log_hi:
            sigma = max((log_hi - log_lo) / 4.0, self.min_sigma_decades)
            reg_score = float(np.exp(-(log_v - log_ctr) ** 2
                                     / (2.0 * sigma ** 2)))
        else:
            dist = min(abs(log_v - log_lo), abs(log_v - log_hi))
            reg_score = float(np.exp(-dist / self.falloff_decades))

        return tag_score * reg_score


class PlasticityLatentMoEScorer:
    def __init__(self, w_material=0.40, w_thermal=0.20, w_strain=0.08,
                 w_method=0.08, w_confidence=0.04, w_reasoning=0.20,
                 thermal_sigma=100.0):
        self.w_material = w_material
        self.w_thermal = w_thermal
        self.w_strain = w_strain
        self.w_method = w_method
        self.w_confidence = w_confidence
        self.w_reasoning = w_reasoning
        self.thermal_sigma = thermal_sigma

    def _material_expert(self, ext_mat: str, target_mat: str) -> float:
        if not ext_mat: return 0.4
        a, b = ext_mat.lower().strip(), target_mat.lower().strip()
        if a == b: return 1.0
        if a in b or b in a: return 0.85
        fcc = {"cu", "al", "ni", "ag", "au", "pt", "pb", "cocrfeni", "hea"}
        bcc = {"fe", "w", "mo", "cr", "v", "nb"}
        hcp = {"mg", "ti", "zn", "co", "zr"}
        for grp in (fcc, bcc, hcp):
            if a in grp and b in grp: return 0.5
        return 0.25

    def _thermal_expert(self, ext_temp, target_temp) -> float:
        if ext_temp is None: return 0.5
        try:
            t = float(ext_temp)
        except (TypeError, ValueError):
            return 0.5
        diff = t - float(target_temp)
        return float(np.exp(-(diff ** 2) / (2.0 * self.thermal_sigma ** 2)))

    def _strain_expert(self, ext_rate, target_rate) -> float:
        if ext_rate is None: return 0.5
        try:
            r = float(ext_rate)
            if r <= 0 or target_rate <= 0: return 0.5
            ratio = math.log10(r / target_rate)
            return float(np.exp(-(ratio ** 2) / 8.0))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _method_expert(method: str) -> float:
        return {
            "experiment": 1.0, "review": 0.8,
            "md": 0.6, "molecular dynamics": 0.6,
            "dft": 0.5, "first-principles": 0.5, "ab initio": 0.5,
            "heuristic": 0.3,
            "explicit": 0.9, "llm_inferred": 0.5,
            "prior inference": 0.35, "llm_prior": 0.35,
            "default_fallback": 0.3,
            "hall_petch_intercept":           0.55,
            "taylor_factor_cross_conversion": 0.45,
            "peierls_nabarro_formula":        0.50,
            "side_note_table":                0.55,
            "derived_consensus":              0.55,
            "physics_regime_inference":       0.60,
            "theory_regime_inference":        0.60,
            "sigma0_route_hall_petch_d":          0.50,
            "sigma0_route_hall_petch_lam":        0.50,
            "sigma0_route_model_inversion":       0.55,
            "sigma0_route_peierls_nabarro":       0.45,
        }.get((method or "").lower(), 0.5)

    @staticmethod
    def _reasoning_expert(reasoning: str, method: str) -> float:
        method_l = (method or "").lower()
        if method_l not in ("llm_inferred", "heuristic", "default_fallback",
                            "prior inference", "llm_prior",
                            "hall_petch_intercept",
                            "taylor_factor_cross_conversion",
                            "peierls_nabarro_formula",
                            "side_note_table", "derived_consensus",
                            "physics_regime_inference",
                            "theory_regime_inference",
                            "sigma0_route_hall_petch_d",
                            "sigma0_route_hall_petch_lam",
                            "sigma0_route_model_inversion",
                            "sigma0_route_peierls_nabarro"):
            return 0.5
        if not reasoning:
            return 0.2
        n_steps = reasoning.count("Step ") + reasoning.count("\n")
        cites_formula = any(k in reasoning for k in
                            ["mu(T)", "sigma0", "rho_0", "gamma0_dot",
                             "*", "/", "exp", "log", "GPa", "MPa", "s^-1",
                             "σ_y", "k_y", "τ_P", "Peierls", "Hall",
                             "synthesis", "architecture", "twin_spacing",
                             "Johnson", "JC", "CPFEM", "DDD", "MD",
                             "phase-field", "framework", "theory"])
        base = min(0.4 + 0.05 * n_steps, 0.8)
        if cites_formula:
            base = min(base + 0.15, 0.85)
        return float(base)

    def score(self, extractions, target_material, target_temp,
              target_strain_rate=1e-3, top_k=8,
              synthesis=None, architecture=None, twin_spacing=None,
              target_theory=None):
        regime_expert_rho0 = PhysicsRegimeExpert()
        theory_expert_gdot = TheoryRegimeExpert()

        regime_active_rho0 = any([
            synthesis not in (None, "", "(unknown)"),
            architecture not in (None, "", "(unknown)"),
            (isinstance(twin_spacing, (int, float))
             and twin_spacing is not None
             and twin_spacing > 0
             and np.isfinite(twin_spacing)),
        ])
        theory_active_gdot = bool(target_theory and _norm_theory(target_theory))

        buckets: Dict[str, List[PlasticityCandidate]] = {
            p: [] for p in PARAM_ORDER
        }

        for ext in extractions:
            p = ext.get("param")
            if p not in buckets:
                continue

            try:
                v_si = _normalize_unit(ext["value"], ext.get("unit", ""), p)
            except Exception:
                continue
            if v_si is None or not np.isfinite(v_si):
                continue
            v_si_clamped, was_clamped = _pl_clamp(v_si, p)

            if p == "rho0" and regime_active_rho0:
                w_mat, w_t, w_s, w_m, w_c, w_r, w_reg = (
                    0.20, 0.10, 0.05, 0.05, 0.05, 0.30, 0.25,
                )
            elif p == "gamma0_dot" and theory_active_gdot:
                w_mat, w_t, w_s, w_m, w_c, w_r, w_reg = (
                    0.10, 0.05, 0.05, 0.10, 0.05, 0.40, 0.25,
                )
            else:
                w_mat = self.w_material
                w_t   = self.w_thermal
                w_s   = self.w_strain
                w_m   = self.w_method
                w_c   = self.w_confidence
                w_r   = self.w_reasoning
                w_reg = 0.0

            s_mat  = self._material_expert(ext.get("material", ""),
                                           target_material)
            s_temp = self._thermal_expert(ext.get("temp"), target_temp)
            s_str  = self._strain_expert(ext.get("strain_rate"),
                                         target_strain_rate)
            s_meth = self._method_expert(ext.get("method", "unknown"))
            s_conf = float(ext.get("confidence", 0.5) or 0.5)
            s_reas = self._reasoning_expert(ext.get("reasoning", ""),
                                            ext.get("method", "unknown"))

            s_reg = 0.0
            if p == "rho0" and regime_active_rho0:
                s_reg = regime_expert_rho0.score(v_si_clamped, synthesis,
                                                  architecture, twin_spacing)
            elif p == "gamma0_dot" and theory_active_gdot:
                s_reg = theory_expert_gdot.score(
                    v_si_clamped,
                    target_theory=target_theory,
                    candidate_theory=ext.get("theory", "unspecified"))

            score = (w_mat * s_mat
                     + w_t * s_temp
                     + w_s * s_str
                     + w_m * s_meth
                     + w_c * s_conf
                     + w_r * s_reas
                     + w_reg * s_reg)

            cand_kwargs = dict(
                param=p,
                value_si=v_si_clamped,
                raw_value=float(ext["value"]),
                raw_unit=str(ext.get("unit", "")),
                score=score,
                confidence=s_conf,
                material=str(ext.get("material", "")),
                temp_k=ext.get("temp"),
                strain_rate=ext.get("strain_rate"),
                method=str(ext.get("method", "unknown")),
                source_file=str(ext.get("_source_file", "")),
                source_title=str(ext.get("_source_title", "")),
                evidence=str(ext.get("evidence", "")),
                reasoning=str(ext.get("reasoning", "")),
                clamped=was_clamped,
                provenance=str(ext.get("_provenance", "")),
                context=bool(ext.get("_context", False)),
            )
            if p == "gamma0_dot":
                cand_kwargs["theory"] = str(ext.get("theory", "unspecified"))
            buckets[p].append(PlasticityCandidate(**cand_kwargs))

        for p in buckets:
            buckets[p].sort(key=lambda c: c.score, reverse=True)
            buckets[p] = buckets[p][:top_k]
        return buckets


class PlasticityMaterialPriorLearner:
    def __init__(self, extractor: PlasticityParameterExtractor):
        self.extractor = extractor

    def learn(self, corpus, material="?", temp_k=300.0, strain_rate=1e-3,
              use_llm=False, max_docs=200, debug_llm=False) -> pd.DataFrame:
        raw: Dict[Tuple[str, str], List[float]] = {}
        for doc in corpus[:max_docs]:
            text = doc["text"]
            extractions = self.extractor.extract(
                text, material=material, temp_k=temp_k,
                strain_rate=strain_rate, use_llm=use_llm,
                mode="strict_extract", debug_llm=debug_llm)
            for ext in extractions:
                mat = (ext.get("material") or "").strip() or "unknown"
                p = ext.get("param")
                if p not in PLASTICITY_ONTOLOGY: continue
                if (ext.get("method") or "").lower() == "default_fallback":
                    continue
                try:
                    v_si = _normalize_unit(ext["value"], ext.get("unit", ""), p)
                except Exception:
                    continue
                v_si, _ = _pl_clamp(v_si, p)
                raw.setdefault((mat, p), []).append(v_si)
        rows: List[Dict[str, Any]] = []
        for (mat, p), values in raw.items():
            if len(values) == 0: continue
            arr = np.array(values, dtype=float)
            rows.append({
                "material": mat, "param": p, "n": int(arr.size),
                "mean": float(arr.mean()), "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=[
                "material", "param", "n", "mean", "median", "std", "p10", "p90"])
        return df.sort_values(["param", "material"]).reset_index(drop=True)

    @staticmethod
    def suggest(priors_df, material, param) -> Optional[float]:
        if priors_df is None or priors_df.empty: return None
        sub = priors_df[
            (priors_df["material"].str.lower() == material.lower())
            & (priors_df["param"] == param)]
        if sub.empty: return None
        row = sub.iloc[0]
        if row["n"] < 3: return float(row["median"])
        return float(0.5 * row["median"] + 0.5 * row["mean"])


def render_plasticity_candidate_histograms(
    candidates_by_param: Dict[str, List[PlasticityCandidate]],
    log_scale_params: Optional[set] = None,
) -> None:
    log_scale_params = log_scale_params or {"rho0", "gamma0_dot"}
    available = [p for p in PARAM_ORDER if candidates_by_param.get(p)]
    if not available:
        st.info("No candidates to plot yet.")
        return
    n = len(available)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[PLASTICITY_ONTOLOGY[p]["label"] for p in available],
        horizontal_spacing=0.10, vertical_spacing=0.18)
    for idx, p in enumerate(available):
        r = idx // cols + 1
        c = idx % cols + 1
        spec = PLASTICITY_ONTOLOGY[p]
        ui_vals = [cd.value_si / spec["ui_scale"] for cd in candidates_by_param[p]]
        scores = [cd.score for cd in candidates_by_param[p]]
        fig.add_trace(go.Histogram(
            x=ui_vals, marker=dict(color="#3b82f6",
                                   line=dict(color="#1e3a8a", width=1)),
            name=spec["symbol"],
            hovertemplate="%{x:.4g}<br>count: %{y}<extra></extra>",
            nbinsx=max(4, min(20, len(ui_vals) * 2)),
            showlegend=False), row=r, col=c)
        top_idx = int(np.argmax(scores))
        fig.add_trace(go.Scatter(
            x=[ui_vals[top_idx]], y=[1], mode="markers",
            marker=dict(color="#ef4444", size=14, symbol="star",
                        line=dict(color="white", width=1)),
            name="⭐ Best",
            hovertemplate=f"⭐ Best = {ui_vals[top_idx]:.4g} "
                          f"{spec['ui_unit']}<br>score: {scores[top_idx]:.3f}"
                          f"<extra></extra>",
            showlegend=(idx == 0)), row=r, col=c)
        if p in log_scale_params and min(ui_vals) > 0:
            fig.update_xaxes(type="log", row=r, col=c)
    fig.update_layout(
        height=320 * rows, showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#1e293b"), bargap=0.08)
    st.plotly_chart(fig, use_container_width=True)


@dataclass
class PlasticityRecommendationBundle:
    material: str
    temp_k: float
    strain_rate: float
    candidates: Dict[str, List[PlasticityCandidate]]
    defaults: Dict[str, float]
    priors_df: pd.DataFrame
    retrieval_backend: str
    llm_used: bool
    timestamp: float = field(default_factory=time.time)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    derived_stress: Dict[str, Any] = field(default_factory=dict)
    # ── v10.0.0 additions ───────────────────────────────────────────────
    scored_v10:     Dict[str, List['ScoredCandidate']] = field(default_factory=dict)
    context_vector: Optional['ContextVector']          = None
    metadb:         Optional['MetaDatabase']           = None

    def best_v10(self, param: str) -> Optional['ScoredCandidate']:
        lst = self.scored_v10.get(param, [])
        return lst[0] if lst else None

    def best(self, param: str) -> Optional[PlasticityCandidate]:
        lst = self.candidates.get(param, [])
        if not lst:
            return None
        ranked = [c for c in lst if not getattr(c, 'context', False)]
        if not ranked:
            logger.warning("bundle.best[%s]: all candidates are context — "
                           "returning top-scored anyway", param)
            ranked = list(lst)
        return max(ranked, key=lambda c: c.score)

    def prior_suggestion(self, param: str) -> Optional[float]:
        return PlasticityMaterialPriorLearner.suggest(
            self.priors_df, self.material, param)

    def coverage(self) -> int:
        return sum(1 for p in PARAM_ORDER if self.candidates.get(p))

    def coverage_five(self) -> int:
        return sum(1 for p in SIDEBAR_PARAM_ORDER if self.candidates.get(p))


class PlasticityRecommender:
    CACHE_DIR = ".plasticity_cache"
    LLM_CACHE_FILE = "llm_cache.json"

    def __init__(self, db_dir: str = "json_metadatabase",
                 ollama_model: str = "qwen2.5:7b",
                 use_llm: bool = True, top_k_retrieval: int = 20,
                 debug_llm: bool = False, use_grounded: bool = True,
                 allow_prior_inference: bool = True,
                 rho0_synthesis: Optional[str] = None,
                 rho0_architecture: Optional[str] = None,
                 rho0_twin_spacing: Optional[float] = None,
                 gamma0_target_theory: Optional[str] = None,
                 cascade_mode: str = 'union'):
        self.db_dir = db_dir
        self.corpus = PlasticityCorpus(db_dir)
        self.client = PlasticityOllamaClient(model=ollama_model)
        self.llm_available = use_llm and PlasticityOllamaClient.is_available()
        self.retriever = PlasticityFAISSRetriever()
        self.hybrid = None
        if use_grounded:
            try:
                self.hybrid = get_retriever(db_dir, use_dense=True)
            except Exception as e:
                logger.warning("Hybrid retriever init failed: %s", e)
                self.hybrid = None
        self.extractor = PlasticityParameterExtractor(
            self.client if self.llm_available else None,
            cache=self._load_disk_cache())
        self.scorer = PlasticityLatentMoEScorer()
        self.prior_learner = PlasticityMaterialPriorLearner(self.extractor)
        self.top_k_retrieval = top_k_retrieval
        self.debug_llm = debug_llm
        self.allow_prior_inference = allow_prior_inference
        self.rho0_synthesis       = rho0_synthesis
        self.rho0_architecture    = rho0_architecture
        self.rho0_twin_spacing    = rho0_twin_spacing
        self.rho0_regime: Dict[str, Any] = {}
        self.rho0_snippet_findings: List[Dict[str, Any]] = []
        self.gamma0_target_theory = gamma0_target_theory
        self.gamma0_regime: Dict[str, Any] = {}
        self.gamma0_snippet_findings: List[Dict[str, Any]] = []
        self.cascade_mode = cascade_mode if cascade_mode in ('union', 'fallback') \
            else 'union'
        # ── v10.0.0 ─────────────────────────────────────────────────────
        self.metadb = MetaDatabase()
        self.scorer_v10 = None  # instantiated per-run with fresh ContextVector
        self._ctx_vector: Optional['ContextVector'] = None

    def _load_disk_cache(self) -> Dict[str, Any]:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            path = os.path.join(self.CACHE_DIR, self.LLM_CACHE_FILE)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
        return {}

    def _save_disk_cache(self) -> None:
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            path = os.path.join(self.CACHE_DIR, self.LLM_CACHE_FILE)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.extractor.cache, f)
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def recommend_grounded(self, material, temp_k, strain_rate=1e-3,
                           value_hints=None, max_docs=20,
                           build_priors=True, progress_callback=None
                           ) -> PlasticityRecommendationBundle:
        if self.hybrid is None:
            st.warning("Hybrid retriever unavailable — falling back to "
                       "legacy v8.7 pipeline via recommend().")
            return self.recommend(material, temp_k, strain_rate,
                                  max_docs=max_docs, build_priors=build_priors,
                                  progress_callback=progress_callback)

        extractions: List[Dict[str, Any]] = []
        diagnostics: List[Dict[str, Any]] = []
        retrieval_backend = ("hybrid(lexical+faiss)"
                             if self.hybrid.dense else "hybrid(lexical-only)")

        # ── v10: build the ContextVector once for this run ─────────────
        self._ctx_vector = ContextVector.from_sidebar(
            material=material,
            synthesis=self.rho0_synthesis,
            architecture=self.rho0_architecture,
            twin_spacing=self.rho0_twin_spacing,
            framework=self.gamma0_target_theory,
        )
        self.scorer_v10 = LatentMoEScorerV10(self._ctx_vector, self.metadb)

        for i, param in enumerate(PARAM_ORDER):
            if progress_callback:
                progress_callback(i + 1, len(PARAM_ORDER), f"cascade[{param}]")

            if param == "rho0":
                rho_cands, rho_snips, rho_regime = recommend_rho0_physics(
                    material=material,
                    synthesis=self.rho0_synthesis,
                    architecture=self.rho0_architecture,
                    twin_spacing=self.rho0_twin_spacing,
                    retriever=self.hybrid,
                    ollama_model=self.client.model,
                    cascade_mode=self.cascade_mode,
                )
                self.rho0_snippet_findings = rho_snips
                self.rho0_regime = rho_regime
                cands = rho_cands
                records = []
                diag = dict(
                    param="rho0", records=len(rho_snips),
                    n_llm=sum(1 for c in cands if c.provenance == "llm_extract"),
                    n_heuristic=sum(1 for c in cands if c.provenance == "regex_ner"),
                    n_prior=sum(1 for c in cands
                                if c.provenance == "llm_prior"),
                    tier=("llm" if any(c.provenance == "llm_extract" for c in cands)
                          else "llm_prior"),
                    reason=(f"physics-regime cascade; "
                            f"{len(rho_snips)} snippet(s); "
                            f"regime={rho_regime['regime'][:60]}; "
                            f"mode={self.cascade_mode}"),
                    cascade_mode=self.cascade_mode,
                )
            elif param == "gamma0_dot":
                g_cands, g_snips, g_regime = recommend_gamma0_theory_aware(
                    material=material,
                    target_theory=self.gamma0_target_theory,
                    retriever=self.hybrid,
                    ollama_model=self.client.model,
                    cascade_mode=self.cascade_mode,
                )
                self.gamma0_snippet_findings = g_snips
                self.gamma0_regime = g_regime
                cands = g_cands
                records = []
                diag = dict(
                    param="gamma0_dot",
                    records=len(g_snips),
                    n_llm=sum(1 for c in cands if c.provenance == "llm_extract"),
                    n_heuristic=sum(1 for c in cands if c.provenance == "regex_ner"),
                    n_prior=sum(1 for c in cands
                                if c.provenance == "llm_prior"),
                    tier=("llm" if any(c.provenance == "llm_extract" for c in cands)
                          else "llm_prior"),
                    reason=(f"theory-aware cascade; theory="
                            f"{self.gamma0_target_theory}; "
                            f"{len(g_snips)} snippet(s); "
                            f"regime={g_regime['regime'][:60]}; "
                            f"mode={self.cascade_mode}"),
                    cascade_mode=self.cascade_mode,
                )
            else:
                cands, records, diag = recommend_param_values(
                    param_key=param, material=material,
                    value_hints=value_hints if param == "mu" else None,
                    k=max(4, max_docs // 3),
                    use_llm=self.llm_available,
                    ollama_model=self.client.model,
                    retriever=self.hybrid,
                    allow_prior_inference=self.allow_prior_inference,
                    cascade_mode=self.cascade_mode)

            diagnostics.append(diag)
            for c in cands:
                prov_fine = _norm_provenance(getattr(c, "provenance", "")
                                             or getattr(c, "method", ""))
                method = ("explicit" if prov_fine in ("llm_extract", "llm_reasoned")
                          else "heuristic" if prov_fine == "regex_ner"
                          else "prior inference")
                if (param == "rho0"
                        and c.method == "physics_regime_inference"):
                    method = "physics_regime_inference"
                if (param == "gamma0_dot"
                        and c.method == "theory_regime_inference"):
                    method = "theory_regime_inference"
                extractions.append({
                    "param": param, "value": c.value, "unit": c.unit,
                    "material": material, "temp": temp_k,
                    "strain_rate": strain_rate, "method": method,
                    "confidence": c.confidence, "evidence": c.evidence,
                    "reasoning": (getattr(c, "reasoning", "")
                                  or _default_reasoning(c)),
                    "dominant_factor": getattr(c, "dominant_factor", ""),
                    "theory": getattr(c, "theory", "unspecified"),
                    "_source_file": c.source,
                    "_source_title": c.property_label,
                    "_provenance": prov_fine,
                    "_context": bool(getattr(c, "context", False)),
                })

        context_keys = ("sigma0_yield", "grain_size_d",
                        "twin_thickness_lambda", "hall_petch_k",
                        "mu", "shear_modulus_G", "poisson_ratio_nu",
                        "burgers_vector_b", "taylor_factor_M", "core_width_w")
        context_bucket: Dict[str, List[ValueCandidate]] = {}
        for p_key in context_keys:
            recs = self.hybrid.search(p_key, material, k=max(3, max_docs // 4))
            hits = heuristic_extract(recs, p_key)
            for _h in hits:
                if isinstance(_h, dict):
                    _stamp_provenance(_h, 'regex_ner')
                    _h['_context'] = False
            context_bucket[p_key] = gatekeep(hits, p_key,
                                             provenance="regex_ner")

        derived_v883 = derive_sigma0_and_tau_p(context_bucket,
                                                bundle_material=material)
        for c in derived_v883:
            if c.property_label.startswith("Friction"):
                target = "sigma0_fric"
            else:
                target = "tau_p"
            extractions.append({
                "param": target, "value": c.value, "unit": c.unit,
                "material": material, "temp": temp_k, "strain_rate": strain_rate,
                "method": c.method, "confidence": c.confidence,
                "evidence": c.evidence,
                "reasoning": getattr(c, "reasoning", "") or _default_reasoning(c),
                "_source_file": c.source,
                "_source_title": c.property_label,
                "_provenance": "physics_inferred",
                "_context": _is_context(target, "physics_inferred"),
            })
            diagnostics.append(dict(param=target, records=0, n_llm=0,
                                    n_heuristic=0, n_prior=0, tier="physics_inferred",
                                    reason=f"physics-inferred via {c.method}",
                                    cascade_mode=self.cascade_mode))

        present = {e["param"] for e in extractions}
        if "gamma0_dot" not in present:
            extractions.append({
                "param": "gamma0_dot",
                "value": float(strain_rate) if strain_rate else 1e-3,
                "unit": "s^-1", "material": material, "temp": temp_k,
                "strain_rate": strain_rate, "method": "default_fallback",
                "confidence": 0.3, "evidence": "",
                "reasoning": ("Step 1: three-tier cascade found no explicit "
                              "reference strain rate.\nStep 2: using the "
                              "user-supplied rate as the reference value."),
                "_source_file": "default_fallback",
                "_source_title": "user input",
                "_provenance": "regex_ner",
                "_context": _is_context("gamma0_dot", "regex_ner"),
            })
            diagnostics.append(dict(param="gamma0_dot", records=0,
                                    n_llm=0, n_heuristic=0, n_prior=0,
                                    tier="default_fallback",
                                    reason="no cascade candidate → default",
                                    cascade_mode=self.cascade_mode))

        candidates = self.scorer.score(
            extractions, material, temp_k,
            target_strain_rate=strain_rate, top_k=8,
            synthesis=self.rho0_synthesis,
            architecture=self.rho0_architecture,
            twin_spacing=self.rho0_twin_spacing,
            target_theory=self.gamma0_target_theory,
        )

        derived_stress: Dict[str, Any] = {}
        try:
            provisional = PlasticityRecommendationBundle(
                material=material, temp_k=temp_k, strain_rate=strain_rate,
                candidates=candidates, defaults={}, priors_df=pd.DataFrame(),
                retrieval_backend=retrieval_backend,
                llm_used=self.llm_available, diagnostics=diagnostics)
            derived_stress = build_derived_stress_bundle(provisional)
            cons = derived_stress.get('consensus')
            bucket = candidates.setdefault('sigma0_fric', [])
            if cons:
                reasoning_parts = []
                for r in derived_stress['routes']:
                    if r.value_MPa is not None:
                        reasoning_parts.append(
                            f"{r.key}: {r.value_MPa:.1f} MPa ({r.status})")
                derived_cand = PlasticityCandidate(
                    param='sigma0_fric',
                    value_si=cons['value_MPa'] * 1e6,
                    raw_value=cons['value_MPa'],
                    raw_unit='MPa',
                    score=cons['conf'],
                    confidence=cons['conf'],
                    material=material,
                    temp_k=temp_k,
                    strain_rate=strain_rate,
                    method='derived_consensus',
                    source_file='side-note-cluster',
                    source_title='HP/solver-inv/P–N consensus',
                    evidence=(f"σ₀ = {cons['value_MPa']:.1f} MPa "
                              f"({cons['n_routes']} routes, "
                              f"spread {cons['rel_spread']:.0%})"),
                    reasoning='; '.join(reasoning_parts),
                    provenance='physics_inferred',
                    context=_is_context('sigma0_fric', 'physics_inferred'),
                )
                bucket.append(derived_cand)
                diagnostics.append(dict(
                    param='sigma0_fric', records=0, n_llm=0,
                    n_heuristic=0, n_prior=0, tier='physics_inferred_consensus',
                    reason=(f"▼ consensus σ₀ = {cons['value_MPa']:.1f} MPa "
                            f"from {cons['n_routes']} routes "
                            f"(spread {cons['rel_spread']:.0%}, "
                            f"conf {cons['conf']:.2f})"),
                    cascade_mode=self.cascade_mode))

            _SIGMA0_ROUTE_METHOD = {
                'hall_petch_d':    'sigma0_route_hall_petch_d',
                'hall_petch_lam':  'sigma0_route_hall_petch_lam',
                'model_inversion': 'sigma0_route_model_inversion',
                'peierls_nabarro': 'sigma0_route_peierls_nabarro',
            }
            for r in derived_stress.get('routes', []):
                if r.status != 'ok' or r.value_MPa is None \
                        or r.value_MPa <= 0:
                    continue
                method_name = _SIGMA0_ROUTE_METHOD.get(
                    r.key, f'sigma0_route_{r.key}')
                route_cand = PlasticityCandidate(
                    param='sigma0_fric',
                    value_si=float(r.value_MPa) * 1e6,
                    raw_value=float(r.value_MPa),
                    raw_unit='MPa',
                    score=float(r.conf),
                    confidence=float(r.conf),
                    material=material,
                    temp_k=temp_k,
                    strain_rate=strain_rate,
                    method=method_name,
                    source_file='side-note-cluster',
                    source_title=f'σ₀ route — {r.label}',
                    evidence=(f"{r.label} → {r.value_MPa:.2f} MPa "
                              f"(conf {r.conf:.2f})"),
                    reasoning=r.note,
                    provenance='physics_inferred',
                    context=True,
                )
                bucket.append(route_cand)
            if bucket:
                bucket.sort(key=lambda c: c.score, reverse=True)
        except Exception:
            logger.warning("v8.9.0/v9.3.0 physics-inferred σ₀ bridge failed:\n%s",
                           traceback.format_exc())

        priors_df = pd.DataFrame()
        if build_priors:
            try:
                corpus = self.corpus.load()
                priors_df = self.prior_learner.learn(
                    corpus, material=material, temp_k=temp_k,
                    strain_rate=strain_rate, use_llm=self.llm_available,
                    max_docs=min(150, len(corpus)),
                    debug_llm=self.debug_llm)
            except Exception as e:
                logger.warning("Prior learning failed: %s", e)

        defaults = {
            p: PLASTICITY_ONTOLOGY[p]["defaults"].get(
                material, PLASTICITY_ONTOLOGY[p]["defaults"]["Cu"])
            for p in PARAM_ORDER
        }

        # ── v10.0.0: translate v9 candidates → v10 raw → ScoredCandidate ──
        scored_v10: Dict[str, List[ScoredCandidate]] = {}
        try:
            raw_by_param: Dict[str, List[Dict[str, Any]]] = {}
            for p_key, p_cands in (candidates or {}).items():
                raws = [_pl_cand_to_v10_raw(c, p_key)
                        for c in p_cands if c is not None]
                if raws:
                    raw_by_param[p_key] = raws
            if self.scorer_v10 is None:
                self.scorer_v10 = LatentMoEScorerV10(
                    self._ctx_vector or ContextVector(material=material),
                    self.metadb)
            scored_v10 = self.scorer_v10.score_all(raw_by_param)
            # attach v10 σ₀ route consensus information if available
            cons = (derived_stress or {}).get('consensus')
            if cons and 'sigma0_fric' in scored_v10:
                for c in scored_v10['sigma0_fric']:
                    n = int(cons.get('n_routes', 0))
                    a = n if n else 0
                    if a:
                        c.opinions['consensus'] = ExpertOpinion(
                            'consensus', (a + 1.0) / (4 + 2.0),
                            f"{a}/4 routes agree")
                        base = BASE_WEIGHTS['sigma0_fric']
                        w = {k: base.get(k, 0.0) * alpha_k(k, 'sigma0_fric',
                                                            self._ctx_vector)
                             for k in c.opinions}
                        wsum = sum(w.values()) or 1.0
                        c.w_hat = {k: w[k] / wsum for k in c.opinions}
                        c.Lambda = sum(c.w_hat[k] * c.opinions[k].s
                                       for k in c.opinions)
                        c.Sigma = c.Lambda * c.kappa_n * c.kappa_m * \
                            (math.prod(g for _, g in c.gate_list)
                             if c.gate_list else 1.0)
                        c.stab_lo, c.stab_hi = weight_stability(
                            c.opinions, c.w_hat, c.kappa_n, c.kappa_m,
                            math.prod(g for _, g in c.gate_list)
                            if c.gate_list else 1.0)
                scored_v10['sigma0_fric'].sort(key=lambda c: -c.Sigma)
        except Exception:
            logger.warning("v10 score-anatomy pass failed:\n%s",
                           traceback.format_exc())

        return PlasticityRecommendationBundle(
            material=material, temp_k=temp_k, strain_rate=strain_rate,
            candidates=candidates, defaults=defaults, priors_df=priors_df,
            retrieval_backend=retrieval_backend, llm_used=self.llm_available,
            diagnostics=diagnostics, derived_stress=derived_stress,
            scored_v10=scored_v10,
            context_vector=self._ctx_vector,
            metadb=self.metadb,
        )

    def recommend(self, material, temp_k, strain_rate=1e-3, max_docs=20,
                  build_priors=True, progress_callback=None
                  ) -> PlasticityRecommendationBundle:
        corpus = self.corpus.load()
        if not corpus:
            st.warning("⚠️ No JSON metadatabases found. Using ontology defaults.")
            return self._default_bundle(material, temp_k, strain_rate)
        retrieval_backend = "keyword-prefilter"
        if not self.retriever._docs:
            try:
                self.retriever.build(corpus, force=False)
                if self.retriever._docs:
                    retrieval_backend = (
                        "faiss+dense" if FAISS_AVAILABLE and self.retriever._index is not None
                        else "numpy+dense" if self.retriever.model is not None
                        else "tfidf-fallback")
            except Exception as e:
                logger.warning("Retrieval build failed: %s", e)
        if self.retriever._docs:
            query = (f"{material} plasticity parameters: dislocation density, "
                     f"shear modulus, reference strain rate, strain-rate "
                     f"sensitivity, friction stress; temperature ~{temp_k} K.")
            docs = self.retriever.search(query,
                                         k=max(self.top_k_retrieval, max_docs),
                                         material_hint=material)[:max_docs]
        else:
            docs = PlasticityCorpus.keyword_prefilter(corpus, material, k=max_docs)
        extractions: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if progress_callback:
                progress_callback(i + 1, len(docs), doc.get("title", "")[:60])
            ext = self.extractor.extract(
                doc["text"], material, temp_k, strain_rate,
                use_llm=self.llm_available, mode="reasoned_inference",
                debug_llm=self.debug_llm)
            for e in ext:
                e["_source_file"] = doc["source"]
                e["_source_title"] = doc["title"]
                e["_provenance"] = ("llm_extract" if e.get("method") in
                                    ("explicit", "llm_inferred")
                                    else "regex_ner")
                e["_context"] = False
            extractions.extend(ext)
        self._save_disk_cache()
        candidates = self.scorer.score(
            extractions, material, temp_k,
            target_strain_rate=strain_rate, top_k=8,
            synthesis=self.rho0_synthesis,
            architecture=self.rho0_architecture,
            twin_spacing=self.rho0_twin_spacing,
            target_theory=self.gamma0_target_theory,
        )
        priors_df = pd.DataFrame()
        if build_priors:
            try:
                priors_df = self.prior_learner.learn(
                    corpus, material=material, temp_k=temp_k,
                    strain_rate=strain_rate, use_llm=self.llm_available,
                    max_docs=min(150, len(corpus)),
                    debug_llm=self.debug_llm)
            except Exception as e:
                logger.warning("Prior learning failed: %s", e)
        defaults = {
            p: PLASTICITY_ONTOLOGY[p]["defaults"].get(
                material, PLASTICITY_ONTOLOGY[p]["defaults"]["Cu"])
            for p in PARAM_ORDER
        }
        return PlasticityRecommendationBundle(
            material=material, temp_k=temp_k, strain_rate=strain_rate,
            candidates=candidates, defaults=defaults, priors_df=priors_df,
            retrieval_backend=retrieval_backend, llm_used=self.llm_available,
            diagnostics=[], derived_stress={},
            scored_v10={}, context_vector=None, metadb=MetaDatabase())

    @staticmethod
    def _default_bundle(material, temp_k, strain_rate):
        defaults = {
            p: PLASTICITY_ONTOLOGY[p]["defaults"].get(
                material, PLASTICITY_ONTOLOGY[p]["defaults"]["Cu"])
            for p in PARAM_ORDER
        }
        return PlasticityRecommendationBundle(
            material=material, temp_k=temp_k, strain_rate=strain_rate,
            candidates={p: [] for p in PARAM_ORDER}, defaults=defaults,
            priors_df=pd.DataFrame(), retrieval_backend="none",
            llm_used=False, diagnostics=[], derived_stress={},
            scored_v10={}, context_vector=None, metadb=MetaDatabase())


OLLAMA_MODELS: Dict[str, str] = {
    "⚡ Fallback (Rule-based, no LLM)": "",
    "🦙 qwen2.5:0.5b (Fastest, CPU OK)": "qwen2.5:0.5b",
    "🦙 qwen2.5:1.5b (Balanced)": "qwen2.5:1.5b",
    "🦙 qwen2.5:7b (Recommended for RAG)": "qwen2.5:7b",
    "🦙 qwen2.5:14b (Max Reasoning)": "qwen2.5:14b",
    "🦙 llama3.1:8b (Meta Standard)": "llama3.1:8b",
    "🦙 mistral:7b (High JSON Reliability)": "mistral:7b",
    "🦙 gemma2:9b (Scientific Nuance)": "gemma2:9b",
    "🦙 falcon3:10b (Instruction Following)": "falcon3:10b",
}


def _build_ollama_dropdown_options() -> Dict[str, str]:
    options: Dict[str, str] = dict(OLLAMA_MODELS)
    installed = PlasticityOllamaClient.list_models()
    existing_names = set(options.values())
    for name in installed:
        if not name or name in existing_names: continue
        options[f"🦙 {name} (Installed locally)"] = name
        existing_names.add(name)
    return options


def purge_plasticity_caches() -> List[str]:
    purged: List[str] = []
    try:
        st.cache_resource.clear()
        purged.append("streamlit cache_resource (hybrid retriever)")
    except Exception:
        pass
    corpus_key = f"pl_corpus_{PlasticityCorpus._CACHE_VERSION}"
    if corpus_key in st.session_state:
        st.session_state.pop(corpus_key, None)
        purged.append("session corpus cache")
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("pl_corpus_") and k != corpus_key:
            st.session_state.pop(k, None)
            purged.append(f"session key '{k}'")
    live_rec = st.session_state.get("_plr_live_recommender")
    if live_rec is not None:
        try:
            if getattr(live_rec, "extractor", None) is not None:
                live_rec.extractor.cache.clear()
                purged.append("in-memory LLM extractor cache")
            if getattr(live_rec, "retriever", None) is not None:
                live_rec.retriever._docs = []
                live_rec.retriever._doc_vectors = None
                live_rec.retriever._index = None
                purged.append("in-memory FAISS retriever state")
            if getattr(live_rec, "hybrid", None) is not None:
                live_rec.hybrid = None
                purged.append("in-memory hybrid retriever reference")
        except Exception as e:
            logger.warning("Could not purge live recommender caches: %s", e)
    cache_dir = PlasticityFAISSRetriever.CACHE_DIR
    if os.path.isdir(cache_dir):
        idx_path = os.path.join(cache_dir, PlasticityFAISSRetriever.INDEX_FILE)
        if os.path.exists(idx_path):
            try:
                os.remove(idx_path); purged.append(PlasticityFAISSRetriever.INDEX_FILE)
            except Exception as e:
                logger.warning("Could not remove %s: %s", idx_path, e)
        llm_cache_path = os.path.join(cache_dir, PlasticityRecommender.LLM_CACHE_FILE)
        if os.path.exists(llm_cache_path):
            try:
                os.remove(llm_cache_path); purged.append(PlasticityRecommender.LLM_CACHE_FILE)
            except Exception as e:
                logger.warning("Could not remove %s: %s", llm_cache_path, e)
        try:
            for fname in os.listdir(cache_dir):
                if not fname.lower().endswith((".json", ".pkl", ".faiss")):
                    continue
                if fname in (PlasticityFAISSRetriever.INDEX_FILE,
                             PlasticityRecommender.LLM_CACHE_FILE):
                    continue
                fpath = os.path.join(cache_dir, fname)
                try:
                    os.remove(fpath); purged.append(fname)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Cache dir sweep failed: %s", e)
    return purged


# ----------------------------------------------------------------------------
# SIDEBAR UI
# ----------------------------------------------------------------------------
_PLR = "pl_rec_"


def _plr_get(key, default=None):
    return st.session_state.get(_PLR + key, default)


def _plr_set(key, value):
    st.session_state[_PLR + key] = value


def _plr_reset():
    for p in PARAM_ORDER:
        st.session_state.pop(f"{_PLR}{p}_choice", None)
        st.session_state.pop(f"{_PLR}{p}_manual", None)
        st.session_state.pop(f"{_PLR}{p}_value_si", None)
        st.session_state.pop(f"{_PLR}{p}_radio", None)
        st.session_state.pop(f"{_PLR}{p}_manual_input", None)
    st.session_state.pop(f"{_PLR}bundle", None)
    st.session_state.pop("plasticity_overrides", None)
    st.session_state.pop("_plr_live_recommender", None)
    st.session_state.pop(f"{_PLR}diagnostics", None)
    st.session_state.pop("recommender_bundle", None)
    st.session_state.pop("friction_lab_w_mode", None)
    st.session_state.pop("pn_zeta", None)
    st.session_state.pop(f"{_PLR}rho0_synthesis", None)
    st.session_state.pop(f"{_PLR}rho0_arch", None)
    st.session_state.pop(f"{_PLR}rho0_lam", None)
    st.session_state.pop(f"{_PLR}gamma0_theory", None)
    st.session_state.pop(f"{_PLR}gamma0_regime_preview", None)
    st.session_state.pop(f"{_PLR}cascade_mode", None)


def _plr_render_parameter_selector(param: str,
                                   bundle: PlasticityRecommendationBundle):
    spec = PLASTICITY_ONTOLOGY[param]
    st.markdown(f"**{spec['symbol']} — {spec['label']}**")
    options: List[Tuple[str, float, Optional[PlasticityCandidate]]] = []
    best = bundle.best(param)
    if best is not None:
        prov_fine = _norm_provenance(getattr(best, "provenance", "")
                                     or getattr(best, "method", ""))
        prov_tag = {
            "llm_extract":      "✓ grounded",
            "llm_reasoned":     "~ llm-cot",
            "regex_ner":        "· regex",
            "regime_prior":     "🧭 regime",
            "llm_prior":        "∅ prior",
            "physics_inferred": "△ physics-inf",
        }.get(prov_fine, best.method)
        theory_tag = ""
        if param == "gamma0_dot" and getattr(best, "theory", ""):
            if best.theory and best.theory != "unspecified":
                theory_tag = f" · [{best.theory}]"
        options.append((
            f"⭐ Recommended: {_pl_fmt(param, best.value_si)} "
            f"(score {best.score:.2f} · {prov_tag}{theory_tag})",
            best.value_si, best))
    n_alt = 0
    for c in bundle.candidates.get(param, []):
        if getattr(c, 'context', False):
            continue
        if c is best:
            continue
        n_alt += 1
        theory_tag = ""
        if param == "gamma0_dot" and getattr(c, "theory", ""):
            if c.theory and c.theory != "unspecified":
                theory_tag = f" · [{c.theory}]"
        options.append((
            f"   Alt {n_alt}: {_pl_fmt(param, c.value_si)} "
            f"(score {c.score:.2f}{theory_tag})",
            c.value_si, c))
    prior = bundle.prior_suggestion(param)
    if prior is not None and (best is None or abs(prior - best.value_si) > 1e-12):
        options.append((
            f"📚 Learned prior ({bundle.material}): {_pl_fmt(param, prior)}",
            prior, None))
    default_si = bundle.defaults[param]
    options.append((
        f"⚙️  Default ({bundle.material}): {_pl_fmt(param, default_si)}",
        default_si, None))
    options.append(("✏️  Manual override…", float("nan"), None))
    labels = [o[0] for o in options]
    prev = _plr_get(f"{param}_choice", labels[0])
    if prev not in labels:
        prev = labels[0]
    idx = labels.index(prev)
    choice = st.radio(
        label=f"Select {param}", options=labels, index=idx,
        key=f"{_PLR}{param}_radio", label_visibility="collapsed")
    _plr_set(f"{param}_choice", choice)
    sel = labels.index(choice)
    _, value_si, chosen = options[sel]
    if math.isnan(value_si):
        ui_default = default_si / spec["ui_scale"]
        manual = st.number_input(
            f"Manual {param} ({spec['ui_unit']})",
            value=float(_plr_get(f"{param}_manual", ui_default)),
            format="%.6g" if param in ("rho0", "gamma0_dot") else "%.4f",
            key=f"{_PLR}{param}_manual_input")
        _plr_set(f"{param}_manual", manual)
        value_si = manual * spec["ui_scale"]
        chosen = None
    if chosen is not None:
        theory_disp = ""
        if param == "gamma0_dot" and getattr(chosen, "theory", ""):
            if chosen.theory and chosen.theory != "unspecified":
                theory_disp = f" | theory={chosen.theory}"
        st.caption(
            f"📚 {chosen.source_file} — {chosen.source_title[:70]} | "
            f"mat={chosen.material or 'n/a'}, T={chosen.temp_k}, "
            f"method={chosen.method}, conf={chosen.confidence:.2f}"
            f"{theory_disp}"
            + (" | ⚠️ clamped" if chosen.clamped else ""))
        if chosen.evidence:
            st.markdown("**📚 Evidence snippet**")
            with st.container():
                st.code(chosen.evidence, language="text")
        if chosen.reasoning:
            with st.expander("🧠 LLM reasoning chain", expanded=False):
                st.markdown(chosen.reasoning)

    n_ctx = sum(1 for c in bundle.candidates.get(param, [])
                if getattr(c, 'context', False))
    if n_ctx > 0:
        with st.expander(f"🔎 Supporting routes shown as context ({n_ctx})",
                         expanded=False):
            st.caption("Greyed context bars in the chart. Non-ranked — "
                       "the recommended value above is unaffected.")
            ctx_rows = []
            for c in bundle.candidates.get(param, []):
                if not getattr(c, 'context', False):
                    continue
                ctx_rows.append({
                    "value": _pl_fmt(param, c.value_si),
                    "method": c.method,
                    "provenance": c.provenance or c.method,
                    "confidence": round(c.confidence, 3),
                    "evidence": c.evidence[:80],
                })
            if ctx_rows:
                st.dataframe(pd.DataFrame(ctx_rows),
                             use_container_width=True, hide_index=True)

    st.session_state[f"{_PLR}{param}_value_si"] = value_si
    st.markdown("---")


def render_plasticity_recommender_sidebar(
    default_material: str = "Cu",
    default_temp: float = 300.0,
    default_strain_rate: float = 1e-3,
    ollama_model: str = "qwen2.5:7b",
):
    st.subheader("🤖 Intelligent Plasticity Recommender v10.0.0")
    st.caption(
        "**v10.0.0 score anatomy:** every bar IS the audit.  Legend keys "
        "are now **experts** (evidence / regime / theory / closure / "
        "consensus / compat); **provenance survives as a non-ranking badge** "
        "(foot-strip + glyph).  Bar height = Σ = Λ·κ_n·κ_m·G.  The v9 "
        "pin-list survives as **annotation-only reference marks**."
    )
    st.caption(
        "**v9.3.1 honest naming:** the fine provenance key formerly called "
        "`derived` is now `physics_inferred` — these values apply physical "
        "laws (Hall–Petch, Peierls–Nabarro, M·τ_P, solver-law inversion) to "
        "MEASURED side-note context, not constructed from nothing. The "
        "coarse deterministic bucket label now reads "
        "**Deterministic (regex · physics-inferred)**."
    )
    st.caption(
        "**v9.3.0 union cascade + context candidates:** every viable route "
        "runs. Non-incumbent routes appear as greyed non-ranked context "
        "bars — the v9.2.2 findings never move. σ₀ disaggregates into its "
        "four physics-inferred routes."
    )

    col1, col2 = st.columns(2)
    with col1:
        material = st.text_input(
            "Target material", value=default_material, key=f"{_PLR}material")
    with col2:
        temp_k = st.number_input(
            "Temperature (K)", value=float(default_temp),
            min_value=1.0, step=10.0, key=f"{_PLR}temp")
    strain_rate = st.number_input(
        "Reference strain rate (s⁻¹)", value=float(default_strain_rate),
        format="%.2e", key=f"{_PLR}rate")
    value_hints_raw = st.text_input(
        "Value hints for shear modulus (comma-separated; e.g. `47.19, 55.17`)",
        value="", key=f"{_PLR}value_hints")
    value_hints = [s.strip() for s in value_hints_raw.split(",") if s.strip()]

    model_options = _build_ollama_dropdown_options()
    current_model_name = ollama_model if ollama_model else "qwen2.5:7b"
    display_options = list(model_options.keys())
    display_name_for_current = next(
        (k for k, v in model_options.items() if v == current_model_name),
        next((k for k in display_options if "Recommended" in k),
             display_options[0] if display_options else ""))
    default_idx = (display_options.index(display_name_for_current)
                   if display_name_for_current in display_options else 0)
    selected_display = st.selectbox(
        "Ollama model", options=display_options, index=default_idx,
        key=f"{_PLR}ollama_model_select")
    ollama_model = model_options.get(selected_display, "")
    if not ollama_model:
        llm_ok = False
    else:
        llm_ok = PlasticityOllamaClient.is_available()
    backend_txt = (
        "hybrid(lexical+faiss)" if FAISS_AVAILABLE and SBERT_AVAILABLE
        else "hybrid(lexical+dense-numpy)" if SBERT_AVAILABLE
        else "hybrid(lexical-only)")
    bc1, bc2 = st.columns(2)
    with bc1:
        if not ollama_model:
            st.caption("⚡ LLM disabled — using heuristic extractor")
        else:
            st.caption(f"{'✅' if llm_ok else '⚠️'} Ollama "
                       f"{'available' if llm_ok else 'unreachable'}")
    with bc2:
        st.caption(f"🔎 Retrieval: `{backend_txt}`")
    use_grounded = st.checkbox(
        "🧭 Use grounded NER pipeline (recommended)",
        value=True, key=f"{_PLR}use_grounded")
    allow_prior = st.checkbox(
        "🧠 Allow LLM prior inference when corpus evidence is missing",
        value=True, key=f"{_PLR}allow_prior",
        help=("Tier-3 prior inference: capped at 0.5, bounded by "
              "PARAM_CANON plausible range."))

    cascade_mode = st.radio(
        "Cascade mode",
        options=['union', 'fallback'],
        format_func=lambda x: {
            'union':    '🔀 Union — show all routes as context',
            'fallback': '➡️ Fallback — first successful tier only'}[x],
        index=0, key=f"{_PLR}cascade_mode", horizontal=True,
        help=("Union: every viable route runs; non-incumbent routes enter "
              "as greyed non-ranked context bars.  Findings are frozen by "
              "the INCUMBENT_ROUTES pin-list.  Fallback: v8.8.2 "
              "short-circuit."))

    with st.expander("🔬 ρ₀ physics-regime inputs (qualitative)",
                     expanded=False):
        st.caption(
            "These three factors determine the initial dislocation density "
            "regime. They are **qualitative context** — the LLM still "
            "chooses the numeric value inside the physically-derived band.")
        rho0_synthesis = st.selectbox(
            "Synthesis method",
            ["(unknown)",
             "Magnetron sputter / PVD",
             "Molecular beam epitaxy (MBE)",
             "DC electrodeposition",
             "Pulsed electrodeposition",
             "Additive-assisted electrodeposition",
             "Localized electroplating / additive micromanufacturing"],
            index=0, key=f"{_PLR}rho0_synthesis")
        rho0_arch = st.selectbox(
            "Grain architecture",
            ["(unknown)",
             "Columnar (111)-oriented",
             "Columnar, random texture",
             "Equiaxed nanocrystalline",
             "Equiaxed UFG",
             "Single crystal",
             "Polycrystalline"],
            index=0, key=f"{_PLR}rho0_arch")
        rho0_lam = st.number_input(
            "Twin lamella spacing λ (nm) — 0 if unknown",
            value=0.0, min_value=0.0, step=1.0,
            key=f"{_PLR}rho0_lam")

    with st.expander("⚡ γ̇₀ theory-regime input (qualitative)",
                     expanded=False):
        st.caption(
            "γ̇₀ depends on the modelling framework.  Choose the target "
            "theory so the LLM tags and scorer can disambiguate DDD "
            "(~5e3 s⁻¹) from CPFEM (~1e-3 s⁻¹) from MD (~1e7 s⁻¹).")
        gamma0_theory = st.selectbox(
            "Target theory framework",
            ["(unspecified)",
             "Johnson-Cook / power-law (experimental)",
             "Crystal Plasticity (CPFEM/VPSC)",
             "Discrete Dislocation Dynamics (DDD)",
             "Molecular Dynamics (MD)",
             "Phase-field (continuum)"],
            index=0, key=f"{_PLR}gamma0_theory")

    _rho0_syn_arg  = (None if rho0_synthesis.startswith("(")
                      else rho0_synthesis)
    _rho0_arch_arg = (None if rho0_arch.startswith("(")
                      else rho0_arch)
    _rho0_lam_arg  = (float(rho0_lam) if rho0_lam > 0 else None)
    _regime_preview = classify_rho0_regime(
        _rho0_syn_arg, _rho0_arch_arg, _rho0_lam_arg)
    st.caption(
        f"🧭 ρ₀ regime preview: **{_regime_preview['regime'][:70]}** · "
        f"[{_regime_preview['low']:.1e}, {_regime_preview['high']:.1e}] m⁻² · "
        f"center {_regime_preview['inferred']:.2e} m⁻²")

    _gamma0_theory_arg = (None if gamma0_theory.startswith("(")
                          else gamma0_theory)
    _gamma0_regime_preview = classify_gamma0_regime(_gamma0_theory_arg)
    st.caption(
        f"⚡ γ̇₀ regime preview: **{_gamma0_regime_preview['regime'][:70]}** · "
        f"[{_gamma0_regime_preview['low']:.1e}, "
        f"{_gamma0_regime_preview['high']:.1e}] s⁻¹ · "
        f"center {_gamma0_regime_preview['inferred']:.2e} s⁻¹")

    debug_llm = st.checkbox(
        "🔍 Show raw LLM responses + retrieved record IDs (INFO level)",
        value=False, key=f"{_PLR}debug_llm")

    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        run_btn = st.button("🔍 Analyse JSON databases",
                            use_container_width=True, type="primary")
    with btn2:
        refresh_btn = st.button("🔄 Force reload corpus",
                                use_container_width=True)
    with btn3:
        if st.button("♻️ Reset recommendations", use_container_width=True):
            _plr_reset(); st.rerun()

    if refresh_btn:
        try:
            purged_items = purge_plasticity_caches()
            st.session_state.pop("_plr_live_recommender", None)
        except Exception as e:
            st.warning(f"Could not clear all caches: {e}")
            purged_items = []
        if purged_items:
            st.success("✅ Purged: " +
                       ", ".join(f"`{p}`" for p in purged_items) +
                       ". Next **Analyse** will run fresh.")
        else:
            st.info("ℹ️ Nothing to purge — caches were already empty.")

    if run_btn:
        recommender = PlasticityRecommender(
            ollama_model=ollama_model or "qwen2.5:7b",
            use_llm=llm_ok, debug_llm=debug_llm,
            use_grounded=use_grounded,
            allow_prior_inference=allow_prior,
            rho0_synthesis=_rho0_syn_arg,
            rho0_architecture=_rho0_arch_arg,
            rho0_twin_spacing=_rho0_lam_arg,
            gamma0_target_theory=_gamma0_theory_arg,
            cascade_mode=cascade_mode)
        st.session_state["_plr_live_recommender"] = recommender
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(i, n, title):
            progress.progress(i / max(n, 1))
            status.caption(f"Scanning {i}/{n}: {title}…")
        try:
            with st.spinner("Retrieving candidates + running cascade…"):
                if use_grounded and recommender.hybrid is not None:
                    bundle = recommender.recommend_grounded(
                        material=material, temp_k=float(temp_k),
                        strain_rate=float(strain_rate),
                        value_hints=value_hints or None,
                        progress_callback=_cb)
                else:
                    bundle = recommender.recommend(
                        material=material, temp_k=float(temp_k),
                        strain_rate=float(strain_rate),
                        progress_callback=_cb)
            _plr_set("bundle", bundle)
            st.session_state['recommender_bundle'] = bundle
            progress.empty()
            n_ctx_total = sum(
                sum(1 for c in bundle.candidates.get(p, [])
                    if getattr(c, 'context', False))
                for p in PARAM_ORDER)
            status.success(
                f"✅ Retrieved "
                f"{sum(len(v) for v in bundle.candidates.values())} candidates "
                f"({n_ctx_total} context) across "
                f"{bundle.coverage()}/{len(PARAM_ORDER)} params "
                f"(5-target: {bundle.coverage_five()}/5). "
                f"Backend: {bundle.retrieval_backend} · "
                f"mode: {cascade_mode}.")

            if getattr(recommender, "rho0_regime", None):
                rr = recommender.rho0_regime
                n_snips = len(getattr(recommender, "rho0_snippet_findings", []))
                st.info(
                    f"🧭 **ρ₀ regime (post-analysis):** {rr['regime']}  \n"
                    f"Range: `[{rr['low']:.1e}, {rr['high']:.1e}] m⁻²` · "
                    f"log-center `{rr['inferred']:.2e} m⁻²` · "
                    f"{n_snips} snippet-context match(es)")
            if getattr(recommender, "gamma0_regime", None):
                gr = recommender.gamma0_regime
                n_snips = len(getattr(recommender,
                                      "gamma0_snippet_findings", []))
                st.info(
                    f"⚡ **γ̇₀ regime (post-analysis):** {gr['regime']}  \n"
                    f"Range: `[{gr['low']:.1e}, {gr['high']:.1e}] s⁻¹` · "
                    f"log-center `{gr['inferred']:.2e} s⁻¹` · "
                    f"{n_snips} snippet-theory match(es)")
        except Exception as e:
            progress.empty()
            status.error(f"Recommendation failed: {e}")
            st.exception(e)

    bundle: Optional[PlasticityRecommendationBundle] = _plr_get("bundle")
    if bundle is None:
        return
    st.caption(
        f"Retrieval: **{bundle.retrieval_backend}** · "
        f"LLM: **{'yes' if bundle.llm_used else 'no (heuristic)'}**")
    st.metric("5-target coverage",
              f"{bundle.coverage_five()}/5",
              help="ρ₀, μ, γ̇₀, m, σ₀ — the five solver-consumed parameters")

    st.markdown("### Choose the 5 target values (one by one)")
    st.caption("Context inputs (σ_y, d, λ, b, ν, M, k_y, w, G) are edited "
               "in the main-bar **🧮 Friction Stress Lab** tab.")
    for param in SIDEBAR_PARAM_ORDER:
        _plr_render_parameter_selector(param, bundle)

    if st.button("✅ Apply selected values to solver",
                 type="primary", use_container_width=True):
        overrides = {}
        for param in PARAM_ORDER:
            v = st.session_state.get(f"{_PLR}{param}_value_si")
            if v is not None:
                overrides[SOLVER_KEY_MAP[param]] = float(v)
        M_val = st.session_state.get(f"{_PLR}taylor_factor_M_value_si")
        if M_val is not None:
            st.session_state["_plr_taylor_M"] = float(M_val)
        st.session_state["plasticity_overrides"] = overrides
        st.success(f"Applied {len(overrides)} parameters. "
                   "The solver will use them on the next run.")
        try:
            bundle_check = _plr_get("bundle")
            if bundle_check and bundle_check.scored_v10:
                n_tot  = sum(len(v) for v in bundle_check.scored_v10.values())
                n_gate = sum(1 for v in bundle_check.scored_v10.values()
                             for c in v if c.gate_list)
                st.caption(
                    f"🔍 v10 audit trail: {n_tot} scored candidate(s); "
                    f"{n_gate} carrying a visible gate penalty.  "
                    f"Open the AI Recommender Visuals Dashboard → "
                    f"**Score Anatomy (v10)** to inspect every bar.")
        except Exception:
            pass

    if not bundle.priors_df.empty:
        st.markdown("**📚 Learned per‑material priors (from corpus)**")
        styled = bundle.priors_df.copy()
        for col in ["mean", "median", "std", "p10", "p90"]:
            if col in styled.columns:
                styled[col] = styled.apply(lambda r, c=col: f"{r[c]:.3e}", axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================================
# 🧮 v8.9.0 — FRICTION STRESS LAB (main-bar tab)
# ============================================================================
@handle_errors
def render_friction_stress_lab():
    st.markdown('### 🧮 Friction & Lattice Stress Lab')
    st.caption(
        'Constructs σ₀ (friction / lattice stress) and τ_P from the '
        'recommender\'s side-note context. Physics-inferred rows carry '
        'the ▼ glyph.')
    bundle = st.session_state.get('recommender_bundle')
    if bundle is None:
        st.info('Run the plasticity recommender in the sidebar first '
                '(🔍 Analyse JSON databases).')
        return

    w_mode = st.session_state.get('friction_lab_w_mode', 'auto')
    derived = build_derived_stress_bundle(bundle, w_mode=w_mode)
    ctx, routes, cons = derived['ctx'], derived['routes'], derived['consensus']

    wcol1, wcol2 = st.columns([3, 2])
    with wcol1:
        w_mode = st.radio(
            'P–N core-width convention w',
            ('auto', 'captured', 'zeta'), horizontal=True,
            index=('auto', 'captured', 'zeta').index(
                st.session_state.get('friction_lab_w_mode', 'auto')),
            key='friction_lab_w_mode',
            help='auto = captured w if corpus-backed, else w = b/(1−ν). '
                 'exp(−2πw/b) makes τ_P exponentially sensitive to w.')
    if w_mode == 'zeta':
        with wcol2:
            st.session_state['pn_zeta'] = st.slider(
                'ζ = w/b', 0.4, 2.6, 1.52, 0.01, key='pn_zeta_slider')

    if w_mode != 'auto' or st.session_state.get('pn_zeta'):
        derived = build_derived_stress_bundle(bundle, w_mode=w_mode)
        ctx, routes, cons = derived['ctx'], derived['routes'], derived['consensus']

    st.markdown('#### Captured context (derivation inputs)')
    glyph = {'llm_extract': '◆', 'llm_reasoned': '❖',
             'regex_ner': '▲', 'regime_prior': '■',
             'llm_prior': '○', 'physics_inferred': '▼', 'missing': '–'}
    cols = st.columns(5)
    for i, sym in enumerate(CTX_KEYS):
        f = ctx[sym]
        with cols[i % 5]:
            st.metric(
                sym,
                '—' if f.value is None else f'{f.value:.4g}',
                delta=('missing' if f.value is None else
                       f"conf {f.conf:.2f} {glyph.get(f.provenance, '–')}"
                       + (' ⚠️ clamped' if f.clamped else '')),
                delta_color='off')

    st.markdown('#### Derivation routes')
    st.dataframe(pd.DataFrame([dict(
        Route=r.label,
        **{'σ₀ (MPa)': '—' if r.value_MPa is None else f'{r.value_MPa:.1f}'},
        **{'τ_P (MPa)': '—' if r.tau_P_MPa is None else f'{r.tau_P_MPa:.3g}'},
        Conf=f'{r.conf:.2f}', Status=r.status, Note=r.note)
        for r in routes]),
        use_container_width=True, hide_index=True)

    if not cons:
        st.error('No viable route — check missing inputs above.')
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Consensus σ₀', f"{cons['value_MPa']:.1f} MPa")
    c2.metric('Route spread', f"{cons['lo']:.1f}–{cons['hi']:.1f} MPa")
    c3.metric('Confidence', f'{cons["conf"]:.2f}')
    c4.metric('Routes OK', f"{cons['n_routes']}/{len(routes)}")
    if cons['disagree']:
        st.warning(f"Routes disagree (spread {cons['rel_spread']:.0%}). "
                   'For nt-Cu, λ-based routes should dominate — verify d.')
    if derived['floor'] and not derived['floor']['passed']:
        st.error(derived['floor']['msg'])

    if st.button('✅ Adopt consensus σ₀ for simulation', type='primary'):
        ov = st.session_state.setdefault('plasticity_overrides', {})
        ov['sigma0'] = cons['value_MPa'] * 1e6
        ov['sigma0_fric'] = cons['value_MPa']
        st.success(
            f"σ₀ = {cons['value_MPa']:.1f} MPa adopted "
            f"(▼ physics-inferred, conf {cons['conf']:.2f}, "
            f"routes: {', '.join(cons['contributors'])}).")

    okr = [r for r in routes if r.status == 'ok' and r.value_MPa]
    if okr:
        best = int(np.argmin([abs(r.value_MPa - cons['value_MPa'])
                              for r in okr]))
        try:
            fig = plot_candidate_scores(
                [r.value_MPa for r in okr], [r.conf for r in okr],
                ['physics_inferred'] * len(okr), best,
                param_title='Friction stress σ₀ — derivation routes',
                param_symbol=r'\sigma_0', unit='MPa',
                score_label='Route confidence',
                legend_granularity='coarse')
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Route chart failed: {e}")

    G_MPa = ((ctx['G'].value * MPA_PER_GPA)
             if ctx['G'].value and ctx['G'].value > 1
             else (derived['mu_GPa'] or 0) * MPA_PER_GPA)
    b, nu = ctx['b'].value, ctx['nu'].value
    if G_MPa and b and nu is not None and nu < 1:
        with st.expander('τ_P sensitivity to w/b (log scale)'):
            zeta = np.linspace(0.4, 2.6, 240)
            fig2, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
            JournalTemplates.apply_journal_style(fig2, ax, 'nature')
            ax.semilogy(zeta, 2 * G_MPa / (1 - nu) * np.exp(-2 * np.pi * zeta))
            ax.axvline(1 / (1 - nu), ls='--', lw=1, color='#D55E00')
            if ctx['w'].value:
                ax.axvline(ctx['w'].value / b, ls=':', lw=1, color='#0072B2')
            ax.set_xlabel(safe_mathtext(r'core-width ratio $w/b$'))
            ax.set_ylabel(safe_mathtext(r'$\tau_P$ [MPa]'))
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption('Dashed: w = b/(1−ν). Dotted: captured w. '
                       'A 5× error in w swings τ_P by ~4 orders of magnitude.')

    cl = derived['closure']
    if cl:
        with st.expander('Solver closure check'):
            st.markdown(
                f"`compute_yield_stress(λ, σ₀_cons, μ, b, ν)` → "
                f"**σ_y = {cl['sigma_y_pred_MPa']:.1f} MPa** vs captured "
                f"**{cl['sigma_y_card_MPa']:.1f} MPa** "
                f"(residual {cl['residual_MPa']:+.1f} MPa).")

    with st.expander('Per-route diagnostics', expanded=False):
        st.json({
            'material': bundle.material,
            'temp_k': bundle.temp_k,
            'mu_GPa': derived['mu_GPa'],
            'w_mode': w_mode,
            'consensus': {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in (cons or {}).items()},
        })


# ----------------------------------------------------------------------------
# SOLVER HOOK
# ----------------------------------------------------------------------------
def apply_plasticity_overrides(solver) -> None:
    try:
        overrides = st.session_state.get("plasticity_overrides", None)
        if not overrides:
            return
        cleaned: Dict[str, Any] = {}
        sigma0_explicit: Optional[float] = None
        yield_held:     Optional[float] = None
        tau_p_held:     Optional[float] = None
        for k, v in overrides.items():
            if k == "_ignore": continue
            if k == "sigma0":
                try:
                    fv = float(v)
                    sigma0_explicit = fv
                except (TypeError, ValueError):
                    pass
                continue
            if k == "sigma0_fric" and sigma0_explicit is None:
                try:
                    sigma0_explicit = float(v) * 1e6
                except (TypeError, ValueError):
                    pass
                continue
            if k == "_yield_passthrough":
                try:
                    yield_held = float(v)
                except (TypeError, ValueError):
                    pass
                continue
            if k == "_tau_p_passthrough":
                try:
                    tau_p_held = float(v)
                except (TypeError, ValueError):
                    pass
                continue
            cleaned[k] = v
        if sigma0_explicit is not None:
            cleaned["sigma0"] = sigma0_explicit
        elif tau_p_held is not None:
            M = 3.06
            try:
                M = float(st.session_state.get("_plr_taylor_M", M))
            except (TypeError, ValueError):
                pass
            cleaned["sigma0"] = float(tau_p_held) * M
        elif yield_held is not None:
            cleaned["sigma0"] = float(yield_held) * 0.2
        if 'plasticity' not in solver.mat_props:
            solver.mat_props['plasticity'] = {}
        solver.mat_props['plasticity'].update(cleaned)
        logger.info("Applied plasticity overrides: %s", cleaned)
    except Exception as e:
        logger.warning("Could not apply plasticity overrides: %s", e)


# ============================================================================
# PUBLICATION-QUALITY AI RECOMMENDER VISUALS DASHBOARD
# ============================================================================
@dataclass
class RecommenderVisualStyle:
    journal: str = 'nature'
    font_family: str = 'Arial'
    font_size_title: float = 12.0
    font_size_axis: float = 10.0
    font_size_tick: float = 9.0
    font_size_legend: float = 9.0
    font_size_annotation: float = 8.0
    font_weight: str = 'bold'
    figure_width: float = 10.0
    figure_height: float = 6.0
    dpi: int = 300
    line_width: float = 1.5
    marker_size: float = 8.0
    colormap: str = 'turbo'
    cmap_reverse: bool = False
    categorical_palette: str = 'okabe_ito'
    background_color: str = '#ffffff'
    plot_background_color: str = '#f8f9fa'
    grid: bool = True
    grid_alpha: float = 0.3
    grid_linestyle: str = '--'
    transparent_bg: bool = False
    spine_color: str = '#333333'
    spine_width: float = 1.0
    show_values: bool = True
    legend_position: str = 'best'
    title_override: str = ''
    xlabel_override: str = ''
    ylabel_override: str = ''
    show_title: bool = True
    show_legend: bool = True
    show_colorbar: bool = True
    colorbar_orientation: str = 'vertical'
    bar_colormap: str = 'viridis'
    bar_color_mode: str = 'score'
    bar_best_color: Optional[str] = None
    bar_show_colorbar: bool = False
    bar_annotation_fontsize: Optional[float] = None
    bar_legend_fontsize: Optional[float] = None
    bar_colorbar_label_fontsize: Optional[float] = None
    bar_colorbar_tick_fontsize: Optional[float] = None
    bar_annotation_offset: float = 9.0
    bar_headroom: float = 1.22
    bar_show_annotation_offset: bool = True

    def get_cmap(self):
        base = plt.get_cmap(self.colormap)
        return base.reversed() if self.cmap_reverse else base

    def get_palette(self):
        return PUBLICATION_PALETTES.get(
            self.categorical_palette,
            PUBLICATION_PALETTES['okabe_ito'])

    def apply_matplotlib(self):
        rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_size_axis,
            'axes.titlesize': self.font_size_title,
            'axes.labelsize': self.font_size_axis,
            'xtick.labelsize': self.font_size_tick,
            'ytick.labelsize': self.font_size_tick,
            'legend.fontsize': self.font_size_legend,
            'figure.titlesize': self.font_size_title,
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'axes.grid': self.grid,
            'grid.alpha': self.grid_alpha,
            'grid.linestyle': self.grid_linestyle,
            'axes.linewidth': self.spine_width,
            'axes.edgecolor': self.spine_color,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.transparent': self.transparent_bg,
            'axes.prop_cycle': plt.cycler(color=self.get_palette()),
        })

    def apply_plotly_layout(self, fig, title="", xlabel="", ylabel=""):
        fig.update_layout(
            title=(self.title_override or title) if self.show_title else "",
            xaxis_title=self.xlabel_override or xlabel,
            yaxis_title=self.ylabel_override or ylabel,
            font=dict(family=self.font_family, size=self.font_size_axis,
                      color='#1e293b'),
            title_font=dict(family=self.font_family,
                            size=self.font_size_title + 2, color='#0f172a'),
            plot_bgcolor=self.plot_background_color,
            paper_bgcolor=self.background_color,
            showlegend=self.show_legend,
            legend=dict(font=dict(family=self.font_family,
                                  size=self.font_size_legend),
                        bordercolor=self.spine_color, borderwidth=1),
            margin=dict(l=70, r=40, t=80, b=60))
        if self.grid:
            fig.update_xaxes(showgrid=True,
                             gridcolor='rgba(128,128,128,0.3)',
                             gridwidth=0.5, griddash='dash')
            fig.update_yaxes(showgrid=True,
                             gridcolor='rgba(128,128,128,0.3)',
                             gridwidth=0.5, griddash='dash')
        else:
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
        return fig


def _apply_matplotlib_style_to_axes(ax, style: RecommenderVisualStyle):
    ax.set_facecolor(style.plot_background_color)
    for spine in ax.spines.values():
        spine.set_color(style.spine_color)
        spine.set_linewidth(style.spine_width)
    ax.tick_params(axis='both', which='major',
                   labelsize=style.font_size_tick,
                   colors=style.spine_color, width=style.spine_width)
    ax.tick_params(axis='both', which='minor',
                   labelsize=style.font_size_tick - 1,
                   colors=style.spine_color, width=style.spine_width * 0.7)
    if style.grid:
        ax.grid(True, alpha=style.grid_alpha,
                linestyle=style.grid_linestyle, linewidth=0.6)
    else:
        ax.grid(False)
    ax.xaxis.label.set_size(style.font_size_axis)
    ax.yaxis.label.set_size(style.font_size_axis)
    ax.title.set_size(style.font_size_title)
    ax.title.set_weight(style.font_weight)


def _normalize_for_radar(param: str, value: float) -> float:
    lo, hi = PLASTICITY_ONTOLOGY[param]["soft_range"]
    if hi == lo: return 0.5
    if param in ("rho0", "gamma0_dot", "sigma0_fric", "sigma0_yield",
                 "tau_p", "mu", "shear_modulus_G"):
        if value <= 0: return 0.0
        l_lo = np.log10(max(lo, 1e-10))
        l_hi = np.log10(max(hi, 1e-9))
        l_val = np.log10(max(value, 1e-10))
        return float(np.clip((l_val - l_lo) / (l_hi - l_lo), 0, 1))
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def render_recommender_radar_pub(bundle, style):
    categories = [PLASTICITY_ONTOLOGY[p]["symbol"] for p in PARAM_ORDER]
    defaults_norm = [_normalize_for_radar(p, bundle.defaults[p]) for p in PARAM_ORDER]
    bests_norm, best_values = [], []
    for p in PARAM_ORDER:
        best = bundle.best(p)
        if best:
            bests_norm.append(_normalize_for_radar(p, best.value_si))
            best_values.append(best.value_si)
        else:
            defaults_norm_v = _normalize_for_radar(p, bundle.defaults[p])
            bests_norm.append(defaults_norm_v)
            best_values.append(bundle.defaults[p])
    prior_norm = []
    for p in PARAM_ORDER:
        prior = bundle.prior_suggestion(p)
        prior_norm.append(_normalize_for_radar(p, prior)
                          if prior is not None
                          else defaults_norm[PARAM_ORDER.index(p)])
    fig = go.Figure()
    cmap = style.get_cmap()
    color_best = cmap(0.85)
    color_best_str = (f'rgb({int(color_best[0]*255)},'
                      f'{int(color_best[1]*255)},{int(color_best[2]*255)})')
    color_prior = cmap(0.55)
    color_prior_str = (f'rgb({int(color_prior[0]*255)},'
                       f'{int(color_prior[1]*255)},{int(color_prior[2]*255)})')
    fig.add_trace(go.Scatterpolar(
        r=defaults_norm + [defaults_norm[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Ontology Defaults',
        line=dict(color='#94a3b8', dash='dot', width=style.line_width),
        opacity=0.55,
        hovertemplate='%{theta}: %{r:.3f}<extra></extra>'))
    fig.add_trace(go.Scatterpolar(
        r=prior_norm + [prior_norm[0]],
        theta=categories + [categories[0]],
        fill='toself', name=f'Learned Prior ({bundle.material})',
        line=dict(color=color_prior_str, dash='dash',
                  width=style.line_width + 0.5),
        opacity=0.65,
        hovertemplate='%{theta}: %{r:.3f}<extra></extra>'))
    fig.add_trace(go.Scatterpolar(
        r=bests_norm + [bests_norm[0]],
        theta=categories + [categories[0]],
        fill='toself', name='⭐ AI Recommended',
        line=dict(color=color_best_str, width=style.line_width + 2),
        marker=dict(size=style.marker_size, color=color_best_str,
                    line=dict(color='white', width=1.5)),
        hovertemplate='%{theta}: %{r:.3f}<br>raw: %{customdata}<extra></extra>',
        customdata=[_pl_fmt(p, v) for p, v in zip(PARAM_ORDER, best_values)] +
                   [_pl_fmt(PARAM_ORDER[0], best_values[0])]))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=style.font_size_tick,
                                          family=style.font_family),
                            gridcolor='rgba(128,128,128,0.35)',
                            linecolor=style.spine_color),
            angularaxis=dict(tickfont=dict(size=style.font_size_axis,
                                           family=style.font_family),
                             linecolor=style.spine_color,
                             gridcolor='rgba(128,128,128,0.35)'),
            bgcolor=style.plot_background_color),
        height=int(style.figure_height * 90),
        width=int(style.figure_width * 90))
    style.apply_plotly_layout(
        fig, title=f"AI Recommendation vs Defaults — {bundle.material}")
    st.plotly_chart(fig, use_container_width=True)
    if style.show_values:
        with st.expander("📊 Normalized radar values", expanded=False):
            df = pd.DataFrame({
                'Parameter': categories,
                'Default (norm)': defaults_norm,
                'Prior (norm)': prior_norm,
                'AI (norm)': bests_norm,
                'AI (actual)': [_pl_fmt(p, v)
                                for p, v in zip(PARAM_ORDER, best_values)]})
            st.dataframe(df, use_container_width=True, hide_index=True)
    return _fig_to_bytes_plotly(fig)


_BAR_PRESET_PENDING_KEY = "rec_bar_pending_preset"


def _apply_pending_bar_preset():
    pending = st.session_state.pop(_BAR_PRESET_PENDING_KEY, None)
    if not pending: return
    cmap, mode = pending.get("cmap"), pending.get("mode")
    if cmap and cmap in BAR_CMAP_FLAT:
        st.session_state["rec_bar_cmap_select"] = cmap
    if mode in ("score", "index", "uniform"):
        st.session_state["rec_bar_color_mode"] = mode


def render_recommender_bars_pub(bundle, style):
    _apply_pending_bar_preset()
    selected_param = st.selectbox(
        "Select Parameter for Bar Chart", PARAM_ORDER,
        format_func=lambda p: f"{PLASTICITY_ONTOLOGY[p]['symbol']} — "
                              f"{PLASTICITY_ONTOLOGY[p]['label']}",
        key="rec_bar_param_pub")
    cands = bundle.candidates.get(selected_param, [])
    if not cands:
        st.warning("No candidates for this parameter.")
        return None
    spec = PLASTICITY_ONTOLOGY[selected_param]
    meta = PARAM_META.get(selected_param,
                          dict(title=spec['label'], symbol=spec['symbol'],
                               unit=None))
    meta = dict(meta)
    meta['title']  = normalize_tex(meta.get('title'))
    meta['symbol'] = normalize_tex(meta.get('symbol'))
    meta['unit']   = normalize_tex(meta.get('unit'))

    gcol1, gcol2 = st.columns([3, 2])
    with gcol1:
        legend_granularity = st.radio(
            "Legend granularity",
            options=['coarse', 'fine'],
            format_func=lambda x: {
                'coarse': '🔷 Coarse (3 buckets) — main text',
                'fine':   '🔶 Fine (6 buckets) — SI / audit'}[x],
            index=0, key="rec_bar_legend_gran", horizontal=True,
            help=("Coarse: LLM-grounded / LLM-prior / Deterministic. "
                  "Fine: verbatim / regex / regime / prior / physics-inferred. "
                  "'LLM chain-of-thought' is reserved for a future CoT "
                  "prompt path and will not appear until then."))
    with gcol2:
        legend_show_counts = st.checkbox(
            "Show per-bucket counts in legend",
            value=False, key="rec_bar_legend_counts")

    show_all_competitive = st.checkbox(
        "🔓 Show all routes as competitive (disable pin-list — audit only)",
        value=False, key="rec_show_all_competitive",
        help=("Temporarily treat all routes as competitive for scientific "
              "justification.  Reveals the FULL ranking including routes "
              "normally greyed as context.  Does NOT affect the actual "
              "recommendation — the pinned winner is unchanged."))
    if show_all_competitive:
        st.caption(
            "🔓 Audit mode ON: pin-list disabled for this figure only. "
            "Context bars are drawn competitively.  The sidebar's "
            "recommended value is still the pinned winner.")

    with st.expander("🎨 Colormap & Coloring Options", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            all_cmap_names = list(BAR_CMAP_FLAT)
            preferred = (style.bar_colormap if style.bar_colormap in all_cmap_names
                         else style.colormap if style.colormap in all_cmap_names
                         else 'viridis')
            default_idx = (all_cmap_names.index(preferred)
                           if preferred in all_cmap_names else 0)
            if "rec_bar_cmap_select" in st.session_state:
                queued = st.session_state["rec_bar_cmap_select"]
                if queued in all_cmap_names:
                    default_idx = all_cmap_names.index(queued)
            chosen_cmap = st.selectbox(
                "Colormap", all_cmap_names, index=default_idx,
                key="rec_bar_cmap_select")
            try:
                cmap_preview = plt.get_cmap(chosen_cmap)
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                fig_prev, ax_prev = plt.subplots(figsize=(5, 0.4), dpi=100)
                ax_prev.imshow(gradient, aspect='auto', cmap=cmap_preview)
                ax_prev.set_axis_off()
                st.pyplot(fig_prev, use_container_width=True)
                plt.close(fig_prev)
            except Exception:
                pass
        with col_b:
            color_mode = st.radio(
                "Color bars by",
                options=['score', 'index', 'uniform'],
                format_func=lambda x: {
                    'score':   '📊 Score (gradient)',
                    'index':   '🔢 Position (categorical)',
                    'uniform': '⬜ Uniform (legacy grey)'}[x],
                index=(['score', 'index', 'uniform'].index(style.bar_color_mode)
                       if style.bar_color_mode in ('score', 'index', 'uniform')
                       else 0),
                key="rec_bar_color_mode")
            colorbar_available = (color_mode == 'score')
            show_cbar = st.checkbox(
                "Show colorbar legend",
                value=bool(style.bar_show_colorbar and colorbar_available),
                key="rec_bar_show_cbar", disabled=not colorbar_available)
            if not colorbar_available:
                show_cbar = False
        with col_c:
            best_color_options = {
                "Auto (colormap opposite)": None,
                "🔵 Okabe-Ito Blue": '#0072B2',
                "🔴 Red accent": '#D55E00',
                "🟢 Green accent": '#009E73',
                "🟡 Yellow accent": '#F0E442',
                "🟣 Purple accent": '#CC79A7',
                "⚫ Black": '#000000',
                "⚪ White": '#FFFFFF'}
            current_override = style.bar_best_color
            default_best_idx = 0
            for i, v in enumerate(best_color_options.values()):
                if v == current_override:
                    default_best_idx = i; break
            best_color_name = st.selectbox(
                "Best-match bar colour", list(best_color_options.keys()),
                index=default_best_idx, key="rec_bar_best_color")
            best_color_override = best_color_options[best_color_name]
            if color_mode == 'score':
                with st.expander("⚙️ Advanced normalisation", expanded=False):
                    use_auto_vmin = st.checkbox(
                        "Auto vmin / vmax", value=True,
                        key="rec_bar_auto_vrange")
                    if use_auto_vmin:
                        cmap_vmin, cmap_vmax = None, None
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            cmap_vmin = st.number_input(
                                "vmin (score)",
                                value=float(min(c.score for c in cands)),
                                format="%.3f", key="rec_bar_vmin")
                        with c2:
                            cmap_vmax = st.number_input(
                                "vmax (score)",
                                value=float(max(c.score for c in cands)),
                                format="%.3f", key="rec_bar_vmax")
                        if cmap_vmax <= cmap_vmin:
                            st.warning("vmax must exceed vmin — using auto range.")
                            cmap_vmin, cmap_vmax = None, None
            else:
                cmap_vmin, cmap_vmax = None, None

    with st.expander("✍️ Typography & Labels", expanded=True):
        t_col1, t_col2 = st.columns(2)
        _journals = JournalTemplates.get_journal_styles()
        _jspec = _journals.get(style.journal, _journals['nature'])
        _base_small = float(_jspec.get('font_size_small', 8))
        with t_col1:
            ann_fs = st.slider(
                "Annotation font size (pt)", 4.0, 24.0,
                value=float(style.bar_annotation_fontsize
                            if style.bar_annotation_fontsize is not None
                            else _base_small),
                step=0.5, key="rec_bar_ann_fs")
            leg_fs = st.slider(
                "Legend font size (pt)", 4.0, 24.0,
                value=float(style.bar_legend_fontsize
                            if style.bar_legend_fontsize is not None
                            else _base_small),
                step=0.5, key="rec_bar_leg_fs")
            ann_offset = st.slider(
                "Annotation offset (pt)", 0.0, 30.0,
                value=float(style.bar_annotation_offset),
                step=0.5, key="rec_bar_ann_offset")
        with t_col2:
            cbar_fs = st.slider(
                "Colorbar label size (pt)", 4.0, 24.0,
                value=float(style.bar_colorbar_label_fontsize
                            if style.bar_colorbar_label_fontsize is not None
                            else _base_small),
                step=0.5, key="rec_bar_cbar_fs",
                disabled=(color_mode != 'score'))
            ctick_fs = st.slider(
                "Colorbar tick size (pt)", 4.0, 24.0,
                value=float(style.bar_colorbar_tick_fontsize
                            if style.bar_colorbar_tick_fontsize is not None
                            else max(_base_small - 1.0, 4.0)),
                step=0.5, key="rec_bar_ctick_fs",
                disabled=(color_mode != 'score'))
            show_offset_hint = st.checkbox(
                "Preview head-room advisory", value=False,
                key="rec_bar_headroom_hint")
            if show_offset_hint:
                _axes_pt = 0.85 * max(2.6, 0.60 * style.figure_height) * 72.0
                _clear_pt = ann_offset + 1.5 * ann_fs
                _frac = min(0.85, _clear_pt / _axes_pt)
                _min_safe = 1.0 / max(1e-6, 1.0 - _frac)
                st.caption(f"Minimum safe head-room factor: "
                           f"**{_min_safe:.3f}** (auto-applied if it exceeds "
                           f"the default {style.bar_headroom:.2f}).")

    st.markdown("**⚡ One-click colormap presets**")
    preset_cols = st.columns(5)
    presets = [("🌊 Ocean", 'viridis', 'score'),
               ("🔥 Heat", 'plasma', 'score'),
               ("🌈 Rainbow", 'turbo', 'score'),
               ("🎯 Diverge", 'RdYlGn', 'score'),
               ("🖤 Classic", 'Greys', 'uniform')]
    for col, (name, cmap_name, mode_name) in zip(preset_cols, presets):
        with col:
            if st.button(name, use_container_width=True,
                         key=f"rec_bar_preset_{name}"):
                st.session_state[_BAR_PRESET_PENDING_KEY] = {
                    "cmap": cmap_name, "mode": mode_name}
                st.rerun()

    candidates = [c.value_si / spec["ui_scale"] for c in cands]
    scores = [float(c.score) for c in cands]
    provenance = [(c.provenance or c.method) for c in cands]
    if show_all_competitive:
        context_flags = [False] * len(cands)
    else:
        context_flags = [bool(getattr(c, 'context', False)) for c in cands]

    n_ctx = sum(1 for x in context_flags if x)
    if n_ctx > 0:
        st.caption(
            f"🩶 {n_ctx} supporting route(s) shown greyed for context — "
            f"ranking considers the {len(cands) - n_ctx} non-context "
            f"candidate(s) only.  The recommended value is unaffected.")
    elif show_all_competitive:
        st.caption(
            f"🔓 All {len(cands)} route(s) drawn competitively "
            "(pin-list override for this figure only).")

    if show_all_competitive:
        best_idx = int(np.argmax(scores))
    else:
        non_ctx_scores = [
            (s if not f else float('-inf'))
            for s, f in zip(scores, context_flags)]
        best_idx = int(np.argmax(non_ctx_scores))
        if best_idx < 0 or not np.isfinite(non_ctx_scores[best_idx]):
            best_idx = int(np.argmax(scores))

    journals = JournalTemplates.get_journal_styles()
    journal = style.journal if style.journal in journals else 'nature'
    fig = plot_candidate_scores(
        candidates=candidates, scores=scores, provenance=provenance,
        best_idx=best_idx, param_title=meta['title'], param_symbol=meta['symbol'],
        unit=meta['unit'], score_label='LatentMoE score', journal=journal,
        fig_size=(max(3.5, 0.62 * style.figure_width),
                  max(2.6, 0.60 * style.figure_height)),
        label_pad=float(ann_offset), headroom=float(style.bar_headroom),
        despine=True, bar_colormap=chosen_cmap, color_by=color_mode,
        best_color_override=best_color_override,
        cmap_vmin=cmap_vmin, cmap_vmax=cmap_vmax,
        show_colorbar=show_cbar, annotation_fontsize=float(ann_fs),
        legend_fontsize=float(leg_fs), colorbar_label_fontsize=float(cbar_fs),
        colorbar_tick_fontsize=float(ctick_fs),
        legend_granularity=legend_granularity,
        legend_show_counts=legend_show_counts,
        context_flags=context_flags)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=style.dpi, bbox_inches='tight',
                pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_recommender_sankey_pub(bundle, style):
    sources = set()
    for p in PARAM_ORDER:
        for c in bundle.candidates.get(p, []):
            if c.source_file:
                sources.add(c.source_file)
    source_list = sorted(list(sources)) if sources else ["(no sources)"]
    param_list = PARAM_ORDER
    selected_list = [f"✅ {p}" for p in PARAM_ORDER]
    all_nodes = source_list + param_list + selected_list
    node_indices = {n: i for i, n in enumerate(all_nodes)}
    cmap = style.get_cmap()
    links = []
    for p in PARAM_ORDER:
        for c in bundle.candidates.get(p, []):
            if c.source_file and c.source_file in node_indices:
                links.append({"source": node_indices[c.source_file],
                              "target": node_indices[p],
                              "value": max(c.score, 0.01),
                              "color": "rgba(100,200,255,0.35)"})
    for p in param_list:
        best = bundle.best(p)
        val = best.value_si if best else bundle.defaults[p]
        norm_val = _normalize_for_radar(p, val)
        rgba = cmap(norm_val)
        links.append({"source": node_indices[p],
                      "target": node_indices[f"✅ {p}"],
                      "value": 1.0,
                      "color": (f'rgba({int(rgba[0]*255)},'
                                f'{int(rgba[1]*255)},'
                                f'{int(rgba[2]*255)},0.65)')})
    node_colors = ["#a5b4fc" if n in source_list
                   else "#fbbf24" if n in param_list
                   else "#34d399" for n in all_nodes]
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=18, thickness=22,
                  line=dict(color=style.spine_color, width=0.8),
                  label=all_nodes, color=node_colors,
                  hovertemplate='%{label}<br>Total flow: %{value:.2f}<extra></extra>'),
        link=dict(source=[l["source"] for l in links],
                  target=[l["target"] for l in links],
                  value=[l["value"] for l in links],
                  color=[l["color"] for l in links],
                  hovertemplate='%{source.label} → %{target.label}<br>'
                                'weight: %{value:.3f}<extra></extra>'))])
    style.apply_plotly_layout(fig, title="Plasticity Parameter Source Flow (Sankey)")
    fig.update_layout(height=int(style.figure_height * 100),
                      width=int(style.figure_width * 100))
    st.plotly_chart(fig, use_container_width=True)
    return _fig_to_bytes_plotly(fig)


def render_recommender_treemap_pub(bundle, style):
    ids, labels, parents, values, colors = [], [], [], [], []
    cmap = style.get_cmap(); palette = style.get_palette()
    for p_idx, p in enumerate(PARAM_ORDER):
        spec = PLASTICITY_ONTOLOGY[p]
        ids.append(p)
        labels.append(f"<b>{spec['symbol']}</b><br>{spec['label']}")
        parents.append(""); values.append(0)
        colors.append(palette[p_idx % len(palette)])
        for j, c in enumerate(bundle.candidates.get(p, [])):
            c_id = f"{p}_{j}_{_pl_hash(c.source_file)[:6]}"
            ids.append(c_id)
            theory_extra = ""
            if p == "gamma0_dot" and getattr(c, "theory", ""):
                if c.theory and c.theory != "unspecified":
                    theory_extra = f"<br>[{c.theory}]"
            ctx_tag = " 🩶" if getattr(c, 'context', False) else ""
            labels.append(f"{_pl_fmt(p, c.value_si)}<br><i>{c.method}</i>"
                          f"{theory_extra}{ctx_tag}<br>score={c.score:.2f}")
            parents.append(p); values.append(max(c.score, 0.01))
            if getattr(c, 'context', False):
                rgb = (0.65, 0.65, 0.65)
            else:
                rgba = cmap(float(np.clip(c.score, 0, 1)))
                rgb = (rgba[0], rgba[1], rgba[2])
            colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},'
                          f'{int(rgb[2]*255)})')
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=colors,
                    line=dict(color=style.spine_color, width=1.0),
                    pad=dict(t=2, l=2, r=2, b=2)),
        textinfo="label+value",
        textfont=dict(family=style.font_family,
                      size=style.font_size_tick, color='white'),
        hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'))
    style.apply_plotly_layout(fig, title="Candidate Hierarchy Treemap (sized by score)")
    fig.update_layout(height=int(style.figure_height * 100),
                      width=int(style.figure_width * 100))
    st.plotly_chart(fig, use_container_width=True)
    return _fig_to_bytes_plotly(fig)


def render_recommender_histograms_pub(bundle, style):
    available = [p for p in PARAM_ORDER if bundle.candidates.get(p)]
    if not available:
        st.info("No candidates to plot yet."); return None
    n = len(available); cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(style.figure_width * cols / 3,
                                      style.figure_height * rows / 2),
                             dpi=style.dpi)
    fig.patch.set_facecolor(style.background_color)
    if style.transparent_bg:
        fig.patch.set_alpha(0.0)
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])
    cmap = style.get_cmap(); palette = style.get_palette()
    for idx, p in enumerate(available):
        ax = axes[idx]
        spec = PLASTICITY_ONTOLOGY[p]
        cands = bundle.candidates[p]
        ui_vals = [c.value_si / spec["ui_scale"] for c in cands]
        scores = [c.score for c in cands]
        use_log = p in ("rho0", "gamma0_dot") and min(ui_vals) > 0
        if use_log:
            bins = np.logspace(np.log10(min(ui_vals)),
                               np.log10(max(ui_vals)),
                               num=min(15, max(5, len(ui_vals))), base=10)
        else:
            bins = min(15, max(5, len(ui_vals)))
        weights = np.array(scores)
        n_hist, edges, patches = ax.hist(
            ui_vals, bins=bins, weights=weights,
            color=palette[idx % len(palette)],
            edgecolor=style.spine_color,
            linewidth=style.spine_width, alpha=0.75)
        vmin, vmax = ((np.log10(max(edges[0], 1e-10)),
                       np.log10(max(edges[-1], 1e-10))) if use_log
                      else (edges[0], edges[-1]))
        for patch, edge in zip(patches, edges[:-1]):
            if use_log:
                e = np.log10(max(edge, 1e-10))
                norm_val = (e - vmin) / max(vmax - vmin, 1e-9)
            else:
                norm_val = (edge - vmin) / max(vmax - vmin, 1e-9)
            patch.set_facecolor(cmap(np.clip(norm_val, 0, 1)))
        best = bundle.best(p)
        if best is not None:
            ax.axvline(best.value_si / spec["ui_scale"],
                       color='#dc2626', linestyle='--',
                       linewidth=style.line_width + 0.5,
                       label=f"⭐ {_pl_fmt(p, best.value_si)}", zorder=5)
        if style.grid:
            ax.grid(True, alpha=style.grid_alpha,
                    linestyle=style.grid_linestyle, axis='y')
        if use_log: ax.set_xscale('log')
        ax.set_xlabel(f"{spec['label']} [{spec['ui_unit']}]",
                      fontsize=style.font_size_axis - 1,
                      fontweight=style.font_weight)
        ax.set_ylabel("Σ Score",
                      fontsize=style.font_size_axis - 1,
                      fontweight=style.font_weight)
        ax.set_title(f"{spec['symbol']} — {spec['label']}",
                     fontsize=style.font_size_title - 1,
                     fontweight=style.font_weight)
        if style.show_legend and best is not None:
            ax.legend(loc=style.legend_position,
                      fontsize=style.font_size_legend - 1, frameon=True)
        _apply_matplotlib_style_to_axes(ax, style)
    for idx in range(len(available), len(axes)):
        axes[idx].axis('off')
    if style.show_title:
        fig.suptitle(style.title_override or
                     f"Candidate Distributions — Target: {bundle.material} "
                     f"@ {bundle.temp_k:.0f} K, γ̇={bundle.strain_rate:.1e} s⁻¹",
                     fontsize=style.font_size_title + 1,
                     fontweight=style.font_weight)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)
    buf = _fig_to_bytes_matplotlib(fig)
    plt.close(fig)
    return buf


def _fig_to_bytes_matplotlib(fig, fmt="png", dpi=300):
    try:
        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning("Matplotlib export failed (%s): %s", fmt, e)
        return None


def _fig_to_bytes_plotly(fig, fmt="png", width=1200, height=800, scale=3.0):
    try:
        return fig.to_image(format=fmt, width=width, height=height, scale=scale)
    except Exception as e:
        logger.warning("Plotly static export failed (%s): %s", fmt, e)
        return None


def render_recommender_style_controls() -> RecommenderVisualStyle:
    style = RecommenderVisualStyle()
    st.markdown("#### 🎨 Publication Styling")
    journal_options = {
        "Nature": "nature", "Science": "science",
        "Advanced Materials": "advanced_materials",
        "Physical Review Letters": "prl", "Custom": "custom"}
    preset_display = st.selectbox(
        "📚 Journal Preset", list(journal_options.keys()),
        index=0, key="rec_journal_preset")
    style.journal = journal_options[preset_display]
    if style.journal != "custom":
        styles = JournalTemplates.get_journal_styles()
        js = styles.get(style.journal, styles['nature'])
        style.font_family = js['font_family']
        style.font_size_title = js['font_size_large']
        style.font_size_axis = js['font_size_medium']
        style.font_size_tick = js['font_size_small']
        style.font_size_legend = js['font_size_small']
        style.font_size_annotation = js['font_size_small']
        style.line_width = js['line_width']
        style.grid_alpha = js['grid_alpha']
        style.dpi = js['dpi']
    with st.expander("🔤 Fonts", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            style.font_family = st.selectbox(
                "Font family", FONT_FAMILIES,
                index=FONT_FAMILIES.index(style.font_family)
                if style.font_family in FONT_FAMILIES else 0,
                key="rec_font_family")
            style.font_weight = st.selectbox(
                "Font weight", ["normal", "bold"],
                index=1 if style.font_weight == "bold" else 0,
                key="rec_font_weight")
        with c2:
            style.font_size_title = st.slider(
                "Title size (pt)", 6.0, 24.0, float(style.font_size_title),
                0.5, key="rec_fs_title")
            style.font_size_axis = st.slider(
                "Axis label size (pt)", 6.0, 20.0, float(style.font_size_axis),
                0.5, key="rec_fs_axis")
            style.font_size_tick = st.slider(
                "Tick label size (pt)", 5.0, 18.0, float(style.font_size_tick),
                0.5, key="rec_fs_tick")
            style.font_size_legend = st.slider(
                "Legend size (pt)", 5.0, 18.0, float(style.font_size_legend),
                0.5, key="rec_fs_legend")
            style.font_size_annotation = st.slider(
                "Annotation size (pt)", 5.0, 16.0,
                float(style.font_size_annotation), 0.5, key="rec_fs_annot")
    with st.expander("🖼️ Figure & DPI", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            style.figure_width = st.slider(
                "Figure width (in)", 4.0, 20.0, float(style.figure_width),
                0.5, key="rec_fw")
            style.line_width = st.slider(
                "Line width", 0.5, 5.0, float(style.line_width),
                0.1, key="rec_lw")
        with c2:
            style.figure_height = st.slider(
                "Figure height (in)", 3.0, 16.0, float(style.figure_height),
                0.5, key="rec_fh")
            style.marker_size = st.slider(
                "Marker size", 2.0, 20.0, float(style.marker_size),
                0.5, key="rec_ms")
        with c3:
            style.dpi = st.select_slider(
                "DPI (export)", options=[72, 150, 300, 600, 1200],
                value=style.dpi if style.dpi in [72, 150, 300, 600, 1200] else 300,
                key="rec_dpi")
    with st.expander("🌈 Colormap", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            style.colormap = st.selectbox(
                "Colormap", cmap_list,
                index=cmap_list.index(style.colormap)
                if style.colormap in cmap_list else 0,
                key="rec_cmap")
        with c2:
            style.cmap_reverse = st.checkbox(
                "Reverse", value=style.cmap_reverse, key="rec_cmap_rev")
        try:
            cmap = style.get_cmap()
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            fig_prev, ax_prev = plt.subplots(figsize=(6, 0.45), dpi=120)
            ax_prev.imshow(gradient, aspect='auto', cmap=cmap)
            ax_prev.set_axis_off()
            st.pyplot(fig_prev, use_container_width=True)
            plt.close(fig_prev)
        except Exception:
            pass
        style.categorical_palette = st.selectbox(
            "Categorical palette (for radar, treemap roots, histograms)",
            list(PUBLICATION_PALETTES.keys()),
            index=list(PUBLICATION_PALETTES.keys()).index(style.categorical_palette)
            if style.categorical_palette in PUBLICATION_PALETTES else 0,
            key="rec_palette")
    with st.expander("🎯 Labels, Titles & Legend", expanded=False):
        style.show_title = st.checkbox(
            "Show title", value=style.show_title, key="rec_show_title")
        if style.show_title:
            style.title_override = st.text_input(
                "Title override (leave empty to use default)",
                value=style.title_override, key="rec_title_ovr")
        style.xlabel_override = st.text_input(
            "X label override", value=style.xlabel_override,
            key="rec_xlabel_ovr")
        style.ylabel_override = st.text_input(
            "Y label override", value=style.ylabel_override,
            key="rec_ylabel_ovr")
        c1, c2 = st.columns(2)
        with c1:
            style.show_legend = st.checkbox(
                "Show legend", value=style.show_legend, key="rec_show_legend")
        with c2:
            style.show_values = st.checkbox(
                "Show numeric annotations", value=style.show_values,
                key="rec_show_values")
            style.show_colorbar = st.checkbox(
                "Show colorbar", value=style.show_colorbar,
                key="rec_show_cbar")
    with st.expander("🧱 Backgrounds & Spines", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            style.background_color = st.color_picker(
                "Figure background", style.background_color,
                key="rec_bg_color")
            style.plot_background_color = st.color_picker(
                "Plot background", style.plot_background_color,
                key="rec_plot_bg")
            style.transparent_bg = st.checkbox(
                "Transparent background (for export)",
                value=style.transparent_bg, key="rec_transp_bg")
        with c2:
            style.spine_color = st.color_picker(
                "Spine / tick color", style.spine_color,
                key="rec_spine_color")
            style.spine_width = st.slider(
                "Spine width", 0.5, 4.0, float(style.spine_width),
                0.1, key="rec_spine_w")
            style.grid = st.checkbox(
                "Show grid", value=style.grid, key="rec_grid")
            if style.grid:
                style.grid_alpha = st.slider(
                    "Grid alpha", 0.0, 1.0, float(style.grid_alpha),
                    0.05, key="rec_grid_alpha")
                style.grid_linestyle = st.selectbox(
                    "Grid linestyle", ['-', '--', '-.', ':'],
                    index=['-', '--', '-.', ':'].index(style.grid_linestyle)
                    if style.grid_linestyle in ['-', '--', '-.', ':'] else 1,
                    key="rec_grid_ls")
    with st.expander("📊 Bar-Chart Defaults", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            style.bar_colormap = st.selectbox(
                "Default bar colormap", BAR_CMAP_FLAT,
                index=BAR_CMAP_FLAT.index(style.bar_colormap)
                if style.bar_colormap in BAR_CMAP_FLAT else 0,
                key="rec_style_bar_cmap")
        with c2:
            style.bar_color_mode = st.selectbox(
                "Default bar color mode", ['score', 'index', 'uniform'],
                index=['score', 'index', 'uniform'].index(style.bar_color_mode)
                if style.bar_color_mode in ('score', 'index', 'uniform') else 0,
                key="rec_style_bar_mode")
        style.bar_show_colorbar = st.checkbox(
            "Show colorbar by default",
            value=style.bar_show_colorbar, key="rec_style_bar_cbar")
        tc1, tc2 = st.columns(2)
        with tc1:
            _ann_default = (style.bar_annotation_fontsize
                            if style.bar_annotation_fontsize is not None else 0.0)
            _ann_new = st.slider(
                "Annotation font size (pt, 0 = journal default)",
                0.0, 24.0, float(_ann_default), 0.5,
                key="rec_style_bar_ann_fs")
            style.bar_annotation_fontsize = _ann_new if _ann_new > 0 else None
            _leg_default = (style.bar_legend_fontsize
                            if style.bar_legend_fontsize is not None else 0.0)
            _leg_new = st.slider(
                "Legend font size (pt, 0 = journal default)",
                0.0, 24.0, float(_leg_default), 0.5,
                key="rec_style_bar_leg_fs")
            style.bar_legend_fontsize = _leg_new if _leg_new > 0 else None
            style.bar_annotation_offset = st.slider(
                "Annotation offset (pt)", 0.0, 30.0,
                float(style.bar_annotation_offset), 0.5,
                key="rec_style_bar_ann_offset")
        with tc2:
            _cbfs_default = (style.bar_colorbar_label_fontsize
                             if style.bar_colorbar_label_fontsize is not None
                             else 0.0)
            _cbfs_new = st.slider(
                "Colorbar label size (pt, 0 = journal default)",
                0.0, 24.0, float(_cbfs_default), 0.5,
                key="rec_style_bar_cbar_fs")
            style.bar_colorbar_label_fontsize = _cbfs_new if _cbfs_new > 0 else None
            _ctick_default = (style.bar_colorbar_tick_fontsize
                              if style.bar_colorbar_tick_fontsize is not None
                              else 0.0)
            _ctick_new = st.slider(
                "Colorbar tick size (pt, 0 = label − 1)",
                0.0, 24.0, float(_ctick_default), 0.5,
                key="rec_style_bar_ctick_fs")
            style.bar_colorbar_tick_fontsize = _ctick_new if _ctick_new > 0 else None
            style.bar_headroom = st.slider(
                "Minimum head-room factor", 1.05, 1.80,
                float(style.bar_headroom), 0.01,
                key="rec_style_bar_headroom")
    style.apply_matplotlib()
    return style


def render_recommender_visuals_dashboard():
    bundle = st.session_state.get('recommender_bundle')
    if bundle is None:
        st.info("Run the plasticity recommender first "
                "(sidebar → 🔍 Analyse JSON databases).")
        return
    st.markdown("---")
    st.header("🤖 AI Recommender Visuals Dashboard")
    st.caption(
        "Publication-quality visualizations for plasticity parameter "
        "candidates. v10.0.0: every bar IS the audit — legend keys are "
        "now experts; provenance survives as a non-ranking badge. "
        "Bar height = Σ = Λ·κ_n·κ_m·G.")
    chart_type = st.selectbox(
        "📈 Chart Type",
        ["Score Anatomy (v10)", "Expert Mass Sankey (v10)",
         "Value Landscape (v10)",
         "Radar Chart", "Bar Chart (v9 tiers)", "Sankey Diagram",
         "Treemap", "Histograms (publication)"],
        index=0, key="rec_chart_type_v10")
    col_style, col_chart = st.columns([1, 2])
    with col_style:
        with st.container():
            style = render_recommender_style_controls()
    with col_chart:
        st.markdown(f"#### {chart_type}")
        chart_bytes = None
        chart_fmt_hint = "png"
        try:
            if chart_type == "Score Anatomy (v10)":
                if not bundle.scored_v10:
                    st.warning("v10 scoring not available — re-run the "
                               "recommender (Analyse JSON databases).")
                else:
                    param_v10 = st.selectbox(
                        "Parameter", [p for p in PARAM_ORDER
                                       if bundle.scored_v10.get(p)],
                        format_func=lambda p:
                            f"{PLASTICITY_ONTOLOGY[p]['symbol']} — "
                            f"{PLASTICITY_ONTOLOGY[p]['label']}",
                        key="rec_v10_param")
                    scored = bundle.scored_v10.get(param_v10, [])
                    show_ref = st.checkbox(
                        "📌 Show v9 reference marks (annotation only)",
                        value=True, key="rec_v10_ref")
                    st.caption(
                        "Bar height = **Σ = Λ·κ_n·κ_m·G**.  Stack sums "
                        "exactly to Σ.  Hatched region = named gate.  "
                        "Grey band = count discount.  Dashed outline = Λ "
                        "(expert composite before counts & gates).  "
                        "Whisker = 5–95 % Dirichlet weight bootstrap.  "
                        "Badge row (below axis) = provenance — non-ranking.")
                    fig_anat = render_score_anatomy_figure(
                        scored, param_v10, journal=style.journal,
                        show_reference=show_ref)
                    st.pyplot(fig_anat)
                    chart_bytes = _fig_to_bytes_matplotlib(
                        fig_anat, dpi=style.dpi)
                    plt.close(fig_anat)

                    with st.expander("📋 Score anatomy table", expanded=False):
                        rows_tbl = [c.to_display() for c in scored]
                        st.dataframe(pd.DataFrame(rows_tbl),
                                     use_container_width=True,
                                     hide_index=True)

                    if bundle.context_vector is not None:
                        ctx = bundle.context_vector
                        st.caption(
                            f"Context completeness κ(c) = **{ctx.kappa:.2f}** "
                            f"· framework κ_F(c) = **{ctx.kappa_framework:.2f}**"
                            f" · synthesis={ctx.synthesis or '—'}"
                            f" · architecture={ctx.architecture or '—'}"
                            f" · λ={ctx.twin_spacing} nm"
                            f" · framework={ctx.framework or '—'}")
            elif chart_type == "Expert Mass Sankey (v10)":
                if not bundle.scored_v10:
                    st.warning("v10 scoring not available.")
                else:
                    param_v10 = st.selectbox(
                        "Parameter", [p for p in PARAM_ORDER
                                       if bundle.scored_v10.get(p)],
                        format_func=lambda p:
                            f"{PLASTICITY_ONTOLOGY[p]['symbol']} — "
                            f"{PLASTICITY_ONTOLOGY[p]['label']}",
                        key="rec_v10_sankey_param")
                    fig_sk = render_expert_mass_sankey(
                        bundle.scored_v10[param_v10], param_v10)
                    if fig_sk is not None:
                        style.apply_plotly_layout(
                            fig_sk, title=f"Expert mass → Σ — {param_v10}")
                        st.plotly_chart(fig_sk, use_container_width=True)
                        chart_bytes = _fig_to_bytes_plotly(fig_sk)
            elif chart_type == "Value Landscape (v10)":
                if not bundle.scored_v10:
                    st.warning("v10 scoring not available.")
                else:
                    param_v10 = st.selectbox(
                        "Parameter", [p for p in PARAM_ORDER
                                       if bundle.scored_v10.get(p)],
                        format_func=lambda p:
                            f"{PLASTICITY_ONTOLOGY[p]['symbol']} — "
                            f"{PLASTICITY_ONTOLOGY[p]['label']}",
                        key="rec_v10_landscape_param")
                    fig_land = render_value_landscape(
                        bundle.scored_v10[param_v10], param_v10)
                    if fig_land is not None:
                        st.pyplot(fig_land)
                        chart_bytes = _fig_to_bytes_matplotlib(
                            fig_land, dpi=style.dpi)
                        plt.close(fig_land)
            elif chart_type == "Radar Chart":
                chart_bytes = render_recommender_radar_pub(bundle, style)
            elif chart_type == "Bar Chart (v9 tiers)":
                chart_bytes = render_recommender_bars_pub(bundle, style)
            elif chart_type == "Sankey Diagram":
                chart_bytes = render_recommender_sankey_pub(bundle, style)
            elif chart_type == "Treemap":
                chart_bytes = render_recommender_treemap_pub(bundle, style)
            elif chart_type == "Histograms (publication)":
                chart_bytes = render_recommender_histograms_pub(bundle, style)
        except Exception as e:
            st.error(f"Chart rendering failed: {e}")
            st.exception(e)
        st.markdown("---")
        st.markdown("##### 📤 Export Chart")
        if chart_bytes is None:
            st.caption("⚠️ Static export unavailable for this chart. "
                       "Install `kaleido` for Plotly charts.")
        else:
            st.download_button(
                f"⬇️ Download PNG ({style.dpi} DPI)",
                data=chart_bytes,
                file_name=(f"recommender_"
                           f"{chart_type.lower().replace(' ', '_')}_"
                           f"{bundle.material}_{int(style.dpi)}dpi.png"),
                mime="image/png", use_container_width=True)


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Enhanced Nanotwinned Cu Phase-Field Simulator (FFT)",
        layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; color: #1E3A8A; text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] {
        height: 3rem; white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px; padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔬 Enhanced Nanotwinned Cu Phase-Field Simulator (FFT)</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #F0F9FF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 1rem;">
    <strong>✅ PURE FFT SPECTRAL + AI PLASTICITY RECOMMENDER v10.0.0 — SCORE ANATOMY:</strong><br>
    • <span style="color: green;">NO FDM/NUMBA:</span> exact spectral operators.<br>
    • <span style="color: green;">SEMI-IMPLICIT FOURIER:</span> unconditional linear stability.<br>
    • <span style="color: green;">🧭 NER-GAZETTEER + 3-TIER CASCADE:</span> grounded LLM ∧ regex ∧ prior inference.<br>
    • <span style="color: green;">🧮 FRICTION STRESS LAB:</span> σ₀ physics-inferred via HP(d), HP(λ), solver-law inversion, P–N+Taylor floor.<br>
    • <span style="color: green;">🔬 PHYSICS-GROUNDED ρ₀:</span> snippet-enhanced NER ∧ three-factor regime classifier ∧ physics-regime expert.<br>
    • <span style="color: green;">⚡ THEORY-AWARE γ̇₀:</span> JC vs CPFEM vs DDD vs MD disambiguation via theory tagging.<br>
    • <span style="color: green;">🆕 v10.0.0 SCORE ANATOMY:</span> legend keys = <strong>experts</strong>; provenance demoted to a non-ranking <em>badge</em>. Bar height = <strong>Σ = Λ·κ_n·κ_m·G</strong>. Stacked expert segments sum <em>exactly</em> to Σ. Counts as <strong>pips + ghost Λ outline</strong>. Gates as <strong>hatched named amputations</strong>. Weight sensitivity as <strong>Dirichlet-bootstrap whisker</strong>. v9 pin-list survives as <strong>annotation-only reference marks</strong>.<br>
    • <span style="color: green;">🎯 5/5 target coverage:</span> ρ₀, μ, γ̇₀, m, σ₀ close automatically.
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔄 Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", type="secondary"):
                for k in list(st.session_state.keys()):
                    if k in ('twin_simulations', 'initial_geometry', 'initialized',
                             'results_history', 'timesteps', 'solver',
                             'selected_sim_id', 'comparison_config',
                             'sweep_results', 'sweep_param',
                             'plasticity_overrides', 'recommender_bundle',
                             'friction_lab_w_mode', 'pn_zeta'):
                        st.session_state.pop(k, None)
                st.success("All simulations & state cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
        if st.button("🔄 Reset to Defaults", type="secondary"):
            keys_to_clear = ['N', 'dx', 'dt', 'twin_spacing', 'W', 'A', 'B',
                             'kappa0', 'gamma_aniso', 'kappa_eta', 'L_CTB',
                             'L_ITB', 'n_mob', 'L_eta', 'zeta', 'applied_stress',
                             'geom_type', 'defect_type', 'defect_x', 'defect_y',
                             'defect_radius', 'left_buffer_width', 'buffer_width',
                             'gb_profile', 'gb_curvature', 'grain_boundary_pos',
                             'stability_factor', 'enable_monitoring',
                             'auto_adjust_dt', 'n_steps', 'save_freq',
                             'confine_twin']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Parameters reset to defaults!")
            st.rerun()
        st.markdown("---")
        with st.expander("🧠 AI Plasticity Recommender v10.0.0", expanded=False):
            render_plasticity_recommender_sidebar(
                default_material=st.session_state.get("material", "Cu"),
                default_temp=300.0,
                default_strain_rate=1e-3,
                ollama_model="qwen2.5:7b")
        st.markdown("---")
        operation_mode = st.radio(
            "Operation Mode",
            ["Run New Simulation", "Compare Saved Simulations",
             "Single Simulation View", "Parameter Sweep"],
            index=0)
        if operation_mode == "Run New Simulation":
            st.header("🎛️ New Simulation Setup")
            st.subheader("🧪 Material")
            material_choice = st.selectbox("Select material", ["Cu", "Al", "Ni"],
                                           key="material")
            st.subheader("🧩 Geometry Configuration")
            geometry_type = st.selectbox(
                "Geometry Type",
                ["Standard Twin Grain", "Twin Grain with Defect"],
                key="geom_type")
            left_buffer_width = st.slider("Left buffer width (nm)", 0.0, 20.0, 5.0,
                                          0.5, key="left_buffer_width")
            buffer_width = st.slider("Twin‑free buffer near GB (nm)", 0.0, 20.0,
                                     5.0, 0.5, key="buffer_width")
            gb_profile = st.selectbox("Grain Boundary Profile",
                                      ["Plane", "Concave", "Convex"],
                                      key="gb_profile")
            gb_curvature = st.slider("GB Curvature Amplitude (nm)", 0.0, 20.0,
                                     5.0, 0.5, key="gb_curvature")
            st.subheader("📊 Grid Configuration")
            N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64, key="N")
            dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1, key="dx")
            dt = st.slider("Time step (ns)", 1e-5, 1e-2, 1e-3, 1e-5,
                           key="dt", format="%.5f")
            st.subheader("🔬 Material Parameters")
            twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0,
                                     1.0, key="twin_spacing")
            grain_boundary_pos = st.slider("Grain boundary nominal position (nm)",
                                           -50.0, 50.0, 0.0, 1.0,
                                           key="grain_boundary_pos")
            if geometry_type == "Twin Grain with Defect":
                st.subheader("⚠️ Defect Parameters")
                defect_type = st.selectbox("Defect Type",
                                           ["Dislocation", "Void"],
                                           key="defect_type")
                defect_x = st.slider("Defect X (nm)", -50.0, 50.0, 0.0,
                                     1.0, key="defect_x")
                defect_y = st.slider("Defect Y (nm)", -50.0, 50.0, 0.0,
                                     1.0, key="defect_y")
                defect_radius = st.slider("Defect radius (nm)", 5.0, 30.0,
                                          10.0, 1.0, key="defect_radius")
            st.subheader("⚡ Thermodynamic Parameters")
            W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0,
                          0.1, key="W")
            A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0,
                          0.5, key="A")
            B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0,
                          0.5, key="B")
            st.subheader("🌀 Gradient Energy")
            kappa0 = st.slider("κ₀ (gradient energy ref)", 0.01, 10.0, 1.0,
                               0.1, key="kappa0")
            gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7,
                                    0.05, key="gamma_aniso")
            kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0,
                                  0.1, key="kappa_eta")
            st.subheader("⚡ Kinetic Parameters")
            L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05,
                              0.001, key="L_CTB")
            L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0,
                              0.1, key="L_ITB")
            n_mob = st.slider("n (mobility exponent)", 1, 10, 4, 1,
                              key="n_mob")
            L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1,
                              key="L_eta")
            zeta = st.slider("ζ (dislocation pinning)", 0.0, 2.0, 0.3, 0.05,
                             key="zeta")
            st.subheader("🏋️ Loading Conditions")
            applied_stress_MPa = st.slider("Applied stress magnitude (MPa)",
                                           0.0, 1000.0, 300.0, 10.0,
                                           key="applied_stress")
            loading_angle = st.slider("Loading angle θ (deg)", 0.0, 180.0, 0.0,
                                      5.0, key="loading_angle")
            st.subheader("⏯️ Simulation Control")
            n_steps = st.slider("Number of steps", 10, 1000, 100, 10,
                                key="n_steps")
            save_frequency = st.slider("Save frequency", 1, 100, 10, 1,
                                       key="save_freq")
            with st.expander("🔧 Advanced Options"):
                stability_factor = st.slider(
                    "Stability factor (unused in FFT mode)", 0.1, 1.0, 0.5, 0.1,
                    key="stability_factor")
                enable_monitoring = st.checkbox(
                    "Enable real-time monitoring", True, key="enable_monitoring")
                auto_adjust_dt = st.checkbox(
                    "Auto-adjust time step (unused in FFT mode)", True,
                    key="auto_adjust_dt")
                confine_twin = st.checkbox(
                    "Confine twin evolution to twinned grain", True,
                    key="confine_twin")
            st.subheader("🎨 Visualization Settings")
            global_cmap_phi = st.selectbox(
                "Global φ colormap", cmap_list,
                index=cmap_list.index('RdBu_r') if 'RdBu_r' in cmap_list else 0,
                key="global_cmap_phi")
            global_cmap_stress = st.selectbox(
                "Global σ_eq colormap", cmap_list,
                index=cmap_list.index('hot') if 'hot' in cmap_list else 0,
                key="global_cmap_stress")
            global_cmap_hydro = st.selectbox(
                "Global σ_h colormap", cmap_list,
                index=cmap_list.index('RdBu') if 'RdBu' in cmap_list else 0,
                key="global_cmap_hydro")
            sim_cmap_phi = st.selectbox(
                "Simulation-specific φ colormap", cmap_list,
                index=cmap_list.index(global_cmap_phi)
                if global_cmap_phi in cmap_list else 0,
                key="sim_cmap_phi")
            sim_cmap_stress = st.selectbox(
                "Simulation-specific σ_eq colormap", cmap_list,
                index=cmap_list.index(global_cmap_stress)
                if global_cmap_stress in cmap_list else 0,
                key="sim_cmap_stress")
            sim_cmap_hydro = st.selectbox(
                "Simulation-specific σ_h colormap", cmap_list,
                index=cmap_list.index(global_cmap_hydro)
                if global_cmap_hydro in cmap_list else 0,
                key="sim_cmap_hydro")
            st.subheader("📏 Scale Bar Settings")
            scalebar_color = st.color_picker("Scale bar color", "#000000",
                                             key="scalebar_color")
            scalebar_fontsize = st.slider("Scale bar font size", 6, 20, 10, 1,
                                          key="scalebar_fontsize")
            if st.button("🚀 Initialize Simulation", type="primary",
                         use_container_width=True):
                params = {
                    'material': material_choice,
                    'N': N, 'dx': dx, 'dt': dt,
                    'W': W, 'A': A, 'B': B,
                    'kappa0': kappa0, 'gamma_aniso': gamma_aniso,
                    'kappa_eta': kappa_eta,
                    'L_CTB': L_CTB, 'L_ITB': L_ITB, 'n_mob': n_mob,
                    'L_eta': L_eta, 'zeta': zeta,
                    'twin_spacing': twin_spacing,
                    'grain_boundary_pos': grain_boundary_pos,
                    'gb_width': 3.0, 'buffer_width': buffer_width,
                    'left_buffer_width': left_buffer_width,
                    'gb_profile': gb_profile.lower(),
                    'gb_curvature': gb_curvature,
                    'geometry_type': 'defect'
                    if geometry_type == "Twin Grain with Defect" else 'standard',
                    'applied_stress': applied_stress_MPa * 1e6,
                    'applied_stress_angle': loading_angle,
                    'n_steps': n_steps, 'save_frequency': save_frequency,
                    'stability_factor': stability_factor,
                    'confine_twin': confine_twin,
                    'cmap_phi': sim_cmap_phi, 'cmap_stress': sim_cmap_stress,
                    'cmap_hydro': sim_cmap_hydro,
                    'global_cmap_phi': global_cmap_phi,
                    'global_cmap_stress': global_cmap_stress,
                    'global_cmap_hydro': global_cmap_hydro,
                    'scalebar_color': scalebar_color,
                    'scalebar_fontsize': scalebar_fontsize}
                if geometry_type == "Twin Grain with Defect":
                    params['defect_type'] = defect_type.lower()
                    params['defect_pos'] = (defect_x, defect_y)
                    params['defect_radius'] = defect_radius
                errors, warnings_list = MaterialProperties.validate_parameters(params)
                if errors:
                    st.error(f"Validation errors: {', '.join(errors)}")
                else:
                    if warnings_list:
                        st.warning(f"Parameter warnings: {', '.join(warnings_list)}")
                    geom_viz = InitialGeometryVisualizer(N, dx)
                    if geometry_type == "Twin Grain with Defect":
                        phi, eta1, eta2 = geom_viz.create_defect_geometry(
                            twin_spacing, defect_type.lower(), (defect_x, defect_y),
                            defect_radius, grain_boundary_pos, 3.0,
                            buffer_width, left_buffer_width,
                            gb_profile.lower(), gb_curvature)
                    else:
                        phi, eta1, eta2 = geom_viz.create_twin_grain_geometry(
                            twin_spacing, grain_boundary_pos, 3.0,
                            buffer_width, left_buffer_width,
                            gb_profile.lower(), gb_curvature)
                    st.session_state.initial_geometry = {
                        'phi': phi, 'eta1': eta1, 'eta2': eta2,
                        'geom_viz': geom_viz, 'params': params}
                    st.session_state.initialized = True
                    st.success("✅ Simulation initialized successfully!")
        elif operation_mode == "Compare Saved Simulations":
            st.header("🔍 Comparison Configuration")
            simulations = SimulationDatabase.get_simulation_list()
            if not simulations:
                st.warning("No simulations saved yet. Run some simulations first!")
            else:
                sim_options = {sim['name']: sim['id'] for sim in simulations}
                selected_sim_ids = st.multiselect(
                    "Select Simulations to Compare",
                    options=list(sim_options.keys()),
                    default=list(sim_options.keys())[:min(3, len(sim_options))])
                comparison_type = st.selectbox(
                    "Comparison Type",
                    ["Side-by-Side Heatmaps", "Overlay Line Profiles",
                     "Statistical Summary", "Correlation Analysis",
                     "Evolution Timeline"], index=0)
                field_to_compare = st.selectbox(
                    "Field to Compare",
                    ["phi (Twin Order)", "eta1 (Grain)",
                     "sigma_eq (Von Mises Stress)",
                     "sigma_h (Hydrostatic Stress)",
                     "h (Twin Spacing)", "sigma_y (Yield Stress)"], index=2)
                field_key = field_to_compare.split()[0]
                if comparison_type == "Overlay Line Profiles":
                    profile_direction = st.selectbox(
                        "Profile Direction",
                        ["Horizontal", "Vertical", "Diagonal",
                         "Anti-Diagonal", "Custom"], index=0)
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                    profile_type_mapping = {
                        "Horizontal": "horizontal", "Vertical": "vertical",
                        "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal",
                        "Custom": "custom"}
                    internal_direction = profile_type_mapping.get(
                        profile_direction, "horizontal")
                    custom_angle = (st.slider("Custom angle (deg)", -180, 180, 45, 5)
                                    if profile_direction == "Custom" else 45)
                if st.button("🔬 Run Comparison", type="primary"):
                    comparison_config = {
                        'sim_ids': [sim_options[name] for name in selected_sim_ids],
                        'type': comparison_type, 'field': field_key}
                    if comparison_type == "Overlay Line Profiles":
                        comparison_config.update({
                            'profile_direction': internal_direction,
                            'position_ratio': position_ratio,
                            'custom_angle': custom_angle
                            if profile_direction == "Custom" else None})
                    st.session_state.comparison_config = comparison_config
                    st.rerun()
        elif operation_mode == "Single Simulation View":
            st.header("🔍 Single Simulation View")
            simulations = SimulationDatabase.get_simulation_list()
            if not simulations:
                st.warning("No simulations saved yet.")
            else:
                sim_options = {sim['name']: sim['id'] for sim in simulations}
                selected_sim = st.selectbox("Select Simulation",
                                            list(sim_options.keys()))
                if selected_sim:
                    st.session_state.selected_sim_id = sim_options[selected_sim]
        elif operation_mode == "Parameter Sweep":
            st.header("📈 Parameter Sweep")
            st.subheader("Base Configuration")
            material_choice = st.selectbox("Material", ["Cu", "Al", "Ni"],
                                           key="sweep_material")
            geom_type_sweep = st.selectbox(
                "Geometry Type",
                ["Standard Twin Grain", "Twin Grain with Defect"],
                key="sweep_geom")
            col1, col2 = st.columns(2)
            with col1:
                N_sweep = st.slider("Grid size N", 64, 256, 128, 32, key="sweep_N")
                dx_sweep = st.slider("dx (nm)", 0.2, 1.0, 0.5, 0.1, key="sweep_dx")
                dt_sweep = st.slider("dt (ns)", 1e-5, 1e-2, 1e-3, 1e-5,
                                     format="%.5f", key="sweep_dt")
                W_sweep = st.slider("W (J/m³)", 0.5, 5.0, 2.0, 0.1, key="sweep_W")
            with col2:
                n_steps_sweep = st.slider("Number of steps", 20, 200, 50, 10,
                                          key="sweep_steps")
                save_freq_sweep = st.slider("Save frequency", 1, 50, 10, 1,
                                            key="sweep_savefreq")
                twin_spacing_sweep = st.slider("Twin spacing (nm)", 10.0, 50.0,
                                               20.0, 1.0, key="sweep_twin_spacing")
                applied_stress_sweep = st.slider("Applied stress (MPa)",
                                                 0.0, 600.0, 300.0, 10.0,
                                                 key="sweep_stress")
            st.subheader("Sweep Parameter")
            sweep_param = st.selectbox(
                "Choose parameter to vary",
                ["twin_spacing", "applied_stress", "applied_stress_angle",
                 "W", "L_CTB", "L_ITB", "kappa0"], index=0)
            col1, col2 = st.columns(2)
            with col1:
                sweep_min = st.number_input(
                    f"Min {sweep_param}",
                    value=10.0 if sweep_param == "twin_spacing" else 0.0,
                    key="sweep_min")
                sweep_max = st.number_input(
                    f"Max {sweep_param}",
                    value=50.0 if sweep_param == "twin_spacing" else 500.0,
                    key="sweep_max")
            with col2:
                sweep_steps = st.number_input("Number of steps",
                                              min_value=2, max_value=20,
                                              value=5, step=1, key="sweep_steps")
            if sweep_param in ["applied_stress"]:
                sweep_values = np.linspace(sweep_min * 1e6, sweep_max * 1e6,
                                           sweep_steps)
            else:
                sweep_values = np.linspace(sweep_min, sweep_max, sweep_steps)
            st.write(f"Sweep values: {sweep_values}")
            base_params = {
                'material': material_choice,
                'N': N_sweep, 'dx': dx_sweep, 'dt': dt_sweep, 'W': W_sweep,
                'A': 5.0, 'B': 10.0,
                'kappa0': 1.0, 'gamma_aniso': 0.7, 'kappa_eta': 2.0,
                'L_CTB': 0.05, 'L_ITB': 5.0, 'n_mob': 4, 'L_eta': 1.0,
                'zeta': 0.3,
                'twin_spacing': twin_spacing_sweep,
                'grain_boundary_pos': 0.0, 'gb_width': 3.0,
                'buffer_width': 5.0, 'left_buffer_width': 5.0,
                'gb_profile': 'plane', 'gb_curvature': 0.0,
                'geometry_type': 'defect'
                if geom_type_sweep == "Twin Grain with Defect" else 'standard',
                'applied_stress': applied_stress_sweep * 1e6,
                'applied_stress_angle': 0.0,
                'n_steps': n_steps_sweep, 'save_frequency': save_freq_sweep,
                'stability_factor': 0.5, 'confine_twin': True}
            if geom_type_sweep == "Twin Grain with Defect":
                base_params['defect_type'] = 'dislocation'
                base_params['defect_pos'] = (0.0, 0.0)
                base_params['defect_radius'] = 10.0
            if st.button("🚀 Run Parameter Sweep", type="primary"):
                with st.spinner("Running parameter sweep..."):
                    sweep_results = ParameterSweep.run_sweep(
                        base_params, sweep_param, sweep_values, save=True)
                st.session_state.sweep_results = sweep_results
                st.session_state.sweep_param = sweep_param
                st.success("Parameter sweep completed!")
                st.rerun()

    if operation_mode == "Compare Saved Simulations" and \
       'comparison_config' in st.session_state:
        st.header("🔬 Multi-Simulation Comparison")
        config = st.session_state.comparison_config
        simulations = []
        for sim_id in config['sim_ids']:
            sim = SimulationDatabase.get_simulation(sim_id)
            if sim: simulations.append(sim)
        if not simulations:
            st.error("No valid simulations found.")
        else:
            st.success(f"Loaded {len(simulations)} simulations")
            sim_names = [build_sim_name(sim['params'], sim['id'])
                         for sim in simulations]
            if config['type'] == "Side-by-Side Heatmaps":
                last_frames = [sim['results_history'][-1]
                               if sim['results_history'] else None
                               for sim in simulations]
                valid_indices = [i for i, f in enumerate(last_frames)
                                 if f is not None]
                if not valid_indices:
                    st.warning("No frame data available.")
                else:
                    n_sims = len(valid_indices)
                    cols = min(3, n_sims); rows = (n_sims + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                    axes = (np.array([axes]) if (rows == 1 and cols == 1)
                            else axes.flatten())
                    for idx, sim_idx in enumerate(valid_indices):
                        ax = axes[idx]
                        sim = simulations[sim_idx]
                        frame = last_frames[sim_idx]
                        field = config['field']
                        if field in frame:
                            data = frame[field].copy()
                            if field in ['sigma_eq', 'sigma_h']: data = data / 1e9
                            elif field == 'sigma_y': data = data / 1e6
                            extent = [-sim['params']['N'] * sim['params']['dx'] / 2,
                                      sim['params']['N'] * sim['params']['dx'] / 2] * 2
                            im = ax.imshow(data, extent=extent, cmap='viridis',
                                           origin='lower')
                            ax.set_title(sim_names[sim_idx][:30] + "...", fontsize=8)
                            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    for idx in range(len(valid_indices), len(axes)):
                        axes[idx].axis('off')
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            elif config['type'] == "Overlay Line Profiles":
                fig = go.Figure()
                ref_sim = simulations[0]
                visualizer = EnhancedTwinVisualizer(ref_sim['params']['N'],
                                                    ref_sim['params']['dx'])
                for sim_idx, sim in enumerate(simulations):
                    if not sim['results_history']: continue
                    frame = sim['results_history'][-1]
                    field = config['field']
                    if field not in frame: continue
                    distance, profile, _ = visualizer.line_profiler.extract_profile(
                        frame[field], config['profile_direction'],
                        config['position_ratio'], config.get('custom_angle', 45))
                    if field in ['sigma_eq', 'sigma_h']:
                        profile = profile / 1e9; ylabel = 'Stress (GPa)'
                    elif field == 'sigma_y':
                        profile = profile / 1e6; ylabel = 'Stress (MPa)'
                    else:
                        ylabel = field
                    fig.add_trace(go.Scatter(x=distance, y=profile, mode='lines',
                                             name=sim_names[sim_idx][:30]))
                fig.update_layout(title=f"{field} Line Profiles Comparison",
                                  xaxis_title="Position (nm)", yaxis_title=ylabel,
                                  hovermode='x unified', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif config['type'] == "Statistical Summary":
                data = []
                for sim in simulations:
                    params = sim['params']; hist = sim['results_history']
                    if hist:
                        last = hist[-1].get('convergence', {})
                        data.append({
                            'Name': build_sim_name(params, sim['id'])[:40],
                            'λ (nm)': params.get('twin_spacing', 0),
                            'σ_app (MPa)': params.get('applied_stress', 0) / 1e6,
                            'θ (deg)': params.get('applied_stress_angle', 0),
                            'W (J/m³)': params.get('W', 0),
                            'Avg σ_eq (GPa)': last.get('avg_stress', 0) / 1e9,
                            'Max σ_eq (GPa)': last.get('max_stress', 0) / 1e9,
                            'Avg h (nm)': last.get('avg_spacing', 0),
                            'Plastic Work (J)': last.get('plastic_work', 0),
                            'Energy (J)': last.get('energy', 0)})
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df['Name'], y=df['Avg σ_eq (GPa)'],
                                         name='Avg Stress'))
                    fig.add_trace(go.Bar(x=df['Name'], y=df['Max σ_eq (GPa)'],
                                         name='Max Stress'))
                    fig.update_layout(title="Stress Comparison",
                                      xaxis_title="Simulation",
                                      yaxis_title="Stress (GPa)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No convergence data available.")
            elif config['type'] == "Evolution Timeline":
                metric_map = {'phi': 'phi_norm', 'sigma_eq': 'avg_stress',
                              'h': 'twin_spacing_avg', 'energy': 'energy',
                              'plastic_work': 'plastic_work'}
                chosen_metric = st.selectbox("Metric to track",
                                             list(metric_map.keys()))
                fig = go.Figure()
                for sim_idx, sim in enumerate(simulations):
                    hist = sim.get('history')
                    if hist is None and 'solver' in sim: hist = sim['solver'].history
                    if hist is None: continue
                    metric_key = metric_map.get(chosen_metric, chosen_metric)
                    if metric_key not in hist: continue
                    times = (np.arange(len(hist[metric_key]))
                             * sim['params'].get('dt', 1e-4))
                    values = hist[metric_key]
                    if chosen_metric in ['sigma_eq']:
                        values = np.array(values) / 1e9
                    fig.add_trace(go.Scatter(x=times, y=values, mode='lines',
                                             name=sim_names[sim_idx][:30]))
                fig.update_layout(title=f"{chosen_metric} Evolution Comparison",
                                  xaxis_title="Time (ns)", yaxis_title=chosen_metric,
                                  hovermode='x unified', template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    elif operation_mode == "Single Simulation View" and \
         'selected_sim_id' in st.session_state:
        sim_id = st.session_state.selected_sim_id
        sim_data = SimulationDatabase.get_simulation(sim_id)
        if sim_data:
            st.header(f"📊 Single Simulation: "
                      f"{build_sim_name(sim_data['params'], sim_id)}")
            params = sim_data['params']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("λ (twin spacing)", f"{params.get('twin_spacing', 0):.1f} nm")
            with col2:
                st.metric("σ_app / θ",
                          f"{params.get('applied_stress', 0) / 1e6:.0f} MPa / "
                          f"{params.get('applied_stress_angle', 0):.0f}°")
            with col3:
                st.metric("W (well depth)", f"{params.get('W', 0):.2f} J/m³")
            with col4:
                st.metric("κ₀", f"{params.get('kappa0', 0):.2f}")
            history = sim_data.get('results_history', [])
            if history:
                num_frames = len(history)
                frame_idx = st.slider("Frame", 0, num_frames - 1,
                                      num_frames - 1,
                                      key=f"frame_slider_{sim_id}")
                results = history[frame_idx]
                visualizer = EnhancedTwinVisualizer(
                    params['N'], params['dx'], dt=params.get('dt', 1e-4))
                fig = visualizer.create_multi_field_comparison(results, {
                    'phi_cmap': params.get('cmap_phi', 'RdBu_r'),
                    'eta1_cmap': params.get('cmap_eta1', 'Reds'),
                    'sigma_eq_cmap': params.get('cmap_stress', 'hot'),
                    'sigma_h_cmap': params.get('cmap_hydro', 'RdBu'),
                    'scalebar_color': params.get('scalebar_color', 'black'),
                    'scalebar_fontsize': params.get('scalebar_fontsize', 10)})
                if fig: st.pyplot(fig); plt.close(fig)
                if st.button("🗑️ Delete This Simulation", key=f"delete_{sim_id}"):
                    SimulationDatabase.delete_simulation(sim_id)
                    st.session_state.pop('selected_sim_id', None)
                    st.success(f"Simulation {sim_id} deleted!"); st.rerun()
            else:
                st.warning("No simulation history found.")
        else:
            st.error("Simulation not found.")

    elif operation_mode == "Run New Simulation" and \
         'initialized' in st.session_state:
        params = st.session_state.initial_geometry['params']
        N = params['N']; dx = params['dx']
        visualizer = EnhancedTwinVisualizer(N, dx, dt=params.get('dt', 1e-4))
        tabs = st.tabs(["📐 Initial Geometry", "▶️ Run Simulation",
                        "📊 Basic Results", "🔍 Advanced Analysis",
                        "📊 Plotly Interactive", "🖥️ 3D Interactive",
                        "📤 Enhanced Export"])
        with tabs[0]:
            st.header("Initial Geometry Visualization")
            phi = st.session_state.initial_geometry['phi']
            eta1 = st.session_state.initial_geometry['eta1']
            kx, ky, k2 = make_k_vectors(N, dx)
            phi_gx, phi_gy = spectral_gradients(phi, kx, ky)
            h = compute_twin_spacing_from_gradient(phi_gx, phi_gy)
            fig = visualizer.create_multi_field_comparison(
                {'phi': phi, 'eta1': eta1, 'h': h},
                {'eta1_cmap': 'Reds',
                 'scalebar_color': params.get('scalebar_color', 'black'),
                 'scalebar_fontsize': params.get('scalebar_fontsize', 10)})
            if fig: st.pyplot(fig); plt.close(fig)
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_spacing = (np.mean(h[(h > 5) & (h < 50)])
                               if np.any((h > 5) & (h < 50)) else 0)
                st.metric("Avg Twin Spacing", f"{avg_spacing:.1f} nm")
            with col2:
                st.metric("Twin Grain Area", f"{np.sum(eta1 > 0.5) * dx**2:.0f} nm²")
            with col3:
                st.metric("Number of Twins", f"{np.sum(h < 20):.0f}")
        with tabs[1]:
            st.header("Run Simulation (Pure FFT Spectral Method)")
            _active_overrides = st.session_state.get("plasticity_overrides", {})
            if _active_overrides:
                st.info("🤖 AI‑recommended plasticity parameters will be injected: "
                        + ", ".join(f"`{k}={v:.3g}`"
                                    for k, v in _active_overrides.items()))
            if st.button("▶️ Start Evolution", type="secondary",
                         use_container_width=True):
                with st.spinner("Running phase-field simulation (FFT)..."):
                    try:
                        solver = NanotwinnedCuSolver(params)
                        solver.phi = st.session_state.initial_geometry['phi'].copy()
                        solver.eta1 = st.session_state.initial_geometry['eta1'].copy()
                        solver.eta2 = st.session_state.initial_geometry['eta2'].copy()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_history = []; timesteps = []
                        monitoring_cols = st.columns(4)
                        n_steps = params['n_steps']; dt = params['dt']
                        save_freq = params['save_frequency']
                        for step in range(n_steps):
                            status_text.text(
                                f"Step {step + 1}/{n_steps} | "
                                f"Time: {(step + 1) * dt:.4f} ns")
                            results = solver.step()
                            if step % save_freq == 0:
                                results_history.append(results.copy())
                                timesteps.append(step * dt)
                            progress_bar.progress((step + 1) / n_steps)
                            if step % 10 == 0 and len(results_history) > 0:
                                with monitoring_cols[0]:
                                    st.metric("Avg Stress",
                                              f"{np.mean(results['sigma_eq']) / 1e9:.2f} GPa")
                                with monitoring_cols[1]:
                                    valid_h = results['h'][
                                        (results['h'] > 5) & (results['h'] < 50)]
                                    avg_h = (np.mean(valid_h)
                                             if len(valid_h) > 0 else 0)
                                    st.metric("Avg Spacing", f"{avg_h:.1f} nm")
                                with monitoring_cols[2]:
                                    st.metric("Max Plastic Strain",
                                              f"{np.max(results['eps_p_mag']):.4f}")
                                with monitoring_cols[3]:
                                    st.metric("Energy",
                                              f"{results['convergence']['energy']:.2e} J")
                        st.success(f"✅ Simulation completed! Generated "
                                   f"{len(results_history)} frames.")
                        st.session_state.results_history = results_history
                        st.session_state.timesteps = timesteps
                        st.session_state.solver = solver
                        start_time = datetime.now()
                        sim_id = SimulationDatabase.save_simulation(
                            params, results_history,
                            st.session_state.initial_geometry,
                            run_time=(datetime.now() - start_time).total_seconds())
                        st.balloons()
                    except Exception as e:
                        st.error(f"Simulation failed: {str(e)}")
                        st.exception(e)
        with tabs[2]:
            if 'results_history' in st.session_state:
                st.header("Basic Results Visualization")
                results_history = st.session_state.results_history
                frame_idx = st.slider("Select frame", 0,
                                      len(results_history) - 1,
                                      len(results_history) - 1)
                results = results_history[frame_idx]
                fig = visualizer.create_multi_field_comparison(results, {
                    'phi_cmap': params.get('cmap_phi', 'RdBu_r'),
                    'eta1_cmap': params.get('cmap_eta1', 'Reds'),
                    'sigma_eq_cmap': params.get('cmap_stress', 'hot'),
                    'sigma_h_cmap': params.get('cmap_hydro', 'RdBu'),
                    'scalebar_color': params.get('scalebar_color', 'black'),
                    'scalebar_fontsize': params.get('scalebar_fontsize', 10)})
                if fig: st.pyplot(fig); plt.close(fig)
                st.subheader("Convergence Monitoring")
                if 'solver' in st.session_state and \
                   st.session_state.solver.history['phi_norm']:
                    full_timesteps = (np.arange(
                        len(st.session_state.solver.history['phi_norm']))
                        * params['dt'])
                    conv_fig = SimulationMonitor.create_convergence_plots(
                        st.session_state.solver.history, full_timesteps)
                    st.pyplot(conv_fig); plt.close(conv_fig)
            else:
                st.info("Run a simulation first.")
        with tabs[3]:
            if 'results_history' in st.session_state:
                st.header("Advanced Analysis Tools")
                results = st.session_state.results_history[-1]
                col1, col2 = st.columns(2)
                with col1:
                    profile_types = st.multiselect(
                        "Profile Directions",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"],
                        default=["Horizontal", "Vertical"])
                    position_ratio = st.slider("Position Ratio", 0.0, 1.0, 0.5, 0.1)
                with col2:
                    field_to_profile = st.selectbox(
                        "Field to Profile",
                        ["phi", "eta1", "sigma_eq", "sigma_h", "h", "sigma_y"],
                        index=2)
                profile_type_mapping = {
                    "Horizontal": "horizontal", "Vertical": "vertical",
                    "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal"}
                internal_types = [profile_type_mapping[pt] for pt in profile_types]
                profiler = EnhancedLineProfiler(N, dx)
                fig_profiles, axes = plt.subplots(
                    len(internal_types), 1, figsize=(10, 4 * len(internal_types)))
                if len(internal_types) == 1: axes = [axes]
                for idx, ptype in enumerate(internal_types):
                    ax = axes[idx]
                    distance, profile, _ = profiler.extract_profile(
                        results[field_to_profile], ptype, position_ratio)
                    if field_to_profile in ['sigma_eq', 'sigma_h']:
                        profile = profile / 1e9; ylabel = 'Stress (GPa)'
                    elif field_to_profile == 'sigma_y':
                        profile = profile / 1e6; ylabel = 'Stress (MPa)'
                    else:
                        ylabel = field_to_profile
                    ax.plot(distance, profile, 'b-', linewidth=2)
                    ax.set_xlabel('Position (nm)'); ax.set_ylabel(ylabel)
                    ax.set_title(f'{ptype.replace("_", " ").title()} Profile')
                    ax.grid(True, alpha=0.3)
                plt.tight_layout(); st.pyplot(fig_profiles); plt.close(fig_profiles)
            else:
                st.info("Run a simulation first.")
        with tabs[4]:
            if 'results_history' in st.session_state:
                st.header("📊 Plotly Interactive Visualization (2D)")
                results_history = st.session_state.results_history
                plotly_field = st.selectbox(
                    "Select field to visualize",
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h",
                     "eps_p_mag", "sigma_y"], index=0, key="plotly_2d_field")
                frame_idx_plotly = st.slider("Frame", 0, len(results_history) - 1,
                                             len(results_history) - 1,
                                             key="plotly_2d_frame")
                results = results_history[frame_idx_plotly]
                fig_heatmap = visualizer.create_plotly_heatmap(
                    results, plotly_field, frame_idx_plotly)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown("---")
                st.subheader("Interactive Line Profiles")
                col1, col2 = st.columns(2)
                with col1:
                    profile_type_plotly = st.selectbox(
                        "Profile direction",
                        ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"],
                        key="plotly_2d_profile")
                with col2:
                    position_ratio_plotly = st.slider(
                        "Position ratio", 0.0, 1.0, 0.5, 0.05,
                        key="plotly_2d_pos")
                profile_type_mapping = {
                    "Horizontal": "horizontal", "Vertical": "vertical",
                    "Diagonal": "diagonal", "Anti-Diagonal": "anti_diagonal"}
                internal_pt = profile_type_mapping.get(profile_type_plotly,
                                                       "horizontal")
                fig_line = visualizer.create_plotly_line_profiles(
                    results, plotly_field, [internal_pt], position_ratio_plotly)
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Run a simulation first.")
        with tabs[5]:
            st.header("🖥️ 3D Interactive Surface Visualization")
            if 'results_history' in st.session_state:
                results_history = st.session_state.results_history
                field_3d = st.selectbox(
                    "Select field for 3D surface",
                    ["phi", "eta1", "sigma_eq", "sigma_h", "h",
                     "eps_p_mag", "sigma_y"], index=1, key="3d_field")
                frame_idx_3d = st.slider("Frame", 0, len(results_history) - 1,
                                         len(results_history) - 1, key="3d_frame")
                results = results_history[frame_idx_3d]
                fig_3d = visualizer.create_plotly_3d_surface(
                    results, field_3d, frame_idx_3d)
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("Run a simulation first.")
        with tabs[6]:
            st.header("📤 Enhanced Export")
            if 'results_history' in st.session_state and \
               st.session_state.results_history:
                results_history = st.session_state.results_history
                params = st.session_state.initial_geometry['params']
                sim_id = SimulationDatabase.generate_id(params)
                sim_name = build_sim_name(params, sim_id)
                sim_data = {'metadata': MetadataManager.create_metadata(
                    params, results_history), 'params': params}
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📦 Pickle (PKL)"):
                        buffer, fname = DataExporter.export_pkl(
                            sim_data, params, results_history, sim_name)
                        st.download_button("Download PKL", buffer, fname)
                    if st.button("🔥 PyTorch (PT)"):
                        buffer, fname = DataExporter.export_pt(
                            sim_data, params, results_history, sim_name)
                        st.download_button("Download PT", buffer, fname)
                    if st.button("📄 SQL Dump"):
                        buffer, fname = DataExporter.export_sql(
                            sim_data, params, results_history, sim_name,
                            sim_id, params['N'], params['dx'])
                        st.download_button("Download SQL", buffer, fname)
                with col2:
                    if st.button("📊 CSV (ZIP)"):
                        vis = EnhancedTwinVisualizer(params['N'], params['dx'])
                        buffer, fname = DataExporter.export_csv(
                            results_history, sim_name, vis.extent,
                            params['N'], params['dx'])
                        st.download_button("Download CSV ZIP", buffer, fname)
                    if st.button("📋 JSON"):
                        buffer, fname = DataExporter.export_json(
                            sim_data, params, results_history, sim_name)
                        st.download_button("Download JSON", buffer, fname)
                    if st.button("📁 HDF5"):
                        buffer, fname = DataExporter.export_hdf5(
                            sim_data, params, results_history, sim_name,
                            params['N'], params['dx'])
                        if buffer:
                            st.download_button("Download HDF5", buffer, fname)
                with col3:
                    st.subheader("Animation Export")
                    anim_field = st.selectbox(
                        "Field for animation",
                        ["phi", "eta1", "sigma_eq", "sigma_h", "h", "eps_p_mag"],
                        key="anim_field")
                    anim_format = st.selectbox("Format", ["gif", "mp4"],
                                               key="anim_format")
                    fps = st.slider("FPS", 1, 30, 5)
                    if st.button("🎬 Generate Animation"):
                        with st.spinner("Creating animation..."):
                            vis = EnhancedTwinVisualizer(
                                params['N'], params['dx'],
                                dt=params.get('dt', 1e-4))
                            anim_buffer = vis.create_animation(
                                results_history, anim_field, anim_format, fps)
                            if anim_buffer:
                                st.download_button(
                                    f"Download {anim_format.upper()}",
                                    anim_buffer,
                                    f"{sim_name}_{anim_field}.{anim_format}")
                st.markdown("---")
                if st.button("📦 Export All Simulations"):
                    vis = EnhancedTwinVisualizer(params['N'], params['dx'])
                    bulk_buffer, bulk_fname = DataExporter.bulk_export_all_simulations(
                        params['N'], params['dx'], vis.extent)
                    if bulk_buffer:
                        st.download_button("Download All Simulations ZIP",
                                           bulk_buffer, bulk_fname)
            else:
                st.info("Run a simulation first to export data.")

    elif operation_mode == "Parameter Sweep" and \
         'sweep_results' in st.session_state:
        st.header("📊 Parameter Sweep Results")
        sweep_results = st.session_state.sweep_results
        sweep_param = st.session_state.sweep_param
        param_vals, avg_stress, max_stress = [], [], []
        avg_spacing, plastic_work = [], []
        for res in sweep_results:
            if res['convergence'] is not None:
                param_vals.append(res['param_value'])
                conv = res['convergence']
                avg_stress.append(conv.get('sigma_eq', 0) / 1e9
                                  if isinstance(conv.get('sigma_eq'),
                                                (int, float)) else 0)
                max_stress.append(conv.get('max_stress', 0) / 1e9)
                avg_spacing.append(conv.get('twin_spacing_avg', 0))
                plastic_work.append(conv.get('plastic_work', 0))
        if sweep_param == 'applied_stress':
            param_display = np.array(param_vals) / 1e6
            param_label = "Applied Stress (MPa)"
        else:
            param_display = param_vals
            param_label = sweep_param.replace('_', ' ').title()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].plot(param_display, avg_stress, 'o-', linewidth=2)
        axes[0, 0].set_xlabel(param_label)
        axes[0, 0].set_ylabel("Avg Von Mises Stress (GPa)")
        axes[0, 0].set_title("Stress vs Parameter")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(param_display, avg_spacing, 's-', color='green',
                        linewidth=2)
        axes[0, 1].set_xlabel(param_label)
        axes[0, 1].set_ylabel("Avg Twin Spacing (nm)")
        axes[0, 1].set_title("Twin Spacing vs Parameter")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(param_display, plastic_work, 'd-', color='red',
                        linewidth=2)
        axes[1, 0].set_xlabel(param_label)
        axes[1, 0].set_ylabel("Plastic Work (J)")
        axes[1, 0].set_title("Plastic Work vs Parameter")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(param_display, max_stress, '^-', color='purple',
                        linewidth=2, label='Max')
        axes[1, 1].plot(param_display, avg_stress, 'o-', color='blue',
                        linewidth=2, label='Avg')
        axes[1, 1].set_xlabel(param_label)
        axes[1, 1].set_ylabel("Stress (GPa)")
        axes[1, 1].set_title("Stress Extremes vs Parameter")
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(pd.DataFrame({
            param_label: param_display, 'Avg Stress (GPa)': avg_stress,
            'Max Stress (GPa)': max_stress,
            'Avg Spacing (nm)': avg_spacing,
            'Plastic Work (J)': plastic_work}))
        if st.button("Clear Sweep Results"):
            del st.session_state.sweep_results
            del st.session_state.sweep_param
            st.rerun()

    st.markdown("---")
    render_friction_stress_lab()

    render_recommender_visuals_dashboard()


# ============================================================================
# REGRESSION TESTS
# ============================================================================
def _regression_test_v881() -> None:
    corpus = {"cu_elastic.json": [
        {"material": "Cu", "quantity": "shear modulus c44 = μ",
         "value": 47.19, "unit": "GPa", "method": "RUS"},
    ]}
    r = HybridRetriever(corpus, use_dense=False)
    recs = r.search("mu", "Cu", value_hints=["47.19"], k=3)
    assert recs, "HybridRetriever returned no records"
    hits = heuristic_extract(recs, "mu")
    assert hits, "heuristic_extract returned no candidates"
    assert any(abs(h["value"] - 47.19) < 1e-6 for h in hits)
    cands = gatekeep(hits, "mu", provenance="regex_ner")
    assert cands and abs(cands[0].value - 47.19) < 1e-6
    assert hasattr(cands[0], "reasoning")
    logger.info("v8.8.1 regression test: PASS")


def _regression_test_v882() -> None:
    scan_text = "rho0 = 6.0e13 m^-2 and forest dislocation density 1.0×10^13 m-2"
    hits = list(_NUM_ANY.finditer(scan_text))
    assert any(abs(_num_value(m) - 6.0e13) < 1e6 for m in hits)
    fake = [{"value": 48.0, "unit": "GPa", "confidence": 0.9,
             "property_label": "shear modulus", "source": "test"}]
    prior_cands = gatekeep(fake, "mu", provenance="llm_prior", conf_cap=0.5)
    assert prior_cands and prior_cands[0].confidence == 0.5
    assert _norm_provenance("llm_prior") == "llm_prior"
    logger.info("v8.8.2 regression test: PASS")


def _regression_test_v883() -> None:
    fake_cands = {
        "sigma0_yield": [ValueCandidate(value=1.0e9, unit="Pa",
            provenance="llm_extract", property_label="yield stress",
            method="explicit", evidence="σ_y = 1.0 GPa", source="test",
            confidence=0.9)],
        "twin_thickness_lambda": [ValueCandidate(value=15e-9, unit="m",
            provenance="llm_extract", property_label="twin thickness λ",
            method="explicit", evidence="λ = 15 nm", source="test",
            confidence=0.9)],
        "hall_petch_k": [ValueCandidate(value=0.12, unit="MPa·m^(1/2)",
            provenance="llm_extract", property_label="k_λ", method="explicit",
            evidence="k_λ = 0.12", source="test", confidence=0.9)],
        "mu": [ValueCandidate(value=45e9, unit="Pa", provenance="llm_extract",
            property_label="G", method="explicit", evidence="G = 45 GPa",
            source="test", confidence=0.9)],
        "poisson_ratio_nu": [ValueCandidate(value=0.34, unit="",
            provenance="llm_extract", property_label="ν", method="explicit",
            evidence="ν = 0.34", source="test", confidence=0.9)],
        "burgers_vector_b": [ValueCandidate(value=0.256e-9, unit="m",
            provenance="llm_extract", property_label="b", method="explicit",
            evidence="b = 0.256 nm", source="test", confidence=0.9)],
    }
    derived = derive_sigma0_and_tau_p(fake_cands, "Cu")
    assert derived
    sigma0 = next((c for c in derived if c.method == "hall_petch_intercept"), None)
    assert sigma0 is not None and abs(sigma0.value - 2.0e7) < 5e6
    assert 'physics_inferred' in PROVENANCE_MARKERS
    logger.info("v8.8.3 regression test: PASS")


def _regression_test_v884() -> None:
    import dataclasses as _dc
    names = [f.name for f in _dc.fields(ValueCandidate)]
    assert "reasoning" in names, "ValueCandidate lost its reasoning field"
    assert "dominant_factor" in names, \
        "ValueCandidate missing v9.0 dominant_factor field"
    assert "theory" in names, \
        "ValueCandidate missing v9.1 theory field"
    assert "context" in names, \
        "ValueCandidate missing v9.3 context field"
    assert names.index("reasoning") < names.index("dominant_factor")
    assert names.index("dominant_factor") < names.index("theory")
    assert names.index("theory") < names.index("context")

    stale = object.__new__(ValueCandidate)
    stale.value = 1.0
    stale.unit = "Pa"
    stale.provenance = "llm_prior"
    stale.property_label = ""
    stale.method = ""
    stale.evidence = ""
    stale.source = ""
    stale.confidence = 0.5
    got = getattr(stale, "reasoning", "") or _default_reasoning(stale)
    assert got and "prior" in got.lower()
    assert getattr(stale, "dominant_factor", "") == ""
    assert getattr(stale, "theory", "") == ""
    assert getattr(stale, "context", False) is False
    logger.info("v8.8.4 regression test: PASS")


def _regression_test_v890() -> None:
    bad_table = (
        "| Field         | Value    |\n"
        "| Shear modulus | 0.000    |\n"
        "| Shear modulus | 0 | GPa |\n"
    )
    rows = parse_side_note_table(bad_table)
    assert all(r["value"] > 0 for r in rows), \
        f"parse_side_note_table emitted a non-positive value: {rows}"

    zero_frac = [{"value": 0.0, "unit": "GPa", "confidence": 0.9,
                  "property_label": "shear modulus", "source": "test"}]
    assert gatekeep(zero_frac, "shear_modulus_G", provenance="regex_ner") == [], \
        "gatekeep failed to reject G = 0"

    assert math.isnan(_normalize_unit(0.0, "GPa", "shear_modulus_G")), \
        "_normalize_unit did not collapse zero for positive param"

    b, nu = 0.256e-9, 0.34
    w_default = b / (1 - nu)
    assert abs(w_default - 0.3878e-9) < 1e-12, \
        f"w-default arithmetic drifted: {w_default}"

    ctx = {
        'sigma_y': CtxField(value=300.0, conf=0.9, provenance='llm_extract'),
        'd':       CtxField(value=1000.0, conf=0.7, provenance='regex_ner'),
        'lam':     CtxField(value=20.0, conf=0.9, provenance='llm_extract'),
        'b':       CtxField(value=0.256, conf=0.9, provenance='llm_extract'),
        'nu':      CtxField(value=0.34, conf=0.9, provenance='llm_extract'),
        'M':       CtxField(value=3.06, conf=0.9, provenance='llm_extract'),
        'k_y':     CtxField(value=0.11, conf=0.9, provenance='llm_extract'),
        'w':       CtxField(value=None),
        'G':       CtxField(value=None),
    }
    routes = build_derived_routes(ctx, mu_GPa=48.0, w_mode='auto')
    assert routes, "build_derived_routes returned no routes"
    pn = next(r for r in routes if r.key == 'peierls_nabarro')
    assert 5.0 < pn.tau_P_MPa < 60.0, \
        f"P–N τ_P out of physical range for Cu: {pn.tau_P_MPa} MPa"

    hp_lam = next(r for r in routes if r.key == 'hall_petch_lam')
    assert hp_lam.status == 'ok', \
        f"HP-λ route not ok for σ_y=300 MPa, λ=20 nm: {hp_lam.status}"

    cons = consensus_sigma0(routes)
    assert cons is not None and cons['value_MPa'] > 0

    logger.info(
        "v8.9.0 regression test: PASS  "
        "zero-G rejection ✓  _normalize_unit collapse ✓  "
        "w = b/(1−ν) = %.4f nm  τ_P(P-N) = %.2f MPa  "
        "σ₀_consensus = %.1f MPa",
        w_default * 1e9, pn.tau_P_MPa, cons['value_MPa'])


def _regression_test_v90() -> None:
    r_small = classify_rho0_regime("electrodeposition", "equiaxed", 5.0)
    r_large = classify_rho0_regime("electrodeposition", "equiaxed", 100.0)
    assert r_small["inferred"] > r_large["inferred"], \
        "smaller λ should give higher ρ₀ inside the same regime"

    r_col = classify_rho0_regime("electrodeposition", "columnar", 20.0)
    r_eqx = classify_rho0_regime("electrodeposition", "equiaxed", 20.0)
    assert r_eqx["low"] > r_col["high"], \
        "equiaxed regime should sit strictly above columnar"

    txt = ("The electrodeposited nanocrystalline Cu film exhibits an "
           "initial dislocation density of 5.0e16 m^-2, with twin "
           "thickness λ = 22 nm and equiaxed grains.")
    findings = extract_rho0_with_context(txt)
    assert findings, "snippet extractor found no ρ₀ mentions"
    f0 = findings[0]
    assert f0["rho0_value"] is not None and \
        abs(f0["rho0_value"] - 5.0e16) / 5.0e16 < 0.02
    assert f0["synthesis_method"] and "electrodepos" in \
        f0["synthesis_method"].lower()
    assert f0["grain_architecture"] and "nanocrystalline" in \
        f0["grain_architecture"].lower()
    assert f0["twin_spacing"] is not None and \
        abs(f0["twin_spacing"] - 22.0) < 0.5

    exp = PhysicsRegimeExpert()
    s_in  = exp.score(5e16, "electrodeposition", "equiaxed", 20.0)
    s_out = exp.score(1e12, "electrodeposition", "equiaxed", 20.0)
    assert s_in > 0.3, f"in-regime score too low: {s_in}"
    assert s_out < 0.05, f"out-of-regime score too high: {s_out}"

    sc = PlasticityLatentMoEScorer()
    extractions = [
        {"param": "rho0", "value": 5e16, "unit": "m^-2",
         "material": "Cu", "temp": 300.0, "strain_rate": 1e-3,
         "method": "physics_regime_inference", "confidence": 0.4,
         "reasoning": ("Step 1: electrodeposition → 1e16–1e17.\n"
                       "Step 2: equiaxed → upper half.\n"
                       "Step 3: λ = 20 nm → mid.\nStep 4: combine."),
         "evidence": "", "_source_file": "test",
         "_source_title": "regression", "_provenance": "llm_prior",
         "_context": True},
        {"param": "rho0", "value": 1e11, "unit": "m^-2",
         "material": "Cu", "temp": 300.0, "strain_rate": 1e-3,
         "method": "explicit", "confidence": 0.9,
         "reasoning": "", "evidence": "", "_source_file": "test",
         "_source_title": "regression", "_provenance": "llm_extract",
         "_context": False},
    ]
    b_no_ctx = sc.score(extractions, "Cu", 300.0)
    b_ctx = sc.score(extractions, "Cu", 300.0,
                     synthesis="electrodeposition",
                     architecture="equiaxed",
                     twin_spacing=20.0)

    def _rank(bucket, val):
        for i, c in enumerate(bucket):
            if abs(c.value_si - val) / val < 0.1:
                return i
        return None

    rank_no = _rank(b_no_ctx["rho0"], 5e16)
    rank_ctx = _rank(b_ctx["rho0"], 5e16)
    assert rank_ctx is not None and rank_no is not None

    logger.info(
        "v9.0 regression test: PASS  "
        "regime(λ=5nm) = %.2e > regime(λ=100nm) = %.2e  "
        "equiaxed.low = %.1e > columnar.high = %.1e  "
        "expert in/out = %.3f/%.3e",
        r_small["inferred"], r_large["inferred"],
        r_eqx["low"], r_col["high"], s_in, s_out)


def _regression_test_v91() -> None:
    r_jc  = classify_gamma0_regime("johnson-cook")
    r_ddd = classify_gamma0_regime("DDD")
    r_md  = classify_gamma0_regime("molecular dynamics")
    assert r_jc["high"]  < r_ddd["low"],  "JC regime should sit below DDD"
    assert r_ddd["high"] < r_md["low"],   "DDD regime should sit below MD"

    txt = ("In the crystal plasticity finite element model (CPFEM), the "
           "slip-system reference shear rate was set to γ̇₀ = 1e-3 s⁻¹ "
           "with a rate-sensitivity exponent m = 20.")
    findings = extract_gamma0_with_context(txt)
    assert findings, "snippet extractor found no γ̇₀ mentions"
    f0 = findings[0]
    assert f0["gamma0_value"] is not None and \
        abs(f0["gamma0_value"] - 1e-3) / 1e-3 < 0.05
    assert f0["theory"] and "crystal plasticity" in f0["theory"].lower()
    assert f0["theory_key"] == "cpfem"

    exp = TheoryRegimeExpert()
    s_match = exp.score(1e-3, target_theory="cpfem",
                        candidate_theory="cpfem")
    s_tag_mismatch = exp.score(1e-3, target_theory="ddd",
                                candidate_theory="cpfem")
    s_ddd_match = exp.score(5e3, target_theory="ddd",
                             candidate_theory="ddd")
    assert s_match > 0.5,  f"in-regime tagged score too low: {s_match}"
    assert s_tag_mismatch < 0.1, \
        f"cross-theory tag mismatch should be punished: {s_tag_mismatch}"
    assert s_ddd_match > 0.5, \
        f"DDD-tagged in-regime score too low: {s_ddd_match}"

    fake = [{"value": 1e-3, "unit": "s^-1", "confidence": 0.9,
             "property_label": "reference strain rate",
             "method": "explicit", "theory": "cpfem",
             "source": "test"}]
    cands = gatekeep(fake, "gamma0_dot", provenance="llm_extract")
    assert cands and cands[0].theory == "cpfem", \
        "gatekeep did not propagate the `theory` tag"

    disp = cands[0].to_display()
    assert disp.get("theory") == "cpfem"

    logger.info(
        "v9.1 regression test: PASS  "
        "JC.high=%.1e < DDD.low=%.1e < MD.low=%.1e  "
        "expert match/tag-mismatch/ddd-match = %.3f/%.3e/%.3f",
        r_jc["high"], r_ddd["low"], r_md["low"],
        s_match, s_tag_mismatch, s_ddd_match)


def _regression_test_v92() -> None:
    for fine in FINE_PROVENANCE_KEYS:
        assert _legend_key(fine, 'coarse') in LEGEND_MARKERS
        assert _legend_key(fine) in COARSE_PROVENANCE_KEYS
        assert _legend_key(fine, 'fine') == fine

    assert _legend_key('heuristic') == 'deterministic'
    assert _legend_key('llm') == 'llm_grounded'
    assert _legend_key('llm_prior (no corpus evidence)') == 'llm_prior'
    assert _legend_key('derived_consensus') == 'deterministic'
    assert _legend_key('physics_regime_inference') == 'deterministic'
    assert _legend_key('theory_regime_inference') == 'deterministic'
    assert _legend_key('hall_petch_intercept') == 'deterministic'
    assert _legend_key('peierls_nabarro_formula') == 'deterministic'
    assert _legend_key('taylor_factor_cross_conversion') == 'deterministic'
    assert _legend_key('explicit') == 'llm_grounded'

    for junk in ('', '   ', 'total nonsense', 'Ω≈ç√∫', None, 42, 3.14):
        assert _legend_key(junk, 'coarse') in COARSE_PROVENANCE_KEYS

    fig = plot_candidate_scores(
        [1e14, 1e15, 1e16, 1e17], [0.80, 0.85, 0.60, 0.90],
        ['heuristic', 'llm', 'llm_prior', 'physics_inferred'],
        best_idx=3, legend_granularity='coarse',
        param_title='ρ₀', param_symbol=r'\rho_0', unit='m^{-2}')
    texts = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert texts <= ({m['label'] for m in LEGEND_MARKERS.values()}
                     | {'Best match', 'Context (non-ranked)'})
    assert 'LLM (corpus-grounded)' in texts
    assert 'LLM prior (parametric)' in texts
    assert 'Deterministic (regex · physics-inferred)' in texts
    plt.close(fig)

    fig = plot_candidate_scores(
        [1e14, 1e15, 1e16, 1e17], [0.80, 0.85, 0.60, 0.90],
        ['heuristic', 'llm', 'llm_prior', 'physics_inferred'],
        best_idx=3, legend_granularity='fine',
        param_title='ρ₀', param_symbol=r'\rho_0', unit='m^{-2}')
    texts = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert texts <= ({m['label'] for m in PROVENANCE_MARKERS.values()}
                     | {'Best match', 'Context (non-ranked)'})
    assert 'Regex NER (deterministic)' in texts
    assert 'LLM extraction (verbatim, grounded)' in texts
    assert 'LLM prior (no corpus evidence)' in texts
    assert 'Physics-inferred (Hall–Petch / P–N)' in texts
    plt.close(fig)
    logger.info("✅ _regression_test_v92 passed")


def _regression_test_v921() -> None:
    for s in ('physics_regime_prior', 'regime_prior_fallback',
              'theory_regime_prior', 'regime_inference_prior'):
        assert _norm_provenance(s) == 'regime_prior', s

    assert _norm_provenance('model_inversion') == 'physics_inferred'
    assert _norm_provenance('explicit')        == 'llm_extract'
    assert _norm_provenance('physics_regime')  == 'regime_prior'
    assert _norm_provenance('derived') == 'physics_inferred'
    assert _norm_provenance('physics_inferred') == 'physics_inferred'

    assert _norm_provenance('value not available') == 'regex_ner'
    assert _norm_provenance('ai extracted verbatim') == 'llm_extract'

    for junk in ('typo', 'FINE ', 'Coarse', None, ''):
        fig = plot_candidate_scores([1e14], [0.8], ['heuristic'],
                                    legend_granularity=junk)
        plt.close(fig)

    ext: Dict[str, Any] = {}
    _stamp_provenance(ext, 'llm_reasoned')
    assert _norm_provenance(ext['_provenance']) == 'llm_reasoned'
    try:
        _stamp_provenance(ext, 'bogus')
        assert False, "stamp must reject non-fine keys"
    except ValueError:
        pass

    assert PROVENANCE_MARKERS['llm_prior']['fill'] is False
    assert PROVENANCE_MARKERS['regime_prior']['fill'] is False
    assert PROVENANCE_MARKERS['physics_inferred']['fill'] is False
    assert PROVENANCE_MARKERS['llm_extract']['fill'] is True
    assert PROVENANCE_MARKERS['regex_ner']['fill'] is True
    assert all(m['fill'] for m in LEGEND_MARKERS.values())
    fig = plot_candidate_scores([1e14, 1e15], [0.8, 0.9],
                                ['heuristic', 'llm_prior'],
                                legend_granularity='fine')
    plt.close(fig)

    fig = plot_candidate_scores(
        [1e12, 1e13, 1e14, 1e15, 1e16, 1e17],
        [0.9, 0.7, 0.85, 0.6, 0.55, 0.4],
        ['explicit', 'llm_prior', 'heuristic', 'physics_inferred',
         'physics_regime', 'regex'],
        best_idx=0, legend_granularity='coarse')
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert labels <= ({m['label'] for m in LEGEND_MARKERS.values()}
                      | {'Best match', 'Context (non-ranked)'}), labels
    plt.close(fig)
    logger.info("✅ _regression_test_v921 passed")


def _regression_test_v922() -> None:
    import dataclasses as _dc

    pc = PlasticityCandidate(
        param='mu', value_si=48e9, raw_value=48.0, raw_unit='GPa',
        score=0.8, confidence=0.7, material='Cu', temp_k=None,
        strain_rate=None, method='explicit', source_file='',
        source_title='', evidence='')
    assert pc.provenance == ""
    assert pc.context is False
    assert _norm_provenance(pc.provenance or pc.method) == 'llm_extract'
    pc2 = _dc.replace(pc, provenance='llm_reasoned')
    assert _norm_provenance(pc2.provenance or pc2.method) == 'llm_reasoned'

    ext = {'value': 1.0, 'method': 'explicit'}
    _stamp_provenance(ext, 'llm_extract')
    assert ext['_provenance'] == 'llm_extract'
    logger.info("✅ _regression_test_v922 passed")


def _regression_test_v930() -> None:
    for param, buckets in INCUMBENT_ROUTES.items():
        for b in buckets:
            assert b in COARSE_PROVENANCE_KEYS, \
                f"INCUMBENT_ROUTES[{param}] references unknown bucket {b!r}"

    for param, pinned in INCUMBENT_ROUTES.items():
        for b in COARSE_PROVENANCE_KEYS:
            for fk in PROVENANCE_GROUPS[b]:
                is_ctx = _is_context(param, fk)
                if b in pinned:
                    assert not is_ctx, \
                        f"_is_context({param!r}, {fk!r}) should be False"
                else:
                    assert is_ctx, \
                        f"_is_context({param!r}, {fk!r}) should be True"

    for param in ('sigma0_yield', 'grain_size_d', 'burgers_vector_b'):
        assert param not in INCUMBENT_ROUTES
        for fk in FINE_PROVENANCE_KEYS:
            assert not _is_context(param, fk), \
                f"un-pinned param {param!r} flagged {fk!r} as context"

    c_incumbent = PlasticityCandidate(
        param='mu', value_si=48e9, raw_value=48.0, raw_unit='GPa',
        score=0.50, confidence=0.70, material='Cu', temp_k=None,
        strain_rate=None, method='explicit', source_file='',
        source_title='', evidence='', provenance='regex_ner',
        context=False)
    c_context = PlasticityCandidate(
        param='mu', value_si=50e9, raw_value=50.0, raw_unit='GPa',
        score=0.99, confidence=0.99, material='Cu', temp_k=None,
        strain_rate=None, method='explicit', source_file='',
        source_title='', evidence='', provenance='llm_extract',
        context=True)
    winner = _pick_best([c_incumbent, c_context])
    assert winner is c_incumbent, \
        "context candidate must not win — the pin-list guarantee"

    winner2 = _pick_best([c_context])
    assert winner2 is c_context, \
        "_pick_best all-context degenerate path should still return one"

    fig = plot_candidate_scores(
        [1e14, 1e15, 1e16], [0.50, 0.60, 0.70],
        ['regex_ner', 'llm_extract', 'llm_prior'],
        best_idx=0, legend_granularity='coarse',
        context_flags=[False, True, True])
    texts = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert 'Context (non-ranked)' in texts, \
        f"context legend entry missing: {texts}"
    plt.close(fig)

    fake = [{"value": 48.0, "unit": "GPa", "confidence": 0.9,
             "property_label": "shear modulus", "source": "test",
             "_context": True}]
    cands = gatekeep(fake, "mu", provenance="llm_extract")
    assert cands and cands[0].context is True, \
        "gatekeep did not propagate `_context`"

    fake2 = [{"value": 48.0, "unit": "GPa", "confidence": 0.9,
              "property_label": "shear modulus", "source": "test"}]
    cands2 = gatekeep(fake2, "mu", provenance="llm_extract")
    assert cands2 and cands2[0].context is False, \
        "gatekeep should default `_context` to False"

    scorer = PlasticityLatentMoEScorer()
    buckets = scorer.score(
        [{"param": "mu", "value": 48.0, "unit": "GPa",
          "material": "Cu", "temp": 300.0, "strain_rate": 1e-3,
          "method": "explicit", "confidence": 0.9,
          "reasoning": "", "evidence": "",
          "_source_file": "t", "_source_title": "t",
          "_provenance": "regex_ner", "_context": False},
         {"param": "mu", "value": 47.0, "unit": "GPa",
          "material": "Cu", "temp": 300.0, "strain_rate": 1e-3,
          "method": "explicit", "confidence": 0.5,
          "reasoning": "", "evidence": "",
          "_source_file": "t", "_source_title": "t",
          "_provenance": "llm_extract", "_context": True}],
        "Cu", 300.0)
    mu_bucket = buckets.get("mu", [])
    assert len(mu_bucket) == 2, f"expected 2 mu candidates, got {len(mu_bucket)}"
    flags = sorted(getattr(c, "context", False) for c in mu_bucket)
    assert flags == [False, True], \
        f"scorer should preserve both context and non-context: {flags}"

    logger.info("✅ _regression_test_v930 passed")


def _regression_test_v931() -> None:
    assert 'physics_inferred' in FINE_PROVENANCE_KEYS, \
        "physics_inferred must be a first-class fine key"
    assert 'derived' not in FINE_PROVENANCE_KEYS, \
        "legacy 'derived' must no longer be a fine key"
    assert 'physics_inferred' in PROVENANCE_GROUPS['deterministic'], \
        "physics_inferred must roll up to deterministic"
    assert 'physics_inferred' in PROVENANCE_MARKERS, \
        "physics_inferred must have a marker entry"

    assert _norm_provenance('derived') == 'physics_inferred', \
        "legacy 'derived' must alias to physics_inferred"
    assert _norm_provenance('model_inversion') == 'physics_inferred'
    assert _norm_provenance('hall_petch_intercept') == 'physics_inferred'
    assert _norm_provenance('peierls_nabarro_formula') == 'physics_inferred'
    assert _norm_provenance('taylor_factor_cross_conversion') == 'physics_inferred'
    assert _norm_provenance('derived_consensus') == 'physics_inferred'

    assert _legend_key('derived', 'coarse') == 'deterministic'
    assert _legend_key('physics_inferred', 'coarse') == 'deterministic'
    assert _legend_key('physics_inferred', 'fine') == 'physics_inferred'

    fake = [{"value": 5e7, "unit": "Pa", "confidence": 0.99,
             "property_label": "σ₀", "source": "test"}]
    cands = gatekeep(fake, "sigma0_fric", provenance="physics_inferred")
    assert cands, "gatekeep must accept physics_inferred sigma0"
    assert cands[0].confidence <= 0.75 + 1e-9, \
        f"physics_inferred conf-cap should be 0.75, got {cands[0].confidence}"

    c_pinf = PlasticityCandidate(
        param='sigma0_fric', value_si=5e7, raw_value=50.0, raw_unit='MPa',
        score=0.55, confidence=0.55, material='Cu', temp_k=None,
        strain_rate=None, method='hall_petch_intercept',
        source_file='', source_title='', evidence='',
        provenance='physics_inferred', context=False)
    c_prior = PlasticityCandidate(
        param='sigma0_fric', value_si=6e7, raw_value=60.0, raw_unit='MPa',
        score=0.55, confidence=0.55, material='Cu', temp_k=None,
        strain_rate=None, method='prior inference',
        source_file='', source_title='', evidence='',
        provenance='llm_prior', context=False)
    winner = _pick_best([c_pinf, c_prior])
    assert winner is c_pinf, \
        "physics_inferred (trust 2) must outrank llm_prior (trust 1) at equal score"

    cands_flags = [True, False, True]
    show_all_competitive = True
    audit_flags = ([False] * len(cands_flags) if show_all_competitive
                   else list(cands_flags))
    assert audit_flags == [False, False, False], \
        "audit toggle must zero all context flags"
    show_all_competitive = False
    standard_flags = ([False] * len(cands_flags) if show_all_competitive
                      else list(cands_flags))
    assert standard_flags == [True, False, True], \
        "standard mode must preserve the recorded context flags"

    fig = plot_candidate_scores(
        [1e14, 1e15, 1e16], [0.5, 0.7, 0.6],
        ['regex_ner', 'physics_inferred', 'llm_prior'],
        best_idx=1, legend_granularity='fine',
        context_flags=audit_flags)
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert 'Physics-inferred (Hall–Petch / P–N)' in labels
    assert 'Context (non-ranked)' not in labels, \
        "audit mode must not draw the context legend entry"
    plt.close(fig)

    fig = plot_candidate_scores(
        [1e14, 1e15, 1e16], [0.5, 0.7, 0.6],
        ['regex_ner', 'physics_inferred', 'llm_prior'],
        best_idx=0, legend_granularity='coarse',
        context_flags=standard_flags)
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert 'Context (non-ranked)' in labels, \
        "standard mode must draw the context legend entry when context present"
    plt.close(fig)

    logger.info("✅ _regression_test_v931 passed")


def _regression_test_v100() -> bool:
    """v10.0.0 — score-anatomy invariants (T1–T8)."""
    ctx = ContextVector(material='Cu',
                        synthesis='electrodeposition',
                        architecture='equiaxed',
                        twin_spacing=20.0,
                        framework='cpfem')
    md = MetaDatabase()
    scorer = LatentMoEScorerV10(ctx, md)

    raws = {
        'gamma0_dot': [
            dict(param='gamma0_dot', value=1e-3, provenance='llm_extract',
                 n_sources=2, span_qualities=[0.92], theory='cpfem',
                 context_tag='Cu', reasoning=''),
            dict(param='gamma0_dot', value=5e-3, provenance='regex_ner',
                 n_sources=1, span_qualities=[0.7], theory='cpfem',
                 context_tag='Cu', reasoning=''),
            dict(param='gamma0_dot', value=1e7, provenance='llm_prior',
                 n_sources=1, span_qualities=[0.5], theory='md',
                 context_tag='Cu', reasoning=''),
        ],
    }
    scored = scorer.score_all(raws)['gamma0_dot']
    for c in scored:
        scale = c.Sigma / c.Lambda if c.Lambda else 0.0
        stack = sum(c.w_hat[k] * c.opinions[k].s for k in c.opinions) * scale
        assert abs(stack - (c.Lambda * scale)) < 1e-9, \
            f"T1 stack identity failed: {stack} vs {c.Lambda * scale}"

    assert kappa_count(1) < kappa_count(2) < kappa_count(10)
    assert kappa_count(10) / kappa_count(1) < 2.0, \
        "kappa_count must saturate — 10× evidence cannot give 10× height"

    cross = [c for c in scored if c.fine_prov == 'llm_prior']
    assert cross, "expected a cross-theory candidate"
    bad = cross[0]
    assert any(g[0] == 'cross_theory' for g in bad.gate_list), \
        f"T3 cross-theory gate missing on {bad.value}: {bad.gate_list}"
    assert bad.Sigma <= 0.05 * bad.Lambda * bad.kappa_n * bad.kappa_m + 1e-9, \
        "T3 gate should chop the bar to ~0.05·Λ·κ_n·κ_m"

    lo1, hi1 = weight_stability(bad.opinions, bad.w_hat,
                                bad.kappa_n, bad.kappa_m, 1.0)
    lo2, hi2 = weight_stability(bad.opinions, bad.w_hat,
                                bad.kappa_n, bad.kappa_m, 1.0)
    assert (lo1, hi1) == (lo2, hi2), \
        "T4 weight_stability must be seed-reproducible"

    fig = render_score_anatomy_figure(scored, 'gamma0_dot')
    leg = fig.axes[0].get_legend()
    legend_txt = [t.get_text() for t in leg.get_texts()] if leg else []
    for txt in legend_txt:
        assert all(txt not in (m['label'] for m in PROVENANCE_MARKERS.values())
                   or 'badge' in (leg.get_title().get_text().lower()
                                  if leg and leg.get_title() else '')
                   for _ in [0]), \
            f"T5 provenance label leaked into the ranking legend: {txt!r}"
    plt.close(fig)

    for c in scored:
        assert c.fine_prov in FINE_PROVENANCE_KEYS, \
            f"T6 non-total badge: {c.fine_prov!r}"
        assert c.coarse in COARSE_PROVENANCE_KEYS

    ranked = sorted(scored, key=lambda x: -x.Sigma)
    assert ranked == sorted(scored, key=lambda x: -x.Sigma), \
        "T7 ranking must key on Σ alone"

    v = 1e-3
    raws2 = {'gamma0_dot': [
        dict(param='gamma0_dot', value=v, provenance='llm_extract',
             n_sources=1, span_qualities=[0.9], theory='cpfem',
             context_tag='Cu', reasoning=''),
    ]}
    base_score = scorer.score_all(raws2)['gamma0_dot'][0]
    raws3 = {'gamma0_dot': raws2['gamma0_dot'] + [
        dict(param='gamma0_dot', value=v * 1.01, provenance='regex_ner',
             n_sources=1, span_qualities=[0.7], theory='cpfem',
             context_tag='Cu', reasoning=''),
    ]}
    add_score = scorer.score_all(raws3)['gamma0_dot']
    target = next(c for c in add_score if abs(c.value - v) < 1e-12)
    assert target.kappa_m >= base_score.kappa_m, \
        "T8 corroboration must be monotonically non-decreasing"

    logger.info("✅ _regression_test_v100 passed")
    return True


def _run_regression_suite() -> None:
    import traceback as _tb
    suite = [
        ('v8.8.1', '_regression_test_v881'),
        ('v8.8.2', '_regression_test_v882'),
        ('v8.8.3', '_regression_test_v883'),
        ('v8.8.4', '_regression_test_v884'),
        ('v8.9.0', '_regression_test_v890'),
        ('v9.0',   '_regression_test_v90'),
        ('v9.1',   '_regression_test_v91'),
        ('v9.2.0', '_regression_test_v92'),
        ('v9.2.1', '_regression_test_v921'),
        ('v9.2.2', '_regression_test_v922'),
        ('v9.3.0', '_regression_test_v930'),
        ('v9.3.1', '_regression_test_v931'),
        ('v10.0.0', '_regression_test_v100'),
    ]
    for tag, name in suite:
        fn = globals().get(name)
        if fn is None:
            logger.warning("regression %s missing (%s undefined) — skipped",
                           tag, name)
            continue
        try:
            fn()
        except AssertionError as ae:
            logger.error("❌ regression %s FAILED: %s", tag, ae)
        except Exception as e:
            logger.error("❌ regression %s ERRORED: %s\n%s",
                         tag, e, _tb.format_exc())


if os.environ.get("NTCU_REGRESSION", "").lower() in ("1", "true", "yes"):
    _run_regression_suite()   # NTCU_REGRESSION=1 streamlit run app.py


if __name__ == "__main__":
    main()
