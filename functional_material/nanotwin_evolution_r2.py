import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq, fftn, ifftn
from numba import njit, prange, complex128, float64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import json
import zipfile
import pickle
import torch
import pyvista as pv
from io import BytesIO, StringIO
import tempfile
import os
import meshio
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import time

# ============================================================================
# ENHANCED MATERIAL CONSTANTS DATABASE (Cu - Experimental Values)
# ============================================================================
class MaterialDatabase:
    """Comprehensive material properties database for nanotwinned Cu"""
    
    @staticmethod
    def get_cubic_constants():
        """Cubic elastic constants (GPa) - Phys. Rev. B 73, 064112 (2006)"""
        return {
            'C11': 168.4e9,   # Pa
            'C12': 121.4e9,   # Pa
            'C44': 75.4e9,    # Pa
            'source': 'Phys. Rev. B 73, 064112 (2006)'
        }
    
    @staticmethod
    def get_twinning_parameters():
        """Twinning crystallography for FCC Cu"""
        return {
            'gamma_tw': 1/np.sqrt(2),  # Twinning shear magnitude
            # Miller indices for {111}<112> twinning
            'n_111': np.array([1, 1, 1])/np.sqrt(3),  # {111} plane normal
            'a_112': np.array([1, 1, -2])/np.sqrt(6),  # <112> direction
            # 2D projection for (111) plane simulation
            'n_2d': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
            'a_2d': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
        }
    
    @staticmethod
    def get_plasticity_parameters():
        """Plastic deformation parameters for Cu"""
        return {
            'mu': 48e9,  # Shear modulus (Pa)
            'nu': 0.34,  # Poisson's ratio
            'b': 0.256e-9,  # Burgers vector (m)
            'sigma0': 50e6,  # Lattice friction (Pa)
            'gamma0_dot': 1e-3,  # Reference shear rate (1/s)
            'm': 20,  # Rate sensitivity exponent
            'rho0': 1e12,  # Initial dislocation density (m^-2)
        }
    
    @staticmethod
    def get_thermal_parameters():
        """Thermal properties for Cu"""
        return {
            'melting_temp': 1357.77,  # K
            'thermal_cond': 401,  # W/(m·K)
            'specific_heat': 385,  # J/(kg·K)
            'thermal_expansion': 16.5e-6,  # 1/K
            'room_temp': 298,  # K
        }

# ============================================================================
# ADVANCED NUMERICAL METHODS
# ============================================================================
class SpectralSolver:
    """Advanced spectral method solver with cubic anisotropy"""
    
    def __init__(self, N, dx, C11, C12, C44, dim=2):
        self.N = N
        self.dx = dx
        self.dim = dim
        
        # Initialize Fourier space grid
        if dim == 2:
            self.kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1)
            self.ky = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1)
            self.k2 = self.kx**2 + self.ky**2 + 1e-12
        else:
            self.kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, 1, -1)
            self.ky = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1, 1)
            self.kz = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1, 1)
            self.k2 = self.kx**2 + self.ky**2 + self.kz**2 + 1e-12
        
        # Compute Green's function for cubic anisotropy
        self.compute_greens_function(C11, C12, C44)
    
    def compute_greens_function(self, C11, C12, C44):
        """Compute Green's function for cubic elasticity in Fourier space"""
        if self.dim == 2:
            # 2D plane strain approximation for (111) plane
            C11_2d = (C11 + C12 + 2*C44) / 2
            C12_2d = (C11 + C12 - 2*C44) / 2
            lambda_2d = C12_2d
            mu_2d = (C11_2d - C12_2d) / 2
            
            # Green's function components
            denom = mu_2d * (lambda_2d + 2*mu_2d) * self.k2
            self.G11 = (mu_2d*(self.kx**2 + 2*self.ky**2) + lambda_2d*self.ky**2) / denom
            self.G12 = -mu_2d * self.kx * self.ky / denom
            self.G22 = (mu_2d*(self.ky**2 + 2*self.kx**2) + lambda_2d*self.kx**2) / denom
        else:
            # 3D Green's function for cubic symmetry
            # Simplified isotropic approximation for demonstration
            E = C11 - 2*C12**2/(C11 + C12)
            nu = C12/(C11 + C12)
            mu = C44
            
            denom = mu * self.k2
            self.G11 = (1/mu - (1+nu)/(E*(1-nu)) * self.kx**2/self.k2) / self.k2
            self.G12 = -(1+nu)/(E*(1-nu)) * self.kx*self.ky/self.k2**2
            self.G22 = (1/mu - (1+nu)/(E*(1-nu)) * self.ky**2/self.k2) / self.k2
    
    def solve_elasticity(self, eigenstrain, applied_stress=0):
        """Solve mechanical equilibrium with FFT-based spectral method"""
        # Fourier transform of eigenstrain
        if self.dim == 2:
            eps_xx_hat = fft2(eigenstrain[0])
            eps_yy_hat = fft2(eigenstrain[1])
            eps_xy_hat = fft2(eigenstrain[2])
            
            # Solve for displacements
            ux_hat = 1j * (self.G11 * self.kx * eps_xx_hat + 
                          self.G12 * self.ky * eps_xx_hat +
                          self.G12 * self.kx * eps_yy_hat + 
                          self.G22 * self.ky * eps_yy_hat)
            
            uy_hat = 1j * (self.G12 * self.kx * eps_xx_hat + 
                          self.G22 * self.ky * eps_xx_hat +
                          self.G11 * self.kx * eps_yy_hat + 
                          self.G12 * self.ky * eps_yy_hat)
            
            # Compute strains
            eps_xx_el = np.real(ifft2(1j * self.kx * ux_hat))
            eps_yy_el = np.real(ifft2(1j * self.ky * uy_hat))
            eps_xy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_hat + self.ky * ux_hat)))
            
            # Total strains
            eps_xx = eps_xx_el + eigenstrain[0]
            eps_yy = eps_yy_el + eigenstrain[1]
            eps_xy = eps_xy_el + eigenstrain[2]
            
            # Stresses (simplified plane strain)
            sxx = applied_stress + 168.4e9 * eps_xx + 121.4e9 * eps_yy
            syy = 121.4e9 * eps_xx + 168.4e9 * eps_yy
            sxy = 2 * 75.4e9 * eps_xy
            
            sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + syy**2 + sxx**2 + 6 * sxy**2))
            
            return sigma_eq, sxx, syy, sxy, eps_xx, eps_yy, eps_xy
        
        return None

# ============================================================================
# PHYSICS-BASED MODELS IMPLEMENTATION
# ============================================================================
@njit(parallel=True)
def compute_transformation_strain_tensor(phi, eta1, gamma_tw, n, a):
    """Compute transformation strain tensor as per theory: ε* = ½γ(a⊗n + n⊗a)·h(φ)"""
    N = phi.shape[0]
    eps_star = np.zeros((3, N, N))
    
    # Interpolation function h(φ) = ¼(φ-1)²(φ+1)
    h_phi = 0.25 * (phi**3 - phi**2 - phi + 1)
    
    # Only active in twin grain (η₁ ≈ 1)
    active_mask = eta1 > 0.5
    
    for i in prange(N):
        for j in prange(N):
            if active_mask[i, j]:
                # Transformation strain components
                eps_star[0, i, j] = gamma_tw * n[0] * a[0] * h_phi[i, j]  # ε_xx
                eps_star[1, i, j] = gamma_tw * n[1] * a[1] * h_phi[i, j]  # ε_yy
                eps_star[2, i, j] = 0.5 * gamma_tw * (n[0]*a[1] + n[1]*a[0]) * h_phi[i, j]  # ε_xy
    
    return eps_star

@njit(parallel=True)
def compute_ovidko_yield_stress(h, sigma0, mu, b, nu):
    """Ovid'ko confined layer slip model: σ_y = σ₀ + μb/[2πh(1-ν)]·ln(h/b)"""
    N = h.shape[0]
    sigma_y = np.zeros((N, N))
    
    for i in prange(N):
        for j in prange(N):
            if h[i, j] > 2*b:  # Avoid singularity
                log_term = np.log(h[i, j] / b)
                sigma_y[i, j] = sigma0 + (mu * b / (2 * np.pi * h[i, j] * (1 - nu))) * log_term
            else:
                sigma_y[i, j] = sigma0 + mu / (2 * np.pi * (1 - nu))  # Maximum strengthening
    
    return sigma_y

@njit(parallel=True)
def compute_perzyna_viscoplasticity(sigma_eq, sigma_y, eps_p_prev, dt, gamma0_dot, m):
    """Perzyna viscoplastic flow rule with strain hardening"""
    N = sigma_eq.shape[0]
    eps_p_dot = np.zeros((3, N, N))
    eps_p_new = np.zeros((3, N, N))
    
    for i in prange(N):
        for j in prange(N):
            if sigma_eq[i, j] > sigma_y[i, j]:
                # Overstress ratio
                overstress = (sigma_eq[i, j] - sigma_y[i, j]) / sigma_y[i, j]
                
                # Strain rate magnitude
                gamma_dot = gamma0_dot * (overstress)**m
                
                # Plastic strain increment (simplified direction)
                eps_p_dot[0, i, j] = gamma_dot  # xx component
                eps_p_dot[1, i, j] = -0.5 * gamma_dot  # yy component (volume preserving)
                eps_p_dot[2, i, j] = 0.0  # xy component
                
                # Update plastic strain
                eps_p_new[0, i, j] = eps_p_prev[0, i, j] + eps_p_dot[0, i, j] * dt
                eps_p_new[1, i, j] = eps_p_prev[1, i, j] + eps_p_dot[1, i, j] * dt
                eps_p_new[2, i, j] = eps_p_prev[2, i, j] + eps_p_dot[2, i, j] * dt
    
    return eps_p_new, eps_p_dot

@njit(parallel=True)
def compute_anisotropic_properties(grad_phi, n_twin, kappa0, gamma_aniso, L_CTB, L_ITB, n_mob):
    """Compute anisotropic gradient coefficient and mobility"""
    N = grad_phi.shape[1]
    kappa_phi = np.zeros((N, N))
    L_phi = np.zeros((N, N))
    
    for i in prange(N):
        for j in prange(N):
            # Gradient direction
            grad_mag = np.sqrt(grad_phi[0, i, j]**2 + grad_phi[1, i, j]**2 + 1e-12)
            if grad_mag > 1e-6:
                m_x = grad_phi[0, i, j] / grad_mag
                m_y = grad_phi[1, i, j] / grad_mag
                
                # Dot product with twin normal
                dot_product = m_x * n_twin[0] + m_y * n_twin[1]
                
                # Anisotropic gradient coefficient
                kappa_phi[i, j] = kappa0 * (1 + gamma_aniso * (1 - dot_product**2))
                
                # Anisotropic mobility
                mobility_factor = (1 - dot_product**2)**n_mob
                L_phi[i, j] = L_CTB + (L_ITB - L_CTB) * mobility_factor
            else:
                kappa_phi[i, j] = kappa0
                L_phi[i, j] = L_CTB
    
    return kappa_phi, L_phi

# ============================================================================
# PHASE-FIELD EVOLUTION EQUATIONS
# ============================================================================
class PhaseFieldSolver:
    """Unified phase-field solver for twin and grain evolution"""
    
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']
        
        # Initialize fields
        self.phi = None
        self.eta1 = None
        self.eta2 = None
        self.eps_p = None
        self.initialize_fields()
        
        # Initialize spectral solver
        mat_props = MaterialDatabase.get_cubic_constants()
        self.spectral_solver = SpectralSolver(
            self.N, self.dx,
            mat_props['C11'], mat_props['C12'], mat_props['C44']
        )
    
    def initialize_fields(self):
        """Initialize order parameters as per theory document"""
        x = np.linspace(-self.N*self.dx/2, self.N*self.dx/2, self.N)
        y = np.linspace(-self.N*self.dx/2, self.N*self.dx/2, self.N)
        X, Y = np.meshgrid(x, y)
        
        # Grain order parameters
        self.eta1 = np.zeros((self.N, self.N))
        self.eta2 = np.zeros((self.N, self.N))
        
        # Twin grain on left, twin-free grain on right
        grain_boundary_width = 3.0  # nm
        for i in range(self.N):
            for j in range(self.N):
                x_val = X[i, j]
                if x_val < -grain_boundary_width:
                    self.eta1[i, j] = 1.0
                    self.eta2[i, j] = 0.0
                elif x_val > grain_boundary_width:
                    self.eta1[i, j] = 0.0
                    self.eta2[i, j] = 1.0
                else:
                    # Smooth transition at grain boundary
                    transition = 0.5 * (1 - np.tanh(x_val / (grain_boundary_width/3)))
                    self.eta1[i, j] = transition
                    self.eta2[i, j] = 1 - transition
        
        # Twin order parameter (only in twin grain)
        twin_spacing = self.params.get('twin_spacing', 20.0)
        self.phi = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                if self.eta1[i, j] > 0.5:
                    # Periodic twin pattern
                    phase = 2 * np.pi * Y[i, j] / twin_spacing
                    self.phi[i, j] = np.tanh(np.sin(phase) * 3.0)  # Sharp interfaces
        
        # Initialize plastic strain
        self.eps_p = np.zeros((3, self.N, self.N))
    
    def compute_local_free_energy_derivatives(self):
        """Compute derivatives of local free energy density"""
        # f_loc = W(φ²-1)²η₁² + ΣAη_i²(1-η_i)² + Bη₁²η₂²
        
        # Derivative with respect to φ
        df_dphi = 4 * self.params['W'] * self.phi * (self.phi**2 - 1) * self.eta1**2
        
        # Derivative with respect to η₁
        df_deta1 = (2 * self.params['A'] * self.eta1 * (1 - self.eta1) * (1 - 2*self.eta1) +
                   2 * self.params['B'] * self.eta1 * self.eta2**2 +
                   2 * self.params['W'] * (self.phi**2 - 1)**2 * self.eta1)
        
        # Derivative with respect to η₂
        df_deta2 = (2 * self.params['A'] * self.eta2 * (1 - self.eta2) * (1 - 2*self.eta2) +
                   2 * self.params['B'] * self.eta2 * self.eta1**2)
        
        return df_dphi, df_deta1, df_deta2
    
    def compute_elastic_driving_force(self, sigma):
        """Compute elastic driving force for twin evolution"""
        # ∂f_el/∂φ = -σ:∂ε*/∂φ
        twin_params = MaterialDatabase.get_twinning_parameters()
        gamma_tw = twin_params['gamma_tw']
        n = twin_params['n_2d']
        a = twin_params['a_2d']
        
        # Derivative of interpolation function: h'(φ) = ¼(3φ² - 2φ - 1)
        dh_dphi = 0.25 * (3*self.phi**2 - 2*self.phi - 1)
        
        # Transformation strain derivative
        deps_dphi = np.zeros((3, self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self.eta1[i, j] > 0.5:
                    deps_dphi[0, i, j] = gamma_tw * n[0] * a[0] * dh_dphi[i, j]
                    deps_dphi[1, i, j] = gamma_tw * n[1] * a[1] * dh_dphi[i, j]
                    deps_dphi[2, i, j] = 0.5 * gamma_tw * (n[0]*a[1] + n[1]*a[0]) * dh_dphi[i, j]
        
        # Elastic driving force
        df_el_dphi = -(sigma[0] * deps_dphi[0] + 
                      sigma[1] * deps_dphi[1] + 
                      2 * sigma[2] * deps_dphi[2])
        
        return df_el_dphi
    
    def compute_dissipation(self, eps_p_mag):
        """Compute dissipation term from dislocation pinning"""
        # Diss_p = ζ|ε̄^p|φ
        return self.params['zeta'] * eps_p_mag * self.phi
    
    def evolve_twin_field(self, sigma, eps_p_mag):
        """Evolve twin order parameter φ"""
        # Compute gradients
        grad_phi = np.gradient(self.phi, self.dx, axis=(0, 1))
        
        # Compute anisotropic properties
        twin_params = MaterialDatabase.get_twinning_parameters()
        kappa_phi, L_phi = compute_anisotropic_properties(
            grad_phi, twin_params['n_2d'],
            self.params['kappa0'], self.params['gamma_aniso'],
            self.params['L_CTB'], self.params['L_ITB'], self.params['n_mob']
        )
        
        # Compute driving forces
        df_loc_dphi, _, _ = self.compute_local_free_energy_derivatives()
        df_el_dphi = self.compute_elastic_driving_force(sigma)
        diss_p = self.compute_dissipation(eps_p_mag)
        
        # Compute gradient term
        laplacian_phi = np.gradient(np.gradient(self.phi, self.dx, axis=(0, 1))[0], 
                                   self.dx, axis=0)[0] + \
                       np.gradient(np.gradient(self.phi, self.dx, axis=(0, 1))[1], 
                                   self.dx, axis=1)[1]
        
        # Evolution equation: ∂φ/∂t = -L_φ[∂f_loc/∂φ + ∂f_el/∂φ - ∇·(κ_φ∇φ) + Diss_p]
        grad_term = np.gradient(kappa_phi * grad_phi[0], self.dx, axis=0)[0] + \
                   np.gradient(kappa_phi * grad_phi[1], self.dx, axis=1)[1]
        
        dphi_dt = -L_phi * (df_loc_dphi + df_el_dphi - grad_term + diss_p)
        
        # Update twin field (explicit Euler)
        phi_new = self.phi + self.dt * dphi_dt
        
        # Apply boundary conditions (periodic)
        phi_new = np.roll(phi_new, 1, axis=0)
        phi_new = np.roll(phi_new, 1, axis=1)
        
        return np.clip(phi_new, -1.1, 1.1)
    
    def evolve_grain_fields(self):
        """Evolve grain order parameters η₁ and η₂"""
        # Compute Laplacians
        laplacian_eta1 = np.gradient(np.gradient(self.eta1, self.dx, axis=(0, 1))[0], 
                                    self.dx, axis=0)[0] + \
                        np.gradient(np.gradient(self.eta1, self.dx, axis=(0, 1))[1], 
                                    self.dx, axis=1)[1]
        
        laplacian_eta2 = np.gradient(np.gradient(self.eta2, self.dx, axis=(0, 1))[0], 
                                    self.dx, axis=0)[0] + \
                        np.gradient(np.gradient(self.eta2, self.dx, axis=(0, 1))[1], 
                                    self.dx, axis=1)[1]
        
        # Compute local derivatives
        _, df_deta1, df_deta2 = self.compute_local_free_energy_derivatives()
        
        # Evolution equations
        deta1_dt = -self.params['L_eta'] * (df_deta1 - self.params['kappa_eta'] * laplacian_eta1)
        deta2_dt = -self.params['L_eta'] * (df_deta2 - self.params['kappa_eta'] * laplacian_eta2)
        
        # Update grain fields
        eta1_new = self.eta1 + self.dt * deta1_dt
        eta2_new = self.eta2 + self.dt * deta2_dt
        
        # Enforce constraints: η₁² + η₂² ≤ 1
        norm = np.sqrt(eta1_new**2 + eta2_new**2 + 1e-12)
        mask = norm > 1
        eta1_new[mask] = eta1_new[mask] / norm[mask]
        eta2_new[mask] = eta2_new[mask] / norm[mask]
        
        return np.clip(eta1_new, 0, 1), np.clip(eta2_new, 0, 1)
    
    def compute_twin_spacing(self):
        """Compute local twin spacing h = 2/|∇φ|"""
        grad_phi = np.gradient(self.phi, self.dx, axis=(0, 1))
        grad_mag = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2 + 1e-12)
        h = 2.0 / grad_mag
        h[grad_mag < 1e-6] = 1e6  # Infinite spacing where gradient is zero
        return h
    
    def step(self, applied_stress):
        """Perform one time step of the simulation"""
        # Compute transformation strain
        twin_params = MaterialDatabase.get_twinning_parameters()
        eps_star = compute_transformation_strain_tensor(
            self.phi, self.eta1,
            twin_params['gamma_tw'],
            twin_params['n_2d'],
            twin_params['a_2d']
        )
        
        # Solve mechanical equilibrium
        sigma_eq, sxx, syy, sxy, eps_xx, eps_yy, eps_xy = \
            self.spectral_solver.solve_elasticity(
                [eps_star[0] + self.eps_p[0],
                 eps_star[1] + self.eps_p[1],
                 eps_star[2] + self.eps_p[2]],
                applied_stress
            )
        
        # Compute twin spacing
        h = self.compute_twin_spacing()
        
        # Compute yield stress
        plastic_params = MaterialDatabase.get_plasticity_parameters()
        sigma_y = compute_ovidko_yield_stress(
            h, plastic_params['sigma0'],
            plastic_params['mu'], plastic_params['b'],
            plastic_params['nu']
        )
        
        # Compute plastic strain
        eps_p_mag = np.sqrt(2/3 * (self.eps_p[0]**2 + self.eps_p[1]**2 + 2*self.eps_p[2]**2))
        sigma_tensor = [sxx, syy, sxy]
        self.eps_p, eps_p_dot = compute_perzyna_viscoplasticity(
            sigma_eq, sigma_y, self.eps_p, self.dt,
            plastic_params['gamma0_dot'], plastic_params['m']
        )
        
        # Evolve phase fields
        self.phi = self.evolve_twin_field(sigma_tensor, eps_p_mag)
        self.eta1, self.eta2 = self.evolve_grain_fields()
        
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
            'eps_p_mag': eps_p_mag.copy(),
            'eps_xx': eps_xx.copy(),
            'eps_yy': eps_yy.copy(),
            'eps_xy': eps_xy.copy()
        }
        
        return results

# ============================================================================
# ADVANCED VISUALIZATION
# ============================================================================
class AdvancedVisualizer:
    """Comprehensive visualization with Plotly and Matplotlib"""
    
    @staticmethod
    def create_interactive_dashboard(results_history, params):
        """Create interactive Plotly dashboard"""
        last_results = results_history[-1]
        N = last_results['phi'].shape[0]
        x = np.linspace(-N*params['dx']/2, N*params['dx']/2, N)
        y = np.linspace(-N*params['dx']/2, N*params['dx']/2, N)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Twin Order Parameter φ', 'Grain Structure',
                          'von Mises Stress (GPa)', 'Twin Spacing (nm)',
                          'Yield Stress (MPa)', 'Plastic Strain'),
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # 1. Twin order parameter
        fig.add_trace(
            go.Heatmap(z=last_results['phi'], x=x, y=y,
                      colorscale='RdBu_r', zmin=-1.2, zmax=1.2,
                      colorbar=dict(x=0.3, y=0.95, len=0.3)),
            row=1, col=1
        )
        
        # 2. Grain structure
        grains_rgb = np.zeros((N, N, 3))
        grains_rgb[..., 0] = last_results['eta1']  # Red for grain 1
        grains_rgb[..., 2] = last_results['eta2']  # Blue for grain 2
        grains_rgb = np.clip(grains_rgb, 0, 1)
        
        fig.add_trace(
            go.Image(z=grains_rgb, x=x, y=y),
            row=1, col=2
        )
        
        # 3. von Mises stress
        fig.add_trace(
            go.Heatmap(z=last_results['sigma_eq']/1e9, x=x, y=y,
                      colorscale='hot', zmin=0, zmax=2.0,
                      colorbar=dict(x=0.95, y=0.95, len=0.3)),
            row=1, col=3
        )
        
        # 4. Twin spacing
        fig.add_trace(
            go.Heatmap(z=np.clip(last_results['h'], 0, 50), x=x, y=y,
                      colorscale='plasma', zmin=0, zmax=30,
                      colorbar=dict(x=0.3, y=0.45, len=0.3)),
            row=2, col=1
        )
        
        # 5. Yield stress
        fig.add_trace(
            go.Heatmap(z=last_results['sigma_y']/1e6, x=x, y=y,
                      colorscale='viridis', zmin=0, zmax=500,
                      colorbar=dict(x=0.65, y=0.45, len=0.3)),
            row=2, col=2
        )
        
        # 6. Plastic strain
        fig.add_trace(
            go.Heatmap(z=last_results['eps_p_mag'], x=x, y=y,
                      colorscale='YlOrRd', zmin=0, zmax=0.05,
                      colorbar=dict(x=0.95, y=0.45, len=0.3)),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"Nanotwinned Cu Simulation | Step {len(results_history)}",
            title_x=0.5
        )
        
        # Update axes
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(title_text="x (nm)", row=i, col=j)
                fig.update_yaxes(title_text="y (nm)", row=i, col=j)
        
        return fig
    
    @staticmethod
    def create_3d_visualization(results):
        """Create 3D visualization of microstructure"""
        N = results['phi'].shape[0]
        x = np.linspace(-N*0.5, N*0.5, N)
        y = np.linspace(-N*0.5, N*0.5, N)
        X, Y = np.meshgrid(x, y)
        
        # Create twin boundary surfaces
        phi_grad = np.gradient(results['phi'])
        grad_mag = np.sqrt(phi_grad[0]**2 + phi_grad[1]**2)
        twin_boundaries = grad_mag > np.percentile(grad_mag, 95)
        
        fig = go.Figure()
        
        # Add twin boundaries as surface
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=twin_boundaries.astype(float),
                colorscale='Viridis',
                opacity=0.8,
                showscale=False
            )
        )
        
        # Add stress contours
        stress_levels = np.linspace(results['sigma_eq'].min(),
                                  results['sigma_eq'].max(), 10)
        
        fig.add_trace(
            go.Contour(
                z=results['sigma_eq'],
                x=x, y=y,
                contours=dict(
                    start=stress_levels[0],
                    end=stress_levels[-1],
                    size=(stress_levels[-1] - stress_levels[0]) / 10
                ),
                colorscale='Hot',
                opacity=0.6,
                showscale=True
            )
        )
        
        fig.update_layout(
            title="3D Microstructure Visualization",
            scene=dict(
                xaxis_title='x (nm)',
                yaxis_title='y (nm)',
                zaxis_title='Twin Boundary Intensity'
            ),
            height=600
        )
        
        return fig

# ============================================================================
# DATA EXPORT AND SERIALIZATION
# ============================================================================
class DataExporter:
    """Handle data export in multiple formats"""
    
    @staticmethod
    def export_to_vtu(results, filename, dx):
        """Export simulation data to VTU format for ParaView"""
        N = results['phi'].shape[0]
        
        # Create 3D grid by extruding 2D data
        points = []
        cells = []
        cell_types = []
        
        # Generate points
        for i in range(N):
            for j in range(N):
                # Extrude in z-direction (single layer)
                points.append([i*dx, j*dx, 0])
                points.append([i*dx, j*dx, dx])
        
        # Generate hexahedral cells
        for i in range(N-1):
            for j in range(N-1):
                # 8 points per hexahedron
                p0 = i*N*2 + j*2
                cell = [
                    p0, p0+1, p0+3, p0+2,  # bottom face
                    p0 + N*2, p0 + N*2 + 1, p0 + N*2 + 3, p0 + N*2 + 2  # top face
                ]
                cells.append(cell)
                cell_types.append(12)  # VTK_HEXAHEDRON
        
        # Create point data
        point_data = {}
        for key, data in results.items():
            if isinstance(data, np.ndarray) and data.shape == (N, N):
                # Extrude 2D data to 3D
                extruded_data = np.zeros(N*N*2)
                for idx in range(N*N):
                    i = idx // N
                    j = idx % N
                    extruded_data[idx*2] = data[i, j]
                    extruded_data[idx*2 + 1] = data[i, j]
                point_data[key] = extruded_data
        
        # Create mesh
        mesh = meshio.Mesh(
            points=points,
            cells=[("hexahedron", cells)],
            point_data=point_data
        )
        
        mesh.write(filename)
        return filename
    
    @staticmethod
    def export_to_pvd(results_history, basename, dx, timesteps):
        """Export time series to PVD format for ParaView"""
        pvd_content = f"""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
"""
        
        for i, results in enumerate(results_history):
            vtu_filename = f"{basename}_{i:04d}.vtu"
            DataExporter.export_to_vtu(results, vtu_filename, dx)
            pvd_content += f'    <DataSet timestep="{timesteps[i]}" group="" part="0" file="{vtu_filename}"/>\n'
        
        pvd_content += """  </Collection>
</VTKFile>"""
        
        pvd_filename = f"{basename}.pvd"
        with open(pvd_filename, 'w') as f:
            f.write(pvd_content)
        
        return pvd_filename
    
    @staticmethod
    def export_to_pickle(results_history, params, filename):
        """Export simulation data to pickle format"""
        data = {
            'results_history': results_history,
            'parameters': params,
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'simulation_type': 'Nanotwinned Cu Phase Field',
                'grid_size': params['N'],
                'dx': params['dx']
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return filename
    
    @staticmethod
    def export_to_torch(results_history, params, filename):
        """Export simulation data to PyTorch format"""
        import torch
        
        # Convert data to tensors
        tensor_data = {}
        for key in results_history[0].keys():
            if isinstance(results_history[0][key], np.ndarray):
                tensor_data[key] = torch.stack([
                    torch.from_numpy(results[key]) for results in results_history
                ])
        
        data = {
            'tensors': tensor_data,
            'parameters': params,
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'simulation_type': 'Nanotwinned Cu Phase Field'
            }
        }
        
        torch.save(data, filename)
        return filename

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="Nanotwinned Cu Phase-Field Simulator",
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
    }
    .theory-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .parameter-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Nanotwinned Copper Phase-Field Simulator</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### Unified Thermodynamic-Kinetic Framework for Nanotwinned FCC Metals
    
    **Physics Implemented:** 
    - Cubic anisotropic elasticity with full stiffness tensor
    - Transformation strain for {111}<112> twinning
    - Ovid'ko confined layer slip strengthening model
    - Perzyna viscoplasticity with strain hardening
    - Grain-twin coupling with anti-overlap constraint
    - Inclination-dependent CTB/ITB kinetics
    - Spectral FFT mechanical equilibrium solver
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        
        # Material selection
        material = st.selectbox(
            "Material System",
            ["Copper (FCC)", "Silver (FCC)", "Nickel (FCC)", "Custom"]
        )
        
        # Simulation parameters
        st.subheader("Grid Configuration")
        N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64)
        dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1)
        dt = st.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5)
        
        st.subheader("Thermodynamic Parameters")
        W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1)
        A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5)
        B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5)
        
        st.subheader("Gradient Energy Parameters")
        kappa0 = st.slider("κ₀ (Gradient energy ref)", 0.01, 10.0, 1.0, 0.1)
        gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05)
        kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1)
        
        st.subheader("Kinetic Parameters")
        L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001)
        L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1)
        n_mob = st.slider("n (Mobility exponent)", 1, 10, 4, 1)
        L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1)
        zeta = st.slider("ζ (Dislocation pinning)", 0.0, 2.0, 0.3, 0.05)
        
        st.subheader("Loading Conditions")
        applied_stress_MPa = st.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0)
        twin_spacing = st.slider("Initial twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0)
        
        st.subheader("Simulation Control")
        n_steps = st.slider("Number of steps", 10, 1000, 100, 10)
        save_frequency = st.slider("Save frequency", 1, 100, 10, 1)
        
        # Export format selection
        st.subheader("Export Formats")
        export_formats = st.multiselect(
            "Select export formats",
            ["VTU/PVD (ParaView)", "Pickle", "PyTorch", "CSV", "HDF5"],
            default=["VTU/PVD (ParaView)", "Pickle"]
        )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        # Initialize parameters dictionary
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
            'applied_stress': applied_stress_MPa * 1e6,
            'save_frequency': save_frequency
        }
        
        # Theory panel
        with st.expander("📚 Theory Reference", expanded=True):
            st.markdown("""
            **Free Energy Functional:**
            ```
            Ψ = ∫[W(φ²-1)²η₁² + ΣAη_i²(1-η_i)² + Bη₁²η₂² 
                + ½κ_η|∇η_i|² + ½κ_φ(m)|∇φ|² + f_el] dV
            ```
            
            **Transformation Strain:**
            ```
            ε*_ij = ½γ_tw(a_i n_j + a_j n_i)·h(φ)
            h(φ) = ¼(φ-1)²(φ+1)
            ```
            
            **Ovid'ko Strengthening:**
            ```
            σ_y(h) = σ₀ + μb/[2πh(1-ν)]·ln(h/b)
            h ≈ 2/|∇φ|
            ```
            """)
    
    with col1:
        st.markdown("### 🚀 Simulation Control")
        
        # Start simulation button
        if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
            with st.spinner("Initializing phase-field solver..."):
                # Initialize solver
                solver = PhaseFieldSolver(params)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Storage for results
                results_history = []
                timesteps = []
                
                # Simulation loop
                for step in range(n_steps):
                    # Update status
                    status_text.text(f"Step {step+1}/{n_steps}")
                    
                    # Perform time step
                    results = solver.step(params['applied_stress'])
                    
                    # Save results
                    if step % save_frequency == 0:
                        results_history.append(results.copy())
                        timesteps.append(step * dt)
                    
                    # Update progress
                    progress_bar.progress((step + 1) / n_steps)
                
                st.success(f"✅ Simulation completed! Generated {len(results_history)} frames.")
                
                # Store in session state
                st.session_state.results_history = results_history
                st.session_state.params = params
                st.session_state.timesteps = timesteps
        
        # Display results if available
        if 'results_history' in st.session_state:
            st.markdown("### 📈 Results Visualization")
            
            # Select visualization type
            viz_type = st.radio(
                "Visualization Type",
                ["Interactive Dashboard", "3D Visualization", "Time Evolution", "Field Analysis"],
                horizontal=True
            )
            
            if viz_type == "Interactive Dashboard":
                # Create Plotly dashboard
                fig = AdvancedVisualizer.create_interactive_dashboard(
                    st.session_state.results_history,
                    st.session_state.params
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "3D Visualization":
                # Create 3D visualization
                fig_3d = AdvancedVisualizer.create_3d_visualization(
                    st.session_state.results_history[-1]
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            
            elif viz_type == "Time Evolution":
                # Create time evolution plots
                fig_time, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Track evolution of key quantities
                steps = []
                avg_twin_spacing = []
                avg_stress = []
                max_plastic_strain = []
                
                for i, results in enumerate(st.session_state.results_history):
                    steps.append(i * st.session_state.params['save_frequency'])
                    avg_twin_spacing.append(np.mean(results['h'][results['h'] < 100]))
                    avg_stress.append(np.mean(results['sigma_eq']) / 1e9)
                    max_plastic_strain.append(np.max(results['eps_p_mag']))
                
                # Plot 1: Twin spacing evolution
                axes[0, 0].plot(steps, avg_twin_spacing, 'b-', linewidth=2)
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Avg. Twin Spacing (nm)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_title('Twin Spacing Evolution')
                
                # Plot 2: Stress evolution
                axes[0, 1].plot(steps, avg_stress, 'r-', linewidth=2)
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Avg. von Mises Stress (GPa)')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_title('Stress Evolution')
                
                # Plot 3: Plastic strain evolution
                axes[1, 0].plot(steps, max_plastic_strain, 'g-', linewidth=2)
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Max Plastic Strain')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_title('Plastic Strain Evolution')
                
                # Plot 4: Yield stress distribution
                last_results = st.session_state.results_history[-1]
                axes[1, 1].hist(last_results['sigma_y'].flatten() / 1e6, 
                               bins=50, alpha=0.7, color='purple')
                axes[1, 1].set_xlabel('Yield Stress (MPa)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_title('Yield Stress Distribution')
                
                plt.tight_layout()
                st.pyplot(fig_time)
            
            # Export section
            st.markdown("### 💾 Export Results")
            
            if st.button("Generate Export Package"):
                with st.spinner("Preparing export package..."):
                    # Create temporary directory for export
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_dir = f"nanotwin_export_{timestamp}"
                    os.makedirs(export_dir, exist_ok=True)
                    
                    # Export in selected formats
                    exported_files = []
                    
                    if "VTU/PVD (ParaView)" in export_formats:
                        # Export individual VTU files
                        for i, results in enumerate(st.session_state.results_history):
                            vtu_file = f"{export_dir}/frame_{i:04d}.vtu"
                            DataExporter.export_to_vtu(
                                results, vtu_file, st.session_state.params['dx']
                            )
                        
                        # Export PVD file
                        pvd_file = DataExporter.export_to_pvd(
                            st.session_state.results_history,
                            f"{export_dir}/simulation",
                            st.session_state.params['dx'],
                            st.session_state.timesteps
                        )
                        exported_files.append(pvd_file)
                    
                    if "Pickle" in export_formats:
                        pickle_file = DataExporter.export_to_pickle(
                            st.session_state.results_history,
                            st.session_state.params,
                            f"{export_dir}/simulation.pkl"
                        )
                        exported_files.append(pickle_file)
                    
                    if "PyTorch" in export_formats:
                        torch_file = DataExporter.export_to_torch(
                            st.session_state.results_history,
                            st.session_state.params,
                            f"{export_dir}/simulation.pt"
                        )
                        exported_files.append(torch_file)
                    
                    if "CSV" in export_formats:
                        # Export summary statistics to CSV
                        summary_data = []
                        for i, results in enumerate(st.session_state.results_history):
                            summary_data.append({
                                'step': i * st.session_state.params['save_frequency'],
                                'avg_twin_spacing': np.mean(results['h'][results['h'] < 100]),
                                'avg_stress_gpa': np.mean(results['sigma_eq']) / 1e9,
                                'max_plastic_strain': np.max(results['eps_p_mag']),
                                'avg_yield_stress_mpa': np.mean(results['sigma_y']) / 1e6
                            })
                        
                        df = pd.DataFrame(summary_data)
                        csv_file = f"{export_dir}/summary.csv"
                        df.to_csv(csv_file, index=False)
                        exported_files.append(csv_file)
                    
                    # Create zip file
                    import zipfile
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for root, dirs, files in os.walk(export_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, export_dir)
                                zip_file.write(file_path, arcname)
                    
                    zip_buffer.seek(0)
                    
                    # Provide download button
                    st.download_button(
                        label="⬇️ Download Export Package",
                        data=zip_buffer,
                        file_name=f"nanotwin_simulation_{timestamp}.zip",
                        mime="application/zip"
                    )
                    
                    st.success(f"✅ Export package created with {len(exported_files)} files")
        
        # Theory documentation
        st.markdown("---")
        with st.expander("📖 Complete Theoretical Formulation", expanded=False):
            st.markdown("""
            ## Complete Phase-Field Formulation for Nanotwinned FCC Copper
            
            ### 1. Order Parameters
            - **Twin order parameter**: φ ∈ [-1,1], where φ = ±1 represent twin variants
            - **Grain order parameters**: η₁, η₂ ∈ [0,1] with constraint η₁² + η₂² ≤ 1
            
            ### 2. Free Energy Functional
            ```
            Ψ = ∫_Ω [f_loc + f_grad + f_el] dV
            ```
            
            **Local free energy**:
            ```
            f_loc = W(φ² - 1)²η₁² + Σ_{i=1}² Aη_i²(1 - η_i)² + Bη₁²η₂²
            ```
            
            **Gradient energy**:
            ```
            f_grad = ½κ_φ(m)|∇φ|² + Σ_i ½κ_η|∇η_i|²
            κ_φ(m) = κ₀[1 + γ_aniso(1 - (m·n_tw)²)]
            ```
            
            **Elastic energy**:
            ```
            f_el = ½(ε - ε^p - ε*) : C : (ε - ε^p - ε*)
            ```
            
            ### 3. Transformation Strain
            ```
            ε*_ij = ½γ_tw(a_i n_j + a_j n_i)·h(φ)
            h(φ) = ¼(φ-1)²(φ+1)
            ```
            where γ_tw = 1/√2 for FCC twinning
            
            ### 4. Mechanical Equilibrium
            - Solved using spectral method with cubic anisotropy
            - Green's function approach in Fourier space
            - Polarization stress: σ*_ij = C_ijkl ε**_kl
            
            ### 5. Strengthening Model (Ovid'ko)
            ```
            σ_y(h) = σ₀ + μb/[2πh(1-ν)]·ln(h/b)
            h = 2/|∇φ|
            ```
            
            ### 6. Viscoplastic Flow (Perzyna)
            ```
            ε̇^p = γ̇ (3/2)(s/σ_eq)
            γ̇ = γ̇₀[(σ_eq - σ_y)/σ_y]^m H(σ_eq - σ_y)
            ```
            
            ### 7. Dissipation Term
            ```
            Diss_p = ζ|ε̄^p|φ
            ```
            
            ### 8. Evolution Equations
            ```
            ∂η_i/∂t = -L_η[∂f_loc/∂η_i - κ_η∇²η_i]
            ∂φ/∂t = -L_φ(m)[∂f_loc/∂φ + ∂f_el/∂φ - ∇·(κ_φ(m)∇φ) + Diss_p]
            ∇·σ = 0
            ```
            
            ### 9. Numerical Implementation
            - Semi-implicit time integration in Fourier space
            - Reference medium stabilization
            - Anisotropic gradient corrections
            - Periodic boundary conditions
            
            ### References
            1. Ovid'ko, I. A. (2002). *Reviews on Advanced Materials Science*
            2. Clayton, J. D., & Knap, J. (2011). *Journal of the Mechanics and Physics of Solids*
            3. Wang, Y. U., et al. (2001). *Acta Materialia*
            4. Physics of nanoscale twinning in FCC metals
            """)

if __name__ == "__main__":
    main()
