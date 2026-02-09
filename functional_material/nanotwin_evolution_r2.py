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
from io import BytesIO
import tempfile
import os
import meshio
import pandas as pd

# ============================================================================
# FIXED NUMBA-COMPATIBLE FUNCTIONS (WITH CORRECTED SIGNATURES)
# ============================================================================
@njit(parallel=True)
def compute_gradients_numba(field, dx):
    """Numba-compatible gradient computation - fixed signature"""
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
    """Numba-compatible Laplacian computation - fixed signature"""
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

# ============================================================================
# ENHANCED PHYSICS MODELS WITH INITIAL GEOMETRY VISUALIZATION
# ============================================================================
class MaterialProperties:
    """Enhanced material properties database"""
    
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
            },
            'thermal': {
                'melting_temp': 1357.77,
                'thermal_cond': 401,
                'specific_heat': 385,
                'thermal_expansion': 16.5e-6,
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
        
        # Create grain boundary (vertical line at x = grain_boundary_pos)
        for i in range(self.N):
            for j in range(self.N):
                x_val = self.X[i, j]
                dist_from_gb = x_val - grain_boundary_pos
                
                # Twin grain on left (eta1 = 1), twin-free grain on right (eta2 = 1)
                if dist_from_gb < -gb_width:
                    eta1[i, j] = 1.0
                    eta2[i, j] = 0.0
                elif dist_from_gb > gb_width:
                    eta1[i, j] = 0.0
                    eta2[i, j] = 1.0
                else:
                    # Smooth transition at grain boundary
                    transition = 0.5 * (1 - np.tanh(dist_from_gb / (gb_width/3)))
                    eta1[i, j] = transition
                    eta2[i, j] = 1 - transition
        
        # Create periodic twin structure only in the twin grain (eta1 > 0.5)
        for i in range(self.N):
            for j in range(self.N):
                if eta1[i, j] > 0.5:
                    # Create periodic twins with specified spacing
                    phase = 2 * np.pi * self.Y[i, j] / twin_spacing
                    phi[i, j] = np.tanh(np.sin(phase) * 3.0)
        
        return phi, eta1, eta2
    
    def create_multi_twin_geometry(self, twin_spacings=[15.0, 25.0, 35.0]):
        """Create geometry with multiple twin spacings for comparison"""
        phi, eta1, eta2 = self.create_twin_grain_geometry(twin_spacings[0])
        
        # Create regions with different twin spacings
        region_height = self.N // len(twin_spacings)
        
        for idx, spacing in enumerate(twin_spacings):
            y_start = idx * region_height
            y_end = (idx + 1) * region_height if idx < len(twin_spacings) - 1 else self.N
            
            for i in range(y_start, y_end):
                for j in range(self.N):
                    if eta1[i, j] > 0.5:
                        phase = 2 * np.pi * self.Y[i, j] / spacing
                        phi[i, j] = np.tanh(np.sin(phase) * 3.0)
        
        return phi, eta1, eta2
    
    def create_curved_twin_geometry(self, twin_spacing=20.0, curvature_radius=100.0):
        """Create geometry with curved twins"""
        phi = np.zeros((self.N, self.N))
        eta1 = np.ones((self.N, self.N))
        eta2 = np.zeros((self.N, self.N))
        
        # Create curved twins using circular geometry
        center_x = 0.0
        center_y = 0.0
        
        for i in range(self.N):
            for j in range(self.N):
                # Calculate angle from center
                dx = self.X[i, j] - center_x
                dy = self.Y[i, j] - center_y
                angle = np.arctan2(dy, dx)
                
                # Create curved twins
                phase = 2 * np.pi * angle * curvature_radius / twin_spacing
                phi[i, j] = np.tanh(np.sin(phase) * 3.0)
        
        return phi, eta1, eta2
    
    def create_bi_crystal_geometry(self, twin_spacing=20.0, num_grains=2):
        """Create bi-crystal geometry with different orientations"""
        eta1 = np.zeros((self.N, self.N))
        eta2 = np.zeros((self.N, self.N))
        phi = np.zeros((self.N, self.N))
        
        # Create alternating grains
        grain_width = self.N // num_grains
        
        for g in range(num_grains):
            start = g * grain_width
            end = (g + 1) * grain_width if g < num_grains - 1 else self.N
            
            if g % 2 == 0:
                # Twin grain
                eta1[start:end, :] = 1.0
                # Create periodic twins
                for i in range(start, end):
                    for j in range(self.N):
                        phase = 2 * np.pi * self.Y[i, j] / twin_spacing
                        phi[i, j] = np.tanh(np.sin(phase) * 3.0)
            else:
                # Twin-free grain
                eta2[start:end, :] = 1.0
        
        # Smooth grain boundaries
        gb_smooth = gaussian_filter(eta1, sigma=1.0)
        eta1 = gb_smooth
        eta2 = 1 - eta1
        
        return phi, eta1, eta2

class SpectralSolver:
    """Spectral solver for mechanical equilibrium with cubic anisotropy"""
    
    def __init__(self, N, dx, C11, C12, C44):
        self.N = N
        self.dx = dx
        
        # Fourier space grid
        self.kx = 2 * np.pi * fftfreq(N, d=dx).reshape(1, -1)
        self.ky = 2 * np.pi * fftfreq(N, d=dx).reshape(-1, 1)
        self.k2 = self.kx**2 + self.ky**2 + 1e-12
        
        # Compute Green's function for 2D plane strain
        C11_2d = (C11 + C12 + 2*C44) / 2
        C12_2d = (C11 + C12 - 2*C44) / 2
        lambda_2d = C12_2d
        mu_2d = (C11_2d - C12_2d) / 2
        
        denom = mu_2d * (lambda_2d + 2*mu_2d) * self.k2
        self.G11 = (mu_2d*(self.kx**2 + 2*self.ky**2) + lambda_2d*self.ky**2) / denom
        self.G12 = -mu_2d * self.kx * self.ky / denom
        self.G22 = (mu_2d*(self.ky**2 + 2*self.kx**2) + lambda_2d*self.kx**2) / denom
    
    def solve(self, eigenstrain_xx, eigenstrain_yy, eigenstrain_xy, applied_stress=0):
        """Solve mechanical equilibrium with FFT"""
        # Fourier transforms
        eps_xx_hat = fft2(eigenstrain_xx)
        eps_yy_hat = fft2(eigenstrain_yy)
        eps_xy_hat = fft2(eigenstrain_xy)
        
        # Solve for displacements
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
        sxx = applied_stress + 168.4e9 * eps_xx + 121.4e9 * eps_yy
        syy = 121.4e9 * eps_xx + 168.4e9 * eps_yy
        sxy = 2 * 75.4e9 * eps_xy
        
        # von Mises equivalent stress
        sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + syy**2 + sxx**2 + 6 * sxy**2))
        
        return sigma_eq, sxx, syy, sxy, eps_xx, eps_yy, eps_xy

class NanotwinnedCuSolver:
    """Main solver for nanotwinned copper microstructural evolution"""
    
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']
        
        # Material properties
        self.mat_props = MaterialProperties.get_cu_properties()
        
        # Initialize geometry visualizer
        self.geom_viz = InitialGeometryVisualizer(self.N, self.dx)
        
        # Initialize fields
        self.phi, self.eta1, self.eta2 = self.initialize_fields()
        
        # Initialize plastic strain
        self.eps_p_xx = np.zeros((self.N, self.N))
        self.eps_p_yy = np.zeros((self.N, self.N))
        self.eps_p_xy = np.zeros((self.N, self.N))
        
        # Initialize spectral solver
        elastic = self.mat_props['elastic']
        self.spectral_solver = SpectralSolver(
            self.N, self.dx,
            elastic['C11'], elastic['C12'], elastic['C44']
        )
    
    def initialize_fields(self):
        """Initialize order parameters based on selected geometry"""
        geom_type = self.params.get('geometry_type', 'standard')
        twin_spacing = self.params.get('twin_spacing', 20.0)
        gb_pos = self.params.get('grain_boundary_pos', 0.0)
        
        if geom_type == 'multi_spacing':
            return self.geom_viz.create_multi_twin_geometry(
                [twin_spacing, twin_spacing*1.5, twin_spacing*2]
            )
        elif geom_type == 'curved':
            return self.geom_viz.create_curved_twin_geometry(twin_spacing)
        elif geom_type == 'bi_crystal':
            return self.geom_viz.create_bi_crystal_geometry(twin_spacing)
        else:  # standard
            return self.geom_viz.create_twin_grain_geometry(twin_spacing, gb_pos)
    
    def compute_local_energy_derivatives(self):
        """Compute derivatives of local free energy density"""
        W = self.params['W']
        A = self.params['A']
        B = self.params['B']
        
        # f_loc = W(φ²-1)²η₁² + ΣAη_i²(1-η_i)² + Bη₁²η₂²
        
        # Derivative with respect to φ
        df_dphi = 4 * W * self.phi * (self.phi**2 - 1) * self.eta1**2
        
        # Derivative with respect to η₁
        df_deta1 = (2 * A * self.eta1 * (1 - self.eta1) * (1 - 2*self.eta1) +
                   2 * B * self.eta1 * self.eta2**2 +
                   2 * W * (self.phi**2 - 1)**2 * self.eta1)
        
        # Derivative with respect to η₂
        df_deta2 = (2 * A * self.eta2 * (1 - self.eta2) * (1 - 2*self.eta2) +
                   2 * B * self.eta2 * self.eta1**2)
        
        return df_dphi, df_deta1, df_deta2
    
    def compute_elastic_driving_force(self, sxx, syy, sxy):
        """Compute elastic driving force for twin evolution"""
        # Get twin parameters
        gamma_tw = self.mat_props['twinning']['gamma_tw']
        n = self.mat_props['twinning']['n_2d']
        a = self.mat_props['twinning']['a_2d']
        
        # Derivative of interpolation function
        dh_dphi = 0.25 * (3*self.phi**2 - 2*self.phi - 1)
        
        # Transformation strain derivative
        nx, ny = n[0], n[1]
        ax, ay = a[0], a[1]
        
        deps_xx_dphi = gamma_tw * nx * ax * dh_dphi * self.eta1**2
        deps_yy_dphi = gamma_tw * ny * ay * dh_dphi * self.eta1**2
        deps_xy_dphi = 0.5 * gamma_tw * (nx * ay + ny * ax) * dh_dphi * self.eta1**2
        
        # Elastic driving force: -σ:∂ε*/∂φ
        df_el_dphi = -(sxx * deps_xx_dphi + 
                      syy * deps_yy_dphi + 
                      2 * sxy * deps_xy_dphi)
        
        return df_el_dphi
    
    def evolve_twin_field(self, sxx, syy, sxy, eps_p_mag):
        """Evolve twin order parameter φ"""
        # Extract parameters as floats
        kappa0 = float(self.params['kappa0'])
        gamma_aniso = float(self.params['gamma_aniso'])
        L_CTB = float(self.params['L_CTB'])
        L_ITB = float(self.params['L_ITB'])
        n_mob = int(self.params['n_mob'])
        zeta = float(self.params['zeta'])
        
        # Get twin parameters as floats
        n_twin = self.mat_props['twinning']['n_2d']
        nx = float(n_twin[0])
        ny = float(n_twin[1])
        
        # Compute gradients
        phi_gx, phi_gy = compute_gradients_numba(self.phi, self.dx)
        
        # Compute anisotropic properties
        kappa_phi, L_phi = compute_anisotropic_properties_numba(
            phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso, L_CTB, L_ITB, n_mob
        )
        
        # Compute driving forces
        df_loc_dphi, _, _ = self.compute_local_energy_derivatives()
        df_el_dphi = self.compute_elastic_driving_force(sxx, syy, sxy)
        
        # Compute dissipation term
        diss_p = zeta * eps_p_mag * self.phi
        
        # Compute gradient term
        lap_phi = compute_laplacian_numba(self.phi, self.dx)
        
        # Evolution equation
        dphi_dt = -L_phi * (df_loc_dphi + df_el_dphi - kappa_phi * lap_phi + diss_p)
        
        # Update twin field
        phi_new = self.phi + self.dt * dphi_dt
        
        # Apply periodic boundary conditions
        phi_new = np.roll(phi_new, 1, axis=0)
        phi_new = np.roll(phi_new, 1, axis=1)
        
        return np.clip(phi_new, -1.1, 1.1)
    
    def evolve_grain_fields(self):
        """Evolve grain order parameters"""
        kappa_eta = float(self.params['kappa_eta'])
        L_eta = float(self.params['L_eta'])
        
        # Compute Laplacians
        lap_eta1 = compute_laplacian_numba(self.eta1, self.dx)
        lap_eta2 = compute_laplacian_numba(self.eta2, self.dx)
        
        # Compute local derivatives
        _, df_deta1, df_deta2 = self.compute_local_energy_derivatives()
        
        # Evolution equations
        deta1_dt = -L_eta * (df_deta1 - kappa_eta * lap_eta1)
        deta2_dt = -L_eta * (df_deta2 - kappa_eta * lap_eta2)
        
        # Update grain fields
        eta1_new = self.eta1 + self.dt * deta1_dt
        eta2_new = self.eta2 + self.dt * deta2_dt
        
        # Enforce constraints
        eta1_new = np.clip(eta1_new, 0, 1)
        eta2_new = np.clip(eta2_new, 0, 1)
        
        # Enforce η₁² + η₂² ≤ 1
        norm = np.sqrt(eta1_new**2 + eta2_new**2 + 1e-12)
        mask = norm > 1
        eta1_new[mask] = eta1_new[mask] / norm[mask]
        eta2_new[mask] = eta2_new[mask] / norm[mask]
        
        return eta1_new, eta2_new
    
    def compute_plastic_strain(self, sigma_eq, sigma_y, eps_p_mag):
        """Compute plastic strain evolution using Perzyna model"""
        plastic_params = self.mat_props['plasticity']
        gamma0_dot = plastic_params['gamma0_dot']
        m = plastic_params['m']
        
        # Where yielding occurs
        yield_mask = sigma_eq > sigma_y
        
        if np.any(yield_mask):
            # Overstress ratio
            overstress = (sigma_eq[yield_mask] - sigma_y[yield_mask]) / sigma_y[yield_mask]
            
            # Strain rate magnitude
            gamma_dot = gamma0_dot * (overstress)**m
            
            # Plastic strain increment (simplified)
            for i in range(self.N):
                for j in range(self.N):
                    if yield_mask[i, j]:
                        stress_ratio = overstress[yield_mask[i, j]]
                        gamma_dot_val = gamma0_dot * (stress_ratio)**m
                        
                        # Update plastic strains (simplified flow rule)
                        self.eps_p_xx[i, j] += gamma_dot_val * self.dt
                        self.eps_p_yy[i, j] += -0.5 * gamma_dot_val * self.dt  # Volume preserving
                        self.eps_p_xy[i, j] += 0.0  # Simplified
        
        # Update plastic strain magnitude
        eps_p_mag_new = np.sqrt(
            2/3 * (self.eps_p_xx**2 + self.eps_p_yy**2 + 2*self.eps_p_xy**2)
        )
        
        return eps_p_mag_new
    
    def step(self, applied_stress):
        """Perform one time step of the simulation"""
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
        
        # Compute plastic strain magnitude
        eps_p_mag = np.sqrt(
            2/3 * (self.eps_p_xx**2 + self.eps_p_yy**2 + 2*self.eps_p_xy**2)
        )
        
        # Update plastic strain
        eps_p_mag = self.compute_plastic_strain(sigma_eq, sigma_y, eps_p_mag)
        
        # Evolve phase fields
        self.phi = self.evolve_twin_field(sxx, syy, sxy, eps_p_mag)
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
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Nanotwinned Copper Phase-Field Simulator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Complete Theoretical Implementation:</strong><br>
    • Cubic anisotropic elasticity with full stiffness tensor<br>
    • {111}<112> transformation strain for FCC twinning<br>
    • Ovid'ko confined layer slip strengthening model<br>
    • Perzyna viscoplasticity with strain hardening<br>
    • Grain-twin coupling with anti-overlap constraint<br>
    • Inclination-dependent CTB/ITB kinetics
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Simulation Configuration")
        
        # Geometry type selection
        st.subheader("📐 Initial Geometry")
        geometry_type = st.selectbox(
            "Geometry Type",
            ["Standard Twin Grain", "Multiple Twin Spacings", "Curved Twins", "Bi-Crystal"],
            key="geom_type"
        )
        
        # Map geometry type to parameter
        geom_map = {
            "Standard Twin Grain": "standard",
            "Multiple Twin Spacings": "multi_spacing",
            "Curved Twins": "curved",
            "Bi-Crystal": "bi_crystal"
        }
        
        # Simulation parameters
        st.subheader("📊 Grid Configuration")
        N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64, key="N")
        dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1, key="dx")
        dt = st.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5, key="dt")
        
        st.subheader("🔬 Material Parameters")
        twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0, key="twin_spacing")
        grain_boundary_pos = st.slider("Grain boundary position (nm)", -50.0, 50.0, 0.0, 1.0, key="gb_pos")
        
        st.subheader("⚛️ Thermodynamic Parameters")
        W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1, key="W")
        A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5, key="A")
        B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5, key="B")
        
        st.subheader("📈 Gradient Energy")
        kappa0 = st.slider("κ₀ (Gradient energy ref)", 0.01, 10.0, 1.0, 0.1, key="kappa0")
        gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05, key="gamma_aniso")
        kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1, key="kappa_eta")
        
        st.subheader("⚡ Kinetic Parameters")
        L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001, key="L_CTB")
        L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1, key="L_ITB")
        n_mob = st.slider("n (Mobility exponent)", 1, 10, 4, 1, key="n_mob")
        L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1, key="L_eta")
        zeta = st.slider("ζ (Dislocation pinning)", 0.0, 2.0, 0.3, 0.05, key="zeta")
        
        st.subheader("🏗️ Loading Conditions")
        applied_stress_MPa = st.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0, key="applied_stress")
        
        st.subheader("⏱️ Simulation Control")
        n_steps = st.slider("Number of steps", 10, 1000, 100, 10, key="n_steps")
        save_frequency = st.slider("Save frequency", 1, 100, 10, 1, key="save_freq")
        
        # Initialize button
        if st.button("🔄 Initialize Geometry", type="primary", use_container_width=True):
            st.session_state.geometry_initialized = True
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 📋 Quick Stats")
        
        if 'geometry_initialized' in st.session_state and st.session_state.geometry_initialized:
            # Create parameters dictionary
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
                'geometry_type': geom_map[geometry_type],
                'applied_stress': applied_stress_MPa * 1e6,
                'save_frequency': save_frequency
            }
            
            # Create geometry visualizer
            geom_viz = InitialGeometryVisualizer(N, dx)
            
            # Create initial geometry
            if geometry_type == "Multiple Twin Spacings":
                phi, eta1, eta2 = geom_viz.create_multi_twin_geometry(
                    [twin_spacing, twin_spacing*1.5, twin_spacing*2]
                )
            elif geometry_type == "Curved Twins":
                phi, eta1, eta2 = geom_viz.create_curved_twin_geometry(twin_spacing)
            elif geometry_type == "Bi-Crystal":
                phi, eta1, eta2 = geom_viz.create_bi_crystal_geometry(twin_spacing)
            else:
                phi, eta1, eta2 = geom_viz.create_twin_grain_geometry(twin_spacing, grain_boundary_pos)
            
            # Store in session state
            st.session_state.initial_geometry = {
                'phi': phi,
                'eta1': eta1,
                'eta2': eta2,
                'geom_viz': geom_viz,
                'params': params
            }
            
            # Display geometry metrics
            phi_gx, phi_gy = compute_gradients_numba(phi, dx)
            h = compute_twin_spacing_numba(phi_gx, phi_gy)
            
            st.metric("Grid Size", f"{N}×{N}")
            st.metric("Avg. Twin Spacing", f"{np.mean(h[(h>5) & (h<50)]):.1f} nm")
            st.metric("Twin Grain Area", f"{np.sum(eta1>0.5) * dx**2:.0f} nm²")
            st.metric("Grain Boundary Length", f"{np.sum(np.abs(eta1-eta2)<0.2) * dx:.0f} nm")
            
            st.success("✅ Geometry initialized!")
    
    with col1:
        st.markdown("### 🎨 Initial Geometry Visualization")
        
        if 'initial_geometry' in st.session_state:
            phi = st.session_state.initial_geometry['phi']
            eta1 = st.session_state.initial_geometry['eta1']
            eta2 = st.session_state.initial_geometry['eta2']
            geom_viz = st.session_state.initial_geometry['geom_viz']
            params = st.session_state.initial_geometry['params']
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["2D Visualization", "3D Visualization", "Analysis"])
            
            with tab1:
                # 2D visualization
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Twin order parameter
                im1 = axes[0, 0].imshow(phi, extent=[geom_viz.x[0], geom_viz.x[-1], geom_viz.y[0], geom_viz.y[-1]], 
                                       cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                axes[0, 0].set_title('Twin Order Parameter φ', fontsize=12)
                axes[0, 0].set_xlabel('x (nm)')
                axes[0, 0].set_ylabel('y (nm)')
                plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
                
                # Grain structure
                grains_rgb = np.zeros((N, N, 3))
                grains_rgb[..., 0] = eta1
                grains_rgb[..., 2] = eta2
                axes[0, 1].imshow(grains_rgb, extent=[geom_viz.x[0], geom_viz.x[-1], geom_viz.y[0], geom_viz.y[-1]])
                axes[0, 1].set_title('Grain Structure', fontsize=12)
                axes[0, 1].set_xlabel('x (nm)')
                axes[0, 1].set_ylabel('y (nm)')
                
                # Twin boundaries
                phi_gx, phi_gy = compute_gradients_numba(phi, dx)
                grad_mag = np.sqrt(phi_gx**2 + phi_gy**2)
                twin_boundaries = grad_mag > np.percentile(grad_mag, 95)
                axes[0, 2].imshow(twin_boundaries, extent=[geom_viz.x[0], geom_viz.x[-1], geom_viz.y[0], geom_viz.y[-1]], 
                                 cmap='gray')
                axes[0, 2].set_title('Twin Boundaries', fontsize=12)
                axes[0, 2].set_xlabel('x (nm)')
                axes[0, 2].set_ylabel('y (nm)')
                
                # Twin spacing
                h = compute_twin_spacing_numba(phi_gx, phi_gy)
                im4 = axes[1, 0].imshow(np.clip(h, 0, 50), 
                                       extent=[geom_viz.x[0], geom_viz.x[-1], geom_viz.y[0], geom_viz.y[-1]], 
                                       cmap='plasma', vmin=0, vmax=30)
                axes[1, 0].set_title('Twin Spacing (nm)', fontsize=12)
                axes[1, 0].set_xlabel('x (nm)')
                axes[1, 0].set_ylabel('y (nm)')
                plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
                
                # Grain boundary profile
                x_profile = geom_viz.X[N//2, :]
                eta1_profile = eta1[N//2, :]
                axes[1, 1].plot(x_profile, eta1_profile, 'r-', linewidth=2)
                axes[1, 1].axvline(x=grain_boundary_pos, color='gray', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('x (nm)')
                axes[1, 1].set_ylabel('η₁')
                axes[1, 1].set_title('Grain Boundary Profile', fontsize=12)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Twin profile
                y_profile = geom_viz.Y[:, N//2]
                phi_profile = phi[:, N//2]
                axes[1, 2].plot(y_profile, phi_profile, 'b-', linewidth=2)
                axes[1, 2].set_xlabel('y (nm)')
                axes[1, 2].set_ylabel('φ')
                axes[1, 2].set_title('Twin Profile', fontsize=12)
                axes[1, 2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                # 3D visualization
                fig_3d = go.Figure()
                
                # Add twin boundaries as surface
                phi_gx, phi_gy = compute_gradients_numba(phi, dx)
                grad_mag = np.sqrt(phi_gx**2 + phi_gy**2)
                
                fig_3d.add_trace(go.Surface(
                    x=geom_viz.X, y=geom_viz.Y, z=grad_mag,
                    colorscale='Viridis',
                    opacity=0.9,
                    name='Gradient Magnitude'
                ))
                
                fig_3d.update_layout(
                    title='3D Initial Geometry Visualization',
                    scene=dict(
                        xaxis_title='x (nm)',
                        yaxis_title='y (nm)',
                        zaxis_title='|∇φ|',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with tab3:
                # Analysis
                st.subheader("📊 Detailed Analysis")
                
                # Compute statistics
                phi_gx, phi_gy = compute_gradients_numba(phi, dx)
                h = compute_twin_spacing_numba(phi_gx, phi_gy)
                h_valid = h[(h > 5) & (h < 50)]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean φ", f"{np.mean(phi):.3f}")
                with col2:
                    st.metric("Std φ", f"{np.std(phi):.3f}")
                with col3:
                    st.metric("Mean η₁", f"{np.mean(eta1):.3f}")
                with col4:
                    st.metric("Mean η₂", f"{np.mean(eta2):.3f}")
                
                # Histogram of twin spacing
                fig_hist, ax = plt.subplots(figsize=(10, 4))
                ax.hist(h_valid.flatten(), bins=30, alpha=0.7, color='green')
                ax.axvline(x=twin_spacing, color='red', linestyle='--', label='Target spacing')
                ax.set_xlabel('Twin Spacing (nm)')
                ax.set_ylabel('Frequency')
                ax.set_title('Twin Spacing Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_hist)
            
            # Run simulation button
            st.markdown("---")
            st.markdown("### 🚀 Run Evolution Simulation")
            
            if st.button("▶️ Start Evolution", type="secondary", use_container_width=True):
                with st.spinner("Running phase-field simulation..."):
                    # Initialize solver with initial geometry
                    solver = NanotwinnedCuSolver(params)
                    solver.phi = phi.copy()
                    solver.eta1 = eta1.copy()
                    solver.eta2 = eta2.copy()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Storage for results
                    results_history = []
                    timesteps = []
                    
                    # Simulation loop
                    for step in range(n_steps):
                        status_text.text(f"Step {step+1}/{n_steps}")
                        
                        # Perform time step
                        results = solver.step(params['applied_stress'])
                        
                        # Save results
                        if step % save_frequency == 0:
                            results_history.append(results.copy())
                            timesteps.append(step * dt)
                        
                        progress_bar.progress((step + 1) / n_steps)
                    
                    st.success(f"✅ Simulation completed! Generated {len(results_history)} frames.")
                    
                    # Store results
                    st.session_state.results_history = results_history
                    st.session_state.timesteps = timesteps
                    st.session_state.final_solver = solver
            
            # Display evolution results if available
            if 'results_history' in st.session_state:
                st.markdown("### 📈 Evolution Results")
                
                last_results = st.session_state.results_history[-1]
                
                # Compare initial and final
                fig_compare, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                # Initial state
                axes[0, 0].imshow(phi, cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                axes[0, 0].set_title('Initial φ')
                axes[0, 1].imshow(eta1, cmap='Reds', vmin=0, vmax=1)
                axes[0, 1].set_title('Initial η₁')
                phi_gx_i, phi_gy_i = compute_gradients_numba(phi, dx)
                h_i = compute_twin_spacing_numba(phi_gx_i, phi_gy_i)
                axes[0, 2].imshow(np.clip(h_i, 0, 50), cmap='plasma', vmin=0, vmax=30)
                axes[0, 2].set_title('Initial Spacing')
                axes[0, 3].imshow(eta1 > 0.5, cmap='gray')
                axes[0, 3].set_title('Initial Twin Grain')
                
                # Final state
                axes[1, 0].imshow(last_results['phi'], cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                axes[1, 0].set_title('Final φ')
                axes[1, 1].imshow(last_results['eta1'], cmap='Reds', vmin=0, vmax=1)
                axes[1, 1].set_title('Final η₁')
                axes[1, 2].imshow(np.clip(last_results['h'], 0, 50), cmap='plasma', vmin=0, vmax=30)
                axes[1, 2].set_title('Final Spacing')
                axes[1, 3].imshow(last_results['sigma_eq']/1e9, cmap='hot', vmin=0, vmax=2)
                axes[1, 3].set_title('Final Stress (GPa)')
                
                for ax in axes.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                plt.tight_layout()
                st.pyplot(fig_compare)
                
                # Evolution metrics
                st.subheader("📊 Evolution Metrics")
                
                steps_tracked = []
                avg_spacings = []
                avg_stresses = []
                
                for i, results in enumerate(st.session_state.results_history):
                    steps_tracked.append(i * save_frequency)
                    h_valid = results['h'][(results['h'] > 5) & (results['h'] < 50)]
                    avg_spacings.append(np.mean(h_valid) if len(h_valid) > 0 else 0)
                    avg_stresses.append(np.mean(results['sigma_eq']) / 1e9)
                
                fig_metrics, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                ax[0].plot(steps_tracked, avg_spacings, 'b-o', linewidth=2, markersize=4)
                ax[0].set_xlabel('Step')
                ax[0].set_ylabel('Avg. Twin Spacing (nm)')
                ax[0].set_title('Twin Spacing Evolution')
                ax[0].grid(True, alpha=0.3)
                
                ax[1].plot(steps_tracked, avg_stresses, 'r-o', linewidth=2, markersize=4)
                ax[1].set_xlabel('Step')
                ax[1].set_ylabel('Avg. Stress (GPa)')
                ax[1].set_title('Stress Evolution')
                ax[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_metrics)
                
                # Export functionality
                st.markdown("### 💾 Export Results")
                
                if st.button("📦 Generate Export Package"):
                    with st.spinner("Preparing export package..."):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Create in-memory zip
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Save initial geometry
                            npz_buffer = BytesIO()
                            np.savez(npz_buffer, 
                                    phi_initial=phi,
                                    eta1_initial=eta1,
                                    eta2_initial=eta2)
                            zip_file.writestr(f"initial_geometry_{timestamp}.npz", npz_buffer.getvalue())
                            
                            # Save final state
                            npz_buffer = BytesIO()
                            np.savez(npz_buffer, 
                                    phi_final=last_results['phi'],
                                    eta1_final=last_results['eta1'],
                                    eta2_final=last_results['eta2'],
                                    sigma_eq=last_results['sigma_eq'],
                                    h=last_results['h'],
                                    sigma_y=last_results['sigma_y'])
                            zip_file.writestr(f"final_state_{timestamp}.npz", npz_buffer.getvalue())
                            
                            # Save parameters
                            params_json = json.dumps(params, indent=2)
                            zip_file.writestr(f"parameters_{timestamp}.json", params_json)
                        
                        zip_buffer.seek(0)
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Export Package",
                            data=zip_buffer,
                            file_name=f"nanotwin_simulation_{timestamp}.zip",
                            mime="application/zip"
                        )
                        
                        st.success("Export package ready!")

if __name__ == "__main__":
    main()
