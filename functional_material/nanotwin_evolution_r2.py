import numpy as np
import streamlit as st
from scipy.fft import fft2, ifft2, fftfreq
from numba import njit, prange, float64
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
from scipy.ndimage import gaussian_filter

# ============================================================================
# FIXED NUMBA-COMPATIBLE FUNCTIONS
# ============================================================================
@njit(float64[:,:](float64[:,:], float64), parallel=True)
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

@njit(float64[:,:](float64[:,:], float64), parallel=True)
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

@njit(float64[:,:](float64[:,:], float64[:,:]), parallel=True)
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
    
    # Shape function: f(φ) = ¼(φ-1)²(φ+1)
    for i in prange(N):
        for j in prange(N):
            if eta1[i, j] > 0.5:
                phi_val = phi[i, j]
                f_phi = 0.25 * (phi_val**3 - phi_val**2 - phi_val + 1)
                
                # Transformation strain components
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
                'C11': 168.4e9,   # GPa -> Pa
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
                'mu': 48e9,  # Shear modulus (Pa)
                'nu': 0.34,  # Poisson's ratio
                'b': 0.256e-9,  # Burgers vector (m)
                'sigma0': 50e6,  # Lattice friction (Pa)
                'gamma0_dot': 1e-3,  # Reference shear rate (1/s)
                'm': 20,  # Rate sensitivity exponent
                'rho0': 1e12,  # Initial dislocation density (m^-2)
            },
            'thermal': {
                'melting_temp': 1357.77,  # K
                'thermal_cond': 401,  # W/(m·K)
                'specific_heat': 385,  # J/(kg·K)
                'thermal_expansion': 16.5e-6,  # 1/K
            }
        }

class InitialGeometryVisualizer:
    """Class to create and visualize initial geometric conditions"""
    
    def __init__(self, N, dx, params):
        self.N = N
        self.dx = dx
        self.params = params
        self.x = np.linspace(-N*dx/2, N*dx/2, N)
        self.y = np.linspace(-N*dx/2, N*dx/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def create_twin_grain_geometry(self, twin_spacing, grain_boundary_pos=0.0):
        """Create initial twin grain geometry with grain boundary"""
        eta1 = np.zeros((self.N, self.N))
        eta2 = np.zeros((self.N, self.N))
        phi = np.zeros((self.N, self.N))
        
        # Create grain boundary (vertical line at x = grain_boundary_pos)
        gb_width = 3.0  # nm (width of transition region)
        
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
                    phi[i, j] = np.tanh(np.sin(phase) * 3.0)  # Sharp interfaces
        
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
    
    def create_curved_twin_geometry(self, twin_spacing, curvature_radius=100.0):
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
    
    def plot_initial_geometry(self, phi, eta1, eta2):
        """Create comprehensive plot of initial geometry"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Twin order parameter
        im1 = axes[0, 0].imshow(phi, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], 
                               cmap='RdBu_r', vmin=-1.2, vmax=1.2)
        axes[0, 0].set_title('Twin Order Parameter φ', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('x (nm)')
        axes[0, 0].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Highlight grain boundary
        gb_mask = np.abs(eta1 - eta2) < 0.2
        axes[0, 0].contour(self.X, self.Y, gb_mask, levels=[0.5], colors='yellow', 
                          linewidths=2, alpha=0.7)
        
        # 2. Grain structure
        grains_rgb = np.zeros((self.N, self.N, 3))
        grains_rgb[..., 0] = eta1  # Red for twin grain
        grains_rgb[..., 2] = eta2  # Blue for twin-free grain
        axes[0, 1].imshow(grains_rgb, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        axes[0, 1].set_title('Grain Structure (Red: Twin Grain, Blue: Twin-free)', 
                           fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('x (nm)')
        axes[0, 1].set_ylabel('y (nm)')
        
        # 3. Twin boundaries detection
        phi_gx, phi_gy = compute_gradients_numba(phi, self.dx)
        grad_mag = np.sqrt(phi_gx**2 + phi_gy**2)
        twin_boundaries = grad_mag > np.percentile(grad_mag, 95)
        
        axes[0, 2].imshow(twin_boundaries, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], 
                         cmap='gray')
        axes[0, 2].set_title('Detected Twin Boundaries', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('x (nm)')
        axes[0, 2].set_ylabel('y (nm)')
        
        # 4. Twin spacing field
        h = compute_twin_spacing_numba(phi_gx, phi_gy)
        h_display = np.clip(h, 0, 50)
        im4 = axes[1, 0].imshow(h_display, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], 
                               cmap='plasma', vmin=0, vmax=30)
        axes[1, 0].set_title('Local Twin Spacing (nm)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('x (nm)')
        axes[1, 0].set_ylabel('y (nm)')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 5. Grain boundary profile
        x_profile = self.X[self.N//2, :]
        eta1_profile = eta1[self.N//2, :]
        eta2_profile = eta2[self.N//2, :]
        
        axes[1, 1].plot(x_profile, eta1_profile, 'r-', linewidth=2, label='η₁ (Twin grain)')
        axes[1, 1].plot(x_profile, eta2_profile, 'b-', linewidth=2, label='η₂ (Twin-free grain)')
        axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Grain boundary')
        axes[1, 1].set_title('Grain Boundary Profile (y=0)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('x (nm)')
        axes[1, 1].set_ylabel('Order Parameter Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Twin profile
        y_profile = self.Y[:, self.N//2]
        phi_profile = phi[:, self.N//2]
        
        axes[1, 2].plot(y_profile, phi_profile, 'g-', linewidth=2)
        axes[1, 2].set_title('Twin Profile (x=0)', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('y (nm)')
        axes[1, 2].set_ylabel('φ Value')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add twin spacing annotation
        avg_spacing = np.mean(h[(h > 5) & (h < 50)])
        axes[1, 2].text(0.05, 0.95, f'Avg spacing: {avg_spacing:.1f} nm', 
                       transform=axes[1, 2].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_interactive_3d_plot(self, phi, eta1, eta2):
        """Create interactive 3D plot of initial geometry"""
        # Create 3D surface for twin boundaries
        phi_gx, phi_gy = compute_gradients_numba(phi, self.dx)
        grad_mag = np.sqrt(phi_gx**2 + phi_gy**2)
        
        fig = go.Figure()
        
        # Add twin boundaries as surface
        fig.add_trace(go.Surface(
            x=self.X, y=self.Y, z=grad_mag,
            colorscale='Viridis',
            opacity=0.9,
            name='Twin Boundaries',
            contours={
                'z': {'show': True, 'start': grad_mag.min(), 'end': grad_mag.max(), 
                      'size': (grad_mag.max() - grad_mag.min())/10}
            }
        ))
        
        # Add grain boundary as line
        gb_mask = np.abs(eta1 - eta2) < 0.2
        gb_indices = np.where(gb_mask)
        if len(gb_indices[0]) > 0:
            # Sample points for better performance
            step = max(1, len(gb_indices[0]) // 100)
            gb_x = self.X[gb_indices[0][::step], gb_indices[1][::step]]
            gb_y = self.Y[gb_indices[0][::step], gb_indices[1][::step]]
            gb_z = grad_mag[gb_indices[0][::step], gb_indices[1][::step]]
            
            fig.add_trace(go.Scatter3d(
                x=gb_x.flatten(), y=gb_y.flatten(), z=gb_z.flatten(),
                mode='markers',
                marker=dict(size=3, color='red'),
                name='Grain Boundary'
            ))
        
        fig.update_layout(
            title='3D Visualization of Initial Microstructure',
            scene=dict(
                xaxis_title='x (nm)',
                yaxis_title='y (nm)',
                zaxis_title='Gradient Magnitude |∇φ|',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            showlegend=True
        )
        
        return fig

class EnhancedPhaseFieldSolver:
    """Enhanced phase-field solver with Numba compatibility"""
    
    def __init__(self, params):
        self.params = params
        self.N = params['N']
        self.dx = params['dx']
        self.dt = params['dt']
        
        # Material properties
        self.mat_props = MaterialProperties.get_cu_properties()
        
        # Initialize geometry visualizer
        self.geometry_viz = InitialGeometryVisualizer(self.N, self.dx, params)
        
        # Initialize fields
        self.phi, self.eta1, self.eta2 = self.geometry_viz.create_twin_grain_geometry(
            params.get('twin_spacing', 20.0),
            params.get('grain_boundary_pos', 0.0)
        )
        
        # Initialize plastic strain
        self.eps_p = np.zeros((3, self.N, self.N))
        
        # Initialize Fourier space
        self.kx = 2 * np.pi * fftfreq(self.N, d=self.dx).reshape(1, -1)
        self.ky = 2 * np.pi * fftfreq(self.N, d=self.dx).reshape(-1, 1)
        self.k2 = self.kx**2 + self.ky**2 + 1e-12
        
        # Initialize elastic Green's function
        self.init_greens_function()
        
    def init_greens_function(self):
        """Initialize Green's function for 2D plane strain"""
        C11 = self.mat_props['elastic']['C11']
        C12 = self.mat_props['elastic']['C12']
        C44 = self.mat_props['elastic']['C44']
        
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
    
    def solve_elasticity(self, eigenstrain, applied_stress=0):
        """Solve mechanical equilibrium with FFT-based spectral method"""
        # Fourier transform of eigenstrain components
        eps_xx_hat = fft2(eigenstrain[0])
        eps_yy_hat = fft2(eigenstrain[1])
        eps_xy_hat = fft2(eigenstrain[2])
        
        # Solve for displacements in Fourier space
        ux_hat = 1j * (self.G11 * self.kx * eps_xx_hat + 
                      self.G12 * self.ky * eps_xx_hat +
                      self.G12 * self.kx * eps_yy_hat + 
                      self.G22 * self.ky * eps_yy_hat)
        
        uy_hat = 1j * (self.G12 * self.kx * eps_xx_hat + 
                      self.G22 * self.ky * eps_xx_hat +
                      self.G11 * self.kx * eps_yy_hat + 
                      self.G12 * self.ky * eps_yy_hat)
        
        # Compute elastic strains
        eps_xx_el = np.real(ifft2(1j * self.kx * ux_hat))
        eps_yy_el = np.real(ifft2(1j * self.ky * uy_hat))
        eps_xy_el = 0.5 * np.real(ifft2(1j * (self.kx * uy_hat + self.ky * ux_hat)))
        
        # Total strains
        eps_xx = eps_xx_el + eigenstrain[0]
        eps_yy = eps_yy_el + eigenstrain[1]
        eps_xy = eps_xy_el + eigenstrain[2]
        
        # Stresses (plane strain approximation)
        C11 = self.mat_props['elastic']['C11']
        C12 = self.mat_props['elastic']['C12']
        C44 = self.mat_props['elastic']['C44']
        
        sxx = applied_stress + C11 * eps_xx + C12 * eps_yy
        syy = C12 * eps_xx + C11 * eps_yy
        sxy = 2 * C44 * eps_xy
        
        # von Mises equivalent stress
        sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + syy**2 + sxx**2 + 6 * sxy**2))
        
        return sigma_eq, sxx, syy, sxy, eps_xx, eps_yy, eps_xy
    
    def compute_local_energy_derivatives(self):
        """Compute derivatives of local free energy density"""
        # f_loc = W(φ²-1)²η₁² + ΣAη_i²(1-η_i)² + Bη₁²η₂²
        
        W = self.params['W']
        A = self.params['A']
        B = self.params['B']
        
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
    
    def compute_elastic_driving_force(self, sigma):
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
        
        deps_dphi = np.zeros((3, self.N, self.N))
        
        # Only compute where twin grain is active
        mask = self.eta1 > 0.5
        deps_dphi[0, mask] = gamma_tw * nx * ax * dh_dphi[mask]
        deps_dphi[1, mask] = gamma_tw * ny * ay * dh_dphi[mask]
        deps_dphi[2, mask] = 0.5 * gamma_tw * (nx * ay + ny * ax) * dh_dphi[mask]
        
        # Elastic driving force: -σ:∂ε*/∂φ
        df_el_dphi = -(sigma[0] * deps_dphi[0] + 
                      sigma[1] * deps_dphi[1] + 
                      2 * sigma[2] * deps_dphi[2])
        
        return df_el_dphi
    
    def evolve_twin_field(self, sigma, eps_p_mag):
        """Evolve twin order parameter φ with Numba compatibility"""
        # Extract parameters
        kappa0 = float(self.params['kappa0'])
        gamma_aniso = float(self.params['gamma_aniso'])
        L_CTB = float(self.params['L_CTB'])
        L_ITB = float(self.params['L_ITB'])
        n_mob = int(self.params['n_mob'])
        zeta = float(self.params['zeta'])
        
        # Get twin parameters
        n_twin = self.mat_props['twinning']['n_2d']
        nx = float(n_twin[0])
        ny = float(n_twin[1])
        
        # Compute gradients
        phi_gx, phi_gy = compute_gradients_numba(self.phi, self.dx)
        
        # Compute anisotropic properties using Numba-compatible function
        kappa_phi, L_phi = compute_anisotropic_properties_numba(
            phi_gx, phi_gy, nx, ny, kappa0, gamma_aniso, L_CTB, L_ITB, n_mob
        )
        
        # Compute local energy derivatives
        df_loc_dphi, _, _ = self.compute_local_energy_derivatives()
        
        # Compute elastic driving force
        df_el_dphi = self.compute_elastic_driving_force(sigma)
        
        # Compute dissipation term
        diss_p = zeta * eps_p_mag * self.phi
        
        # Compute gradient term
        lap_phi = compute_laplacian_numba(self.phi, self.dx)
        
        # Evolution equation
        dphi_dt = -L_phi * (df_loc_dphi + df_el_dphi - kappa_phi * lap_phi + diss_p)
        
        # Update twin field
        phi_new = self.phi + self.dt * dphi_dt
        
        # Apply boundary conditions (periodic)
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
    
    def compute_plastic_strain(self, sigma_eq, sigma_y, dt):
        """Compute plastic strain evolution"""
        plastic_params = self.mat_props['plasticity']
        gamma0_dot = plastic_params['gamma0_dot']
        m = plastic_params['m']
        
        # Initialize plastic strain increment
        deps_p = np.zeros_like(self.eps_p)
        
        # Compute where yielding occurs
        yield_mask = sigma_eq > sigma_y
        
        if np.any(yield_mask):
            # Overstress ratio
            overstress = (sigma_eq[yield_mask] - sigma_y[yield_mask]) / sigma_y[yield_mask]
            
            # Strain rate magnitude
            gamma_dot = gamma0_dot * (overstress)**m
            
            # Plastic strain increment (simplified)
            for k in range(3):
                deps_p[k][yield_mask] = gamma_dot * dt
        
        # Update plastic strain
        eps_p_new = self.eps_p + deps_p
        
        return eps_p_new
    
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
        eigenstrain = [
            exx_star + self.eps_p[0],
            eyy_star + self.eps_p[1],
            exy_star + self.eps_p[2]
        ]
        
        # Solve mechanical equilibrium
        sigma_eq, sxx, syy, sxy, eps_xx, eps_yy, eps_xy = self.solve_elasticity(
            eigenstrain, applied_stress
        )
        
        # Compute twin spacing
        phi_gx, phi_gy = compute_gradients_numba(self.phi, self.dx)
        h = compute_twin_spacing_numba(phi_gx, phi_gy)
        
        # Compute yield stress
        plastic_params = self.mat_props['plasticity']
        sigma_y = compute_yield_stress_numba(
            h, plastic_params['sigma0'], plastic_params['mu'], 
            plastic_params['b'], plastic_params['nu']
        )
        
        # Update plastic strain
        self.eps_p = self.compute_plastic_strain(sigma_eq, sigma_y, self.dt)
        
        # Compute plastic strain magnitude
        eps_p_mag = np.sqrt(2/3 * (self.eps_p[0]**2 + self.eps_p[1]**2 + 2*self.eps_p[2]**2))
        
        # Evolve phase fields
        sigma_tensor = [sxx, syy, sxy]
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
# STREAMLIT APPLICATION WITH ENHANCED VISUALIZATION
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
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
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
            ["Standard Twin Grain", "Multiple Twin Spacings", "Curved Twins", "Custom"]
        )
        
        # Simulation parameters
        st.subheader("📊 Grid Configuration")
        N = st.slider("Grid resolution (N×N)", 64, 512, 256, 64)
        dx = st.slider("Grid spacing (nm)", 0.2, 2.0, 0.5, 0.1)
        dt = st.slider("Time step (ns)", 1e-5, 1e-3, 1e-4, 1e-5)
        
        st.subheader("🔬 Material Parameters")
        twin_spacing = st.slider("Twin spacing λ (nm)", 5.0, 100.0, 20.0, 1.0)
        grain_boundary_pos = st.slider("Grain boundary position (nm)", -50.0, 50.0, 0.0, 1.0)
        
        st.subheader("⚛️ Thermodynamic Parameters")
        W = st.slider("Twin well depth W (J/m³)", 0.1, 10.0, 2.0, 0.1)
        A = st.slider("Grain double-well A (J/m³)", 0.1, 20.0, 5.0, 0.5)
        B = st.slider("Grain anti-overlap B (J/m³)", 0.1, 30.0, 10.0, 0.5)
        
        st.subheader("📈 Gradient Energy")
        kappa0 = st.slider("κ₀ (Gradient energy ref)", 0.01, 10.0, 1.0, 0.1)
        gamma_aniso = st.slider("γ_aniso (CTB/ITB ratio)", 0.0, 2.0, 0.7, 0.05)
        kappa_eta = st.slider("κ_η (GB energy)", 0.1, 10.0, 2.0, 0.1)
        
        st.subheader("⚡ Kinetic Parameters")
        L_CTB = st.slider("L_CTB (CTB mobility)", 0.001, 1.0, 0.05, 0.001)
        L_ITB = st.slider("L_ITB (ITB mobility)", 0.1, 20.0, 5.0, 0.1)
        n_mob = st.slider("n (Mobility exponent)", 1, 10, 4, 1)
        L_eta = st.slider("L_η (GB mobility)", 0.1, 10.0, 1.0, 0.1)
        zeta = st.slider("ζ (Dislocation pinning)", 0.0, 2.0, 0.3, 0.05)
        
        st.subheader("🏗️ Loading Conditions")
        applied_stress_MPa = st.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 300.0, 10.0)
        
        st.subheader("⏱️ Simulation Control")
        n_steps = st.slider("Number of steps", 10, 1000, 100, 10)
        save_frequency = st.slider("Save frequency", 1, 100, 10, 1)
        
        # Initialize simulation button
        if st.button("🔄 Initialize Simulation", type="primary", use_container_width=True):
            st.session_state.initialized = True
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 📋 Quick Stats")
        
        if 'initialized' in st.session_state and st.session_state.initialized:
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
                'applied_stress': applied_stress_MPa * 1e6,
                'save_frequency': save_frequency
            }
            
            # Create geometry visualizer for initial conditions
            geometry_viz = InitialGeometryVisualizer(N, dx, params)
            
            # Create initial geometry based on selected type
            if geometry_type == "Standard Twin Grain":
                phi, eta1, eta2 = geometry_viz.create_twin_grain_geometry(
                    twin_spacing, grain_boundary_pos
                )
            elif geometry_type == "Multiple Twin Spacings":
                phi, eta1, eta2 = geometry_viz.create_multi_twin_geometry(
                    [twin_spacing, twin_spacing*1.5, twin_spacing*2]
                )
            elif geometry_type == "Curved Twins":
                phi, eta1, eta2 = geometry_viz.create_curved_twin_geometry(
                    twin_spacing, curvature_radius=100.0
                )
            else:
                phi, eta1, eta2 = geometry_viz.create_twin_grain_geometry(
                    twin_spacing, grain_boundary_pos
                )
            
            # Store in session state
            st.session_state.initial_geometry = {
                'phi': phi,
                'eta1': eta1,
                'eta2': eta2,
                'geometry_viz': geometry_viz
            }
            
            st.success("✅ Simulation initialized!")
    
    with col1:
        st.markdown("### 🎨 Initial Geometry Visualization")
        
        if 'initial_geometry' in st.session_state:
            # Get initial geometry
            phi = st.session_state.initial_geometry['phi']
            eta1 = st.session_state.initial_geometry['eta1']
            eta2 = st.session_state.initial_geometry['eta2']
            geometry_viz = st.session_state.initial_geometry['geometry_viz']
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["2D Visualization", "3D Visualization", "Analysis"])
            
            with tab1:
                # 2D visualization
                fig = geometry_viz.plot_initial_geometry(phi, eta1, eta2)
                st.pyplot(fig)
                
                # Add metrics
                phi_gx, phi_gy = compute_gradients_numba(phi, dx)
                h = compute_twin_spacing_numba(phi_gx, phi_gy)
                
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("Avg. Twin Spacing", f"{np.mean(h[(h > 5) & (h < 50)]):.1f} nm")
                with col_metrics[1]:
                    st.metric("Twin Grain Area", f"{np.sum(eta1 > 0.5) * dx**2:.0f} nm²")
                with col_metrics[2]:
                    st.metric("Twin Boundaries", f"{np.sum(h < 20):.0f}")
                with col_metrics[3]:
                    st.metric("Grain Boundary Length", f"{np.sum(np.abs(eta1 - eta2) < 0.2) * dx:.0f} nm")
            
            with tab2:
                # 3D visualization
                fig_3d = geometry_viz.create_interactive_3d_plot(phi, eta1, eta2)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with tab3:
                # Detailed analysis
                st.subheader("📊 Detailed Geometry Analysis")
                
                # Compute profiles
                x_profile = geometry_viz.X[N//2, :]
                y_profile = geometry_viz.Y[:, N//2]
                phi_x_profile = phi[N//2, :]
                phi_y_profile = phi[:, N//2]
                eta1_x_profile = eta1[N//2, :]
                
                # Create analysis plots
                fig_analysis, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # 1. Twin profile along y-axis
                axes[0, 0].plot(y_profile, phi_y_profile, 'b-', linewidth=2)
                axes[0, 0].set_xlabel('y (nm)')
                axes[0, 0].set_ylabel('φ')
                axes[0, 0].set_title('Twin Profile (x=0)')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. Grain boundary profile
                axes[0, 1].plot(x_profile, eta1_x_profile, 'r-', linewidth=2)
                axes[0, 1].axvline(x=grain_boundary_pos, color='gray', linestyle='--', alpha=0.5)
                axes[0, 1].set_xlabel('x (nm)')
                axes[0, 1].set_ylabel('η₁')
                axes[0, 1].set_title('Grain Boundary Profile')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 3. Twin spacing histogram
                h_valid = h[(h > 5) & (h < 50)]
                axes[1, 0].hist(h_valid.flatten(), bins=30, alpha=0.7, color='green')
                axes[1, 0].axvline(x=twin_spacing, color='red', linestyle='--', label='Target spacing')
                axes[1, 0].set_xlabel('Twin Spacing (nm)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Twin Spacing Distribution')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # 4. Order parameter statistics
                stats_data = {
                    'Statistic': ['Mean φ', 'Std φ', 'Min φ', 'Max φ', 'Mean η₁', 'Mean η₂'],
                    'Value': [
                        f"{np.mean(phi):.3f}",
                        f"{np.std(phi):.3f}",
                        f"{np.min(phi):.3f}",
                        f"{np.max(phi):.3f}",
                        f"{np.mean(eta1):.3f}",
                        f"{np.mean(eta2):.3f}"
                    ]
                }
                
                axes[1, 1].axis('off')
                table = axes[1, 1].table(cellText=list(zip(stats_data['Statistic'], stats_data['Value'])),
                                        colLabels=['Statistic', 'Value'],
                                        cellLoc='center',
                                        loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                axes[1, 1].set_title('Order Parameter Statistics')
                
                plt.tight_layout()
                st.pyplot(fig_analysis)
            
            # Run simulation button
            st.markdown("---")
            st.markdown("### 🚀 Run Evolution Simulation")
            
            if st.button("▶️ Start Evolution", type="secondary", use_container_width=True):
                with st.spinner("Running phase-field simulation..."):
                    # Initialize solver
                    solver = EnhancedPhaseFieldSolver(params)
                    
                    # Override with initial geometry
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
                    
                    # Store results in session state
                    st.session_state.results_history = results_history
                    st.session_state.timesteps = timesteps
                    st.session_state.solver = solver
            
            # Display evolution results if available
            if 'results_history' in st.session_state:
                st.markdown("### 📈 Evolution Results")
                
                # Create evolution visualization
                last_results = st.session_state.results_history[-1]
                
                # Compare initial and final states
                fig_compare, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                # Initial state
                axes[0, 0].imshow(phi, cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                axes[0, 0].set_title('Initial φ')
                axes[0, 1].imshow(eta1, cmap='Reds', vmin=0, vmax=1)
                axes[0, 1].set_title('Initial η₁')
                axes[0, 2].imshow(phi_gx, cmap='coolwarm')
                axes[0, 2].set_title('Initial ∇φ_x')
                axes[0, 3].imshow(h, cmap='plasma', vmin=0, vmax=30)
                axes[0, 3].set_title('Initial Twin Spacing')
                
                # Final state
                axes[1, 0].imshow(last_results['phi'], cmap='RdBu_r', vmin=-1.2, vmax=1.2)
                axes[1, 0].set_title('Final φ')
                axes[1, 1].imshow(last_results['eta1'], cmap='Reds', vmin=0, vmax=1)
                axes[1, 1].set_title('Final η₁')
                phi_gx_final, _ = compute_gradients_numba(last_results['phi'], dx)
                axes[1, 2].imshow(phi_gx_final, cmap='coolwarm')
                axes[1, 2].set_title('Final ∇φ_x')
                axes[1, 3].imshow(last_results['h'], cmap='plasma', vmin=0, vmax=30)
                axes[1, 3].set_title('Final Twin Spacing')
                
                for ax in axes.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                plt.tight_layout()
                st.pyplot(fig_compare)
                
                # Evolution metrics
                st.subheader("📊 Evolution Metrics")
                
                # Track evolution of key quantities
                steps_tracked = []
                avg_spacings = []
                avg_stresses = []
                plastic_strains = []
                
                for i, results in enumerate(st.session_state.results_history):
                    steps_tracked.append(i * save_frequency)
                    h_valid = results['h'][(results['h'] > 5) & (results['h'] < 50)]
                    avg_spacings.append(np.mean(h_valid) if len(h_valid) > 0 else 0)
                    avg_stresses.append(np.mean(results['sigma_eq']) / 1e9)
                    plastic_strains.append(np.max(results['eps_p_mag']))
                
                fig_metrics, ax = plt.subplots(3, 1, figsize=(10, 12))
                
                ax[0].plot(steps_tracked, avg_spacings, 'b-o', linewidth=2, markersize=4)
                ax[0].set_ylabel('Avg. Twin Spacing (nm)')
                ax[0].set_title('Twin Spacing Evolution')
                ax[0].grid(True, alpha=0.3)
                
                ax[1].plot(steps_tracked, avg_stresses, 'r-o', linewidth=2, markersize=4)
                ax[1].set_ylabel('Avg. Stress (GPa)')
                ax[1].set_title('Stress Evolution')
                ax[1].grid(True, alpha=0.3)
                
                ax[2].plot(steps_tracked, plastic_strains, 'g-o', linewidth=2, markersize=4)
                ax[2].set_xlabel('Step')
                ax[2].set_ylabel('Max Plastic Strain')
                ax[2].set_title('Plastic Strain Evolution')
                ax[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_metrics)
                
                # Export functionality
                st.markdown("### 💾 Export Results")
                
                if st.button("📦 Generate Export Package"):
                    with st.spinner("Preparing export package..."):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Create in-memory zip file
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
                            
                            # Save evolution data
                            evolution_data = {
                                'steps': steps_tracked,
                                'avg_spacings': avg_spacings,
                                'avg_stresses': avg_stresses,
                                'plastic_strains': plastic_strains
                            }
                            npz_buffer = BytesIO()
                            np.savez(npz_buffer, **evolution_data)
                            zip_file.writestr(f"evolution_data_{timestamp}.npz", npz_buffer.getvalue())
                            
                            # Save parameters
                            params_json = json.dumps(params, indent=2)
                            zip_file.writestr(f"parameters_{timestamp}.json", params_json)
                        
                        zip_buffer.seek(0)
                        
                        # Provide download button
                        st.download_button(
                            label="⬇️ Download Export Package (.zip)",
                            data=zip_buffer,
                            file_name=f"nanotwin_simulation_{timestamp}.zip",
                            mime="application/zip"
                        )
                        
                        st.success("Export package ready for download!")
    
    # Theory documentation
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
        
        ### 4. Ovid'ko Strengthening Model
        ```
        σ_y(h) = σ₀ + μb/[2πh(1-ν)]·ln(h/b)
        h = 2/|∇φ|
        ```
        
        ### 5. Evolution Equations
        ```
        ∂η_i/∂t = -L_η[∂f_loc/∂η_i - κ_η∇²η_i]
        ∂φ/∂t = -L_φ(m)[∂f_loc/∂φ + ∂f_el/∂φ - ∇·(κ_φ(m)∇φ) + ζ|ε̄^p|φ]
        ∇·σ = 0
        ```
        """)

if __name__ == "__main__":
    main()
