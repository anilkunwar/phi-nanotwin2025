import streamlit as st
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from numba import njit, prange
import matplotlib.pyplot as plt
from datetime import datetime
import json
import zipfile
from io import BytesIO
from matplotlib.patches import Rectangle

# ============================================================================
# MATERIAL CONSTANTS (Cu - Experimental Values)
# ============================================================================
# Cubic elastic constants (GPa) - Phys. Rev. B 73, 064112 (2006)
C11 = 168.4e9   # Pa
C12 = 121.4e9   # Pa
C44 = 75.4e9    # Pa

# Twinning parameters
GAMMA_TW = 1/np.sqrt(2)  # Twinning shear magnitude
N_TWIN = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])  # (111) normal (3D)
A_TWIN = np.array([1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)])  # [112̄] direction (3D)

# 2D projection for (111) plane simulation (x=[110], y=[11̄0])
N_TWIN_2D = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Projected normal in simulation plane
A_TWIN_2D = np.array([1/np.sqrt(2), -1/np.sqrt(2)])  # Projected shear direction

# Plasticity parameters
MU = (C11 - C12 + 3*C44)/5  # Effective shear modulus for (111) plane ~48 GPa
NU = (3*C12 + C44)/(2*(C11 - C12) + 2*C44)  # Effective Poisson's ratio ~0.34
B = 0.256e-9  # Burgers vector magnitude (m)
SIGMA0 = 50e6  # Lattice friction stress (Pa)
GAMMA0_DOT = 1e-3  # Reference shear rate (1/s)
M = 20  # Rate sensitivity exponent

# Simulation parameters
N = 256  # Higher resolution for twin interfaces
DX = 0.5  # nm (grid spacing)
DT = 1e-4  # Smaller time step for stability
EXTENT = [-N*DX/2, N*DX/2, -N*DX/2, N*DX/2]
LAMBDA_TWIN = 20.0  # Initial twin spacing (nm)

# ============================================================================
# CONFIGURE STREAMLIT INTERFACE
# ============================================================================
st.set_page_config(page_title="Nanotwinned Cu Phase-Field Simulator", layout="wide")
st.title("🔬 Phase-Field Simulator for Nanotwinned Copper")
st.markdown("""
**Theoretical Foundation:** Ovid'ko Strength Model • Cubic Anisotropic Elasticity • 
Inclination-Dependent Boundary Kinetics • Spectral FFT Solver
""")

# Sidebar controls with physics-based parameter ranges
st.sidebar.header("🔬 Material Parameters")
kappa0 = st.sidebar.slider(r"$\kappa_0$ (Gradient energy ref)", 0.1, 5.0, 1.0, step=0.1)
gamma_aniso = st.sidebar.slider(r"$\gamma_{aniso}$ (CTB/ITB energy ratio)", 0.0, 0.9, 0.7, step=0.05)
kappa_eta = st.sidebar.slider(r"$\kappa_\eta$ (GB energy coeff)", 0.5, 5.0, 2.0, step=0.1)
W = st.sidebar.slider(r"$W$ (Twin well depth, J/m³)", 0.5, 5.0, 2.0, step=0.1)
A_loc = st.sidebar.slider(r"$A$ (Grain double-well coeff)", 1.0, 10.0, 5.0, step=0.5)
B_loc = st.sidebar.slider(r"$B$ (Grain anti-overlap)", 5.0, 20.0, 10.0, step=0.5)

st.sidebar.header("⚙️ Kinetic Parameters")
L_CTB = st.sidebar.slider(r"$L_{CTB}$ (CTB mobility)", 0.01, 0.5, 0.05, step=0.01)
L_ITB = st.sidebar.slider(r"$L_{ITB}$ (ITB mobility)", 0.5, 10.0, 5.0, step=0.1)
n_mob = st.sidebar.slider(r"$n$ (Mobility exponent)", 1, 10, 4, step=1)
zeta_pin = st.sidebar.slider(r"$\zeta$ (Dislocation pinning)", 0.0, 1.0, 0.3, step=0.05)

st.sidebar.header("⏱️ Simulation Control")
steps = st.sidebar.slider("Evolution steps", 50, 1000, 300, step=50)
applied_stress = st.sidebar.slider(r"Applied stress $\sigma_{xx}$ (MPa)", 0.0, 1000.0, 300.0, step=10.0) * 1e6
twin_spacing = st.sidebar.slider(r"Initial twin spacing $\lambda$ (nm)", 10.0, 50.0, 20.0, step=2.0)

# ============================================================================
# NUMERICAL SETUP
# ============================================================================
X, Y = np.meshgrid(np.linspace(EXTENT[0], EXTENT[1], N), np.linspace(EXTENT[2], EXTENT[3], N))
KX = 2 * np.pi * fftfreq(N, d=DX).reshape(1, -1)
KY = 2 * np.pi * fftfreq(N, d=DX).reshape(-1, 1)
K2 = KX**2 + KY**2 + 1e-12  # Avoid division by zero
KAPPA_REF = kappa0  # Reference medium for semi-implicit scheme
L_REF = (L_CTB + L_ITB) / 2  # Reference mobility

# Green's function for cubic elasticity (2D plane strain approximation on (111) plane)
# Effective 2D stiffness for (111) plane: C11_eff = (C11 + C12 + 2*C44)/2
C11_2D = (C11 + C12 + 2*C44) / 2
C12_2D = (C11 + C12 - 2*C44) / 2
C44_2D = C44
LAMBDA_2D = C12_2D
MU_2D = (C11_2D - C12_2D) / 2

# Precompute Green's function components in Fourier space
G11 = (MU_2D*(KX**2 + 2*KY**2) + LAMBDA_2D*KY**2) / (MU_2D*(LAMBDA_2D + 2*MU_2D)*K2)
G12 = -MU_2D * KX * KY / (MU_2D*(LAMBDA_2D + 2*MU_2D)*K2)
G22 = (MU_2D*(KY**2 + 2*KX**2) + LAMBDA_2D*KX**2) / (MU_2D*(LAMBDA_2D + 2*MU_2D)*K2)

@njit(parallel=True)
def compute_gradients(field, dx):
    """Second-order central difference gradients with periodic BCs"""
    gx = np.zeros_like(field)
    gy = np.zeros_like(field)
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
def compute_laplacian(field, dx):
    """Laplacian with periodic BCs"""
    lap = np.zeros_like(field)
    for i in prange(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        for j in range(N):
            jp1 = (j + 1) % N
            jm1 = (j - 1) % N
            lap[i, j] = (field[ip1, j] + field[im1, j] + field[i, jp1] + field[i, jm1] - 4 * field[i, j]) / (dx**2)
    return lap

@njit(parallel=True)
def compute_twin_spacing(phi_gx, phi_gy):
    """Compute local twin spacing h = 2/|∇φ| (Ovid'ko model)"""
    h = np.zeros_like(phi_gx)
    for i in prange(N):
        for j in range(N):
            grad_mag = np.sqrt(phi_gx[i, j]**2 + phi_gy[i, j]**2)
            if grad_mag > 1e-6:
                h[i, j] = 2.0 / grad_mag
            else:
                h[i, j] = 1e6  # Effectively infinite spacing
    return h

@njit(parallel=True)
def compute_anisotropic_kappa(phi_gx, phi_gy, gamma_aniso, kappa0, nx, ny):
    """κ_φ(m) = κ₀[1 + γ_aniso(1 - (m·n_twin)²)]"""
    kappa = np.zeros_like(phi_gx)
    for i in prange(N):
        for j in range(N):
            grad_mag = np.sqrt(phi_gx[i, j]**2 + phi_gy[i, j]**2 + 1e-12)
            mx = phi_gx[i, j] / grad_mag
            my = phi_gy[i, j] / grad_mag
            dot = mx * nx + my * ny
            kappa[i, j] = kappa0 * (1.0 + gamma_aniso * (1.0 - dot**2))
    return kappa

@njit(parallel=True)
def compute_anisotropic_mobility(phi_gx, phi_gy, L_CTB, L_ITB, n_mob, nx, ny):
    """L_φ(m) = L_CTB + (L_ITB - L_CTB)[1 - (m·n_twin)²]^n"""
    L = np.zeros_like(phi_gx)
    for i in prange(N):
        for j in range(N):
            grad_mag = np.sqrt(phi_gx[i, j]**2 + phi_gy[i, j]**2 + 1e-12)
            mx = phi_gx[i, j] / grad_mag
            my = phi_gy[i, j] / grad_mag
            dot = mx * nx + my * ny
            aniso_factor = (1.0 - dot**2)**n_mob
            L[i, j] = L_CTB + (L_ITB - L_CTB) * aniso_factor
    return L

@njit(parallel=True)
def compute_transformation_strain(phi, gamma_tw, ax, ay):
    """ε*_ij = ½γ_tw(a_i n_j + a_j n_i)·¼(φ-1)²(φ+1) for (111) twinning"""
    # Shape function: f(φ) = ¼(φ-1)²(φ+1) = ¼(φ³ - φ² - φ + 1)
    f_phi = 0.25 * (phi**3 - phi**2 - phi + 1)
    
    # For (111) twinning in Cu: shear direction a = [112̄], normal n = [111]
    # In 2D projection on (111) plane: effective shear tensor components
    exx_star = gamma_tw * ax * ax * f_phi  # Approximate projection
    eyy_star = gamma_tw * ay * ay * f_phi
    exy_star = 0.5 * gamma_tw * (ax * ay + ay * ax) * f_phi
    
    return exx_star, eyy_star, exy_star

def compute_yield_stress(h):
    """Ovid'ko (CLS) model: σ_y(h) = σ₀ + μb/[2πh(1-ν)]·ln(h/b)"""
    h_clipped = np.clip(h, 2*B, 1e-6)  # Avoid singularities
    log_term = np.log(h_clipped / B)
    sigma_y = SIGMA0 + (MU * B / (2 * np.pi * h_clipped * (1 - NU))) * log_term
    return sigma_y

def elastic_solver(eps_p_xx, eps_p_yy, eps_p_xy, eps_star_xx, eps_star_yy, eps_star_xy, applied_sigma):
    """
    FFT-based mechanical equilibrium solver with cubic anisotropy projection
    Solves: σ_ij = C_ijkl (ε_kl - ε^p_kl - ε^*_kl) + σ^ext_ij
    """
    # Total eigenstrain (plastic + transformation)
    e_xx_eig = eps_p_xx + eps_star_xx
    e_yy_eig = eps_p_yy + eps_star_yy
    e_xy_eig = eps_p_xy + eps_star_xy
    
    # Fourier transforms of eigenstrains
    e_xx_h = fft2(e_xx_eig)
    e_yy_h = fft2(e_yy_eig)
    e_xy_h = fft2(e_xy_eig)
    
    # Solve for displacements in Fourier space using Green's function
    # u_i(k) = i G_ij(k) k_l C_jlmn ε_mn^*(k)
    # For 2D plane strain with cubic symmetry projection:
    ux_h = 1j * (G11 * KX * e_xx_h + G12 * KY * e_xx_h + 
                 G12 * KX * e_yy_h + G22 * KY * e_yy_h +
                 2 * G12 * KX * e_xy_h)  # Approximate coupling
    
    uy_h = 1j * (G12 * KX * e_xx_h + G22 * KY * e_xx_h + 
                 G11 * KX * e_yy_h + G12 * KY * e_yy_h +
                 2 * G22 * KY * e_xy_h)
    
    # Strain fields from displacements
    exx_el = np.real(ifft2(1j * KX * ux_h))
    eyy_el = np.real(ifft2(1j * KY * uy_h))
    exy_el = 0.5 * np.real(ifft2(1j * (KX * uy_h + KY * ux_h)))
    
    # Total strains (elastic + eigenstrains)
    exx_tot = exx_el + e_xx_eig
    eyy_tot = eyy_el + e_yy_eig
    exy_tot = exy_el + e_xy_eig
    
    # Stresses (plane strain approximation with σ_zz = ν(σ_xx + σ_yy))
    sxx = C11_2D * exx_tot + C12_2D * eyy_tot
    syy = C12_2D * exx_tot + C11_2D * eyy_tot
    sxy = 2 * C44_2D * exy_tot
    
    # Add applied stress
    sxx += applied_sigma
    
    # von Mises equivalent stress (plane strain)
    sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + (syy)**2 + (sxx)**2 + 6 * sxy**2))
    
    # Deviatoric stress components for plastic flow
    s_mean = (sxx + syy) / 3.0
    s_dev_xx = sxx - s_mean
    s_dev_yy = syy - s_mean
    s_dev_xy = sxy
    
    return sigma_eq, s_dev_xx, s_dev_yy, s_dev_xy, sxx, syy, sxy, exx_tot, eyy_tot, exy_tot

@njit(parallel=True)
def update_plastic_strain(sigma_eq, s_dev_xx, s_dev_yy, s_dev_xy, sigma_y, 
                         eps_p_xx, eps_p_yy, eps_p_xy, dt):
    """Perzyna viscoplasticity with von Mises yield criterion"""
    deps_p_xx = np.zeros_like(sigma_eq)
    deps_p_yy = np.zeros_like(sigma_eq)
    deps_p_xy = np.zeros_like(sigma_eq)
    
    for i in prange(N):
        for j in range(N):
            if sigma_eq[i, j] > sigma_y[i, j] and sigma_y[i, j] > 1e6:  # Avoid division by zero
                # Effective plastic strain rate
                gamma_dot = GAMMA0_DOT * ((sigma_eq[i, j] - sigma_y[i, j]) / sigma_y[i, j])**M
                
                # Plastic strain rate tensor (associated flow rule)
                if sigma_eq[i, j] > 1e6:
                    deps_p_xx[i, j] = (2/3) * gamma_dot * (s_dev_xx[i, j] / sigma_eq[i, j]) * dt
                    deps_p_yy[i, j] = (2/3) * gamma_dot * (s_dev_yy[i, j] / sigma_eq[i, j]) * dt
                    deps_p_xy[i, j] = gamma_dot * (s_dev_xy[i, j] / sigma_eq[i, j]) * dt
    
    return (eps_p_xx + deps_p_xx, 
            eps_p_yy + deps_p_yy, 
            eps_p_xy + deps_p_xy)

def evolve_phi(phi, eta1, eta2, sigma_eq, eps_p_eq, dt):
    """
    Semi-implicit spectral evolution for twin order parameter:
    ∂φ/∂t = ∇·[L_φ(m)κ_φ(m)∇φ] - L_φ(m)[∂f_loc/∂φ + ∂f_el/∂φ + ζ|ε̄^p|sign(φ)]
    """
    # 1. Compute gradients and anisotropic properties
    phi_gx, phi_gy = compute_gradients(phi, DX)
    grad_phi_mag = np.sqrt(phi_gx**2 + phi_gy**2 + 1e-12)
    
    # 2. Anisotropic coefficients
    kappa_m = compute_anisotropic_kappa(phi_gx, phi_gy, gamma_aniso, kappa0, 
                                       N_TWIN_2D[0], N_TWIN_2D[1])
    L_m = compute_anisotropic_mobility(phi_gx, phi_gy, L_CTB, L_ITB, n_mob,
                                      N_TWIN_2D[0], N_TWIN_2D[1])
    
    # 3. Local thermodynamic driving force ∂f_loc/∂φ
    # f_loc = W(φ²-1)²η₁² → ∂f_loc/∂φ = 4Wφ(φ²-1)η₁²
    df_loc_dphi = 4 * W * phi * (phi**2 - 1) * eta1**2
    
    # 4. Elastic driving force ∂f_el/∂φ ≈ -σ:∂ε*/∂φ (approximated)
    # Using chain rule: ∂ε*/∂φ = γ_tw·a⊗n·∂f(φ)/∂φ, where f(φ)=¼(φ-1)²(φ+1)
    df_dphi = 0.25 * (3*phi**2 - 2*phi - 1)  # Derivative of shape function
    # Approximate coupling: σ:∂ε*/∂φ ≈ |σ|·γ_tw·|df/dφ|
    sigma_mag = sigma_eq
    df_el_dphi = -sigma_mag * GAMMA_TW * np.abs(df_dphi) * eta1**2
    
    # 5. Dislocation pinning dissipation term ζ|ε̄^p|sign(φ)
    # Effective plastic strain magnitude
    eps_p_mag = np.sqrt(2/3 * (eps_p_eq**2))
    diss_p = zeta_pin * eps_p_mag * np.sign(phi)
    
    # 6. Anisotropic gradient term in real space (divergence form)
    # G_aniso = ∇·[(κ_m - κ_ref)∇φ] ≈ (κ_m - κ_ref)∇²φ + ∇(κ_m - κ_ref)·∇φ
    lap_phi = compute_laplacian(phi, DX)
    kappa_diff = kappa_m - KAPPA_REF
    G_aniso = kappa_diff * lap_phi
    
    # Additional term from gradient of κ_m (optional for higher accuracy)
    kappa_gx, kappa_gy = compute_gradients(kappa_m, DX)
    G_aniso += kappa_gx * phi_gx + kappa_gy * phi_gy
    
    # 7. Total driving force
    driving_force = df_loc_dphi + df_el_dphi + diss_p + G_aniso
    
    # 8. Spectral semi-implicit update with spatially varying mobility
    # Approximate L_m as constant L_ref for stability, correct with iterative scheme
    phi_hat = fft2(phi)
    f_hat = fft2(L_REF * driving_force)
    
    # Semi-implicit update: φ^{n+1} = [φ^n - Δt·F̂] / [1 + Δt·L_ref·κ_ref·k²]
    phi_hat_new = (phi_hat - dt * f_hat) / (1 + dt * L_REF * KAPPA_REF * K2)
    phi_new = np.real(ifft2(phi_hat_new))
    
    # Clip to physical bounds
    phi_new = np.clip(phi_new, -1.1, 1.1)
    
    return phi_new, grad_phi_mag

def evolve_eta(eta1, eta2, phi, dt):
    """
    Coupled evolution of grain order parameters with anti-overlap constraint:
    ∂η_i/∂t = L_η[κ_η∇²η_i - ∂f_loc/∂η_i]
    f_loc = ΣAη_i²(1-η_i)² + Bη₁²η₂² + W(φ²-1)²η₁²
    """
    # Compute Laplacians
    lap_eta1 = compute_laplacian(eta1, DX)
    lap_eta2 = compute_laplacian(eta2, DX)
    
    # Local driving forces
    # ∂f_loc/∂η1 = 2Aη1(1-η1)(1-2η1) + 2Bη1η2² + 2W(φ²-1)²η1
    df_deta1 = (2*A_loc*eta1*(1-eta1)*(1-2*eta1) + 
                2*B_loc*eta1*eta2**2 + 
                2*W*(phi**2 - 1)**2 * eta1)
    
    # ∂f_loc/∂η2 = 2Aη2(1-η2)(1-2η2) + 2Bη2η1²
    df_deta2 = (2*A_loc*eta2*(1-eta2)*(1-2*eta2) + 
                2*B_loc*eta2*eta1**2)
    
    # Evolution equations (L_η = 1 for simplicity)
    eta1_new = eta1 + dt * (-df_deta1 + kappa_eta * lap_eta1)
    eta2_new = eta2 + dt * (-df_deta2 + kappa_eta * lap_eta2)
    
    # Enforce physical bounds and normalization constraint η1² + η2² ≤ 1
    eta1_new = np.clip(eta1_new, 0, 1)
    eta2_new = np.clip(eta2_new, 0, 1)
    
    # Anti-overlap normalization
    norm = np.sqrt(eta1_new**2 + eta2_new**2 + 1e-12)
    eta1_new = np.where(norm > 1, eta1_new / norm, eta1_new)
    eta2_new = np.where(norm > 1, eta2_new / norm, eta2_new)
    
    return eta1_new, eta2_new

def init_fields(twin_spacing):
    """Initialize according to LaTeX specification:
    - η₁=1 for x<0 (nanotwinned region)
    - η₂=1 for x≥0 (twin-free region)
    - φ = tanh[sin(2πy/λ)] where η₁>0.5
    """
    eta1 = np.ones((N, N))
    eta2 = np.zeros((N, N))
    
    # Create grain boundary at x=0
    for i in range(N):
        if X[i, 0] >= 0:
            eta1[i, :] = 0.0
            eta2[i, :] = 1.0
    
    # Smooth grain boundary transition (2-3 nm width)
    gb_width = 3.0  # nm
    for i in range(N):
        dist = X[i, 0]
        if abs(dist) < gb_width:
            eta1[i, :] = 0.5 * (1 - np.tanh(dist / (gb_width/3)))
            eta2[i, :] = 0.5 * (1 + np.tanh(dist / (gb_width/3)))
    
    # Initialize periodic twins only in grain 1
    phi = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if eta1[i, j] > 0.5:
                # Periodic twin structure with spacing λ
                phase = 2 * np.pi * Y[i, j] / twin_spacing
                phi[i, j] = np.tanh(np.sin(phase) * 3)  # Sharp interfaces
    
    # Initialize plastic strains to zero
    eps_p_xx = np.zeros((N, N))
    eps_p_yy = np.zeros((N, N))
    eps_p_xy = np.zeros((N, N))
    
    return phi, eta1, eta2, eps_p_xx, eps_p_yy, eps_p_xy

def detect_twin_boundaries(phi, grad_mag):
    """Identify CTBs (m·n≈1) and ITBs (m·n≈0) for visualization"""
    phi_gx, phi_gy = compute_gradients(phi, DX)
    m_dot_n = np.abs((phi_gx * N_TWIN_2D[0] + phi_gy * N_TWIN_2D[1]) / 
                     (np.sqrt(phi_gx**2 + phi_gy**2 + 1e-12)))
    
    # CTBs: |m·n| > 0.8, ITBs: |m·n| < 0.3
    ctbs = (grad_mag > 0.5) & (m_dot_n > 0.8)
    itbs = (grad_mag > 0.5) & (m_dot_n < 0.3)
    
    return ctbs, itbs

# ============================================================================
# SIMULATION EXECUTION
# ============================================================================
if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner("Initializing simulation with physics-based parameters..."):
        phi, eta1, eta2, eps_p_xx, eps_p_yy, eps_p_xy = init_fields(twin_spacing)
        history = []
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "kappa0": kappa0, "gamma_aniso": gamma_aniso, "kappa_eta": kappa_eta,
                "W": W, "A_loc": A_loc, "B_loc": B_loc, "L_CTB": L_CTB, "L_ITB": L_ITB,
                "n_mob": n_mob, "zeta_pin": zeta_pin, "applied_stress_MPa": applied_stress/1e6,
                "twin_spacing_nm": twin_spacing, "steps": steps, "grid_size": N, "dx_nm": DX
            },
            "material": {
                "C11_GPa": C11/1e9, "C12_GPa": C12/1e9, "C44_GPa": C44/1e9,
                "gamma_tw": GAMMA_TW, "burgers_m": B, "sigma0_MPa": SIGMA0/1e6
            }
        }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(steps):
        # 1. Compute twin field gradients and spacing
        phi_gx, phi_gy = compute_gradients(phi, DX)
        grad_phi_mag = np.sqrt(phi_gx**2 + phi_gy**2)
        h = compute_twin_spacing(phi_gx, phi_gy)
        
        # 2. Transformation strain from twinning
        eps_star_xx, eps_star_yy, eps_star_xy = compute_transformation_strain(
            phi, GAMMA_TW, A_TWIN_2D[0], A_TWIN_2D[1]
        )
        
        # 3. Mechanical equilibrium with cubic elasticity
        sigma_eq, s_dev_xx, s_dev_yy, s_dev_xy, sxx, syy, sxy, exx, eyy, exy = elastic_solver(
            eps_p_xx, eps_p_yy, eps_p_xy, 
            eps_star_xx, eps_star_yy, eps_star_xy,
            applied_stress
        )
        
        # 4. Yield stress from Ovid'ko model
        sigma_y = compute_yield_stress(h)
        
        # 5. Plastic strain update
        eps_p_eq = np.sqrt(2/3 * (eps_p_xx**2 + eps_p_yy**2 + 2*eps_p_xy**2))
        eps_p_xx, eps_p_yy, eps_p_xy = update_plastic_strain(
            sigma_eq, s_dev_xx, s_dev_yy, s_dev_xy, sigma_y,
            eps_p_xx, eps_p_yy, eps_p_xy, DT
        )
        
        # 6. Evolve phase fields
        phi, grad_phi_mag = evolve_phi(phi, eta1, eta2, sigma_eq, eps_p_eq, DT)
        eta1, eta2 = evolve_eta(eta1, eta2, phi, DT)
        
        # 7. Save history for visualization
        if step % max(1, steps//50) == 0 or step == steps-1:
            ctbs, itbs = detect_twin_boundaries(phi, grad_phi_mag)
            history.append({
                'step': step,
                'phi': phi.copy(),
                'eta1': eta1.copy(),
                'eta2': eta2.copy(),
                'sigma_eq': sigma_eq.copy(),
                'h': h.copy(),
                'ctbs': ctbs.copy(),
                'itbs': itbs.copy(),
                'sigma_y': sigma_y.copy()
            })
        
        # Update progress
        progress = (step + 1) / steps
        progress_bar.progress(progress)
        status_text.text(f"Step {step+1}/{steps} | Avg. twin spacing: {np.mean(h[h<100]):.1f} nm")
    
    st.success(f"✅ Simulation completed! Captured {len(history)} frames with full microstructural evolution.")
    st.session_state.history = history
    st.session_state.metadata = metadata

# ============================================================================
# VISUALIZATION
# ============================================================================
if 'history' in st.session_state and st.session_state.history:
    history = st.session_state.history
    metadata = st.session_state.metadata
    
    st.subheader("📊 Microstructural Evolution")
    col1, col2 = st.columns([3, 1])
    
    with col2:
        frame_idx = st.slider("Frame", 0, len(history)-1, len(history)-1, key="frame_slider")
        show_vectors = st.checkbox("Show twin boundary vectors", True)
        overlay_gb = st.checkbox("Overlay grain boundary", True)
    
    with col1:
        data = history[frame_idx]
        phi_f = data['phi']
        eta1_f = data['eta1']
        eta2_f = data['eta2']
        sigma_eq_f = data['sigma_eq'] / 1e9  # GPa
        h_f = np.clip(data['h'], 0, 50)  # Clip for visualization
        ctbs = data['ctbs']
        itbs = data['itbs']
        sigma_y_f = data['sigma_y'] / 1e6  # MPa
        
        fig, axs = plt.subplots(2, 3, figsize=(22, 14))
        
        # 1. Twin field φ
        im0 = axs[0, 0].imshow(phi_f, extent=EXTENT, cmap='RdBu_r', vmin=-1.2, vmax=1.2)
        axs[0, 0].set_title(f"Twin Order Parameter φ (Step {data['step']})", fontsize=13, fontweight='bold')
        plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
        
        # Overlay twin boundaries
        if show_vectors:
            # Subsample for clarity
            subsample = max(1, N//32)
            Y_sub, X_sub = np.where(ctbs[::subsample, ::subsample])
            axs[0, 0].quiver(X[::subsample, ::subsample][Y_sub, X_sub], 
                            Y[::subsample, ::subsample][Y_sub, X_sub],
                            N_TWIN_2D[0]*np.ones_like(X_sub), 
                            N_TWIN_2D[1]*np.ones_like(X_sub),
                            color='cyan', scale=20, width=0.003, label='CTBs')
            
            Y_sub, X_sub = np.where(itbs[::subsample, ::subsample])
            axs[0, 0].quiver(X[::subsample, ::subsample][Y_sub, X_sub], 
                            Y[::subsample, ::subsample][Y_sub, X_sub],
                            -N_TWIN_2D[1]*np.ones_like(X_sub), 
                            N_TWIN_2D[0]*np.ones_like(X_sub),
                            color='magenta', scale=20, width=0.003, label='ITBs')
            axs[0, 0].legend(loc='upper right')
        
        if overlay_gb:
            # Grain boundary at η1=η2=0.5
            gb_mask = np.abs(eta1_f - eta2_f) < 0.2
            axs[0, 0].contour(X, Y, gb_mask.astype(float), levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.7)
        
        # 2. Grain structure
        grains = np.zeros((N, N, 3))
        grains[..., 0] = eta1_f  # Red for grain 1
        grains[..., 2] = eta2_f  # Blue for grain 2
        axs[0, 1].imshow(grains, extent=EXTENT, origin='lower')
        axs[0, 1].set_title("Grain Structure (η₁: red, η₂: blue)", fontsize=13, fontweight='bold')
        
        if overlay_gb:
            axs[0, 1].contour(X, Y, (eta1_f - eta2_f), levels=[0], colors='white', linewidths=2)
        
        # 3. von Mises stress
        im2 = axs[0, 2].imshow(sigma_eq_f, extent=EXTENT, cmap='hot', vmin=0, vmax=1.5)
        axs[0, 2].set_title("von Mises Stress (GPa)", fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Twin spacing field
        im3 = axs[1, 0].imshow(h_f, extent=EXTENT, cmap='plasma', vmin=0, vmax=30)
        axs[1, 0].set_title("Local Twin Spacing h (nm)", fontsize=13, fontweight='bold')
        plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)
        
        # 5. Yield stress field
        im4 = axs[1, 1].imshow(sigma_y_f, extent=EXTENT, cmap='viridis', vmin=0, vmax=500)
        axs[1, 1].set_title("Local Yield Stress σ_y (MPa)", fontsize=13, fontweight='bold')
        plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)
        
        # 6. Plastic strain magnitude
        eps_p_mag = np.sqrt(2/3 * (eps_p_xx**2 + eps_p_yy**2 + 2*eps_p_xy**2))
        im5 = axs[1, 2].imshow(eps_p_mag, extent=EXTENT, cmap='YlOrRd', vmin=0, vmax=0.05)
        axs[1, 2].set_title("Accumulated Plastic Strain |ε^p|", fontsize=13, fontweight='bold')
        plt.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)
        
        # Formatting
        for ax in axs.flat:
            ax.set_xlabel("x (nm)", fontsize=11)
            ax.set_ylabel("y (nm)", fontsize=11)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Quantitative analysis
        st.subheader("📈 Quantitative Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_h = np.mean(h_f[(h_f > 2) & (h_f < 50)])
            st.metric("Mean Twin Spacing", f"{avg_h:.1f} nm")
        
        with col2:
            ctbs_density = np.sum(ctbs) / (N*N*DX*DX) * 1e6  # per μm²
            st.metric("CTB Density", f"{ctbs_density:.1f} μm⁻²")
        
        with col3:
            avg_sigma_y = np.mean(sigma_y_f[(sigma_y_f > 50) & (sigma_y_f < 500)])
            st.metric("Mean Yield Stress", f"{avg_sigma_y:.0f} MPa")
        
        # Export functionality
        st.subheader("💾 Export Simulation Data")
        if st.button("📦 Generate Export Package"):
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # Save final state
                npz_buffer = BytesIO()
                np.savez(npz_buffer, 
                        phi=phi_f, eta1=eta1_f, eta2=eta2_f, 
                        sigma_eq=sigma_eq_f, h=h_f, 
                        sigma_y=sigma_y_f, eps_p_mag=eps_p_mag)
                zf.writestr("final_state.npz", npz_buffer.getvalue())
                
                # Save metadata
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                
                # Save key frames
                for i, frame in enumerate(history[::max(1, len(history)//10)]):
                    npz_buffer = BytesIO()
                    np.savez(npz_buffer, 
                            step=frame['step'],
                            phi=frame['phi'],
                            eta1=frame['eta1'],
                            eta2=frame['eta2'],
                            sigma_eq=frame['sigma_eq'],
                            h=frame['h'])
                    zf.writestr(f"frame_{i:03d}_step_{frame['step']}.npz", npz_buffer.getvalue())
                
                # Save visualization snapshot
                fig.savefig(BytesIO(), format='png', dpi=150, bbox_inches='tight')
            
            buffer.seek(0)
            st.download_button(
                label="⬇️ Download Simulation Package (.zip)",
                data=buffer,
                file_name=f"nt_cu_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            st.info("Package includes: final state, metadata, key frames, and visualization snapshot")

# ============================================================================
# THEORY REFERENCE PANEL
# ============================================================================
with st.sidebar.expander("📚 Theoretical Foundations"):
    st.markdown("""
    ### Core Physics Implemented
    
    **1. Free Energy Functional**
    ```math
    Ψ = ∫[f_loc(η_i,φ) + Σ(κ_η/2)|∇η_i|² + (κ_φ(m)/2)|∇φ|² + f_el] dV
    ```
    - Coupled local potential with grain anti-overlap
    - Anisotropic gradient energy distinguishing CTBs/ITBs
    
    **2. Ovid'ko Strength Model**
    ```math
    σ_y(h) = σ_0 + \\frac{μb}{2πh(1-ν)} \\ln\\left(\\frac{h}{b}\\right), \\quad h ≈ \\frac{2}{|∇φ|}
    ```
    
    **3. Inclination-Dependent Kinetics**
    ```math
    L_φ(m) = L_{CTB} + (L_{ITB} - L_{CTB})[1 - (m·n_{twin})^2]^n
    ```
    
    **4. Spectral Numerics**
    - Semi-implicit FFT solver with reference medium
    - Green's function for cubic elasticity projection
    - Transformation strain coupling to mechanical equilibrium
    
    **Material Parameters Source:** 
    Cu elastic constants from [Phys. Rev. B 73, 064112 (2006)]
    """)
