import streamlit as st
import numpy as np
from scipy.fft import fft2, ifft2
from numba import njit, prange
import matplotlib.pyplot as plt
from datetime import datetime
import json
import zipfile
from io import BytesIO

# Material constants for Cu (typical values)
MU = 48e9  # Shear modulus (Pa)
NU = 0.34  # Poisson's ratio
B = 0.256e-9  # Burgers vector (nm)
SIGMA0 = 50e6  # Lattice friction (Pa)
GAMMA0_DOT = 1e-3  # Ref shear rate (1/s)
M = 20  # Rate sensitivity

# Simulation parameters
N = 128  # Grid size
DX = 0.5  # nm
DT = 1e-3  # Time step
EXTENT = [-N*DX/2, N*DX/2, -N*DX/2, N*DX/2]

# Configure page
st.set_page_config(page_title="Nanotwinned Cu Phase-Field Simulator", layout="wide")
st.title("🔬 Phase-Field Simulator for Nanotwinned Copper")
st.markdown("**Dislocation-Based Plasticity • Anisotropic Twins • Spectral FFT Implementation**")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")
kappa0 = st.sidebar.slider("κ_0 (Gradient coeff for φ)", 0.1, 2.0, 0.5)
gamma_aniso = st.sidebar.slider("γ_aniso (Anisotropy strength)", 0.0, 1.0, 0.5)
kappa_eta = st.sidebar.slider("κ_η (Grain boundary coeff)", 0.1, 2.0, 0.3)
alpha = st.sidebar.slider("α (Local potential coeff)", 1.0, 5.0, 2.0)
beta = st.sidebar.slider("β (Local potential coeff)", 1.0, 5.0, 3.0)
gamma = st.sidebar.slider("γ (Grain coupling)", 1.0, 5.0, 2.0)
W = st.sidebar.slider("W (Twin well depth)", 0.1, 1.0, 0.5)
L_phi = st.sidebar.slider("L_φ (Twin mobility)", 0.1, 10.0, 1.0)
L_eta = st.sidebar.slider("L_η (Grain mobility)", 0.1, 10.0, 1.0)
steps = st.sidebar.slider("Evolution steps", 10, 500, 100)
applied_stress = st.sidebar.slider("Applied stress σ_xx (MPa)", 0.0, 1000.0, 200.0) * 1e6  # Pa

# Grid setup
X, Y = np.meshgrid(np.linspace(EXTENT[0], EXTENT[1], N), np.linspace(EXTENT[2], EXTENT[3], N))
KX, KY = np.meshgrid(2*np.pi*np.fft.fftfreq(N, DX), 2*np.pi*np.fft.fftfreq(N, DX))
K2 = KX**2 + KY**2 + 1e-12  # Avoid div by zero
KAPPA_REF = kappa0  # Reference for semi-implicit

# Twin plane normal (e.g., [111] projected to 2D as [1,1]/sqrt(2))
N_TWIN = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

@njit(parallel=True)
def compute_gradients(field, dx):
    """Finite difference gradients"""
    gx = np.zeros_like(field)
    gy = np.zeros_like(field)
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            gx[i,j] = (field[i+1,j] - field[i-1,j]) / (2*dx)
            gy[i,j] = (field[i,j+1] - field[i,j-1]) / (2*dx)
    return gx, gy

@njit
def compute_h(grad_phi_mag):
    """Twin spacing h"""
    h = np.zeros_like(grad_phi_mag)
    for i in range(N):
        for j in range(N):
            if grad_phi_mag[i,j] > 1e-6:
                h[i,j] = 2 / grad_phi_mag[i,j]
            else:
                h[i,j] = np.inf  # No twin
    return h

@njit
def update_plastic_strain(sigma_eq, s_dev, sigma_y, eps_p, dt):
    """Viscoplastic update"""
    deps_p = np.zeros_like(eps_p)
    for i in range(N):
        for j in range(N):
            if sigma_eq[i,j] > sigma_y[i,j]:
                rate = GAMMA0_DOT * (sigma_eq[i,j] / sigma_y[i,j]) ** M
                deps_p[i,j] = rate * dt * (3 * s_dev[i,j] / (2 * sigma_eq[i,j]))
    return eps_p + deps_p

def compute_sigma_y(h):
    """CLS yield stress"""
    log_term = np.log(h / B)
    log_term[np.isinf(log_term)] = 0
    return SIGMA0 + (MU * B / (2 * np.pi * h * (1 - NU))) * log_term

def elastic_solver(eps_p, applied_sigma):
    """FFT elastic solver (simplified isotropic 2D plane strain)"""
    # Green's tensor for isotropic (simplified)
    # For 2D, approximate
    lambda_ = 2 * MU * NU / (1 - 2*NU)
    C11 = lambda_ + 2*MU
    C12 = lambda_
    C44 = MU

    # Polarization stress (here, no transformation strain for simplicity; add if needed)
    tau_xx = -applied_sigma  # External load as polarization
    tau_xx_h = fft2(tau_xx * np.ones((N,N)))  # Uniform applied

    # Displacement (simplified for demo; full Green's needed)
    ux_h = -1j * (tau_xx_h * KX / (C11 * K2))  # Approximate
    uy_h = np.zeros_like(ux_h)
    ux = np.real(ifft2(ux_h))
    uy = np.real(ifft2(uy_h))

    # Strains
    exx = np.real(ifft2(1j * KX * ux_h))
    eyy = np.real(ifft2(1j * KY * uy_h))
    exy = 0.5 * np.real(ifft2(1j * (KX * uy_h + KY * ux_h)))

    # Stresses (subtract plastic contribution)
    sxx = C11 * (exx - eps_p) + C12 * eyy + applied_sigma
    syy = C12 * (exx - eps_p) + C11 * eyy
    sxy = 2 * C44 * exy

    # von Mises
    sigma_eq = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - sxx)**2 + 6 * sxy**2))  # Approx szz=0
    s_dev = sxx - (sxx + syy)/3  # Deviatoric xx (2D approx)

    return sigma_eq, s_dev, sxx, syy

@njit
def kappa_phi(gx, gy, gamma_aniso, kappa0, nx_twin, ny_twin):
    """Anisotropic kappa_phi"""
    mag = np.sqrt(gx**2 + gy**2 + 1e-12)
    mx = gx / mag
    my = gy / mag
    dot = mx * nx_twin + my * ny_twin
    return kappa0 * (1 + gamma_aniso * (1 - dot**2))

def evolve_phi(phi, eta_sum, sigma_eq, dt, L_phi):
    """Semi-implicit evolution for phi"""
    # Local deriv
    df_loc_dphi = 4 * W * phi * (phi**2 - 1) * eta_sum  # From W(phi^2-1)^2 * eta_sum

    # Elastic deriv (simplified: assume df_el/dphi ~ something; for demo, 0)
    df_el_dphi = np.zeros_like(phi)

    # Gradients
    phi_gx, phi_gy = compute_gradients(phi, DX)
    grad_mag = np.sqrt(phi_gx**2 + phi_gy**2 + 1e-12)

    # Anisotropic kappa
    kappa_m = kappa_phi(phi_gx, phi_gy, gamma_aniso, kappa0, N_TWIN[0], N_TWIN[1])

    # G_aniso in real space
    g_aniso = (kappa_m - KAPPA_REF) * grad_mag**2  # Approx; full is div( (kappa- ref) grad )

    # Fourier
    phi_hat = fft2(phi)
    f_term = fft2(df_loc_dphi + df_el_dphi + g_aniso)

    # Update
    phi_hat_new = (phi_hat - dt * L_phi * f_term) / (1 + dt * L_phi * KAPPA_REF * K2)
    return np.real(ifft2(phi_hat_new))

def evolve_eta(eta, phi, dt, L_eta):
    """Isotropic evolution for eta (single grain for simplicity)"""
    # Assume one eta for now
    df_loc_deta = -alpha * eta + beta * eta**3 + 2 * gamma * eta * (1 - eta**2) + W * (phi**2 - 1)**2 * eta  # Simplified
    lap_eta = compute_laplacian(eta, DX)
    eta_new = eta + dt * L_eta * (-df_loc_deta + kappa_eta * lap_eta)
    return np.clip(eta_new, 0, 1)

@njit(parallel=True)
def compute_laplacian(field, dx):
    lap = np.zeros_like(field)
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap[i,j] = (field[i+1,j] + field[i-1,j] + field[i,j+1] + field[i,j-1] - 4*field[i,j]) / (dx**2)
    return lap

# Initialization
def init_fields():
    phi = np.tanh(Y / 2)  # Simple horizontal twin band
    eta = np.ones((N,N))  # Single grain
    eps_p = np.zeros((N,N))  # Plastic strain (scalar approx for exx)
    return phi, eta, eps_p

# Run simulation
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        phi, eta, eps_p = init_fields()
        history = []

        for step in range(steps):
            # Step 1: Gradients and h
            phi_gx, phi_gy = compute_gradients(phi, DX)
            grad_phi_mag = np.sqrt(phi_gx**2 + phi_gy**2)
            h = compute_h(grad_phi_mag)

            # Step 2: Yield stress and plastic update
            sigma_y = compute_sigma_y(h)
            sigma_eq, s_dev, sxx, syy = elastic_solver(eps_p, applied_stress)
            eps_p = update_plastic_strain(sigma_eq, s_dev, sigma_y, eps_p, DT)

            # Step 3: Evolve fields
            eta_sum = eta**2  # For single eta
            phi = evolve_phi(phi, eta_sum, sigma_eq, DT, L_phi)
            eta = evolve_eta(eta, phi, DT, L_eta)

            if step % 10 == 0:
                history.append((phi.copy(), eta.copy(), sigma_eq.copy(), h.copy()))

    st.success(f"Simulation complete! {len(history)} frames saved.")

    # Visualization
    frame = st.slider("Frame", 0, len(history)-1, len(history)-1)
    phi_f, eta_f, sigma_eq_f, h_f = history[frame]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axs[0].imshow(phi_f, extent=EXTENT, cmap='coolwarm')
    axs[0].set_title("Twin Field φ")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(eta_f, extent=EXTENT, cmap='viridis')
    axs[1].set_title("Grain Field η")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(sigma_eq_f / 1e9, extent=EXTENT, cmap='hot')
    axs[2].set_title("von Mises Stress (GPa)")
    plt.colorbar(im2, ax=axs[2])

    im3 = axs[3].imshow(h_f, extent=EXTENT, cmap='plasma', vmin=0, vmax=10)
    axs[3].set_title("Twin Spacing h (nm)")
    plt.colorbar(im3, ax=axs[3])

    st.pyplot(fig)

# Export
if st.sidebar.button("Export Data"):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i, (phi, eta, sigma, h) in enumerate(history):
            np.savez(BytesIO(), phi=phi, eta=eta, sigma=sigma, h=h)
    buffer.seek(0)
    st.sidebar.download_button("Download ZIP", buffer, "nt_cu_simulation.zip")
