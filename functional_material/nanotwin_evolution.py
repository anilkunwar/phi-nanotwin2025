import streamlit as st
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
import pandas as pd
from io import BytesIO
import zipfile
from datetime import datetime

st.set_page_config(page_title="Nanotwin Evolution in Ag FCC", layout="wide")
st.title("Simplified Nanotwin Evolution in Ag FCC Matrix")
st.markdown("**Structural multi-order-parameter phase-field model+ long-range elasticity**")

# ==================================== USER INPUT ====================================
col1, col2 = st.columns(2)

with col1:
    N = st.slider("Grid size", 128, 512, 256, 64)
    dx = st.number_input("Grid spacing (nm)", 0.5, 3.0, 1.0, 0.5)
    steps = st.slider("Evolution steps", 1000, 20000, 8000, 500)
    dt = st.number_input("Time step", 0.005, 0.1, 0.02, 0.005)

with col2:
    T = st.slider("Temperature (K)", 300, 1000, 700)
    init = st.selectbox("Initial twin structure", 
                       ["Single twin lamella", "Multiple twins", "Random"])
    twin_width_nm = st.slider("Twin thickness (nm)", 5, 50, 15)

# ==================================== MATERIAL PARAMETERS (Ag FCC) ====================================
p = 2  # 0: matrix, 1: twin
w = 1.5          # interface energy coefficient
kappa = 0.5      # gradient coefficient (isotropic, no inclination dependence)
mobility = 1.178e-5 * np.exp(-0.155 / (8.617e-5 * T))  # MD-derived Arrhenius for ITB mobility

# Eigenstrains for FCC twin (shear-based)
ε00 = [
    np.zeros((2,2)),                                        # matrix
    np.array([[ 0.0, 0.707], [0.707,  0.0]]) / np.sqrt(2)   # twin (simplified {111}<112> shear)
]

# Elastic constants for Ag (GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1

# ==================================== NUMBA KERNELS ====================================
@njit(parallel=True)
def compute_laplacian(eta, dx):
    n = eta.shape[0]
    lap = np.zeros((n,n))
    inv_dx2 = 1.0 / (dx * dx)
    for i in prange(1, n-1):
        for j in prange(1, n-1):
            lap[i,j] = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) * inv_dx2
    return lap

@njit
def local_free_energy_deriv(etas):
    df = np.zeros((p, etas[0].shape[0], etas[0].shape[1]))
    for i in range(p):
        e2 = etas[i]**2
        e4 = e2 * e2
        df[i] += etas[i]**3 - etas[i] + 2 * etas[i] * sum(w * etas[j]**2 for j in range(p) if j != i)
    return df

# ==================================== ELASTIC SOLVER ====================================
def compute_stress_strain(etas, dx):
    nx = etas[0].shape[0]
    kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(nx, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0,0] = 1e-12

    # Total eigenstrain
    e11 = np.zeros((nx,nx))
    e22 = np.zeros((nx,nx))
    e12 = np.zeros((nx,nx))
    for i in range(p):
        e11 += etas[i] * ε00[i][0,0]
        e22 += etas[i] * ε00[i][1,1]
        e12 += etas[i] * ε00[i][0,1]

    e11h = fftn(e11)
    e22h = fftn(e22)
    e12h = fftn(e12)

    # Plane-strain reduced constants
    C11p = C11 - C12**2 / C11
    C12p = C12 - C12**2 / C11
    C44p = C44

    n1 = KX / np.sqrt(K2)
    n2 = KY / np.sqrt(K2)

    # Stress in Fourier space (simplified isotropic)
    sigma11h = C11p * e11h + C12p * e22h - (n1 * (C11p * n1 * e11h + C12p * n1 * e22h + 2*C44p * n2 * e12h) + n2 * (C12p * n2 * e11h + C11p * n2 * e22h + 2*C44p * n1 * e12h))
    sigma22h = C11p * e22h + C12p * e11h - (n2 * (C11p * n2 * e22h + C12p * n2 * e11h + 2*C44p * n1 * e12h) + n1 * (C12p * n1 * e22h + C11p * n1 * e11h + 2*C44p * n2 * e12h))
    sigma12h = 2*C44p * e12h - (n1 * (2*C44p * n2 * e12h) + n2 * (2*C44p * n1 * e12h))

    s11 = np.real(ifftn(sigma11h))
    s22 = np.real(ifftn(sigma22h))
    s12 = np.real(ifftn(sigma12h))

    hydro = (s11 + s22) / 3
    vm = np.sqrt(0.5 * ((s11 - s22)**2 + (s11 - hydro*3)**2 + (s22 - hydro*3)**2 + 6*s12**2))

    return s11, s22, s12, hydro, vm

# ==================================== INITIALIZATION ====================================
def init_structure(choice):
    etas = [np.zeros((N,N)) for _ in range(p)]
    mid = N // 2
    tw = int(twin_width_nm / dx)

    if choice == "Single twin lamella":
        etas[0][:] = 1.0
        etas[1][:, mid-tw//2:mid+tw//2] = 1.0
    elif choice == "Multiple twins":
        etas[0][:] = 1.0
        etas[1][:, mid-2*tw//2:mid-tw//2] = 1.0
        etas[1][:, mid+tw//2:mid+2*tw//2] = 1.0
    else:  # Random
        rnd = np.random.rand(N,N)
        etas[0][rnd < 0.5] = 1.0
        etas[1][rnd >= 0.5] = 1.0

    total = np.sum(etas, axis=0)
    for i in range(p):
        etas[i] /= (total + 1e-12)
    return etas

# ==================================== RUN SIMULATION ====================================
if st.button("Run Nanotwin Evolution + Stress Analysis", type="primary"):
    etas = init_structure(init)

    progress = st.progress(0)
    placeholder = st.empty()

    for step in range(steps + 1):
        if step > 0:
            laps = [compute_laplacian(etas[i], dx) for i in range(p)]
            df_chem = local_free_energy_deriv(etas)
            s11, s22, s12, hydro, vm = compute_stress_strain(etas, dx)

            df_elas = np.zeros((p, N, N))
            for i in range(p):
                df_elas[i] = -(s11 * ε00[i][0,0] + s22 * ε00[i][1,1] + 2 * s12 * ε00[i][0,1])

            for i in range(p):
                df = df_chem[i] - kappa * laps[i] + df_elas[i]
                etas[i] += dt * mobility * df * etas[i]**2 * (1 - etas[i])**2
                etas[i] = np.clip(etas[i], 0, 1)

            total = np.sum(etas, axis=0)
            for i in range(p):
                etas[i] /= (total + 1e-12)

        if step % max(1, steps//50) == 0 or step == steps:
            progress.progress((step + 1) / (steps + 1))

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

            # Twins (gray for matrix, red for twin)
            ax1 = axes[0]
            img = np.zeros((N,N))
            img += 0.5 * etas[0] + etas[1]
            ax1.imshow(img, cmap='Reds', origin='lower', extent=extent, vmin=0, vmax=1)
            ax1.set_title(f"Twin Structure (Step {step})")
            ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")

            # von Mises
            ax2 = axes[1]
            ax2.imshow(vm, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=np.percentile(vm, 99))
            ax2.set_title("von Mises Stress (GPa)")

            # Hydrostatic
            ax3 = axes[2]
            ax3.imshow(hydro, cmap='coolwarm', origin='lower', extent=extent, symmetric=True)
            ax3.set_title("Hydrostatic Stress (GPa)")

            placeholder.pyplot(fig)
            plt.close(fig)

    st.success("Simplified nanotwin evolution complete!")

    # Final downloads
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = plt.figure()
        plt.imshow(np.zeros((N,N)) + 0.5 * etas[0] + etas[1], cmap='Reds', vmin=0, vmax=1)
        plt.title("Final Twins")
        st.pyplot(fig1)
        buf = BytesIO()
        fig1.savefig(buf, format='png')
        buf.seek(0)
        st.download_button("Download Twins PNG", buf, "twins.png")

    with col2:
        fig2 = plt.figure()
        plt.imshow(vm, cmap='hot')
        plt.title("Final von Mises")
        st.pyplot(fig2)
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        st.download_button("Download von Mises PNG", buf2, "vm.png")

    with col3:
        fig3 = plt.figure()
        plt.imshow(hydro, cmap='coolwarm')
        plt.title("Final Hydrostatic")
        st.pyplot(fig3)
        buf3 = BytesIO()
        fig3.savefig(buf3, format='png')
        buf3.seek(0)
        st.download_button("Download Hydrostatic PNG", buf3, "hydro.png")

    # ZIP export
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i in range(p):
            pd.DataFrame(etas[i]).to_csv(BytesIO(), index=False)
            zf.writestr(f"eta_{i}.csv", BytesIO(pd.DataFrame(etas[i]).to_csv(index=False)).getvalue())
        zf.writestr("von_mises.csv", BytesIO(pd.DataFrame(vm).to_csv(index=False)).getvalue())
        zf.writestr("hydrostatic.csv", BytesIO(pd.DataFrame(hydro).to_csv(index=False)).getvalue())
    buffer.seek(0)
    st.download_button("Download All Data ZIP", buffer, "nanotwin_ag.zip")
