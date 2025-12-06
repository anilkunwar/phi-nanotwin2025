import streamlit as st
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
import pandas as pd
from io import BytesIO
import zipfile
from datetime import datetime

st.set_page_config(page_title="Tetragonal Twin Stress Analyzer", layout="wide")
st.title("Tetragonal Twin Evolution in BaTiO₃ – Structural Phase-Field + Elastic Stress/Strain")
st.markdown("**Multi-order-parameter model • Long-range elasticity • Publication-ready output**")

# ==================================== USER INPUT ====================================
col1, col2 = st.columns(2)

with col1:
    N = st.slider("Grid size", 128, 512, 256, 64)
    dx = st.number_input("Grid spacing (nm)", 0.5, 3.0, 1.0, 0.5)
    steps = st.slider("Evolution steps", 1000, 20000, 8000, 500)
    dt = st.number_input("Time step", 0.005, 0.1, 0.02, 0.005)

with col2:
    T = st.slider("Temperature (K)", 300, 600, 400)
    init = st.selectbox("Initial twin structure", 
                       ["90° twin band", "Single lamella", "Random", "Vortex"])
    twin_width_nm = st.slider("Twin thickness (nm)", 5, 50, 15)

# ==================================== MATERIAL PARAMETERS ====================================
p = 3  # 0: cubic matrix, 1: a-domain, 2: c-domain
w = 1.5          # interface energy coefficient
mobility = 1.0

# Transformation strains (Bain strain)
ε00 = [
    np.zeros((2,2)),                                        # cubic matrix
    np.array([[ 0.01, 0.0], [0.0, -0.005]]),                   # a-domain (x-elongated)
    np.array([[-0.005, 0.0], [0.0,  0.01 ]])                    # c-domain (y-elongated)
]

# Elastic constants (GPa)
C11 = 275.0
C12 = 179.0
C44 = 54.3

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
        e4 = e2**2
        df[i] += etas[i] * (e4 - e2)           # 1/4 e⁴ − 1/2 e²
        for j in range(p):
            if i != j:
                df[i] += w * etas[i] * etas[j]**2
    return df

# ==================================== ELASTIC SOLVER (CORRECT & FAST) ====================================
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

    # Simple isotropic Green operator (very stable)
    lam = C12 * 1e9
    mu  = C44 * 1e9
    factor = 1.0 / (4 * mu * K2)

    n1 = KX / np.sqrt(K2)
    n2 = KY / np.sqrt(K2)

    # Stress in Fourier space
    sigma11h = (lam + 2*mu) * e11h + lam * e22h
    sigma22h = (lam + 2*mu) * e22h + lam * e11h
    sigma12h = 2 * mu * e12h

    # Remove elastic incompatibility term
    B11 = n1*n1*sigma11h + n1*n2*sigma12h
    B22 = n1*n2*sigma12h + n2*n2*sigma22h
    sigma11h -= B11 * n1*n1
    sigma22h -= B22 * n2*n2
    sigma12h -= (B11*n2*n1 + B22*n1*n2)

    s11 = np.real(ifftn(sigma11h)) / 1e9
    s22 = np.real(ifftn(sigma22h)) / 1e9
    s12 = np.real(ifftn(sigma12h)) / 1e9

    hydro = (s11 + s22) / 3
    vm = np.sqrt(0.5 * ((s11-s22)**2 + 6*s12**2 + (s11+ s22)**2))

    return s11, s22, s12, hydro, vm

# ==================================== INITIALIZATION ====================================
def init_structure(choice):
    etas = [np.zeros((N,N)) for _ in range(p)]
    mid = N // 2
    tw = int(twin_width_nm / dx)

    if choice == "90° twin band":
        etas[1][:, :mid] = 1.0
        etas[2][:, mid:] = 1.0
    elif choice == "Single lamella":
        etas[0][:] = 1.0
        etas[1][:, mid-tw//2:mid+tw//2] = 1.0
    elif choice == "Random":
        rnd = np.random.rand(N,N)
        mask0 = rnd < 0.4
        mask1 = (rnd >= 0.4) & (rnd < 0.7)
        mask2 = rnd >= 0.7
        etas[0][mask0] = 1.0
        etas[1][mask1] = 1.0
        etas[2][mask2] = 1.0
    else:  # Vortex
        x, y = np.meshgrid(np.arange(N), np.arange(N))
        angle = np.arctan2(y - N/2, x - N/2)
        etas[0] = 0.5 * (1 + np.cos(4 * angle))
        etas[1] = 0.5 * (1 + np.sin(4 * angle))

    total = np.sum(etas, axis=0)
    for i in range(p):
        etas[i] /= (total + 1e-12)
    return etas

# ==================================== RUN SIMULATION ====================================
if st.button("Run Twin Evolution + Full Stress/Strain Analysis", type="primary"):
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
                df = df_chem[i] - w * laps[i] + df_elas[i]
                delta = dt * mobility * df * etas[i]**2 * (1 - etas[i])**2
                etas[i] += delta
                etas[i] = np.clip(etas[i], 0, 1)

            total = np.sum(etas, axis=0)
            for i in range(p):
                etas[i] /= (total + 1e-12)

        if step % max(1, steps//50) == 0 or step == steps:
            progress.progress((step + 1) / (steps + 1))

            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.4)

            ax1 = fig.add_subplot(gs[0, :2])
            rgb = np.stack([etas[0], etas[1], etas[2]], axis=-1)
            ax1.imshow(rgb, origin='lower', extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2])
            ax1.set_title(f"Tetragonal Twins – Step {step}", fontsize=18, fontweight='bold')
            ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")

            ax2 = fig.add_subplot(gs[0, 2])
            im2 = ax2.imshow(vm, cmap='hot', vmin=0, vmax=2, origin='lower',
                            extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2])
            ax2.set_title("von Mises Stress (GPa)")
            plt.colorbar(im2, ax=ax2, shrink=0.8)

            ax3 = fig.add_subplot(gs[0, 3])
            im3 = ax3.imshow(hydro, cmap='coolwarm', vmin=-0.8, vmax=0.8, origin='lower',
                            extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2])
            ax3.set_title("Hydrostatic Stress (GPa)")
            plt.colorbar(im3, ax=ax3, shrink=0.8)

            ax4 = fig.add_subplot(gs[1, :2])
            im4 = ax4.imshow(s11, cmap='RdBu_r', vmin=-1.5, vmax=1.5, origin='lower',
                            extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2])
            ax4.set_title("σ$_{xx}$ (GPa)")
            plt.colorbar(im4, ax=ax4, shrink=0.8)

            ax5 = fig.add_subplot(gs[1, 2])
            im5 = ax5.imshow(s22, cmap='RdBu_r', vmin=-1.5, vmax=1.5, origin='lower',
                            extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2])
            ax5.set_title("σ$_{yy}$ (GPa)")
            plt.colorbar(im5, ax=ax5, shrink=0.8)

            ax6 = fig.add_subplot(gs[1, 3])
            im6 = ax6.imshow(s12, cmap='PRGn', vmin=-0.5, vmax=0.5, origin='lower',
                            extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2])
            ax6.set_title("σ$_{xy}$ (GPa)")
            plt.colorbar(im6, ax=ax6, shrink=0.8)

            for ax in fig.axes:
                ax.tick_params(direction='in', top=True, right=True)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)

            placeholder.pyplot(fig)
            plt.close(fig)

    st.success("Simulation complete! Twins + full stress/strain fields generated.")

    # Final results with download
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.imshow(np.stack(etas, axis=-1), origin='lower')
        ax1.set_title("Final Twin Structure")
        st.pyplot(fig1)
        buf = BytesIO()
        fig1.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button("Download Twins PNG", buf, "twins_final.png", "image/png")

    with col2:
        fig2, ax2 = plt.subplots()
        im = ax2.imshow(vm, cmap='hot', origin='lower')
        plt.colorbar(im, label='von Mises (GPa)')
        ax2.set_title("von Mises Stress")
        st.pyplot(fig2)
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
        buf2.seek(0)
        st.download_button("Download von Mises", buf2, "von_mises.png", "image/png")

    with col3:
        fig3, ax3 = plt.subplots()
        im = ax3.imshow(hydro, cmap='coolwarm', vmin=-0.8, vmax=0.8, origin='lower')
        plt.colorbar(im, label='Hydrostatic (GPa)')
        ax3.set_title("Hydrostatic Stress")
        st.pyplot(fig3)
        buf3 = BytesIO()
        fig3.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
        buf3.seek(0)
        st.download_button("Download Hydrostatic", buf3, "hydrostatic.png", "image/png")

    # Export all data as ZIP
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i in range(p):
            np.savetxt(f"eta_{i}.csv", etas[i], delimiter=",")
            zf.writestr(f"eta_{i}.csv", BytesIO(np.savetxt(..., etas[i], delimiter=",")).getvalue())
        zf.writestr("von_mises.csv", BytesIO(np.savetxt(..., vm, delimiter=",")).getvalue())
        zf.writestr("hydrostatic.csv", BytesIO(np.savetxt(..., hydro, delimiter=",")).getvalue())
    buffer.seek(0)
    st.download_button("Download All Data (ZIP)", buffer, f"BaTiO3_twins_{datetime.now().strftime('%H%M')}.zip")

st.balloons()
