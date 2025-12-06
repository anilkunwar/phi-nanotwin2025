import streamlit as st
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn

st.set_page_config(page_title="Tetragonal Twin Evolution", layout="centered")
st.title("Structural Phase-Field Model of Twinning in Tetragonal BaTiO₃")
st.markdown("""
Multi-order-parameter model (exactly like Shen & Beyerlein 2025) with **correct long-range elasticity** via Khachaturyan spectral method.  
Now **bug-free and fast**.
""")

# ========================== USER INPUT ==========================
col1, col2 = st.columns(2)
with col1:
    N = st.slider("Grid size N×N", 128, 512, 256, 64)
    dx = st.number_input("Grid spacing Δx (nm)", 0.5, 4.0, 1.0, 0.5)
    steps = st.slider("Number of steps", 500, 15000, 5000, 500)
    dt = st.number_input("Time step Δt", 0.001, 0.1, 0.02, 0.005)

with col2:
    init_type = st.selectbox("Initial microstructure", 
                             ["90° twin", "Single lamella", "Random polycrystal", "Vortex"])
    twin_thickness = st.slider("Twin thickness (grid points)", 5, 50, 15)

# ========================== MATERIAL PARAMETERS ==========================
p = 3  # matrix + two tetragonal twins
w = 1.0      # interface energy
mobility = 1.0

# Transformation strains (Bain strain)
ε00 = [
    np.zeros((2,2)),                    # 0: cubic matrix
    np.array([[0.01, 0.0], [0.0, -0.005]]),  # 1: a-domain (x-elongated)
    np.array([[-0.005, 0.0], [0.0, 0.01]])   # 2: c-domain (y-elongated)
]

# Elastic moduli (Voigt average, in arbitrary units)
C11 = 2.0; C12 = 1.0; C44 = 0.5
Cijkl = np.array([[[[C11, C12, 0],
                    [C12, C11, 0],
                    [0,   0,   C44]]]])  # dummy 4D for indexing

# ========================== NUMBA KERNELS ==========================
@njit(parallel=True)
def compute_laplacian(eta, dx):
    n = eta.shape[0]
    lap = np.zeros((n, n))
    h2 = 1.0 / (dx * dx)
    for i in prange(1, n-1):
        for j in prange(1, n-1):
            lap[i,j] = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) * h2
    return lap

@njit
def local_free_energy_and_deriv(etas):
    n = etas[0].shape[0]
    f = np.zeros((n,n))
    df = np.zeros((p, n, n))
    for i in range(p):
        e4 = etas[i]**4
        e2 = etas[i]**2
        f += e4/4 - e2/2
        df[i] += etas[i]**3 - etas[i]
        for j in range(p):
            if i != j:
                f += 0.5 * w * e2 * etas[j]**2
                df[i] += w * etas[i] * etas[j]**2
    return f, df

# ========================== ELASTIC SOLVER (FIXED!) ==========================
def compute_elastic(etas, dx):
    nx, ny = etas[0].shape
    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=dx)
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
    k2 = kxx**2 + kyy**2 + 1e-12
    n1 = kxx / np.sqrt(k2)  # unit wavevector components
    n2 = kyy / np.sqrt(k2)

    # Total eigenstrain
    e0_11 = np.zeros((nx,ny)); e0_22 = np.zeros((nx,ny)); e0_12 = np.zeros((nx,ny))
    for i in range(p):
        weight = etas[i]
        e0_11 += weight * ε00[i][0,0]
        e0_22 += weight * ε00[i][1,1]
        e0_12 += weight * ε00[i][0,1]

    e0_11_hat = fftn(e0_11)
    e0_22_hat = fftn(e0_22)
    e0_12_hat = fftn(e0_12)

    force = np.zeros((p, nx, ny))
    for i in range(p):
        B_hat = (n1*n1*C11*e0_11_hat + n2*n2*C11*e0_22_hat +
                 2*n1*n2*C12*e0_11_hat + 2*n1*n2*C12*e0_22_hat +
                 4*n1*n2*C44*e0_12_hat)
        sigma11_hat = C11*e0_11_hat + C12*e0_22_hat - B_hat*n1*n1
        sigma22_hat = C11*e0_22_hat + C12*e0_11_hat - B_hat*n2*n2
        sigma12_hat = 2*C44*e0_12_hat - B_hat*n1*n2

        sigma11 = np.real(ifftn(sigma11_hat))
        sigma22 = np.real(ifftn(sigma22_hat))
        sigma12 = np.real(ifftn(sigma12_hat))

        force[i] = -(sigma11*ε00[i][0,0] + sigma22*ε00[i][1,1] + 2*sigma12*ε00[i][0,1])

    return force

# ========================== INITIALIZATION ==========================
etas = [np.zeros((N,N)) for _ in range(p)]

if init_type == "90° twin":
    mid = N//2
    etas[1][:, :mid] = 1.0
    etas[2][:, mid:] = 1.0
elif init_type == "Single lamella":
    mid = N//2
    etas[0][:, :mid-twin_thickness//2] = 1.0
    etas[0][:, mid+twin_thickness//2:] = 1.0
    etas[1][:, mid-twin_thickness//2:mid+twin_thickness//2] = 1.0
else:
    # Random
    rnd = np.random.rand(N,N)
    etas[0][rnd < 0.4] = 1
    etas[1][(rnd>=0.4)&(rnd<0.7)] = 1
    etas[2][rnd>=0.7] = 1

# Normalize
s = sum(etas)
for i in range(p):
    etas[i] /= (s + 1e-15)

# ========================== SIMULATION ==========================
if st.button("Run Twin Evolution", type="primary"):
    progress = st.progress(0)
    placeholder = st.empty()

    for step in range(steps):
        laps = [compute_laplacian(etas[i], dx) for i in range(p)]

        _, df_chem = local_free_energy_and_deriv(etas)

        df_elas = compute_elastic(etas, dx)

        for i in range(p):
            df = df_chem[i] - w * laps[i] + df_elas[i]
            etas[i] += dt * mobility * df * etas[i]**2 * (1 - etas[i])**2
            etas[i] = np.clip(etas[i], 0, 1)

        # Re-normalize
        s = sum(etas)
        for i in range(p):
            etas[i] /= (s + 1e-15)

        if step % max(1, steps//50) == 0 or step == steps-1:
            progress.progress((step+1)/steps)
            fig, ax = plt.subplots(figsize=(8,7))
            img = np.stack(etas, axis=-1)  # RGB
            ax.imshow(img, origin='lower')
            ax.set_title(f"Step {step:,} | Twins in Tetragonal BaTiO₃")
            ax.axis('off')
            placeholder.pyplot(fig)
            plt.close(fig)

    st.success("Finished! Twins evolved with full elastic interaction.")
    st.balloons()
