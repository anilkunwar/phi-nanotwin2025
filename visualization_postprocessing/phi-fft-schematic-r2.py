import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Phase-Field FFT Schematic",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Phase-Field FFT Spectral Method Schematic")
st.markdown(
    """
This interactive visualization demonstrates the decomposition of a real-space spatial field $\phi(\mathbf{r})$ 
into constituent Fourier modes in the wavenumber domain $\hat{\phi}(\mathbf{k})$.

**Key improvement**: The Fourier domain now correctly shows a **spectrum** (amplitude vs. wavenumber) 
with discrete peaks, rather than oscillating waves.
"""
)

# -------------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------------
st.sidebar.header("Signal Parameters")

f1 = st.sidebar.slider("Wavenumber $k_1$", 0.1, 3.0, 0.5, 0.1)
amp1 = st.sidebar.slider("Amplitude $A_1$", 0.1, 2.0, 1.0, 0.1)

f2 = st.sidebar.slider("Wavenumber $k_2$", 0.1, 3.0, 1.5, 0.1)
amp2 = st.sidebar.slider("Amplitude $A_2$", 0.1, 2.0, 0.5, 0.1)

st.sidebar.subheader("3D View Angle")
elev = st.sidebar.slider("Elevation", 0, 90, 15, 5)
azim = st.sidebar.slider("Azimuth", -180, 180, -60, 5)

# -------------------------------------------------------------------------
# Figure Generation
# -------------------------------------------------------------------------
plt.style.use("default")
fig = plt.figure(figsize=(14, 7), facecolor="white")
ax = fig.add_subplot(111, projection="3d")

# Spatial domain
x = np.linspace(0, 10, 500)

# Signal components
s1 = amp1 * np.sin(2 * np.pi * f1 * x)
s2 = amp2 * np.sin(2 * np.pi * f2 * x)
s_total = s1 + s2

# Colors
c_total = "#1f77b4"  # Blue
c_c1 = "#ff7f0e"     # Orange
c_c2 = "#2ca02c"     # Green

# =========================================================================
# 1. REAL SPACE DOMAIN (front plane at y = 0)
# =========================================================================
y_real = 0

# Draw the real-space plane
xx_plane, zz_plane = np.meshgrid([0, 10], [-3, 3])
yy_plane = np.full_like(xx_plane, y_real)
ax.plot_surface(xx_plane, yy_plane, zz_plane, color="#e6f2ff", alpha=0.3)

# Combined field φ(r) - solid thick line
ax.plot(x, np.full_like(x, y_real), s_total,
        color=c_total, linewidth=2.5, label=r"$\phi(\mathbf{r})$")

# Component modes - dashed thinner lines (superimposed, NOT separated)
ax.plot(x, np.full_like(x, y_real), s1,
        color=c_c1, linewidth=1.5, linestyle="--", alpha=0.7,
        label=r"Mode 1: $A_1\sin(k_1 r)$")
ax.plot(x, np.full_like(x, y_real), s2,
        color=c_c2, linewidth=1.5, linestyle="--", alpha=0.7,
        label=r"Mode 2: $A_2\sin(k_2 r)$")

# =========================================================================
# 2. FOURIER / WAVENUMBER DOMAIN (back plane at x = 10)
# =========================================================================
x_fourier = 10
k_max = max(f1, f2) + 1.0

# Draw the Fourier plane
y_plane_vals = np.linspace(0, k_max, 2)
z_plane_vals = np.linspace(-0.5, max(amp1, amp2) + 0.5, 2)
Y_p, Z_p = np.meshgrid(y_plane_vals, z_plane_vals)
X_p = np.full_like(Y_p, x_fourier)
ax.plot_surface(X_p, Y_p, Z_p, color="#ffe6e6", alpha=0.4)

# Baseline (k-axis) in Fourier domain
ax.plot([x_fourier, x_fourier], [0, k_max], [0, 0],
        color="black", linewidth=1.5)

# SPECTRAL PEAKS (stem plot) - THIS IS THE KEY FIX
# Peak at k1
ax.plot([x_fourier, x_fourier], [f1, f1], [0, amp1],
        color=c_c1, linewidth=3.5)
ax.scatter([x_fourier], [f1], [amp1], color=c_c1, s=80, zorder=5,
           edgecolors="black", linewidth=0.5)

# Peak at k2
ax.plot([x_fourier, x_fourier], [f2, f2], [0, amp2],
        color=c_c2, linewidth=3.5)
ax.scatter([x_fourier], [f2], [amp2], color=c_c2, s=80, zorder=5,
           edgecolors="black", linewidth=0.5)

# =========================================================================
# 3. PROJECTION / TRANSFORMATION LINES
# =========================================================================
# Dotted lines showing the mapping from spatial modes to spectral peaks
ax.plot([10, x_fourier], [0, f1], [0, 0],
        color=c_c1, linestyle=":", linewidth=1.5, alpha=0.6)
ax.plot([10, x_fourier], [0, f2], [0, 0],
        color=c_c2, linestyle=":", linewidth=1.5, alpha=0.6)

# FT arrow (curved path in 3D)
arrow_x = np.linspace(10.5, 10.5, 20)
arrow_y = np.linspace(0.5, 1.5, 20)
arrow_z = np.linspace(0, 0, 20)
# Simplified: just annotate with text

# =========================================================================
# 4. ANNOTATIONS
# =========================================================================
ax.set_axis_off()
ax.view_init(elev=elev, azim=azim)

# Real space label
ax.text(5, -0.8, -2.8,
        "Real Space Domain\n" + r"$\phi(\mathbf{r}) = A_1\sin(k_1 r) + A_2\sin(k_2 r)$",
        fontsize=10, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round", facecolor="#e6f2ff", edgecolor="#1f77b4", alpha=0.8))

# Fourier domain label
ax.text(x_fourier + 0.5, k_max/2, -0.8,
        "Fourier / Wavenumber Domain\n" + r"$\hat{\phi}(\mathbf{k})$",
        fontsize=10, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round", facecolor="#ffe6e6", edgecolor="#d62728", alpha=0.8))

# FFT operation annotation
fig.text(0.5, 0.08,
         "Fast Fourier Transform (FFT)\n"
         + r"$\nabla^2 \phi(\mathbf{r}) \longrightarrow -k^2 \hat{\phi}(\mathbf{k})$",
         fontsize=12, fontweight="bold", color="#a00000", ha="center",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0f0", edgecolor="#d62728"))

# Legend
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
st.pyplot(fig)
