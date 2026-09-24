import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Set Streamlit page configuration
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
"""
)

# -------------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------------
st.sidebar.header("Plot & Signal Parameters")

f1 = st.sidebar.slider("Frequency 1 ($f_1$)", 0.1, 3.0, 0.5, 0.1)
amp1 = st.sidebar.slider("Amplitude 1", 0.1, 2.0, 1.0, 0.1)

f2 = st.sidebar.slider("Frequency 2 ($f_2$)", 0.1, 3.0, 1.5, 0.1)
amp2 = st.sidebar.slider("Amplitude 2", 0.1, 2.0, 0.5, 0.1)

st.sidebar.subheader("3D View Angle")
elev = st.sidebar.slider("Elevation Angle", 0, 90, 20, 5)
azim = st.sidebar.slider("Azimuth Angle", -180, 180, -55, 5)

# -------------------------------------------------------------------------
# Figure Generation
# -------------------------------------------------------------------------
plt.style.use("default")
fig = plt.figure(figsize=(12, 6), facecolor="white")
ax = fig.add_subplot(111, projection="3d")

# Grid parameters
t = np.linspace(0, 10, 500)

# Signal Components
s1 = amp1 * np.sin(2 * np.pi * f1 * t)
s2 = amp2 * np.sin(2 * np.pi * f2 * t)
s_total = s1 + s2

# Colors
c_total = "#1f77b4"  # Dark Blue
c_c1 = "#ff7f0e"  # Orange
c_c2 = "#2ca02c"  # Green
c_fft = "#d62728"  # Red/Purple

# 1. Real Space / Time Domain Plane (Front Plane: y = 0)
y_real = 0
ax.plot_surface(
    np.array([[0, 10], [0, 10]]),
    np.array([[y_real, y_real], [y_real, y_real]]),
    np.array([[-2.5, -2.5], [2.5, 2.5]]),
    color="#e6f2ff",
    alpha=0.4,
)

# Combined field profile
ax.plot(
    t,
    np.full_like(t, y_real),
    s_total,
    color=c_total,
    linewidth=2.5,
    label=r"Combined Field $\phi(\mathbf{r})$",
)

# Underlying modes
ax.plot(
    t,
    np.full_like(t, y_real),
    s1,
    color=c_c1,
    linewidth=1.0,
    linestyle="--",
    alpha=0.7,
)
ax.plot(
    t,
    np.full_like(t, y_real),
    s2,
    color=c_c2,
    linewidth=1.0,
    linestyle="--",
    alpha=0.7,
)

# 2. Decomposed Wave Modes in Frequency/Wavenumber Space
y_k1 = 2.0
y_k2 = 4.0

# Wave Mode 1
ax.plot(t, np.full_like(t, y_k1), s1, color=c_c1, linewidth=2.0)
ax.plot(
    [0, 10], [y_k1, y_k1], [0, 0], color="gray", linestyle=":", linewidth=0.8
)

# Wave Mode 2
ax.plot(t, np.full_like(t, y_k2), s2, color=c_c2, linewidth=2.0)
ax.plot(
    [0, 10], [y_k2, y_k2], [0, 0], color="gray", linestyle=":", linewidth=0.8
)

# Connecting dashed projection lines
ax.plot([10, 10], [0, y_k1], [0, 0], color=c_c1, linestyle="--", linewidth=1.2)
ax.plot([10, 10], [0, y_k2], [0, 0], color=c_c2, linestyle="--", linewidth=1.2)

# 3. Fourier / Wavenumber Domain Plane (Back Plane: x = 10)
x_fourier = 10

# Back plane boundary
y_plane = np.linspace(0, 5, 2)
z_plane = np.linspace(-0.2, max(amp1, amp2) + 0.8, 2)
Y_p, Z_p = np.meshgrid(y_plane, z_plane)
X_p = np.full_like(Y_p, x_fourier)
ax.plot_surface(X_p, Y_p, Z_p, color="#ffe6e6", alpha=0.4)

# Baseline in Fourier Domain
ax.plot([x_fourier, x_fourier], [0, 5], [0, 0], color="black", linewidth=1.5)

# Discrete Spectral Peaks at k1 and k2
ax.plot(
    [x_fourier, x_fourier], [y_k1, y_k1], [0, amp1], color=c_c1, linewidth=3
)
ax.scatter([x_fourier], [y_k1], [amp1], color=c_c1, s=40)

ax.plot(
    [x_fourier, x_fourier], [y_k2, y_k2], [0, amp2], color=c_c2, linewidth=3
)
ax.scatter([x_fourier], [y_k2], [amp2], color=c_c2, s=40)

# 4. Annotations & Formatting
ax.set_axis_off()
ax.view_init(elev=elev, azim=azim)

# Labels
ax.text(
    5,
    -0.5,
    -2.0,
    "Real Space Domain\n$\phi(\mathbf{r})$",
    fontsize=11,
    fontweight="bold",
    ha="center",
)
ax.text(
    10.2,
    2.5,
    -0.6,
    "Fourier / Wavenumber Domain\n$\hat{\phi}(\mathbf{k}) = \mathcal{F}\{\phi(\mathbf{r})\}$",
    fontsize=11,
    fontweight="bold",
    ha="center",
)

# Mathematical operation annotation box
fig.text(
    0.51,
    0.12,
    "Fast Fourier Transform (FFT)\n"
    r"$\nabla^2 \phi(\mathbf{r}) \xrightarrow{\quad\mathcal{F}\quad} -k^2 \hat{\phi}(\mathbf{k})$",
    fontsize=12,
    fontweight="bold",
    color="#a00000",
    ha="center",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0f0", edgecolor="#d62728"),
)

plt.tight_layout()

# Render in Streamlit
st.pyplot(fig)
