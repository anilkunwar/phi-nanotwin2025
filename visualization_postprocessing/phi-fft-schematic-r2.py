import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import streamlit as st


# ---------- Helper: true 3D arrow ----------
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


# ---------- Streamlit config ----------
st.set_page_config(
    page_title="Phase-Field FFT Schematic",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Phase-Field FFT Spectral Method Schematic")
st.markdown(
    r"""
This interactive visualization shows the transformation of $\phi(\mathbf{r})$ into
$\hat{\phi}(\mathbf{k}) = \mathcal{F}\{\phi(\mathbf{r})\}$ via the **Fast Fourier Transform (FFT)**.

The real-space and wavenumber domains are shown side-by-side with sufficient padding,
connected by a directional FFT arrow (and an inverse-FFT return path).
"""
)

# ---------- Sidebar controls ----------
st.sidebar.header("Signal Parameters")
f1   = st.sidebar.slider("Wavenumber $k_1$", 0.1, 3.0, 0.5, 0.1)
amp1 = st.sidebar.slider("Amplitude $A_1$",   0.1, 2.0, 1.0, 0.1)
f2   = st.sidebar.slider("Wavenumber $k_2$", 0.1, 3.0, 1.5, 0.1)
amp2 = st.sidebar.slider("Amplitude $A_2$",   0.1, 2.0, 0.5, 0.1)

st.sidebar.subheader("3D View Angle")
elev = st.sidebar.slider("Elevation", 0, 90, 20, 5)
azim = st.sidebar.slider("Azimuth", -180, 180, -50, 5)

# ---------- Figure ----------
plt.style.use("default")
fig = plt.figure(figsize=(14, 7.5), facecolor="white")
ax  = fig.add_subplot(111, projection="3d")

# Spatial coordinate
x   = np.linspace(0, 10, 500)
s1  = amp1 * np.sin(2 * np.pi * f1 * x)
s2  = amp2 * np.sin(2 * np.pi * f2 * x)
s_t = s1 + s2

# Harmonious palette
C_PHI  = "#1f3a93"   # deep blue  – φ(r)
C_M1   = "#e67e22"   # warm orange – mode 1
C_M2   = "#27ae60"   # emerald    – mode 2
C_REAL = "#dbeafe"   # light blue panel
C_FOUR = "#fce4ec"   # light pink panel
C_FFT  = "#c0392b"   # crimson    – FFT arrow / label
C_INVT = "#7f8c8d"   # slate gray – inverse FFT

# =====================================================================
# 1. REAL SPACE DOMAIN  (front plane at y = 0)
# =====================================================================
y_real = 0
xx_p, zz_p = np.meshgrid([0, 10], [-3, 3])
yy_p = np.full_like(xx_p, y_real)
ax.plot_surface(xx_p, yy_p, zz_p, color=C_REAL, alpha=0.35,
                edgecolor=C_PHI, linewidth=0.3)

# subtle gridlines on real-space panel
for gx in np.linspace(0, 10, 6):
    ax.plot([gx, gx], [y_real, y_real], [-3, 3], color=C_PHI, alpha=0.08, lw=0.5)
for gz in np.linspace(-3, 3, 7):
    ax.plot([0, 10], [y_real, y_real], [gz, gz], color=C_PHI, alpha=0.08, lw=0.5)

# r-axis baseline
ax.plot([0, 10], [y_real, y_real], [-3, -3], color="#34495e", lw=1.2)
ax.text(10.2, y_real, -3, "r", fontsize=11, fontweight="bold", color="#34495e")

# Combined field φ(r)  – solid, thick
ax.plot(x, np.full_like(x, y_real), s_t,
        color=C_PHI, linewidth=3.0, label=r"$\phi(\mathbf{r})$")
# Component modes
ax.plot(x, np.full_like(x, y_real), s1,
        color=C_M1, lw=1.5, ls="--",  alpha=0.75, label=r"Mode 1: $A_1\sin(k_1 r)$")
ax.plot(x, np.full_like(x, y_real), s2,
        color=C_M2, lw=1.5, ls="-.",  alpha=0.75, label=r"Mode 2: $A_2\sin(k_2 r)$")

# =====================================================================
# 2. WAVENUMBER DOMAIN  (back plane — sufficient padding)
# =====================================================================
x_fourier = 13                       # <-- padding between domains
k_max     = max(f1, f2) + 1.2

y_kv = np.linspace(0, k_max, 2)
z_kv = np.linspace(-0.5, max(amp1, amp2) + 0.5, 2)
Y_p, Z_p = np.meshgrid(y_kv, z_kv)
X_p = np.full_like(Y_p, x_fourier)
ax.plot_surface(X_p, Y_p, Z_p, color=C_FOUR, alpha=0.45,
                edgecolor=C_FFT, linewidth=0.3)

# gridlines on Fourier panel
for gk in np.linspace(0, k_max, 6):
    ax.plot([x_fourier, x_fourier], [gk, gk],
            [-0.5, max(amp1, amp2) + 0.5], color=C_FFT, alpha=0.08, lw=0.5)
for gz in np.linspace(-0.5, max(amp1, amp2) + 0.5, 5):
    ax.plot([x_fourier, x_fourier], [0, k_max], [gz, gz],
            color=C_FFT, alpha=0.08, lw=0.5)

# k-axis baseline
ax.plot([x_fourier, x_fourier], [0, k_max], [0, 0], color="#34495e", lw=1.2)
ax.text(x_fourier + 0.3, k_max, 0, "k", fontsize=11, fontweight="bold", color="#34495e")

# Spectral peaks (stem plot)
ax.plot([x_fourier, x_fourier], [f1, f1], [0, amp1], color=C_M1, lw=4)
ax.scatter([x_fourier], [f1], [amp1], color=C_M1, s=130, zorder=5,
           edgecolors="black", linewidths=0.8)
ax.plot([x_fourier, x_fourier], [f2, f2], [0, amp2], color=C_M2, lw=4)
ax.scatter([x_fourier], [f2], [amp2], color=C_M2, s=130, zorder=5,
           edgecolors="black", linewidths=0.8)

# k-axis tick labels
ax.text(x_fourier - 0.25, f1, -0.4, r"$k_1$", fontsize=10, color=C_M1,
        ha="right", fontweight="bold")
ax.text(x_fourier - 0.25, f2, -0.4, r"$k_2$", fontsize=10, color=C_M2,
        ha="right", fontweight="bold")

# =====================================================================
# 3. FFT / IFFT ARROWS connecting the two domains
# =====================================================================
# Forward FFT arrow  (red, bold)
ax.add_artist(Arrow3D(10.5, 0.5,  1.8, 2.0, 0, 0,
                      mutation_scale=28, lw=2.8,
                      arrowstyle="-|>", color=C_FFT))
# Inverse FFT arrow (gray, dashed)
ax.add_artist(Arrow3D(12.5, 0.5, -1.8, -2.0, 0, 0,
                      mutation_scale=16, lw=1.5,
                      arrowstyle="-|>", color=C_INVT, linestyle="--"))

# FFT label box
ax.text(11.5, 0.5, 2.9, "FFT", fontsize=14, fontweight="bold",
        color=C_FFT, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8",
                  edgecolor=C_FFT, linewidth=1.5))
# IFFT label
ax.text(11.5, 0.5, -2.2, "IFFT", fontsize=9, fontweight="bold",
        color=C_INVT, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                  edgecolor=C_INVT, linewidth=1.0))

# =====================================================================
# 4. ANNOTATIONS  – simplified labels
# =====================================================================
ax.set_axis_off()
ax.view_init(elev=elev, azim=azim)

# Real-space label: just φ(r)
ax.text(5, -0.8, -3.6,
        r"Real Space:  $\phi(\mathbf{r})$",
        fontsize=13, fontweight="bold", ha="center", color=C_PHI,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C_REAL,
                  edgecolor=C_PHI, alpha=0.9, linewidth=1.2))

# Fourier label: φ̂(k) = F{φ(r)}
ax.text(x_fourier, k_max + 0.7, -1.6,
        r"Fourier:  $\hat{\phi}(\mathbf{k}) = \mathcal{F}\{\phi(\mathbf{r})\}$",
        fontsize=13, fontweight="bold", ha="center", color=C_FFT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C_FOUR,
                  edgecolor=C_FFT, alpha=0.9, linewidth=1.2))

# Bottom: FFT operating equations
# NOTE: the entire \dfrac{...}{...} must live inside ONE raw string
fig.text(0.5, 0.025,
         r"$\mathcal{F}\!\left[\nabla^{2}\phi(\mathbf{r})\right] = -k^{2}\,\hat{\phi}(\mathbf{k})$"
         "          "
         r"$\hat{\phi}^{\,t+\Delta t}(\mathbf{k}) = "
         r"\dfrac{\hat{\phi}^{\,t}(\mathbf{k}) + \Delta t\,\hat{R}_{\mathrm{explicit}}(\mathbf{k})}"
         r"{1 + \Delta t\, L_{\mathrm{ref}}\,\kappa_{\mathrm{ref}}\, k^{2}}$",
         fontsize=12, fontweight="bold", color="#a00000", ha="center",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0f0",
                   edgecolor="#d62728", linewidth=1.2))

# Legend
ax.legend(loc="upper left", fontsize=10, framealpha=0.95,
          edgecolor="#34495e", fancybox=True, shadow=True)

# Axis limits (explicit padding)
ax.set_xlim(-0.5, 14)
ax.set_ylim(-1.5, k_max + 1.2)
ax.set_zlim(-4, 4)

plt.tight_layout()
st.pyplot(fig)
