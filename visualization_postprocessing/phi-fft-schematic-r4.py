import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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

st.sidebar.subheader("Layout & Padding")
padding = st.sidebar.slider("Padding between domains", 11.0, 20.0, 14.0, 0.5)
gradient_line = st.sidebar.checkbox("Apply colormap gradient to signal line", value=True)

st.sidebar.subheader("Typography")
lbl_fs = st.sidebar.slider("Domain Label Font Size", 8, 22, 13)
eq_fs  = st.sidebar.slider("Equation Font Size", 8, 20, 12)
leg_fs = st.sidebar.slider("Legend Font Size", 8, 18, 10)
ax_fs  = st.sidebar.slider("Axis Tick Font Size", 8, 18, 11)

# 60+ Colormaps
cmaps = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
    'turbo', 'jet', 'rainbow', 'gist_rainbow', 'hsv', 
    'coolwarm', 'bwr', 'seismic', 'RdBu', 'RdYlBu', 
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 
    'tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 
    'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 
    'PuBu', 'PuRd', 'BuPu', 'OrRd', 'YlOrRd', 'YlOrBr', 'YlGn', 'YlGnBu', 
    'afmhot', 'binary', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'copper'
]
st.sidebar.subheader("Colormap Selection")
selected_cmap = st.sidebar.selectbox("Choose a colormap", cmaps, index=cmaps.index('turbo'))
cmap = plt.colormaps.get_cmap(selected_cmap)

st.sidebar.subheader("3D View Angle")
elev = st.sidebar.slider("Elevation", 0, 90, 20, 5)
azim = st.sidebar.slider("Azimuth", -180, 180, -50, 5)

# ---------- Figure ----------
plt.style.use("default")
fig = plt.figure(figsize=(14, 7.5), facecolor="white")
ax  = fig.add_subplot(111, projection="3d")

# Hide default 3D panes for a clean look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Spatial coordinate
x   = np.linspace(0, 10, 500)
s1  = amp1 * np.sin(2 * np.pi * f1 * x)
s2  = amp2 * np.sin(2 * np.pi * f2 * x)
s_t = s1 + s2

# Dynamic palette based on colormap
C_PHI  = cmap(0.85)   # Main signal color
C_M1   = cmap(0.15)   # Mode 1 color
C_M2   = cmap(0.50)   # Mode 2 color
C_REAL = cmap(0.10)   # Real space plane color
C_FOUR = cmap(0.70)   # Fourier space plane color
C_FFT  = cmap(0.95)   # FFT arrow color
C_INVT = "#7f8c8d"    # Inverse FFT color (kept gray for distinction)
C_AXIS = "#34495e"    # Axis color

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
ax.plot([0, 10], [y_real, y_real], [-3, -3], color=C_AXIS, lw=1.2)
ax.text(10.2, y_real, -3, "r", fontsize=ax_fs, fontweight="bold", color=C_AXIS)

# Combined field φ(r)  – solid, thick (or gradient line)
if gradient_line:
    # Create a gradient line using Line3DCollection
    points = np.array([x, np.full_like(x, y_real), s_t]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(s_t.min(), s_t.max())
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=3.0)
    lc.set_array(s_t)
    ax.add_collection3d(lc)
else:
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
x_fourier = padding
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
ax.plot([x_fourier, x_fourier], [0, k_max], [0, 0], color=C_AXIS, lw=1.2)
ax.text(x_fourier + 0.3, k_max, 0, "k", fontsize=ax_fs, fontweight="bold", color=C_AXIS)

# Spectral peaks (stem plot)
ax.plot([x_fourier, x_fourier], [f1, f1], [0, amp1], color=C_M1, lw=4)
ax.scatter([x_fourier], [f1], [amp1], color=C_M1, s=130, zorder=5,
           edgecolors="black", linewidths=0.8)
ax.plot([x_fourier, x_fourier], [f2, f2], [0, amp2], color=C_M2, lw=4)
ax.scatter([x_fourier], [f2], [amp2], color=C_M2, s=130, zorder=5,
           edgecolors="black", linewidths=0.8)

# k-axis tick labels
ax.text(x_fourier - 0.25, f1, -0.4, r"$k_1$", fontsize=ax_fs, color=C_M1,
        ha="right", fontweight="bold")
ax.text(x_fourier - 0.25, f2, -0.4, r"$k_2$", fontsize=ax_fs, color=C_M2,
        ha="right", fontweight="bold")

# =====================================================================
# 3. FFT / IFFT ARROWS connecting the two domains
# =====================================================================
# Forward FFT arrow  (colored, bold)
ax.add_artist(Arrow3D(10.5, 0.5,  1.8, (x_fourier - 10.5), 0, 0,
                      mutation_scale=28, lw=2.8,
                      arrowstyle="-|>", color=C_FFT))
# Inverse FFT arrow (gray, dashed)
ax.add_artist(Arrow3D(x_fourier - 0.5, 0.5, -1.8, -(x_fourier - 10.5), 0, 0,
                      mutation_scale=16, lw=1.5,
                      arrowstyle="-|>", color=C_INVT, linestyle="--"))

# FFT label box
mid_x = 10.5 + (x_fourier - 10.5) / 2
ax.text(mid_x, 0.5, 2.9, "FFT", fontsize=lbl_fs, fontweight="bold",
        color=C_FFT, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=cmap(0.95, alpha=0.15),
                  edgecolor=C_FFT, linewidth=1.5))
# IFFT label
ax.text(mid_x, 0.5, -2.2, "IFFT", fontsize=ax_fs, fontweight="bold",
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
        fontsize=lbl_fs, fontweight="bold", ha="center", color=C_PHI,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=cmap(0.10, alpha=0.2),
                  edgecolor=C_PHI, alpha=0.9, linewidth=1.2))

# Fourier label: φ̂(k) = F{φ(r)}
ax.text(x_fourier, k_max + 0.7, -1.6,
        r"Fourier:  $\hat{\phi}(\mathbf{k}) = \mathcal{F}\{\phi(\mathbf{r})\}$",
        fontsize=lbl_fs, fontweight="bold", ha="center", color=C_FFT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=cmap(0.70, alpha=0.2),
                  edgecolor=C_FFT, alpha=0.9, linewidth=1.2))

# Bottom: FFT operating equations (matching the paper)
fft_eq_1 = r"$\mathcal{F}\left[\nabla^{2}\phi(\mathbf{r})\right] = -k^{2}\,\hat{\phi}(\mathbf{k})$"
fft_eq_2 = r"$\hat{\phi}^{t+\Delta t}(\mathbf{k}) = \dfrac{\hat{\phi}^{t}(\mathbf{k}) + \Delta t\,\hat{R}_{\mathrm{explicit}}(\mathbf{k})}{1 + \Delta t\, L_{\mathrm{ref}}\,\kappa_{\mathrm{ref}}\, k^{2}}$"

fig.text(0.5, 0.02,
         fft_eq_1 + "          " + fft_eq_2,
         fontsize=eq_fs, fontweight="bold", color="#a00000", ha="center",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0f0",
                   edgecolor="#d62728", linewidth=1.2))

# Legend
ax.legend(loc="upper left", fontsize=leg_fs, framealpha=0.95,
          edgecolor=C_AXIS, fancybox=True, shadow=True)

# Axis limits (explicit padding)
ax.set_xlim(-0.5, x_fourier + 1.5)
ax.set_ylim(-1.5, k_max + 1.2)
ax.set_zlim(-4, 4)

plt.tight_layout()
st.pyplot(fig)

