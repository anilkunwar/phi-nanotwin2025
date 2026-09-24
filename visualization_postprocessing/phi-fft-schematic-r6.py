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

> **Tip:** Use the sidebar to edit, hide, or recolor every label and palette element independently.
"""
)

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================

# ---------- Signal parameters ----------
st.sidebar.header("Signal Parameters")
f1   = st.sidebar.slider("Wavenumber $k_1$", 0.1, 3.0, 0.5, 0.1)
amp1 = st.sidebar.slider("Amplitude $A_1$",   0.1, 2.0, 1.0, 0.1)
f2   = st.sidebar.slider("Wavenumber $k_2$", 0.1, 3.0, 1.5, 0.1)
amp2 = st.sidebar.slider("Amplitude $A_2$",   0.1, 2.0, 0.5, 0.1)

# ---------- Layout ----------
st.sidebar.subheader("Layout & Padding")
padding = st.sidebar.slider("Padding between domains", 11.0, 20.0, 14.0, 0.5)
gradient_line = st.sidebar.checkbox("Apply colormap gradient to signal line", value=True)

# ---------- Typography ----------
st.sidebar.subheader("Typography")
lbl_fs = st.sidebar.slider("Domain Label Font Size", 8, 22, 13)
eq_fs  = st.sidebar.slider("Equation Font Size", 8, 20, 12)
leg_fs = st.sidebar.slider("Legend Font Size", 8, 18, 10)
ax_fs  = st.sidebar.slider("Axis Tick Font Size", 8, 18, 11)

# ---------- Colormaps ----------
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
use_custom_colors = st.sidebar.checkbox("Override colormap with custom colors", value=False)
selected_cmap = st.sidebar.selectbox("Choose a colormap", cmaps, index=cmaps.index('turbo'))
cmap = plt.colormaps.get_cmap(selected_cmap)

# ---------- 3D View ----------
st.sidebar.subheader("3D View Angle")
elev = st.sidebar.slider("Elevation", 0, 90, 20, 5)
azim = st.sidebar.slider("Azimuth", -180, 180, -50, 5)

# =====================================================================
# LABEL CUSTOMIZATION (visibility + text + colors)
# =====================================================================
st.sidebar.markdown("---")
st.sidebar.header("🏷️ Label Customization")

def label_controls(name, default_text, default_text_color="#000000", default_box_color="#ffffff"):
    """Return (show, text, text_color, box_color) for a single label."""
    with st.sidebar.expander(f"Label: {name}", expanded=False):
        show = st.checkbox(f"Show {name}", value=True, key=f"show_{name}")
        text = st.text_input(f"{name} text", value=default_text, key=f"text_{name}")
        tc   = st.color_picker(f"{name} text color", value=default_text_color, key=f"tc_{name}")
        bc   = st.color_picker(f"{name} box color", value=default_box_color, key=f"bc_{name}")
    return show, text, tc, bc

# Legend
show_leg_m1, txt_leg_m1, tc_leg_m1, _ = label_controls(
    "Legend Mode 1", r"Mode 1: $A_1\sin(k_1 r)$", "#1f77b4", "#ffffff")
show_leg_m2, txt_leg_m2, tc_leg_m2, _ = label_controls(
    "Legend Mode 2", r"Mode 2: $A_2\sin(k_2 r)$", "#2ca02c", "#ffffff")
show_legend_box = st.sidebar.checkbox("Show legend box at all", value=True)

# FFT / IFFT arrows
show_fft,  txt_fft,  tc_fft,  bc_fft  = label_controls("FFT arrow",  "FFT",  "#a00000", "#ffe0e0")
show_ifft, txt_ifft, tc_ifft, bc_ifft = label_controls("IFFT arrow", "IFFT", "#7f8c8d", "#ecf0f1")

# Domain labels
show_real, txt_real, tc_real, bc_real = label_controls(
    "Real Space", r"Real Space:  $\phi(\mathbf{r})$", "#1a5276", "#d6eaf8")
show_four, txt_four, tc_four, bc_four = label_controls(
    "Fourier", r"Fourier:  $\hat{\phi}(\mathbf{k}) = \mathcal{F}\{\phi(\mathbf{r})\}$",
    "#7d6608", "#fcf3cf")

# Bottom equation box
show_eq, txt_eq_1, tc_eq_1, bc_eq_1 = label_controls(
    "Equation 1", r"$\mathcal{F}\left[\nabla^{2}\phi(\mathbf{r})\right] = -k^{2}\,\hat{\phi}(\mathbf{k})$",
    "#a00000", "#fff0f0")
show_eq2, txt_eq_2, tc_eq_2, bc_eq_2 = label_controls(
    "Equation 2",
    r"$\hat{\phi}^{t+\Delta t}(\mathbf{k}) = \dfrac{\hat{\phi}^{t}(\mathbf{k}) + \Delta t\,\hat{R}_{\mathrm{explicit}}(\mathbf{k})}{1 + \Delta t\, L_{\mathrm{ref}}\,\kappa_{\mathrm{ref}}\, k^{2}}$",
    "#a00000", "#fff0f0")
show_eq_box = st.sidebar.checkbox("Show equation box border", value=True)

# Axis labels
show_r,  txt_r,  tc_r,  _  = label_controls("r-axis",  "r", "#34495e", "#ffffff")
show_k,  txt_k,  tc_k,  _  = label_controls("k-axis",  "k", "#34495e", "#ffffff")
show_k1, txt_k1, tc_k1, _  = label_controls("k₁ tick", r"$k_1$", "#1f77b4", "#ffffff")
show_k2, txt_k2, tc_k2, _  = label_controls("k₂ tick", r"$k_2$", "#2ca02c", "#ffffff")

# =====================================================================
# GLOBAL PALETTE OVERRIDES
# =====================================================================
st.sidebar.markdown("---")
st.sidebar.header("🎨 Global Palette Overrides")

def color_override(name, default_hex, key_prefix):
    """If use_custom_colors is on, let user pick; else return default."""
    if use_custom_colors:
        return st.sidebar.color_picker(f"{name}", value=default_hex, key=f"{key_prefix}_{name}")
    return default_hex

C_PHI  = color_override("Signal φ(r)",       cmap(0.85), "c_phi")
C_M1   = color_override("Mode 1 color",      cmap(0.15), "c_m1")
C_M2   = color_override("Mode 2 color",      cmap(0.50), "c_m2")
C_REAL = color_override("Real-space plane",  cmap(0.10), "c_real")
C_FOUR = color_override("Fourier plane",     cmap(0.70), "c_four")
C_FFT  = color_override("FFT arrow",         cmap(0.95), "c_fft")
C_INVT = color_override("IFFT arrow",        "#7f8c8d", "c_invt")
C_AXIS = color_override("Axis lines",        "#34495e", "c_axis")

# =====================================================================
# FIGURE
# =====================================================================
plt.style.use("default")
fig = plt.figure(figsize=(14, 7.5), facecolor="white")
ax  = fig.add_subplot(111, projection="3d")

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

x   = np.linspace(0, 10, 500)
s1  = amp1 * np.sin(2 * np.pi * f1 * x)
s2  = amp2 * np.sin(2 * np.pi * f2 * x)
s_t = s1 + s2

# =====================================================================
# 1. REAL SPACE DOMAIN
# =====================================================================
y_real = 0
xx_p, zz_p = np.meshgrid([0, 10], [-3, 3])
yy_p = np.full_like(xx_p, y_real)
ax.plot_surface(xx_p, yy_p, zz_p, color=C_REAL, alpha=0.35,
                edgecolor=C_PHI, linewidth=0.3)

for gx in np.linspace(0, 10, 6):
    ax.plot([gx, gx], [y_real, y_real], [-3, 3], color=C_PHI, alpha=0.08, lw=0.5)
for gz in np.linspace(-3, 3, 7):
    ax.plot([0, 10], [y_real, y_real], [gz, gz], color=C_PHI, alpha=0.08, lw=0.5)

ax.plot([0, 10], [y_real, y_real], [-3, -3], color=C_AXIS, lw=1.2)
if show_r:
    ax.text(10.2, y_real, -3, txt_r, fontsize=ax_fs, fontweight="bold", color=tc_r)

if gradient_line:
    points = np.array([x, np.full_like(x, y_real), s_t]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(s_t.min(), s_t.max())
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=3.0)
    lc.set_array(s_t)
    ax.add_collection3d(lc)
else:
    ax.plot(x, np.full_like(x, y_real), s_t,
            color=C_PHI, linewidth=3.0, label=r"$\phi(\mathbf{r})$")

ax.plot(x, np.full_like(x, y_real), s1,
        color=C_M1, lw=1.5, ls="--",  alpha=0.75,
        label=txt_leg_m1 if show_leg_m1 else None)
ax.plot(x, np.full_like(x, y_real), s2,
        color=C_M2, lw=1.5, ls="-.",  alpha=0.75,
        label=txt_leg_m2 if show_leg_m2 else None)

# =====================================================================
# 2. WAVENUMBER DOMAIN
# =====================================================================
x_fourier = padding
k_max     = max(f1, f2) + 1.2

y_kv = np.linspace(0, k_max, 2)
z_kv = np.linspace(-0.5, max(amp1, amp2) + 0.5, 2)
Y_p, Z_p = np.meshgrid(y_kv, z_kv)
X_p = np.full_like(Y_p, x_fourier)
ax.plot_surface(X_p, Y_p, Z_p, color=C_FOUR, alpha=0.45,
                edgecolor=C_FFT, linewidth=0.3)

for gk in np.linspace(0, k_max, 6):
    ax.plot([x_fourier, x_fourier], [gk, gk],
            [-0.5, max(amp1, amp2) + 0.5], color=C_FFT, alpha=0.08, lw=0.5)
for gz in np.linspace(-0.5, max(amp1, amp2) + 0.5, 5):
    ax.plot([x_fourier, x_fourier], [0, k_max], [gz, gz],
            color=C_FFT, alpha=0.08, lw=0.5)

ax.plot([x_fourier, x_fourier], [0, k_max], [0, 0], color=C_AXIS, lw=1.2)
if show_k:
    ax.text(x_fourier + 0.3, k_max, 0, txt_k, fontsize=ax_fs, fontweight="bold", color=tc_k)

ax.plot([x_fourier, x_fourier], [f1, f1], [0, amp1], color=C_M1, lw=4)
ax.scatter([x_fourier], [f1], [amp1], color=C_M1, s=130, zorder=5,
           edgecolors="black", linewidths=0.8)
ax.plot([x_fourier, x_fourier], [f2, f2], [0, amp2], color=C_M2, lw=4)
ax.scatter([x_fourier], [f2], [amp2], color=C_M2, s=130, zorder=5,
           edgecolors="black", linewidths=0.8)

if show_k1:
    ax.text(x_fourier - 0.25, f1, -0.4, txt_k1, fontsize=ax_fs, color=tc_k1,
            ha="right", fontweight="bold")
if show_k2:
    ax.text(x_fourier - 0.25, f2, -0.4, txt_k2, fontsize=ax_fs, color=tc_k2,
            ha="right", fontweight="bold")

# =====================================================================
# 3. FFT / IFFT ARROWS
# =====================================================================
ax.add_artist(Arrow3D(10.5, 0.5,  1.8, (x_fourier - 10.5), 0, 0,
                      mutation_scale=28, lw=2.8,
                      arrowstyle="-|>", color=C_FFT))
ax.add_artist(Arrow3D(x_fourier - 0.5, 0.5, -1.8, -(x_fourier - 10.5), 0, 0,
                      mutation_scale=16, lw=1.5,
                      arrowstyle="-|>", color=C_INVT, linestyle="--"))

mid_x = 10.5 + (x_fourier - 10.5) / 2

if show_fft:
    ax.text(mid_x, 0.5, 2.9, txt_fft, fontsize=lbl_fs, fontweight="bold",
            color=tc_fft, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=bc_fft,
                      edgecolor=tc_fft, linewidth=1.5))
if show_ifft:
    ax.text(mid_x, 0.5, -2.2, txt_ifft, fontsize=ax_fs, fontweight="bold",
            color=tc_ifft, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bc_ifft,
                      edgecolor=tc_ifft, linewidth=1.0))

# =====================================================================
# 4. ANNOTATIONS
# =====================================================================
ax.set_axis_off()
ax.view_init(elev=elev, azim=azim)

if show_real:
    ax.text(5, -0.8, -3.6, txt_real,
            fontsize=lbl_fs, fontweight="bold", ha="center", color=tc_real,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bc_real,
                      edgecolor=tc_real, alpha=0.9, linewidth=1.2))

if show_four:
    ax.text(x_fourier, k_max + 0.7, -1.6, txt_four,
            fontsize=lbl_fs, fontweight="bold", ha="center", color=tc_four,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bc_four,
                      edgecolor=tc_four, alpha=0.9, linewidth=1.2))

# Bottom equation box
if show_eq or show_eq2:
    eq_parts = []
    if show_eq:  eq_parts.append(txt_eq_1)
    if show_eq2: eq_parts.append(txt_eq_2)
    eq_combined = "          ".join(eq_parts)

    box_kwargs = dict(boxstyle="round,pad=0.5", linewidth=1.2)
    if show_eq_box:
        box_kwargs["facecolor"] = bc_eq_1
        box_kwargs["edgecolor"] = tc_eq_1
    else:
        box_kwargs["facecolor"] = "none"
        box_kwargs["edgecolor"] = "none"

    fig.text(0.5, 0.02, eq_combined,
             fontsize=eq_fs, fontweight="bold", color=tc_eq_1, ha="center",
             bbox=box_kwargs)

# Legend
if show_legend_box:
    ax.legend(loc="upper left", fontsize=leg_fs, framealpha=0.95,
              edgecolor=C_AXIS, fancybox=True, shadow=True)

ax.set_xlim(-0.5, x_fourier + 1.5)
ax.set_ylim(-1.5, k_max + 1.2)
ax.set_zlim(-4, 4)

plt.tight_layout()
st.pyplot(fig)
