"""
Phase-Field FFT Spectral Method Schematic
=========================================
Interactive 3D visualization (Matplotlib + Streamlit) of

        phi(r)  --FFT-->  phi_hat(k)  --IFFT-->  phi(r)

Fully customizable: signal parameters, view angles, domain padding,
font sizes, line widths, panel styling and 90+ colormaps
(incl. rainbow, inferno, jet, turbo, viridis, plasma, ...).
"""

import io

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import streamlit as st


# =====================================================================
# 0. PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Phase-Field FFT Schematic",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# 1. HELPER: TRUE 3D ARROW
# =====================================================================
class Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch that lives in 3D data space and is projected
    onto the 2D canvas at draw time."""

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


# =====================================================================
# 2. COLORMAP REGISTRY  (90+ entries)
# =====================================================================
CMAP_NAMES = [
    # ---- Perceptually uniform sequential -------------------------
    "viridis", "plasma", "inferno", "magma", "cividis",
    # ---- Sequential (ColorBrewer) --------------------------------
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Oranges",
    "OrRd", "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds",
    "YlGn", "YlGnBu", "YlOrBr", "YlOrRd",
    # ---- Sequential (misc) ---------------------------------------
    "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
    "spring", "summer", "autumn", "winter", "cool", "Wistia", "hot",
    "afmhot", "gist_heat", "copper",
    # ---- Diverging -----------------------------------------------
    "PiYG", "PRGn", "BrBG", "PuOr", "RdBu", "RdGy", "RdYlBu",
    "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic",
    "berlin", "managua", "vanimo",
    # ---- Cyclic ---------------------------------------------------
    "twilight", "twilight_shifted", "hsv",
    # ---- Qualitative ---------------------------------------------
    "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1",
    "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c",
    # ---- Rainbow / spectral / misc -------------------------------
    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
    "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg",
    "gist_rainbow", "rainbow", "jet", "turbo", "nipy_spectral",
    "gist_ncar",
]

# Keep only colormaps actually installed in this Matplotlib build
try:
    _available = set(plt.colormaps())
except Exception:                                    # very old Matplotlib
    _available = set(matplotlib.colormaps)

CMAP_CHOICES = [c for c in CMAP_NAMES if c in _available]
if not CMAP_CHOICES:                                  # ultimate fallback
    CMAP_CHOICES = ["viridis", "jet", "turbo", "rainbow", "inferno"]


def get_cmap(name, reverse=False):
    """Return a Matplotlib colormap object (robust across versions)."""
    try:
        cmap = matplotlib.colormaps[name]
    except Exception:
        cmap = plt.get_cmap(name)
    return cmap.reversed() if reverse else cmap


def cmap_rgb(cmap, frac):
    """RGB tuple sampled from `cmap` at normalized position `frac`."""
    return tuple(np.array(cmap(float(np.clip(frac, 0.0, 1.0)))[:3]))


def cmap_hex(cmap, frac):
    """Hex colour string sampled from `cmap` at normalized position `frac`."""
    return mcolors.to_hex(cmap(float(np.clip(frac, 0.0, 1.0))))


# =====================================================================
# 3. SIDEBAR — CONTROLS
# =====================================================================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # -----------------------------------------------------------------
    # 3.1 Signal parameters
    # -----------------------------------------------------------------
    with st.expander("📈 Signal parameters", expanded=True):
        f1 = st.slider("Wavenumber $k_1$", 0.1, 3.0, 0.5, 0.1)
        amp1 = st.slider("Amplitude $A_1$", 0.1, 2.0, 1.0, 0.1)
        f2 = st.slider("Wavenumber $k_2$", 0.1, 3.0, 1.5, 0.1)
        amp2 = st.slider("Amplitude $A_2$", 0.1, 2.0, 0.5, 0.1)
        n_samples = st.slider("Samples along $r$", 100, 2000, 500, 50)

    # -----------------------------------------------------------------
    # 3.2 3D view
    # -----------------------------------------------------------------
    with st.expander("🧭 3D view angle", expanded=False):
        elev = st.slider("Elevation", 0, 90, 20, 5)
        azim = st.slider("Azimuth", -180, 180, -50, 5)
        roll = st.slider("Roll", -90, 90, 0, 5)

    # -----------------------------------------------------------------
    # 3.3 Layout & padding
    # -----------------------------------------------------------------
    with st.expander("📐 Layout & padding", expanded=False):
        gap = st.slider(
            "Domain gap (padding between panels)", 1.0, 8.0, 3.0, 0.25,
            help="Horizontal distance between the real-space plane and "
                 "the Fourier plane.",
        )
        real_half_h = st.slider(
            "Real-space panel half-height", 1.0, 6.0, 3.0, 0.5
        )
        axes_margin = st.slider("Outer axes margin", 0.0, 3.0, 0.6, 0.1)
        fig_w = st.slider("Figure width (in)", 8.0, 24.0, 14.0, 0.5)
        fig_h = st.slider("Figure height (in)", 5.0, 16.0, 7.5, 0.5)
        dpi = st.slider("Export DPI", 80, 400, 200, 20)

    # -----------------------------------------------------------------
    # 3.4 Font sizes
    # -----------------------------------------------------------------
    with st.expander("🔤 Font sizes", expanded=False):
        ui_title_fs = st.slider("Streamlit page title", 16, 60, 34, 2)
        domain_title_fs = st.slider("Domain titles (Real / Fourier)", 6, 40, 13, 1)
        axis_label_fs = st.slider("Axis labels  r , k", 6, 40, 11, 1)
        tick_label_fs = st.slider("Peak tick labels  k₁ , k₂", 6, 36, 10, 1)
        arrow_label_fs = st.slider("FFT / IFFT labels", 6, 40, 14, 1)
        legend_fs = st.slider("Legend", 6, 36, 10, 1)
        equation_fs = st.slider("Bottom equations", 6, 36, 12, 1)
        font_family = st.selectbox(
            "Font family", ["sans-serif", "serif", "monospace"], index=0
        )

    # -----------------------------------------------------------------
    # 3.5 Line widths & markers
    # -----------------------------------------------------------------
    with st.expander("✏️ Lines, arrows & markers", expanded=False):
        curve_lw = st.slider("Field curve φ(r) width", 0.5, 10.0, 3.0, 0.5)
        mode_lw = st.slider("Component mode width", 0.5, 8.0, 1.5, 0.5)
        stem_lw = st.slider("Spectral stem width", 1.0, 15.0, 4.0, 0.5)
        marker_size = st.slider("Spectral peak marker size", 20, 400, 130, 10)
        arrow_lw = st.slider("FFT arrow width", 0.5, 8.0, 2.8, 0.2)
        arrow_scale = st.slider("FFT arrow head scale", 5, 60, 28, 1)
        panel_edge_lw = st.slider("Panel edge width", 0.0, 3.0, 0.3, 0.1)

    # -----------------------------------------------------------------
    # 3.6 Colours, colormaps & styling
    # -----------------------------------------------------------------
    with st.expander("🎨 Colormap & styling", expanded=True):
        cmap_name = st.selectbox(
            "Colormap", CMAP_CHOICES,
            index=CMAP_CHOICES.index("viridis") if "viridis" in CMAP_CHOICES else 0,
            help="90+ Matplotlib colormaps, incl. rainbow, inferno, jet, turbo.",
        )
        reverse_cmap = st.checkbox("Reverse colormap", value=False)
        apply_target = st.selectbox(
            "Apply colormap to",
            ["Panels + Curves", "Panels only", "Curves only", "Neither (classic)"],
            index=0,
        )
        panel_alpha = st.slider("Panel opacity", 0.05, 1.0, 0.40, 0.05)
        shade_panels = st.checkbox("Shade 3D panels", value=False)
        bg_color = st.color_picker("Figure background", "#ffffff")
        eq_box_color = st.color_picker("Equation box face", "#fff0f0")
        eq_edge_color = st.color_picker("Equation box edge", "#d62728")
        eq_text_color = st.color_picker("Equation text", "#a00000")
        eq_y = st.slider("Equation vertical position", 0.0, 0.25, 0.025, 0.005)
        show_legend = st.checkbox("Show legend", value=True)

    # Build the actual colormap object now (needed for the preview strip)
    cmap = get_cmap(cmap_name, reverse_cmap)

    # ---- Colormap preview strip -------------------------------------
    st.markdown("**Colormap preview**")
    _grad = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    _fig_c, _ax_c = plt.subplots(figsize=(3.2, 0.42), dpi=110)
    _ax_c.imshow(_grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
    _ax_c.set_xticks([])
    _ax_c.set_yticks([])
    for _sp in _ax_c.spines.values():
        _sp.set_visible(False)
    _fig_c.patch.set_alpha(0.0)
    st.pyplot(_fig_c)
    plt.close(_fig_c)


# =====================================================================
# 4. DERIVED COLOURS
# =====================================================================
apply_panels = apply_target in ("Panels + Curves", "Panels only")
apply_curves = apply_target in ("Panels + Curves", "Curves only")

# ---- Curve / text palette -------------------------------------------
if apply_curves:
    C_PHI = cmap_hex(cmap, 0.02)     # combined field φ(r)
    C_M1 = cmap_hex(cmap, 0.38)      # mode 1
    C_M2 = cmap_hex(cmap, 0.66)      # mode 2
    C_FFT = cmap_hex(cmap, 0.97)     # FFT arrow / Fourier accents
    C_INVT = cmap_hex(cmap, 0.50)    # inverse FFT
else:
    C_PHI = "#1f3a93"                # deep blue
    C_M1 = "#e67e22"                 # warm orange
    C_M2 = "#27ae60"                 # emerald
    C_FFT = "#c0392b"                # crimson
    C_INVT = "#7f8c8d"               # slate gray

# ---- Panel fills -----------------------------------------------------
if apply_panels:
    real_panel_rgb = cmap_rgb(cmap, 0.15)
    four_panel_rgb = cmap_rgb(cmap, 0.82)
else:
    real_panel_rgb = mcolors.to_rgb("#dbeafe")   # light blue
    four_panel_rgb = mcolors.to_rgb("#fce4ec")   # light pink


# =====================================================================
# 5. STREAMLIT HEADER (font size controlled from the sidebar)
# =====================================================================
st.markdown(
    f"""
    <h1 style="font-size:{ui_title_fs}px; margin-bottom:0.2rem; line-height:1.15;">
        Phase-Field FFT Spectral Method Schematic
    </h1>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    r"""
This interactive visualization shows the transformation of $\phi(\mathbf{r})$ into
$\hat{\phi}(\mathbf{k}) = \mathcal{F}\{\phi(\mathbf{r})\}$ via the **Fast Fourier Transform (FFT)**.

The real-space and wavenumber domains are shown side-by-side with adjustable padding,
connected by a directional FFT arrow (and an inverse-FFT return path).
"""
)


# =====================================================================
# 6. MATPLOTLIB GLOBAL STYLE
# =====================================================================
matplotlib.rcParams["font.family"] = font_family
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 7. BUILD THE FIGURE
# =====================================================================
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg_color)
ax = fig.add_subplot(111, projection="3d")

# ---------------------------------------------------------------------
# 7.1 Signal
# ---------------------------------------------------------------------
x = np.linspace(0.0, 10.0, int(n_samples))
s1 = amp1 * np.sin(2 * np.pi * f1 * x)
s2 = amp2 * np.sin(2 * np.pi * f2 * x)
s_t = s1 + s2

# Derived geometry -----------------------------------------------------
y_real = 0.0                                  # real-space plane position
x_fourier = 10.0 + gap                        # Fourier plane position
k_max = max(f1, f2) + 1.2                     # k-axis extent
z_panel_lo, z_panel_hi = -real_half_h, real_half_h

z_top = max(
    real_half_h,
    float(np.max(np.abs(s_t))) if s_t.size else 1.0,
    max(amp1, amp2),
) + 1.0
z_bot = -z_top

x_lo = -0.6 - axes_margin
x_hi = x_fourier + 1.0 + axes_margin
y_lo = -1.5
y_hi = k_max + 1.6

# =====================================================================
# 8. REAL-SPACE DOMAIN  (front plane at y = 0)
# =====================================================================
xx_p, zz_p = np.meshgrid([0.0, 10.0], [z_panel_lo, z_panel_hi])
yy_p = np.full_like(xx_p, y_real)

ax.plot_surface(
    xx_p, yy_p, zz_p,
    color=real_panel_rgb,
    alpha=panel_alpha,
    edgecolor=C_PHI,
    linewidth=panel_edge_lw,
    shade=shade_panels,
    antialiased=True,
)

# Subtle gridlines on the real-space panel
for gx in np.linspace(0.0, 10.0, 6):
    ax.plot([gx, gx], [y_real, y_real], [z_panel_lo, z_panel_hi],
            color=C_PHI, alpha=0.08, lw=0.5)
for gz in np.linspace(z_panel_lo, z_panel_hi, 7):
    ax.plot([0.0, 10.0], [y_real, y_real], [gz, gz],
            color=C_PHI, alpha=0.08, lw=0.5)

# r-axis baseline
ax.plot([0.0, 10.0], [y_real, y_real], [z_panel_lo, z_panel_lo],
        color="#34495e", lw=1.2)
ax.text(10.2, y_real, z_panel_lo, "r",
        fontsize=axis_label_fs, fontweight="bold", color="#34495e")

# Combined field φ(r)
ax.plot(x, np.full_like(x, y_real), s_t,
        color=C_PHI, linewidth=curve_lw, label=r"$\phi(\mathbf{r})$")

# Component modes
ax.plot(x, np.full_like(x, y_real), s1,
        color=C_M1, lw=mode_lw, ls="--", alpha=0.75,
        label=r"Mode 1: $A_1\sin(k_1 r)$")
ax.plot(x, np.full_like(x, y_real), s2,
        color=C_M2, lw=mode_lw, ls="-.", alpha=0.75,
        label=r"Mode 2: $A_2\sin(k_2 r)$")


# =====================================================================
# 9. WAVENUMBER DOMAIN  (back plane — padding controlled by `gap`)
# =====================================================================
k_panel_lo = -0.5
k_panel_hi = max(amp1, amp2) + 0.5

y_kv = np.linspace(0.0, k_max, 2)
z_kv = np.linspace(k_panel_lo, k_panel_hi, 2)
Y_p, Z_p = np.meshgrid(y_kv, z_kv)
X_p = np.full_like(Y_p, x_fourier)

ax.plot_surface(
    X_p, Y_p, Z_p,
    color=four_panel_rgb,
    alpha=panel_alpha,
    edgecolor=C_FFT,
    linewidth=panel_edge_lw,
    shade=shade_panels,
    antialiased=True,
)

# Gridlines on the Fourier panel
for gk in np.linspace(0.0, k_max, 6):
    ax.plot([x_fourier, x_fourier], [gk, gk], [k_panel_lo, k_panel_hi],
            color=C_FFT, alpha=0.08, lw=0.5)
for gz in np.linspace(k_panel_lo, k_panel_hi, 5):
    ax.plot([x_fourier, x_fourier], [0.0, k_max], [gz, gz],
            color=C_FFT, alpha=0.08, lw=0.5)

# k-axis baseline
ax.plot([x_fourier, x_fourier], [0.0, k_max], [0.0, 0.0],
        color="#34495e", lw=1.2)
ax.text(x_fourier + 0.3, k_max, 0.0, "k",
        fontsize=axis_label_fs, fontweight="bold", color="#34495e")

# Spectral peaks (stem plot)
ax.plot([x_fourier, x_fourier], [f1, f1], [0.0, amp1],
        color=C_M1, lw=stem_lw)
ax.scatter([x_fourier], [f1], [amp1],
           color=C_M1, s=marker_size, zorder=5,
           edgecolors="black", linewidths=0.8)

ax.plot([x_fourier, x_fourier], [f2, f2], [0.0, amp2],
        color=C_M2, lw=stem_lw)
ax.scatter([x_fourier], [f2], [amp2],
           color=C_M2, s=marker_size, zorder=5,
           edgecolors="black", linewidths=0.8)

# k-axis tick labels
ax.text(x_fourier - 0.25, f1, k_panel_lo - 0.15, r"$k_1$",
        fontsize=tick_label_fs, color=C_M1, ha="right", fontweight="bold")
ax.text(x_fourier - 0.25, f2, k_panel_lo - 0.15, r"$k_2$",
        fontsize=tick_label_fs, color=C_M2, ha="right", fontweight="bold")


# =====================================================================
# 10. FFT / IFFT ARROWS CONNECTING THE TWO DOMAINS
# =====================================================================
z_arrow = 0.45 * z_top

# Geometry scales automatically with the padding gap
fwd_start_x = 10.0 + 0.35 * gap
fwd_len = 0.50 * gap
inv_start_x = x_fourier - 0.35 * gap
inv_len = -0.50 * gap

# Forward FFT arrow (bold)
ax.add_artist(
    Arrow3D(
        fwd_start_x, 0.5, z_arrow, fwd_len, 0.0, 0.0,
        mutation_scale=arrow_scale, lw=arrow_lw,
        arrowstyle="-|>", color=C_FFT,
    )
)

# Inverse FFT arrow (dashed, lighter)
ax.add_artist(
    Arrow3D(
        inv_start_x, 0.5, -z_arrow, inv_len, 0.0, 0.0,
        mutation_scale=arrow_scale * 0.6, lw=arrow_lw * 0.55,
        arrowstyle="-|>", color=C_INVT, linestyle="--",
    )
)

# ---- FFT label box ---------------------------------------------------
ax.text(
    10.0 + gap / 2.0, 0.5, z_arrow + 1.1, "FFT",
    fontsize=arrow_label_fs, fontweight="bold",
    color=C_FFT, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.5",
              facecolor=mcolors.to_hex(cmap_rgb(cmap, 0.92)) if apply_curves else "#fadbd8",
              edgecolor=C_FFT, linewidth=1.5),
)

# ---- IFFT label ------------------------------------------------------
ax.text(
    10.0 + gap / 2.0, 0.5, -z_arrow - 0.5, "IFFT",
    fontsize=max(6, arrow_label_fs - 5), fontweight="bold",
    color=C_INVT, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
              edgecolor=C_INVT, linewidth=1.0),
)


# =====================================================================
# 11. ANNOTATIONS
# =====================================================================
ax.set_axis_off()
for _axis in (ax.xaxis, ax.yaxis, ax.zaxis):     # belt-and-braces pane hiding
    _axis.pane.set_visible(False)
    _axis.pane.set_alpha(0.0)

ax.view_init(elev=elev, azim=azim, roll=roll)

# Real-space label
ax.text(
    5.0, -0.8, z_bot - 0.6,
    r"Real Space:  $\phi(\mathbf{r})$",
    fontsize=domain_title_fs, fontweight="bold", ha="center", color=C_PHI,
    bbox=dict(boxstyle="round,pad=0.4",
              facecolor=mcolors.to_hex(real_panel_rgb),
              edgecolor=C_PHI, alpha=0.9, linewidth=1.2),
)

# Fourier label
ax.text(
    x_fourier, k_max + 0.7, -1.6,
    r"Fourier:  $\hat{\phi}(\mathbf{k}) = \mathcal{F}\{\phi(\mathbf{r})\}$",
    fontsize=domain_title_fs, fontweight="bold", ha="center", color=C_FFT,
    bbox=dict(boxstyle="round,pad=0.4",
              facecolor=mcolors.to_hex(four_panel_rgb),
              edgecolor=C_FFT, alpha=0.9, linewidth=1.2),
)

# Legend
if show_legend:
    ax.legend(
        loc="upper left", fontsize=legend_fs, framealpha=0.95,
        edgecolor="#34495e", fancybox=True, shadow=True,
    )

# Axis limits (explicit padding)
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_zlim(z_bot, z_top)


# =====================================================================
# 12. BOTTOM EQUATIONS (operators in spectral space)
# =====================================================================
fig.text(
    0.5, eq_y,
    r"$\mathcal{F}\!\left[\nabla^{2}\phi(\mathbf{r})\right] = -k^{2}\,\hat{\phi}(\mathbf{k})$"
    "          "
    r"$\hat{\phi}^{\,t+\Delta t}(\mathbf{k}) = "
    r"\dfrac{\hat{\phi}^{\,t}(\mathbf{k}) + \Delta t\,\hat{R}_{\mathrm{explicit}}(\mathbf{k})}"
    r"{1 + \Delta t\, L_{\mathrm{ref}}\,\kappa_{\mathrm{ref}}\, k^{2}}$",
    fontsize=equation_fs, fontweight="bold", color=eq_text_color, ha="center",
    bbox=dict(boxstyle="round,pad=0.5", facecolor=eq_box_color,
              edgecolor=eq_edge_color, linewidth=1.2),
)

plt.tight_layout()


# =====================================================================
# 13. RENDER + DOWNLOAD
# =====================================================================
st.pyplot(fig)

try:
    _buf = io.BytesIO()
    fig.savefig(_buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    st.download_button(
        "⬇️  Download figure (PNG)",
        data=_buf.getvalue(),
        file_name="phase_field_fft_schematic.png",
        mime="image/png",
    )
except Exception as _exc:                     # never break the app on export
    st.caption(f"PNG export unavailable: {_exc}")

plt.close(fig)
