"""
LatentMoE Clock Diagram — Streamlit + Plotly
============================================
A standalone Python/Streamlit port of the HTML clock diagram.

Draws the full pipeline:
    Input candidate → Pre-gate (norm + clamp + provenance)
       → 10 experts (s_e × w_e) arranged radially around a central Σ node
       → Physical gates (context-aware gating + pin-list + context flag)
       → Ranked output

Run:
    streamlit run latentmoe_clock_streamlit.py
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# 1. THE 10 EXPERTS — order, weights, threads, s_e rules
#    Mirrors PlasticityLatentMoEScorer.EXPERT_ORDER + EXPERT_COLORS
# ============================================================================
EXPERTS: List[Dict] = [
    dict(name="Material",        w=0.15, thread="core",
         rule="string match: exact 1.0 · substring 0.85 · family 0.5 · else 0.25"),
    dict(name="Thermal",         w=0.10, thread="core",
         rule="Gaussian on ΔT: exp(−ΔT²/2σ²), σ=100 K"),
    dict(name="Strain",          w=0.05, thread="core",
         rule="log-Gaussian on log₁₀(Ṫ/Ṫ_t): exp(−r²/8)"),
    dict(name="Method",          w=0.10, thread="core",
         rule="lookup: exp=1.0 · rev=0.8 · md=0.6 · phys_reg=0.60 · prior=0.35"),
    dict(name="Confidence",      w=0.05, thread="core",
         rule="pass-through of extractor conf (clamped by provenance cap)"),
    dict(name="Reasoning",       w=0.10, thread="core",
         rule="CoT: 0.4 + 0.05·n_steps + 0.15 formula bonus, cap 0.85"),
    dict(name="Framework (A)",   w=0.15, thread="physics",
         rule="tag_score × log-Gaussian on theory band"),
    dict(name="Genesis (B)",     w=0.15, thread="physics",
         rule="log-Gaussian on (synth × arch) regime band"),
    dict(name="Thermo-Mech (C)", w=0.10, thread="physics",
         rule="T>400 & m<10, or T≤300 & 15≤m≤75 → 1.0"),
    dict(name="Corpus Density",  w=0.05, thread="physics",
         rule="prov bin: regex=1.0 · phys_inf=0.7 · regime=0.4 · prior=0.2"),
]

# ============================================================================
# 2. WORKED EXAMPLES — concrete s_e values per parameter
#    Numbers come from running PlasticityLatentMoEScorer.score() on the
#    illustrative candidate.
# ============================================================================
WORKED_EXAMPLES: Dict[str, Dict] = {
    "rho0": dict(
        title="ρ₀ = 5×10¹⁶ m⁻²",
        candidate="Cu · electrodeposit · equiaxed · λ unknown · "
                  "method=physics_regime_inference · _provenance=llm_prior",
        s_e=[1.00, 1.00, 1.00, 0.60, 0.45, 0.85, 0.00, 1.00, 0.00, 0.20],
        # Material 1.0 (Cu==Cu) · Thermal 1.0 (ΔT=0) · Strain 1.0 (rate match)
        # Method 0.60 (physics_regime_inference) · Confidence 0.45 (raw)
        # Reasoning 0.85 (4 steps + formula bonus) · Framework 0 (param≠γ̇₀/μ)
        # Genesis 1.0 (at regime center) · Thermo-Mech 0 (param≠srs)
        # Corpus Density 0.2 (llm_prior bucket)
        fires_note="Framework A → 0 (param ≠ γ̇₀/μ) · "
                   "Thermo-Mech C → 0 (param ≠ srs) · "
                   "Genesis B fires (regime_active=True)",
        pin_list="INCUMBENT_ROUTES['rho0'] = ('deterministic',) — "
                 "llm_grounded candidates become grey context bars",
        output_lines=[
            "C1 ρ₀=5e16 · deterministic · 0.628  →  ⭐ winner",
            "C5 ρ₀=8e16 · deterministic · 0.586  →  ranked #2",
            "C3 ρ₀=5e16 · llm_extract · 0.660    →  grey context bar",
            "C2 ρ₀=1e12 · llm_extract · 0.080    →  grey context bar",
        ],
    ),
    "gamma0_dot": dict(
        title="γ̇₀ = 1×10⁻³ s⁻¹",
        candidate="Cu · T=300 K · target_theory=cpfem · "
                  "method=explicit · _provenance=llm_extract · theory=cpfem",
        s_e=[1.00, 1.00, 1.00, 0.90, 0.95, 0.50, 1.00, 0.00, 0.00, 1.00],
        # Material 1.0 · Thermal 1.0 · Strain 1.0 · Method 0.90 (explicit)
        # Confidence 0.95 · Reasoning 0.5 (non-CoT method, returns neutral)
        # Framework 1.0 (CPFEM tag match + value in band)
        # Genesis 0 (param≠ρ₀/σ₀_fric) · Thermo-Mech 0 (param≠srs)
        # Corpus Density 1.0 (llm_extract bucket)
        fires_note="Genesis B → 0 (param ≠ ρ₀/σ₀_fric) · "
                   "Thermo-Mech C → 0 (param ≠ srs) · "
                   "Framework A fires (theory_active=True, tag match)",
        pin_list="INCUMBENT_ROUTES['gamma0_dot'] = ('llm_grounded',) — "
                 "llm_extract candidates can win",
        output_lines=[
            "G1 γ̇₀=1e-3 · llm_grounded · 0.688       →  ⭐ winner",
            "G4 γ̇₀=1e-3 · regex_ner · 0.620          →  grey context bar",
            "G2 γ̇₀=5e3 · physics_inferred · 0.450    →  grey context bar",
            "G3 γ̇₀=1e7 · physics_inferred · 0.380    →  grey context bar",
        ],
    ),
}

# ============================================================================
# 3. GEOMETRY — clock layout constants (CSS-pixel coordinates, Y up)
# ============================================================================
CANVAS_W = 1440
CANVAS_H = 1320

# Top stage boxes
TOP_BOX_Y0, TOP_BOX_Y1 = 1130, 1230
INPUT_BOX   = dict(x0=60,  x1=660,  y0=TOP_BOX_Y0, y1=TOP_BOX_Y1)
PREGATE_BOX = dict(x0=780, x1=1380, y0=TOP_BOX_Y0, y1=TOP_BOX_Y1)

# Clock section
CLOCK_CX, CLOCK_CY = 700, 670
CLOCK_R = 370                  # radius for expert card centers
CARD_W, CARD_H = 210, 110

# Center Σ node (260×100 centered at (700, 670))
CENTER_NODE = dict(x0=570, y0=620, x1=830, y1=720)

# Bottom stage boxes
BOT_BOX_Y0, BOT_BOX_Y1 = 60, 200
GATES_BOX  = dict(x0=60,  x1=660,  y0=BOT_BOX_Y0, y1=BOT_BOX_Y1)
OUTPUT_BOX = dict(x0=780, x1=1380, y0=BOT_BOX_Y0, y1=BOT_BOX_Y1)


def expert_card_center(i: int, n: int = 10) -> Tuple[float, float]:
    """Return (x, y) center of expert i, arranged clockwise from top."""
    angle = math.radians(90 - i * (360 / n))   # i=0 → top, i=5 → bottom
    return CLOCK_CX + CLOCK_R * math.cos(angle), CLOCK_CY + CLOCK_R * math.sin(angle)


def wrap_rule(rule: str, max_chars: int = 48) -> str:
    """Greedy word-wrap into 1–2 lines separated by <br>."""
    if len(rule) <= max_chars:
        return rule
    words = rule.split()
    line1, line2 = "", ""
    for w in words:
        if len(line1) + len(w) + 1 <= max_chars:
            line1 = (line1 + " " + w) if line1 else w
        else:
            line2 = (line2 + " " + w) if line2 else w
    return f"{line1}<br>{line2}"


# ============================================================================
# 4. BUILD THE PLOTLY FIGURE
# ============================================================================
def build_clock_figure(param_key: str = "rho0") -> go.Figure:
    example = WORKED_EXAMPLES[param_key]
    s_e = example["s_e"]
    total = sum(EXPERTS[i]["w"] * s_e[i] for i in range(10))

    shapes: List[Dict] = []
    annotations: List[Dict] = []

    # ── Helper to add a stage box ────────────────────────────────────────
    def add_stage_box(box: Dict, label: str, title: str, body: str,
                      body_family: str = "sans-serif") -> None:
        shapes.append(dict(
            type="rect", **box,
            line=dict(color="#475569", width=1.5),
            fillcolor="#FFFFFF",
        ))
        cx = (box["x0"] + box["x1"]) / 2
        annotations.append(dict(
            x=cx, y=box["y1"] - 12,
            text=f"<b>{label}</b>",
            showarrow=False, font=dict(size=10, color="#64748B"),
        ))
        annotations.append(dict(
            x=cx, y=box["y1"] - 32,
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(size=13, color="#1F2937"),
        ))
        annotations.append(dict(
            x=cx, y=box["y0"] + 22,
            text=body,
            showarrow=False,
            font=dict(size=10.5, color="#475569",
                      family=body_family if body_family != "sans-serif"
                      else "sans-serif"),
            align="center",
        ))

    # ── Helper to add an arrowhead ───────────────────────────────────────
    def add_arrowhead(x: float, y: float, dx: float = -10, dy: float = 0) -> None:
        annotations.append(dict(
            x=x, y=y, ax=x + dx, ay=y + dy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=2.2,
            arrowcolor="#1E293B", arrowsize=1.2,
        ))

    # ── Helper to add a labeled straight arrow ───────────────────────────
    def add_straight_arrow(x0: float, y0: float, x1: float, y1: float,
                           label: str = "") -> None:
        shapes.append(dict(
            type="line", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#1E293B", width=2.2),
        ))
        if label:
            annotations.append(dict(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 10,
                text=label,
                showarrow=False,
                font=dict(size=10, color="#64748B", family="monospace"),
            ))
        # Arrowhead at the (x1, y1) end
        dx, dy = x1 - x0, y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            add_arrowhead(x1, y1, dx=-10 * dx / length, dy=-10 * dy / length)

    # ── Helper to add a polyline path ────────────────────────────────────
    def add_polyline(points: List[Tuple[float, float]],
                     arrow_at_end: bool = True) -> None:
        path = "M " + " L ".join(f"{x},{y}" for x, y in points)
        shapes.append(dict(
            type="path", path=path,
            line=dict(color="#1E293B", width=2.2),
            fillcolor="rgba(0,0,0,0)",
        ))
        if arrow_at_end:
            x1, y1 = points[-1]
            x0, y0 = points[-2]
            dx, dy = x1 - x0, y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                add_arrowhead(x1, y1,
                              dx=-10 * dx / length,
                              dy=-10 * dy / length)

    # ── Top boxes: Input + Pre-gate ──────────────────────────────────────
    if param_key == "rho0":
        input_body = ("param=<b>rho0</b><br>"
                      "value=<b>5e16</b>, unit=m⁻²<br>"
                      "material=Cu, temp=300, strain_rate=1e-3<br>"
                      "method=physics_regime_inference<br>"
                      "_provenance=llm_prior")
        pregate_body = ("_normalize_unit(5e16, 'm⁻²', 'rho0') → v_si=5e16<br>"
                        "_pl_clamp(v_si, 'rho0') → [1e10, 1e17]; not clamped<br>"
                        "_norm_provenance('physics_regime_inference')<br>"
                        "  → 'regime_prior' (matches physics_regime hint)<br>"
                        "_stamp_provenance(ext, 'regime_prior')")
    else:
        input_body = ("param=<b>gamma0_dot</b><br>"
                      "value=<b>1e-3</b>, unit=s⁻¹<br>"
                      "material=Cu, temp=300, strain_rate=1e-3<br>"
                      "method=explicit, theory=cpfem<br>"
                      "_provenance=llm_extract")
        pregate_body = ("_normalize_unit(1e-3, 's⁻¹', 'gamma0_dot') → v_si=1e-3<br>"
                        "_pl_clamp(v_si, 'gamma0_dot') → [1e-8, 1e6]; not clamped<br>"
                        "_norm_provenance('explicit')<br>"
                        "  → 'llm_extract' (matches 'explicit' hint)<br>"
                        "_stamp_provenance(ext, 'llm_extract')")

    add_stage_box(INPUT_BOX, "STAGE 1 · INPUT",
                  "Candidate extraction (ext dict)", input_body)
    add_stage_box(PREGATE_BOX, "STAGE 2 · PRE-GATE",
                  "Unit norm · Plausibility clamp · Provenance canonicalize",
                  pregate_body, body_family="monospace")

    # Arrow: Input → Pre-gate
    arrow_y = (INPUT_BOX["y0"] + INPUT_BOX["y1"]) / 2
    add_straight_arrow(INPUT_BOX["x1"] + 5, arrow_y,
                       PREGATE_BOX["x0"] - 5, arrow_y,
                       label="ext dict")

    # Arrow: Pre-gate → Center (polyline routed AROUND the top expert card)
    pg_cx = (PREGATE_BOX["x0"] + PREGATE_BOX["x1"]) / 2
    pg_by = PREGATE_BOX["y0"]
    bypass_x = 460  # left of top expert card (which sits at x=595..805)
    center_cy = (CENTER_NODE["y0"] + CENTER_NODE["y1"]) / 2
    center_left = CENTER_NODE["x0"]
    add_polyline([
        (pg_cx, pg_by),       # pre-gate bottom-center
        (pg_cx, 1080),        # bend down
        (bypass_x, 1080),     # go LEFT past the top expert card
        (bypass_x, center_cy),# down to center node level
        (center_left, center_cy),  # enter center from the LEFT
    ])

    # ── Center Σ node ────────────────────────────────────────────────────
    shapes.append(dict(
        type="rect", **CENTER_NODE,
        line=dict(color="#1E293B", width=0),
        fillcolor="#1E293B",
    ))
    cx_node = (CENTER_NODE["x0"] + CENTER_NODE["x1"]) / 2
    cy_node = (CENTER_NODE["y0"] + CENTER_NODE["y1"]) / 2
    annotations += [
        dict(x=cx_node, y=cy_node + 18, text="<b>Σ</b>",
             showarrow=False, font=dict(size=28, color="#FFFFFF")),
        dict(x=cx_node, y=cy_node - 4,
             text="score = Σ w<sub>e</sub> · s<sub>e</sub>",
             showarrow=False,
             font=dict(size=12, color="#CBD5E1", family="monospace")),
        dict(x=cx_node, y=cy_node - 25,
             text=f"<b>{total:.3f}</b>",
             showarrow=False,
             font=dict(size=16, color="#FACC15", family="monospace")),
    ]

    # ── 10 expert cards + radial spokes ──────────────────────────────────
    for i, expert in enumerate(EXPERTS):
        cx, cy = expert_card_center(i)
        x0, y0 = cx - CARD_W / 2, cy - CARD_H / 2
        x1, y1 = cx + CARD_W / 2, cy + CARD_H / 2

        s = s_e[i]
        w = expert["w"]
        ws = w * s
        is_physics = expert["thread"] == "physics"
        is_active = s > 0

        # Radial spoke — from center node edge to card edge
        dx, dy = cx - CLOCK_CX, cy - CLOCK_CY
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            ux, uy = dx / dist, dy / dist
            # Guard against division by zero when the card is directly
            # above/below/left/right of the center (ux==0 or uy==0).
            cHalfW = (CENTER_NODE["x1"] - CENTER_NODE["x0"]) / 2
            cHalfH = (CENTER_NODE["y1"] - CENTER_NODE["y0"]) / 2
            tCx = cHalfW / abs(ux) if abs(ux) > 1e-9 else float("inf")
            tCy = cHalfH / abs(uy) if abs(uy) > 1e-9 else float("inf")
            tC = min(tCx, tCy)
            sx, sy = CLOCK_CX + tC * ux, CLOCK_CY + tC * uy
            tEx = (CARD_W / 2) / abs(ux) if abs(ux) > 1e-9 else float("inf")
            tEy = (CARD_H / 2) / abs(uy) if abs(uy) > 1e-9 else float("inf")
            tE = min(tEx, tEy)
            tx, ty = cx - tE * ux, cy - tE * uy

            shapes.append(dict(
                type="line", x0=sx, y0=sy, x1=tx, y1=ty,
                line=dict(
                    color="#CBD5E1" if not is_active else "#94A3B8",
                    width=1.2 if not is_active else 1.6,
                    dash="dot" if not is_active else "dash",
                ),
            ))

        # Card rectangle
        if is_physics:
            line_color = "#1E3A8A" if is_active else "#94A3B8"
            line_width = 2.5 if is_active else 1.5
            fill = "#F8FAFC" if is_active else "#FAFAFA"
            dash = "solid" if is_active else "dash"
        else:
            line_color = "#475569"
            line_width = 1.5
            fill = "#FFFFFF"
            dash = "solid"

        shapes.append(dict(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=line_color, width=line_width, dash=dash),
            fillcolor=fill,
        ))

        # Card text
        name_color = "#1E3A8A" if is_physics else "#1F2937"
        if not is_active:
            name_color = "#94A3B8"
        annotations.append(dict(
            x=x0 + 8, y=y1 - 12,
            text=f"<b>{i+1}. {expert['name']}</b>",
            showarrow=False, font=dict(size=11, color=name_color),
            align="left",
        ))
        annotations.append(dict(
            x=x1 - 8, y=y1 - 12,
            text=f"w={w:.2f}",
            showarrow=False,
            font=dict(size=10, color="#64748B", family="monospace"),
            align="right",
        ))
        annotations.append(dict(
            x=cx, y=cy + 4,
            text=wrap_rule(expert["rule"]),
            showarrow=False,
            font=dict(size=9,
                      color="#94A3B8" if not is_active else "#475569"),
            align="center",
        ))

        # Contribution badge
        if is_active:
            badge_bg, badge_fg = "#FACC15", "#422006"
            badge_text = f"w·s = {ws:.3f}"
        else:
            badge_bg, badge_fg = "#F1F5F9", "#94A3B8"
            badge_text = "w·s = 0.000"
        annotations.append(dict(
            x=x0 + 8, y=y0 + 12,
            text=f"<b>{badge_text}</b>",
            showarrow=False,
            font=dict(size=10, color=badge_fg, family="monospace"),
            align="left",
            bgcolor=badge_bg,
            borderpad=3,
        ))

    # Arrow: Center → Gates (polyline routed AROUND the bottom expert card)
    center_cx = (CENTER_NODE["x0"] + CENTER_NODE["x1"]) / 2
    center_by = CENTER_NODE["y0"]
    bypass2_x = 380  # left of bottom expert card (x=595..805)
    gates_cx = (GATES_BOX["x0"] + GATES_BOX["x1"]) / 2
    gates_ty = GATES_BOX["y1"]
    add_polyline([
        (center_cx, center_by),         # center node bottom-center
        (center_cx, 540),               # bend down (above bottom card top Y=415)
        (bypass2_x, 540),               # go LEFT past the bottom expert card
        (bypass2_x, gates_ty),          # down to gates box top
        (gates_cx, gates_ty),           # enter gates box from above
    ])

    # ── Bottom boxes: Gates + Output ─────────────────────────────────────
    gates_body = (f"<b>(a) Context-aware gating — zeroes s<sub>e</sub>, not w<sub>e</sub>:</b><br>"
                  f"{example['fires_note']}<br><br>"
                  f"<b>(b) Pin-list:</b><br>"
                  f"{example['pin_list']}<br><br>"
                  f"<b>(c) Context flag:</b><br>"
                  f"_is_context(param, prov) stamps c.context=True<br>"
                  f"for non-pinned-bucket candidates")
    add_stage_box(GATES_BOX, "STAGE 4 · PHYSICAL GATES",
                  "Context-aware gating · Pin-list · Context flag",
                  gates_body)

    output_body = "<br>".join(example["output_lines"])
    add_stage_box(OUTPUT_BOX, "STAGE 5 · OUTPUT",
                  "Ranked candidates · Winner selection",
                  output_body, body_family="monospace")

    # Arrow: Gates → Output
    arrow_y2 = (GATES_BOX["y0"] + GATES_BOX["y1"]) / 2
    add_straight_arrow(GATES_BOX["x1"] + 5, arrow_y2,
                       OUTPUT_BOX["x0"] - 5, arrow_y2,
                       label="ranked list")

    # ── Title + subtitle ────────────────────────────────────────────────
    annotations += [
        dict(x=CANVAS_W / 2, y=CANVAS_H - 30,
             text="<b>LatentMoE Value Quantification — 10-Expert Clock Diagram</b>",
             showarrow=False, font=dict(size=22, color="#0F172A")),
        dict(x=CANVAS_W / 2, y=CANVAS_H - 55,
             text=f"Worked example: {example['title']} · {example['candidate']}",
             showarrow=False, font=dict(size=12, color="#64748B")),
    ]

    # ── Build the figure ────────────────────────────────────────────────
    fig = go.Figure()
    # Invisible trace (required to keep the plot area alive)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#FFFFFF", size=1),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, CANVAS_W]),
        yaxis=dict(visible=False, range=[0, CANVAS_H], scaleanchor="x"),
        width=1100, height=1000,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


# ============================================================================
# 5. STREAMLIT UI
# ============================================================================
def main() -> None:
    st.set_page_config(page_title="LatentMoE Clock Diagram", layout="wide")
    st.title("🧮 LatentMoE Value Quantification — 10-Expert Clock Diagram")
    st.markdown(
        "A Streamlit + Plotly port of the static clock diagram. Shows how a "
        "candidate material property value flows through **10 experts** "
        "(each emitting s<sub>e</sub> ∈ [0,1] weighted by a fixed "
        "w<sub>e</sub>), gets summed at the central Σ node, then passes "
        "through **physical gates** to produce the final ranked output.")

    st.sidebar.header("Worked example")
    param_key = st.sidebar.radio(
        "Parameter",
        options=["rho0", "gamma0_dot"],
        format_func=lambda p: {
            "rho0":        "ρ₀  —  dislocation density (m⁻²)",
            "gamma0_dot":  "γ̇₀  —  reference strain rate (s⁻¹)",
        }[p],
        help="Switch between the two parameters that trigger different "
             "physics-thread experts (Genesis B for ρ₀, Framework A for γ̇₀).",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Legend**")
    st.sidebar.markdown(
        "• **White card** = v9.x core expert (6)\n"
        "• **Blue-bordered card** = v10.x physics thread (4)\n"
        "• **Dashed border + grey** = inactive for this param "
        "(s<sub>e</sub> = 0)\n"
        "• **Yellow badge** = firing w<sub>e</sub>·s<sub>e</sub> contribution\n"
        "• **Grey badge** = gated to zero (w<sub>e</sub>·s<sub>e</sub> = 0)"
    )

    fig = build_clock_figure(param_key)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": False, "staticPlot": False,
    })

    # ── Per-expert breakdown table ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Per-expert breakdown")
    example = WORKED_EXAMPLES[param_key]
    rows = []
    for i, expert in enumerate(EXPERTS):
        s = example["s_e"][i]
        w = expert["w"]
        ws = w * s
        rows.append({
            "#":             i + 1,
            "Expert":        expert["name"],
            "Thread":        expert["thread"],
            "w_e":           w,
            "s_e":           s,
            "w_e · s_e":     round(ws, 4),
            "fires?":        "yes" if s > 0 else "no (gated → 0)",
        })
    rows.append({
        "#": "Σ", "Expert": "TOTAL", "Thread": "—",
        "w_e": round(sum(e["w"] for e in EXPERTS), 2),
        "s_e": "—",
        "w_e · s_e": round(example["s_e"][0] * EXPERTS[0]["w"]
                           + sum(EXPERTS[i]["w"] * example["s_e"][i]
                                 for i in range(1, 10)), 4),
        "fires?": "—",
    })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Why the winner emerges (short narrative) ────────────────────────
    st.markdown("---")
    st.subheader("Why the winner emerges")
    firing = [(i, EXPERTS[i], example["s_e"][i]) for i in range(10)
              if example["s_e"][i] > 0]
    non_firing = [(i, EXPERTS[i]) for i in range(10)
                  if example["s_e"][i] == 0]
    total = sum(EXPERTS[i]["w"] * example["s_e"][i] for i in range(10))

    st.markdown(
        f"**Firing experts ({len(firing)}/10):** "
        + ", ".join(f"{e['name']} (s={s:.2f}, w·s={e['w']*s:.3f})"
                    for _, e, s in firing)
        + f"\n\n**Gated to zero ({len(non_firing)}/10):** "
        + ", ".join(e["name"] for _, e in non_firing)
        + f"\n\n**Total score = Σ w_e · s_e = {total:.3f}**"
        + (f"  (ceiling for this param = "
           f"{sum(EXPERTS[i]['w'] for i in range(10) if example['s_e'][i] > 0 or EXPERTS[i]['thread'] == 'core'):.2f})"
           if non_firing else "")
    )


if __name__ == "__main__":
    main()
