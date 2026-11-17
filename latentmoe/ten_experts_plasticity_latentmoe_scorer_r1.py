"""
LatentMoE Score — Interactive Streamlit Visualization
=====================================================

A self-contained teaching app that shows, with concrete numbers, how the
**LatentMoE score** for 10 experts is computed — a faithful, deterministic
re-implementation of `PlasticityLatentMoEScorer` (v10.x).

The score is a *linear, deterministic mixture of experts*:

    score(c) = Σ_e  w_e · s_e(c)        (e = 1 .. 10 experts)

Because the MoE is linear and deterministic, the per-expert weighted
contribution `w_e · s_e` is an **exact feature attribution** — no SHAP/LIME
approximation is needed.  This app makes that decomposition visible.

Ten experts (weights sum to 1.00):
    v9.x core                    v10.x physics threads
    ---------                    ---------------------
    Material        0.15         Framework (A)    0.15   ← gamma0_dot / mu
    Thermal         0.10         Genesis   (B)    0.15   ← rho0 / sigma0_fric
    Strain          0.05         Thermo-Mech (C)  0.10   ← srs
    Method          0.10         Corpus Density   0.05   ← provenance-binned
    Confidence      0.05
    Reasoning       0.10

Run with:
    streamlit run latentmoe_streamlit_app.py

Numbers below are illustrative — they exist only to visualize how the
score is assembled expert-by-expert.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
#  PART A — THE 10-EXPERT LATENT-MoE SCORER  (verified faithful to source)
#  Mirrors scripts/latentmoe_scorer.py (all unit tests pass).
# ============================================================================

# ── 1. Expert order, weights (Σ = 1.00) and publication colours ────────────
EXPERT_ORDER: Tuple[str, ...] = (
    'Material', 'Thermal', 'Strain', 'Method', 'Confidence',
    'Reasoning', 'Framework (A)', 'Genesis (B)',
    'Thermo-Mech (C)', 'Corpus Density',
)
EXPERT_WEIGHTS: Dict[str, float] = {
    'Material': 0.15, 'Thermal': 0.10, 'Strain': 0.05,
    'Method': 0.10, 'Confidence': 0.05, 'Reasoning': 0.10,
    'Framework (A)': 0.15, 'Genesis (B)': 0.15,
    'Thermo-Mech (C)': 0.10, 'Corpus Density': 0.05,
}
EXPERT_COLORS: Dict[str, str] = {
    'Material': '#E69F00', 'Thermal': '#56B4E9', 'Strain': '#009E73',
    'Method': '#CC79A7', 'Confidence': '#999999', 'Reasoning': '#F0E442',
    'Framework (A)': '#0072B2', 'Genesis (B)': '#D55E00',
    'Thermo-Mech (C)': '#000000', 'Corpus Density': '#88CCEE',
}
# which experts actually fire (non-zero) for each parameter
EXPERT_FIRES_FOR: Dict[str, Tuple[str, ...]] = {
    'rho0':        ('Material', 'Thermal', 'Strain', 'Method', 'Confidence',
                    'Reasoning', 'Genesis (B)', 'Corpus Density'),
    'gamma0_dot':  ('Material', 'Thermal', 'Strain', 'Method', 'Confidence',
                    'Reasoning', 'Framework (A)', 'Corpus Density'),
}

# ── 2. Regime tables + normalizers (for the physics-thread experts) ────────
RHO0_REGIMES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("sputter", "columnar"):       dict(low=1e13, high=1e14, bench=5e13,
        desc="Sputter/PVD columnar — very low residual density"),
    ("sputter", "equiaxed"):       dict(low=1e14, high=1e15, bench=3e14,
        desc="Sputter nanocrystalline — moderate density"),
    ("electrodeposit", "columnar"):dict(low=1e14, high=1e15, bench=3.9e14,
        desc="Electrodeposited columnar — moderate density"),
    ("electrodeposit", "equiaxed"):dict(low=1e16, high=1e17, bench=5e16,
        desc="Equiaxed electrodeposited — high density (organics + H)"),
    ("additive", "equiaxed"):      dict(low=5e16, high=8.4e16, bench=8.38e16,
        desc="Additive-assisted electrodeposited — very high density"),
    ("additive", "columnar"):      dict(low=1e15, high=1e16, bench=5e15,
        desc="Additive + columnar — intermediate"),
    ("mbe", "single"):             dict(low=1e11, high=1e12, bench=5e11,
        desc="MBE single crystal — lowest measurable density"),
    ("electrodeposit", "single"):  dict(low=1e12, high=1e13, bench=5e12,
        desc="Electrodeposited epitaxial — low density"),
}
GAMMA0_DOT_REGIMES: Dict[str, Dict[str, Any]] = {
    "johnson_cook": dict(low=1e-4, high=1e-3, bench=1e-3,
        desc="Johnson-Cook — quasi-static reference rate"),
    "power_law":    dict(low=1e-4, high=1e-3, bench=1e-3,
        desc="Power-law fit — quasi-static reference rate"),
    "cpfem":        dict(low=1e-4, high=1e-3, bench=1e-3,
        desc="Crystal Plasticity FEM — slip-system reference γ̇₀"),
    "ddd":          dict(low=1e3,  high=1e4,  bench=5e3,
        desc="Discrete Dislocation Dynamics — high computational reference"),
    "md":           dict(low=1e6,  high=1e8,  bench=1e7,
        desc="Molecular Dynamics — atomistic reference rate"),
    "phase_field":  dict(low=1e-4, high=1e-2, bench=1e-3,
        desc="Phase-field continuum — quasi-static reference rate"),
    "experimental": dict(low=1e-4, high=1e-3, bench=1e-3,
        desc="Experimental tensile / SHPB — quasi-static reference rate"),
}
_GAMMA0_QUASI_STATIC_GROUP = frozenset(
    {"johnson_cook", "power_law", "experimental", "phase_field"})

# theory phrases the user can pick (left) -> normalized regime key (right)
THEORY_OPTIONS: List[Tuple[str, str]] = [
    ("Crystal Plasticity FEM (cpfem)",          "cpfem"),
    ("Discrete Dislocation Dynamics (ddd)",     "ddd"),
    ("Molecular Dynamics (md)",                 "md"),
    ("Johnson-Cook (johnson_cook)",             "johnson_cook"),
    ("Power-law fit (power_law)",               "power_law"),
    ("Experimental tensile / SHPB (experimental)", "experimental"),
    ("Phase-field (phase_field)",               "phase_field"),
]
# the *phrase* passed to _norm_theory (it matches substrings, not bare keys)
THEORY_PHRASE: Dict[str, str] = {
    "cpfem": "cpfem", "ddd": "ddd", "md": "molecular dynamics",
    "johnson_cook": "johnson-cook", "power_law": "power law",
    "experimental": "tensile test", "phase_field": "phase field",
}


def _norm_synthesis(s: Optional[str]) -> Optional[str]:
    s = (s or "").lower()
    if any(k in s for k in ("sputter", "pvd", "physical vapor")): return "sputter"
    if "mbe" in s or "molecular beam" in s:                      return "mbe"
    if any(k in s for k in ("additive", "additively")):          return "additive"
    if any(k in s for k in ("electrodeposit", "electroplat", "electroform")):
        return "electrodeposit"
    return None


def _norm_architecture(a: Optional[str]) -> Optional[str]:
    a = (a or "").lower()
    if "columnar" in a: return "columnar"
    if "single" in a: return "single"
    if any(k in a for k in ("equiaxed", "nanocrystalline", "nc-cu",
                            "nc cu", "ufg", "ultrafine", "polycrystal")):
        return "equiaxed"
    return None


def _norm_theory(t: Optional[str]) -> Optional[str]:
    t = (t or "").lower()
    if any(k in t for k in ("johnson-cook", "johnson cook", "johnson–cook",
                            "jc model", "jc plasticity", "jc constitutive",
                            "j-c model", "j-c")):                              return "johnson_cook"
    if any(k in t for k in ("power law", "power-law", "empirical",
                            "constitutive fit", "constitutive law",
                            "constitutive model", "curve fit")):               return "power_law"
    if any(k in t for k in ("discrete dislocation", "dislocation dynamics",
                            "ddd", "paradis", "mesoscale dislocation")):        return "ddd"
    if any(k in t for k in ("crystal plasticity", "cpfem", "cp-fem",
                            "cp fem", "taylor model", "taylor polycrystal",
                            "vpsc", "viscoplastic self-consistent",
                            "self-consistent model", "slip system",
                            "slip rate", "reference slip rate",
                            "reference shear rate", "reference shear",
                            "schmid law")):                                     return "cpfem"
    if any(k in t for k in ("molecular dynamics", "md simulation",
                            "atomistic")):                                      return "md"
    if any(k in t for k in ("phase field", "phase-field", "allen-cahn",
                            "cahn-hilliard")):                                  return "phase_field"
    if any(k in t for k in ("tensile test", "compression test",
                            "quasi-static", "shpb", "split hopkinson",
                            "dynamic compression")):                           return "experimental"
    return None


def classify_rho0_regime(synthesis, architecture, twin_spacing_nm):
    syn_key, arch_key = _norm_synthesis(synthesis), _norm_architecture(architecture)
    regime = (RHO0_REGIMES.get((syn_key, arch_key))
              if (syn_key and arch_key) else None)
    if regime is None:
        return dict(regime="unclassified — fallback plausible range",
                    low=1e10, high=1e17, bench=1e12, inferred=1e12, source="fallback")
    if (isinstance(twin_spacing_nm, (int, float)) and twin_spacing_nm
            and twin_spacing_nm > 0 and np.isfinite(twin_spacing_nm)):
        lam = float(twin_spacing_nm)
        log_lam = np.log10(max(lam, 1.0))
        log_5, log_100 = np.log10(5.0), np.log10(100.0)
        pos = float(np.clip(1.0 - (log_lam - log_5) / (log_100 - log_5), 0.10, 0.90))
        log_lo, log_hi = np.log10(regime["low"]), np.log10(regime["high"])
        inferred = float(10.0 ** (log_lo + pos * (log_hi - log_lo)))
    else:
        inferred = regime["bench"]
    return dict(regime=regime["desc"], low=regime["low"], high=regime["high"],
                bench=regime["bench"], inferred=inferred,
                source=f"classifier({syn_key},{arch_key},λ={twin_spacing_nm})")


def classify_gamma0_regime(theory):
    key = _norm_theory(theory)
    if key is None:
        return dict(regime="unclassified — fallback plausible range",
                    low=1e-8, high=1e6, bench=1e-3, inferred=1e-3, source="fallback")
    r = GAMMA0_DOT_REGIMES[key]
    return dict(regime=r["desc"], low=r["low"], high=r["high"],
                bench=r["bench"], inferred=r["bench"], source=f"classifier({key})")


# ── 3. The ten experts — each returns s_e ∈ [0, 1] ─────────────────────────
def s_material(ext_mat, target_mat):
    if not ext_mat: return 0.4
    a, b = ext_mat.lower().strip(), target_mat.lower().strip()
    if a == b: return 1.0
    if a in b or b in a: return 0.85
    fcc = {"cu", "al", "ni", "ag", "au", "pt", "pb", "cocrfeni", "hea"}
    bcc = {"fe", "w", "mo", "cr", "v", "nb"}
    hcp = {"mg", "ti", "zn", "co", "zr"}
    for grp in (fcc, bcc, hcp):
        if a in grp and b in grp: return 0.5
    return 0.25


def s_thermal(ext_temp, target_temp, sigma=100.0):
    if ext_temp is None: return 0.5
    try: t = float(ext_temp)
    except (TypeError, ValueError): return 0.5
    diff = t - float(target_temp)
    return float(np.exp(-(diff ** 2) / (2.0 * sigma ** 2)))


def s_strain(ext_rate, target_rate):
    if ext_rate is None: return 0.5
    try:
        r = float(ext_rate)
        if r <= 0 or target_rate <= 0: return 0.5
        ratio = math.log10(r / target_rate)
        return float(np.exp(-(ratio ** 2) / 8.0))
    except (TypeError, ValueError):
        return 0.5


_METHOD_SCORES = {
    "experiment": 1.0, "review": 0.8,
    "md": 0.6, "molecular dynamics": 0.6,
    "dft": 0.5, "first-principles": 0.5, "ab initio": 0.5,
    "heuristic": 0.3,
    "explicit": 0.9, "llm_inferred": 0.5,
    "prior inference": 0.35, "llm_prior": 0.35,
    "default_fallback": 0.3,
    "hall_petch_intercept": 0.55,
    "taylor_factor_cross_conversion": 0.45,
    "peierls_nabarro_formula": 0.50,
    "side_note_table": 0.55,
    "derived_consensus": 0.55,
    "physics_regime_inference": 0.60,
    "theory_regime_inference": 0.60,
}
_METHOD_OPTIONS = ["experiment", "review", "md", "dft", "explicit",
                   "llm_inferred", "hall_petch_intercept",
                   "derived_consensus", "physics_regime_inference",
                   "default_fallback"]


def s_method(method):
    return _METHOD_SCORES.get((method or "").lower(), 0.5)


_COG_METHODS = (
    "llm_inferred", "heuristic", "default_fallback",
    "prior inference", "llm_prior",
    "hall_petch_intercept", "taylor_factor_cross_conversion",
    "peierls_nabarro_formula", "side_note_table",
    "derived_consensus", "physics_regime_inference",
    "theory_regime_inference",
    "sigma0_route_hall_petch_d", "sigma0_route_hall_petch_lam",
    "sigma0_route_model_inversion", "sigma0_route_peierls_nabarro",
)


def s_reasoning(reasoning, method):
    """0.5 for first-principles methods; for chain-of-thought methods it
    counts reasoning steps and rewards formula citations."""
    method_l = (method or "").lower()
    if method_l not in _COG_METHODS:
        return 0.5
    if not reasoning:
        return 0.2
    n_steps = reasoning.count("Step ") + reasoning.count("\n")
    cites_formula = any(k in reasoning for k in [
        "mu(T)", "sigma0", "rho_0", "gamma0_dot",
        "*", "/", "exp", "log", "GPa", "MPa", "s^-1",
        "synthesis", "architecture", "twin_spacing",
        "Johnson", "JC", "CPFEM", "DDD", "MD",
        "phase-field", "framework", "theory", "Hall", "Peierls",
    ])
    base = min(0.4 + 0.05 * n_steps, 0.8)
    if cites_formula:
        base = min(base + 0.15, 0.85)
    return float(base)


# Thread A — Theory/Framework expert (gamma0_dot only)
def _theory_regime(value_si, target_theory, candidate_theory=None,
                   falloff=1.5, min_sigma=0.25):
    try: v = float(value_si)
    except (TypeError, ValueError): return 0.0
    if v <= 0 or not np.isfinite(v): return 0.0
    regime = classify_gamma0_regime(target_theory)
    log_v = np.log10(v)
    log_lo, log_hi = np.log10(regime["low"]), np.log10(regime["high"])
    log_ctr = np.log10(max(regime["inferred"], 1e-30))
    tag_score = 1.0
    if target_theory and candidate_theory:
        t_t, t_c = _norm_theory(target_theory), _norm_theory(candidate_theory)
        if t_t and t_c:
            if t_t == t_c: tag_score = 1.0
            elif {t_t, t_c} <= _GAMMA0_QUASI_STATIC_GROUP: tag_score = 0.85
            else: tag_score = 0.05
        elif t_c and t_t and t_c != t_t:
            tag_score = 0.10
    if log_lo <= log_v <= log_hi:
        sigma = max((log_hi - log_lo) / 4.0, min_sigma)
        reg = float(np.exp(-(log_v - log_ctr) ** 2 / (2.0 * sigma ** 2)))
    else:
        reg = float(np.exp(-min(abs(log_v - log_lo), abs(log_v - log_hi)) / falloff))
    return tag_score * reg


def s_framework(param, value_si, candidate_theory, target_theory, theory_active):
    """Thread A — fires for gamma0_dot; mu → neutral 0.8; else 0."""
    if param == "gamma0_dot":
        if theory_active and target_theory:
            return _theory_regime(value_si, target_theory, candidate_theory)
        return 0.8
    if param == "mu":
        return 0.8
    return 0.0


# Thread B — Genesis expert (rho0 / sigma0_fric)
def _physics_regime(value_si, synthesis, architecture, twin_spacing_nm,
                    falloff=1.0, min_sigma=0.25):
    try: v = float(value_si)
    except (TypeError, ValueError): return 0.0
    if v <= 0 or not np.isfinite(v): return 0.0
    regime = classify_rho0_regime(synthesis, architecture, twin_spacing_nm)
    log_v = np.log10(v)
    log_lo, log_hi = np.log10(regime["low"]), np.log10(regime["high"])
    log_ctr = np.log10(max(regime["inferred"], 1e-10))
    if log_lo <= log_v <= log_hi:
        sigma = max((log_hi - log_lo) / 4.0, min_sigma)
        return float(np.exp(-(log_v - log_ctr) ** 2 / (2.0 * sigma ** 2)))
    return float(np.exp(-min(abs(log_v - log_lo), abs(log_v - log_hi)) / falloff))


def s_genesis(param, value_si, synthesis, architecture, twin_spacing,
              regime_active, provenance):
    """Thread B — origin of the microstructure. rho0 → physics regime;
    sigma0_fric → physics_inferred check; else 0."""
    if param == "rho0":
        if regime_active:
            return _physics_regime(value_si, synthesis, architecture, twin_spacing)
        return 0.5
    if param == "sigma0_fric":
        return 0.9 if provenance == "physics_inferred" else 0.5
    return 0.0


def s_thermomech(param, value_si, target_temp):
    """Thread C — thermo-mechanical consistency (srs only)."""
    if param != "srs": return 0.0
    try: v = float(value_si)
    except (TypeError, ValueError): return 0.5
    if target_temp and target_temp > 400 and v < 10: return 1.0
    if target_temp and target_temp <= 300 and 15 <= v <= 75: return 1.0
    return 0.5


_PROV_OPTIONS = ["llm_extract", "llm_reasoned", "regex_ner",
                 "physics_inferred", "regime_prior", "llm_prior"]


def s_density(prov_fine):
    """Thread D — corpus evidence density (provenance-binned)."""
    if prov_fine in ("llm_extract", "llm_reasoned", "regex_ner"): return 1.0
    if prov_fine == "physics_inferred": return 0.7
    if prov_fine == "regime_prior": return 0.4
    return 0.2


# ── 4. The scorer: breakdown[e] = w_e · s_e;  total = Σ breakdown ──────────
@dataclass
class TargetContext:
    target_material: str = "Cu"
    target_temp: float = 300.0
    target_strain_rate: float = 1e-3
    synthesis: Optional[str] = None
    architecture: Optional[str] = None
    twin_spacing: Optional[float] = None
    target_theory: Optional[str] = None

    @property
    def regime_active_rho0(self) -> bool:
        return any([
            self.synthesis not in (None, "", "(unknown)"),
            self.architecture not in (None, "", "(unknown)"),
            (isinstance(self.twin_spacing, (int, float))
             and self.twin_spacing is not None
             and self.twin_spacing > 0
             and np.isfinite(self.twin_spacing)),
        ])

    @property
    def theory_active_gdot(self) -> bool:
        return bool(self.target_theory and _norm_theory(self.target_theory))


def score_candidate(ext: Dict[str, Any], ctx: TargetContext):
    """Return (raw_scores s_e, weighted_breakdown w_e·s_e, total)."""
    param = ext["param"]
    value_si = float(ext["value_si"])
    raw = {
        'Material':        s_material(ext.get("material", ""), ctx.target_material),
        'Thermal':         s_thermal(ext.get("temp"), ctx.target_temp),
        'Strain':          s_strain(ext.get("strain_rate"), ctx.target_strain_rate),
        'Method':          s_method(ext.get("method", "unknown")),
        'Confidence':      float(ext.get("confidence", 0.5) or 0.5),
        'Reasoning':       s_reasoning(ext.get("reasoning", ""), ext.get("method", "unknown")),
        'Framework (A)':   s_framework(param, value_si,
                                       ext.get("theory", "unspecified"),
                                       ctx.target_theory, ctx.theory_active_gdot),
        'Genesis (B)':     s_genesis(param, value_si, ctx.synthesis,
                                     ctx.architecture, ctx.twin_spacing,
                                     ctx.regime_active_rho0,
                                     ext.get("provenance", "")),
        'Thermo-Mech (C)': s_thermomech(param, value_si, ctx.target_temp),
        'Corpus Density':  s_density(ext.get("provenance", "")),
    }
    breakdown = {e: EXPERT_WEIGHTS[e] * raw[e] for e in EXPERT_ORDER}
    return raw, breakdown, float(sum(breakdown.values()))


# ============================================================================
#  PART B — ILLUSTRATIVE EXAMPLE CANDIDATES (numbers for visualization only)
# ============================================================================
PARAM_META = {
    'rho0':       dict(symbol='ρ₀',       unit='m⁻²',  label='dislocation density'),
    'gamma0_dot': dict(symbol='γ̇₀',       unit='s⁻¹',  label='reference strain rate'),
}

# Per-candidate reasoning text (read-only; drives the Reasoning expert).
REASONING_TEXT = {
    "C1": "Step 1: extract rho_0 from XRD broadening\n"
          "Step 2: classify synthesis regime\nStep 3: cross-check with sigma0",
    "C2": "Step 1: extract rho_0\nStep 2: classify regime",
    "C3": "Step 1: build MD cell\nStep 2: count dislocation segments",
    "C4": "Step 1: parse text\nStep 2: infer from Hall-Petch\nStep 3: cite sigma0",
    "C5": "",
    "G1": "Step 1: fit CPFEM slip law\nStep 2: calibrate gamma0_dot",
    "G2": "Step 1: DDD simulation\nStep 2: extract gamma0_dot",
    "G3": "Step 1: MD simulation\nStep 2: record strain rate",
    "G4": "",
    "G5": "Step 1: tensile test\nStep 2: extract reference rate",
}

# Default candidate tables (value is in SI units already).
RHO0_DEFAULT = pd.DataFrame([
    dict(candidate="C1", value=5e16, material="Cu", temp_K=300, strain_rate=1e-3,
         method="experiment", confidence=0.95, provenance="llm_extract",
         sample_origin="electrodeposit + equiaxed"),
    dict(candidate="C2", value=1e12, material="Cu", temp_K=300, strain_rate=1e-3,
         method="experiment", confidence=0.90, provenance="llm_extract",
         sample_origin="sputter + columnar (regime mismatch)"),
    dict(candidate="C3", value=5e16, material="Ni", temp_K=350, strain_rate=1e-2,
         method="md", confidence=0.70, provenance="physics_inferred",
         sample_origin="MD cell"),
    dict(candidate="C4", value=1e14, material="Cu", temp_K=300, strain_rate=1e-3,
         method="llm_inferred", confidence=0.50, provenance="llm_prior",
         sample_origin="LLM extract from review"),
    dict(candidate="C5", value=8e16, material="Cu", temp_K=300, strain_rate=1e-3,
         method="review", confidence=0.80, provenance="regex_ner",
         sample_origin="literature review"),
])
GDOT_DEFAULT = pd.DataFrame([
    dict(candidate="G1", value=1e-3, material="Cu", temp_K=300, strain_rate=1e-3,
         method="experiment", confidence=0.95, provenance="llm_extract",
         theory="cpfem"),
    dict(candidate="G2", value=5e3,  material="Cu", temp_K=300, strain_rate=1e-3,
         method="md", confidence=0.85, provenance="physics_inferred",
         theory="ddd"),
    dict(candidate="G3", value=1e7,  material="Cu", temp_K=300, strain_rate=1e-3,
         method="md", confidence=0.80, provenance="physics_inferred",
         theory="md"),
    dict(candidate="G4", value=1e-3, material="Cu", temp_K=300, strain_rate=1e-3,
         method="review", confidence=0.75, provenance="regex_ner",
         theory="johnson_cook"),
    dict(candidate="G5", value=1e-3, material="Cu", temp_K=300, strain_rate=1e-3,
         method="experiment", confidence=0.90, provenance="llm_extract",
         theory="experimental"),
])

MATERIAL_OPTIONS = ["Cu", "Ni", "Al", "Fe", "Mg", "Ti", "W", "CoCrFeNi"]


# ============================================================================
#  PART C — STREAMLIT UI
# ============================================================================
def sidebar_target_context(param: str) -> TargetContext:
    st.sidebar.header("🎯 Target context")
    st.sidebar.caption("The scenario you are selecting parameters for. "
                       "Context-aware gating activates the physics-thread "
                       "experts (Thread A / B).")
    target_material = st.sidebar.selectbox("Target material", MATERIAL_OPTIONS, index=0)
    target_temp = st.sidebar.number_input("Target temperature (K)", value=300,
                                          min_value=0, max_value=2000, step=10)
    target_strain_rate = st.sidebar.number_input(
        "Target strain rate (s⁻¹)", value=1e-3, format="%.1e")

    synthesis = architecture = twin_spacing = None
    target_theory = None
    if param == "rho0":
        st.sidebar.markdown("**Thread B — Genesis (microstructure origin)**")
        activate = st.sidebar.checkbox("Activate Genesis expert "
                                       "(supply synthesis / architecture)",
                                       value=True)
        if activate:
            synthesis = st.sidebar.selectbox(
                "Synthesis route",
                ["electrodeposit", "sputter", "additive", "mbe", "(unknown)"],
                index=0)
            architecture = st.sidebar.selectbox(
                "Architecture",
                ["equiaxed", "columnar", "single", "(unknown)"], index=0)
            twin_spacing = st.sidebar.number_input(
                "Twin spacing λ (nm, 0 = none)", value=0.0, min_value=0.0,
                step=1.0)
            twin_spacing = twin_spacing if twin_spacing > 0 else None
        else:
            st.sidebar.info("Deactivated → Genesis falls back to neutral "
                            "prior s_e = 0.5 for every candidate.")
    else:  # gamma0_dot
        st.sidebar.markdown("**Thread A — Framework (theory target)**")
        activate = st.sidebar.checkbox("Activate Framework expert "
                                       "(supply target theory)", value=True)
        if activate:
            idx = st.sidebar.selectbox(
                "Target theory",
                range(len(THEORY_OPTIONS)),
                format_func=lambda i: THEORY_OPTIONS[i][0], index=0)
            target_theory = THEORY_PHRASE[THEORY_OPTIONS[idx][1]]
        else:
            st.sidebar.info("Deactivated → Framework falls back to neutral "
                            "prior s_e = 0.8 for every candidate.")

    return TargetContext(
        target_material=target_material, target_temp=float(target_temp),
        target_strain_rate=float(target_strain_rate),
        synthesis=synthesis, architecture=architecture,
        twin_spacing=twin_spacing, target_theory=target_theory)


def expert_weights_table(param: str):
    st.subheader("1 · The 10 experts and their weights")
    st.latex(r"\text{score}(c) \;=\; \sum_{e=1}^{10} w_e \cdot s_e(c)"
             r"\qquad \textstyle\sum_e w_e = 1.00")
    fires = set(EXPERT_FIRES_FOR[param])
    rows = []
    for e in EXPERT_ORDER:
        rows.append({
            "expert": e,
            "weight w_e": EXPERT_WEIGHTS[e],
            "fires for this param?": "yes" if e in fires else "no (→ 0)",
            "color": EXPERT_COLORS[e],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({"weight w_e": "{:.2f}"}),
        column_config={
            "expert": st.column_config.TextColumn("Expert", width="medium"),
            "weight w_e": st.column_config.NumberColumn("weight w_e", format="%.2f"),
            "fires for this param?": st.column_config.TextColumn("active?"),
            "color": st.column_config.TextColumn("colour"),
        },
        use_container_width=True, hide_index=True)
    cap = ("Experts marked *no (→ 0)* do not fire for the selected parameter — "
           "e.g. **Framework (A)** only fires for `gamma0_dot` and `mu`; "
           "**Genesis (B)** only fires for `rho0` and `sigma0_fric`; "
           "**Thermo-Mech (C)** only fires for `srs`. "
           "Their weight is therefore 'spent' at 0, so the practical ceiling "
           "for these examples is 0.85.")
    st.caption(cap)


def candidate_editor(param: str) -> pd.DataFrame:
    st.subheader("2 · Example candidate extractions (edit the numbers)")
    st.caption("Each row is one extracted parameter value with its metadata. "
               "Edit any cell — the score recomputes live.")
    default = RHO0_DEFAULT if param == "rho0" else GDOT_DEFAULT
    if f"df_{param}" not in st.session_state:
        st.session_state[f"df_{param}"] = default.copy()
    if st.button("↺ Reset to example numbers"):
        st.session_state[f"df_{param}"] = default.copy()
    df = st.session_state[f"df_{param}"]
    common_cfg = {
        "candidate": st.column_config.TextColumn("ID", disabled=True),
        "value": st.column_config.NumberColumn(
            f"value ({PARAM_META[param]['unit']})", format="%.3e"),
        "material": st.column_config.SelectboxColumn(
            "material", options=MATERIAL_OPTIONS),
        "temp_K": st.column_config.NumberColumn("temp (K)", format="%d"),
        "strain_rate": st.column_config.NumberColumn(
            "strain rate (s⁻¹)", format="%.1e"),
        "method": st.column_config.SelectboxColumn(
            "method", options=_METHOD_OPTIONS),
        "confidence": st.column_config.NumberColumn(
            "confidence", format="%.2f", min_value=0.0, max_value=1.0, step=0.05),
        "provenance": st.column_config.SelectboxColumn(
            "provenance", options=_PROV_OPTIONS),
    }
    if param == "rho0":
        common_cfg["sample_origin"] = st.column_config.TextColumn(
            "sample origin (note)", disabled=True)
    else:
        common_cfg["theory"] = st.column_config.SelectboxColumn(
            "candidate theory", options=[k for _, k in THEORY_OPTIONS])
    edited = st.data_editor(
        df, column_config=common_cfg, num_rows="fixed",
        use_container_width=True, hide_index=True, key=f"editor_{param}")
    st.session_state[f"df_{param}"] = edited
    return edited


def df_to_ext(row, param: str) -> Dict[str, Any]:
    ext = dict(param=param, value_si=float(row["value"]),
               material=str(row["material"]), temp=float(row["temp_K"]),
               strain_rate=float(row["strain_rate"]),
               method=str(row["method"]),
               confidence=float(row["confidence"]),
               provenance=str(row["provenance"]),
               reasoning=REASONING_TEXT.get(str(row["candidate"]), ""))
    if param == "gamma0_dot":
        ext["theory"] = THEORY_PHRASE.get(str(row.get("theory", "")), str(row.get("theory", "")))
    return ext


def breakdown_tables(param: str, df: pd.DataFrame, ctx: TargetContext):
    st.subheader("3 · Per-expert score breakdown (the arithmetic)")
    st.caption("For every candidate: raw expert score `s_e ∈ [0,1]`, weight "
               "`w_e`, and weighted contribution `w_e · s_e`. The row sum is "
               "the candidate's total LatentMoE score.")
    results = []
    for _, row in df.iterrows():
        ext = df_to_ext(row, param)
        raw, bd, total = score_candidate(ext, ctx)
        results.append((row["candidate"], ext, raw, bd, total))
    # rank by total desc
    results.sort(key=lambda r: r[4], reverse=True)
    winner_id = results[0][0]
    cols = st.columns(len(results))
    for col, (cid, ext, raw, bd, total) in zip(cols, results):
        is_win = (cid == winner_id)
        with col:
            title = f"{'⭐ ' if is_win else ''}{cid}"
            st.markdown(f"**{title}**  ·  total = `{total:.4f}`")
            st.caption(f"value = {ext['value_si']:.2e} {PARAM_META[param]['unit']}")
            sub = pd.DataFrame({
                "expert": EXPERT_ORDER,
                "s_e": [round(raw[e], 4) for e in EXPERT_ORDER],
                "w_e": [EXPERT_WEIGHTS[e] for e in EXPERT_ORDER],
                "w_e·s_e": [round(bd[e], 5) for e in EXPERT_ORDER],
            })
            st.dataframe(sub.style.format({"s_e": "{:.4f}", "w_e": "{:.2f}",
                                           "w_e·s_e": "{:.5f}"}),
                         use_container_width=True, hide_index=True, height=260)
    return results, winner_id


def stacked_bar(param: str, results, winner_id: str):
    st.subheader("4 · Stacked LatentMoE score chart")
    st.caption("Bar height = total LatentMoE score; each coloured segment = "
               "one expert's exact weighted contribution `w_e · s_e`. Bars "
               "sorted ascending so the winner is rightmost (orange underline).")
    # sort ascending by total
    asc = sorted(results, key=lambda r: r[4])
    labels = [r[0] for r in asc]
    totals = [r[4] for r in asc]
    fig = go.Figure()
    for e in EXPERT_ORDER:
        heights = [r[3][e] for r in asc]
        if sum(heights) <= 1e-6:
            continue
        fig.add_trace(go.Bar(
            name=e, x=labels, y=heights,
            marker_color=EXPERT_COLORS[e],
            marker_line_color="black", marker_line_width=0.5,
            hovertemplate=f"{e}<br>w·s = %{{y:.5f}}<extra></extra>"))
    fig.update_layout(
        barmode="stack",
        yaxis_title="Total LatentMoE score",
        yaxis_range=[0, 1.0],
        xaxis_title=f"candidate  ({PARAM_META[param]['symbol']} "
                    f"[{PARAM_META[param]['unit']}])",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0.0, font=dict(size=9)),
        margin=dict(l=50, r=20, t=60, b=40),
        height=480,
    )
    # annotate totals; bold + orange the winner
    for lab, tot in zip(labels, totals):
        is_win = (lab == winner_id)
        fig.add_annotation(x=lab, y=tot + 0.02,
                           text=f"{tot:.3f}",
                           showarrow=False,
                           font=dict(size=11, color="#D55E00" if is_win else "black",
                                     family="Arial Black" if is_win else "Arial"))
    # orange underline under the winner bar
    win_idx = labels.index(winner_id)
    fig.add_shape(type="line",
                  xref="x", yref="paper",
                  x0=win_idx - 0.33, x1=win_idx + 0.33,
                  y0=-0.02, y1=-0.02,
                  line=dict(color="#D55E00", width=4))
    st.plotly_chart(fig, use_container_width=True)


def winner_walkthrough(param: str, results, winner_id: str, ctx: TargetContext):
    st.subheader("5 · How the winner's score is assembled (step by step)")
    winner = next(r for r in results if r[0] == winner_id)
    cid, ext, raw, bd, total = winner
    meta = PARAM_META[param]
    st.markdown(
        f"**Candidate {cid}** — {meta['symbol']} = "
        f"`{ext['value_si']:.3e}` {meta['unit']}, material "
        f"`{ext['material']}`, method `{ext['method']}`, "
        f"provenance `{ext['provenance']}`.")
    st.markdown(
        f"`score({cid}) = Σ_e w_e · s_e = "
        + " + ".join(f"{bd[e]:.4f}" for e in EXPERT_ORDER if bd[e] > 1e-9)
        + f" = {total:.4f}`")

    with st.expander("Show the full expert-by-expert walk-through"):
        rows = []
        for e in EXPERT_ORDER:
            rows.append({
                "expert": e,
                "weight w_e": f"{EXPERT_WEIGHTS[e]:.2f}",
                "raw s_e": f"{raw[e]:.4f}",
                "contribution w_e · s_e": f"{bd[e]:.5f}",
            })
        rows.append({"expert": "Σ TOTAL",
                     "weight w_e": f"{sum(EXPERT_WEIGHTS.values()):.2f}",
                     "raw s_e": "—",
                     "contribution w_e · s_e": f"{total:.5f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Why the physics-thread expert fired (context-aware gating)"):
        if param == "rho0":
            if ctx.regime_active_rho0:
                reg = classify_rho0_regime(ctx.synthesis, ctx.architecture,
                                           ctx.twin_spacing)
                st.markdown(
                    f"**Thread B active.** Target synthesis = "
                    f"`{ctx.synthesis}`, architecture = "
                    f"`{ctx.architecture}` → regime "
                    f"`{reg['regime']}`  \n"
                    f"Plausible band: low = `{reg['low']:.1e}`, "
                    f"high = `{reg['high']:.1e}`, inferred centre = "
                    f"`{reg['inferred']:.1e}` {meta['unit']}.  \n"
                    f"Candidate value `{ext['value_si']:.1e}` is scored by a "
                    f"log-Gaussian centred on `{reg['inferred']:.1e}`; "
                    f"values far below the band (e.g. C2's 1e12) get a steep "
                    f"exponential falloff → low Genesis score. This is what "
                    f"makes an electrodeposited+equiaxed ρ₀=5e16 outrank a "
                    f"sputtered+columnar ρ₀=1e12.")
            else:
                st.markdown("**Thread B inactive** (no synthesis context) → "
                            "Genesis returns the neutral prior `s_e = 0.5` "
                            "for every candidate. Toggle it on in the sidebar "
                            "to see context-aware gating.")
        else:
            if ctx.theory_active_gdot:
                reg = classify_gamma0_regime(ctx.target_theory)
                cand_key = _norm_theory(ext.get("theory", ""))
                st.markdown(
                    f"**Thread A active.** Target theory = "
                    f"`{_norm_theory(ctx.target_theory)}` → regime "
                    f"`{reg['regime']}`  \n"
                    f"Plausible band: low = `{reg['low']:.1e}`, "
                    f"high = `{reg['high']:.1e}`, inferred centre = "
                    f"`{reg['inferred']:.1e}` s⁻¹.  \n"
                    f"Candidate theory tag = `{cand_key}`. The tag_score "
                    f"penalizes mismatched theory families (e.g. a DDD-tagged "
                    f"γ̇₀ vs a CPFEM target → tag_score = 0.05) even when the "
                    f"value is in-band. This is what makes a DDD-tagged "
                    f"γ̇₀=5e3 outrank an MD-tagged γ̇₀=1e7 for a CPFEM target "
                    f"(both are far out of band, but 5e3 is log-closer).")
            else:
                st.markdown("**Thread A inactive** (no target theory) → "
                            "Framework returns the neutral prior `s_e = 0.8` "
                            "for every candidate. Toggle it on in the sidebar "
                            "to see context-aware gating.")

    with st.expander("Reasoning text driving the Reasoning expert"):
        txt = REASONING_TEXT.get(cid, "")
        if not txt:
            st.markdown(f"_{cid} has no reasoning text._ For "
                        f"`{ext['method']}` (not a chain-of-thought method) "
                        f"the Reasoning expert returns the neutral 0.5; for "
                        f"CoT methods with empty reasoning it returns 0.2.")
        else:
            st.text(txt)
            n_steps = txt.count("Step ") + txt.count("\n")
            cites = any(k in txt for k in [
                "mu(T)", "sigma0", "rho_0", "gamma0_dot", "Hall", "Peierls",
                "CPFEM", "DDD", "MD", "synthesis", "sigma0"])
            st.markdown(f"method = `{ext['method']}` → CoT-eligible? "
                        f"`{ext['method'] in _COG_METHODS}`  ·  "
                        f"step count = `{n_steps}`  ·  cites formula? "
                        f"`{cites}`  →  `s_reasoning = {raw['Reasoning']:.3f}`")


def main():
    st.set_page_config(page_title="LatentMoE Score — 10-expert visualization",
                       page_icon="🧮", layout="wide")
    st.title("🧮 LatentMoE Score — how 10 experts compute one rank")
    st.markdown(
        "An interactive, **numerically worked** illustration of the Latent "
        "Mixture-of-Experts (LatentMoE) scorer: ten experts, each emitting a "
        "score `s_e ∈ [0,1]`, combined by a fixed weight vector `w_e` that "
        "sums to 1.00.  \n"
        "Because the combination is **linear and deterministic**, the "
        "per-expert contribution `w_e · s_e` is an *exact* feature "
        "attribution — the coloured segments of the bar chart below literally "
        "sum to the total score.")
    st.info("ℹ️ All candidate numbers below are illustrative — they exist "
            "only to visualize how the score is assembled. The scorer logic "
            "mirrors `PlasticityLatentMoEScorer` (v10.x) and was verified "
            "against the source by unit test.")

    param = st.radio(
        "Parameter to rank candidates for",
        options=["rho0", "gamma0_dot"],
        format_func=lambda p: f"{PARAM_META[p]['symbol']}  —  "
                               f"{PARAM_META[p]['label']}  "
                               f"[{PARAM_META[p]['unit']}]",
        horizontal=True)
    st.caption(f"ρ₀ triggers the **Genesis (B)** physics thread; γ̇₀ triggers "
               f"the **Framework (A)** physics thread. Switch between them to "
               f"see context-aware gating in action.")

    ctx = sidebar_target_context(param)
    expert_weights_table(param)
    df = candidate_editor(param)
    results, winner_id = breakdown_tables(param, df, ctx)
    stacked_bar(param, results, winner_id)
    winner_walkthrough(param, results, winner_id, ctx)

    st.divider()
    st.subheader("Reference — expert scoring rules at a glance")
    rules = {
        "Material (0.15)":        "string match: exact 1.0 · substring 0.85 · same lattice family 0.5 · else 0.25 · empty 0.4",
        "Thermal (0.10)":         "Gaussian on |ΔT|, σ = 100 K: exp(−ΔT²/2σ²); missing → 0.5",
        "Strain (0.05)":          "log-Gaussian on log₁₀(Ṫ/Ṫ_target), σ²=8: exp(−ratio²/8); missing → 0.5",
        "Method (0.10)":          "lookup: experiment 1.0 · review 0.8 · explicit 0.9 · md 0.6 · dft 0.5 · llm_inferred 0.5 · default_fallback 0.3",
        "Confidence (0.05)":      "raw extractor confidence, clamped to [0,1]; missing → 0.5",
        "Reasoning (0.10)":       "non-CoT methods → 0.5; CoT methods: 0.4 + 0.05·n_steps (+0.15 if cites a formula), capped 0.85; empty → 0.2",
        "Framework (A) (0.15)":   "Thread A — γ̇₀: TheoryRegimeExpert (log-Gaussian on theory band × theory-tag penalty); mu → 0.8; inactive → 0.8",
        "Genesis (B) (0.15)":     "Thread B — ρ₀: PhysicsRegimeExpert (log-Gaussian on synthesis×architecture band); σ0_fric: 0.9 if physics_inferred else 0.5; inactive → 0.5",
        "Thermo-Mech (C) (0.10)": "Thread C — srs only: 1.0 if (T>400 & srs<10) or (T≤300 & 15≤srs≤75); else 0.5; other params → 0",
        "Corpus Density (0.05)":  "Thread D — provenance bin: llm_extract/llm_reasoned/regex_ner → 1.0 · physics_inferred → 0.7 · regime_prior → 0.4 · other → 0.2",
    }
    st.table(pd.DataFrame({"expert (weight)": list(rules.keys()),
                           "scoring rule": list(rules.values())}))


if __name__ == "__main__":
    main()
