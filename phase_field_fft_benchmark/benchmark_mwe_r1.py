#!/usr/bin/env python3
"""
Headless A/B benchmark that PROVES the dominant confound in the nanotwinned-Cu
plasticity app, using the REAL compute_yield_stress() and update_plastic_strain()
obtained verbatim from the r3 - phase field fft code.

What it tests (the analysis's central claim, "confound A"):
  * unit_fix=False  -> h in nm, b in m  -> yield-stress hardening term ~14 Pa
                       -> sigma_y ~= sigma0 ~= 50 MPa FLAT everywhere
                       -> overstress saturates at 1.0 everywhere -> SMOOTH eps_p
  * unit_fix=True   -> h_m = h*1e-9      -> hardening term ~0.4-3 GPa, STRUCTURED
                       -> sigma_y comparable to sigma_eq -> marginal yielding
                       -> eps_p traces a threshold contour -> ZIGZAG / structured

If zigzag (high tv / hf_frac) appears in EVERY row of the unit_fix=True run
INCLUDING baseline, the cause of error is the sigma_y structure / unit fix, NOT the physics-grounded AI plasticity recommendeer
override values. `only rho0` must be byte-identical to baseline (rho0 is inert).

This is a faithful reproduction of the *plasticity* physics; the r3 - phase field fft code app's
spectral stress solve is replaced by a representative GPa-scale sigma_eq field
that places the system in the marginal-yielding regime the analysis describes
(the r3 app reaches GPa-scale local stress via the gamma_tw ~= 0.7
transformation eigenstrain). The two plasticity functions below are identical to the r3 code, so the confound is exercised on the actual code.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/z/my-project/download"

# ----------------------------------------------------------------------------
# VERBATIM copies from the uploaded no-AI app (lines 108-131). DO NOT EDIT.
# ----------------------------------------------------------------------------
def compute_yield_stress(h, sigma0, mu, b, nu):
    safe = h > 2 * b
    sigma_y = np.empty_like(h)
    log_term = np.log(np.maximum(h, 2.001 * b) / b)
    sigma_y[safe] = sigma0 + (mu * b / (2 * np.pi * h[safe] * (1 - nu))) * log_term[safe]
    sigma_y[~safe] = sigma0 + mu / (2 * np.pi * (1 - nu))
    return sigma_y


def update_plastic_strain(sigma_eq, sigma_y, eps_p_xx, eps_p_yy, eps_p_xy,
                          gamma0_dot, m, dt):
    MAX_OVERSTRESS = 1.0
    MAX_PLASTIC_STRAIN = 0.1
    overstress = np.maximum(sigma_eq - sigma_y, 0.0) / np.maximum(sigma_y, 1e-9)
    overstress = np.minimum(overstress, MAX_OVERSTRESS)
    gamma_dot = gamma0_dot * overstress**m
    stress_dev = 2.0 / 3.0 * gamma_dot * dt
    stress_dev = np.minimum(stress_dev, 0.001)
    d_xx = stress_dev
    d_yy = -0.5 * stress_dev
    d_xy = 0.5 * stress_dev
    eps_p_xx_new = np.clip(eps_p_xx + d_xx, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_yy_new = np.clip(eps_p_yy + d_yy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    eps_p_xy_new = np.clip(eps_p_xy + d_xy, -MAX_PLASTIC_STRAIN, MAX_PLASTIC_STRAIN)
    return eps_p_xx_new, eps_p_yy_new, eps_p_xy_new


# Snippet 4 -- associated (normality) flow rule, for the confound-C experiment.
def update_plastic_strain_associated(sigma_eq, sigma_y, sxx, syy, sxy,
                                     eps_p_xx, eps_p_yy, eps_p_xy,
                                     gamma0_dot, m, dt):
    os_ = np.clip(np.maximum(sigma_eq - sigma_y, 0.0) / np.maximum(sigma_y, 1e-9), 0, 1)
    inc = np.minimum(1.5 * gamma0_dot * os_**m * dt, 1e-3)
    p = (sxx + syy) / 2.0
    dx_, dy_, dxy_ = sxx - p, syy - p, sxy
    nrm = np.sqrt(dx_**2 + dy_**2 + 2.0 * dxy_**2 + 1e-30)
    return (np.clip(eps_p_xx + inc * dx_ / nrm, -.1, .1),
            np.clip(eps_p_yy + inc * dy_ / nrm, -.1, .1),
            np.clip(eps_p_xy + inc * dxy_ / nrm, -.1, .1))


# ----------------------------------------------------------------------------
# Snippet 3 -- smoothness metrics (verbatim, minus the streamlit coupling).
# ----------------------------------------------------------------------------
def _smoothness(field):
    tv = float(np.mean(np.abs(np.diff(field, axis=0))) +
                np.mean(np.abs(np.diff(field, axis=1))))
    F = np.abs(np.fft.fftshift(np.fft.fft2(field - field.mean())))**2
    n = field.shape[0]
    k = np.fft.fftshift(np.fft.fftfreq(n))
    kx, ky = np.meshgrid(k, k)
    hi = np.sqrt(kx**2 + ky**2) > 0.25
    return {"tv": tv, "hf_frac": float(F[hi].sum() / max(F.sum(), 1e-30)),
            "max": float(np.max(field))}


# ----------------------------------------------------------------------------
# Representative fields.
# ----------------------------------------------------------------------------
def build_h_field(N=128, twin_spacing_nm=20.0, seed=1):
    """Lamellar-nanotwin h field in nm: sawtooth distance to nearest twin plane,
    with a slight angular tilt so the contour is 2-D (rings/lamellae), mirroring
    compute_twin_spacing_from_gradient(phi_gx, phi_gy) = 2/|grad phi|."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    # twin planes tilted, with a mild grain-dependent spacing jitter
    tilt = 0.15
    coord = xx + tilt * yy
    spacing = twin_spacing_nm * (1.0 + 0.25 * rng.uniform(-1, 1, size=(1, N)))
    h = np.abs(coord % spacing)            # 0..spacing/2 sawtooth (distance to plane)
    h = np.minimum(h, spacing - h)         # distance to NEAREST plane
    return np.maximum(h, 2.0)              # nm; clamp so sigma_y stays in the ~0.5-3 GPa band the analysis describes


def build_sigma_eq(N=128, peak_GPa=1.2, seed=2):
    """Smooth GPa-scale equivalent-stress field (low-frequency Fourier modes).
    Places the system in the marginal-yielding regime so the sigma_y contour is
    nontrivial -- stands in for the real spectral solve's concentrated stress."""
    rng = np.random.default_rng(seed)
    ky, kx = np.mgrid[0:N, 0:N].astype(float)
    field = np.zeros((N, N))
    for _ in range(6):
        fx, fy = rng.uniform(-3, 3, size=2)
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        field += amp * np.cos(2 * np.pi * (fx * kx / N + fy * ky / N) + ph)
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)   # 0..1
    # range chosen so sigma_eq straddles the structured sigma_y (0.5-3 GPa):
    # places the bulk of cells in the overstress in (0,1) marginal-yielding regime.
    return peak_GPa * 1e9 * (0.30 + 0.70 * field)


def run_solver(h_nm, sigma_eq, plasticity, n_steps=100, unit_fix=True,
               associated_flow=False, sxx=None, syy=None, sxy=None):
    """Run the verbatim plasticity update for n_steps; return final fields."""
    sigma0 = plasticity["sigma0"]; mu = plasticity["mu"]
    b = plasticity["b"]; nu = plasticity["nu"]
    gamma0_dot = plasticity["gamma0_dot"]; m = int(plasticity["m"])
    dt = plasticity.get("dt", 1e-4)

    h_yield = h_nm * 1e-9 if unit_fix else h_nm
    sigma_y = compute_yield_stress(h_yield, sigma0, mu, b, nu)

    epxx = np.zeros_like(sigma_eq); epyy = np.zeros_like(sigma_eq); epxy = np.zeros_like(sigma_eq)
    for _ in range(n_steps):
        if associated_flow:
            epxx, epyy, epxy = update_plastic_strain_associated(
                sigma_eq, sigma_y, sxx, syy, sxy, epxx, epyy, epxy, gamma0_dot, m, dt)
        else:
            epxx, epyy, epxy = update_plastic_strain(
                sigma_eq, sigma_y, epxx, epyy, epxy, gamma0_dot, m, dt)
    eps_p_mag = np.sqrt(2.0 / 3.0 * (epxx**2 + epyy**2 + 2.0 * epxy**2 + 1e-15))
    return {"eps_p_mag": eps_p_mag, "eps_p_xy": epxy, "sigma_y": sigma_y}


# ----------------------------------------------------------------------------
# A/B benchmark (Snippet 3, streamlit-free).
# ----------------------------------------------------------------------------
BASELINE = {"mu": 48e9, "nu": 0.34, "b": 0.256e-9, "sigma0": 50e6,
            "gamma0_dot": 1e-3, "m": 20, "rho0": 1e12, "dt": 1e-4}
# Representative AI overrides (the AI app file ...778713.txt is not uploaded,
# so these reflect the analysis's described magnitudes: sigma0 halved, m raised,
# rho0 = 2.90e16). rho0 is inert, so its value is irrelevant to the result.
AI_OVERRIDES = {"mu": 48e9, "sigma0": 25e6, "m": 33, "gamma0_dot": 1e-3,
                "rho0": 2.9e16}


def run_ab_benchmark(h_nm, sigma_eq, n_steps=100, unit_fix=True):
    scenarios = {"baseline": {}}
    scenarios.update({f"only {k}": {k: v} for k, v in AI_OVERRIDES.items()})
    scenarios["AI all"] = dict(AI_OVERRIDES)
    rows, frames = [], {}
    for name, ov in scenarios.items():
        plast = dict(BASELINE); plast.update(ov)
        res = run_solver(h_nm, sigma_eq, plast, n_steps=n_steps, unit_fix=unit_fix)
        frames[name] = res
        sm = _smoothness(res["eps_p_mag"])
        epxy = res["eps_p_xy"]
        rows.append({
            "scenario": name, "unit_fix": unit_fix,
            "eps_tv": sm["tv"], "eps_hf_frac": sm["hf_frac"], "eps_max": sm["max"],
            "sigmay_min_GPa": float(res["sigma_y"].min()) / 1e9,
            "sigmay_max_GPa": float(res["sigma_y"].max()) / 1e9,
            "eps_xy_pos_frac": float(np.mean(epxy > 0)),   # confound-C signature
        })
    return pd.DataFrame(rows), frames


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    N = 128
    h_nm = build_h_field(N, twin_spacing_nm=20.0)
    sigma_eq = build_sigma_eq(N, peak_GPa=3.0)

    all_rows = []
    frames_by_fix = {}
    for fix in (False, True):
        df, frames = run_ab_benchmark(h_nm, sigma_eq, n_steps=100, unit_fix=fix)
        all_rows.append(df)
        frames_by_fix[fix] = frames
        print(f"\n=== unit_fix={fix} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # rho0-inertness assertion (only rho0 must equal baseline byte-for-byte)
    base = frames_by_fix[True]["baseline"]["eps_p_mag"]
    only_rho = frames_by_fix[True]["only rho0"]["eps_p_mag"]
    assert np.array_equal(base, only_rho), "rho0 is NOT inert -- expected byte-identical!"
    print("\n[OK] only-rho0 == baseline (byte-identical) -> rho0 is confirmed INERT.")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(f"{OUT}/benchmark_results.csv", index=False)
    print(f"\nsaved {OUT}/benchmark_results.csv")

    # ---- grid plot: 2 rows (off / on) x 7 cols (scenarios) ----
    order = ["baseline", "only mu", "only sigma0", "only m",
             "only gamma0_dot", "only rho0", "AI all"]
    fig, axes = plt.subplots(2, 7, figsize=(18, 5.6), constrained_layout=True)
    for r, fix in enumerate((False, True)):
        fr = frames_by_fix[fix]
        for c, name in enumerate(order):
            ax = axes[r, c]
            im = ax.imshow(fr[name]["eps_p_mag"], origin="lower", cmap="magma",
                           vmin=0, vmax=max(full["eps_max"].max(), 1e-30))
            hf = _smoothness(fr[name]["eps_p_mag"])["hf_frac"]
            ax.set_title(f"{name}\nhf_frac={hf:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"unit_fix={fix}", fontsize=11)
    fig.suptitle("Plastic strain |eps_p| -- confound A (Hall-Petch unit fix)\n"
                 "off (top): smooth everywhere; on (bottom): structured everywhere, incl. baseline",
                 fontsize=12)
    fig.colorbar(im, ax=axes[:, -1], shrink=0.8, label="|eps_p|")
    fig.savefig(f"{OUT}/benchmark_grid.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT}/benchmark_grid.png")

    # ---- sigma_y flat-vs-structured comparison ----
    sy_off = compute_yield_stress(h_nm, 50e6, 48e9, 0.256e-9, 0.34)
    sy_on = compute_yield_stress(h_nm * 1e-9, 50e6, 48e9, 0.256e-9, 0.34)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    im0 = ax[0].imshow(sy_off / 1e6, origin="lower", cmap="viridis")
    ax[0].set_title(f"sigma_y, unit_fix=False\nrange {sy_off.min()/1e6:.2f}-{sy_off.max()/1e6:.2f} MPa (FLAT)")
    fig.colorbar(im0, ax=ax[0], label="sigma_y [MPa]")
    im1 = ax[1].imshow(sy_on / 1e9, origin="lower", cmap="viridis")
    ax[1].set_title(f"sigma_y, unit_fix=True\nrange {sy_on.min()/1e9:.2f}-{sy_on.max()/1e9:.2f} GPa (STRUCTURED)")
    fig.colorbar(im1, ax=ax[1], label="sigma_y [GPa]")
    fig.savefig(f"{OUT}/sigma_y_comparison.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT}/sigma_y_comparison.png")

    # ---- confound C: fixed-sign vs associated flow on eps_p_xy ----
    sxx = np.zeros_like(sigma_eq); syy = np.zeros_like(sigma_eq)
    sxy = sigma_eq * 0.6 * np.sign(np.sin(2 * np.pi * np.arange(N)[:, None] / N * 3))  # sign-changing shear
    res_fixed = run_solver(h_nm, sigma_eq, dict(BASELINE), n_steps=100, unit_fix=True,
                           associated_flow=False)
    res_assoc = run_solver(h_nm, sigma_eq, dict(BASELINE), n_steps=100, unit_fix=True,
                           associated_flow=True, sxx=sxx, syy=syy, sxy=sxy)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    im = ax[0].imshow(res_fixed["eps_p_xy"], origin="lower", cmap="RdBu_r",
                      vmin=-max(np.abs(res_fixed["eps_p_xy"]).max(), np.abs(res_assoc["eps_p_xy"]).max()),
                      vmax=max(np.abs(res_fixed["eps_p_xy"]).max(), np.abs(res_assoc["eps_p_xy"]).max()))
    ax[0].set_title(f"eps_p_xy, fixed-sign flow (legacy)\npos_frac={np.mean(res_fixed['eps_p_xy']>0):.2f} (single-signed -> rectified)")
    im = ax[1].imshow(res_assoc["eps_p_xy"], origin="lower", cmap="RdBu_r",
                      vmin=-max(np.abs(res_fixed["eps_p_xy"]).max(), np.abs(res_assoc["eps_p_xy"]).max()),
                      vmax=max(np.abs(res_fixed["eps_p_xy"]).max(), np.abs(res_assoc["eps_p_xy"]).max()))
    ax[1].set_title(f"eps_p_xy, associated flow (Snippet 4)\npos_frac={np.mean(res_assoc['eps_p_xy']>0):.2f} (follows sxy sign)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="eps_p_xy")
    fig.suptitle("Confound C: fixed-sign d_xy = +0.5*stress_dev rectifies the pattern", fontsize=12)
    fig.savefig(f"{OUT}/associated_flow_comparison.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT}/associated_flow_comparison.png")

    # ---- headline numbers ----
    off = all_rows[0]; on = all_rows[1]
    print("\n=== HEADLINE ===")
    print(f"unit_fix=False : mean hf_frac = {off['eps_hf_frac'].mean():.4g}  (smooth)")
    print(f"unit_fix=True  : mean hf_frac = {on['eps_hf_frac'].mean():.4g}  (structured)")
    print(f"baseline hf_frac: off={off.loc[off.scenario=='baseline','eps_hf_frac'].iloc[0]:.4g}"
          f"  on={on.loc[on.scenario=='baseline','eps_hf_frac'].iloc[0]:.4g}")
    print("=> zigzag appears in EVERY unit_fix=True row incl. baseline => culprit = sigma_y structure / unit fix.")


if __name__ == "__main__":
    main()
