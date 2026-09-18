"""
INTEGRATION PATCH -- Enhanced Nanotwinned Cu Phase-Field Simulator (FFT)
=========================================================================
Three surgical edits wire plasticity_recommender.py into code 1.
Nothing else in the simulator changes.

EDIT 1 -- make rho0 consequential (it is currently unused in the solver).
Replace the existing compute_yield_stress with this version
(Taylor hardening term added; signature gains rho0 with default None so the
old call sites keep working until EDIT 3 updates them):

    def compute_yield_stress(h, sigma0, mu, b, nu, rho0=None, alpha_taylor=0.3):
        """Hall-Petch-like yield stress + Taylor dislocation hardening, vectorized."""
        safe = h > 2 * b
        sigma_y = np.empty_like(h)
        log_term = np.log(np.maximum(h, 2.001 * b) / b)
        sigma_y[safe] = sigma0 + (mu * b / (2 * np.pi * h[safe] * (1 - nu))) * log_term[safe]
        sigma_y[~safe] = sigma0 + mu / (2 * np.pi * (1 - nu))
        if rho0:  # Taylor term: 0.5 * mu * b * alpha * sqrt(rho0)
            sigma_y = sigma_y + 0.5 * mu * b * alpha_taylor * np.sqrt(rho0)
        return sigma_y

EDIT 2 -- let the solver accept an LLM-recommended override, and pass rho0.
In NanotwinnedCuSolver.__init__, immediately after:

        self.mat_props = MaterialProperties.get_material(material_name)
        self.params['material'] = material_name

insert:

        # ---- LLM-recommended plasticity override (Stage F) ----
        override = params.get('plasticity_override')
        if override:
            self.mat_props['plasticity'].update(
                {k: v for k, v in override.items() if v is not None})

In NanotwinnedCuSolver.step, replace the compute_yield_stress call with:

            sigma_y = compute_yield_stress(
                h, plastic_params['sigma0'], plastic_params['mu'],
                plastic_params['b'], plastic_params['nu'],
                rho0=plastic_params.get('rho0'),
                alpha_taylor=plastic_params.get('alpha_taylor', 0.3))

EDIT 3 -- sidebar "Let LLM recommend" button.
In main(), in operation_mode == "Run New Simulation", immediately AFTER:

            st.subheader("🧪 Material")
            material_choice = st.selectbox("Select material", ["Cu", "Al", "Ni"], key="material")

insert this block:

            # ---------------- LLM plasticity recommendation (Stage F) --------------
            from plasticity_recommender import (
                load_evidence_corpus, get_backend, recommend)

            st.subheader("🦙 LLM Plasticity Recommendation")
            json_db_dir = st.text_input("Metadatabase directory",
                                        value="./json_metadatabase")
            if st.button("🦙 Let LLM recommend", type="primary",
                         use_container_width=True):
                corpus = load_evidence_corpus(json_db_dir)
                if not corpus:
                    st.warning("No metadatabase records found; keeping defaults.")
                else:
                    mat_props = MaterialProperties.get_material(material_choice)
                    backend = get_backend()
                    if not backend["llm_available"]:
                        st.info("Ollama not reachable -- running deterministic "
                                "(rule + regex + weighted-median) pipeline.")
                    with st.spinner("Categorizing -> NER -> graph reasoning -> MoE fusion..."):
                        rec = recommend(corpus, mat_props,
                                        material=material_choice.lower(),
                                        backend=backend, json_dir=json_db_dir)
                    # Provenance FIRST, values second:
                    with st.expander(
                            f"Evidence: {rec['n_evidence']} records "
                            f"({rec['n_excluded']} excluded, e.g. non-Cu materials)"):
                        import pandas as pd
                        if rec["evidence_table"]:
                            st.dataframe(pd.DataFrame(rec["evidence_table"]))
                        st.caption(f"MoE gate: {rec['gate']}")
                        for note in rec["notes"]:
                            st.caption(note)
                        if rec["top_attended_papers"]:
                            st.caption("Attention top-3: "
                                       + "; ".join(rec["top_attended_papers"]))
                    override = {k: rec.get(k) for k in
                                ("mu", "sigma0", "rho0", "gamma0_dot", "m")}
                    st.session_state["plasticity_override"] = override
                    st.success(
                        f"Recommended mu={rec['mu']/1e9:.1f} GPa, "
                        f"sigma0={rec['sigma0']/1e6:.0f} MPa, "
                        f"m={rec['m']} (from SRS={rec['srs']:.3f}), "
                        f"rho0={rec['rho0']:.2e}, "
                        f"gamma0_dot={rec['gamma0_dot']:.1e}"
                        if all(override.values()) else
                        "Partial recommendation -- missing parameters keep defaults.")

            if "plasticity_override" in st.session_state:
                ov = st.session_state["plasticity_override"]
                st.caption("Active LLM override: "
                           + ", ".join(f"{k}={v:g}" for k, v in ov.items()
                                       if v is not None))

Then, inside the "🚀 Initialize Simulation" handler, add to the params dict:

                    'plasticity_override': st.session_state.get('plasticity_override'),

NOTES
-----
* The recommender is driven by the ALREADY-SELECTED material. For Cu it should
  reproduce mu ~= 47-48 GPa (validated against the VRH guard using the app's
  own elastic constants); for Al/Ni a Cu-centric corpus yields wide intervals
  or an honest fallback to defaults -- that is surfaced in the provenance
  expander rather than silently recommending Cu values for Ni.
* rho0 is now consequential via the Taylor term (EDIT 1). Until you are
  comfortable with alpha_taylor (0.2-0.5 literature range), keep it at 0.3.
* Every stage degrades gracefully: no Ollama -> gazetteer + regex + weighted
  median; no sentence-transformers -> rule-only categorization; no networkx/
  torch -> list-based relational collection.
* If you want the optional trainable LatentMoE v2, train with leave-one-paper-
  out: hold out one paper per fold, predict its value from the remaining
  evidence latents, minimize NLL of (mu, var) -- see LatentMoE in the module.
