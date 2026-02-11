# Phase field method for modeling evolution of nanotwins in FCC matrix

[![Visualization via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nanotwinevolution.streamlit.app/)

[![continuummodelnt2d](https://img.shields.io/badge/nanotwinag1-streamlit-red)](https://nanotwinevolution2.streamlit.app/)  (faster model)



# Multiphysics simulations

Model Development

[![continuummodelnt2d](https://img.shields.io/badge/nanotwinag1-streamlit-red)](https://nanotwin-structureevolution1.streamlit.app/) (Debug phase)

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincu2-streamlit-red)](https://nanotwin-structureevolution2.streamlit.app/) (Solutions of phase field elasto plastic equations without post-processing)

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincu3-streamlit-red)](https://nanotwin-structureevolution3.streamlit.app/) (Realistic plasticity - Volume-preserving update with shear strain evolution: Solutions of phase field elasto plastic equations with post-processing, twin containing grain more stable than the one without twin, detwinning happens at both ends of the twins, Key difference in plastic strain updates: Code 3 uses -0.5 and +0.5 factors for yy and xy strains, enabling shear, unlike Code 4's full volume preservation with no shear, Code 3 has Shear Plastic Strain and So Compressibility is Modeled with J2 Flow Theory for Plasiticity in Metals.)

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincu4-streamlit-red)](https://nanotwin-structureevolution4.streamlit.app/) (Unphysical constraint- Volume-preserving update WITHOUT shear strain evolution: Solutions of phase field elasto plastic equations with post-processing, both grains are in equilbrium, detwinning happens only at the end that does not touch the grain boundary,  Key difference in plastic strain updates: Code 3 uses -0.5 and +0.5 factors for yy and xy strains, enabling shear, unlike Code 4's full volume preservation with no shear, Incompressibility Preserved and No Shear Plastic Strain .)


Engineering Design with Advanced Models (nanotwincu3 model as the physically realistic base model)

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincu5-streamlit-red)](https://nanotwin-structurecomparison5.streamlit.app/) (theoretically inconsistent and no evolution)

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincu6-streamlit-red)](https://nanotwin-structurecomparison6.streamlit.app/) (theoretically consistent and detwinning occurs)

Data Generation

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincudata7-streamlit-red)](https://nanotwinstructure-datagenerator7.streamlit.app/) (theoretically consistent and detwinning occurs)
