# Hybrid FDM + FFT approach

FDM: Phase-field evolution (Allen-Cahn equations for ϕ, η1, η2) is solved using explicit FDM (via Numba parallel kernels) for the gradients and Laplacians.

FFT: Mechanical equilibrium (calculating stress and strain from eigenstrains) is solved using the FFT spectral method.

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincudata17-streamlit-red)](https://nanotwinstructure-datagenerator17.streamlit.app/) ( variable distance between the ITB and GB (right) and ITB and left edge, planar and curved GB interface, interactive visualization)
