# Demonstration to Verify the Input Parameters are Passed On to the FFT Solver

Demonstration of Plasticity Parameters Influence

[![continuummodelnt2d](https://img.shields.io/badge/demo1-streamlit-red)](https://benchmark-mwe1.streamlit.app/) ()


Benchmark Phase-field evolution (Allen-Cahn equations for ϕ, η1, η2) and  Mechanical equilibrium (calculating stress and strain from eigenstrains) are solved using the FFT spectral method.

[![continuummodelnt2d](https://img.shields.io/badge/pffftbm1-streamlit-turquoise)](https://nanotwin-evolution-mechanics-benchmark1.streamlit.app/) (r2 with colormap functionality for phase field simulation results visualizations, default delta t =1e-3 s, in update_plastic_strain func, max overstress = 1.0, max plastic strain = 0.1 and stress_dev = np.minimum(stress_dev,0.001), the plastic strain is now within reasonable limit )
