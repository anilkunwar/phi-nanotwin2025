# Pure FFT-based Semi-Implicit Spectral Phase-Field model

  Phase-field evolution (Allen-Cahn equations for ϕ, η1, η2) and  Mechanical equilibrium (calculating stress and strain from eigenstrains) are solved using the FFT spectral method.

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincudata1-streamlit-red)](https://nanotwin-evolution-mechanics1.streamlit.app/) (default delta t =1e-3 s, in update_plastic_strain func, max overstress = 3.0, max plastic strain = 1.0 and stress_dev = np.minimum(stress_dev,0.05) , throws the warning "Large plastic strain detected: 0.500" during simulation run)

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincudata2-streamlit-turquoise)](https://nanotwin-evolution-mechanics2.streamlit.app/) (default delta t =1e-3 s, in update_plastic_strain func, max overstress = 1.0, max plastic strain = 0.1 and stress_dev = np.minimum(stress_dev,0.001), the plastic strain is now within reasonable limit )

[![continuummodelnt2d](https://img.shields.io/badge/nanotwincudata3-streamlit-turquoise)](https://nanotwin-evolution-mechanics3.streamlit.app/) (r2 with colormap functionality for phase field simulation results visualizations, default delta t =1e-3 s, in update_plastic_strain func, max overstress = 1.0, max plastic strain = 0.1 and stress_dev = np.minimum(stress_dev,0.001), the plastic strain is now within reasonable limit )
