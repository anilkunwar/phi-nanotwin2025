import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="LLM & AI Tab Architecture Schematic",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 LLM Architecture with AI Agent Tab Interface")
st.markdown(
    """
This schematic illustrates the system architecture of an **AI Collaboration Assistant** integrated into a web dashboard (such as Streamlit or a workspace tab). 
It maps the flow of user queries, LLM reasoning, tool calling (e.g., spectral Phase-Field FFT solvers), and real-time visualization.
"""
)

# -------------------------------------------------------------------------
# Sidebar Options
# -------------------------------------------------------------------------
st.sidebar.header("Diagram Styling")
theme_color = st.sidebar.selectbox(
    "Color Theme", ["Default Blue/Teal", "Dark Mode Accent", "Minimalist Gray"]
)

if theme_color == "Default Blue/Teal":
    c_user = "#E3F2FD"
    c_tab = "#BBDEFB"
    c_llm = "#D1C4E9"
    c_tools = "#C8E6C9"
    c_out = "#FFE0B2"
elif theme_color == "Dark Mode Accent":
    c_user = "#263238"
    c_tab = "#37474F"
    c_llm = "#4A148C"
    c_tools = "#1B5E20"
    c_out = "#E65100"
else:
    c_user = "#F5F5F5"
    c_tab = "#EEEEEE"
    c_llm = "#E0E0E0"
    c_tools = "#D6D6D6"
    c_out = "#CCCCCC"

# -------------------------------------------------------------------------
# Graphviz DOT Architecture Definition
# -------------------------------------------------------------------------
dot_code = f"""
digraph LLM_AI_Tab_Architecture {{
    graph [rankdir=TB, splines=ortho, nodesep=0.6, ranksep=0.8, fontname="Helvetica"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", margin="0.2,0.15", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#555555", penwidth=1.2];

    # 1. User & Front-end UI Group
    subgraph cluster_frontend {{
        label = "User Interface Layer (Streamlit / Workspace)";
        style = "dashed";
        color = "#1976D2";
        fontcolor = "#1976D2";
        fontsize = 12;

        User [label="👤 User / Researcher\\n(Prompt, Parameters, Files)", fillcolor="{c_user}", color="#1976D2"];
        AITab [label="📱 AI Agent Tab\\n(Chat Box, History, Dynamic Widgets)", fillcolor="{c_tab}", color="#1565C0"];
        Viewer [label="📊 Interactive Canvas\\n(Matplotlib, Plotly, Renderers)", fillcolor="{c_out}", color="#E65100"];
    }}

    # 2. Core Reasoning LLM Orchestrator
    subgraph cluster_backend {{
        label = "LLM Core & Reasoning Engine";
        style = "dashed";
        color = "#7B1FA2";
        fontcolor = "#7B1FA2";
        fontsize = 12;

        Context [label="🧠 System Instructions &\\nConversation Memory", fillcolor="{c_llm}", color="#6A1B9A"];
        LLM [label="⚡ Large Language Model (LLM)\\n(Prompt Understanding, Code Gen, Tool Routing)", fillcolor="{c_llm}", color="#4A148C", shape=rect];
    }}

    # 3. External Tools & Solvers
    subgraph cluster_tools {{
        label = "Tool Execution Environment (Python Runtime)";
        style = "dashed";
        color = "#388E3C";
        fontcolor = "#388E3C";
        fontsize = 12;

        FFT_Solver [label="🌊 Spectral FFT Solver\\n(scipy.fft, k-space derivation)", fillcolor="{c_tools}", color="#2E7D32"];
        PF_Model [label="🔬 Phase-Field Simulation\\n(Cahn-Hilliard / Allen-Cahn)", fillcolor="{c_tools}", color="#2E7D32"];
        DataStore [label="💾 Data & File Store\\n(Pickle, HDF5, NumPy arrays)", fillcolor="{c_tools}", color="#2E7D32"];
    }}

    # Data Flow Connections
    User -> AITab [label=" Input Query / Parameter Changes "];
    AITab -> Context [label=" Send Context & Prompt "];
    Context -> LLM;
    
    # LLM Tool Call Loop
    LLM -> FFT_Solver [label=" Function Call / Execute Code "];
    LLM -> PF_Model [label=" Trigger Simulation Run "];
    
    FFT_Solver -> DataStore [label=" Array Computation "];
    PF_Model -> DataStore [label=" Field Updates "];
    
    # Returning Execution Results
    DataStore -> AITab [label=" Processed Data & Figures "];
    AITab -> Viewer [label=" Render Graphics & Output "];
    Viewer -> User [label=" Visual Feedback / Response "];
}}
"""

# Render the graph in Streamlit
st.graphviz_chart(dot_code, use_container_width=True)

# -------------------------------------------------------------------------
# Architecture Explanation
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("Component Breakdown")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. UI Layer (AI Tab)")
    st.markdown(
        """
    * **Chat Interface:** Captures user requests, parameter tweaks, and custom mathematical formulas.
    * **State Management:** Preserves user history and active session data across reruns.
    * **Interactive Canvas:** Displays real-time matplotlib plots, Plotly graphics, and dataframes.
    """
    )

with col2:
    st.markdown("### 2. LLM Core Engine")
    st.markdown(
        """
    * **System Context:** Injects specialized domain knowledge (e.g., Phase-Field modeling, spectral methods).
    * **Function Calling:** Decides whether to generate text responses or trigger numerical execution scripts.
    * **Code Generator:** Dynamically creates or adjusts Python routines for data post-processing.
    """
    )

with col3:
    st.markdown("### 3. Execution Environment")
    st.markdown(
        """
    * **Spectral Solvers:** Computes FFT, wavenumber vectors ($k_x, k_y, k^2$), and laplacians.
    * **Phase-Field Models:** Evolves order parameters over time steps.
    * **Data Storage:** Caches large array operations safely in memory or disk formats (HDF5, NumPy).
    """
    )
