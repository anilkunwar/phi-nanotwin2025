import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Neper Tessellation Visualizer",
    page_icon="🧊",
    layout="wide",
)

st.title("🧊 Neper Tessellation Visualizer")
st.markdown(
    "Generate and visualize polycrystalline tessellations with "
    "[Neper](https://neper.info/) directly from the browser."
)

# ---------------------------------------------------------------------------
# Sidebar – user inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parameters")

    n_cells = st.number_input(
        "Number of cells (n)",
        min_value=1,
        max_value=10_000,
        value=20,
        step=1,
        help="Passed to `neper -T -n <value>`.",
    )

    morpho = st.selectbox(
        "Morphology (-morpho)",
        options=["gg", "voronoi", "lamellar", "cubic"],
        index=0,
        help="Tessellation morphology. `gg` = grain growth.",
    )

    id_seed = st.number_input(
        "Id seed (-id)",
        min_value=1,
        max_value=1_000,
        value=1,
        step=1,
        help="Random seed used by Neper.",
    )

    img_width = st.number_input("Image width (px)", min_value=100, max_value=4000, value=800, step=50)
    img_height = st.number_input("Image height (px)", min_value=100, max_value=4000, value=400, step=50)

    st.divider()
    st.caption("💡 Make sure `neper` is available on PATH (e.g. `conda activate neperenv`).")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_neper() -> str | None:
    """Return the path to the `neper` executable, or None if not found."""
    return shutil.which("neper")


def configure_neperrc(width: int, height: int, rc_path: Path) -> str:
    """Ensure ~/.neperrc contains the requested `-imagesize` line."""
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    config_line = f"neper -V -imagesize {width}:{height}"

    existing = rc_path.read_text() if rc_path.exists() else ""
    if config_line in existing:
        return f"✅ Configuration already present in {rc_path}"

    # Remove any previous `neper -V -imagesize ...` line to avoid duplicates
    cleaned = re.sub(r"^neper\s+-V\s+-imagesize\s+\d+:\d+\s*$\n?", "", existing, flags=re.MULTILINE)
    with open(rc_path, "a" if cleaned else "w") as f:
        if cleaned:
            f.write(cleaned.rstrip() + "\n")
        f.write(config_line + "\n")
    return f"📝 Wrote `{config_line}` to {rc_path}"


def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a subprocess and capture stdout/stderr."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
if st.button("🚀 Generate tessellation", type="primary", use_container_width=True):

    # 1. Check that Neper is installed
    neper_bin = find_neper()
    if neper_bin is None:
        st.error(
            "❌ `neper` executable not found on PATH. "
            "Please activate your conda environment (e.g. `conda activate neperenv`) "
            "and restart Streamlit from that environment."
        )
        st.stop()

    st.info(f"Using Neper at: `{neper_bin}`")

    # 2. Configure ~/.neperrc
    rc_path = Path.home() / ".neperrc"
    rc_msg = configure_neperrc(img_width, img_height, rc_path)
    st.success(rc_msg)

    # 3. Work in a temporary directory so files don't collide
    workdir = Path(tempfile.mkdtemp(prefix="neper_st_"))
    tess_name = f"n{n_cells}-id{id_seed}"
    tess_file = workdir / f"{tess_name}.tess"
    img_prefix = "img1"
    img_file = workdir / f"{img_prefix}.png"

    log_box = st.expander("📜 Console log", expanded=True)

    # --- Step 2: generate tessellation -----------------------------------
    log_box.markdown(f"**Step 1/2** – Generating {n_cells}-cell `{morpho}` tessellation…")
    rc, out, err = run(
        ["neper", "-T", "-n", str(n_cells), "-morpho", morpho, "-id", str(id_seed)],
        cwd=workdir,
    )
    log_box.code(out + err, language="bash")

    if rc != 0 or not tess_file.exists():
        st.error(f"❌ Tessellation generation failed (exit code {rc}).")
        st.stop()

    # --- Step 3: visualize & print ---------------------------------------
    log_box.markdown("**Step 2/2** – Rendering image…")
    rc, out, err = run(
        ["neper", "-V", str(tess_file), "-print", img_prefix],
        cwd=workdir,
    )
    log_box.code(out + err, language="bash")

    if rc != 0 or not img_file.exists():
        st.error(f"❌ Visualization failed (exit code {rc}).")
        st.stop()

    # 4. Display the result
    st.success(f"✅ Generated `{img_file.name}` ({img_file.stat().st_size:,} bytes)")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(str(img_file), caption=f"{tess_name} · {img_width}×{img_height}px", use_container_width=True)
    with col2:
        st.download_button(
            "⬇️ Download PNG",
            data=img_file.read_bytes(),
            file_name=img_file.name,
            mime="image/png",
        )
        st.markdown(
            f"""
            **Files produced**
            - Tessellation: `{tess_file.name}`
            - Image: `{img_file.name}`
            - Working dir: `{workdir}`
            """
        )
else:
    st.info("Adjust the parameters on the left and click **Generate tessellation**.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Equivalent to: `neper -T -n {n} -morpho {m} -id {id}` → "
    "`neper -V <tess> -print img1` (with `~/.neperrc` set to `-imagesize {w}:{h}`)."
)
