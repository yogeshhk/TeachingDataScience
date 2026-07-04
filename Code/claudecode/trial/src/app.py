"""Streamlit app for stlinspector: upload, 3D preview, validation report."""

from __future__ import annotations

import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from core import inspect_mesh, load_mesh

st.set_page_config(page_title="STL Inspector", layout="wide")
st.title("STL Inspector")

uploaded = st.file_uploader("Upload an STL file", type=["stl"])

if uploaded is None:
    st.info("Upload an STL file to inspect it.")
    st.stop()

tmp_path: Path | None = None
try:
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    mesh = load_mesh(tmp_path)
    report = inspect_mesh(mesh)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not inspect file: {exc}")
    st.stop()
finally:
    if tmp_path is not None:
        tmp_path.unlink(missing_ok=True)

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color="lightblue",
                opacity=0.9,
            )
        ]
    )
    fig.update_layout(scene=dict(aspectmode="data"), height=600)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Triangle count", report.triangle_count)
    st.metric("Volume", f"{report.volume:.4f}")
    st.metric("Surface area", f"{report.surface_area:.4f}")
    bb_min, bb_max = report.bounding_box
    st.write("Bounding box (min):", [round(v, 4) for v in bb_min])
    st.write("Bounding box (max):", [round(v, 4) for v in bb_max])

    st.subheader("Validation")
    if report.issues:
        for issue in report.issues:
            st.error(issue)
    else:
        st.success("All checks passed")
