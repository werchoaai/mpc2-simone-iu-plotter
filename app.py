"""
Simone I-U Plotter · Streamlit Cloud version.

Same functionality as standalone HTML, but cloud-hosted with password gate.
"""
from __future__ import annotations
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────
st.set_page_config(page_title="Simone I-U Plotter · MPC²", page_icon="📈", layout="wide")

# Password gate
if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    st.title("MPC² · Simone I-U Plotter")
    st.caption("Geschützter Zugang · bitte Passwort eingeben")
    pw = st.text_input("Passwort", type="password")
    if pw and pw == st.secrets.get("password", ""):
        st.session_state.auth = True
        st.rerun()
    elif pw:
        st.error("Falsches Passwort.")
    st.stop()

# ─────────────────────────────────────────
# Language
LANG = st.sidebar.radio("Sprache · Language", ["DE", "EN"], horizontal=True)

T = {
    "title": ("I-U Curve Plotter · für Simone", "I-U Curve Plotter · for Simone"),
    "subtitle": ("Stromdichte vs. Potential · läuft in Streamlit Cloud", "Current density vs. potential · runs in Streamlit Cloud"),
    "upload": ("ASC-Datei(en) hochladen", "Upload ASC file(s)"),
    "settings": ("Diagramm-Einstellungen", "Plot settings"),
    "title_input": ("Titel (frei wählbar)", "Title (custom)"),
    "xmin": ("X-Achse min [mV]", "X-axis min [mV]"),
    "xmax": ("X-Achse max [mV]", "X-axis max [mV]"),
    "ymin": ("Y-Achse min [mA/cm²]", "Y-axis min [mA/cm²]"),
    "ymax": ("Y-Achse max [mA/cm²]", "Y-axis max [mA/cm²]"),
    "yscale": ("Y-Achse Skala", "Y-axis scale"),
    "smoothing": ("Glättung (Punkte)", "Smoothing (points)"),
    "decimate": ("Datenpunkte (max)", "Data points (max)"),
    "show_ocp": ("OCP markieren", "Mark OCP"),
    "show_rpp": ("RPP markieren", "Mark RPP"),
    "curves": ("Kurven · Farben · Sichtbarkeit", "Curves · colors · visibility"),
    "no_curves": ("Lade ASC-Files hoch um zu starten.", "Upload ASC files to start."),
    "x_label": ("Potential E [mV vs. Ag/AgCl]", "Potential E [mV vs. Ag/AgCl]"),
    "y_label": ("Stromdichte i [mA/cm²]", "Current density i [mA/cm²]"),
    "default_title": ("I-U-Kurve", "I-U Curve"),
    "info_loaded": ("Geladen", "Loaded"),
    "info_total": ("Datenpunkte gesamt", "Total points"),
    "info_displayed": ("Angezeigt nach Reduktion", "Displayed after decimation"),
    "export_png": ("PNG speichern", "Save PNG"),
    "export_csv": ("CSV speichern", "Save CSV"),
    "values_panel": ("Werte zum Kopieren", "Values to copy"),
    "copy_tsv": ("Alle Werte als TSV (Excel) herunterladen", "Download all values as TSV (Excel)"),
}
def t(key): return T[key][0 if LANG == "DE" else 1]

# ─────────────────────────────────────────
DEFAULT_COLORS = ["#16a34a","#dc2626","#2563eb","#b45309","#7c3aed","#0891b2","#db2777","#65a30d","#ea580c","#475569"]

# ─────────────────────────────────────────
@dataclass
class Curve:
    name: str
    V: np.ndarray   # mV
    i: np.ndarray   # mA/cm²
    color: str
    visible: bool = True
    ocp: float | None = None
    rpp: float | None = None


def parse_asc(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse Simone-style ASC: 4 cols, returns potential[mV] and current density[mA/cm²]."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", ";")):
            continue
        parts = s.replace(",", ".").split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(p) for p in parts[:4]])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("ASC must have at least 4 columns")
    V_mV = arr[:, 1] * 1000.0
    iden = arr[:, 3]
    return V_mV, iden


def detect_ocp_rpp(V: np.ndarray, i: np.ndarray) -> tuple[float | None, float | None]:
    """OCP = mean V during initial rest plateau. RPP = last reverse-scan crossing of i ~ 0."""
    N = len(V)
    if N < 100:
        return None, None
    # OCP: leading low-current rows
    ocp_end = 0
    limit = min(N, 5000)
    for k in range(limit):
        if abs(i[k]) > 1e-6:
            ocp_end = k
            break
    rest_window = V[max(0, ocp_end - 200):max(1, ocp_end)]
    ocp = float(np.mean(rest_window)) if rest_window.size else None
    # Vertex
    vertex = int(np.argmax(V))
    # RPP: reverse scan where i drops below threshold
    threshold = 0.01
    rpp = None
    for k in range(N - 1, vertex, -1):
        if abs(i[k]) > threshold and abs(i[k - 1]) <= threshold:
            rpp = float(V[k])
            break
    if rpp is None:
        for k in range(vertex + 1, N - 1):
            if i[k] > threshold and i[k + 1] <= threshold:
                rpp = float(V[k])
    return ocp, rpp


def moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    if win < 2:
        return arr
    pad = win // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def decimate(V: np.ndarray, i: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    if len(V) <= target:
        return V, i
    idx = np.linspace(0, len(V) - 1, target, dtype=int)
    return V[idx], i[idx]


# ─────────────────────────────────────────
# UI
st.title(t("title"))
st.caption(t("subtitle"))

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📂 " + t("upload"))
    files = st.file_uploader("ASC", type=["asc", "ASC", "txt"], accept_multiple_files=True, label_visibility="collapsed")

    # Curve management state
    if "curves" not in st.session_state:
        st.session_state.curves = []
    if "loaded_names" not in st.session_state:
        st.session_state.loaded_names = set()

    # ingest new files
    if files:
        for f in files:
            if f.name in st.session_state.loaded_names:
                continue
            try:
                text = f.read().decode("utf-8", errors="replace")
                V, ii = parse_asc(text)
                ocp, rpp = detect_ocp_rpp(V, ii)
                st.session_state.curves.append(Curve(
                    name=f.name.rsplit(".", 1)[0],
                    V=V, i=ii,
                    color=DEFAULT_COLORS[len(st.session_state.curves) % len(DEFAULT_COLORS)],
                    ocp=ocp, rpp=rpp,
                ))
                st.session_state.loaded_names.add(f.name)
            except Exception as e:
                st.error(f"{f.name}: {e}")

    # ─── Werte-Copy-Panel
    if st.session_state.curves:
        st.subheader("📋 " + t("values_panel"))
        with st.container(border=True):
            for c in st.session_state.curves:
                vc1, vc2, vc3 = st.columns([2, 1, 1])
                vc1.markdown(f"<span style='color:{c.color};font-weight:500'>{c.name}</span>", unsafe_allow_html=True)
                vc2.code(f"{c.ocp:.1f}" if c.ocp is not None else "—", language=None)
                vc3.code(f"{c.rpp:.1f}" if c.rpp is not None else "—", language=None)
            # TSV download — build cleanly handling None values
            tsv_lines = ["Kurve\tOCP_mV\tRPP_mV"]
            for c in st.session_state.curves:
                ocp_s = f"{c.ocp:.1f}" if c.ocp is not None else ""
                rpp_s = f"{c.rpp:.1f}" if c.rpp is not None else ""
                tsv_lines.append(f"{c.name}\t{ocp_s}\t{rpp_s}")
            tsv_text = "\n".join(tsv_lines)
            st.download_button(
                "📋 " + t("copy_tsv"),
                data=tsv_text,
                file_name="IU_OCP_RPP_values.tsv",
                mime="text/tab-separated-values",
                use_container_width=True,
            )

    st.subheader("🎨 " + t("curves"))
    if not st.session_state.curves:
        st.info(t("no_curves"))
    else:
        for idx, c in enumerate(st.session_state.curves):
            with st.expander(f"{'✓' if c.visible else '✗'}  {c.name}", expanded=False):
                c.visible = st.checkbox("Sichtbar / Visible", value=c.visible, key=f"vis{idx}")
                c.color = st.color_picker("Farbe / Color", value=c.color, key=f"col{idx}")
                c.name = st.text_input("Name", value=c.name, key=f"nm{idx}")
                stats = f"n={len(c.V):,} · OCP={'—' if c.ocp is None else f'{c.ocp:.0f} mV'} · RPP={'—' if c.rpp is None else f'{c.rpp:.0f} mV'}"
                st.caption(stats)
                if st.button("Entfernen / Remove", key=f"rm{idx}"):
                    st.session_state.curves.pop(idx)
                    st.rerun()

    st.subheader("⚙️ " + t("settings"))
    title_in = st.text_input(t("title_input"), placeholder="z. B. ON2026-0001 · 1.4410 · 65 °C")
    c1, c2 = st.columns(2)
    with c1:
        xmin = st.text_input(t("xmin"), placeholder="auto")
        ymin = st.text_input(t("ymin"), placeholder="auto")
    with c2:
        xmax = st.text_input(t("xmax"), placeholder="auto")
        ymax = st.text_input(t("ymax"), placeholder="auto")
    yscale = st.selectbox(t("yscale"), ["linear", "log (|i|)"])
    smooth = st.slider(t("smoothing"), 0, 50, 0)
    dec_target = st.slider(t("decimate"), 100, 5000, 2000, step=100)
    show_ocp = st.checkbox(t("show_ocp"), value=True)
    show_rpp = st.checkbox(t("show_rpp"), value=True)

with col_right:
    if not st.session_state.curves:
        st.info("📈 " + t("no_curves"))
    else:
        traces, shapes, annotations = [], [], []
        total = displayed = 0
        for idx, c in enumerate(st.session_state.curves):
            total += len(c.V)
            if not c.visible:
                continue
            i_proc = moving_average(c.i, smooth) if smooth > 1 else c.i
            Vd, id_ = decimate(c.V, i_proc, dec_target)
            displayed += len(Vd)
            y = np.abs(id_) if yscale.startswith("log") else id_
            traces.append(go.Scatter(
                x=Vd, y=y, mode="lines", name=c.name,
                line=dict(color=c.color, width=1.6),
                hovertemplate=f"<b>{c.name}</b><br>E = %{{x:.1f}} mV<br>i = %{{y:.4f}} mA/cm²<extra></extra>",
            ))
            if show_ocp and c.ocp is not None:
                shapes.append(dict(type="line", xref="x", yref="paper", x0=c.ocp, x1=c.ocp, y0=0, y1=1,
                                   line=dict(color=c.color, width=1, dash="dash"), opacity=0.5))
                annotations.append(dict(xref="x", yref="paper", x=c.ocp, y=0.96 - idx * 0.05,
                                        text=f"OCP {c.ocp:.0f} mV", showarrow=False,
                                        font=dict(size=9, color=c.color), bgcolor="rgba(255,255,255,0.7)"))
            if show_rpp and c.rpp is not None:
                shapes.append(dict(type="line", xref="x", yref="paper", x0=c.rpp, x1=c.rpp, y0=0, y1=1,
                                   line=dict(color=c.color, width=1, dash="dot"), opacity=0.5))
                annotations.append(dict(xref="x", yref="paper", x=c.rpp, y=0.90 - idx * 0.05,
                                        text=f"RPP {c.rpp:.0f} mV", showarrow=False,
                                        font=dict(size=9, color=c.color), bgcolor="rgba(255,255,255,0.7)"))

        xaxis = dict(title=t("x_label"), gridcolor="#e5e7eb", zerolinecolor="#d1d5db")
        yaxis = dict(title=t("y_label"), gridcolor="#e5e7eb", zerolinecolor="#d1d5db",
                     type="log" if yscale.startswith("log") else "linear")
        try:
            if xmin: xaxis["range"] = [float(xmin), xaxis.get("range", [None, None])[1]]
            if xmax:
                cur = xaxis.get("range", [None, None])
                xaxis["range"] = [cur[0] if cur else None, float(xmax)]
            if ymin: yaxis["range"] = [float(ymin), yaxis.get("range", [None, None])[1]]
            if ymax:
                cur = yaxis.get("range", [None, None])
                yaxis["range"] = [cur[0] if cur else None, float(ymax)]
        except ValueError:
            pass

        layout = go.Layout(
            title=title_in or t("default_title"),
            xaxis=xaxis, yaxis=yaxis,
            shapes=shapes, annotations=annotations,
            legend=dict(orientation="h", y=-0.18, x=0),
            margin=dict(l=70, r=30, t=60, b=80),
            paper_bgcolor="white", plot_bgcolor="white",
            hovermode="closest", height=620,
        )
        fig = go.Figure(data=traces, layout=layout)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": (title_in or "IU_curve").replace(" ", "_")[:50],
                    "width": 1400, "height": 800, "scale": 2,
                },
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
        st.caption("📸 PNG speichern: Klicke auf das Kamera-Icon oben rechts im Diagramm.")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric(t("info_loaded"), f"{sum(1 for c in st.session_state.curves if c.visible)}/{len(st.session_state.curves)}")
        sc2.metric(t("info_total"), f"{total:,}")
        sc3.metric(t("info_displayed"), f"{displayed:,}")

        # CSV export (PNG via Plotly toolbar camera icon)
        rows = []
        for c in st.session_state.curves:
            if not c.visible:
                continue
            i_proc = moving_average(c.i, smooth) if smooth > 1 else c.i
            Vd, id_ = decimate(c.V, i_proc, dec_target)
            for v, ii in zip(Vd, id_):
                rows.append({"Curve": c.name, "E_mV": v, "i_mA_per_cm2": ii})
        if rows:
            df = pd.DataFrame(rows)
            csv = df.to_csv(sep=";", index=False).encode("utf-8")
            st.download_button(
                "📥 " + t("export_csv"),
                csv,
                file_name="IU_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.caption("MPC² × werchota.ai · Simone I-U Plotter v1.0 · 24.04.2026")
