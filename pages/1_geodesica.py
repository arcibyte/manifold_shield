
import sys; sys.path.append("..")
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.math_manifold import (
    make_surface, geodesic_path, linear_path,
    geodesic_length, euclidean_length, gaussian_curvature,
)
from utils.plot_theme import (
    C_ACCENT, C_ATTACK, C_GOLD, C_BG, C_MUTED, C_TEXT, C_BORDER, C_PANEL,
    surface_trace, path_trace_3d, point_trace_3d, scene_3d, base_layout,
)

st.set_page_config(page_title="Geodésica | Manifold Shield", layout="wide", page_icon="⌬")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@800&display=swap');
html,[class*="css"]{font-family:'Space Mono',monospace;}
.formula-box{background:#04060f;border-left:3px solid #7209b7;padding:10px 14px;
border-radius:0 4px 4px 0;font-size:12px;color:#c8d8f0;margin:10px 0;line-height:1.9;}
[data-testid="metric-container"]{background:#0d1528;border:1px solid #1a2540;border-radius:4px;padding:10px;}
</style>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Syne,monospace;font-size:28px;font-weight:800;color:#00f5d4;letter-spacing:-1px;margin-bottom:4px;'>
⌬ Geodésica vs Interpolación Euclídea
</div>
<div style='font-size:10px;color:#4a6080;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;'>
Módulo 1 — Cálculo Multivariable en el Espacio Latente
</div>""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⌬ Controles")
    st.markdown("---")
    k = st.slider("Curvatura k", 0.5, 3.0, 1.0, 0.1,
                  help="Controla qué tan curva es la variedad f(x,y) = sin(k·x)·cos(k·y)")

    st.markdown("**Punto A (origen)**")
    ax = st.slider("A — x", -2.5, 0.0, -1.5, 0.1)
    ay = st.slider("A — y", -2.5, 0.0, -1.0, 0.1)

    st.markdown("**Punto B (destino)**")
    bx = st.slider("B — x", 0.0, 2.5, 1.5, 0.1)
    by = st.slider("B — y", 0.0, 2.5, 1.0, 0.1)

    t_val = st.slider("Posición t en trayectoria", 0.0, 1.0, 0.5, 0.01,
                      help="Mueve el punto a lo largo del camino")
    show_error = st.checkbox("Mostrar error de interpolación", True)
    show_gradient = st.checkbox("Mostrar vectores gradiente", False)
    n_surface = st.select_slider("Resolución superficie", [30, 50, 70], 50)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#4a6080;line-height:1.8;'>
    <b style='color:#00f5d4'>Geodésica</b>: camino más corto sobre la superficie<br>
    <b style='color:#f72585'>Línea recta</b>: falla — sale de la variedad
    </div>""", unsafe_allow_html=True)

# ── Cálculos ──────────────────────────────────────────────────────────────────
XX, YY, ZZ = make_surface(n=n_surface, k=k)
xs_geo, ys_geo, zs_geo = geodesic_path(ax, ay, bx, by, steps=150, k=k)
xs_lin, ys_lin, zs_lin, zs_man = linear_path(ax, ay, bx, by, steps=150, k=k)

az = float(np.sin(k * ax) * np.cos(k * ay))
bz = float(np.sin(k * bx) * np.cos(k * by))

geo_len = geodesic_length(ax, ay, bx, by, k=k)
euc_len = euclidean_length(ax, ay, bx, by, k=k)
error = float(np.max(np.abs(zs_lin - zs_man)))
curv_mid = float(abs(gaussian_curvature((ax+bx)/2, (ay+by)/2, k)))

# Punto actual
idx = int(t_val * 149)
gp = (xs_geo[idx], ys_geo[idx], zs_geo[idx])
lp = (xs_lin[idx], ys_lin[idx], zs_lin[idx])
mp_z = float(np.sin(k * lp[0]) * np.cos(k * lp[1]))  # dónde DEBERÍA estar

# ── Métricas ──────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dist. Geodésica",   f"{geo_len:.3f}",  help="Longitud real sobre la variedad")
m2.metric("Dist. Euclídea",    f"{euc_len:.3f}",
          delta=f"+{euc_len - geo_len:.3f} (sobreestima)", delta_color="inverse")
m3.metric("Error máx. lineal", f"{error:.4f}",    help="|z_lineal − z_variedad|")
m4.metric("Curvatura K (med)", f"{curv_mid:.3f}", help="Curvatura de Gauss en el punto medio")

st.markdown("<br>", unsafe_allow_html=True)

# ── Figura 3D ─────────────────────────────────────────────────────────────────
col_plot, col_info = st.columns([3, 1])

with col_plot:
    fig = go.Figure()

    # Superficie
    fig.add_trace(surface_trace(XX, YY, ZZ, opacity=0.55))

    # Geodésica
    fig.add_trace(path_trace_3d(xs_geo, ys_geo, zs_geo,
                                "Geodésica (correcto)", C_GOLD, width=5))

    # Camino lineal
    fig.add_trace(path_trace_3d(xs_lin, ys_lin, zs_lin,
                                "Línea recta (falla)", C_ATTACK, dash="dash", width=3))

    # Error de interpolación (líneas verticales)
    if show_error:
        for i in range(0, 150, 10):
            z_err = zs_man[i]
            z_off = zs_lin[i]
            if abs(z_off - z_err) > 0.005:
                fig.add_trace(go.Scatter3d(
                    x=[xs_lin[i], xs_lin[i]],
                    y=[ys_lin[i], ys_lin[i]],
                    z=[z_off, z_err],
                    mode="lines",
                    line=dict(color="rgba(247,37,133,0.5)", width=2, dash="dot"),
                    name="Error" if i == 0 else None,
                    showlegend=(i == 0),
                ))

    # Vectores gradiente
    if show_gradient:
        for xi in np.linspace(-2, 2, 6):
            for yi in np.linspace(-2, 2, 6):
                zi = float(np.sin(k * xi) * np.cos(k * yi))
                gx = float(k * np.cos(k * xi) * np.cos(k * yi))
                gy = float(-k * np.sin(k * xi) * np.sin(k * yi))
                scale = 0.18
                fig.add_trace(go.Scatter3d(
                    x=[xi, xi + scale * gx],
                    y=[yi, yi + scale * gy],
                    z=[zi, zi],
                    mode="lines",
                    line=dict(color="rgba(114,9,183,0.6)", width=2),
                    showlegend=False,
                ))

    # Puntos A y B
    fig.add_trace(point_trace_3d(ax, ay, az, "A", C_ACCENT, size=11))
    fig.add_trace(point_trace_3d(bx, by, bz, "B", C_ATTACK, size=11))

    # Punto actual (geodésica)
    fig.add_trace(go.Scatter3d(
        x=[gp[0]], y=[gp[1]], z=[gp[2]],
        mode="markers", name=f"γ(t={t_val:.2f})",
        marker=dict(color=C_GOLD, size=10, symbol="diamond",
                    line=dict(color=C_BG, width=2)),
    ))

    # Punto actual (lineal) + error visual
    fig.add_trace(go.Scatter3d(
        x=[lp[0]], y=[lp[1]], z=[lp[2]],
        mode="markers", name=f"Lineal(t={t_val:.2f})",
        marker=dict(color=C_ATTACK, size=8, symbol="cross",
                    line=dict(color=C_BG, width=2)),
    ))
    fig.add_trace(go.Scatter3d(
        x=[lp[0], lp[0]], y=[lp[1], lp[1]], z=[lp[2], mp_z],
        mode="lines",
        line=dict(color=C_ATTACK, width=4),
        name="Error en t",
        showlegend=True,
    ))

    fig.update_layout(**scene_3d("Variedad Latente — Geodésica vs Línea Recta"), height=580)
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.markdown("""
    <div class='formula-box'>
    <b>Variedad:</b><br>
    f(x,y) = sin(k·x)·cos(k·y)<br><br>
    <b>Geodésica:</b><br>
    γ*(t) = argmin ∫‖γ'‖ dt<br><br>
    <b>Interp. lineal:</b><br>
    z_t = (1−t)·z_A + t·z_B<br><br>
    <b>Error:</b><br>
    Δz = |z_lin − z_man|
    </div>""", unsafe_allow_html=True)

    st.markdown("**¿Por qué importa?**")
    st.markdown("""
    <div style='font-size:11px;color:#c8d8f0;line-height:1.8;'>
    En un espacio latente (GAN, VAE), los datos reales <em>siempre</em>
    viven sobre la variedad.<br><br>
    Si interpolamos en línea recta, el punto sale
    de la variedad → imagen borrosa, cara deformada, artefacto.<br><br>
    La geodésica permanece <em>sobre</em> la variedad → resultado coherente.
    </div>""", unsafe_allow_html=True)

# ── Sección 2D: Error plot ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Error de Interpolación a lo largo del camino")

fig2 = go.Figure()
ts = np.linspace(0, 1, 150)
err_arr = np.abs(zs_lin - zs_man)

fig2.add_trace(go.Scatter(
    x=ts, y=err_arr,
    fill="tozeroy",
    fillcolor="rgba(247,37,133,0.15)",
    line=dict(color=C_ATTACK, width=2),
    name="|z_lineal − z_variedad|",
))
fig2.add_vline(x=t_val, line_color=C_GOLD, line_dash="dash",
               annotation_text=f"t={t_val:.2f}", annotation_font_color=C_GOLD)
fig2.add_hline(y=0, line_color=C_ACCENT, line_dash="dot",
               annotation_text="Error = 0 (sobre la variedad)", annotation_font_color=C_ACCENT)

fig2.update_layout(
    **base_layout("Error de la Interpolación Lineal: cuánto se aleja de la variedad"),
    height=280,
    xaxis=dict(title="t", gridcolor=C_BORDER, color=C_MUTED),
    yaxis=dict(title="|Δz|", gridcolor=C_BORDER, color=C_MUTED),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div style='font-size:10px;color:#4a6080;letter-spacing:1px;'>
El error es 0 en t=0 (punto A) y t=1 (punto B) porque ambos pertenecen a la variedad.
El error máximo ocurre en la región de mayor curvatura.
</div>""", unsafe_allow_html=True)
