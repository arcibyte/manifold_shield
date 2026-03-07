
import sys; sys.path.append("..")
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from utils.math_manifold import (
    make_surface, manifold_z, manifold_gradient,
    apply_shield_grid, adversarial_perturbation, gaussian_curvature,
)
from utils.plot_theme import (
    C_ACCENT, C_ATTACK, C_GOLD, C_BG, C_PANEL, C_BORDER, C_MUTED, C_TEXT,
    surface_trace, scene_3d, base_layout, COLORSCALE_SHIELD,
)

st.set_page_config(page_title="Protección | Manifold Shield", layout="wide", page_icon="◈")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@800&display=swap');
html,[class*="css"]{font-family:'Space Mono',monospace;}
.formula-box{background:#04060f;border-left:3px solid #7209b7;padding:10px 14px;
border-radius:0 4px 4px 0;font-size:12px;color:#c8d8f0;margin:10px 0;line-height:1.9;}
.shield-active{background:rgba(255,214,10,0.08);border:1px solid rgba(255,214,10,0.35);
border-radius:4px;padding:10px 16px;font-size:12px;color:#ffd60a;margin:10px 0;}
.shield-off{background:rgba(247,37,133,0.08);border:1px solid rgba(247,37,133,0.35);
border-radius:4px;padding:10px 16px;font-size:12px;color:#f72585;margin:10px 0;}
[data-testid="metric-container"]{background:#0d1528;border:1px solid #1a2540;border-radius:4px;padding:10px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style='font-family:Syne,monospace;font-size:28px;font-weight:800;color:#ffd60a;letter-spacing:-1px;margin-bottom:4px;'>
◈ Protección Adversarial
</div>
<div style='font-size:10px;color:#4a6080;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;'>
Módulo 3 — Escudo Matemático: Perturbación por Gradiente para Proteger Datos Biométricos
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◈ Controles del Escudo")
    st.markdown("---")
    k = st.slider("Curvatura k", 0.5, 2.5, 1.0, 0.1)
    epsilon = st.slider("ε (epsilon — intensidad del escudo)", 0.01, 0.5, 0.12, 0.01,
                        help="Perturbación: x' = x + ε·∇f(x,y)")
    iters = st.slider("Iteraciones del escudo", 1, 20, 6,
                      help="Más iteraciones = protección más profunda")
    show_vectors = st.checkbox("Mostrar vectores ∇f", True)
    show_original = st.checkbox("Mostrar superficie original", True)
    n_surf = st.select_slider("Resolución", [25, 40, 55], 40)

    st.markdown("---")
    st.markdown("""
    <div class='formula-box'>
    <b>Ataque adversarial:</b><br>
    x' = x + ε · ∇ₓ f(x,y)<br><br>
    El gradiente ∇f apunta en la dirección de mayor cambio en la variedad.<br><br>
    Al desplazar los datos en esa dirección, los movemos fuera del manifold sin cambiarlos visualmente.
    </div>""", unsafe_allow_html=True)

#Cálculos
XX, YY, ZZ = make_surface(n=n_surf, k=k)

# Escudo sobre la grilla
XX_s, YY_s = apply_shield_grid(XX, YY, epsilon=epsilon, iters=iters, k=k)
ZZ_s = manifold_z(XX_s, YY_s, k) + np.random.default_rng(0).normal(0, epsilon * 0.3, XX_s.shape)

# Métricas
distortion = float(np.mean(np.sqrt((XX_s - XX)**2 + (YY_s - YY)**2)))
z_error = float(np.mean(np.abs(ZZ_s - ZZ)))
shield_pct = min(100, distortion / epsilon * 100 * 2)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Distorsión media", f"{distortion:.4f}", help="Desplazamiento promedio en (x,y)")
m2.metric("Error en z", f"{z_error:.4f}", help="Distancia media al manifold original")
m3.metric("ε aplicado", f"{epsilon:.3f}")
m4.metric("Escudo efectivo", f"{shield_pct:.0f}%",
          delta="✓ Activo" if shield_pct > 20 else "⚠ Débil")

st.markdown("<br>", unsafe_allow_html=True)

col3d, col_info = st.columns([3, 1])

with col3d:
    fig = go.Figure()

    # Superficie original
    if show_original:
        fig.add_trace(surface_trace(XX, YY, ZZ, name="Variedad Original",
                                   opacity=0.3))

    # Superficie protegida (distorsionada)
    fig.add_trace(go.Surface(
        x=XX_s, y=YY_s, z=ZZ_s,
        colorscale=COLORSCALE_SHIELD,
        opacity=0.7,
        name="Variedad Protegida",
        showscale=False,
        hovertemplate="x'=%{x:.2f}<br>y'=%{y:.2f}<br>z'=%{z:.2f}<extra>Protegida</extra>",
    ))

    # Vectores gradiente (escudo)
    if show_vectors:
        xs_v = np.linspace(-2.5, 2.5, 10)
        ys_v = np.linspace(-2.5, 2.5, 10)
        for xv in xs_v:
            for yv in ys_v:
                zv = float(manifold_z(xv, yv, k))
                gx, gy = manifold_gradient(xv, yv, k)
                scale = epsilon * 0.7
                gn = np.sqrt(gx**2 + gy**2) + 1e-8
                ex = xv + scale * gx / gn
                ey = yv + scale * gy / gn
                ez = float(manifold_z(ex, ey, k))
                fig.add_trace(go.Scatter3d(
                    x=[xv, ex], y=[yv, ey], z=[zv, ez],
                    mode="lines",
                    line=dict(color=f"rgba(247,37,133,{min(0.7, epsilon*3)})", width=2),
                    showlegend=False,
                ))

    fig.update_layout(**scene_3d("Superficie Original vs Superficie Protegida (Escudo Adversarial)"),
                      height=560)
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    if shield_pct > 30:
        st.markdown(f"""
        <div class='shield-active'>
        ESCUDO ACTIVO<br><br>
        Efectividad: {shield_pct:.0f}%<br>
        ε = {epsilon:.3f}<br>
        Iteraciones: {iters}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='shield-off'>
        ESCUDO DÉBIL<br><br>
        Aumenta ε o iteraciones
        </div>""", unsafe_allow_html=True)

    st.markdown("**Principio del Escudo**")
    st.markdown("""
    <div style='font-size:11px;color:#c8d8f0;line-height:1.9;'>
    El gradiente ∇f apunta hacia la zona de mayor curvatura.<br><br>
    Al mover los datos en esa dirección, salen de la "zona reconocible" de la variedad.<br><br>
    Una IA que intente replicarlos encontrará una geometría deformada → no puede generar un deepfake coherente.
    </div>""", unsafe_allow_html=True)

#Vista 2D del efecto del escudo
st.markdown("---")
st.markdown("### Desplazamiento del Escudo — Vista Superior (XY)")

col2d_a, col2d_b = st.columns(2)

#Grilla 2D de puntos
pts_x = np.linspace(-2.5, 2.5, 20)
pts_y = np.linspace(-2.5, 2.5, 20)
PX, PY = np.meshgrid(pts_x, pts_y)
PX_s, PY_s = apply_shield_grid(PX, PY, epsilon=epsilon, iters=iters, k=k)

with col2d_a:
    fig2a = go.Figure()
    fig2a.add_trace(go.Scatter(
        x=PX.flatten(), y=PY.flatten(),
        mode="markers", name="Datos Originales",
        marker=dict(color=C_ACCENT, size=6, opacity=0.8,
                    line=dict(color=C_BG, width=1)),
    ))
    fig2a.update_layout(
        **base_layout("Datos Originales (sin escudo)"), height=320,
        xaxis=dict(title="x", gridcolor=C_BORDER, color=C_MUTED, range=[-3.2, 3.2]),
        yaxis=dict(title="y", gridcolor=C_BORDER, color=C_MUTED, range=[-3.2, 3.2]),
    )
    st.plotly_chart(fig2a, use_container_width=True)

with col2d_b:
    fig2b = go.Figure()
    for i in range(0, len(PX.flatten()), 3):
        x0 = PX.flatten()[i]; y0 = PY.flatten()[i]
        x1 = PX_s.flatten()[i]; y1 = PY_s.flatten()[i]
        fig2b.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor=f"rgba(247,37,133,0.5)",
        )
    fig2b.add_trace(go.Scatter(
        x=PX_s.flatten(), y=PY_s.flatten(),
        mode="markers", name="Datos Protegidos",
        marker=dict(color=C_ATTACK, size=6, opacity=0.8,
                    line=dict(color=C_BG, width=1)),
    ))
    fig2b.update_layout(
        **base_layout(f"Datos Protegidos (ε={epsilon:.2f}, {iters} iteraciones)"), height=320,
        xaxis=dict(title="x'", gridcolor=C_BORDER, color=C_MUTED, range=[-3.2, 3.2]),
        yaxis=dict(title="y'", gridcolor=C_BORDER, color=C_MUTED, range=[-3.2, 3.2]),
    )
    st.plotly_chart(fig2b, use_container_width=True)

st.markdown("---")
st.markdown("### Efecto del ε sobre la Distorsión y Detectabilidad")

epsilons = np.linspace(0.01, 0.5, 50)
distortions = []
detectabilities = []

rng = np.random.default_rng(0)
sample_x = rng.uniform(-2, 2, 100)
sample_y = rng.uniform(-2, 2, 100)

for ep in epsilons:
    sx, sy = apply_shield_grid(sample_x.copy(), sample_y.copy(), epsilon=ep, iters=iters, k=k)
    dist = float(np.mean(np.sqrt((sx - sample_x)**2 + (sy - sample_y)**2)))
    detect = min(1.0, dist * 5) 
    distortions.append(dist)
    detectabilities.append(detect)

fig_ep = go.Figure()
fig_ep.add_trace(go.Scatter(
    x=epsilons, y=distortions,
    name="Distorsión (protección ↑)",
    line=dict(color=C_ACCENT, width=2),
    fill="tozeroy", fillcolor="rgba(0,245,212,0.08)",
))
fig_ep.add_trace(go.Scatter(
    x=epsilons, y=detectabilities,
    name="Detectabilidad visual (riesgo ↑)",
    line=dict(color=C_ATTACK, width=2, dash="dash"),
))
fig_ep.add_vline(x=epsilon, line_color=C_GOLD, line_dash="dot",
                 annotation_text=f"ε actual = {epsilon:.2f}",
                 annotation_font_color=C_GOLD)

# Zona óptima
fig_ep.add_vrect(x0=0.08, x1=0.20,
                 fillcolor="rgba(0,245,212,0.04)",
                 layer="below", line_width=0,
                 annotation_text="Zona óptima",
                 annotation_font_color=C_ACCENT,
                 annotation_position="top left")

fig_ep.update_layout(
    **base_layout("Trade-off: Protección vs Detectabilidad del Escudo"),
    height=300,
    xaxis=dict(title="ε (epsilon)", gridcolor=C_BORDER, color=C_MUTED),
    yaxis=dict(title="Magnitud", gridcolor=C_BORDER, color=C_MUTED),
)
st.plotly_chart(fig_ep, use_container_width=True)
st.markdown("""
<div style='font-size:10px;color:#4a6080;'>
La <b style="color:#00f5d4">zona óptima</b> balancea protección máxima sin ser visualmente detectable por un humano.
Un ε demasiado grande protege pero también alerta al atacante.
</div>""", unsafe_allow_html=True)
