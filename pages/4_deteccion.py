
import sys; sys.path.append("..")
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.math_manifold import (
    make_surface, manifold_z, gaussian_curvature,
    mean_curvature, curvature_map,
)
from utils.plot_theme import (
    C_ACCENT, C_ATTACK, C_GOLD, C_BG, C_BORDER, C_MUTED, C_TEXT, C_PANEL,
    surface_trace, scene_3d, base_layout, COLORSCALE_CURVATURE,
)

st.set_page_config(page_title="Detección | Manifold Shield", layout="wide", page_icon="◉")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@800&display=swap');
html,[class*="css"]{font-family:'Space Mono',monospace;}
.formula-box{background:#04060f;border-left:3px solid #7209b7;padding:10px 14px;
border-radius:0 4px 4px 0;font-size:12px;color:#c8d8f0;margin:10px 0;line-height:1.9;}
.detected{background:rgba(247,37,133,0.1);border:1px solid rgba(247,37,133,0.4);
border-radius:4px;padding:10px 16px;font-size:13px;color:#f72585;margin:10px 0;text-align:center;}
.authentic{background:rgba(0,245,212,0.08);border:1px solid rgba(0,245,212,0.35);
border-radius:4px;padding:10px 16px;font-size:13px;color:#00f5d4;margin:10px 0;text-align:center;}
[data-testid="metric-container"]{background:#0d1528;border:1px solid #1a2540;border-radius:4px;padding:10px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style='font-family:Syne,monospace;font-size:28px;font-weight:800;color:#b45dff;letter-spacing:-1px;margin-bottom:4px;'>
◉ Detección por Curvatura de Gauss
</div>
<div style='font-size:10px;color:#4a6080;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;'>
Módulo 4 — Análisis Forense de Deepfakes mediante Cálculo Diferencial
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◉ Controles de Detección")
    st.markdown("---")
    k_real = st.slider("Curvatura real k", 0.5, 3.0, 1.0, 0.1,
                       help="Curvatura de los datos auténticos")
    threshold = st.slider("Umbral de detección |K|", 0.0, 2.0, 0.5, 0.05,
                          help="Zonas con |K| > umbral se marcan como sospechosas")
    fake_regions = st.slider("Regiones falsificadas (%)", 0, 60, 25,
                             help="Porcentaje de la superficie que ha sido manipulada")
    noise_level = st.slider("Ruido de manipulación", 0.0, 1.0, 0.4, 0.05)
    n_surf = st.select_slider("Resolución", [30, 45, 60], 45)
    show_3d_curv = st.checkbox("Mostrar curvatura en 3D", True)

    st.markdown("---")
    st.markdown("""
    <div class='formula-box'>
    <b>Curvatura de Gauss:</b><br>
    K = f_xx·f_yy − (f_xy)²<br><br>
    <b>Curvatura media:</b><br>
    H = (f_xx + f_yy) / 2<br><br>
    Un deepfake introduce discontinuidades en K que no existen en datos reales.
    </div>""", unsafe_allow_html=True)

#Simulación de superficie con región falsificada
rng = np.random.default_rng(42)

XX, YY, ZZ = make_surface(n=n_surf, k=k_real)
XX_curv, YY_curv, KK = curvature_map(n=n_surf, k=k_real)

# Inyectar regiones falsificadas
ZZ_fake = ZZ.copy()
KK_fake = KK.copy()
fake_mask = np.zeros_like(ZZ, dtype=bool)

if fake_regions > 0:
    n_fake_pts = int(n_surf * n_surf * fake_regions / 100)
    fake_idx_r = rng.integers(n_surf // 4, 3 * n_surf // 4, n_fake_pts)
    fake_idx_c = rng.integers(n_surf // 4, 3 * n_surf // 4, n_fake_pts)

    for r, c in zip(fake_idx_r, fake_idx_c):
        t_fake = rng.uniform(0.2, 0.8)
        ZZ_fake[r, c] = ZZ[r, c] * (1 - t_fake) + rng.normal(0, noise_level)
        fake_mask[r, c] = True

    KK_fake[fake_mask] += rng.normal(0, noise_level * 2, fake_mask.sum())

# Detección
KK_abs = np.abs(KK_fake)
detected_mask = KK_abs > threshold
true_positives = np.sum(detected_mask & fake_mask)
false_positives = np.sum(detected_mask & ~fake_mask)
false_negatives = np.sum(~detected_mask & fake_mask)
total_fakes = fake_mask.sum()

precision = true_positives / (true_positives + false_positives + 1e-9)
recall    = true_positives / (total_fakes + 1e-9)
f1        = 2 * precision * recall / (precision + recall + 1e-9)

#Métricas
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Regiones falsas",   f"{total_fakes}")
m2.metric("Detectadas (TP)",   f"{true_positives}", delta=f"{recall*100:.0f}% recall")
m3.metric("Falsos positivos",  f"{false_positives}")
m4.metric("Precisión",         f"{precision:.2%}")
m5.metric("F1-Score",          f"{f1:.3f}",
          delta="✓ Bueno" if f1 > 0.6 else "⚠ Ajusta umbral")

# Status
if fake_regions == 0:
    st.markdown('<div class="authentic">✓ DATOS AUTÉNTICOS — Sin anomalías de curvatura detectadas</div>',
                unsafe_allow_html=True)
elif f1 > 0.5:
    st.markdown(f'<div class="detected">⚠ DEEPFAKE DETECTADO — F1={f1:.3f} | {true_positives} regiones comprometidas</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="detected" style="color:#ffd60a;border-color:rgba(255,214,10,0.4);background:rgba(255,214,10,0.06);">⚡ DEEPFAKE PARCIALMENTE DETECTADO — Ajusta el umbral</div>',
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#Gráficos
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Superficie con Regiones Falsificadas")
    if show_3d_curv:
        fig3d = go.Figure()
        fig3d.add_trace(surface_trace(XX, YY, ZZ, "Superficie Auténtica", opacity=0.3))
        fig3d.add_trace(go.Surface(
            x=XX, y=YY, z=ZZ_fake,
            colorscale=COLORSCALE_CURVATURE,
            surfacecolor=KK_abs,
            cmin=0, cmax=float(np.percentile(KK_abs, 95)),
            opacity=0.8,
            name="Superficie Manipulada",
            showscale=True,
            colorbar=dict(
                title=dict(text="|K|", font=dict(color=C_MUTED)),
                tickfont=dict(color=C_MUTED, size=9),
                len=0.6, thickness=10,
                bgcolor=C_PANEL, bordercolor=C_BORDER,
            ),
            hovertemplate="K=%{surfacecolor:.3f}<extra>Curvatura</extra>",
        ))

        # Puntos detectados como deepfake
        det_r, det_c = np.where(detected_mask)
        if len(det_r) > 0:
            fig3d.add_trace(go.Scatter3d(
                x=XX[det_r, det_c].flatten(),
                y=YY[det_r, det_c].flatten(),
                z=ZZ_fake[det_r, det_c].flatten(),
                mode="markers", name="⚠ Sospechoso",
                marker=dict(color=C_ATTACK, size=4, symbol="x",
                            line=dict(color=C_BG, width=1)),
            ))

        fig3d.update_layout(**scene_3d("Mapa de Curvatura |K| — Detección de Manipulación"),
                            height=440)
        st.plotly_chart(fig3d, use_container_width=True)

with col_right:
    st.markdown("#### Mapa de Calor de Curvatura |K|")
    fig_hm = go.Figure()
    fig_hm.add_trace(go.Heatmap(
        z=KK_abs,
        colorscale=COLORSCALE_CURVATURE,
        showscale=True,
        name="|K|",
        colorbar=dict(
            title=dict(text="|K|", font=dict(color=C_MUTED)),
            tickfont=dict(color=C_MUTED, size=9),
            bgcolor=C_PANEL, bordercolor=C_BORDER, thickness=10,
        ),
        hovertemplate="i=%{x}, j=%{y}<br>|K|=%{z:.3f}<extra></extra>",
    ))

    # Marcar detecciones
    det_r2, det_c2 = np.where(detected_mask)
    if len(det_r2) > 0:
        fig_hm.add_trace(go.Scatter(
            x=det_c2, y=det_r2, mode="markers",
            name="⚠ Detectado",
            marker=dict(color=C_ATTACK, size=5, symbol="x",
                        line=dict(color=C_ATTACK, width=1)),
        ))

    # Marcar falsificaciones reales
    real_r, real_c = np.where(fake_mask)
    if len(real_r) > 0:
        fig_hm.add_trace(go.Scatter(
            x=real_c, y=real_r, mode="markers",
            name="■ Falsificado",
            marker=dict(color=C_GOLD, size=4, symbol="square-open",
                        line=dict(color=C_GOLD, width=1.5)),
        ))

    fig_hm.update_layout(
        **base_layout(f"Mapa |K| — Umbral = {threshold:.2f}"),
        height=440,
        xaxis=dict(title="j", gridcolor=C_BORDER, color=C_MUTED),
        yaxis=dict(title="i", gridcolor=C_BORDER, color=C_MUTED),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

#Distribución de curvatura
st.markdown("---")
st.markdown("### Distribución de Curvatura: Real vs Falsificado")

_, _, KK_real = curvature_map(n=n_surf, k=k_real)

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=KK_real.flatten(),
    name="Curvatura Auténtica",
    nbinsx=60,
    marker_color=C_ACCENT,
    opacity=0.6,
    histnorm="probability density",
))
fig_dist.add_trace(go.Histogram(
    x=KK_fake.flatten(),
    name="Curvatura con Deepfake",
    nbinsx=60,
    marker_color=C_ATTACK,
    opacity=0.6,
    histnorm="probability density",
))
fig_dist.add_vline(x=threshold, line_color=C_GOLD, line_dash="dash",
                   annotation_text=f"Umbral = {threshold:.2f}",
                   annotation_font_color=C_GOLD)
fig_dist.add_vline(x=-threshold, line_color=C_GOLD, line_dash="dash")

fig_dist.update_layout(
    **base_layout("Distribución de Curvatura K — Datos Reales vs Manipulados"),
    barmode="overlay",
    height=300,
    xaxis=dict(title="K (curvatura de Gauss)", gridcolor=C_BORDER, color=C_MUTED),
    yaxis=dict(title="Densidad", gridcolor=C_BORDER, color=C_MUTED),
)
st.plotly_chart(fig_dist, use_container_width=True)

#Curva ROC simplificada
st.markdown("### Curva ROC del Detector")
thresholds = np.linspace(0.01, 3.0, 100)
tprs, fprs = [], []
for thr in thresholds:
    det = np.abs(KK_fake) > thr
    tp = np.sum(det & fake_mask)
    fp = np.sum(det & ~fake_mask)
    fn = np.sum(~det & fake_mask)
    tn = np.sum(~det & ~fake_mask)
    tprs.append(tp / (tp + fn + 1e-9))
    fprs.append(fp / (fp + tn + 1e-9))

auc = float(np.trapz(tprs[::-1], fprs[::-1]))

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fprs, y=tprs,
    mode="lines", name=f"ROC (AUC = {auc:.3f})",
    line=dict(color="#b45dff", width=2.5),
    fill="tozeroy", fillcolor="rgba(114,9,183,0.1)",
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines", name="Azar (AUC=0.5)",
    line=dict(color=C_MUTED, width=1, dash="dash"),
))

# Punto actual
cur_tpr = true_positives / (total_fakes + 1e-9)
cur_fpr = false_positives / (np.sum(~fake_mask) + 1e-9)
fig_roc.add_trace(go.Scatter(
    x=[cur_fpr], y=[cur_tpr],
    mode="markers", name=f"Umbral actual ({threshold:.2f})",
    marker=dict(color=C_GOLD, size=12, symbol="diamond"),
))

fig_roc.update_layout(
    **base_layout("Curva ROC — Detector de Curvatura"),
    height=320,
    xaxis=dict(title="Tasa de Falsos Positivos", gridcolor=C_BORDER, color=C_MUTED,
               range=[0, 1]),
    yaxis=dict(title="Tasa de Verdaderos Positivos", gridcolor=C_BORDER, color=C_MUTED,
               range=[0, 1]),
)
st.plotly_chart(fig_roc, use_container_width=True)
st.markdown(f"""
<div style='font-size:10px;color:#4a6080;'>
AUC = <b style="color:#b45dff">{auc:.3f}</b> — cuanto más cercano a 1.0, mejor el detector.
Un clasificador aleatorio tendría AUC = 0.5.
Mueve el umbral para encontrar el balance TPR/FPR óptimo para tu caso de uso.
</div>""", unsafe_allow_html=True)
