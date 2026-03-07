import streamlit as st

st.set_page_config(
    page_title="The Manifold Shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Space Mono', monospace; }

.main { background: #04060f; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080e1e;
    border-right: 1px solid #1a2540;
}

/* Cards */
.manifold-card {
    background: #080e1e;
    border: 1px solid #1a2540;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.manifold-card h4 {
    font-family: 'Syne', sans-serif;
    color: #00f5d4;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Math formula blocks */
.formula-box {
    background: #04060f;
    border-left: 3px solid #7209b7;
    padding: 10px 14px;
    border-radius: 0 4px 4px 0;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #c8d8f0;
    margin: 10px 0;
    line-height: 1.8;
}

/* Status badges */
.badge-safe   { color:#00f5d4; background:rgba(0,245,212,.1); border:1px solid rgba(0,245,212,.3); padding:3px 10px; border-radius:2px; font-size:10px; letter-spacing:1.5px; }
.badge-risk   { color:#f72585; background:rgba(247,37,133,.1); border:1px solid rgba(247,37,133,.3); padding:3px 10px; border-radius:2px; font-size:10px; letter-spacing:1.5px; }
.badge-warn   { color:#ffd60a; background:rgba(255,214,10,.08); border:1px solid rgba(255,214,10,.3); padding:3px 10px; border-radius:2px; font-size:10px; letter-spacing:1.5px; }

/* Metric overrides */
[data-testid="metric-container"] {
    background: #0d1528;
    border: 1px solid #1a2540;
    border-radius: 4px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

#Header
st.markdown("""
<div style="padding:8px 0 20px">
  <div style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; letter-spacing:-1px; line-height:1;">
    The <span style="color:#00f5d4">Manifold</span> <span style="color:#f72585">Shield</span>
  </div>
  <div style="font-size:10px; color:#4a6080; letter-spacing:2.5px; text-transform:uppercase; margin-top:6px;">
    Geometría Latente · Deepfakes · Protección de Datos · Cálculo Multivariable
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

#Navigation cards 
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_geodesica.py",    label=" Geodésica vs Euclídea",    use_container_width=True)
with col2:
    st.page_link("pages/2_deepfake.py",     label=" Simulador Deepfake",       use_container_width=True)
with col3:
    st.page_link("pages/3_proteccion.py",   label=" Protección Adversarial",    use_container_width=True)
with col4:
    st.page_link("pages/4_deteccion.py",    label=" Detección por Curvatura",   use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

#Overview
c1, c2 = st.columns([3, 2])

with c1:
    st.markdown("""
    <div class="manifold-card">
    <h4>¿Qué es esta simulación?</h4>
    <p style="font-size:12px; color:#c8d8f0; line-height:1.9;">
    Esta app muestra cómo el <b style="color:#00f5d4">Cálculo Multivariable</b> y la
    <b style="color:#00f5d4">Geometría Diferencial</b> son la base matemática de los
    deepfakes, la protección de datos biométricos y su detección forense.
    <br><br>
    Usa el menú lateral para navegar por los 4 módulos interactivos.
    </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="manifold-card">
    <h4>Stack Tecnológico</h4>
    <div style="font-size:11px; color:#c8d8f0; line-height:2.2;">
    <b>Streamlit</b> — Frontend / UI<br>
    <b>Plotly</b> — Visualización 3D<br>
    <b>NumPy / SciPy</b> — Cálculo<br>
    <b>Hugging Face Spaces</b> — Deploy
    </div>
    </div>
    """, unsafe_allow_html=True)

#Concept map
st.markdown("### Conexión Matemática")
st.markdown("""
<div class="formula-box">
 <b>Variedad (Manifold)</b>: superficie curva donde viven los datos reales<br>
     f : ℝⁿ → ℝ   →   f(x,y) = sin(k·x)·cos(k·y)<br><br>
  <b>Geodésica</b>: trayectoria mínima sobre la variedad<br>
     γ*(t) = argmin ∫₀¹ ‖γ'(t)‖ dt<br><br>
  <b>Deepfake lineal (falla)</b>: z_t = (1−t)·z_A + t·z_B  →  sale de la variedad<br><br>
  <b>Escudo adversarial</b>: x' = x + ε · ∇ₓ L(x)<br><br>
  <b>Curvatura de Gauss</b>: K = f_xx·f_yy − (f_xy)²
</div>
""", unsafe_allow_html=True)
