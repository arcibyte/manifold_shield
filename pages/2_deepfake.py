
import sys; sys.path.append("..")
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFilter
import io, math
from utils.math_manifold import make_surface, geodesic_path, linear_path, manifold_z
from utils.plot_theme import (
    C_ACCENT, C_ATTACK, C_GOLD, C_BG, C_PANEL, C_BORDER, C_MUTED, C_TEXT,
    surface_trace, path_trace_3d, point_trace_3d, scene_3d, base_layout,
    COLORSCALE_SURFACE,
)

st.set_page_config(page_title="Deepfake | Manifold Shield", layout="wide", page_icon="⚡")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@800&display=swap');
html,[class*="css"]{font-family:'Space Mono',monospace;}
.formula-box{background:#04060f;border-left:3px solid #7209b7;padding:10px 14px;
border-radius:0 4px 4px 0;font-size:12px;color:#c8d8f0;margin:10px 0;line-height:1.9;}
.face-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#4a6080;
text-align:center;margin-top:8px;}
.artifact-warn{background:rgba(247,37,133,0.1);border:1px solid rgba(247,37,133,0.4);
border-radius:4px;padding:8px 14px;font-size:11px;color:#f72585;margin:8px 0;}
.geo-ok{background:rgba(0,245,212,0.08);border:1px solid rgba(0,245,212,0.3);
border-radius:4px;padding:8px 14px;font-size:11px;color:#00f5d4;margin:8px 0;}
[data-testid="metric-container"]{background:#0d1528;border:1px solid #1a2540;border-radius:4px;padding:10px;}
</style>""", unsafe_allow_html=True)

# FACE GENERATOR  (100% sintético con PIL — sin imágenes externas)
def _lerp(a, b, t): return a + (b - a) * t
def _lerp_color(ca, cb, t):
    return tuple(int(_lerp(ca[i], cb[i], t)) for i in range(3))

class FaceParams:
    """Parámetros de un rostro sintético en el espacio latente."""
    def __init__(self, skin, eye_color, mouth_curve, eye_size,
                 brow_angle, face_width, head_tilt, nose_width, cheek_blush):
        self.skin        = skin          # (r,g,b)
        self.eye_color   = eye_color     # (r,g,b)
        self.mouth_curve = mouth_curve   # -1..1 (fruncido→sonrisa)
        self.eye_size    = eye_size      # 0.5..1.5
        self.brow_angle  = brow_angle    # -20..20 deg
        self.face_width  = face_width    # 0.8..1.2
        self.head_tilt   = head_tilt     # -15..15 deg
        self.nose_width  = nose_width    # 5..18 px
        self.cheek_blush = cheek_blush   # 0..1

# Dos identidades extremas
FACE_A = FaceParams(
    skin=(220, 170, 130), eye_color=(60, 100, 180),
    mouth_curve=0.7,  eye_size=1.1,  brow_angle=-5,
    face_width=1.0,   head_tilt=0,   nose_width=8,  cheek_blush=0.35,
)
FACE_B = FaceParams(
    skin=(160, 110, 80),  eye_color=(50, 140, 70),
    mouth_curve=-0.4, eye_size=0.8,  brow_angle=12,
    face_width=1.15,  head_tilt=5,   nose_width=14, cheek_blush=0.1,
)

def interp_face_linear(fa: FaceParams, fb: FaceParams, t: float) -> FaceParams:
    """Interpolación lineal (euclídea) — produce artefactos."""
    return FaceParams(
        skin        = _lerp_color(fa.skin, fb.skin, t),
        eye_color   = _lerp_color(fa.eye_color, fb.eye_color, t),
        mouth_curve = _lerp(fa.mouth_curve, fb.mouth_curve, t),
        eye_size    = _lerp(fa.eye_size, fb.eye_size, t),
        brow_angle  = _lerp(fa.brow_angle, fb.brow_angle, t),
        face_width  = _lerp(fa.face_width, fb.face_width, t),
        head_tilt   = _lerp(fa.head_tilt, fb.head_tilt, t),
        nose_width  = _lerp(fa.nose_width, fb.nose_width, t),
        cheek_blush = _lerp(fa.cheek_blush, fb.cheek_blush, t),
    )

def geodesic_t(t: float, k: float = 1.0) -> float:
    """
    Función de reparametrización geodésica.
    Curva el parámetro t según la curvatura de la variedad,
    lo que mantiene coherencia semántica.
    """
    # Aproximamos la geodésica con una parametrización basada en sin²
    # que ralentiza el movimiento en zonas de alta curvatura
    s = np.sin(np.pi * t / 2) ** 2
    correction = 0.5 * np.sin(2 * np.pi * t) * (k - 1) * 0.15
    return float(np.clip(s + correction, 0, 1))

def interp_face_geodesic(fa: FaceParams, fb: FaceParams, t: float, k: float = 1.0) -> FaceParams:
    """Interpolación geodésica — coherente sobre la variedad."""
    tg = geodesic_t(t, k)
    return interp_face_linear(fa, fb, tg)

def add_artifact(img: Image.Image, t: float, intensity: float) -> Image.Image:
    """Añade artefactos visuales que aparecen con la interpolación lineal."""
    arr = np.array(img).astype(float)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(seed=42 + int(t * 1000))

    # Artefacto 1: banda fantasma horizontal en zona media (ojos→boca)
    band_strength = intensity * abs(math.sin(t * math.pi)) * 35
    y0, y1 = int(h * 0.38), int(h * 0.62)
    noise = rng.normal(0, band_strength, (y1 - y0, w, 3))
    arr[y0:y1] = np.clip(arr[y0:y1] + noise, 0, 255)

    # Artefacto 2: ghosting (borde borroso de la cara)
    ghost_alpha = intensity * abs(math.sin(t * math.pi)) * 0.4
    if ghost_alpha > 0.05:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=3 + intensity * 4))
        b_arr = np.array(blurred).astype(float)
        arr = arr * (1 - ghost_alpha) + b_arr * ghost_alpha

    # Artefacto 3: desaturación en bordes
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    for c in range(3):
        edge_desat = intensity * 0.3 * abs(math.sin(t * math.pi))
        arr[:, :, c] = arr[:, :, c] * (1 - edge_desat) + gray * edge_desat

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def draw_face(params: FaceParams, size: int = 220) -> Image.Image:
    """Dibuja un rostro sintético a partir de parámetros."""
    img = Image.new("RGB", (size, size), (18, 22, 40))
    d = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    fw = int(size * 0.34 * params.face_width)
    fh = int(size * 0.42)

    # Sombra de cabeza
    shadow_col = tuple(max(0, c - 40) for c in params.skin)
    d.ellipse([cx - fw + 4, cy - fh + 4 + 4, cx + fw + 4, cy + fh + 4],
              fill=(*shadow_col, 80))

    # Cabeza
    d.ellipse([cx - fw, cy - fh + 4, cx + fw, cy + fh], fill=params.skin)

    # Cuello
    neck_w = int(fw * 0.35)
    neck_col = tuple(max(0, c - 15) for c in params.skin)
    d.rectangle([cx - neck_w, cy + fh - 10, cx + neck_w, cy + fh + 30], fill=neck_col)

    # Ropa (silueta básica)
    d.ellipse([cx - fw - 20, cy + fh + 20, cx + fw + 20, cy + fh + 80],
              fill=(40, 50, 80))

    # Mejillas (rubor)
    if params.cheek_blush > 0:
        blush_a = int(params.cheek_blush * 60)
        blush_col = (220, 130, 130, blush_a)
        blush_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blush_img)
        br = int(fw * 0.28)
        bd.ellipse([cx - fw + 8, cy + 4, cx - fw + 8 + br * 2, cy + br * 2],
                   fill=blush_col)
        bd.ellipse([cx + fw - 8 - br * 2, cy + 4, cx + fw - 8, cy + br * 2],
                   fill=blush_col)
        img = Image.alpha_composite(img.convert("RGBA"), blush_img).convert("RGB")
        d = ImageDraw.Draw(img)

    # Ojos
    ey = cy - int(fh * 0.15)
    ex_l = cx - int(fw * 0.45)
    ex_r = cx + int(fw * 0.45)
    er = int(12 * params.eye_size)

    for ex in [ex_l, ex_r]:
        # Blanco del ojo
        d.ellipse([ex - er - 4, ey - er + 2, ex + er + 4, ey + er - 2], fill=(240, 240, 240))
        # Iris
        d.ellipse([ex - er + 2, ey - er + 3, ex + er - 2, ey + er - 3], fill=params.eye_color)
        # Pupila
        pr = max(3, er - 5)
        d.ellipse([ex - pr, ey - pr, ex + pr, ey + pr], fill=(15, 10, 20))
        # Brillo
        d.ellipse([ex - 2, ey - er + 5, ex + 3, ey - er + 10], fill=(255, 255, 255))

    # Cejas
    bw = int(fw * 0.3)
    ba_rad = math.radians(params.brow_angle)
    for sign, ex in [(-1, ex_l), (1, ex_r)]:
        dx = bw * math.cos(ba_rad)
        dy = bw * math.sin(ba_rad) * sign
        by0 = ey - er - 10
        x1 = int(ex - dx); y1 = int(by0 - dy)
        x2 = int(ex + dx); y2 = int(by0 + dy)
        brow_col = tuple(max(0, c - 80) for c in params.skin)
        d.line([x1, y1, x2, y2], fill=brow_col, width=4)

    # Nariz
    ny = cy + int(fh * 0.08)
    nw = int(params.nose_width)
    nose_col = tuple(max(0, c - 25) for c in params.skin)
    d.ellipse([cx - nw, ny - 3, cx + nw, ny + 7], fill=nose_col)
    d.line([cx - nw + 2, ny - 12, cx - nw + 2, ny + 4], fill=nose_col, width=3)
    d.line([cx + nw - 2, ny - 12, cx + nw - 2, ny + 4], fill=nose_col, width=3)

    # Boca
    my = cy + int(fh * 0.32)
    mw = int(fw * 0.45)
    curve = int(params.mouth_curve * 14)
    lip_col = tuple(max(0, c - 50) for c in params.skin)
    lip_col = (min(255, lip_col[0] + 30), max(0, lip_col[1] - 20), max(0, lip_col[2] - 20))

    # Labio superior
    d.line([cx - mw, my, cx, my - curve, cx + mw, my], fill=lip_col, width=4)
    # Labio inferior
    d.line([cx - mw, my, cx, my + abs(curve) // 2 + 4, cx + mw, my],
           fill=lip_col, width=3)

    # Aplicar inclinación si hay head_tilt
    if abs(params.head_tilt) > 0.5:
        img = img.rotate(params.head_tilt, fillcolor=(18, 22, 40), resample=Image.BICUBIC)

    # Suavizado final
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# UI
st.markdown("""
<div style='font-family:Syne,monospace;font-size:28px;font-weight:800;color:#f72585;letter-spacing:-1px;margin-bottom:4px;'>
 Simulador de Deepfake
</div>
<div style='font-size:10px;color:#4a6080;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;'>
Módulo 2 — Interpolación Lineal vs Geodésica en Espacio Latente
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Controles Deepfake")
    st.markdown("---")
    t_val = st.slider("Paso de interpolación t", 0.0, 1.0, 0.5, 0.01,
                      help="0 = Identidad A pura | 1 = Identidad B pura")
    k_curv = st.slider("Curvatura de la variedad k", 0.5, 3.0, 1.0, 0.1)
    artifact_intensity = st.slider("Intensidad de artefactos (lineal)", 0.0, 1.0, 0.6, 0.05,
                                   help="Cuánto se degrada la interpolación lineal")
    face_size = st.select_slider("Tamaño de rostro", [160, 200, 240], 200)

    st.markdown("---")
    st.markdown("""
    <div class='formula-box'>
    <b>Lineal:</b><br>
    z_t = (1−t)·z_A + t·z_B<br>
    → SALE de la variedad<br><br>
    <b>Geodésica:</b><br>
    z_t = γ*(t) sobre M<br>
    → PERMANECE coherente
    </div>""", unsafe_allow_html=True)

# Generar rostros
face_a_img  = draw_face(FACE_A, face_size)
face_b_img  = draw_face(FACE_B, face_size)

# Interpolación lineal (con artefactos)
params_lin  = interp_face_linear(FACE_A, FACE_B, t_val)
face_lin    = draw_face(params_lin, face_size)
if t_val > 0.05 and t_val < 0.95:
    face_lin = add_artifact(face_lin, t_val, artifact_intensity)

# Interpolación geodésica (limpia)
params_geo  = interp_face_geodesic(FACE_A, FACE_B, t_val, k_curv)
face_geo    = draw_face(params_geo, face_size)

# Fila de rostros
st.markdown("### Comparación Visual de Interpolación")

ca, arr1, cl, arr2, cg, arr3, cb_ = st.columns([2, 0.4, 2, 0.4, 2, 0.4, 2])

with ca:
    st.image(face_a_img, caption="Identidad A (original)", use_container_width=True)
    st.markdown('<div class="face-label">z_A ∈ Variedad</div>', unsafe_allow_html=True)

with arr1:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:24px;text-align:center;color:#4a6080;">→</div>', unsafe_allow_html=True)

with cl:
    st.image(face_lin, caption=f"Lineal t={t_val:.2f} ⚠ artefactos", use_container_width=True)
    err_z = abs(manifold_z(
        -1.5 + 3.0 * t_val,
        -1.0 + 2.0 * t_val, k_curv
    ) - (manifold_z(-1.5, -1.0, k_curv) * (1 - t_val) + manifold_z(1.5, 1.0, k_curv) * t_val))

    if t_val > 0.05 and t_val < 0.95:
        st.markdown(
            f'<div class="artifact-warn">⚠ Error Δz = {err_z:.4f} — punto fuera de la variedad</div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="geo-ok">✓ En extremo — error mínimo</div>', unsafe_allow_html=True)

with arr2:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:24px;text-align:center;color:#4a6080;">→</div>', unsafe_allow_html=True)

with cg:
    st.image(face_geo, caption=f"Geodésica t={t_val:.2f} ✓ coherente", use_container_width=True)
    st.markdown('<div class="geo-ok">✓ Sobre la variedad — resultado realista</div>',
                unsafe_allow_html=True)

with arr3:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:24px;text-align:center;color:#4a6080;">→</div>', unsafe_allow_html=True)

with cb_:
    st.image(face_b_img, caption="Identidad B (destino)", use_container_width=True)
    st.markdown('<div class="face-label">z_B ∈ Variedad</div>', unsafe_allow_html=True)

# Métricas
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)

tg = geodesic_t(t_val, k_curv)
m1.metric("t lineal",     f"{t_val:.3f}")
m2.metric("t geodésico",  f"{tg:.3f}", delta=f"Δ = {tg - t_val:+.3f}")
m3.metric("Error Δz",     f"{err_z:.4f}", delta="fuera del manifold" if err_z > 0.01 else "sobre el manifold")
m4.metric("Curvatura k",  f"{k_curv:.1f}")

# Gráfico 3D del espacio latente
st.markdown("---")
st.markdown("### Trayectorias en el Espacio Latente 3D")

col3d, col_exp = st.columns([3, 1])

with col3d:
    XX, YY, ZZ = make_surface(n=45, k=k_curv)
    ax, ay = -1.5, -1.0
    bx, by =  1.5,  1.0
    xs_g, ys_g, zs_g = geodesic_path(ax, ay, bx, by, 120, k_curv)
    xs_l, ys_l, zs_l, zs_m = linear_path(ax, ay, bx, by, 120, k_curv)

    idx = int(t_val * 119)

    fig = go.Figure()
    fig.add_trace(surface_trace(XX, YY, ZZ, opacity=0.5))

    # Camino geodésico
    fig.add_trace(path_trace_3d(xs_g, ys_g, zs_g, "Geodésica ✓", C_GOLD, width=5))

    # Camino lineal
    fig.add_trace(path_trace_3d(xs_l, ys_l, zs_l, "Lineal ✗", C_ATTACK, dash="dash", width=3))

    # Puntos actuales
    fig.add_trace(go.Scatter3d(
        x=[xs_g[idx]], y=[ys_g[idx]], z=[zs_g[idx]],
        mode="markers", name="Punto geodésico",
        marker=dict(color=C_GOLD, size=12, symbol="diamond",
                    line=dict(color=C_BG, width=2)),
    ))
    fig.add_trace(go.Scatter3d(
        x=[xs_l[idx]], y=[ys_l[idx]], z=[zs_l[idx]],
        mode="markers", name="Punto lineal (artefacto)",
        marker=dict(color=C_ATTACK, size=10, symbol="cross",
                    line=dict(color=C_BG, width=2)),
    ))

    # Error gap
    fig.add_trace(go.Scatter3d(
        x=[xs_l[idx], xs_l[idx]], y=[ys_l[idx], ys_l[idx]],
        z=[zs_l[idx], zs_m[idx]],
        mode="lines", line=dict(color=C_ATTACK, width=5),
        name=f"Artefacto Δz={err_z:.3f}",
    ))

    fig.add_trace(point_trace_3d(ax, ay, float(np.sin(k_curv*ax)*np.cos(k_curv*ay)), "A", C_ACCENT, 11))
    fig.add_trace(point_trace_3d(bx, by, float(np.sin(k_curv*bx)*np.cos(k_curv*by)), "B", "#b45dff", 11))

    fig.update_layout(**scene_3d("Espacio Latente — Identidades A y B"), height=520)
    st.plotly_chart(fig, use_container_width=True)

with col_exp:
    st.markdown("**¿Qué ves?**")
    st.markdown("""
    <div style='font-size:11px;color:#c8d8f0;line-height:1.9;'>
    🟡 <b>Línea dorada</b>:<br>
    geodésica — sigue la curvatura del manifold.<br><br>
    🔴 <b>Línea roja</b>:<br>
    interpolación lineal — va en línea recta y <em>sale</em> de la superficie.<br><br>
    🔴 <b>Barra vertical</b>:<br>
    error Δz — distancia al manifold. Cuanto mayor, más artefactos en el rostro.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='formula-box'>
    <b>Por qué falla el deepfake lineal:</b><br><br>
    z_t = (1−t)·z_A + t·z_B<br><br>
    Este punto <em>no existe</em> en la distribución real de datos. El decoder de la GAN genera algo que "no conoce" → artefactos, caras fantasmas, bordes borrosos.
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Strip de Interpolación (t = 0 → 1)")
st.markdown(
    '<div style="font-size:10px;color:#4a6080;margin-bottom:12px;">Comparación cuadro a cuadro: arriba = lineal, abajo = geodésico</div>',
    unsafe_allow_html=True)

n_frames = 7
cols_strip = st.columns(n_frames)
ts_strip = np.linspace(0, 1, n_frames)

for i, (col_s, t_s) in enumerate(zip(cols_strip, ts_strip)):
    p_lin = interp_face_linear(FACE_A, FACE_B, t_s)
    f_lin = draw_face(p_lin, 100)
    if t_s > 0.05 and t_s < 0.95:
        f_lin = add_artifact(f_lin, t_s, artifact_intensity * 0.7)

    p_geo = interp_face_geodesic(FACE_A, FACE_B, t_s, k_curv)
    f_geo = draw_face(p_geo, 100)

    with col_s:
        st.image(f_lin, caption=f"L t={t_s:.2f}", use_container_width=True)
        st.image(f_geo, caption=f"G t={t_s:.2f}", use_container_width=True)
