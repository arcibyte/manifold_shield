---
title: The Manifold Shield
emoji: ⬡
colorFrom: cyan
colorTo: pink
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# ⬡ The Manifold Shield

**Geometría Latente · Deepfakes · Protección de Datos · Cálculo Multivariable**

Simulación interactiva que conecta el **Cálculo Multivariable** con la **Ingeniería en Ciencia de Datos** a través del problema de deepfakes y protección biométrica.

---

## Módulos

### ⌬ 1. Geodésica vs Interpolación Euclídea
Visualiza en 3D cómo la interpolación lineal sale de la variedad de datos mientras que la geodésica permanece sobre ella.

- Controla la curvatura `k` de la variedad `f(x,y) = sin(k·x)·cos(k·y)`
- Mueve el punto a lo largo de ambas trayectorias con el slider `t`
- Observa el error `Δz = |z_lineal − z_variedad|`

**Matemáticas:** `γ*(t) = argmin ∫₀¹ ‖γ'(t)‖ dt`

---

### ⚡ 2. Simulador de Deepfake
Compara visualmente el resultado de interpolar entre dos identidades usando:
- **Interpolación lineal** → genera artefactos (ghosting, bordes borrosos)
- **Interpolación geodésica** → resultado coherente y realista

Incluye un strip de 7 frames mostrando la transición completa.

**Matemáticas:** `z_t = (1−t)·z_A + t·z_B` vs `z_t = γ*(t)`

---

### ◈ 3. Protección Adversarial
Simula el "Escudo" matemático: perturbación del gradiente que desplaza los datos fuera de la variedad, haciéndolos inaccesibles para una IA generativa.

- Visualiza los vectores `∇f` que definen la dirección del escudo
- Controla `ε` e iteraciones para encontrar la zona óptima
- Analiza el trade-off Protección vs Detectabilidad

**Matemáticas:** `x' = x + ε · ∇ₓ f(x,y)`

---

### ◉ 4. Detección por Curvatura de Gauss
Detector forense que analiza discontinuidades en la curvatura de Gauss `K` para identificar regiones manipuladas.

- Mapa de calor de `|K|` sobre la superficie
- Curva ROC del detector
- Métricas de Precisión, Recall y F1-Score

**Matemáticas:** `K = f_xx · f_yy − (f_xy)²`

---

## Stack Tecnológico

| Componente | Herramienta |
|---|---|
| Frontend/UI | Streamlit |
| Visualización 3D | Plotly |
| Cálculo | NumPy / SciPy |
| Imágenes | Pillow |
| Deploy | Hugging Face Spaces |

---

## Ejecución Local

```bash
git clone https://huggingface.co/spaces/TU_USUARIO/manifold-shield
cd manifold-shield
pip install -r requirements.txt
streamlit run app.py
```

---

## Conexión Matemática

```
Variedad:    f : ℝ² → ℝ   →   f(x,y) = sin(k·x)·cos(k·y)
Geodésica:   γ*(t) = argmin ∫‖γ'(t)‖ dt
Gradiente:   ∇f = (∂f/∂x, ∂f/∂y) = (k·cos(kx)·cos(ky), −k·sin(kx)·sin(ky))
Hessiana:    H = [[f_xx, f_xy], [f_xy, f_yy]]
Curvatura:   K = f_xx·f_yy − (f_xy)²
Escudo:      x' = x + ε·∇f(x,y)
```

---

*Proyecto académico — Ingeniería en Ciencia de Datos · Bucaramanga, Colombia*
