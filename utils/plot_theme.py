import plotly.graph_objects as go

C_BG       = "#04060f"
C_SURFACE  = "#080e1e"
C_PANEL    = "#0d1528"
C_BORDER   = "#1a2540"
C_ACCENT   = "#00f5d4"
C_ATTACK   = "#f72585"
C_GOLD     = "#ffd60a"
C_VIOLET   = "#7209b7"
C_TEXT     = "#c8d8f0"
C_MUTED    = "#4a6080"

COLORSCALE_SURFACE = [
    [0.0,  "#0a1628"],
    [0.3,  "#0d3060"],
    [0.5,  "#00838f"],
    [0.7,  "#00f5d4"],
    [1.0,  "#b8fff4"],
]

COLORSCALE_CURVATURE = [
    [0.0,  "#04060f"],
    [0.3,  "#7209b7"],
    [0.6,  "#f72585"],
    [1.0,  "#ffd60a"],
]

COLORSCALE_SHIELD = [
    [0.0,  "#04060f"],
    [0.4,  "#7209b7"],
    [0.7,  "#f72585"],
    [1.0,  "#ff6b9d"],
]


def base_layout(title="", height=560):
    return dict(
        title=dict(text=title, font=dict(family="Syne, monospace", size=14, color=C_ACCENT)),
        height=height,
        paper_bgcolor=C_BG,
        plot_bgcolor=C_SURFACE,
        font=dict(family="Space Mono, monospace", size=11, color=C_TEXT),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            bgcolor=C_PANEL,
            bordercolor=C_BORDER,
            borderwidth=1,
            font=dict(size=10, color=C_TEXT),
        ),
    )


def scene_3d(title=""):
    axis = dict(
        backgroundcolor=C_SURFACE,
        gridcolor=C_BORDER,
        showbackground=True,
        zerolinecolor=C_MUTED,
        tickfont=dict(size=9, color=C_MUTED),
    )
    return dict(
        scene=dict(
            xaxis=dict(**axis, title="x"),
            yaxis=dict(**axis, title="y"),
            zaxis=dict(**axis, title="z (latente)"),
            bgcolor=C_BG,
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
            aspectmode="cube",
        ),
        **base_layout(title),
    )


def surface_trace(XX, YY, ZZ, name="Variedad f(x,y)", opacity=0.65, colorscale=None):
    return go.Surface(
        x=XX, y=YY, z=ZZ,
        colorscale=colorscale or COLORSCALE_SURFACE,
        opacity=opacity,
        name=name,
        showscale=False,
        contours=dict(
            z=dict(show=True, color=C_BORDER, width=1),
        ),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra>%s</extra>" % name,
    )


def path_trace_3d(xs, ys, zs, name, color, dash="solid", width=5):
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate=f"{name}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>",
    )


def point_trace_3d(x, y, z, name, color, size=10, symbol="circle"):
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers+text",
        name=name,
        text=[name],
        textposition="top center",
        textfont=dict(color=color, size=11),
        marker=dict(color=color, size=size, symbol=symbol,
                    line=dict(color=C_BG, width=2)),
    )
