"""Funciones que construyen las figuras de Plotly reutilizadas por las paginas."""
import numpy as np
import plotly.graph_objects as go

from . import data as D
from .data import CLASES, COLOR, CORTO, ETIQ, LARGO, fmt, miles

LR_F1 = D.BASE["test"]["F1 macro"]
LR_ACC = D.BASE["test"]["Accuracy"]


GRADIENTE = {  # p5, p25, mediana, p75, p95 de la elevación por cobertura (m)
    4: (2051, 2143, 2232, 2304, 2375), 3: (2052, 2261, 2403, 2547, 2694), 6: (2066, 2319, 2428, 2545, 2715),
    5: (2588, 2732, 2796, 2858, 2929), 2: (2595, 2794, 2936, 3042, 3227), 1: (2838, 3034, 3146, 3236, 3362),
    7: (3218, 3300, 3363, 3409, 3520),
}


def fig_gradiente():
    fig = go.Figure()
    for c, (p5, p25, p50, p75, p95) in GRADIENTE.items():
        h = (f"<b>{LARGO[c]}</b><br>Mediana: {miles(p50)} m<br>50 % central: {miles(p25)}–{miles(p75)} m"
             f"<br>90 % central: {miles(p5)}–{miles(p95)} m<extra></extra>")
        y = [ETIQ[c]] * 2
        fig.add_scatter(x=[p5, p95], y=y, mode="lines", line=dict(color=COLOR[c], width=3), hovertemplate=h, showlegend=False)
        fig.add_scatter(x=[p25, p75], y=y, mode="lines", line=dict(color=COLOR[c], width=16), hovertemplate=h, showlegend=False)
        fig.add_scatter(x=[p50], y=[ETIQ[c]], mode="markers", hovertemplate=h, showlegend=False,
                        marker=dict(color="white", size=11, line=dict(color=COLOR[c], width=3)))
    for x0, x1, col, txt in [(1850, 2450, "rgba(224,165,38,0.18)", "Montano inferior"),
                             (2450, 2750, "rgba(76,154,42,0.16)", "Montano superior"),
                             (2750, 3250, "rgba(27,111,92,0.16)", "Subalpino"),
                             (3250, 3900, "rgba(142,94,162,0.18)", "Límite arbóreo")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=col, line_width=0, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=1, yref="y domain", text=f"<b>{txt}</b>", showarrow=False,
                           yanchor="top", font=dict(color="#4f6159"))
    fig.update_layout(height=400, margin=dict(t=40), xaxis=dict(title="Elevación (m s. n. m.)", range=[1850, 3900]),
                      yaxis=dict(showgrid=False))
    return fig


COMPOSICION = {  # % de celdas de cada área por cobertura
    1: [0, 34.546, 62.224, 40.536], 2: [8.185, 49.373, 30.066, 56.058], 3: [58.034, 5.644, 0, 0],
    4: [7.431, 0, 0, 0], 5: [0, 2.254, 0, 1.450], 6: [26.350, 3.010, 0, 0], 7: [0, 5.172, 7.710, 1.956],
}


def fig_composicion():
    ys = ["Cache la Poudre", "Comanche Peak", "Neota", "Rawah"]
    fig = go.Figure()
    for c in CLASES:
        fig.add_bar(x=COMPOSICION[c], y=ys, orientation="h", name=ETIQ[c],
                    marker=dict(color=COLOR[c], line=dict(color="white", width=1.5)),
                    hovertemplate=f"<b>%{{y}}</b><br>{LARGO[c]}: %{{x:.1f}} %<extra></extra>")
    fig.update_layout(barmode="stack", height=330, margin=dict(t=20), xaxis=dict(title="% de las celdas del área", range=[0, 100]),
                      legend=dict(orientation="h", y=-0.28, x=0))
    return fig


ESPECIALES = {(1, 2), (1, 7), (2, 5), (2, 6), (3, 4), (3, 6)}


def _tasas(cm):
    return cm / cm.sum(axis=1, keepdims=True)


def fig_solapamiento():
    r = _tasas(D.CM_LR)
    xs, ys, cd, col, txt, tam, pos = [], [], [], [], [], [], []
    maxc = max(D.CM_LR[a - 1, b - 1] + D.CM_LR[b - 1, a - 1] for a, b in D.SOLAPE)
    for (a, b), s in D.SOLAPE.items():
        y = r[a - 1, b - 1] + r[b - 1, a - 1]
        n = int(D.CM_LR[a - 1, b - 1] + D.CM_LR[b - 1, a - 1])
        par = f"{CORTO[a]} ↔ {CORTO[b]}"
        xs.append(s), ys.append(y), cd.append([a, b, n, par])
        destacado = y > 0.2
        col.append("#d9643a" if destacado else "#26a6e8")
        txt.append(par if destacado else "")
        tam.append(10 + 30 * np.sqrt(n / maxc))
        pos.append("top center" if (a, b) == (1, 2) else "middle left" if (a, b) == (1, 7) else "middle right")
    rcorr = np.corrcoef(xs, ys)[0, 1]
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=txt, textposition=pos, textfont=dict(size=10), customdata=cd,
        marker=dict(size=tam, color=col, opacity=0.85, line=dict(color="white", width=2)),
        hovertemplate="<b>%{customdata[3]}</b><br>Solapamiento altitudinal: %{x:.2f}<br>Confusión bidireccional: %{y:.2f}"
                      "<br>Celdas confundidas: %{customdata[2]:,}<extra></extra>"))
    fig.update_layout(height=470, title=f"r = {fmt(rcorr, 2)} entre nicho compartido y confusión · clic en un punto 👆",
                      xaxis=dict(title="Solapamiento de elevación (0 = nada, 1 = idéntico)", range=[-0.32, 1.42],
                                 tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1]),
                      yaxis=dict(title="Confusión bidireccional (a→b + b→a)"))
    return fig


def fig_halving():
    fig = go.Figure()
    m = D.MULTI.copy()
    for cand, g in m.groupby("candidato"):
        conf = g["config"].iloc[0].replace("'", "").replace("max_depth", "prof.").replace("min_samples_leaf", "hoja")
        ganador = len(g) == m.groupby("candidato").size().max()
        fig.add_scatter(x=g["n_estimators"], y=g["f1_macro"], mode="lines+markers", name=conf.strip("{}"),
                        line=dict(width=4 if ganador else 1.5, color="#0f9d58" if ganador else "#b0bec5"),
                        marker=dict(size=10 if ganador else 7),
                        hovertemplate=f"<b>{conf}</b><br>%{{x}} árboles<br>F1 macro: %{{y:.3f}}<extra></extra>")
    fig.update_layout(height=380, title="Cada ronda conserva la mitad mejor y duplica el presupuesto",
                      xaxis=dict(title="Presupuesto (número de árboles, escala log)", type="log", tickvals=[20, 40, 80, 160]),
                      yaxis=dict(title="F1 macro (validación)"), legend=dict(font=dict(size=10)))
    return fig


def fig_confusion(cm, modo, titulo_extra=""):
    etiquetas = [CORTO[c] for c in CLASES]
    if modo == "conteo":
        z = np.log10(cm + 1)
        texto = [[miles(v) for v in fila] for fila in cm]
        custom, hover, escala, titulo = cm, "%{customdata:,} celdas", dict(title="celdas"), "Conteos absolutos"
        mostrar = False
    else:
        z = cm / cm.sum(axis=1, keepdims=True) if modo == "real" else cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
        texto = [[f"{v:.2f}" for v in fila] for fila in z]
        custom, hover, escala, mostrar = z, "%{customdata:.1%}", dict(title="proporción"), True
        titulo = "Normalizada por fila (diagonal = recall)" if modo == "real" else "Normalizada por columna (diagonal = precisión)"
    fig = go.Figure(go.Heatmap(
        z=z, x=etiquetas, y=etiquetas, text=texto, texttemplate="%{text}", textfont=dict(size=12), customdata=custom,
        xgap=3, ygap=3, zmin=0, zmax=1 if modo != "conteo" else None, showscale=mostrar, colorbar=escala,
        colorscale=[[0, "#f1fbf3"], [0.15, "#b9e8c6"], [0.4, "#4fc38a"], [0.7, "#0f9d58"], [1, "#0b4f34"]],
        hovertemplate="Real: <b>%{y}</b><br>Predicha: <b>%{x}</b><br>" + hover + "<extra></extra>"))
    fig.update_layout(height=470, title=titulo + titulo_extra, yaxis=dict(title="Cobertura real", autorange="reversed", showgrid=False),
                      xaxis=dict(title="Cobertura predicha", tickangle=30, showgrid=False))
    return fig


def fig_areas_no_vistas(metrica, solo_lr=False):
    df = D.AREAS_NV.copy()
    fig = go.Figure()
    for modelo, color, nombre in [("Regresión logística", "#d9643a", "Regresión logística"),
                                  ("XGBoost", "#0f9d58", "XGBoost")]:
        if solo_lr and modelo != "Regresión logística":
            continue
        g = df[df["modelo"] == modelo]
        fig.add_bar(x=g["área_no_vista"], y=g[metrica], name=nombre, marker_color=color,
                    text=[fmt(v) for v in g[metrica]], textposition="outside",
                    customdata=g[["celdas"]].to_numpy(),
                    hovertemplate=f"<b>{nombre}</b><br>Área no vista: %{{x}}<br>%{{y:.3f}}<br>%{{customdata[0]:,}} celdas de prueba<extra></extra>")
    referencia = LR_ACC if metrica == "accuracy" else LR_F1
    fig.add_hline(y=referencia, line_dash="dash", line_color="#4f6159",
                  annotation_text=f"Regresión logística con partición aleatoria ({fmt(referencia)})", annotation_position="top left")
    fig.update_layout(barmode="group", height=400, yaxis=dict(range=[0, 1], title="Accuracy" if metrica == "accuracy" else "F1 macro"),
                      xaxis=dict(title="Área silvestre dejada fuera del entrenamiento", showgrid=False),
                      legend=dict(orientation="h", y=-0.22, x=0), title="Entrenar con 3 áreas, evaluar en la cuarta")
    return fig

