"""Seccion 2: Analisis exploratorio con filtros globales.

Los filtros (areas silvestres, coberturas y rango de elevacion) actualizan todos los
graficos de la pagina. Trabaja sobre la muestra estratificada de 40.000 celdas que se
carga una sola vez (src/data.py). No hay filtro de fechas: Covertype es una instantanea
unica sin dimension temporal.
"""
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from src import data as D
from src.data import CLASES, COLOR, ETIQ, LIBRO, fmt, miles, vacio
from src.plots import fig_areas_no_vistas
from src.ui import control, grafica, nota, radio, tarjeta

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Covertype")

NUMERICAS = list(D.VARIABLES)
CORTA = {"Elevation": "Elevación", "Aspect": "Orientación", "Slope": "Pendiente",
         "Horizontal_Distance_To_Hydrology": "D. horiz. agua", "Vertical_Distance_To_Hydrology": "D. vert. agua",
         "Horizontal_Distance_To_Roadways": "D. carreteras", "Hillshade_9am": "Sombra 9 a. m.",
         "Hillshade_Noon": "Sombra mediodía", "Hillshade_3pm": "Sombra 3 p. m.",
         "Horizontal_Distance_To_Fire_Points": "D. incendios"}
ELEV = (int(D.EDA["Elevation"].min()), int(D.EDA["Elevation"].max()))
ENTRADAS = [Input("f-areas", "value"), Input("f-clases", "value"), Input("f-elevacion", "value")]


@lru_cache(maxsize=64)
def filtrar(areas, clases, elev):
    d = D.EDA
    return d[d["Area"].isin(areas) & d["Clase"].isin(clases) & d["Elevation"].between(*elev)]


def _f(areas, clases, elev):
    return filtrar(tuple(areas or ()), tuple(clases or ()), tuple(elev or ELEV))


def filtros():
    return html.Div(dbc.Row([
        dbc.Col(control("🏞️ Áreas silvestres", dbc.Checklist(
            id="f-areas", inline=True, value=D.AREAS, options=[{"label": f" {a}", "value": a} for a in D.AREAS])), lg=4),
        dbc.Col(control("🌲 Coberturas", dcc.Dropdown(
            id="f-clases", multi=True, value=CLASES, options=[{"label": ETIQ[c], "value": c} for c in CLASES])), lg=4),
        dbc.Col(control("🏔️ Elevación (m)", dcc.RangeSlider(
            id="f-elevacion", min=ELEV[0], max=ELEV[1], step=50, value=list(ELEV), allowCross=False,
            marks={v: f"{v}" for v in range(2000, 3900, 400)}, tooltip={"placement": "bottom"})), lg=4),
        dbc.Col(html.Div(id="f-resumen", className="etiqueta-control mt-1"), lg=12),
    ]), className="barra-filtros mb-3")


def layout():
    return html.Div([
        html.H2("📊 Análisis exploratorio", className="seccion-titulo"),
        html.P("Los filtros de esta barra actualizan todos los gráficos de la página.", className="text-muted"),
        filtros(),
        dbc.Row([
            dbc.Col(tarjeta("Distribución de la variable objetivo", "mdi:scale-unbalanced", "cab-ocre", [
                radio("escala-desbalance", [("log", "Escala logarítmica"), ("lineal", "Escala lineal")], "log"),
                grafica("fig-desbalance"), html.Div(id="nota-desbalance", className="nota"),
            ]), lg=6, className="mb-3"),
            dbc.Col(tarjeta("Poder discriminante de cada variable (η²)", "mdi:podium-gold", "cab-verde", [
                grafica("fig-eta2"), html.Div(id="nota-eta2", className="nota"),
            ]), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Distribución de cada variable por cobertura", "mdi:chart-bell-curve", "cab-lavanda", [
                control("🔎 Variable", dcc.Dropdown(id="variable-distribucion", clearable=False, value="Elevation",
                                                    options=[{"label": l, "value": v} for v, l in D.VARIABLES.items()])),
                grafica("fig-distribucion"), html.Div(id="nota-distribucion", className="nota"),
            ]), lg=6, className="mb-3"),
            dbc.Col(tarjeta("Correlación entre variables continuas", "mdi:grid", "cab-azul", [
                grafica("fig-correlacion"), html.Div(id="nota-correlacion", className="nota"),
            ]), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Explorador bivariado", "mdi:chart-scatter-plot", "cab-lago", [
                dbc.Row([
                    dbc.Col(control("Eje x", dcc.Dropdown(id="biv-x", clearable=False, value="Elevation",
                                                          options=[{"label": l, "value": v} for v, l in D.VARIABLES.items()])), md=4),
                    dbc.Col(control("Eje y", dcc.Dropdown(id="biv-y", clearable=False, value="Horizontal_Distance_To_Roadways",
                                                          options=[{"label": l, "value": v} for v, l in D.VARIABLES.items()])), md=4),
                    dbc.Col(control("Color", dcc.Dropdown(id="biv-color", clearable=False, value="Clase",
                                                          options=[{"label": "Cobertura", "value": "Clase"},
                                                                   {"label": "Área silvestre", "value": "Area"}])), md=4),
                ], className="mb-2"),
                grafica("fig-bivariado"), html.Div(id="nota-bivariado", className="nota"),
            ]), lg=7, className="mb-3"),
            dbc.Col(tarjeta("Valores faltantes y atípicos", "mdi:alert-octagon-outline", "cab-terracota", [
                grafica("fig-atipicos"), html.Div(id="nota-atipicos", className="nota"),
            ]), lg=5, className="mb-3"),
        ]),
        tarjeta("¿Generaliza un modelo a un área que nunca vio?", "mdi:map-search", "cab-terracota", [
            radio("metrica-areas-eda", [("accuracy", "Accuracy"), ("f1_macro_clases_presentes", "F1 macro (clases presentes)")],
                  "accuracy"),
            grafica("fig-areas-eda"),
            nota("Este experimento usa todos los datos (no depende de los filtros): se entrena con tres áreas y se evalúa en la "
                 "cuarta. La accuracy cae de 0,72 (partición aleatoria) a 0,44–0,62: las celdas vecinas se parecen entre sí "
                 "(autocorrelación espacial) y la partición aleatoria es optimista.", "02b_fuga_estructura.html", "2.7"),
        ], "mb-3"),
    ], className="contenido-tab")


@callback(Output("f-resumen", "children"), *ENTRADAS)
def cb_resumen(areas, clases, elev):
    d = _f(areas, clases, elev)
    return f"{miles(len(d))} de {miles(len(D.EDA))} celdas de la muestra estratificada cumplen los filtros."


@callback(Output("fig-desbalance", "figure"), Output("nota-desbalance", "children"),
          Input("escala-desbalance", "value"), *ENTRADAS)
def cb_desbalance(escala, areas, clases, elev):
    d = _f(areas, clases, elev)
    if d.empty:
        return vacio("Ningún dato cumple los filtros"), ""
    conteo = d["Clase"].value_counts().reindex(CLASES).dropna().sort_values()
    total = conteo.sum()
    fig = go.Figure(go.Bar(
        x=conteo.values, y=[ETIQ[c] for c in conteo.index], orientation="h",
        text=[f"{miles(v)} ({fmt(100 * v / total, 1)} %)" for v in conteo.values], textposition="outside",
        marker=dict(color=[COLOR[c] for c in conteo.index], line=dict(color="white", width=2)),
        hovertemplate="<b>%{y}</b><br>%{x:,} celdas<extra></extra>"))
    eje = dict(title="Celdas en la muestra (escala log)", type="log") if escala == "log" else \
        dict(title="Celdas en la muestra", range=[0, conteo.max() * 1.35])
    razon = conteo.max() / max(conteo.min(), 1)
    fig.update_layout(height=360, title=f"Razón de desbalance {razon:.0f} : 1", xaxis=eje, yaxis=dict(showgrid=False))
    texto = [html.B("💡 Interpretación: "), f"{ETIQ[conteo.index[-1]]} concentra el {fmt(100 * conteo.iloc[-1] / total, 1)} % "
             f"de las celdas filtradas y {ETIQ[conteo.index[0]]} solo el {fmt(100 * conteo.iloc[0] / total, 2)} %. "
             "Con este desbalance la exactitud premia a las clases grandes; por eso se optimiza el F1 macro."]
    return fig, texto


@callback(Output("fig-eta2", "figure"), Output("nota-eta2", "children"), *ENTRADAS)
def cb_eta2(areas, clases, elev):
    d = _f(areas, clases, elev)
    if d["Clase"].nunique() < 2:
        return vacio("Se necesitan al menos dos coberturas"), ""
    eta = {}
    for v in NUMERICAS:
        media = d[v].mean()
        sct = ((d[v] - media) ** 2).sum()
        sce = d.groupby("Clase")[v].apply(lambda s: len(s) * (s.mean() - media) ** 2).sum()
        eta[v] = sce / sct if sct > 0 else 0.0
    orden = sorted(eta, key=eta.get)
    fig = go.Figure(go.Bar(x=[eta[v] for v in orden], y=[CORTA[v] for v in orden], orientation="h",
                           text=[fmt(eta[v], 2) for v in orden], textposition="outside",
                           marker_color=["#0f9d58" if eta[v] > 0.3 else "#9ccc65" if eta[v] > 0.05 else "#cfd8dc" for v in orden],
                           hovertemplate="<b>%{y}</b><br>η² = %{x:.3f}<extra></extra>"))
    fig.update_layout(height=360, title="η² = varianza explicada por la cobertura",
                      xaxis=dict(title="η² (0 = no separa, 1 = separa por completo)", range=[0, 1.05]),
                      yaxis=dict(showgrid=False))
    top = orden[-1]
    return fig, [html.B("💡 Interpretación: "), f"{CORTA[top]} es la variable que más separa las coberturas en la "
                 f"selección (η² = {fmt(eta[top], 2)}). ", html.A("Sección 2.3 del Jupyter Book →",
                                                                  href=LIBRO + "02_eda.html", target="_blank")]


@callback(Output("fig-distribucion", "figure"), Output("nota-distribucion", "children"),
          Input("variable-distribucion", "value"), *ENTRADAS)
def cb_distribucion(variable, areas, clases, elev):
    d = _f(areas, clases, elev)
    if d.empty:
        return vacio("Ningún dato cumple los filtros", 420), ""
    etiqueta = D.VARIABLES[variable]
    orden = d.groupby("Clase")[variable].median().sort_values().index
    fig = go.Figure()
    for c in orden:
        fig.add_violin(y=d.loc[d["Clase"] == c, variable], name=ETIQ[c], fillcolor=COLOR[c], line=dict(color=COLOR[c]),
                       opacity=0.78, box=dict(visible=True, width=0.22, fillcolor="white"), points=False,
                       showlegend=False, spanmode="hard", scalemode="width", width=0.6)
    fig.update_layout(height=420, title=f"{etiqueta} por cobertura", xaxis=dict(showgrid=False),
                      yaxis=dict(title=etiqueta))
    med = d.groupby("Clase")[variable].median()
    return fig, [html.B("💡 Interpretación: "), f"La mediana de {etiqueta.lower()} va de {fmt(med.min(), 0)} "
                 f"({ETIQ[med.idxmin()]}) a {fmt(med.max(), 0)} ({ETIQ[med.idxmax()]}); cuanto menos se solapan las "
                 "cajas, más útil es la variable para separar esas coberturas."]


@callback(Output("fig-correlacion", "figure"), Output("nota-correlacion", "children"), *ENTRADAS)
def cb_correlacion(areas, clases, elev):
    d = _f(areas, clases, elev)
    if len(d) < 10:
        return vacio("Muy pocas celdas para correlacionar", 440), ""
    r = d[NUMERICAS].corr().to_numpy()
    n = len(NUMERICAS)
    z = [[r[i, j] if j <= i else None for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=z, x=[CORTA[v] for v in NUMERICAS], y=[CORTA[v] for v in NUMERICAS], zmin=-1, zmax=1,
        text=[[f"{r[i, j]:.2f}" if j <= i else "" for j in range(n)] for i in range(n)], texttemplate="%{text}",
        textfont=dict(size=10), xgap=2, ygap=2, colorbar=dict(title="r"),
        colorscale=[[0, "#b5432a"], [0.5, "#fbf7ef"], [1, "#0d47a1"]],
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(height=440, title="Correlación de Pearson (triángulo inferior)",
                      yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(tickangle=35, showgrid=False))
    abs_r = np.abs(np.tril(r, -1))
    i, j = np.unravel_index(abs_r.argmax(), abs_r.shape)
    return fig, [html.B("💡 Interpretación: "), f"El par más correlacionado es {CORTA[NUMERICAS[i]]} – "
                 f"{CORTA[NUMERICAS[j]]} (r = {fmt(r[i, j], 2)}). Los índices de sombreado se calculan con la orientación "
                 "y la pendiente, por eso son redundantes entre sí."]


@callback(Output("fig-bivariado", "figure"), Output("nota-bivariado", "children"),
          Input("biv-x", "value"), Input("biv-y", "value"), Input("biv-color", "value"), *ENTRADAS)
def cb_bivariado(x, y, color, areas, clases, elev):
    d = _f(areas, clases, elev)
    if d.empty:
        return vacio("Ningún dato cumple los filtros", 440), ""
    m = d.sample(min(len(d), 4000), random_state=42)
    fig = go.Figure()
    grupos = sorted(m[color].unique())
    for g in grupos:
        s = m[m[color] == g]
        nombre = ETIQ[g] if color == "Clase" else g
        col = COLOR[g] if color == "Clase" else D.COLOR_AREA[g]
        fig.add_scattergl(x=s[x], y=s[y], mode="markers", name=nombre, marker=dict(size=5, color=col, opacity=0.55),
                          hovertemplate=f"<b>{nombre}</b><br>{D.VARIABLES[x]}: %{{x}}<br>{D.VARIABLES[y]}: %{{y}}<extra></extra>")
    fig.update_layout(height=440, title=f"{miles(len(m))} celdas (submuestra para el gráfico)",
                      xaxis=dict(title=D.VARIABLES[x]), yaxis=dict(title=D.VARIABLES[y]),
                      legend=dict(orientation="h", y=-0.2, font=dict(size=11)))
    rho = d[[x, y]].corr(method="spearman").iloc[0, 1]
    return fig, [html.B("💡 Interpretación: "), f"Correlación de Spearman entre las dos variables: {fmt(rho, 2)}. "
                 "Si los colores forman franjas separadas, la combinación de ambas variables ayuda a distinguir los grupos."]


@callback(Output("fig-atipicos", "figure"), Output("nota-atipicos", "children"), *ENTRADAS)
def cb_atipicos(areas, clases, elev):
    d = _f(areas, clases, elev)
    if d.empty:
        return vacio("Ningún dato cumple los filtros", 440), ""
    q1, q3 = d[NUMERICAS].quantile(0.25), d[NUMERICAS].quantile(0.75)
    iqr = q3 - q1
    pct = (((d[NUMERICAS] < q1 - 1.5 * iqr) | (d[NUMERICAS] > q3 + 1.5 * iqr)).mean() * 100).sort_values()
    faltantes = d[NUMERICAS].isna().mean() * 100
    fig = go.Figure()
    fig.add_bar(x=pct.values, y=[CORTA[v] for v in pct.index], orientation="h", name="Atípicos (regla 1,5 × IQR)",
                marker_color="#d9643a", text=[f"{fmt(v, 1)} %" for v in pct.values], textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.2f} % de celdas atípicas<extra></extra>")
    fig.add_bar(x=faltantes[pct.index].values, y=[CORTA[v] for v in pct.index], orientation="h", name="Faltantes",
                marker_color="#5b6fb3")
    fig.update_layout(height=440, barmode="group", title="Porcentaje de celdas por variable",
                      xaxis=dict(title="% de celdas", range=[0, max(pct.max() * 1.3, 1)]), yaxis=dict(showgrid=False),
                      legend=dict(orientation="h", y=-0.2))
    top = pct.index[-1]
    return fig, [html.B("💡 Interpretación: "), "No hay valores faltantes en ninguna variable (barras azules en cero). "
                 f"{CORTA[top]} tiene la mayor proporción de atípicos ({fmt(pct[top], 1)} %); son valores reales del "
                 "terreno (por ejemplo, celdas muy lejanas al agua), no errores, y por eso se conservan."]


@callback(Output("fig-areas-eda", "figure"), Input("metrica-areas-eda", "value"))
def cb_areas_eda(metrica):
    return fig_areas_no_vistas(metrica, solo_lr=False)
