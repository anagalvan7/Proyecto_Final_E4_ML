"""Seccion 3: Modelos. Tabla de lideres, prediccion en vivo con explicacion local,
diagnostico del modelo elegido, comparacion estadistica y el modelo original (ECLC)."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

from src import data as D
from src import modelos as MOD
from src import seccion_e2 as E2
from src.data import CLASES, COLOR, CORTO, ETIQ, LIBRO, fmt, vacio
from src.preprocesamiento import NUM_COLS
from src.ui import control, grafica, intro, kpi, nota, radio, tarjeta

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Covertype")

NOMBRES = {"KNN": "K vecinos", "NB": "Naive Bayes", "LR": "Regresión logística", "DT": "Árbol de decisión",
           "RF": "Random Forest", "XGB": "XGBoost", "SVM": "SVM (RBF)", "ECLC": "ECLC (original)",
           "HGB": "HistGradientBoosting"}
CLAVE_EST = {**{m: f"base:{m}" for m in MOD.BASES}, "ECLC": "ECLC", "HGB": "HGB (estado del arte)"}
ORDEN = ["ECLC"] + MOD.BASES
COLOR_M = {**D.COLOR_MODELO, "ECLC": "#b0413e", "HGB": "#455a64"}
GRUPO_ETIQ = {**{c: D.VARIABLES[c] for c in NUM_COLS}, "Wilderness_Area": "Área silvestre", "Soil_Type": "Tipo de suelo"}
EJEMPLOS = pd.read_csv(D.E3 / "ejemplos.csv")
V = D.VEREDICTO


# ======================================================================== tabla de lideres
def tabla_lideres():
    est = D.EST_RESUMEN.set_index("metodo")
    est_c = D.EST_RESUMEN_COMPLETO.set_index("metodo")
    filas = []
    for m in ORDEN + ["HGB"]:
        h = D.HOLDOUT[m]
        e = est.loc[CLAVE_EST[m]]
        ec = est_c.loc[CLAVE_EST[m]]
        filas.append({
            "Modelo": NOMBRES[m], "Balanceo": D.BALANCEOS.get(h.get("balanceo"), h.get("balanceo") or "—"),
            "Optimizador": D.OPTIMIZADORES.get(h.get("optimizacion"), "Random Search" if m == "HGB" else "—"),
            "F1 anidada 3×3": f"{fmt(h['f1_cv_media'])} ± {fmt(h['f1_cv_sd'])}"
            if h.get("f1_cv_media") is not None else "—",
            "F1 30 pliegues": f"{fmt(e['f1_media'])} ± {fmt(e['f1_sd'])}",
            "F1 prueba reservada (40.000)": round(h["prueba_reservada"]["f1_macro"], 4),
            "F1 completo 10 pliegues": f"{fmt(ec['f1_media'])} ± {fmt(ec['f1_sd'])}",
            "f1_c": round(float(ec["f1_media"]), 4),
            "F1 completo prueba reservada": round(float(ec["f1_prueba_reservada"]), 4),
            "Inferencia (ms / 1.000 celdas)": round(1000 * h["tiempo_inferencia_s_por_1000"], 1)
            if h.get("tiempo_inferencia_s_por_1000") else None,
        })
    df = pd.DataFrame(filas).sort_values("f1_c", ascending=False).drop(columns="f1_c")
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
        sort_action="native", filter_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Nunito, Segoe UI, sans-serif", "fontSize": 14, "padding": "6px", "textAlign": "left"},
        style_header={"backgroundColor": "#1b6f5c", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[{"if": {"filter_query": '{Modelo} contains "ECLC"'}, "backgroundColor": "#fdecea",
                                 "fontWeight": "bold"}],
    )


def sub_lideres():
    return html.Div([
        intro(html.B("Mejor combinación de cada modelo. "), "Para cada uno de los 7 algoritmos se eligió la combinación de "
              "balanceo y optimizador con mayor F1 macro medio en el bucle externo de la validación anidada (desempate: menor "
              "desviación estándar). Se agregan el modelo original (ECLC) y HistGradientBoosting como referencia del estado "
              "del arte. Ordena o filtra cualquier columna."),
        tarjeta("Tabla de líderes", "mdi:trophy", "cab-ocre", [
            tabla_lideres(),
            nota("«F1 anidada 3×3» es la estimación del benchmark corregido (3 pliegues externos). «F1 30 pliegues» repite la "
                 "evaluación en 10 particiones de 3 pliegues, las mismas para todos los modelos, y alimenta las pruebas de "
                 "hipótesis. Las columnas «completo» repiten la evaluación entrenando con las 464.809 celdas (5 × 2 pliegues y "
                 "prueba reservada); la tabla se ordena por ellas. La prueba reservada (116.203 celdas) se usó una sola vez. "
                 "La predicción en vivo usa los modelos de la submuestra, que caben en la memoria del servidor.", "06_benchmark_modelos.html", "5.4"),
        ], "mb-3"),
    ])


# ======================================================================== prediccion en vivo
def formulario():
    rangos = {e["nombre"]: e for e in MOD.esquema()}
    campos = []
    for c in NUM_COLS:
        e = rangos[c]
        medio = int(EJEMPLOS[c].median())
        campos.append(dbc.Col(control(f"{D.VARIABLES[c]} [{e['min']}, {e['max']}]", dbc.Input(
            id={"tipo": "entrada", "var": c}, type="number", min=e["min"], max=e["max"], step=1, value=medio,
            debounce=True)), md=6, lg=4, className="mb-2"))
    campos.append(dbc.Col(control("Área silvestre", dcc.Dropdown(
        id="entrada-area", clearable=False, value=1, options=[{"label": a, "value": i} for i, a in enumerate(D.AREAS, 1)])),
        md=6, lg=4, className="mb-2"))
    campos.append(dbc.Col(control("Tipo de suelo (1–40)", dcc.Dropdown(
        id="entrada-suelo", clearable=False, value=29, options=[{"label": f"Suelo {i}", "value": i} for i in range(1, 41)])),
        md=6, lg=4, className="mb-2"))
    return dbc.Row(campos)


def sub_prediccion():
    return html.Div([
        intro(html.B("Predicción en vivo. "), "Describe una celda del terreno (o carga una celda real de la prueba reservada), "
              "elige un modelo y presiona «Predecir». Los modelos ya están entrenados: el dashboard solo los carga."),
        dbc.Row([
            dbc.Col(tarjeta("Datos de la celda", "mdi:form-select", "cab-lago", [
                dbc.Row([
                    dbc.Col(control("📂 Cargar una celda de ejemplo", dcc.Dropdown(
                        id="ejemplo", placeholder="Elige una celda real…",
                        options=[{"label": f"Celda {i + 1} · cobertura real: {CORTO[int(r.Cover_Type)]}", "value": i}
                                 for i, r in EJEMPLOS.iterrows()])), md=7),
                    dbc.Col(control("🤖 Modelo", dcc.Dropdown(
                        id="modelo-pred", clearable=False, value="ECLC",
                        options=[{"label": NOMBRES[m], "value": m} for m in ORDEN])), md=5),
                ], className="mb-3"),
                formulario(),
                dbc.Button("🔮 Predecir", id="boton-predecir", color="success", className="mt-2"),
                html.Div(id="alerta-pred", className="mt-2"),
            ]), lg=6, className="mb-3"),
            dbc.Col(tarjeta("Resultado y explicación local", "mdi:lightbulb-on", "cab-verde", [
                dcc.Loading(html.Div(id="salida-pred", children=html.P(
                    "Completa los datos y presiona «Predecir».", className="text-muted"))),
            ]), lg=6, className="mb-3"),
        ]),
    ])


@callback(*[Output({"tipo": "entrada", "var": c}, "value") for c in NUM_COLS],
          Output("entrada-area", "value"), Output("entrada-suelo", "value"),
          Input("ejemplo", "value"), prevent_initial_call=True)
def cb_ejemplo(i):
    if i is None:
        return [no_update] * (len(NUM_COLS) + 2)
    fila = EJEMPLOS.iloc[i]
    area = int(np.argmax([fila[f"Wilderness_Area_{a}"] for a in range(1, 5)]) + 1)
    suelo = int(np.argmax([fila[f"Soil_Type_{s}"] for s in range(1, 41)]) + 1)
    return [int(fila[c]) for c in NUM_COLS] + [area, suelo]


@callback(Output("salida-pred", "children"), Output("alerta-pred", "children"),
          Input("boton-predecir", "n_clicks"),
          State("modelo-pred", "value"), State("entrada-area", "value"), State("entrada-suelo", "value"),
          *[State({"tipo": "entrada", "var": c}, "value") for c in NUM_COLS], prevent_initial_call=True)
def cb_predecir(_, modelo, area, suelo, *valores):
    datos = dict(zip(NUM_COLS, valores))
    errores = MOD.validar(datos)
    if errores:
        return no_update, dbc.Alert([html.B("Revisa los datos: "), html.Ul([html.Li(e) for e in errores])], color="danger")
    try:
        X = MOD.fila_desde_formulario(datos, area, suelo)
        p = MOD.predecir_proba(modelo, X)[0]
        c = int(np.argmax(p)) + 1
        expl, base, fx = MOD.explicar(modelo, X, D.FONDO_SHAP, c)
    except Exception as e:  # noqa: BLE001 - una entrada rara no debe tumbar la aplicacion
        return no_update, dbc.Alert(f"No fue posible predecir: {e}", color="danger")
    fig_p = go.Figure(go.Bar(x=p, y=[ETIQ[k] for k in CLASES], orientation="h", marker_color=[COLOR[k] for k in CLASES],
                             text=[f"{100 * v:.1f} %" for v in p], textposition="outside",
                             hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.3f}<extra></extra>"))
    fig_p.update_layout(height=300, title="Probabilidad de cada cobertura", xaxis=dict(range=[0, 1.15], title="Probabilidad"),
                        yaxis=dict(autorange="reversed", showgrid=False), margin=dict(t=40))
    e = expl.assign(etq=expl["variable"].map(GRUPO_ETIQ)).sort_values("shap", key=np.abs).tail(8)
    fig_s = go.Figure(go.Bar(x=e["shap"], y=e["etq"], orientation="h",
                             marker_color=["#0f9d58" if v > 0 else "#d9643a" for v in e["shap"]],
                             text=[f"{v:+.3f}" for v in e["shap"]], textposition="outside",
                             hovertemplate="<b>%{y}</b><br>Aporte: %{x:+.3f}<extra></extra>", cliponaxis=False))
    lo, hi = min(float(e["shap"].min()), 0.0), max(float(e["shap"].max()), 0.0)
    margen = 0.22 * (hi - lo or 1.0)                     # espacio para que las etiquetas no se corten
    fig_s.update_layout(height=320, title=f"¿Por qué {CORTO[c]}? (SHAP local, P = {fx:.2f}; base = {base:.2f})",
                        xaxis=dict(title="Aporte a la probabilidad (verde sube, terracota baja)",
                                   range=[lo - (margen if lo < 0 else 0), hi + margen]),
                        yaxis=dict(showgrid=False), margin=dict(t=40))
    hijos = [html.H4([f"{ETIQ[c]}", dbc.Badge(f"{100 * p[c - 1]:.1f} %", color="success", className="ms-2")]),
             grafica(figura=fig_p), grafica(figura=fig_s),
             html.Div([html.B("💡 Cómo leerlo: "), "cada barra es la contribución de una variable (o grupo de indicadoras) "
                       "a la probabilidad de la cobertura predicha, frente a un fondo de 10 celdas típicas. Las contribuciones "
                       "suman la diferencia entre la probabilidad predicha y la base (KernelSHAP agrupado)."], className="nota")]
    if modelo == "ECLC":
        W = MOD.pesos_eclc(X)[0][:, c - 1]
        bases = MOD.metadata("ECLC")["modelos_base"]
        hijos.append(html.Div([html.B("⚖️ Peso de cada modelo para esta clase en esta celda: "),
                               ", ".join(f"{NOMBRES[m]} {100 * w:.0f} %" for m, w in sorted(zip(bases, W), key=lambda t: -t[1])
                                         if w >= 0.01)], className="nota"))
    return hijos, None


# ======================================================================== diagnostico
def sub_diagnostico():
    return html.Div([
        intro(html.B("Diagnóstico en la prueba reservada. "), "Elige un modelo para ver su matriz de confusión, sus curvas ROC "
              "y de precisión-recall por clase, su calibración y su interpretabilidad global (SHAP)."),
        dbc.Row([
            dbc.Col(control("🤖 Modelo", dcc.Dropdown(id="modelo-diag", clearable=False, value="ECLC",
                                                     options=[{"label": NOMBRES[m], "value": m} for m in ORDEN])), md=5),
            dbc.Col(control("🗂️ Modelos entrenados con", radio("entrenamiento-diag", [
                ("completo", "el conjunto completo (464.809 celdas)"),
                ("submuestra", "la submuestra de 40.000 celdas (los de la predicción en vivo)")], "completo")), md=7),
        ], className="mb-2"),
        html.Div(id="kpis-diag", className="mb-2"),
        dbc.Row([
            dbc.Col(tarjeta("Matriz de confusión (recall por fila)", "mdi:grid", "cab-azul", [grafica("fig-cm-diag"),
                                                                                         html.Div(id="nota-cm-diag", className="nota")]), lg=6, className="mb-3"),
            dbc.Col(tarjeta("Curvas ROC por clase (uno contra el resto)", "mdi:chart-line", "cab-verde", [
                grafica("fig-roc-diag"),
                nota("Cuanto más se acerca cada curva a la esquina superior izquierda, mejor separa esa cobertura del resto. "
                     "Con 7 clases desbalanceadas, el AUC alto puede convivir con un recall bajo en las clases raras.")]),
                    lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Precisión-recall por clase", "mdi:chart-bell-curve-cumulative", "cab-lavanda", [
                grafica("fig-pr-diag"),
                nota("Más informativa que la ROC para las clases raras: muestra cuánta precisión se pierde al exigir más recall.")]),
                    lg=6, className="mb-3"),
            dbc.Col(tarjeta("Calibración (clase predicha)", "mdi:target-variant", "cab-ocre", [
                grafica("fig-cal-diag"), html.Div(id="nota-cal-diag", className="nota")]), lg=6, className="mb-3"),
        ]),
        tarjeta("Interpretabilidad global: SHAP agrupado sobre 60 celdas", "mdi:magnify-scan", "cab-terracota", [
            grafica("fig-shap-diag"), html.Div(id="nota-shap-diag", className="nota")], "mb-3"),
    ])


@callback(Output("kpis-diag", "children"), Output("fig-cm-diag", "figure"), Output("nota-cm-diag", "children"),
          Output("fig-roc-diag", "figure"), Output("fig-pr-diag", "figure"), Output("fig-cal-diag", "figure"),
          Output("nota-cal-diag", "children"), Output("fig-shap-diag", "figure"), Output("nota-shap-diag", "children"),
          Input("modelo-diag", "value"), Input("entrenamiento-diag", "value"))
def cb_diagnostico(m, entrenamiento):
    completo = entrenamiento == "completo" and m in D.DIAG_COMPLETO
    if completo:
        d = h = D.DIAG_COMPLETO[m]
        detalle = "prueba reservada · entrenamiento completo"
    else:
        d, h = D.DIAG[m], D.HOLDOUT[m]["prueba_reservada"]
        detalle = "prueba reservada · submuestra de 40.000"
    kpis = dbc.Row([
        dbc.Col(kpi("F1 macro", fmt(h["f1_macro"]), detalle, 1), md=3, xs=6),
        dbc.Col(kpi("Accuracy", fmt(h["accuracy"]), detalle, 2), md=3, xs=6),
        dbc.Col(kpi("Brier", fmt(d["brier"]), "menor es mejor", 3), md=3, xs=6),
        dbc.Col(kpi("ECE", fmt(d["ece"]), "error de calibración", 4), md=3, xs=6),
    ], className="g-2")
    cm = np.array(d["confusion"], dtype=float)
    r = cm / cm.sum(1, keepdims=True)
    fig_cm = go.Figure(go.Heatmap(z=r, x=[CORTO[c] for c in CLASES], y=[CORTO[c] for c in CLASES], zmin=0, zmax=1,
                                  colorscale="Greens", text=[[f"{v:.2f}" for v in fila] for fila in r], texttemplate="%{text}",
                                  hovertemplate="Real <b>%{y}</b> → predicha <b>%{x}</b><br>%{z:.3f}<extra></extra>",
                                  colorbar=dict(title="Tasa")))
    fig_cm.update_layout(height=420, xaxis=dict(title="Predicha", tickangle=30), yaxis=dict(title="Real", autorange="reversed"))
    peor = int(np.argmin(np.diag(r))) + 1
    nota_cm = [html.B("💡 Interpretación: "), f"La clase con menor recall es {ETIQ[peor]} ({fmt(r[peor - 1, peor - 1], 2)}); "
               f"se confunde sobre todo con {ETIQ[int(np.argsort(r[peor - 1])[-2]) + 1]}."]
    fig_roc, fig_pr = go.Figure(), go.Figure()
    for c in CLASES:
        roc, pr = d["roc"][str(c)], d["pr"][str(c)]
        fig_roc.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"{CORTO[c]} (AUC {roc['auc']:.3f})",
                            line=dict(color=COLOR[c], width=2))
        fig_pr.add_scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=f"{CORTO[c]} (AP {pr['ap']:.3f})",
                           line=dict(color=COLOR[c], width=2))
    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#9e9e9e"), showlegend=False)
    fig_roc.update_layout(height=420, xaxis=dict(title="Tasa de falsos positivos"), yaxis=dict(title="Tasa de verdaderos positivos"),
                          legend=dict(font=dict(size=10)))
    fig_pr.update_layout(height=420, xaxis=dict(title="Recall"), yaxis=dict(title="Precisión", range=[0, 1.02]),
                         legend=dict(font=dict(size=10)))
    cal = pd.DataFrame(d["calibracion"])
    fig_cal = go.Figure()
    fig_cal.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#9e9e9e"), name="Calibración perfecta")
    fig_cal.add_scatter(x=cal["confianza"], y=cal["exactitud"], mode="lines+markers", name=NOMBRES[m],
                        marker=dict(size=8 + 20 * np.sqrt(cal["n"] / cal["n"].max()), color=COLOR_M[m]),
                        customdata=cal["n"], hovertemplate="Confianza %{x:.2f}<br>Exactitud %{y:.2f}<br>%{customdata:,} celdas<extra></extra>")
    fig_cal.update_layout(height=420, xaxis=dict(title="Probabilidad de la clase predicha", range=[0, 1]),
                          yaxis=dict(title="Fracción de aciertos", range=[0, 1]), legend=dict(orientation="h", y=-0.2))
    nota_cal = [html.B("💡 Interpretación: "), f"ECE = {fmt(d['ece'])}. Puntos bajo la diagonal indican exceso de confianza; "
                "sobre la diagonal, falta de confianza. El tamaño del punto es el número de celdas del intervalo."]
    s = D.SHAP_GLOBAL.get(m)
    if not s:
        return kpis, fig_cm, nota_cm, fig_roc, fig_pr, fig_cal, nota_cal, vacio("SHAP no disponible"), ""
    imp = pd.Series(s["importancia"]).sort_values()
    fig_s = go.Figure(go.Bar(x=imp.values, y=[GRUPO_ETIQ[g] for g in imp.index], orientation="h", marker_color=COLOR_M[m],
                             text=[f"{v:.3f}" for v in imp.values], textposition="outside",
                             hovertemplate="<b>%{y}</b><br>media |SHAP| = %{x:.4f}<extra></extra>"))
    fig_s.update_layout(height=430, title="Importancia media |SHAP| para la clase predicha",
                        xaxis=dict(title="Aporte medio absoluto a la probabilidad"), yaxis=dict(showgrid=False))
    nota_s = [html.B("💡 Interpretación: "), f"Para {NOMBRES[m]} la variable que más mueve la probabilidad es "
              f"{GRUPO_ETIQ[imp.index[-1]].lower()}, seguida de {GRUPO_ETIQ[imp.index[-2]].lower()}. ",
              "(El SHAP global se calculó con los modelos de la submuestra, que son los que carga el dashboard.) " if completo else "",
              html.A("Comparación SHAP del ECLC y del mejor benchmark en la sección 11.9 →", href=LIBRO + "e3_05_modelo_original.html",
                     target="_blank")]
    return kpis, fig_cm, nota_cm, fig_roc, fig_pr, fig_cal, nota_cal, fig_s, nota_s


# ======================================================================== comparacion estadistica
def fig_cd():
    f = V["friedman"]
    rangos = pd.Series(f["rangos"]).sort_values()
    cd = f["cd"]
    fig = go.Figure()
    for i, (n, r) in enumerate(rangos.items()):
        es = n.startswith("ECLC")
        fig.add_scatter(x=[r], y=[i], mode="markers+text", text=[f"  {n} ({r:.2f})"], textposition="middle right",
                        marker=dict(size=14 if es else 10, color="#b0413e" if es else "#1b6f5c",
                                    symbol="star" if es else "circle"), showlegend=False,
                        hovertemplate=f"<b>{n}</b><br>Rango promedio: {r:.2f}<extra></extra>")
    mejor = rangos.iloc[0]
    fig.add_shape(type="rect", x0=mejor, x1=mejor + cd, y0=-0.5, y1=len(rangos) - 0.5, fillcolor="rgba(91,111,179,0.12)",
                  line_width=0)
    fig.add_annotation(x=mejor + cd / 2, y=len(rangos) - 0.3, text=f"CD = {cd:.2f} (no distinguible del primero)",
                       showarrow=False, font=dict(color="#5b6fb3"))
    fig.update_layout(height=420, title=f"Rango promedio en {f['N']} pliegues (1 = mejor) · Friedman p = {f['p_valor']:.1e}",
                      xaxis=dict(title="Rango promedio de F1 macro", range=[0.5, len(rangos) + 3.5]),
                      yaxis=dict(visible=False, autorange="reversed"))
    return fig


def tabla_pruebas(p=None):
    p = D.EST_PAREADAS if p is None else p
    df = pd.DataFrame({
        "Competidor": p["competidor"], "Δ F1 (ECLC − comp.)": p["dif_media_f1"].round(4),
        "IC 95 % BCa": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(p["dif_ic95_inf"], p["dif_ic95_sup"])],
        "Wilcoxon p (Holm)": p["wilcoxon_p_holm"].map(lambda v: f"{v:.2e}"),
        "t corregida p (Holm)": p["nb_p_holm"].map(lambda v: f"{v:.2e}"),
        "McNemar p (Holm)": p["mcnemar_p_holm"].map(lambda v: f"{v:.2e}"),
        "P(ECLC mejor) bayesiana": p["bayes_p_eclc_mejor"].round(3),
        "δ de Cliff": [f"{v:.2f} ({m})" for v, m in zip(p["cliff_delta"], p["magnitud_cliff"])],
    })
    return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
                                sort_action="native", style_table={"overflowX": "auto"},
                                style_cell={"fontFamily": "Nunito, Segoe UI, sans-serif", "fontSize": 13, "padding": "5px"},
                                style_header={"backgroundColor": "#5b6fb3", "color": "white", "fontWeight": "bold"})


def veredicto():
    """Veredicto principal: protocolo de datos completos. El de la submuestra se reporta como complemento."""
    vc = D.VEREDICTO_COMPLETO
    ok = vc["R3"]
    gana = int(D.EST_PAREADAS_COMPLETO.set_index("competidor").loc[vc["mejor_benchmark"], "ecl_gana_pliegues"])
    return dbc.Alert([
        html.H5(("✅ Superioridad estadística verificada (R3)" if ok else "⚠️ Superioridad no verificada (R3 no se cumple)")),
        html.P([f"Entrenamiento completo (464.809 celdas): F1 macro medio en {vc['n_pliegues']} pliegues ",
                html.B(f"ECLC {fmt(vc['f1_eclc'])}"), f" frente a {fmt(vc['f1_mejor_benchmark'])} del mejor benchmark "
                f"({vc['mejor_benchmark']}); prueba reservada {fmt(vc['f1_eclc_prueba_reservada'])}. El ECLC gana en {gana} de "
                f"{vc['n_pliegues']} pliegues; δ de Cliff = {fmt(vc['cliff_vs_mejor'], 2)} (grande)."]),
        html.P("Pruebas frente al mejor benchmark (α = 0,05; Holm ajusta por las 8 comparaciones):"),
        lista_pruebas(vc, D.EST_PAREADAS_COMPLETO),
        html.P(["Con la submuestra de 40.000 celdas (30 pliegues) la ventaja es menor: ECLC ", fmt(V["f1_eclc"]), " frente a ",
                fmt(V["f1_mejor_benchmark"]), f" ({V['mejor_benchmark']}); gana en 27 de 30 pliegues y la confirman Friedman y "
                "Wilcoxon-Holm. Detalle de ambos protocolos en la pestaña de comparación estadística."],
               className="mb-0 small"),
    ], color="success" if ok else "warning")


def lista_pruebas(v, pareadas):
    """Cada prueba frente al mejor benchmark con su estadístico y su p-valor (crudo y ajustado por Holm)."""
    f, mb = v["friedman"], v["mejor_benchmark"]
    fila = pareadas.set_index("competidor").loc[mb]
    ok = v["pruebas_vs_mejor"]
    pe = lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"
    textos = {
        "Friedman (global)": f"Friedman (global): χ² = {f['estadistico']:.1f}, p = {pe(f['p_valor'])}",
        "Nemenyi (rango ECLC vs mejor benchmark > CD)":
            f"Nemenyi: diferencia de rangos {f['rangos'][mb] - f['rangos']['ECLC (original)']:.2f} frente a CD = {f['cd']:.2f}",
        "Wilcoxon-Holm": f"Wilcoxon de rangos con signo: p = {pe(fila['wilcoxon_p'])}; p ajustado (Holm) = {pe(fila['wilcoxon_p_holm'])}",
        "t corregida Nadeau-Bengio (Holm)":
            f"t corregida de Nadeau-Bengio: t = {fila['nb_t']:.2f}, p = {pe(fila['nb_p'])}; p ajustado (Holm) = {pe(fila['nb_p_holm'])}",
        "McNemar en prueba reservada (Holm)":
            f"McNemar (prueba reservada): p = {pe(fila['mcnemar_p'])}; p ajustado (Holm) = {pe(fila['mcnemar_p_holm'])}",
        "Bayesiana (P(ECLC mejor) > 0,95)":
            f"Bayesiana (ROPE ±0,01): P(ECLC mejor) = {fila['bayes_p_eclc_mejor']:.3f}; P(equivalencia práctica) = "
            f"{fila['bayes_p_rope']:.3f}",
    }
    return html.Ul([html.Li(("✅ " if ok[k] else "❌ ") + textos.get(k, k)) for k in ok], className="mb-1")


def veredicto_submuestra():
    """Veredicto con la submuestra de 40.000 celdas (30 pliegues), reportado como complemento."""
    ok = V["R3"]
    return dbc.Alert([
        html.H6("R3 se cumple también en la submuestra" if ok else
                "En la submuestra la ventaja es menor y R3 no alcanza las tres pruebas"),
        html.P(f"F1 macro medio en {V['n_pliegues']} pliegues: ECLC {fmt(V['f1_eclc'])} frente a {fmt(V['f1_mejor_benchmark'])} "
               f"del mejor benchmark ({V['mejor_benchmark']}); gana en 27 de 30 pliegues; δ de Cliff = {fmt(V['cliff_vs_mejor'], 2)}."),
        html.P("Pruebas frente al mejor benchmark (α = 0,05; Holm ajusta por las 8 comparaciones):"),
        lista_pruebas(V, D.EST_PAREADAS),
    ], color="success" if ok else "secondary")


NOMBRE_EST = {"ECLC": "ECLC (original)", "HGB (estado del arte)": "HistGradientBoosting", "stacking lineal": "Stacking lineal",
              **{f"base:{m}": NOMBRES[m] for m in MOD.BASES}}


def tarjeta_completo():
    vc = D.VEREDICTO_COMPLETO
    r = D.EST_RESUMEN_COMPLETO[D.EST_RESUMEN_COMPLETO["metodo"].isin(NOMBRE_EST)].copy()
    r["Modelo"] = r["metodo"].map(NOMBRE_EST)
    r["F1 macro (10 pliegues)"] = [f"{fmt(a)} ± {fmt(b)}" for a, b in zip(r["f1_media"], r["f1_sd"])]
    r["F1 prueba reservada"] = r["f1_prueba_reservada"].round(4)
    tabla = dash_table.DataTable(
        data=r[["Modelo", "F1 macro (10 pliegues)", "F1 prueba reservada"]].to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["Modelo", "F1 macro (10 pliegues)", "F1 prueba reservada"]],
        sort_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Nunito, Segoe UI, sans-serif", "fontSize": 14, "padding": "6px", "textAlign": "left"},
        style_header={"backgroundColor": "#1b6f5c", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[{"if": {"filter_query": '{Modelo} contains "ECLC"'}, "backgroundColor": "#fdecea",
                                 "fontWeight": "bold"}])
    return tarjeta("Con el entrenamiento completo (464.809 celdas)", "mdi:database-check", "cab-verde", [
        html.P(["Misma comparación, pero entrenando a todos los modelos con el conjunto de entrenamiento completo "
                "(validación cruzada 5 × 2 y prueba reservada). El ECLC alcanza ", html.B(f"{fmt(vc['f1_eclc'])}"),
                " de F1 macro en 10 pliegues y ", html.B(fmt(vc["f1_eclc_prueba_reservada"])), " en la prueba reservada, frente a ",
                f"{fmt(vc['f1_mejor_benchmark'])} del mejor benchmark ({vc['mejor_benchmark']})."]),
        dbc.Alert([html.B("✅ R3 se cumple con datos completos " if vc["R3"] else "⚠️ R3 no se cumple con datos completos "),
                   f"(δ de Cliff = {fmt(vc['cliff_vs_mejor'], 2)}):", lista_pruebas(vc, D.EST_PAREADAS_COMPLETO)],
                  color="success" if vc["R3"] else "warning"),
        tabla,
        html.H6("Pruebas pareadas con datos completos: ECLC frente a cada competidor (p ajustados por Holm)", className="mt-3"),
        tabla_pruebas(D.EST_PAREADAS_COMPLETO),
        nota("Los modelos desplegados en este dashboard (pestaña de predicción) están entrenados con 40.000 celdas para "
             "caber en la memoria de Render; las cifras de esta tarjeta corresponden al entrenamiento completo.",
             "e3_06_validacion.html", "12.8"),
    ], "mb-3")


def sub_estadistica():
    return html.Div([
        intro(html.B("¿El modelo original es mejor de verdad? "), "Todos los modelos se evaluaron en los mismos 30 pliegues "
              "externos (10 repeticiones × 3, la primera es la partición del benchmark) y en la misma prueba reservada, "
              "y además con el entrenamiento completo (tarjeta verde)."),
        tarjeta_completo(),
        html.H5("Con la submuestra de 40.000 celdas del benchmark", className="seccion-titulo mt-2"),
        veredicto_submuestra(),
        dbc.Row([
            dbc.Col(tarjeta("Diagrama de diferencia crítica (Friedman → Nemenyi)", "mdi:ruler", "cab-azul", [
                grafica(figura=fig_cd()),
                nota("Los modelos dentro de la franja azul no se distinguen del primero según Nemenyi (α = 0,05). Nemenyi es "
                     "conservador porque controla todas las comparaciones a la vez; por eso se complementa con pruebas pareadas "
                     "dirigidas con corrección de Holm.", "e3_06_validacion.html", "12.3")]), lg=12, className="mb-3"),
        ]),
        tarjeta("Pruebas pareadas: ECLC frente a cada competidor", "mdi:table-check", "cab-lavanda", [
            tabla_pruebas(),
            nota("Wilcoxon y t corregida de Nadeau-Bengio usan los 30 pliegues; McNemar usa la prueba reservada. Todas son "
                 "unilaterales (H₁: el ECLC es mejor) y se ajustan por Holm. La prueba bayesiana usa una región de equivalencia "
                 "práctica de ±0,01 en F1 macro.", "e3_06_validacion.html", "12.4")], "mb-3"),
    ])


# ======================================================================== modelo original
MATEMATICA = r"""
**1. Pérdida de cada modelo por clase.** Con predicciones fuera de pliegue $\hat p_{mc}(x_j)$ (ningún modelo se evalúa
sobre celdas que vio), la pérdida logarítmica binaria del modelo $m$ en la clase $c$ y la celda $j$ es

$$\ell_{mc}(j) = -\,\mathbb{1}[y_j=c]\log \hat p_{mc}(x_j) - \mathbb{1}[y_j\neq c]\log\bigl(1-\hat p_{mc}(x_j)\bigr).$$

Vale casi 0 si el modelo acertó con seguridad y crece sin límite si se equivocó estando seguro.

**2. Competencia local con contracción.** Para una celda nueva $x$ con vecinos $j \in N_k(x)$ a distancias
$d_1 \le \dots \le d_k$:

$$a_j = s_{y_j}\,e^{-(d_j/d_k)^2},\quad s_c = \frac{n}{C\,n_c},\qquad
L_{mc}(x) = \frac{\sum_j a_j\,\ell_{mc}(j) + \tau\,\bar L_{mc}}{\sum_j a_j + \tau}.$$

El núcleo gaussiano da más peso a los vecinos cercanos y $s_c$ a los de clases raras. $\tau$ actúa como $\tau$
«vecinos imaginarios» con la pérdida global $\bar L_{mc}$: si la vecindad aporta poca evidencia, manda el historial global.

**3. Pesos de Gibbs y combinación.**

$$w_{mc}(x) = \frac{e^{-\beta L_{mc}(x)}}{\sum_{m'} e^{-\beta L_{m'c}(x)}},\qquad
p(c\mid x) \propto \sum_m w_{mc}(x)\,\hat p_{mc}(x).$$

**Proposición 1 (de dónde sale la regla).** $w_{\cdot c}(x)$ es el **único** minimizador en el símplex
$\{w \ge 0, \sum_m w_m = 1\}$ de

$$F(w) = \sum_m w_m L_{mc}(x) + \frac{1}{\beta}\sum_m w_m \log w_m .$$

*Idea de la prueba:* $F$ es lineal más $\tfrac1\beta$ veces la entropía negativa, estrictamente convexa, así que el
mínimo es único. Con el multiplicador de Lagrange de $\sum_m w_m = 1$:
$L_{mc} + \tfrac1\beta(1+\log w_m) + \lambda = 0 \Rightarrow w_m \propto e^{-\beta L_{mc}}$. La entropía impide poner todo el
peso en un modelo con poca evidencia. Casos límite: $\beta = 0$ da el promedio simple; $\beta\to\infty$, la selección del
mejor modelo local (OLA); $\tau\to\infty$, pesos globales por clase.

**Proposición 2 (garantía).** En la vecindad de $x$ y para cada clase,

$$\tilde L_c(q) \;\le\; \sum_m w_{mc}\,L_{mc}(x) \;\le\; \min_m L_{mc}(x) + \frac{\log M}{\beta}.$$

*Idea de la prueba:* la pérdida logarítmica es convexa, así que por la desigualdad de Jensen la pérdida de la mezcla es
a lo sumo la mezcla de las pérdidas (primera desigualdad). Como $F(w) \le F(e_{m^\star}) = \min_m L_{mc}$ y la entropía es
a lo sumo $\log M$, se obtiene la segunda. **El ensamble nunca es mucho peor que el mejor modelo de la zona:** como
máximo $\log(7)/\beta$ por encima.

**Proposición 3 (complejidad).** Ajustar la combinación cuesta $O(nMC)$; predecir $n_q$ celdas,
$O(n_q\,n\,p + n_q\,k\,M\,C)$, dominado por la búsqueda exacta de vecinos (medido: el tiempo crece linealmente con $n$).

Las proposiciones 1 y 2 se verifican numéricamente en las pruebas unitarias del modelo.
"""

PSEUDO = r"""
**Entrada:** entrenamiento $(X, y)$; modelos base $f_1,\dots,f_M$; hiperparámetros $k, \beta, \tau$.

1. Para cada modelo $m$: probabilidades fuera de pliegue $\hat P_m(x_j)$ con 3 pliegues; reentrenar $f_m$ con todo $(X, y)$.
2. Pérdida binaria de referencia: $\ell_{mc}(j) = -[y_j=c]\log \hat p_{mc}(x_j) - [y_j\neq c]\log(1-\hat p_{mc}(x_j))$.
3. Pérdida global por clase $\bar L_{mc}$ (promedio ponderado por $s_{y_j} = n / (C\,n_{y_j})$).

**Predicción de $x$:**

4. Buscar los $k$ vecinos de $x$ en el espacio preprocesado; pesos $a_j = s_{y_j}\exp(-(d_j/d_k)^2)$, $A=\sum_j a_j$.
5. $L_{mc}(x) = \dfrac{\sum_j a_j\ell_{mc}(j) + \tau \bar L_{mc}}{A + \tau}$ (contracción hacia la pérdida global).
6. $w_{mc}(x) = \dfrac{\exp(-\beta L_{mc}(x))}{\sum_{m'}\exp(-\beta L_{m'c}(x))}$ (solución cerrada del problema regularizado con entropía).
7. $q_c(x) = \sum_m w_{mc}(x)\,\hat p_{mc}(x)$; $\;p(c\mid x) = q_c / \sum_{c'} q_{c'}$; predecir $\arg\max_c p(c\mid x)$.

**Salida:** probabilidades por clase. **Parada:** no iterativo (forma cerrada); el costo lo domina la búsqueda de vecinos.
"""

DIFERENCIAS = [
    ("OLA / LCA (Woods et al., 1997)", "Exactitud local", "Elige un solo modelo", "Pesos suaves por clase con pérdida logarítmica"),
    ("KNORA-E (Ko et al., 2008)", "Aciertos en los k vecinos", "Selecciona un subconjunto", "No descarta modelos; pondera con forma cerrada"),
    ("META-DES (Cruz et al., 2015)", "Meta-clasificador de competencia", "Selección", "Sin meta-clasificador: la competencia es la pérdida"),
    ("DES-RRC / DES-KL (Woloszynski y Kurzynski, 2011; Woloszynski et al., 2012)", "Competencia probabilística",
     "Pesos por modelo", "Pesos distintos para cada clase y contracción global"),
    ("Stacking lineal (Wolpert, 1992)", "Ninguna (global)", "Combinación aprendida", "Pesos que cambian con la región del espacio"),
    ("Hedge / pesos exponenciales (Freund y Schapire, 1997)", "Pérdida acumulada global", "Pesos exponenciales", "Pérdida local en vecinos y por clase"),
]


ETIQ_ABLACION = {"sin nucleo de distancia": "sin núcleo de distancia", "sin vecindario (tau = inf)": "sin vecindario (τ = ∞)",
                 "sin contraccion (tau = 0)": "sin contracción (τ = 0)", "promedio simple (beta = 0)": "promedio simple (β = 0)"}


def fig_ablacion():
    a = D.EST_ABLACION.sort_values("f1_media")
    eclc = D.EST_RESUMEN.set_index("metodo").loc["ECLC", "f1_media"]
    # El valor va junto al nombre (eje y) para no chocar con la barra de error ni con la línea del ECLC
    etiquetas = [f"{ETIQ_ABLACION.get(v, v)}  <b>{f:.4f}</b>" for v, f in zip(a["variante"], a["f1_media"])]
    fig = go.Figure(go.Bar(x=a["f1_media"], y=etiquetas, orientation="h",
                           error_x=dict(type="data", array=a["f1_sd"], color="#37474f", thickness=1.5),
                           marker_color=["#5b6fb3" if v.startswith("sin") else "#9e9e9e" for v in a["variante"]],
                           hovertemplate="<b>%{y}</b><br>F1 macro %{x:.4f} ± %{error_x.array:.4f}<extra></extra>"))
    fig.add_vline(x=eclc, line_dash="dash", line_color="#b0413e")
    fig.add_annotation(x=eclc, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"<b>ECLC completo {eclc:.4f}</b>", font=dict(color="#b0413e"))
    fig.update_layout(height=450, title="Ablación (azul) y reglas de combinación estándar (gris), 30 pliegues",
                      xaxis=dict(title="F1 macro medio (± desviación estándar)",
                                 range=[a["f1_media"].min() - 0.02, max(a["f1_media"].max(), eclc) + 0.015]),
                      yaxis=dict(showgrid=False), margin=dict(t=70))
    return fig


def fig_convergencia():
    c = D.CONVERGENCIA
    g = c.groupby("ensayo")["mejor_hasta_ahora"].agg(["mean", "min", "max"]).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=g["ensayo"], y=g["max"], mode="lines", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=g["ensayo"], y=g["min"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(176,65,62,0.15)", name="rango entre pliegues")
    fig.add_scatter(x=g["ensayo"], y=g["mean"], mode="lines+markers", line=dict(color="#b0413e", width=3), name="media")
    fig.update_layout(height=360, title="Convergencia de la optimización bayesiana del ECLC (TPE, 6 ensayos)",
                      xaxis=dict(title="Ensayo", dtick=1), yaxis=dict(title="Mejor F1 macro interno hasta el ensayo"),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def fig_datasets():
    d = D.DATASETS_AD
    metodos = ["ECLC", "stacking lineal", "promedio simple (beta = 0)", "KNORA-E", "HGB (estado del arte)"]
    mejor_base = d[d["metodo"].str.startswith("base:")].groupby(["conjunto", "metodo"])["f1_macro"].mean() \
        .groupby("conjunto").max()
    fig = go.Figure()
    fig.add_bar(x=mejor_base.index, y=mejor_base.values, name="Mejor modelo base", marker_color="#1b6f5c")
    for m, col in zip(metodos, ["#b0413e", "#5b6fb3", "#9e9e9e", "#c8a13a", "#455a64"]):
        s = d[d["metodo"] == m].groupby("conjunto")["f1_macro"].mean()
        fig.add_bar(x=s.index, y=s.values, name=m, marker_color=col)
    fig.update_layout(barmode="group", height=380, title="Conjuntos adicionales (5 × 3 pliegues, mismo presupuesto)",
                      yaxis=dict(title="F1 macro medio", range=[0.85, 1.0]), legend=dict(orientation="h", y=-0.25))
    return fig


def sub_original():
    filas = [html.Tr([html.Td(a), html.Td(b), html.Td(c), html.Td(d)]) for a, b, c, d in DIFERENCIAS]
    return html.Div([
        intro(html.B("Ensamble por competencia local y por clase (ECLC). "), "Para cada celda nueva, cada uno de los siete "
              "modelos recibe un peso según su pérdida logarítmica en las celdas más parecidas, y ese peso es distinto para cada "
              "cobertura. Los modelos base vienen de librerías; la regla de combinación, la búsqueda de vecinos y la "
              "contracción están programadas con numpy (dashboard/src/original_model.py)."),
        veredicto(),
        dbc.Row([
            dbc.Col(tarjeta("Pseudocódigo", "mdi:code-braces", "cab-verde", [dcc.Markdown(PSEUDO, mathjax=True)]),
                    lg=6, className="mb-3"),
            dbc.Col(tarjeta("¿Qué es nuevo frente a la literatura?", "mdi:compare", "cab-lavanda", [
                dbc.Table([html.Thead(html.Tr([html.Th("Método afín"), html.Th("Competencia"), html.Th("Decisión"),
                                               html.Th("Diferencia del ECLC")])), html.Tbody(filas)],
                          size="sm", striped=True, responsive=True),
                nota("Matriz de novedad, cadenas de búsqueda y declaración completa en el capítulo 4 del Jupyter Book.",
                     "e3_04_literatura.html", "10"),
            ]), lg=6, className="mb-3"),
        ]),
        tarjeta("Fundamento matemático: formulación y propiedades demostradas", "mdi:sigma", "cab-azul", [
            dcc.Markdown(MATEMATICA, mathjax=True),
            nota("Formulación completa, demostraciones y verificación numérica en las secciones 11.2 y 11.3 del Jupyter Book.",
                 "e3_05_modelo_original.html", "11.3"),
        ], "mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Curva de convergencia", "mdi:chart-timeline-variant", "cab-ocre", [
                grafica(figura=fig_convergencia()),
                nota("La regla de combinación es de forma cerrada (no itera); lo que converge es la búsqueda de (k, β, τ). "
                     "Con el mismo presupuesto del benchmark (6 ensayos) la mejora se estabiliza en los primeros ensayos.")]),
                    lg=6, className="mb-3"),
            dbc.Col(tarjeta("Ablación: el aporte de cada componente", "mdi:puzzle-remove", "cab-azul", [
                grafica(figura=fig_ablacion()),
                nota("Cada barra azul retira un componente del ECLC con los mismos hiperparámetros; las grises son reglas "
                     "estándar sobre las mismas probabilidades. Si una barra supera a la línea roja, ese componente no aporta.",
                     "e3_05_modelo_original.html", "11.7")]), lg=6, className="mb-3"),
        ]),
        tarjeta("Evaluación en otros problemas de clasificación", "mdi:database-search", "cab-terracota", [
            grafica(figura=fig_datasets()),
            nota("Dígitos manuscritos (10 clases), vinos (3 clases) y cáncer de mama (2 clases), con los mismos 7 modelos, "
                 "espacios de búsqueda y presupuesto.", "e3_06_validacion.html", "12.6")], "mb-3"),
    ])


# ======================================================================== layout
def layout():
    return html.Div([
        html.H2("🤖 Modelos", className="seccion-titulo"),
        dbc.Tabs([
            dbc.Tab(sub_lideres(), label="🏆 Tabla de líderes", tab_id="lideres"),
            dbc.Tab(sub_prediccion(), label="🔮 Predicción en vivo", tab_id="prediccion"),
            dbc.Tab(sub_diagnostico(), label="🩺 Diagnóstico", tab_id="diagnostico"),
            dbc.Tab(sub_estadistica(), label="📐 Comparación estadística", tab_id="estadistica"),
            dbc.Tab(sub_original(), label="🧪 Modelo original (ECLC)", tab_id="original"),
            dbc.Tab(dbc.Tabs([
                dbc.Tab(E2.sub_modelo_base(), label="📏 Modelo base"),
                dbc.Tab(E2.sub_benchmark(), label="🏁 Benchmark 112 combinaciones"),
                dbc.Tab(E2.sub_optimizacion(), label="🎛️ Optimizadores"),
                dbc.Tab(E2.sub_eficiencia(), label="⚡ Eficiencia"),
            ], className="subpestanas nav-pills mt-2"), label="📚 Benchmark detallado", tab_id="e2"),
        ], active_tab="lideres", className="mb-3"),
    ], className="contenido-tab")
