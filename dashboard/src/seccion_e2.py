"""Secciones del Entregable 2 que siguen vigentes tras las correcciones del Entregable 3.

Modelo base, benchmark combinatorio (leido de la tabla maestra corregida), comparacion de
optimizadores y eficiencia computacional. Los callbacks usan dash.callback, de modo que
quedan registrados al importar este modulo desde pages/modelos.py.
"""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, ctx, dcc, html

from . import data as D
from .data import CLASES, CORTO, EMOJI, LARGO, LIBRO, fmt, miles, vacio
from .plots import _tasas, fig_confusion, fig_halving, fig_solapamiento
from .ui import control, grafica, intro, kpi, nota, radio, tarjeta


LR_F1 = D.BASE["test"]["F1 macro"]


LR_ACC = D.BASE["test"]["Accuracy"]


N_COMB = int(D.TABLA["aplicable"].sum())


def sub_modelo_base():
    t = D.BASE["test"]
    mejor_dummy = max(v["F1 macro"] for v in D.BASE["dummies"].values())
    brecha = abs(D.BASE["train"]["F1 macro"] - t["F1 macro"])
    return html.Div([
        intro(html.B("Paso 1 · Modelo base. "), "Una regresión logística multinomial con ingeniería de variables, "
              "ajustada con validación cruzada estratificada. Sirve de referencia: cualquier modelo más complejo tiene que superarla."),
        dbc.Row([
            dbc.Col(kpi("🎯 Accuracy en prueba", fmt(t["Accuracy"]), f"vs. {fmt(D.BASE['dummies']['Dummy (más frecuente)']['Accuracy'])} del clasificador mayoritario", 1), lg=3, md=6, className="mb-3"),
            dbc.Col(kpi("⚖️ F1 macro en prueba", fmt(t["F1 macro"]), f"vs. {fmt(mejor_dummy)} del mejor dummy", 2), lg=3, md=6, className="mb-3"),
            dbc.Col(kpi("🚀 Ganancia en F1 macro", f"+{(t['F1 macro'] / mejor_dummy - 1) * 100:.0f} %", "sobre la inferencia ingenua", 3), lg=3, md=6, className="mb-3"),
            dbc.Col(kpi("🩺 Brecha entrenamiento–prueba", fmt(brecha, 4), "F1 macro: sin sobreajuste", 4), lg=3, md=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Desempeño frente a la línea base", "mdi:trophy", "cab-verde", [
                radio("vista-metricas", [("base", "🎯 Modelo vs. dummies"), ("ajuste", "🩺 Diagnóstico de ajuste")], "base"),
                grafica("fig-metricas"),
                nota("Predecir siempre lodgepole acierta el 49 % por puro desbalance, pero su F1 macro es 0,09. El modelo "
                     "multiplica el F1 macro, y entrenamiento y prueba casi coinciden: su límite es el sesgo de un modelo lineal, "
                     "no la falta de datos.", "03_modelo_base.html", "3.5"),
            ]), lg=6, className="mb-3"),
            dbc.Col(tarjeta("¿Qué pesa en cada cobertura? Coeficientes", "mdi:tune-variant", "cab-lavanda", [
                dbc.Row([
                    dbc.Col(control("🌲 Cobertura", dcc.Dropdown(id="clase-coef", clearable=False, value=7,
                                                                options=[{"label": f"{EMOJI[c]} {LARGO[c]}", "value": c} for c in CLASES])), md=6),
                    dbc.Col(control("🧩 Tipo de variable", dcc.Dropdown(id="grupo-coef", clearable=False, value="todas", options=[
                        {"label": "Todas", "value": "todas"}, {"label": "Continuas", "value": "Continua"},
                        {"label": "Áreas silvestres", "value": "Área silvestre"}, {"label": "Tipos de suelo", "value": "Tipo de suelo"}])), md=6),
                ]),
                control("🔢 Número de variables", dcc.Slider(id="n-coef", min=5, max=20, step=1, value=10,
                                                             marks={i: str(i) for i in (5, 10, 15, 20)}), "mt-2"),
                grafica("fig-coef"),
                html.Div(id="nota-coef", className="nota"),
            ]), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Matriz de confusión", "mdi:grid-large", "cab-terracota", [
                radio("modo-confusion", [("real", "% de la clase real (recall)"), ("pred", "% de la predicción (precisión)"), ("conteo", "Conteos")], "real"),
                grafica("fig-confusion"),
                nota("Los errores no se dispersan: se concentran entre coberturas vecinas del gradiente. El álamo temblón casi "
                     "desaparece (más del 90 % → lodgepole) y el abeto Douglas se confunde con el ponderosa.", "03_modelo_base.html", "3.5"),
            ]), lg=6, className="mb-3"),
            dbc.Col(tarjeta("Los errores siguen el nicho compartido", "mdi:vector-intersection", "cab-lago", [
                grafica("fig-solapamiento", fig_solapamiento()),
                nota("Cada punto es un par de coberturas: cuanto más comparten su franja de elevación, más las confunde el "
                     "modelo. El tamaño indica cuántas celdas se confundieron.", "03_modelo_base.html", "3.7"),
            ]), lg=6, className="mb-3"),
        ]),
        tarjeta("Explorador de confusiones: la ecología detrás de cada error", "mdi:magnify", "cab-ocre",
                html.Div(id="explicacion-confusion", className="panel-explicacion"), "mb-3"),
    ])


def sub_benchmark():
    return html.Div([
        intro(html.B("Paso 2 · Benchmark. "), f"Se entrenaron 7 algoritmos con 4 estrategias de balanceo y 4 métodos de "
              f"optimización ({N_COMB} combinaciones aplicables) con validación cruzada anidada: 3 pliegues externos para "
              "medir el desempeño y 3 internos para elegir hiperparámetros. Así el F1 reportado nunca se usó para ajustar el modelo."),
        dbc.Row([
            dbc.Col(tarjeta("Mapa del benchmark: algoritmo × balanceo", "mdi:view-grid-plus", "cab-verde", [
                dbc.Row([
                    dbc.Col(control("🎛️ Método de optimización", dcc.Dropdown(id="opt-heatmap", clearable=False, value="mejor", options=[
                        {"label": "El mejor de los cuatro", "value": "mejor"}] + [{"label": l, "value": v} for v, l in D.OPTIMIZADORES.items()])), md=6),
                    dbc.Col(control("📏 Qué mostrar", dcc.Dropdown(id="metrica-heatmap", clearable=False, value="f1_macro_media", options=[
                        {"label": "F1 macro (media de 3 pliegues)", "value": "f1_macro_media"},
                        {"label": "Accuracy (media)", "value": "accuracy_media"},
                        {"label": "Tiempo total (minutos)", "value": "tiempo_min"}])), md=6),
                ], className="mb-2"),
                grafica("fig-heatmap"),
                nota("Haz clic en una celda para ver su detalle. Con el protocolo corregido (mismo espacio y presupuesto para todas las combinaciones) Random Forest con SMOTE lidera; Naive Bayes queda último "
                     "porque supone variables independientes y gaussianas, algo que estos datos violan. KNN no admite pesos de clase (N/A).",
                     "06_benchmark_modelos.html", "5.4"),
            ]), lg=7, className="mb-3"),
            dbc.Col(tarjeta("Detalle de la combinación", "mdi:clipboard-text-search", "cab-lago", [
                html.Div(id="detalle-combinacion"),
            ]), lg=5, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("¿Vale la pena el costo? Tiempo frente a desempeño", "mdi:timer-sand", "cab-azul", [
                dbc.Checklist(id="modelos-dispersion", inline=True, value=list(D.MODELOS),
                              options=[{"label": f" {n}", "value": k} for k, n in D.MODELOS.items()]),
                grafica("fig-dispersion"),
                nota("Cada punto es una combinación. Arriba a la izquierda está lo ideal (alto F1, poco tiempo). El tiempo crece "
                     "con el balanceo sintético (SMOTE/ADASYN multiplican las filas) y no siempre trae más F1.",
                     "06_benchmark_modelos.html", "5.4"),
            ]), lg=7, className="mb-3"),
            dbc.Col(tarjeta("¿Ayuda balancear las clases?", "mdi:scale-balance", "cab-ocre", [
                grafica("fig-balanceo"),
                nota("Diferencia de F1 macro de cada estrategia frente a no balancear, por algoritmo. El balanceo no afecta a los "
                     "algoritmos por igual: en Random Forest SMOTE y ADASYN suben el F1 macro en 0,033 y los pesos de clase en 0,025, "
                     "mientras que en XGBoost las tres estrategias lo reducen entre 0,023 y 0,040 (protocolo corregido).",
                     "06_benchmark_modelos.html", "5.4"),
            ]), lg=5, className="mb-3"),
        ]),
    ])


def sub_optimizacion():
    ahorro = D.MULTI_RES["ahorro_relativo"] * 100
    return html.Div([
        intro(html.B("Paso 3 · Optimización de hiperparámetros. "), "Grid Search prueba todas las combinaciones; Random Search "
              "sortea; la optimización bayesiana (TPE) aprende de los intentos previos; el algoritmo genético cruza y muta las "
              "mejores configuraciones. ¿Cuál encuentra antes una buena solución?"),
        dbc.Row([
            dbc.Col(tarjeta("Experimento controlado: la carrera de los optimizadores", "mdi:run-fast", "cab-verde", [
                dbc.Row([
                    dbc.Col(control("🏁 Métodos", dbc.Checklist(id="metodos-anytime", inline=True,
                                                               value=list(D.COLOR_OPT_NOMBRE), options=[{"label": f" {m}", "value": m} for m in D.COLOR_OPT_NOMBRE])), md=8),
                    dbc.Col(control("⏱️ Eje horizontal", dbc.RadioItems(id="eje-anytime", inline=True, value="evaluacion", options=[
                        {"label": " Evaluaciones", "value": "evaluacion"}, {"label": " Segundos", "value": "tiempo_acumulado_s"}])), md=4),
                ], className="mb-2"),
                grafica("fig-anytime"),
                nota("Árbol de decisión, espacio de 5 hiperparámetros, 30 evaluaciones y 5 semillas. La línea es el mejor F1 "
                     "encontrado hasta ese momento (media de las semillas) y la banda, su rango. TPE obtiene la mejor solución y la más "
                     "estable (0,7226) y la mayor área bajo la curva (0,700). En el benchmark, con rejillas pequeñas y seis ensayos, "
                     "esa ventaja no aparecía.",
                     "07_optimizacion.html", "6.4"),
            ]), lg=7, className="mb-3"),
            dbc.Col(tarjeta("Los cuatro métodos en el benchmark", "mdi:tune", "cab-azul", [
                radio("metrica-opt", [("f1_macro_media", "F1 macro"), ("tiempo_total_s", "Tiempo"), ("evaluaciones_totales", "Evaluaciones")], "f1_macro_media"),
                grafica("fig-optimizadores"),
                nota("Sobre las 108 combinaciones los cuatro métodos alcanzan F1 muy parecidos; lo que cambia es el costo. La "
                     "elección del algoritmo pesa mucho más que la del optimizador.", "07_optimizacion.html", "6.2"),
            ]), lg=5, className="mb-3"),
        ]),
        tarjeta("Successive Halving: descartar pronto a los candidatos malos", "mdi:filter-variant", "cab-lavanda", [
            dbc.Row([
                dbc.Col(grafica(figura=fig_halving()), lg=8),
                dbc.Col([
                    html.Div([html.Div("1", className="numero"), html.Div("Se evalúan 8 configuraciones de un bosque con poco presupuesto (20 árboles).")], className="paso"),
                    html.Div([html.Div("2", className="numero"), html.Div("Solo la mitad mejor pasa a la siguiente ronda, con el doble de árboles (η = 2).")], className="paso"),
                    html.Div([html.Div("3", className="numero"), html.Div("Se repite hasta quedar un ganador entrenado con 160 árboles.")], className="paso"),
                    html.Div(kpi("💰 Ahorro de cómputo", f"{ahorro:.0f} %", "frente a evaluar todo con el presupuesto completo", 5), className="mt-3"),
                ], lg=4),
            ]),
            nota("El ganador (profundidad 20, hoja mínima 1) ya era el primero con 20 árboles y el orden entre configuraciones "
                 "se mantuvo en todas las rondas: por eso descartar pronto no costó calidad.", "07_optimizacion.html", "6.6"),
        ], "mb-3"),
    ])


def sub_eficiencia():
    p = D.PERFIL
    return html.Div([
        intro(html.B("Paso 6 · Costo computacional. "), "Un buen modelo también tiene que ser entrenable y usable. Aquí se "
              "compara cómo crece el tiempo con el tamaño de los datos y qué ganan las implementaciones optimizadas."),
        dbc.Row([
            dbc.Col(tarjeta("¿Cómo crece el tiempo con los datos?", "mdi:chart-timeline-variant", "cab-verde", [
                dbc.Checklist(id="modelos-escala", inline=True, value=list(D.ESC["modelo"].unique()),
                              options=[{"label": f" {m}", "value": m} for m in D.ESC["modelo"].unique()]),
                grafica("fig-escala"),
                nota("En escala log-log una recta de pendiente b significa tiempo ∝ nᵇ. La SVM con núcleo RBF crece casi "
                     "cuadráticamente (b ≈ 1,9): por eso se entrenó con una submuestra de 9.000 celdas. XGBoost hist escala mejor que lineal.",
                     "10_conclusiones_finales.html", "9.1"),
            ]), lg=7, className="mb-3"),
            dbc.Col(tarjeta("Implementación estándar frente a optimizada", "mdi:rocket-launch", "cab-azul", [
                control("🧩 Algoritmo", dcc.Dropdown(id="algoritmo-comp", clearable=False, value="XGBoost (200 árboles)",
                                                    options=[{"label": m, "value": m} for m in D.COMP["modelo"].unique()])),
                grafica("fig-comp"),
                html.Div(id="nota-comp", className="nota"),
            ]), lg=5, className="mb-3"),
        ]),
        tarjeta("Costo de afinar XGBoost con el conjunto completo", "mdi:cash-clock", "cab-ocre", dbc.Row([
            dbc.Col(kpi("⏱️ Entrenamiento", f"{p.iloc[1]['tiempo_entrenamiento_s']:.0f} s",
                        f"vs. {p.iloc[0]['tiempo_entrenamiento_s']:.0f} s por defecto", 2), md=3, className="mb-2"),
            dbc.Col(kpi("📈 F1 macro", fmt(p.iloc[1]["F1_macro_prueba"]),
                        f"vs. {fmt(p.iloc[0]['F1_macro_prueba'])} por defecto: los hiperparámetros elegidos con 40.000 "
                        "celdas se quedan cortos con todos los datos", 1), md=3, className="mb-2"),
            dbc.Col(kpi("💾 Memoria pico", f"{p.iloc[1]['memoria_pico_entrenamiento_MB']:.0f} MB",
                        "igual en ambas versiones", 5), md=3, className="mb-2"),
            dbc.Col(kpi("🎲 Estabilidad", f"± {fmt(D.SEMILLAS['F1_macro'].std(), 4)}",
                        "DE del F1 macro entre 5 semillas", 3), md=3, className="mb-2"),
        ]), "mb-3"),
    ])


@callback(Output("fig-metricas", "figure"), Input("vista-metricas", "value"))
def cb_metricas(vista):
    met = ["Accuracy", "Precisión macro", "Recall macro", "F1 macro"]
    fig = go.Figure()
    if vista == "base":
        for (nombre, valores), color in zip(D.BASE["dummies"].items(), ["#ffcc80", "#ffab91", "#ce93d8"]):
            fig.add_bar(x=met, y=[valores[m] for m in met], name=nombre, marker_color=color,
                        hovertemplate=f"<b>{nombre}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>")
        t, ic = D.BASE["test"], D.BASE["ic95"]
        fig.add_bar(x=met, y=[t[m] for m in met], name="Regresión logística", marker_color="#0f9d58",
                    text=[fmt(t[m]) for m in met], textposition="outside",
                    error_y=dict(type="data", array=[ic[m][1] - t[m] for m in met], arrayminus=[t[m] - ic[m][0] for m in met],
                                 color="#0b4f34", thickness=2, width=6),
                    hovertemplate="<b>Regresión logística</b><br>%{x}: %{y:.3f}<extra></extra>")
        titulo = f"🎯 Regresión logística frente a la inferencia ingenua (prueba, {miles(D.BASE['n_test'])} celdas)"
    else:
        for parte, nombre, color in [("train", "Entrenamiento", "#26a6e8"), ("test", "Prueba", "#0f9d58")]:
            v = D.BASE[parte]
            fig.add_bar(x=met, y=[v[m] for m in met], name=nombre, marker_color=color, text=[fmt(v[m]) for m in met],
                        textposition="outside", hovertemplate=f"<b>{nombre}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>")
        titulo = "Entrenamiento vs. prueba: brecha < 0,01 en todas las métricas"
    fig.update_layout(barmode="group", bargap=0.25, height=420, title=titulo, yaxis=dict(range=[0, 0.88], title="Valor de la métrica"),
                      xaxis=dict(showgrid=False), legend=dict(orientation="h", y=-0.15, x=0))
    return fig


NOTAS_COEF = {
    1: "Cada 100 m de ascenso multiplican sus odds por ≈ {or_}. Rawah y los suelos subalpinos (9, 21 y 22) la favorecen; los suelos pedregosos 2, 4 y 37 la excluyen.",
    2: "Efecto altitudinal moderado (≈ ×{or_} por cada 100 m): ocupa el centro del gradiente. Sus coeficientes más fuertes son geográficos y edáficos: Rawah y el suelo 12 a favor; los suelos 1, 4 y 5 del cañón en contra.",
    3: "La altitud la penaliza (≈ ×{or_} por cada 100 m). Los suelos 1 a 4, complejos de afloramiento rocoso del cañón, son su firma edáfica; Rawah la excluye.",
    4: "Las áreas silvestres actúan como atajo geográfico: Cache la Poudre ({cp}), la única con bosque ribereño, la favorece y Comanche Peak la excluye. Cada 100 m de ascenso reducen sus odds a ≈ {or_}.",
    5: "Sin variables de perturbación (incendios, edad del rodal), ninguna predictora le da suficiente evidencia: su recall es apenas {rec}.",
    6: "Los índices de sombreado tienen signos opuestos por colinealidad: solo su efecto conjunto tiene lectura física, a favor de laderas poco expuestas.",
    7: "El efecto altitudinal más fuerte del modelo: +100 m multiplican sus odds por ≈ {or_}. Los suelos pedregosos de alta montaña completan su firma.",
}


@callback(Output("fig-coef", "figure"), Output("nota-coef", "children"),
              Input("clase-coef", "value"), Input("grupo-coef", "value"), Input("n-coef", "value"))
def cb_coef(clase, grupo, n):
    coef = pd.Series(D.BASE["coeficientes"][str(clase)])
    df = pd.DataFrame({"var": coef.index, "coef": coef.values})
    df["grupo"] = df["var"].map(D.grupo_variable)
    df["nombre"] = df["var"].map(D.NOMBRE_ING)
    if grupo != "todas":
        df = df[df["grupo"] == grupo]
    df = df.reindex(df["coef"].abs().sort_values(ascending=False).index).head(n).sort_values("coef")
    fig = go.Figure(go.Bar(
        x=df["coef"], y=df["nombre"], orientation="h", customdata=np.c_[np.exp(df["coef"]), df["grupo"]],
        marker=dict(color=["#0f9d58" if v > 0 else "#d9643a" for v in df["coef"]]),
        hovertemplate="<b>%{y}</b> (%{customdata[1]})<br>Coeficiente: %{x:+.2f}<br>Razón de odds: ×%{customdata[0]:.2f}<extra></extra>"))
    fig.update_layout(height=max(360, 26 * len(df) + 120), title=f"{EMOJI[clase]} {LARGO[clase]}: 🟢 aumenta · 🟠 reduce la probabilidad",
                      xaxis=dict(title="Coeficiente (log-odds; continuas por desviación estándar)"), yaxis=dict(showgrid=False))
    or_ = D.BASE["or_100m"][str(clase)]
    texto = NOTAS_COEF[clase].format(or_=fmt(or_, 2), cp=f"{coef['Wilderness_Area_4']:+.2f}".replace(".", ","),
                                     rec=fmt(D.PC_LR.loc[5, "Recall"], 2))
    hijos = [html.B("💡 Interpretación: "), texto, " Razón de odds por +100 m de elevación (efecto conjunto de la elevación y "
             "la elevación relativa al agua): ", html.B(f"×{fmt(or_, 2)}"), ". ",
             html.A("Profundiza en la sección 3.10 del Jupyter Book →", href=LIBRO + "03_modelo_base.html", target="_blank")]
    return fig, hijos


@callback(Output("fig-confusion", "figure"), Input("modo-confusion", "value"))
def cb_confusion(modo):
    return fig_confusion(D.CM_LR, modo, " · haz clic en una celda 👆")


EXPLICACIONES = {
    (1, 2): ("🔥 Transición sucesional", "El pino lodgepole coloniza los sitios tras incendios de reemplazo y, con los siglos, la pícea y el abeto lo sustituyen bajo su dosel. Entre 2.800 y 3.300 m el mismo terreno puede sostener cualquiera de las dos: decide el tiempo desde el último incendio, que no está en los datos.{extra}"),
    (1, 7): ("🏔️ Ecotono del límite arbóreo", "El krummholz es la forma achaparrada de las mismas especies del bosque subalpino, moldeada por el viento y el hielo. Lo determinan la exposición y la nieve a escalas menores que la celda de 30 m, y es 10 veces menos frecuente: la duda se resuelve a favor de la pícea/abeto."),
    (2, 5): ("🍂 Cobertura definida por la perturbación", "El álamo temblón es clonal y rebrota tras incendios: su presencia depende de la historia del sitio, no de la topografía. Su nicho altitudinal está contenido en el del lodgepole, 30 veces más frecuente, y el modelo casi nunca lo elige (recall {rec5})."),
    (2, 6): ("📏 Vecinos en el gradiente", "El abeto Douglas ocupa el borde inferior del nicho del lodgepole; en las laderas frescas del montano medio ambos comparten elevación y suelos."),
    (3, 4): ("💧 Nicho ribereño no monótono", "El álamo/sauce se concentra en celdas casi al nivel del cauce, y por encima y por debajo desaparece: un pico que una recta no representa. Además, el corredor ribereño es compartido: incluso junto al agua el pino ponderosa ocupa buena parte de las celdas."),
    (3, 6): ("🧭 Partición por orientación", "Es el par con mayor solapamiento altitudinal. El abeto Douglas ocupa laderas frescas orientadas al norte y el ponderosa, laderas cálidas. El efecto de la orientación depende de la pendiente (interacción no lineal) y está repartido entre índices de sombreado colineales."),
    (4, 6): ("💧 Corredor ribereño compartido", "El álamo/sauce comparte con el abeto Douglas la elevación, el área (Cache la Poudre) y buena parte de los suelos; la capa hidrológica no registra el nivel freático ni las crecidas que delimitan el bosque de galería."),
}


def explicar_par(a, b):
    if a == b:
        pc = D.PC_LR.loc[a]
        return [html.H6(f"{EMOJI[a]} {LARGO[a]}: celda diagonal (aciertos)"),
                html.P(f"Recall = {fmt(pc['Recall'])} · precisión = {fmt(pc['Precisión'])}. En el mejor benchmark ({D.NOMBRE_MEJOR}, prueba reservada): "
                       f"recall = {fmt(D.PC_MEJOR.loc[a, 'Recall'])} · precisión = {fmt(D.PC_MEJOR.loc[a, 'Precisión'])}."),
                html.Small("Haz clic fuera de la diagonal para entender por qué el modelo confunde dos coberturas.", className="text-muted")]
    clave = tuple(sorted((a, b)))
    r = _tasas(D.CM_LR)
    rx = _tasas(D.CM_MEJOR)
    n = int(D.CM_LR[a - 1, b - 1] + D.CM_LR[b - 1, a - 1])
    if clave in EXPLICACIONES:
        titulo, texto = EXPLICACIONES[clave]
    else:
        titulo, texto = "📏 Vecinos en el gradiente", ("Estas coberturas comparten parte de su franja altitudinal; cuanto mayor es "
                                                     "el solapamiento, más se confunden. Los pares con solapamiento inferior a 0,2 apenas registran errores.")
    med = D.BASE["elev_mediana_error"].get("1-2")
    extra = f" Las celdas de pícea/abeto que el modelo llama lodgepole están a una mediana de {miles(med)} m." if med else ""
    texto = texto.format(extra=extra, rec5=fmt(D.PC_LR.loc[5, "Recall"], 2))
    return [html.H6(f"{titulo}: {CORTO[a]} ↔ {CORTO[b]}"), html.P(texto),
            html.Small(f"Solapamiento altitudinal = {fmt(D.SOLAPE[clave], 2)} · {CORTO[a]} → {CORTO[b]}: {fmt(100 * r[a - 1, b - 1], 1)} % · "
                       f"{CORTO[b]} → {CORTO[a]}: {fmt(100 * r[b - 1, a - 1], 1)} % · {miles(n)} celdas confundidas en total. "
                       f"Con {D.NOMBRE_MEJOR}: {fmt(100 * rx[a - 1, b - 1], 1)} % y {fmt(100 * rx[b - 1, a - 1], 1)} %.", className="text-muted")]


@callback(Output("explicacion-confusion", "children"),
              Input("fig-confusion", "clickData"), Input("fig-solapamiento", "clickData"))
def cb_explicacion(click_conf, click_sol):
    try:
        disparador = ctx.triggered_id
        if disparador == "fig-solapamiento" and click_sol:
            a, b = click_sol["points"][0]["customdata"][:2]
            return explicar_par(int(a), int(b))
        if disparador == "fig-confusion" and click_conf:
            p = click_conf["points"][0]
            a = int(str(p["y"]).split()[0])
            b = int(str(p["x"]).split()[0])
            return explicar_par(a, b)
    except Exception:
        pass
    return explicar_par(1, 2)


def _tabla_heatmap(opt, metrica):
    df = D.APLIC.copy()
    df["tiempo_min"] = df["tiempo_total_s"] / 60
    if opt == "mejor":
        df = df.loc[df.groupby(["modelo", "balanceo"])["f1_macro_media"].idxmax()]
    else:
        df = df[df["optimizacion"] == opt]
    return df.pivot(index="modelo", columns="balanceo", values=metrica), df


@callback(Output("fig-heatmap", "figure"), Input("opt-heatmap", "value"), Input("metrica-heatmap", "value"))
def cb_heatmap(opt, metrica):
    piv, _ = _tabla_heatmap(opt, metrica)
    filas = [m for m in D.MODELOS if m in piv.index]
    piv = piv.reindex(index=filas, columns=list(D.BALANCEOS))
    z = piv.to_numpy()
    if metrica == "tiempo_min":
        texto = [["N/A" if np.isnan(v) else f"{v:.0f}" for v in fila] for fila in z]
        escala = [[0, "#e3f2fd"], [1, "#0d47a1"]]
        zz = np.log10(z)
        barra = dict(title="min (log)", tickvals=[0, 1, 2], ticktext=["1", "10", "100"])
    else:
        texto = [["N/A" if np.isnan(v) else f"{v:.3f}" for v in fila] for fila in z]
        escala = [[0, "#fff3e0"], [0.4, "#f6c344"], [0.7, "#4fc38a"], [1, "#0b4f34"]]
        zz, barra = z, dict(title="F1" if metrica.startswith("f1") else "Acc.")
    fig = go.Figure(go.Heatmap(
        z=zz, x=[D.BALANCEOS[b] for b in piv.columns], y=[D.MODELOS[m] for m in piv.index], text=texto,
        texttemplate="%{text}", textfont=dict(size=13), xgap=3, ygap=3, colorscale=escala, colorbar=barra,
        customdata=[[[m, b] for b in piv.columns] for m in piv.index],
        hovertemplate="<b>%{y}</b> · %{x}<br>%{text}<extra>clic para ver el detalle</extra>"))
    nombre = "el mejor optimizador" if opt == "mejor" else D.OPTIMIZADORES[opt]
    fig.update_layout(height=430, title=f"Cada celda: una combinación con {nombre} · clic 👆",
                      yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(side="top", showgrid=False))
    return fig


@callback(Output("detalle-combinacion", "children"),
              Input("fig-heatmap", "clickData"), Input("opt-heatmap", "value"))
def cb_detalle(click, opt):
    modelo, balanceo = "XGB", "ninguno"
    try:
        if click and ctx.triggered_id == "fig-heatmap":
            punto = click["points"][0]
            modelo = {v: k for k, v in D.MODELOS.items()}[punto["y"]]
            balanceo = {v: k for k, v in D.BALANCEOS.items()}[punto["x"]]
    except Exception:
        pass
    filas = D.TABLA[(D.TABLA["modelo"] == modelo) & (D.TABLA["balanceo"] == balanceo)]
    if not filas["aplicable"].any():
        return html.P(f"{D.MODELOS[modelo]} no admite {D.BALANCEOS[balanceo].lower()}: la combinación no es aplicable.")
    filas = filas[filas["aplicable"]].copy()
    ranking = D.APLIC["f1_macro_media"].rank(ascending=False, method="min")
    cab = html.Tr([html.Th("Optimizador"), html.Th("F1 macro"), html.Th("Pliegues"), html.Th("Tiempo"), html.Th("Puesto")])
    cuerpo = []
    mejor = filas["f1_macro_media"].max()
    for _, f in filas.sort_values("f1_macro_media", ascending=False).iterrows():
        pl = " · ".join(fmt(v) for v in f["f1_pliegues"])
        cuerpo.append(html.Tr([html.Td(D.OPTIMIZADORES[f["optimizacion"]]), html.Td(fmt(f["f1_macro_media"]), className="num"),
                               html.Td(pl, className="num"), html.Td(f"{f['tiempo_total_s'] / 60:.0f} min", className="num"),
                               html.Td(f"{int(ranking[f.name])} / {len(D.APLIC)}", className="num")],
                              className="resaltada" if f["f1_macro_media"] == mejor else ""))
    top = filas.loc[filas["f1_macro_media"].idxmax()]
    hp = top["hiper"][0] if top["hiper"] else {}
    hp_txt = ", ".join(f"{k} = {('sin límite' if v is None else (fmt(v, 3).rstrip('0').rstrip(',') if isinstance(v, float) else v))}" for k, v in hp.items())
    return [
        html.H5(f"{D.MODELOS[modelo]} · {D.BALANCEOS[balanceo]}", style={"fontWeight": 900, "color": "#1b6f5c"}),
        html.Table([html.Thead(cab), html.Tbody(cuerpo)], className="tabla-bosque mb-3"),
        html.P([html.B("Hiperparámetros elegidos (mejor optimizador, pliegue 1): "), hp_txt or "—"]),
        html.P([html.B("¿Cómo leer los pliegues? "), "Son los F1 del bucle externo de la validación anidada: tres estimaciones "
                "independientes del desempeño. Si varían poco, el resultado es estable."], className="small text-muted mb-0"),
    ]


@callback(Output("fig-dispersion", "figure"), Input("modelos-dispersion", "value"))
def cb_dispersion(modelos):
    df = D.APLIC[D.APLIC["modelo"].isin(modelos or [])]
    if df.empty:
        return vacio("Selecciona al menos un algoritmo", 420)
    simbolos = {"ninguno": "circle", "smote": "diamond", "adasyn": "square", "balanced": "triangle-up"}
    fig = go.Figure()
    for m in [k for k in D.MODELOS if k in modelos]:
        g = df[df["modelo"] == m]
        fig.add_scatter(x=g["tiempo_total_s"] / 60, y=g["f1_macro_media"], mode="markers", name=D.MODELOS[m],
                        marker=dict(color=D.COLOR_MODELO[m], size=11, symbol=[simbolos[b] for b in g["balanceo"]],
                                    line=dict(color="white", width=1)),
                        customdata=np.c_[g["balanceo"].map(D.BALANCEOS), g["optimizacion"].map(D.OPTIMIZADORES)],
                        hovertemplate=f"<b>{D.MODELOS[m]}</b><br>%{{customdata[0]}} · %{{customdata[1]}}<br>F1 macro: %{{y:.3f}}"
                                      "<br>Tiempo: %{x:.1f} min<extra></extra>")
    fig.update_layout(height=420, title="● sin balanceo · ◆ SMOTE · ■ ADASYN · ▲ pesos de clase",
                      xaxis=dict(type="log", title="Tiempo total de la validación anidada (min, escala log)"),
                      yaxis=dict(title="F1 macro (media de 3 pliegues)"), legend=dict(orientation="h", y=-0.2, x=0))
    return fig


@callback(Output("fig-balanceo", "figure"), Input("opt-heatmap", "value"))
def cb_balanceo(opt):
    piv, _ = _tabla_heatmap(opt, "f1_macro_media")
    piv = piv.reindex(index=[m for m in D.MODELOS if m in piv.index])
    fig = go.Figure()
    for b, color in [("smote", "#1e88e5"), ("adasyn", "#8e5ea2"), ("balanced", "#e0a526")]:
        delta = piv[b] - piv["ninguno"]
        fig.add_bar(y=[D.MODELOS[m] for m in piv.index], x=delta, orientation="h", name=D.BALANCEOS[b], marker_color=color,
                    hovertemplate=f"<b>%{{y}}</b><br>{D.BALANCEOS[b]}: %{{x:+.3f}} de F1 frente a no balancear<extra></extra>")
    fig.add_vline(x=0, line_color="#1f2d27", line_width=1.5)
    fig.update_layout(barmode="group", height=420, title="Δ F1 macro frente a «sin balanceo»",
                      xaxis=dict(title="Diferencia de F1 macro"), yaxis=dict(autorange="reversed", showgrid=False),
                      legend=dict(orientation="h", y=-0.2, x=0))
    return fig


@callback(Output("fig-anytime", "figure"), Input("metodos-anytime", "value"), Input("eje-anytime", "value"))
def cb_anytime(metodos, eje):
    if not metodos:
        return vacio("Selecciona al menos un método", 420)
    fig = go.Figure()
    for metodo in metodos:
        g = D.ANYTIME[D.ANYTIME["metodo"] == metodo]
        piv = g.pivot(index="evaluacion", columns="semilla", values="mejor_hasta_ahora")
        x = piv.index if eje == "evaluacion" else g.groupby("evaluacion")["tiempo_acumulado_s"].mean().reindex(piv.index)
        color = D.COLOR_OPT_NOMBRE[metodo]
        fig.add_scatter(x=list(x) + list(x)[::-1], y=list(piv.max(axis=1)) + list(piv.min(axis=1))[::-1], fill="toself",
                        fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=x, y=piv.mean(axis=1), mode="lines", name=metodo, line=dict(color=color, width=3),
                        hovertemplate=f"<b>{metodo}</b><br>%{{x:.0f}}<br>Mejor F1 hasta ahora: %{{y:.4f}}<extra></extra>")
    fig.update_layout(height=420, title="Mejor F1 macro encontrado hasta cada momento (media de 5 semillas)",
                      xaxis=dict(title="Evaluaciones realizadas" if eje == "evaluacion" else "Tiempo acumulado (s)"),
                      yaxis=dict(title="Mejor F1 macro hasta ahora"), legend=dict(orientation="h", y=-0.2, x=0))
    return fig


@callback(Output("fig-optimizadores", "figure"), Input("metrica-opt", "value"))
def cb_optimizadores(metrica):
    g = D.APLIC.groupby("optimizacion")[metrica].agg(["mean", "median"]).reindex(list(D.OPTIMIZADORES))
    if metrica == "tiempo_total_s":
        g = g / 60
    etiqueta = {"f1_macro_media": "F1 macro medio", "tiempo_total_s": "Tiempo medio por combinación (min)",
                "evaluaciones_totales": "Evaluaciones por combinación"}[metrica]
    d = 3 if metrica == "f1_macro_media" else 0
    fig = go.Figure(go.Bar(x=[D.OPTIMIZADORES[o] for o in g.index], y=g["mean"], text=[fmt(v, d) for v in g["mean"]],
                           textposition="outside", marker_color=[D.COLOR_OPT[o] for o in g.index],
                           hovertemplate="<b>%{x}</b><br>" + etiqueta + ": %{y:.3f}<extra></extra>"))
    fig.update_layout(height=420, title=etiqueta + " (108 combinaciones)", yaxis=dict(title=etiqueta,
                      range=[0, g["mean"].max() * 1.2]), xaxis=dict(showgrid=False))
    return fig


@callback(Output("fig-escala", "figure"), Input("modelos-escala", "value"))
def cb_escala(modelos):
    if not modelos:
        return vacio("Selecciona al menos un algoritmo", 430)
    paleta = ["#0f9d58", "#1e88e5", "#c8a13a", "#1b6f5c", "#8e5ea2", "#d9643a", "#00acc1"]
    colores = dict(zip(D.ESC["modelo"].unique(), paleta))
    exp = D.ESC_EXP.set_index("modelo")["exponente_empirico"]
    fig = go.Figure()
    for m in modelos:
        g = D.ESC[D.ESC["modelo"] == m]
        fig.add_scatter(x=g["n"], y=g["tiempo_s"], mode="lines+markers", name=f"{m} (b = {fmt(exp[m], 2)})",
                        line=dict(color=colores[m], width=3), marker=dict(size=8),
                        hovertemplate=f"<b>{m}</b><br>n = %{{x:,}}<br>%{{y:.3f}} s<extra></extra>")
    fig.update_layout(height=430, title="Tiempo de entrenamiento (o de predicción en KNN) según el tamaño de la muestra",
                      xaxis=dict(type="log", title="Celdas de entrenamiento (escala log)"),
                      yaxis=dict(type="log", title="Segundos (escala log)"), legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)))
    return fig


NOTAS_COMP = {
    "KNN (k = 5)": "Los árboles KD y Ball Tree aceleran la búsqueda de vecinos solo en pocas dimensiones. Con 57 variables pierden su ventaja y la fuerza bruta vectorizada es la más rápida (KD-Tree 4,5 veces más lento; Ball Tree 24 veces).",
    "Naive Bayes": "partial_fit procesa los datos por lotes: mismo resultado y mucha menos memoria (≈ 7,6 veces menos). Útil cuando los datos no caben en memoria.",
    "Regresión logística (L2)": "SAGA, un optimizador estocástico, converge unas 4 veces más rápido que L-BFGS con una tolerancia algo más laxa, sin perder F1.",
    "XGBoost (200 árboles)": "El método hist agrupa cada variable en histogramas y evita evaluar todos los cortes posibles: ≈ 12 veces más rápido que el método exact con un F1 prácticamente igual. La parada temprana ahorra árboles innecesarios.",
    "SVM": "LinearSVC y SGD escalan de forma lineal, pero pierden la frontera curva del núcleo RBF: el F1 macro cae de 0,65 a ≈ 0,50. Más rápido no siempre es mejor.",
}


@callback(Output("fig-comp", "figure"), Output("nota-comp", "children"), Input("algoritmo-comp", "value"))
def cb_comp(algoritmo):
    g = D.COMP[D.COMP["modelo"] == algoritmo]
    versiones = g["versión"].tolist()
    fig = go.Figure()
    fig.add_bar(x=versiones, y=g["tiempo_entrenamiento_s"], name="Entrenamiento (s)", marker_color="#1e88e5",
                text=[fmt(v, 2) for v in g["tiempo_entrenamiento_s"]], textposition="outside", cliponaxis=False)
    fig.add_bar(x=versiones, y=g["tiempo_inferencia_s"], name="Inferencia (s)", marker_color="#e0a526",
                text=[fmt(v, 2) for v in g["tiempo_inferencia_s"]], textposition="outside", cliponaxis=False)
    fig.add_scatter(x=versiones, y=g["F1_macro"], name="F1 macro", yaxis="y2", mode="markers+text",
                    marker=dict(size=16, color="#0f9d58", symbol="diamond"), text=[fmt(v) for v in g["F1_macro"]],
                    textposition="top center")
    fig.update_layout(barmode="group", height=430, title="Tiempo (barras, escala log) y F1 macro (rombos)", margin=dict(t=70),
                      yaxis=dict(type="log", title="Segundos"), yaxis2=dict(overlaying="y", side="right", range=[0, 1.05],
                                                                             title="F1 macro", showgrid=False),
                      xaxis=dict(showgrid=False), legend=dict(orientation="h", y=-0.25, x=0))
    texto = [html.B("💡 Interpretación: "), NOTAS_COMP.get(algoritmo, ""), " ",
             html.A("Profundiza en la sección 9.2 del Jupyter Book →", href=LIBRO + "10_conclusiones_finales.html", target="_blank")]
    return fig, texto

