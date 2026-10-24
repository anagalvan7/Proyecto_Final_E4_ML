"""Seccion 1: Contexto. Narrativa estatica: problema, pregunta, datos, metodologia y limites."""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src import data as D
from src.contexto_contenido import AREAS_INFO, ESPECIES
from src.data import COLOR, EMOJI, LARGO, LIBRO
from src.plots import fig_composicion, fig_gradiente
from src.ui import grafica, icono, nota, tarjeta

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Covertype")

DICCIONARIO = [
    ("Elevation", "Elevación", "m", "Continua"), ("Aspect", "Orientación (azimut)", "°", "Continua (circular)"),
    ("Slope", "Pendiente", "°", "Continua"),
    ("Horizontal_Distance_To_Hydrology", "Distancia horizontal al agua más cercana", "m", "Continua"),
    ("Vertical_Distance_To_Hydrology", "Distancia vertical al agua más cercana", "m", "Continua"),
    ("Horizontal_Distance_To_Roadways", "Distancia horizontal a la carretera más cercana", "m", "Continua"),
    ("Hillshade_9am / Noon / 3pm", "Índice de sombreado en el solsticio de verano", "0–255", "Continua (3 variables)"),
    ("Horizontal_Distance_To_Fire_Points", "Distancia a puntos de ignición de incendios", "m", "Continua"),
    ("Wilderness_Area_1…4", "Área silvestre (Rawah, Neota, Comanche Peak, Cache la Poudre)", "0/1", "Indicadora (4)"),
    ("Soil_Type_1…40", "Tipo de suelo (clasificación USFS ELU)", "0/1", "Indicadora (40)"),
    ("Cover_Type", "Cobertura forestal dominante (variable objetivo)", "1–7", "Nominal, 7 clases"),
]

ETAPAS = [
    ("1. ETL", "581.012 celdas, partición 80/20 estratificada, auditoría de fuga", "#1e88e5"),
    ("2. EDA", "Gradiente altitudinal, desbalance 103:1, autocorrelación espacial", "#0f9d58"),
    ("3. Benchmark", "7 modelos × 4 balanceos × 4 optimizadores = 112 combinaciones", "#c8a13a"),
    ("4. Validación anidada", "3 × 3 pliegues, mismo espacio y presupuesto para todas", "#d9643a"),
    ("5. Comparación", "Friedman → Nemenyi y tamaño del efecto", "#8e5ea2"),
    ("6. Modelo original", "ECLC: ensamble por competencia local y por clase", "#1b6f5c"),
    ("7. Validación del original", "30 pliegues pareados, 5 pruebas, Holm, Cliff", "#5b6fb3"),
]


def diagrama_pipeline():
    pasos = []
    for i, (t, d, c) in enumerate(ETAPAS):
        pasos.append(html.Div([html.Div(t, className="paso-titulo"), html.Div(d, className="paso-desc")],
                              className="paso", style={"borderTopColor": c}))
        if i < len(ETAPAS) - 1:
            pasos.append(html.Div("➜", className="paso-flecha", **{"aria-hidden": "true"}))
    return html.Div(pasos, className="pipeline", role="img",
                    **{"aria-label": "Diagrama del pipeline: ETL, EDA, benchmark, validación anidada, comparación "
                                     "estadística, modelo original y su validación"})


def pregunta_y_datos():
    filas = [html.Tr([html.Td(html.Code(a)), html.Td(b), html.Td(c), html.Td(d)]) for a, b, c, d in DICCIONARIO]
    return dbc.Row([
        dbc.Col(tarjeta("Pregunta de investigación y objetivos", "mdi:help-circle", "cab-lago", [
            html.P(html.B("¿Puede un ensamble que pondera a cada modelo según su desempeño local y por clase superar, "
                          "de forma estadísticamente significativa, a los mejores modelos del benchmark?")),
            html.Ol([
                html.Li("Comparar 7 algoritmos, 4 balanceos y 4 optimizadores en igualdad de condiciones."),
                html.Li("Diseñar un modelo original (ECLC) y validarlo con al menos tres pruebas de hipótesis."),
                html.Li("Poner los modelos al alcance de un usuario no experto."),
            ], className="mb-0"),
        ]), lg=5, className="mb-3"),
        dbc.Col(tarjeta("Datos y diccionario de variables", "mdi:table", "cab-azul", [
            html.P(["Fuente: ", html.B("UCI Machine Learning Repository, Covertype"), " (Blackard y Dean, 1998; CC BY 4.0). "
                    "Variables cartográficas derivadas de datos del U.S. Geological Survey y del U.S. Forest Service (Región 2); "
                    "la cobertura proviene del Resource Information System del USFS. Es una instantánea única, sin dimensión "
                    "temporal, publicada en 1998: 581.012 observaciones, 54 predictoras, tarea de ",
                    html.B("clasificación multiclase"), "."]),
            dbc.Table([html.Thead(html.Tr([html.Th("Variable"), html.Th("Descripción"), html.Th("Unidad"), html.Th("Tipo")])),
                       html.Tbody(filas)], size="sm", striped=True, responsive=True, className="tabla-dicc"),
        ]), lg=7, className="mb-3"),
    ])


def territorio():
    areas = dbc.Row([
        dbc.Col(html.Div([
            html.H5(n), dbc.Badge(exp, color="light", text_color="dark", className="mb-2"),
            html.Div(desc, className="mb-2"), html.Div([html.B(celdas), f" · {rango}"]), html.Div(extra),
        ], className="area", style={"background": f"linear-gradient(135deg, {col}, {col}bb)"}), md=6, className="mb-3")
        for n, exp, desc, celdas, rango, extra, col in AREAS_INFO
    ])
    especies = dbc.Row([
        dbc.Col(html.Div([
            html.H6(f"{EMOJI[c]} {LARGO[c]}"), html.Div(cient, className="cientifico mb-1"),
            dbc.Badge(piso, className="mb-2", style={"backgroundColor": COLOR[c]}), html.Div(desc),
        ], className="especie", style={"borderTopColor": COLOR[c]}), xl=3, lg=4, md=6, className="mb-3")
        for c, cient, piso, desc in ESPECIES
    ])
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta("¿Qué problema resolvemos?", "mdi:target", "cab-lago", [
                html.P(["Predecir la ", html.B("cobertura forestal dominante"), " de cada celda del terreno a partir solo de su ",
                        html.B("cartografía"), ": elevación, orientación, pendiente, distancias al agua, a carreteras y a puntos "
                        "de incendio, área silvestre y tipo de suelo."]),
                html.P(["Los inventarios de campo son costosos en zonas remotas; un modelo cartográfico apoya la ",
                        html.B("planificación forestal, la evaluación del riesgo de incendios y la conservación"),
                        ". Blackard y Dean (1999), quienes publicaron estos datos, alcanzaron 70,6 % de exactitud con una "
                        "red neuronal."]),
                html.P(["Primero se construyó un ", html.B("modelo base"), " (regresión logística) y luego se compararon ",
                        html.B("7 algoritmos × 4 estrategias de balanceo × 4 métodos de optimización"),
                        " para elegir el mejor modelo de forma estadísticamente defendible. Finalmente se diseñó un ",
                        html.B("modelo original (ECLC)"), " que combina a los siete."]),
                html.Div([dbc.Badge("🎯 Clasificación multiclase", color="success", className="me-1 mb-1"),
                          dbc.Badge("⚖️ Desbalance 103 : 1", color="warning", className="me-1 mb-1"),
                          dbc.Badge("🤖 7 algoritmos · 108 combinaciones", color="info", className="me-1 mb-1"),
                          dbc.Badge(f"🏆 Mejor benchmark: {D.NOMBRE_MEJOR}", color="danger", className="me-1 mb-1"),
                          dbc.Badge(f"🧪 ECLC: F1 macro {D.fmt(D.VEREDICTO_COMPLETO['f1_eclc_prueba_reservada'])}",
                                    color="primary", className="me-1 mb-1")]),
            ]), lg=4, className="mb-3"),
            dbc.Col(tarjeta("Gradiente altitudinal: cada especie en su piso", "mdi:image-filter-hdr", "cab-verde", [
                grafica(figura=fig_gradiente()),
                nota("La temperatura baja ≈ 6 °C por cada 1.000 m de ascenso y cada cobertura ocupa la franja que tolera. "
                     "Las barras gruesas de clases vecinas se solapan: esas transiciones serán las zonas de duda de los modelos.",
                     "02_eda.html", "2.3"),
            ]), lg=8, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Composición de cada área silvestre", "mdi:map-marker-radius", "cab-azul", [
                grafica(figura=fig_composicion()),
                nota("Cada área ocupa un tramo distinto del gradiente. Cache la Poudre es otro mundo: sin pícea/abeto ni "
                     "krummholz, y es la única con bosque ribereño de álamo/sauce.", "02b_fuga_estructura.html", "2.7"),
            ]), lg=5, className="mb-3"),
            dbc.Col([html.H4("🏞️ Las cuatro áreas silvestres", className="seccion-titulo"), areas], lg=7),
        ]),
        html.H4("🌲 Las siete coberturas forestales", className="seccion-titulo mt-2"),
        especies,
    ])


def metodologia_y_limites():
    return html.Div([
        tarjeta("Metodología: del dato al modelo original", "mdi:sitemap", "cab-verde", [
            diagrama_pipeline(),
            nota("Todas las combinaciones comparten la partición, la semilla (42), el preprocesamiento y los pliegues externos. "
                 "Cada combinación tuvo el mismo espacio de búsqueda y el mismo presupuesto de optimización.",
                 "06_benchmark_modelos.html", "5.1"),
        ], "mb-3"),
        tarjeta("Limitaciones, ética y sesgos", "mdi:alert-circle-outline", "cab-terracota", [
            html.Ul([
                html.Li("Las celdas vecinas se parecen (autocorrelación espacial): la partición aleatoria es optimista; "
                        "entrenar con 3 áreas y evaluar en la cuarta reduce mucho el desempeño."),
                html.Li("El benchmark se seleccionó con una submuestra estratificada de 40.000 celdas por costo computacional "
                        "(el modelo original se validó además con el entrenamiento completo); las clases raras (álamo/sauce) "
                        "tienen pocas observaciones."),
                html.Li("Los datos describen un solo bosque de Colorado en una fecha: no deben usarse para decidir sobre "
                        "otros ecosistemas sin validarlos allí."),
                html.Li("Un error en una clase rara (bosque ribereño) puede ocultar un ecosistema valioso; por eso se usa "
                        "F1 macro y no exactitud."),
                html.Li("El uso de herramientas de IA está declarado en el capítulo final del Jupyter Book."),
            ]),
        ], "mb-3"),
    ])


def layout():
    return html.Div([
        territorio(), pregunta_y_datos(), metodologia_y_limites(),
        html.Div(dbc.Button([icono("mdi:book-open-page-variant"), "Leer el informe completo en Jupyter Book"],
                            href=LIBRO, target="_blank", className="boton-libro mt-2"), className="text-center"),
    ], className="contenido-tab")
