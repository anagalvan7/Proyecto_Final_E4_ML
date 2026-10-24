"""Seccion 1: Contexto. Narrativa estatica: problema, pregunta, datos, metodologia y limites."""
import dash
import dash_bootstrap_components as dbc
from dash import html

from src import data as D
from src.contexto_contenido import AREAS_INFO, ESPECIES
from src.data import COLOR, EMOJI, LARGO, LIBRO
from src.plots import fig_composicion, fig_gradiente
from src.ui import grafica, icono, kpi, nota, tarjeta

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


def heroe():
    return html.Div([
        html.Div(dbc.Row([
            dbc.Col([
                html.H1("🌲 Clasificación de la cobertura forestal en el Roosevelt National Forest"),
                html.Div("¿Qué especie domina cada celda de 30 × 30 m, sabiendo solo su cartografía?", className="subtitulo"),
                html.Div([dbc.Badge("👩‍🔬 Ana Galván", color="light", text_color="success"),
                          dbc.Badge("👩‍🔬 Valeria Oñate", color="light", text_color="primary"),
                          dbc.Badge("👩‍🔬 María José Ruiz", color="light", text_color="danger")], className="mt-2"),
            ], lg=9),
            dbc.Col(dbc.Button([icono("mdi:book-open-variant", 20), "Jupyter Book"], href=LIBRO, target="_blank",
                               className="boton-libro"), lg=3,
                    className="d-flex align-items-center justify-content-lg-end mt-3 mt-lg-0"),
        ]), className="hero mb-4"),
        dbc.Row([
            dbc.Col(kpi("🗺️ Celdas de 30 × 30 m", "581.012", "sin valores faltantes", 1), xs=6, md=True, className="mb-3"),
            dbc.Col(kpi("📐 Predictoras", "54", "10 continuas + 44 indicadoras", 2), xs=6, md=True, className="mb-3"),
            dbc.Col(kpi("🌲 Coberturas", "7", "clases a predecir", 3), xs=6, md=True, className="mb-3"),
            dbc.Col(kpi("📚 Referencia original", "70,6 %", "exactitud de la red neuronal de Blackard y Dean (1999)", 5),
                    xs=6, md=True, className="mb-3"),
            dbc.Col(kpi("🧪 Modelo original (ECLC)", D.fmt(D.VEREDICTO_COMPLETO["f1_eclc_prueba_reservada"]),
                        f"F1 macro en la prueba reservada con datos completos vs. "
                        f"{D.fmt(D.VEREDICTO_COMPLETO['f1_mejor_benchmark'])} del mejor benchmark", 4),
                    xs=12, md=True, className="mb-3"),
        ], className="g-3"),
        html.Img(src="/assets/banner_montana.svg", className="banner mb-4",
                 alt="Ilustración de los pisos altitudinales de las Montañas Rocosas, del bosque ribereño al krummholz"),
    ])


def pregunta_y_datos():
    filas = [html.Tr([html.Td(html.Code(a)), html.Td(b), html.Td(c), html.Td(d)]) for a, b, c, d in DICCIONARIO]
    return dbc.Row([
        dbc.Col(tarjeta("Pregunta de investigación y objetivos", "mdi:help-circle", "cab-lago", [
            html.P(html.B("¿Puede un ensamble que pondera a cada modelo según su desempeño local y por clase superar, "
                          "de forma estadísticamente significativa, a los mejores modelos del benchmark al predecir la "
                          "cobertura forestal a partir de variables cartográficas?")),
            html.Ol([
                html.Li("Corregir el benchmark del Entregable 2 para que las 112 combinaciones usen el mismo espacio de "
                        "búsqueda y el mismo presupuesto de optimización."),
                html.Li("Seleccionar con criterios estadísticos la mejor combinación de cada uno de los 7 modelos."),
                html.Li("Diseñar, implementar y probar un modelo original (ECLC) y validarlo con al menos tres pruebas de hipótesis."),
                html.Li("Poner los modelos a disposición de un usuario no experto en este dashboard."),
            ]),
            nota("Por qué importa: los inventarios forestales de campo son costosos en zonas remotas. Blackard y Dean (1999), "
                 "quienes publicaron estos datos, mostraron que un modelo cartográfico predice la cobertura con 70,6 % de "
                 "exactitud (red neuronal) frente a 58,4 % del análisis discriminante, lo que permite apoyar la "
                 "planificación forestal y el manejo de incendios sin visitar cada celda.", "01_base_datos.html", "1.1"),
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
                html.P(["Primero se construyó un ", html.B("modelo base"), " (regresión logística); luego se compararon ",
                        html.B("7 algoritmos × 4 estrategias de balanceo × 4 métodos de optimización"),
                        " y, finalmente, se diseñó un ", html.B("modelo original"), " que combina a los siete."]),
                html.Div([dbc.Badge("🎯 Clasificación multiclase", color="success", className="me-1 mb-1"),
                          dbc.Badge("⚖️ Desbalance 103 : 1", color="warning", className="me-1 mb-1"),
                          dbc.Badge("🤖 7 algoritmos · 108 combinaciones", color="info", className="me-1 mb-1"),
                          dbc.Badge(f"🏆 Mejor benchmark: {D.NOMBRE_MEJOR}", color="danger", className="me-1 mb-1"),
                          dbc.Badge("🧪 Modelo original: ECLC", color="primary", className="me-1 mb-1")]),
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
                 "En el Entregable 3 se repitieron 37 combinaciones que habían usado un espacio o presupuesto mayor.",
                 "06_benchmark_modelos.html", "5.1"),
        ], "mb-3"),
        tarjeta("Limitaciones, ética y sesgos", "mdi:alert-circle-outline", "cab-terracota", [
            html.Ul([
                html.Li("Las celdas vecinas se parecen (autocorrelación espacial): la partición aleatoria es optimista; "
                        "entrenar con 3 áreas y evaluar en la cuarta reduce mucho el desempeño."),
                html.Li("El benchmark y el modelo original se entrenan con una submuestra estratificada de 40.000 celdas "
                        "por costo computacional; las clases raras (álamo/sauce) tienen pocas observaciones."),
                html.Li("Los datos describen un solo bosque de Colorado en una fecha: no deben usarse para decidir sobre "
                        "otros ecosistemas sin validarlos allí."),
                html.Li("Un error en una clase rara (bosque ribereño) puede ocultar un ecosistema valioso; por eso se usa "
                        "F1 macro y no exactitud."),
                html.Li("El uso de herramientas de IA está declarado en el capítulo final del Jupyter Book."),
            ]),
        ], "mb-3"),
    ])


def layout():
    return html.Div([heroe(), pregunta_y_datos(), territorio(), metodologia_y_limites()], className="contenido-tab")
