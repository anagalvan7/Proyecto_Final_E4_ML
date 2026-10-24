"""Dashboard: clasificacion de la cobertura forestal del Bosque Nacional Roosevelt.

Aplicacion multipagina (pages/contexto.py, pages/eda.py, pages/modelos.py).
Ejecutar en local:   python app.py            (http://127.0.0.1:8050)
Despliegue (Render): gunicorn app:server
"""
import dash
import dash_bootstrap_components as dbc
from dash import Dash, html

from src.data import LIBRO
from src.ui import icono, kpi

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap",
    ],
    title="Cobertura Forestal · Roosevelt National Forest",
    suppress_callback_exceptions=True,
)
server = app.server

VERSION = "4.0"
ACTUALIZADO = "24 de octubre de 2026"


# Encabezado, cifras clave y banner: comunes a las tres secciones (como pestañas de una sola vista)
encabezado = html.Div(dbc.Container(dbc.Row([
    dbc.Col([
        html.H1("🌲 Clasificación de Cobertura Forestal - Roosevelt National Forest"),
        html.Div("Curso Machine Learning · Proyecto final", className="subtitulo"),
        html.Div([dbc.Badge("👩‍🔬 Ana Galván", color="light", text_color="success"),
                  dbc.Badge("👩‍🔬 Valeria Oñate", color="light", text_color="primary"),
                  dbc.Badge("👩‍🔬 María José Ruiz", color="light", text_color="danger")], className="mt-2"),
    ], lg=9),
    dbc.Col(dbc.Button([icono("mdi:book-open-variant", 20), "Jupyter Book"], href=LIBRO, target="_blank",
                       className="boton-libro"), lg=3, className="d-flex align-items-center justify-content-lg-end mt-3 mt-lg-0"),
]), fluid="xl"), className="hero mb-4")

cifras = dbc.Row([
    dbc.Col(kpi("🗺️ Celdas de 30 × 30 m", "581.012", "sin valores faltantes", 1), xs=6, md=True, className="mb-3"),
    dbc.Col(kpi("📐 Predictoras", "54", "10 continuas + 44 indicadoras", 2), xs=6, md=True, className="mb-3"),
    dbc.Col(kpi("🌲 Coberturas", "7", "clases a predecir", 3), xs=6, md=True, className="mb-3"),
    dbc.Col(kpi("🏞️ Áreas silvestres", "4", "Roosevelt National Forest", 5), xs=6, md=True, className="mb-3"),
    dbc.Col(kpi("🏔️ Gradiente altitudinal", "≈ 2.000 m", "de 1.859 a 3.858 m s. n. m.", 4), xs=12, md=True, className="mb-3"),
], className="g-3")

# Navegación entre las tres páginas con el aspecto de pestañas; queda fija al desplazarse
pestanas = dbc.Nav([
    dbc.NavItem(dbc.NavLink("🌲 Contexto y Territorio", href="/", active="exact")),
    dbc.NavItem(dbc.NavLink("📊 Análisis Exploratorio", href="/eda", active="exact")),
    dbc.NavItem(dbc.NavLink("🤖 Modelos de Machine Learning", href="/modelos", active="exact")),
], className="nav-tabs pestanas-fijas")

app.layout = html.Div([
    encabezado,
    dbc.Container([
        cifras,
        html.Img(src="/assets/banner_montana.svg", className="banner mb-4", alt="Pisos altitudinales de las Montañas Rocosas"),
        pestanas,
        dash.page_container,
        html.Div([
            "Datos: Covertype, UCI Machine Learning Repository (Blackard y Dean, 1998; CC BY 4.0) · ",
            html.A("Informe técnico en Jupyter Book", href=LIBRO, target="_blank"),
            html.Br(),
            f"Ana Galván, Valeria Oñate y María José Ruiz · versión {VERSION} · actualizado el {ACTUALIZADO}",
            html.Br(), "Fotografía del encabezado: Jaye Travel Blog (2022), uso académico.",
        ], className="pie"),
    ], fluid="xl"),
])

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
