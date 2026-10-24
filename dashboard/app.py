"""Dashboard: clasificacion de la cobertura forestal del Bosque Nacional Roosevelt.

Aplicacion multipagina (pages/contexto.py, pages/eda.py, pages/modelos.py).
Ejecutar en local:   python app.py            (http://127.0.0.1:8050)
Despliegue (Render): gunicorn app:server
"""
import dash
import dash_bootstrap_components as dbc
from dash import Dash, html

from src.data import LIBRO
from src.ui import icono

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

VERSION = "3.0 (Entregable 3)"
ACTUALIZADO = "23 de octubre de 2026"

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("🌲 Covertype · Roosevelt National Forest", href="/", className="fw-bold"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
            dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact")),
            dbc.NavItem(dbc.NavLink([icono("mdi:book-open-variant", 18), "Jupyter Book"], href=LIBRO,
                                    target="_blank")),
        ], navbar=True, className="ms-auto"),
    ], fluid="xl"),
    color="#1b6f5c", dark=True, sticky="top", className="mb-3 barra-nav",
)

app.layout = html.Div([
    navbar,
    dbc.Container([
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
