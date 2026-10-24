"""Componentes visuales comunes del dashboard (tarjetas, KPI, notas de interpretacion)."""
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_iconify import DashIconify

from .data import CONFIG, LIBRO


def icono(nombre, ancho=22):
    return DashIconify(icon=nombre, width=ancho, className="me-2")


def tarjeta(titulo, icon, cabecera, cuerpo, clase=""):
    return dbc.Card(
        [dbc.CardHeader([icono(icon), titulo], className=f"{cabecera} d-flex align-items-center"),
         dbc.CardBody(cuerpo)],
        className=f"tarjeta {clase}",
    )


def nota(texto, pagina=None, seccion=None):
    hijos = [html.B("💡 Interpretación: "), texto]
    if pagina:
        hijos += [" ", html.A(f"Profundiza en la sección {seccion} del Jupyter Book →",
                              href=LIBRO + pagina, target="_blank")]
    return html.Div(hijos, className="nota")


def kpi(etiqueta, valor, detalle, n):
    return html.Div([html.Div(etiqueta, className="etiqueta"), html.Div(valor, className="valor"),
                     html.Div(detalle, className="detalle")], className=f"kpi kpi-{n}")


def grafica(id_=None, figura=None):
    props = {"config": CONFIG}
    if id_:
        props["id"] = id_
    if figura is not None:
        props["figure"] = figura
    return dcc.Graph(**props)


def control(etiqueta, componente, clase=""):
    return html.Div([html.Div(etiqueta, className=f"etiqueta-control {clase}"), componente])


def radio(id_, opciones, valor):
    return dbc.RadioItems(id=id_, inline=True, value=valor,
                          options=[{"label": f" {l}", "value": v} for v, l in opciones])


def intro(*hijos):
    return html.Div(list(hijos), className="intro-seccion")

