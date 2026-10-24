"""Pruebas de humo del dashboard: las tres paginas se construyen y los callbacks principales
responden sin errores a traves del servidor Flask (como lo haria el navegador)."""
import json
import sys
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASH))


@pytest.fixture(scope="module")
def cliente():
    import os
    os.chdir(DASH)
    import app as aplicacion
    return aplicacion.app.server.test_client()


def llamar(cliente, salida, entradas, estados=()):
    """Dispara un callback como lo haria dash-renderer."""
    partes = [x for x in salida.strip(".").split("...")]
    outs = [{"id": x.rsplit(".", 1)[0], "property": x.rsplit(".", 1)[1]} for x in partes]
    if len(outs) == 1:
        outs = outs[0]
    else:
        salida = ".." + "...".join(partes) + ".."
    cuerpo = {"output": salida, "outputs": outs, "inputs": [{"id": i, "property": p, "value": v} for i, p, v in entradas],
              "state": [{"id": i, "property": p, "value": v} for i, p, v in estados],
              "changedPropIds": [f"{entradas[0][0]}.{entradas[0][1]}"]}
    r = cliente.post("/_dash-update-component", data=json.dumps(cuerpo), content_type="application/json")
    assert r.status_code in (200, 204), r.data[:500]
    return r


def test_paginas(cliente):
    for ruta in ["/", "/eda", "/modelos"]:
        assert cliente.get(ruta).status_code == 200
    capa = json.loads(cliente.get("/_dash-layout").data)
    assert capa


def test_layouts_de_paginas():
    import dash
    for p in dash.page_registry.values():
        lay = p["layout"]() if callable(p["layout"]) else p["layout"]
        assert lay is not None


FILTROS = [("f-areas", "value", ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]),
           ("f-clases", "value", [1, 2, 3, 4, 5, 6, 7]), ("f-elevacion", "value", [1859, 3858])]


def test_callbacks_eda(cliente):
    llamar(cliente, "fig-desbalance.figure...nota-desbalance.children", [("escala-desbalance", "value", "log")] + FILTROS)
    llamar(cliente, "fig-eta2.figure...nota-eta2.children", FILTROS)
    llamar(cliente, "fig-correlacion.figure...nota-correlacion.children", FILTROS)
    llamar(cliente, "fig-atipicos.figure...nota-atipicos.children", FILTROS)
    llamar(cliente, "fig-bivariado.figure...nota-bivariado.children",
           [("biv-x", "value", "Elevation"), ("biv-y", "value", "Slope"), ("biv-color", "value", "Area")] + FILTROS)
    # filtros que dejan la seleccion vacia no deben romper la pagina
    llamar(cliente, "fig-desbalance.figure...nota-desbalance.children",
           [("escala-desbalance", "value", "log"), ("f-areas", "value", []), ("f-clases", "value", []),
            ("f-elevacion", "value", [1859, 1860])])


@pytest.mark.parametrize("entrenamiento", ["completo", "submuestra"])
@pytest.mark.parametrize("modelo", ["ECLC", "RF", "KNN", "XGB", "SVM", "NB", "LR", "DT"])
def test_diagnostico(cliente, modelo, entrenamiento):
    llamar(cliente, "..kpis-diag.children...fig-cm-diag.figure...nota-cm-diag.children...fig-roc-diag.figure..."
                    "fig-pr-diag.figure...fig-cal-diag.figure...nota-cal-diag.children...fig-shap-diag.figure..."
                    "nota-shap-diag.children..", [("modelo-diag", "value", modelo), ("entrenamiento-diag", "value", entrenamiento)])


def _estados(valores):
    from src.preprocesamiento import NUM_COLS
    return [("modelo-pred", "value", valores["modelo"]), ("entrada-area", "value", 1), ("entrada-suelo", "value", 29)] + \
        [(json.dumps({"tipo": "entrada", "var": c}, separators=(",", ":"), sort_keys=True), "value", valores.get(c, 0))
         for c in NUM_COLS]


def test_prediccion_valida_y_invalida(cliente):
    import pandas as pd
    ej = pd.read_csv(DASH / "datos" / "e3" / "ejemplos.csv").iloc[0].to_dict()
    r = llamar(cliente, "..salida-pred.children...alerta-pred.children..", [("boton-predecir", "n_clicks", 1)],
               _estados({**ej, "modelo": "ECLC"}))
    assert b"Probabilidad" in r.data or b"probabilidad" in r.data
    malo = {**ej, "modelo": "RF", "Elevation": 99999}
    r = llamar(cliente, "..salida-pred.children...alerta-pred.children..", [("boton-predecir", "n_clicks", 2)],
               _estados(malo))
    assert "fuera del rango".encode() in r.data
