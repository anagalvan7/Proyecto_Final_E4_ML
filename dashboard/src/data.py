"""Carga de datos (una sola vez al iniciar), constantes y plantilla grafica del dashboard.

Todo lo que se muestra sale de la carpeta datos/, generada por preparar_datos.py a partir
de la misma partición, el mismo preprocesamiento y los mismos resultados del libro.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

DATOS = Path(__file__).resolve().parents[1] / "datos"
ARTEFACTOS = Path(__file__).resolve().parents[1] / "artifacts"
RES = DATOS / "resultados"
LIBRO = "https://anagalvan7.github.io/Proyecto_Final_E4_ML/"

# ------------------------------------------------------------------ constantes
CLASES = list(range(1, 8))
COLOR = {1: "#1b6f5c", 2: "#4c9a2a", 3: "#c8a13a", 4: "#b0413e", 5: "#e07b39", 6: "#5b6fb3", 7: "#8e5ea2"}
EMOJI = {1: "🌲", 2: "🌲", 3: "🌳", 4: "💧", 5: "🍂", 6: "🌲", 7: "❄️"}
CORTO = {1: "1 Abeto/pícea", 2: "2 P. lodgepole", 3: "3 P. ponderosa", 4: "4 Álamo/sauce",
         5: "5 Á. temblón", 6: "6 A. Douglas", 7: "7 Krummholz"}
LARGO = {1: "Abeto/pícea (Spruce/Fir)", 2: "Pino lodgepole", 3: "Pino ponderosa",
         4: "Álamo/sauce (Cottonwood/Willow)", 5: "Álamo temblón (Aspen)", 6: "Abeto de Douglas", 7: "Krummholz"}
ETIQ = {c: f"{EMOJI[c]} {CORTO[c]}" for c in CLASES}
AREAS = ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]
COLOR_AREA = {"Rawah": "#1e88e5", "Neota": "#8e5ea2", "Comanche Peak": "#0f9d58", "Cache la Poudre": "#d9643a"}

VARIABLES = {
    "Elevation": "Elevación (m)",
    "Aspect": "Orientación (° azimut)",
    "Slope": "Pendiente (°)",
    "Horizontal_Distance_To_Hydrology": "Dist. horizontal a agua (m)",
    "Vertical_Distance_To_Hydrology": "Dist. vertical a agua (m)",
    "Horizontal_Distance_To_Roadways": "Dist. a carreteras (m)",
    "Hillshade_9am": "Sombreado 9 a. m. (0-255)",
    "Hillshade_Noon": "Sombreado mediodía (0-255)",
    "Hillshade_3pm": "Sombreado 3 p. m. (0-255)",
    "Horizontal_Distance_To_Fire_Points": "Dist. a puntos de incendio (m)",
}
# Nombres de las variables después de la ingeniería (coeficientes y SHAP)
NOMBRE_ING = {
    "Elevation": "Elevación (m)", "Slope": "Pendiente (°)",
    "Horizontal_Distance_To_Hydrology": "log Dist. horizontal a agua",
    "Vertical_Distance_To_Hydrology": "Dist. vertical a agua (m)",
    "Horizontal_Distance_To_Roadways": "log Dist. a carreteras",
    "Hillshade_9am": "Sombreado 9 a. m. (0-255)", "Hillshade_Noon": "Sombreado mediodía (0-255)",
    "Hillshade_3pm": "Sombreado 3 p. m. (0-255)",
    "Horizontal_Distance_To_Fire_Points": "log Dist. a puntos de incendio",
    "Aspect_sin": "sen(orientación) → este", "Aspect_cos": "cos(orientación) → norte",
    "Dist_Hydrology_Euclid": "Dist. euclidiana a agua", "Elevation_Minus_VDH": "Elevación relativa al agua",
}
for i, a in enumerate(AREAS, 1):
    NOMBRE_ING[f"Wilderness_Area_{i}"] = f"Área: {a}"
for i in range(1, 41):
    NOMBRE_ING[f"Soil_Type_{i}"] = f"Suelo {i}"


def grupo_variable(v):
    if v.startswith("Wilderness"):
        return "Área silvestre"
    if v.startswith("Soil"):
        return "Tipo de suelo"
    return "Continua"


MODELOS = {"XGB": "XGBoost", "RF": "Random Forest", "KNN": "K vecinos más cercanos", "DT": "Árbol de decisión",
           "SVM": "SVM (núcleo RBF)", "LR": "Regresión logística", "NB": "Naive Bayes gaussiano"}
COLOR_MODELO = {"XGB": "#0f9d58", "RF": "#1b6f5c", "KNN": "#1e88e5", "DT": "#c8a13a", "SVM": "#8e5ea2",
                "LR": "#d9643a", "NB": "#9e9e9e"}
BALANCEOS = {"ninguno": "Sin balanceo", "smote": "SMOTE", "adasyn": "ADASYN", "balanced": "Pesos de clase"}
OPTIMIZADORES = {"grid": "Grid Search", "random": "Random Search", "bayes": "Bayesiana (TPE)", "genetico": "Genético (DEAP)"}
COLOR_OPT = {"grid": "#1e88e5", "random": "#e0a526", "bayes": "#0f9d58", "genetico": "#8e5ea2"}
COLOR_OPT_NOMBRE = {"Grid Search": "#1e88e5", "Random Search": "#e0a526", "Bayesiana (TPE)": "#0f9d58",
                    "Genético (DEAP)": "#8e5ea2"}


def fmt(x, d=3):
    """Número con coma decimal y punto de miles (convención del libro)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def miles(x):
    return f"{int(round(x)):,}".replace(",", ".")


# ------------------------------------------------------------------ plantilla gráfica
_t = go.layout.Template(pio.templates["plotly_white"])
_t.layout.update(
    font=dict(family="Nunito, Segoe UI, sans-serif", size=13, color="#1f2d27"),
    title=dict(x=0.01, font=dict(size=16, color="#1b6f5c")),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fbfdfb",
    hoverlabel=dict(bgcolor="white", bordercolor="#0f9d58", font=dict(family="Nunito, Segoe UI, sans-serif", size=13)),
    xaxis=dict(gridcolor="#e3ece5", linecolor="#cfdcd3", zerolinecolor="#cfdcd3", zerolinewidth=2, automargin=True,
               title=dict(standoff=15)),
    yaxis=dict(gridcolor="#e3ece5", linecolor="#cfdcd3", zerolinecolor="#cfdcd3", zerolinewidth=2, automargin=True,
               title=dict(standoff=15)),
    legend=dict(bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=10, r=20, t=50, b=10),
)
pio.templates["bosque"] = _t
pio.templates.default = "bosque"
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def vacio(texto, alto=380):
    fig = go.Figure()
    fig.add_annotation(text=texto, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=15, color="#4f6159"))
    fig.update_layout(height=alto, xaxis_visible=False, yaxis_visible=False)
    return fig


# ------------------------------------------------------------------ datos: Parte I
EDA = pd.read_csv(DATOS / "eda_muestra.csv.gz")
with open(DATOS / "modelo_base.json", encoding="utf-8") as f:
    BASE = json.load(f)
CM_LR = np.array(BASE["confusion"])                 # filas = real, columnas = predicha
N_TEST_CLASE = CM_LR.sum(axis=1)
CONTEO_TRAIN = {int(k): v for k, v in BASE["conteo_train"].items()}

# ------------------------------------------------------------------ datos: Parte II
TABLA = pd.read_csv(RES / "tabla_maestra.csv", encoding="utf-8-sig")
TABLA["f1_pliegues"] = TABLA["f1_macro_pliegues"].apply(lambda s: json.loads(s) if isinstance(s, str) else None)
TABLA["hiper"] = TABLA["mejores_hiperparametros_por_pliegue"].apply(lambda s: json.loads(s) if isinstance(s, str) else None)
APLIC = TABLA[TABLA["aplicable"]].copy()

FIN_MET = pd.read_csv(RES / "finalista_metricas.csv", encoding="utf-8-sig").set_index("modelo")
FIN_IC = pd.read_csv(RES / "finalista_bootstrap_ic.csv", encoding="utf-8-sig")
FIN_CM_NORM = pd.read_csv(RES / "finalista_matriz_confusion.csv", encoding="utf-8-sig", index_col=0).to_numpy()
AREAS_NV = pd.read_csv(RES / "finalista_areas_no_vistas.csv", encoding="utf-8-sig")
DELONG = pd.read_csv(RES / "delong_finalistas.csv", encoding="utf-8-sig")
PERFIL = pd.read_csv(RES / "xgb_perfilamiento.csv", encoding="utf-8-sig")
SEMILLAS = pd.read_csv(RES / "xgb_sensibilidad_semilla.csv", encoding="utf-8-sig")
with open(RES / "resumen_finalista.json", encoding="utf-8") as f:
    RESUMEN = json.load(f)
with open(RES / "friedman_resumen.json", encoding="utf-8") as f:
    FRIEDMAN = json.load(f)
with open(RES / "finalista_calibracion_resumen.json", encoding="utf-8") as f:
    CALIB = json.load(f)
ANYTIME = pd.read_csv(RES / "anytime_historial.csv", encoding="utf-8-sig")
ANYTIME_RES = pd.read_csv(RES / "anytime_resumen.csv", encoding="utf-8-sig")
MULTI = pd.read_csv(RES / "multifidelidad_historial.csv", encoding="utf-8-sig")
with open(RES / "multifidelidad_resumen.json", encoding="utf-8") as f:
    MULTI_RES = json.load(f)
with open(RES / "shap_lime_casos.json", encoding="utf-8") as f:
    LIME = json.load(f)
COMP = pd.read_csv(RES / "computacional_estandar_vs_optimizado.csv", encoding="utf-8-sig")
ESC = pd.read_csv(RES / "escalamiento_tiempos.csv", encoding="utf-8-sig")
ESC_EXP = pd.read_csv(RES / "escalamiento_exponentes.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------ datos: Entregable 3
E3 = DATOS / "e3"
with open(E3 / "holdout_resumen.json", encoding="utf-8") as f:
    HOLDOUT = {r["modelo"]: r for r in json.load(f)}
with open(E3 / "estadistica_veredicto.json", encoding="utf-8") as f:
    VEREDICTO = json.load(f)
EST_RESUMEN = pd.read_csv(E3 / "estadistica_resumen.csv", encoding="utf-8-sig")
EST_PAREADAS = pd.read_csv(E3 / "estadistica_pareadas.csv", encoding="utf-8-sig")
EST_ABLACION = pd.read_csv(E3 / "estadistica_ablacion.csv", encoding="utf-8-sig")
with open(E3 / "estadistica_veredicto_completo.json", encoding="utf-8") as f:
    VEREDICTO_COMPLETO = json.load(f)
EST_RESUMEN_COMPLETO = pd.read_csv(E3 / "estadistica_resumen_completo.csv", encoding="utf-8-sig")
EST_PAREADAS_COMPLETO = pd.read_csv(E3 / "estadistica_pareadas_completo.csv", encoding="utf-8-sig")
with open(E3 / "diagnosticos.json", encoding="utf-8") as f:
    DIAG = json.load(f)
with open(E3 / "shap_global.json", encoding="utf-8") as f:
    SHAP_GLOBAL = json.load(f)
CONVERGENCIA = pd.read_csv(E3 / "convergencia_optuna.csv", encoding="utf-8-sig")
DATASETS_AD = pd.read_csv(E3 / "datasets_adicionales.csv", encoding="utf-8-sig")
ROBUSTEZ = pd.read_csv(E3 / "robustez.csv", encoding="utf-8-sig") if (E3 / "robustez.csv").exists() else None
FONDO_SHAP = pd.read_csv(E3 / "fondo_shap.csv")
MEJOR = max((m for m in HOLDOUT if m not in ("ECLC", "HGB")), key=lambda m: HOLDOUT[m]["f1_cv_media"])
NOMBRE_MEJOR = HOLDOUT[MEJOR]["nombre"]
CM_MEJOR = pd.read_csv(E3 / f"holdout_confusion_{MEJOR}.csv", index_col=0).to_numpy()

# (Los valores SHAP del XGBoost del Entregable 2 ya no se cargan: ese modelo dejo de ser el finalista.)


# ------------------------------------------------------------------ métricas derivadas
def por_clase(cm):
    tp = np.diag(cm).astype(float)
    rec = tp / cm.sum(axis=1)
    pre = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros_like(tp), where=(pre + rec) > 0)
    return pd.DataFrame({"Recall": rec, "Precisión": pre, "F1": f1}, index=CLASES)


PC_LR = por_clase(CM_LR)

# Solapamiento altitudinal entre pares de coberturas (sección 3.7 del libro)
SOLAPE = {
    (1, 2): 0.5172, (1, 3): 0.0402, (1, 4): 0.0002, (1, 5): 0.1468, (1, 6): 0.0430, (1, 7): 0.3309,
    (2, 3): 0.1750, (2, 4): 0.0102, (2, 5): 0.4979, (2, 6): 0.1966, (2, 7): 0.1043,
    (3, 4): 0.4799, (3, 5): 0.1677, (3, 6): 0.8676, (3, 7): 0.0006,
    (4, 5): 0.0043, (4, 6): 0.3844, (4, 7): 0.0,
    (5, 6): 0.1893, (5, 7): 0.0039, (6, 7): 0.0010,
}
