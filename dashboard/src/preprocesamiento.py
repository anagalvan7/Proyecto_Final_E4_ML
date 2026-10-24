"""Ingenieria de variables del proyecto (copia de docs/_utils.engineer_features).

Vive dentro del paquete del dashboard para que los artefactos serializados con joblib
referencien src.preprocesamiento.engineer_features y se puedan cargar en Render sin el
resto del repositorio. El script que entrena los artefactos sustituye la funcion del
libro por esta antes de ajustar los pipelines (son identicas; tests/ lo verifica).
"""
import numpy as np

NUM_COLS = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am",
    "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
]
WILD_COLS = [f"Wilderness_Area_{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type_{i}" for i in range(1, 41)]
FEATURES = NUM_COLS + WILD_COLS + SOIL_COLS
LOG_COLS = ["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points"]


def engineer_features(X):
    """Orientacion circular (seno y coseno), log1p de distancias horizontales, distancia
    euclidiana al agua y elevacion relativa al agua. Sin parametros aprendidos."""
    X = X.copy()
    rad = np.deg2rad(X["Aspect"])
    X["Aspect_sin"] = np.sin(rad)
    X["Aspect_cos"] = np.cos(rad)
    X["Dist_Hydrology_Euclid"] = np.hypot(
        X["Horizontal_Distance_To_Hydrology"], X["Vertical_Distance_To_Hydrology"]
    )
    X["Elevation_Minus_VDH"] = X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    for col in LOG_COLS:
        X[col] = np.log1p(X[col])
    return X.drop(columns="Aspect")
