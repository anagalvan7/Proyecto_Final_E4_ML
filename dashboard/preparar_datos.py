"""Genera los archivos livianos que consume el dashboard (carpeta datos/).

Se ejecuta una sola vez, desde la raíz del proyecto, después de compilar el libro:

    python dashboard/preparar_datos.py

Usa la misma partición, el mismo preprocesamiento y la misma semilla del libro
(docs/_utils.py) y copia los resultados del benchmark (resultados/). El dashboard no
necesita scikit-learn ni el CSV completo para funcionar: solo lee datos/.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "docs"))
import _utils as U  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "datos"
SALIDA.mkdir(exist_ok=True)

# Hiperparámetros elegidos en la sección 3.3 del libro
MEJOR_C = 10.0
MEJOR_ENG = True


def metricas(y, p):
    return {
        "Accuracy": accuracy_score(y, p),
        "Precisión macro": precision_score(y, p, average="macro", zero_division=0),
        "Recall macro": recall_score(y, p, average="macro", zero_division=0),
        "F1 macro": f1_score(y, p, average="macro"),
    }


def main():
    print("Cargando datos…")
    df, X, y = U.load_covertype()
    X_train, X_test, y_train, y_test = U.split_covertype(X, y)

    # ---------------- Muestra para las distribuciones del EDA ----------------
    Xs, ys = U.stratified_sample(X_train, y_train, 40_000)
    muestra = Xs[U.NUM_COLS].copy()
    muestra["Area"] = U.wilderness_label(Xs).values
    muestra["Clase"] = ys.values
    muestra.to_csv(SALIDA / "eda_muestra.csv.gz", index=False, compression="gzip")

    # ---------------- Modelo base: regresión logística ----------------
    print("Entrenando la regresión logística del capítulo 3…")
    modelo = U.build_logreg(engineered=MEJOR_ENG, C=MEJOR_C, max_iter=3000)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    pred_tr = modelo.predict(X_train)

    base = {"test": metricas(y_test, pred), "train": metricas(y_train, pred_tr)}

    rng = np.random.default_rng(U.RANDOM_STATE)
    yt, pt = y_test.to_numpy(), pred
    boot = {m: [] for m in base["test"]}
    for _ in range(200):
        i = rng.integers(0, len(yt), len(yt))
        for m, v in metricas(yt[i], pt[i]).items():
            boot[m].append(v)
    base["ic95"] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in boot.items()}

    dummies = {}
    for nombre, estr in [("Dummy (más frecuente)", "most_frequent"),
                         ("Dummy (uniforme)", "uniform"),
                         ("Dummy (estratificado)", "stratified")]:
        d = DummyClassifier(strategy=estr, random_state=U.RANDOM_STATE).fit(X_train, y_train)
        dummies[nombre] = metricas(y_test, d.predict(X_test))
    base["dummies"] = dummies

    cm = confusion_matrix(y_test, pred, labels=U.CLASES if hasattr(U, "CLASES") else list(range(1, 8)))
    base["confusion"] = cm.tolist()

    # Elevación mediana de cada tipo de error (para el explorador de confusiones)
    elev = X_test["Elevation"].to_numpy()
    med = {}
    for a in range(1, 8):
        for b in range(1, 8):
            m = (yt == a) & (pt == b)
            med[f"{a}-{b}"] = float(np.median(elev[m])) if m.any() else None
    base["elev_mediana_error"] = med

    # Coeficientes (continuas por desviación estándar) y razón de odds por +100 m
    clf = modelo.named_steps["clf"]
    prep = modelo.named_steps["prep"]
    escalador = prep[-1].named_transformers_["num"]
    nombres = list(U.ENG_NUM_COLS) + list(U.BIN_COLS)
    sd = dict(zip(escalador.feature_names_in_, escalador.scale_))
    coef = pd.DataFrame(clf.coef_, index=clf.classes_, columns=nombres)
    base["coeficientes"] = {int(c): coef.loc[c].round(5).to_dict() for c in coef.index}
    or100 = {}
    for c in coef.index:
        lo = coef.loc[c, "Elevation"] * 100 / sd["Elevation"] + coef.loc[c, "Elevation_Minus_VDH"] * 100 / sd["Elevation_Minus_VDH"]
        or100[int(c)] = float(np.exp(lo))
    base["or_100m"] = or100
    base["n_test"] = int(len(y_test))
    base["n_train"] = int(len(y_train))
    base["conteo_train"] = {int(k): int(v) for k, v in y_train.value_counts().sort_index().items()}

    with open(SALIDA / "modelo_base.json", "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=1)
    print("  F1 macro prueba:", round(base["test"]["F1 macro"], 4), "· accuracy:", round(base["test"]["Accuracy"], 4))

    # ---------------- Resultados de la Parte II ----------------
    res = RAIZ / "resultados"
    destino = SALIDA / "resultados"
    destino.mkdir(exist_ok=True)
    for nombre in [
        "tabla_maestra.csv", "friedman_resumen.json", "finalista_metricas.csv",
        "finalista_bootstrap_ic.csv", "finalista_matriz_confusion.csv",
        "finalista_areas_no_vistas.csv", "delong_finalistas.csv", "finalista_tamano_efecto.json",
        "finalista_calibracion_resumen.json", "finalista_perfilamiento.csv",
        "finalista_sensibilidad_semilla.csv", "resumen_finalista.json",
        "anytime_historial.csv", "anytime_resumen.csv", "multifidelidad_historial.csv",
        "multifidelidad_resumen.json", "shap_importancia_global_muestra.csv",
        "shap_lime_casos.json", "computacional_estandar_vs_optimizado.csv",
        "escalamiento_tiempos.csv", "escalamiento_exponentes.csv",
    ]:
        shutil.copy(res / nombre, destino / nombre)

    # SHAP: valores de la muestra de 3.000 celdas de prueba (sección 8 del libro)
    z = np.load(res / "shap_valores_muestra.npz", allow_pickle=True)
    idx = z["indices"]
    crudos = X_test.iloc[idx]
    np.savez_compressed(
        SALIDA / "shap_muestra.npz",
        valores=z["valores"].astype(np.float16),        # (celdas, variables, clases)
        base=z["base"][0].astype(np.float32),            # valor base por clase
        nombres=z["nombres"],
        clase_real=y_test.iloc[idx].to_numpy(),
        elevacion=crudos["Elevation"].to_numpy(),
        area=U.wilderness_label(crudos).to_numpy(),
        crudos=crudos[U.NUM_COLS].to_numpy().astype(np.float32),
        crudos_nombres=np.array(U.NUM_COLS),
    )
    print("Listo:", SALIDA)


if __name__ == "__main__":
    main()
