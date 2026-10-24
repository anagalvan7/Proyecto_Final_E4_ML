"""Interpretabilidad, áreas no vistas, costo de afinar y semillas de XGBoost con su mejor configuración
del espacio de búsqueda común (150 árboles, profundidad 6, tasa de aprendizaje 0,1), entrenado con las
464.809 celdas de entrenamiento.

Regenera los insumos del capítulo 8 y de las secciones 9.2 y 9.4:
    shap_valores_muestra.npz, shap_casos_waterfall.json, figuras/shap_*.png,
    shap_importancia_global_muestra.csv, shap_lime_casos.json, finalista_areas_no_vistas.csv,
    xgb_perfilamiento.csv, xgb_sensibilidad_semilla.csv
"""
import time
import tracemalloc

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline as SkPipeline
from xgboost import XGBClassifier

import _e3 as e3
import _utils3 as u3
import experimentos_complementarios as ec

HP = e3.mejores_combinaciones().set_index("modelo").loc["XGB", "params_moda"]


def xgb(n_jobs=-1, **extra):
    return XGBClassifier(objective="multi:softprob", num_class=7, eval_metric="mlogloss", tree_method="hist",
                         n_jobs=n_jobs, **extra)


def perfilamiento_y_semillas():
    _, _, X_test, y_test, X_train, y_train = u3.get_data(n=u3.N_BENCHMARK)
    filas = []
    for nombre, params in [("por defecto (100 árboles, profundidad 6, tasa 0,3)", {}),
                           ("espacio común (150 árboles, profundidad 6, tasa 0,1)", HP)]:
        pipe = SkPipeline(list(u3.build_preprocessor_steps()) + [("clf", xgb(random_state=u3.RANDOM_STATE, **params))])
        tracemalloc.start()
        t0 = time.time()
        pipe.fit(X_train, y_train - 1)
        t_fit = time.time() - t0
        _, pico_fit = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        t0 = time.time()
        pred = pipe.predict(X_test) + 1
        t_pred = time.time() - t0
        _, pico_pred = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        filas.append({"versión": nombre, "tiempo_entrenamiento_s": t_fit, "tiempo_inferencia_s": t_pred,
                      "memoria_pico_entrenamiento_MB": pico_fit / 1e6, "memoria_pico_inferencia_MB": pico_pred / 1e6,
                      "F1_macro_prueba": f1_score(y_test, pred, average="macro")})
        print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(ec.RESULTS / "xgb_perfilamiento.csv", index=False, encoding="utf-8-sig")
    semillas = []
    for s in range(5):
        pipe = SkPipeline(list(u3.build_preprocessor_steps()) + [
            ("clf", xgb(random_state=s, subsample=0.8, colsample_bytree=0.8, **HP))])
        pipe.fit(X_train, y_train - 1)
        semillas.append({"semilla": s, "F1_macro": f1_score(y_test, pipe.predict(X_test) + 1, average="macro")})
        print(semillas[-1], flush=True)
    pd.DataFrame(semillas).to_csv(ec.RESULTS / "xgb_sensibilidad_semilla.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    print("Configuración de XGBoost:", HP, flush=True)
    ec.HP_XGB = HP
    for tarea in (ec.shap_figuras, ec.shap_figuras_es, ec.lime_casos, ec.areas, perfilamiento_y_semillas):
        print(f"=== {tarea.__name__} ===", flush=True)
        tarea()
