"""Artefacto de despliegue de Random Forest: los primeros 80 de sus 200 arboles.

El plan gratuito de Render tiene 512 MB de memoria. Con el bosque completo, el dashboard
ocupa unos 625 MB con los 8 modelos cargados. Un bosque es un promedio de arboles
independientes, asi que quedarse con un subconjunto no requiere reentrenar. Con 80 arboles
el F1 macro en la prueba reservada cambia de 0,7934 a 0,7932 (RF) y de 0,7959 a 0,7957
(ECLC). Es la reduccion de arboles equivalente en desempeno que admite la guia (seccion 4.2).

El bosque completo se conserva en artifacts_completos/RF_200.joblib. Todas las metricas del
libro y de las pruebas estadisticas usan el bosque completo.
"""
import copy
import json
import shutil

import joblib
import numpy as np
from sklearn.metrics import f1_score

import _e3 as e3
import _utils3 as u3
from src import preprocesamiento  # noqa: F401  (necesario para cargar los pipelines)

N_ARBOLES = 80
ART = e3.RAIZ / "dashboard" / "artifacts"
COMPLETOS = e3.RAIZ / "artifacts_completos"
COMPLETOS.mkdir(exist_ok=True)


def main():
    if not (COMPLETOS / "RF_200.joblib").exists():
        shutil.copy2(ART / "RF.joblib", COMPLETOS / "RF_200.joblib")
    rf = joblib.load(COMPLETOS / "RF_200.joblib")
    reducido = copy.deepcopy(rf)
    bosque = reducido.steps[-1][1]
    bosque.estimators_ = bosque.estimators_[:N_ARBOLES]
    bosque.n_estimators = N_ARBOLES

    _, _, X_test, y_test, _, _ = u3.get_data(n=u3.N_BENCHMARK)
    y = y_test.to_numpy()
    f1_completo = f1_score(y, rf.predict(X_test), average="macro")
    f1_reducido = f1_score(y, reducido.predict(X_test), average="macro")
    joblib.dump(reducido, ART / "RF.joblib", compress=3)
    recargado = joblib.load(ART / "RF.joblib")
    ida_vuelta = bool(np.array_equal(recargado.predict(X_test.iloc[:2000]), reducido.predict(X_test.iloc[:2000])))

    meta_ruta = ART / "RF.metadata.json"
    meta = json.load(open(meta_ruta, encoding="utf-8"))
    meta["despliegue"] = {"arboles": N_ARBOLES, "arboles_modelo_evaluado": 200,
                          "f1_macro_prueba_reservada_200": f1_completo, "f1_macro_prueba_reservada_80": f1_reducido,
                          "ida_y_vuelta_ok": ida_vuelta,
                          "motivo": "memoria del plan gratuito de Render (512 MB); reduccion equivalente en desempeno"}
    json.dump(meta, open(meta_ruta, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"RF 200 arboles: F1 {f1_completo:.4f} · 80 arboles: F1 {f1_reducido:.4f} · ida/vuelta {ida_vuelta}")


if __name__ == "__main__":
    main()
