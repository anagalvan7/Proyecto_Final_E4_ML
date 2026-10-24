"""Reentrena el finalista del benchmark corregido con el entrenamiento COMPLETO (464.809 filas).

Envoltorio de finalistas.py para el Entregable 3:
- Tras la correccion del protocolo, el finalista es Random Forest + SMOTE (tabla_maestra.csv).
  Con SMOTE, el entrenamiento completo pasa a ~1,6 millones de filas; el bosque se entrena con
  n_jobs=20 (en el benchmark se usaba n_jobs=1 porque se paralelizaba entre combinaciones).
- Se omite el TreeSHAP de finalistas.py: sobre 200 arboles de profundidad 20 ajustados con
  1,6 millones de filas tardaria horas. La interpretabilidad del finalista se hace con el
  KernelSHAP agrupado comparable con el del ECLC (capitulo 11).
Salidas: resultados/resumen_finalista.json y resultados/delong_finalistas.csv.
"""
import sys

from sklearn.ensemble import RandomForestClassifier

import _utils3 as u3

sys.modules["shap"] = None  # finalistas.py captura el ImportError y deja constancia
u3.MODELOS["RF"]["build"] = lambda **kw: RandomForestClassifier(random_state=u3.RANDOM_STATE, n_jobs=20, **kw)

import finalistas  # noqa: E402

if __name__ == "__main__":
    finalistas.main()
