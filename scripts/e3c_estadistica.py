"""Verificación estadística en el protocolo de datos completos (10 pliegues 5 × 2 + prueba reservada).

Mismas pruebas que e3_estadistica.py (Friedman + Nemenyi, Wilcoxon-Holm, t corregida de
Nadeau-Bengio con n_prueba/n_entrenamiento = 0,25, McNemar en la prueba reservada, prueba
bayesiana con ROPE de 0,01, delta de Cliff, d de Cohen e intervalos BCa), aplicadas al ECLC frente a
cada competidor. Salidas: resultados/e3/completo/estadistica_*_completo.*
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multitest import multipletests

import _e3 as e3
from e3_estadistica import (ALFA, COMPETIDORES, NOMBRE, bayes_signed_rank, bca, cliff, diagrama_cd,
                            magnitud_cliff, mcnemar, nadeau_bengio)

DIR = e3.RES_E3 / "completo"
COMBINADOR = "ECLC"


def cargar():
    filas, n_te, n_tr = [], None, None
    for ruta in sorted(glob.glob(str(DIR / "meta_rep*_f*.json"))):
        d = json.load(open(ruta, encoding="utf-8"))
        n_te, n_tr = d["n_prueba"], d["n_entrenamiento"]
        for metodo, m in d["metricas"].items():
            filas.append({"pliegue": d["nombre"], "metodo": metodo, "f1_macro": m["f1_macro"],
                          "accuracy": m["accuracy"]})
    return pd.DataFrame(filas), n_te, n_tr


def main():
    df, n_te, n_tr = cargar()
    piv = df.pivot_table(index="pliegue", columns="metodo", values="f1_macro")
    resumen = []
    for metodo in piv.columns:
        v = piv[metodo].to_numpy()
        lo, hi = bca(v)
        resumen.append({"metodo": metodo, "f1_media": v.mean(), "f1_sd": v.std(ddof=1), "ic95_bca_inf": lo,
                        "ic95_bca_sup": hi, "accuracy_media": df[df.metodo == metodo]["accuracy"].mean()})
    resumen = pd.DataFrame(resumen).sort_values("f1_media", ascending=False)
    hold = json.load(open(DIR / "meta_holdout.json", encoding="utf-8"))["metricas"]
    resumen["f1_prueba_reservada"] = resumen["metodo"].map(lambda m: hold.get(m, {}).get("f1_macro", np.nan))
    resumen.to_csv(DIR / "estadistica_resumen_completo.csv", index=False, encoding="utf-8-sig")

    universo = [COMBINADOR] + COMPETIDORES
    M = piv[universo].to_numpy()
    chi2, p_fr = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    rangos = pd.Series(stats.rankdata(-M, axis=1).mean(0), index=[NOMBRE[u] for u in universo])
    k, N = len(universo), M.shape[0]
    cd = qsturng(1 - ALFA, k, 1e6) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    diagrama_cd(rangos, cd, DIR / "diagrama_cd_completo.png", n_pliegues=N)

    pred = np.load(DIR / "pred_holdout.npz")
    y = pred["y_te"]
    clave = {f"base:{m}": f"base_{m}" for m in e3.MODELOS_ORDEN} | {"HGB (estado del arte)": "HGB_(estado_del_arte)",
                                                                    "ECLC": "ECLC"}
    acierto = {k_: pred[v] == y for k_, v in clave.items()}
    filas = []
    for comp in COMPETIDORES:
        dif = piv[COMBINADOR].to_numpy() - piv[comp].to_numpy()
        w = stats.wilcoxon(dif, alternative="greater", zero_method="zsplit")
        t_nb, p_nb = nadeau_bengio(dif, n_tr, n_te)
        b01, b10, chi_mc, p_mc = mcnemar(acierto[COMBINADOR], acierto[comp])
        p_der, p_rope, p_izq = bayes_signed_rank(dif)
        lo, hi = bca(dif)
        sd = dif.std(ddof=1)
        filas.append({"competidor": NOMBRE[comp], "dif_media_f1": dif.mean(), "dif_ic95_inf": lo, "dif_ic95_sup": hi,
                      "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue), "nb_t": t_nb, "nb_p": p_nb,
                      "mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_chi2": chi_mc, "mcnemar_p": p_mc,
                      "bayes_p_eclc_mejor": p_der, "bayes_p_rope": p_rope, "bayes_p_competidor_mejor": p_izq,
                      "cliff_delta": cliff(piv[COMBINADOR], piv[comp]),
                      "cohen_dz": float(dif.mean() / sd) if sd > 0 else np.inf,
                      "ecl_gana_pliegues": int((dif > 0).sum())})
    comp_df = pd.DataFrame(filas)
    for prueba in ["wilcoxon", "nb", "mcnemar"]:
        comp_df[f"{prueba}_p_holm"] = multipletests(comp_df[f"{prueba}_p"], method="holm")[1]
    comp_df["magnitud_cliff"] = comp_df["cliff_delta"].apply(magnitud_cliff)
    comp_df.to_csv(DIR / "estadistica_pareadas_completo.csv", index=False, encoding="utf-8-sig")

    bases = [c for c in COMPETIDORES if c.startswith("base:")]
    mejor = max(bases, key=lambda c: piv[c].mean())
    fila = comp_df[comp_df.competidor == NOMBRE[mejor]].iloc[0]
    pruebas = {"Friedman (global)": p_fr < ALFA,
               "Nemenyi (rango ECLC vs mejor benchmark > CD)": rangos[NOMBRE[mejor]] - rangos[NOMBRE[COMBINADOR]] > cd,
               "Wilcoxon-Holm": fila["wilcoxon_p_holm"] < ALFA, "t corregida Nadeau-Bengio (Holm)": fila["nb_p_holm"] < ALFA,
               "McNemar en prueba reservada (Holm)": fila["mcnemar_p_holm"] < ALFA,
               "Bayesiana (P(ECLC mejor) > 0,95)": fila["bayes_p_eclc_mejor"] > 0.95}
    rechazos = [k_ for k_, v in pruebas.items() if v and not k_.startswith(("Bayes", "Friedman"))]
    veredicto = {"protocolo": "datos completos, 5 x 2 pliegues + prueba reservada", "mejor_benchmark": NOMBRE[mejor],
                 "f1_eclc": float(piv[COMBINADOR].mean()), "f1_mejor_benchmark": float(piv[mejor].mean()),
                 "f1_eclc_prueba_reservada": hold[COMBINADOR]["f1_macro"],
                 "condicion_1_media_mejor_que_los_7": bool(all(piv[COMBINADOR].mean() > piv[c].mean() for c in bases)),
                 "pruebas_vs_mejor": {k_: bool(v) for k_, v in pruebas.items()},
                 "condicion_2_tres_pruebas": bool(pruebas["Friedman (global)"] and len(rechazos) >= 2),
                 "condicion_3_efecto_no_despreciable": bool(abs(fila["cliff_delta"]) >= 0.147),
                 "cliff_vs_mejor": float(fila["cliff_delta"]), "n_pliegues": int(N),
                 "friedman": {"estadistico": float(chi2), "p_valor": float(p_fr), "k": k, "N": int(N), "cd": float(cd),
                              "rangos": rangos.round(3).to_dict()}}
    veredicto["R3"] = bool(veredicto["condicion_1_media_mejor_que_los_7"] and veredicto["condicion_2_tres_pruebas"]
                           and veredicto["condicion_3_efecto_no_despreciable"])
    json.dump(veredicto, open(DIR / "estadistica_veredicto_completo.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(resumen.round(4).to_string())
    print(comp_df[["competidor", "dif_media_f1", "ecl_gana_pliegues", "wilcoxon_p_holm", "nb_p_holm", "mcnemar_p_holm",
                   "bayes_p_eclc_mejor", "cliff_delta"]].round(4).to_string())
    print(json.dumps(veredicto, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
