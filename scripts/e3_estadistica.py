"""Verificacion estadistica de la superioridad del ECLC (bloque D de la guia).

Datos: 30 pliegues externos pareados (10 repeticiones x 3 pliegues; la repeticion 0 es la
particion del benchmark) y la prueba reservada de 116.203 filas.

Pruebas (Cuadro 3 de la guia, columna de datos independientes / clasificacion):
  T1. Friedman + Nemenyi (global) con el ECLC dentro del universo y diagrama CD.
  T2. Wilcoxon de rangos con signo, ECLC vs cada competidor, Holm.
  T3. t corregida de Nadeau y Bengio (2003), ECLC vs cada competidor, Holm.
  T4. McNemar en la prueba reservada, ECLC vs cada competidor, Holm.
  T5. Prueba bayesiana de rangos con signo (Benavoli et al., 2017), ROPE = 0,01.
Tamano del efecto: delta de Cliff y d de Cohen pareada. Intervalos bootstrap BCa.

Salidas en resultados/e3/: estadistica_*.csv / .json y figura del diagrama CD.
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multitest import multipletests

import _e3 as e3

ALFA = 0.05
ROPE = 0.01
COMPETIDORES = ["base:KNN", "base:NB", "base:LR", "base:DT", "base:RF", "base:XGB", "base:SVM",
                "HGB (estado del arte)"]
NOMBRE = {"ECLC": "ECLC (original)", "base:KNN": "KNN", "base:NB": "Naive Bayes",
          "base:LR": "Reg. logística", "base:DT": "Árbol de decisión", "base:RF": "Random Forest",
          "base:XGB": "XGBoost", "base:SVM": "SVM", "HGB (estado del arte)": "HistGradientBoosting"}
RNG = np.random.default_rng(42)


def cargar():
    filas = []
    for ruta in sorted(glob.glob(str(e3.RES_E3 / "meta" / "rep*_f*.json"))):
        d = json.load(open(ruta, encoding="utf-8"))
        for metodo, m in d["metricas"].items():
            filas.append({"rep": d["rep"], "pliegue": d["pliegue"], "metodo": metodo, **{
                k: v for k, v in m.items() if k != "f1_por_clase"}})
    return pd.DataFrame(filas)


def bca(datos, estadistico=np.mean, B=5000, alfa=0.05):
    """Intervalo bootstrap BCa (Efron, 1987) de un estadistico de una muestra."""
    datos = np.asarray(datos)
    n = len(datos)
    theta = estadistico(datos)
    boot = np.array([estadistico(datos[RNG.integers(0, n, n)]) for _ in range(B)])
    z0 = stats.norm.ppf(np.clip((boot < theta).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([estadistico(np.delete(datos, i)) for i in range(n)])
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5) + 1e-300)
    z = stats.norm.ppf([alfa / 2, 1 - alfa / 2])
    q = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return float(np.quantile(boot, q[0])), float(np.quantile(boot, q[1]))


def cliff(a, b):
    """Delta de Cliff: P(a > b) - P(a < b) sobre todos los pares."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((a > b).mean() - (a < b).mean())


def magnitud_cliff(d):
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeno" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def nadeau_bengio(dif, n_tr, n_te):
    """t corregida: var * (1/J + n_te/n_tr) (Nadeau y Bengio, 2003). Unilateral (ECLC mejor)."""
    J = len(dif)
    var = np.var(dif, ddof=1)
    t = dif.mean() / np.sqrt((1 / J + n_te / n_tr) * var + 1e-300)
    return float(t), float(stats.t.sf(t, J - 1))


def bayes_signed_rank(dif, rope=ROPE, muestras=20000):
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017), con pseudo-observacion
    en 0 y pesos de Dirichlet(1, ..., 1). Devuelve P(ECLC mejor), P(ROPE), P(competidor mejor)."""
    z = np.concatenate([[0.0], dif])
    n = len(z)
    suma = z[:, None] + z[None, :]
    der = (suma > 2 * rope).astype(float)
    izq = (suma < -2 * rope).astype(float)
    W = RNG.dirichlet(np.ones(n), size=muestras)
    p_der = np.einsum("si,ij,sj->s", W, der, W)
    p_izq = np.einsum("si,ij,sj->s", W, izq, W)
    p_rope = 1 - p_der - p_izq
    gana = np.argmax(np.stack([p_der, p_rope, p_izq]), axis=0)
    return [float((gana == i).mean()) for i in range(3)]


def mcnemar(correcto_a, correcto_b):
    """McNemar con correccion de continuidad; unilateral a favor de A cuando b01 > b10."""
    b01 = int(np.sum(correcto_a & ~correcto_b))
    b10 = int(np.sum(~correcto_a & correcto_b))
    chi2 = (abs(b01 - b10) - 1) ** 2 / max(b01 + b10, 1)
    p_bilateral = float(stats.chi2.sf(chi2, 1))
    p = p_bilateral / 2 if b01 > b10 else 1 - p_bilateral / 2
    return b01, b10, float(chi2), float(p)


def diagrama_cd(rangos, cd, ruta, n_pliegues=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orden = rangos.sort_values()
    k = len(orden)
    fig, ax = plt.subplots(figsize=(9, 0.42 * k + 1.6))
    ax.set_xlim(1, k)
    ax.set_ylim(0, k + 2.0)
    ax.invert_xaxis()
    ax.axis("off")
    ax.hlines(k + 0.6, 1, k, color="k", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, k + 0.5, k + 0.7, color="k", lw=1)
        ax.text(r, k + 0.95, str(r), ha="center", fontsize=9)
    ax.hlines(k + 1.3, 1, 1 + cd, color="#b0413e", lw=3)
    ax.text(1 + cd / 2, k + 1.45, f"CD = {cd:.2f}", ha="center", fontsize=9, color="#b0413e")
    for i, (nombre, r) in enumerate(orden.items()):
        yy = k - i - 0.3
        color = "#1b6f5c" if nombre.startswith("ECLC") else "#333"
        ax.vlines(r, yy, k + 0.6, color=color, lw=0.8)
        ax.text(r, yy - 0.15, f"{nombre} ({r:.2f})", ha="center", va="top", fontsize=9, color=color,
                fontweight="bold" if nombre.startswith("ECLC") else "normal")
    vals = orden.to_numpy()
    nivel = 0
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(vals[i] >= a and vals[j] <= b for a, b in getattr(diagrama_cd, "_g", [])):
            ax.hlines(k + 0.25 - 0.18 * nivel, vals[i] - 0.03, vals[j] + 0.03, color="#5b6fb3", lw=4)
            nivel += 1
            diagrama_cd._g = getattr(diagrama_cd, "_g", []) + [(vals[i], vals[j])]
    diagrama_cd._g = []
    ax.set_title(f"Diagrama de diferencia crítica (Nemenyi, α = 0,05): F1 macro en {n_pliegues} pliegues",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    df = cargar()
    n_pliegues = df.groupby("metodo").size().iloc[0]
    meta0 = json.load(open(e3.RES_E3 / "meta" / "rep0_f0.json", encoding="utf-8"))
    n_te = meta0["n_prueba"]
    n_tr = 40_000 - n_te
    piv = df.pivot_table(index=["rep", "pliegue"], columns="metodo", values="f1_macro")

    # ---------------- tabla resumen
    tiempos = []
    for ruta in sorted(glob.glob(str(e3.RES_E3 / "meta" / "rep*_f*.json"))):
        d = json.load(open(ruta, encoding="utf-8"))
        t = d["meta_probas"]["tiempos"]
        fila = {f"base:{m}": v["ajuste_s"] for m, v in t.items() if m != "HGB"}
        fila.update({f"inf:base:{m}": v["inferencia_s"] for m, v in t.items() if m != "HGB"})
        fila["ECLC"] = sum(v["oof_s"] + v["ajuste_s"] for m, v in t.items() if m != "HGB") + \
            d["tiempo_optimizacion_s"] + d["tiempo_ajuste_meta_s"]
        fila["inf:ECLC"] = sum(v["inferencia_s"] for m, v in t.items() if m != "HGB") + d["tiempo_inferencia_meta_s"]
        fila["HGB (estado del arte)"] = t["HGB"]["busqueda_y_ajuste_s"]
        fila["inf:HGB (estado del arte)"] = t["HGB"]["inferencia_s"]
        tiempos.append(fila)
    tiempos = pd.DataFrame(tiempos).mean()
    resumen = []
    for metodo in piv.columns:
        v = piv[metodo].to_numpy()
        lo, hi = bca(v)
        resumen.append({"metodo": metodo, "f1_media": v.mean(), "f1_sd": v.std(ddof=1), "ic95_bca_inf": lo,
                        "ic95_bca_sup": hi,
                        "accuracy_media": df[df.metodo == metodo]["accuracy"].mean(),
                        "tiempo_entrenamiento_s": tiempos.get(metodo, np.nan),
                        "tiempo_inferencia_s": tiempos.get(f"inf:{metodo}", np.nan)})
    resumen = pd.DataFrame(resumen).sort_values("f1_media", ascending=False)
    resumen.to_csv(e3.RES_E3 / "estadistica_resumen.csv", index=False, encoding="utf-8-sig")

    # ---------------- T1: Friedman + Nemenyi
    universo = ["ECLC"] + COMPETIDORES
    M = piv[universo].to_numpy()
    chi2, p_fr = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    rangos = pd.Series(stats.rankdata(-M, axis=1).mean(0), index=[NOMBRE[u] for u in universo])
    k, N = len(universo), M.shape[0]
    cd = qsturng(1 - ALFA, k, 1e6) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    diagrama_cd(rangos, cd, e3.RES_E3 / "diagrama_cd_eclc.png")
    friedman = {"estadistico": float(chi2), "p_valor": float(p_fr), "k": k, "N": int(N), "cd": float(cd),
                "rangos": rangos.round(3).to_dict()}

    # ---------------- comparaciones pareadas
    hold = np.load(e3.RES_E3 / "holdout_probas.npz")
    y_test = hold["y_test"]
    acierto = {k_: (e3.CLASES[hold[k_].argmax(1)] == y_test) for k_ in hold.files if k_ != "y_test"}
    clave_holdout = {f"base:{m}": m for m in e3.MODELOS_ORDEN} | {"HGB (estado del arte)": "HGB"}
    filas = []
    for comp in COMPETIDORES:
        dif = piv["ECLC"].to_numpy() - piv[comp].to_numpy()
        w = stats.wilcoxon(dif, alternative="greater", zero_method="zsplit")
        t_nb, p_nb = nadeau_bengio(dif, n_tr, n_te)
        b01, b10, chi_mc, p_mc = mcnemar(acierto["ECLC"], acierto[clave_holdout[comp]])
        p_der, p_rope, p_izq = bayes_signed_rank(dif)
        lo, hi = bca(dif)
        sd = dif.std(ddof=1)
        filas.append({"competidor": NOMBRE[comp], "dif_media_f1": dif.mean(), "dif_ic95_inf": lo,
                      "dif_ic95_sup": hi, "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
                      "nb_t": t_nb, "nb_p": p_nb, "mcnemar_b01": b01, "mcnemar_b10": b10,
                      "mcnemar_chi2": chi_mc, "mcnemar_p": p_mc, "bayes_p_eclc_mejor": p_der,
                      "bayes_p_rope": p_rope, "bayes_p_competidor_mejor": p_izq,
                      "cliff_delta": cliff(piv["ECLC"], piv[comp]),
                      "cohen_dz": float(dif.mean() / sd) if sd > 0 else np.inf,
                      "ecl_gana_pliegues": int((dif > 0).sum())})
    comp_df = pd.DataFrame(filas)
    for prueba in ["wilcoxon", "nb", "mcnemar"]:
        comp_df[f"{prueba}_p_holm"] = multipletests(comp_df[f"{prueba}_p"], method="holm")[1]
    comp_df["magnitud_cliff"] = comp_df["cliff_delta"].apply(magnitud_cliff)
    comp_df.to_csv(e3.RES_E3 / "estadistica_pareadas.csv", index=False, encoding="utf-8-sig")

    # ---------------- veredicto R3
    mejor_bench = resumen[resumen.metodo.str.startswith("base:")].iloc[0]["metodo"]
    fila_mejor = comp_df[comp_df.competidor == NOMBRE[mejor_bench]].iloc[0]
    media_eclc = piv["ECLC"].mean()
    cond1 = all(media_eclc > piv[c].mean() for c in COMPETIDORES[:7])
    rechazos_mejor = {
        "Friedman (global)": p_fr < ALFA,
        "Nemenyi (rango ECLC vs mejor benchmark > CD)":
            rangos[NOMBRE[mejor_bench]] - rangos[NOMBRE["ECLC"]] > cd,
        "Wilcoxon-Holm": fila_mejor["wilcoxon_p_holm"] < ALFA,
        "t corregida Nadeau-Bengio (Holm)": fila_mejor["nb_p_holm"] < ALFA,
        "McNemar en prueba reservada (Holm)": fila_mejor["mcnemar_p_holm"] < ALFA,
        "Bayesiana (P(ECLC mejor) > 0,95)": fila_mejor["bayes_p_eclc_mejor"] > 0.95,
    }
    pruebas_frecuentistas = [k_ for k_ in rechazos_mejor if not k_.startswith("Bayes") and rechazos_mejor[k_]]
    cond2 = len([k_ for k_ in pruebas_frecuentistas if k_ != "Friedman (global)"]) >= 2 and rechazos_mejor["Friedman (global)"]
    cond3 = abs(fila_mejor["cliff_delta"]) >= 0.147
    veredicto = {"mejor_benchmark": NOMBRE[mejor_bench], "f1_eclc": float(media_eclc),
                 "f1_mejor_benchmark": float(piv[mejor_bench].mean()),
                 "condicion_1_media_mejor_que_los_7": bool(cond1),
                 "pruebas_vs_mejor": {k_: bool(v) for k_, v in rechazos_mejor.items()},
                 "condicion_2_tres_pruebas": bool(cond2),
                 "condicion_3_efecto_no_despreciable": bool(cond3),
                 "cliff_vs_mejor": float(fila_mejor["cliff_delta"]),
                 "R3": bool(cond1 and cond2 and cond3),
                 "n_comparaciones_por_prueba": len(COMPETIDORES), "n_pliegues": int(n_pliegues),
                 "friedman": friedman}
    json.dump(veredicto, open(e3.RES_E3 / "estadistica_veredicto.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------------- ablacion y reglas de combinacion
    reglas = ["ECLC", "sin pesos por clase", "sin vecindario (tau = inf)", "sin contraccion (tau = 0)",
              "sin nucleo de distancia", "sin balanceo de vecinos", "promedio simple (beta = 0)",
              "pesos por error global", "stacking lineal", "KNORA-E", "OLA"]
    abl = []
    for r in reglas[1:]:
        dif = piv["ECLC"].to_numpy() - piv[r].to_numpy()
        w = stats.wilcoxon(dif, alternative="greater", zero_method="zsplit") if np.any(dif != 0) else None
        abl.append({"variante": r, "f1_media": piv[r].mean(), "f1_sd": piv[r].std(ddof=1),
                    "dif_vs_eclc": -dif.mean(), "wilcoxon_p": float(w.pvalue) if w else 1.0,
                    "cliff_delta": cliff(piv["ECLC"], piv[r])})
    abl = pd.DataFrame(abl)
    abl["wilcoxon_p_holm"] = multipletests(abl["wilcoxon_p"], method="holm")[1]
    abl.to_csv(e3.RES_E3 / "estadistica_ablacion.csv", index=False, encoding="utf-8-sig")

    print(resumen.round(4).to_string())
    print(comp_df[["competidor", "dif_media_f1", "wilcoxon_p_holm", "nb_p_holm", "mcnemar_p_holm",
                   "bayes_p_eclc_mejor", "cliff_delta"]].round(4).to_string())
    print(abl.round(4).to_string())
    print(json.dumps(veredicto, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
