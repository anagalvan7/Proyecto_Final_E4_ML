"""Busqueda documentada para la matriz de novedad (seccion 5.2 de la guia).

Consulta cuatro bases bibliograficas con API publica y registra, para cada cadena de
busqueda, el numero de resultados y los titulos de los primeros resultados (que se
revisan a mano en el capitulo 4 del libro):
    OpenAlex (indice abierto que agrega Crossref, PubMed, arXiv, repositorios...),
    Semantic Scholar, arXiv y Crossref.
Scopus y Web of Science requieren suscripcion; sus conteos provienen de la revision
preliminar del equipo (documento de propuesta) y se reportan aparte.

Salida: resultados/e3/literatura/<fecha>_busquedas.json y matriz_novedad.csv
"""
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "e3" / "literatura"
SALIDA.mkdir(parents=True, exist_ok=True)
N_TITULOS = 25
CABECERA = {"User-Agent": "covertype-e3-revision/1.0 (proyecto academico)"}

# (identificador, cadena, idioma, componentes: modelo / aplicacion / sinonimos)
CADENAS = [
    ("C1", '"dynamic ensemble selection" covertype', "en", "familia + problema"),
    ("C2", '"dynamic ensemble selection" "forest cover type"', "en", "familia + problema (sinonimo)"),
    ("C3", '"dynamic classifier selection" covtype', "en", "familia + nombre del conjunto"),
    ("C4", '"local competence" ensemble "cover type"', "en", "mecanismo + problema"),
    ("C5", '"dynamic weighting" ensemble "forest cover"', "en", "mecanismo + problema"),
    ("C6", 'KNORA covertype', "en", "metodo afin + problema"),
    ("C7", '"META-DES" covertype', "en", "metodo afin + problema"),
    ("C8", '"local log loss" ensemble weighting nearest neighbors', "en", "mecanismo (cualquier aplicacion)"),
    ("C9", '"class-specific" "dynamic weighting" ensemble classifiers', "en", "mecanismo (cualquier aplicacion)"),
    ("C10", 'ensamble "competencia local" "cobertura forestal"', "es", "familia + problema (espanol)"),
    ("C11", 'seleccion dinamica de clasificadores cobertura vegetal', "es", "familia + problema (espanol)"),
    # Segunda ronda (busqueda ampliada): combinador regional y finalistas alternativos
    ("C12", 'stacking covertype', "en", "familia (stacking) + nombre del conjunto"),
    ("C13", '"stacked generalization" "forest cover type"', "en", "familia + problema"),
    ("C14", '"clustered stacking" OR "cluster-based stacking" OR "regional stacking"', "en", "mecanismo (cualquier aplicacion)"),
    ("C15", '"local stacking" OR "locally weighted stacking" ensemble', "en", "mecanismo (cualquier aplicacion)"),
    ("C16", '"mixture of experts" covertype', "en", "familia afin + nombre del conjunto"),
    ("C17", '"hierarchical shrinkage" stacking ensemble', "en", "mecanismo (cualquier aplicacion)"),
    ("C18", '"logistic model tree" covertype', "en", "finalista 2 + problema"),
    ("C19", '"functional trees" OR "LMT" "forest cover type"', "en", "finalista 2 + problema"),
    ("C20", '"fuzzy k-nearest neighbor" covertype', "en", "finalista 3 + problema"),
    ("C21", 'stacking "cobertura forestal" OR "cobertura vegetal" ensamble', "es", "familia + problema (espanol)"),
]


def _get(url, intentos=6):
    for i in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=CABECERA), timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == intentos - 1:
                raise
            time.sleep(10 * (i + 1))


def openalex(q):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": q, "per-page": N_TITULOS, "mailto": "proyecto@example.org"})
    d = json.loads(_get(url))
    return d["meta"]["count"], [(w.get("title"), w.get("publication_year"), w.get("doi")) for w in d["results"]]


def semantic_scholar(q):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": q.replace('"', ""), "limit": N_TITULOS, "fields": "title,year,externalIds"})
    d = json.loads(_get(url))
    return d.get("total", 0), [(p.get("title"), p.get("year"), (p.get("externalIds") or {}).get("DOI"))
                               for p in d.get("data", [])]


def arxiv(q):
    terminos = " AND ".join(f'all:{t}' if not t.startswith('"') else f"all:{t}" for t in _tokens(q))
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": terminos, "max_results": N_TITULOS})
    raiz = ET.fromstring(_get(url))
    ns = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}
    total = int(raiz.find("os:totalResults", ns).text)
    return total, [(e.find("a:title", ns).text.strip().replace("\n", " "), e.find("a:published", ns).text[:4],
                    e.find("a:id", ns).text) for e in raiz.findall("a:entry", ns)]


def crossref(q):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query": q.replace('"', ""), "rows": N_TITULOS})
    d = json.loads(_get(url))["message"]
    return d["total-results"], [((it.get("title") or [""])[0], (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                                 it.get("DOI")) for it in d["items"]]


def dblp(q):
    """DBLP (bibliografia de computacion): busca en titulos; las comillas se ignoran."""
    url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode(
        {"q": q.replace('"', ""), "format": "json", "h": N_TITULOS})
    d = json.loads(_get(url))["result"]["hits"]
    return int(d.get("@total", 0)), [(h["info"].get("title"), h["info"].get("year"), h["info"].get("doi"))
                                     for h in d.get("hit", [])]


def _tokens(q):
    """Separa frases entre comillas y palabras sueltas."""
    out, actual, dentro = [], "", False
    for ch in q:
        if ch == '"':
            if dentro:
                out.append(f'"{actual}"')
                actual = ""
            dentro = not dentro
        elif ch == " " and not dentro:
            if actual:
                out.append(actual)
            actual = ""
        else:
            actual += ch
    if actual:
        out.append(actual)
    return out


# Crossref se excluye: su busqueda es por relevancia con OR implicito y el total no es un conteo de
# coincidencias (devuelve millones de registros para cualquier cadena).
BASES = {"OpenAlex": openalex, "arXiv": arxiv, "DBLP": dblp, "Semantic Scholar": semantic_scholar}


def reintentar(base="Semantic Scholar", pausa=25):
    """Repite solo las consultas de 'base' que fallaron (p. ej. HTTP 429 por limite de tasa)."""
    fecha = dt.date.today().isoformat()
    ruta = SALIDA / f"{fecha}_busquedas.json"
    registro = json.load(open(ruta, encoding="utf-8"))
    for r in registro:
        if r["base"] == base and r["error"]:
            time.sleep(pausa)
            try:
                r["resultados"], r["titulos"] = BASES[base](r["cadena"])
                r["error"] = None
            except Exception as e:  # noqa: BLE001
                r["error"] = str(e)
            print(r["id"], base, r["resultados"], flush=True)
    json.dump(registro, open(ruta, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.DataFrame([{k: v for k, v in r.items() if k != "titulos"} for r in registro]).to_csv(
        SALIDA / "matriz_busquedas.csv", index=False, encoding="utf-8-sig")


def ampliar():
    """Agrega al registro del dia las cadenas que aun no tiene (segunda ronda), sin repetir las demas."""
    fecha = dt.date.today().isoformat()
    ruta = SALIDA / f"{fecha}_busquedas.json"
    registro = json.load(open(ruta, encoding="utf-8")) if ruta.exists() else []
    hechas = {(r["id"], r["base"]) for r in registro}
    for cid, cadena, idioma, comp in CADENAS:
        for base in ("OpenAlex", "arXiv"):
            if (cid, base) in hechas:
                continue
            try:
                total, titulos = BASES[base](cadena)
                error = None
            except Exception as e:  # noqa: BLE001
                total, titulos, error = None, [], str(e)
            registro.append({"id": cid, "base": base, "cadena": cadena, "idioma": idioma, "componentes": comp,
                             "fecha": fecha, "resultados": total, "titulos": titulos, "error": error})
            print(cid, base, total, flush=True)
            time.sleep(0.5)
    json.dump(registro, open(ruta, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.DataFrame([{k: v for k, v in r.items() if k != "titulos"} for r in registro]).to_csv(
        SALIDA / "matriz_busquedas.csv", index=False, encoding="utf-8-sig")


def main():
    import sys
    if "--reintentar" in sys.argv:
        reintentar()
        return
    if "--ampliar" in sys.argv:
        ampliar()
        return
    fecha = dt.date.today().isoformat()
    registro = []
    for cid, cadena, idioma, comp in CADENAS:
        for base, fn in BASES.items():
            try:
                total, titulos = fn(cadena)
                error = None
            except Exception as e:  # noqa: BLE001
                total, titulos, error = None, [], str(e)
            registro.append({"id": cid, "base": base, "cadena": cadena, "idioma": idioma,
                             "componentes": comp, "fecha": fecha, "resultados": total,
                             "titulos": titulos, "error": error})
            print(f"{cid} {base:16s} {total}", flush=True)
            time.sleep(6 if base == "Semantic Scholar" else 0.5)
    json.dump(registro, open(SALIDA / f"{fecha}_busquedas.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    pd.DataFrame([{k: v for k, v in r.items() if k != "titulos"} for r in registro]).to_csv(
        SALIDA / "matriz_busquedas.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
