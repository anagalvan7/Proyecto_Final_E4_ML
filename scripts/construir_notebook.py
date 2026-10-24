"""Une los capítulos ejecutados del Jupyter Book en un solo cuaderno (Entregable4_Covertype.ipynb).

Usa los cuadernos que Jupyter Book deja en docs/_build/jupyter_execute (con sus salidas), en el orden
de docs/_toc.yml. Convierte la sintaxis MyST que Jupyter no interpreta: figuras (se incrustan como
adjuntos), admoniciones (citas en bloque), citas bibliográficas (autor, año) y referencias a ecuaciones.
Requiere haber compilado antes el libro: jupyter-book build docs
"""
import base64
import json
import re
from pathlib import Path

import nbformat
import yaml

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "docs"
EJEC = DOCS / "_build" / "jupyter_execute"
SALIDA = RAIZ / "Entregable4_Covertype.ipynb"

TITULOS_ADMON = {"note": "Nota", "warning": "Advertencia", "important": "Importante", "admonition": ""}


def archivos_toc():
    toc = yaml.safe_load(open(DOCS / "_toc.yml", encoding="utf-8"))
    orden = [toc["root"]]

    def recorrer(items):
        for it in items:
            orden.append(it["file"])
            recorrer(it.get("sections", []))

    for parte in toc["parts"]:
        recorrer(parte["chapters"])
    return orden


def bibliografia():
    texto = open(DOCS / "references.bib", encoding="utf-8").read()
    refs = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", texto, re.S):
        clave, cuerpo = m.group(1).strip(), m.group(2)
        autor = re.search(r"author\s*=\s*\{(.+?)\}\s*,", cuerpo, re.S)
        anio = re.search(r"year\s*=\s*\{?(\d{4})", cuerpo)
        titulo = re.search(r"title\s*=\s*\{(.+?)\}\s*,", cuerpo, re.S)
        if not autor:
            autor = re.search(r"(?:institution|organization|publisher)\s*=\s*\{(.+?)\}", cuerpo, re.S)
        nombres = [a.strip() for a in (autor.group(1) if autor else clave).replace("\n", " ").split(" and ")]
        apellidos = [re.sub(r"[{}\\']", "", n.split(",")[0] if "," in n else n.split()[-1]) for n in nombres]
        corto = apellidos[0] + (" et al." if len(apellidos) > 2 else f" y {apellidos[1]}" if len(apellidos) == 2 else "")
        refs[clave] = {"corto": corto, "anio": anio.group(1) if anio else "s. f.",
                       "completo": f"{'; '.join(nombres)} ({anio.group(1) if anio else 's. f.'}). "
                                   f"{re.sub(r'[{}]', '', titulo.group(1)) if titulo else ''}."}
    return refs


REFS = bibliografia()
CITADAS = []


def citas(texto):
    def cita(m, narrativa=False):
        partes = []
        for clave in m.group(1).split(","):
            clave = clave.strip()
            r = REFS.get(clave, {"corto": clave, "anio": ""})
            if clave not in CITADAS:
                CITADAS.append(clave)
            partes.append(f"{r['corto']} ({r['anio']})" if narrativa else f"{r['corto']}, {r['anio']}")
        return "; ".join(partes) if narrativa else "(" + "; ".join(partes) + ")"

    texto = re.sub(r"\{cite:t\}`([^`]+)`", lambda m: cita(m, True), texto)
    return re.sub(r"\{cite(?::p)?\}`([^`]+)`", cita, texto)


def ecuaciones(texto, numeros):
    def etiqueta(m):
        numeros.setdefault(m.group(1), len(numeros) + 1)
        return f"$$ \\tag{{{numeros[m.group(1)]}}}"
    texto = re.sub(r"\$\$\s*\(([\w-]+)\)", lambda m: "$$", texto)  # se numeran por orden de aparición
    return texto


def convertir_markdown(texto, adjuntos, numeros):
    # Ecuaciones con etiqueta: "$$ ... $$ (eq-x)" -> \tag{n}
    def eq(m):
        numeros.setdefault(m.group(2), len(numeros) + 1)
        return f"$$\n{m.group(1).strip()} \\tag{{{numeros[m.group(2)]}}}\n$$"
    texto = re.sub(r"\$\$(.+?)\$\$\s*\(([\w-]+)\)", eq, texto, flags=re.S)
    texto = re.sub(r"\{eq\}`([\w-]+)`", lambda m: f"({numeros.setdefault(m.group(1), len(numeros) + 1)})", texto)

    # Figuras
    def figura(m):
        ruta, cuerpo = m.group(1).strip(), m.group(2)
        leyenda = "\n".join(l for l in cuerpo.splitlines() if not l.startswith(":")).strip()
        archivo = DOCS / ruta
        nombre = archivo.name
        if archivo.exists():
            adjuntos[nombre] = {"image/png": base64.b64encode(archivo.read_bytes()).decode()}
        return f"![{nombre}](attachment:{nombre})\n\n*{leyenda}*"
    texto = re.sub(r"```\{figure\}\s*(\S+)\n(.*?)```", figura, texto, flags=re.S)

    # Admoniciones
    def admon(m):
        tipo, titulo_extra, cuerpo = m.group(1), m.group(2).strip(), m.group(3)
        cuerpo = "\n".join(l for l in cuerpo.splitlines() if not l.startswith(":class:")).strip()
        titulo = titulo_extra or TITULOS_ADMON.get(tipo, tipo.capitalize())
        lineas = [f"> **{titulo}.**" if titulo else ">"] + [f"> {l}" if l else ">" for l in cuerpo.splitlines()]
        return "\n".join(lineas)
    texto = re.sub(r"```\{(note|warning|important|admonition)\}([^\n]*)\n(.*?)```", admon, texto, flags=re.S)
    return citas(texto)


def celdas_de(nombre, numeros):
    """Celdas del capítulo con las salidas que Jupyter Book guardó en su caché de ejecución."""
    from jupyter_cache import get_cache
    from myst_nb.core.read import read_myst_markdown_notebook
    texto = (DOCS / f"{nombre}.md").read_text(encoding="utf-8")
    if texto.lstrip().startswith("---") and "kernelspec" in texto.split("---")[1]:
        nb = read_myst_markdown_notebook(texto)
        try:
            _, nb = get_cache(DOCS / "_build" / ".jupyter_cache").merge_match_into_notebook(nb)
        except KeyError:
            print(f"Aviso: {nombre} no está en la caché; se incluye sin salidas")
        celdas = nb.cells
    else:
        celdas = [nbformat.v4.new_markdown_cell(texto)]
    salida = []
    for c in celdas:
        if c.cell_type == "markdown":
            if c.source.strip().startswith("```{bibliography}"):
                continue
            adjuntos = {}
            c.source = convertir_markdown(c.source, adjuntos, numeros)
            if adjuntos:
                c["attachments"] = adjuntos
        c.pop("id", None)
        salida.append(c)
    return salida


def main():
    numeros = {}
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}}
    nb.cells.append(nbformat.v4.new_code_cell(
        "# Configuración: el código de los capítulos se escribió para ejecutarse desde docs/\n"
        "import os\nimport sys\nfrom pathlib import Path\n\nif Path(\"docs\").is_dir():\n    os.chdir(\"docs\")\n"
        "sys.path.insert(0, str(Path.cwd()))\nprint(\"Directorio de trabajo:\", Path.cwd())"))
    for nombre in archivos_toc():
        if nombre == "bibliografia":
            continue
        nb.cells.extend(celdas_de(nombre, numeros))
    refs = sorted(CITADAS, key=lambda k: (REFS.get(k, {}).get("corto", k), REFS.get(k, {}).get("anio", "")))
    nb.cells.append(nbformat.v4.new_markdown_cell(
        "# Bibliografía\n\n" + "\n".join(f"- {REFS[k]['completo']}" for k in refs if k in REFS)))
    for c in nb.cells:
        c.pop("id", None)
    nbformat.write(nbformat.validator.normalize(nb)[1] if hasattr(nbformat.validator, "normalize") else nb, SALIDA)
    print(f"{SALIDA.name}: {len(nb.cells)} celdas, {SALIDA.stat().st_size / 2**20:.1f} MB, {len(refs)} referencias")


if __name__ == "__main__":
    main()
