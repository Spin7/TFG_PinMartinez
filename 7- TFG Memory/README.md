# TFG Memory — Thesis Document

This folder contains the **final thesis document** and all associated presentation materials for the defense.

## Contents

| File / Folder | Description |
|---|---|
| `TFG_PinMartinez.pdf` | **Full thesis memory** — complete thesis document |
| `TFG_PinMartinez_ResumenEjecutivo.pdf` | **Executive summary** — condensed overview of the thesis |
| `TFG_PinMartinez_Presentacion_large_version.pdf` | **Defense presentation** — slides used for the oral defense |
| `Latex ZIP projects/` | LaTeX source files for all three documents |

---

## Thesis Summary

**Title**: *Aplicación de Deep Learning y Modelado Basado en Agentes para el Desarrollo de una Trampa Inteligente de Mosquitos*

**Institution**: Universidad Nacional de Asunción — Facultad de Ingeniería (FIUNA)

**Author**: Lucas Pin Martínez
---

## LaTeX Source

The `Latex ZIP projects/` folder contains the full LaTeX projects. To compile:

```bash
# Requires a LaTeX distribution (TeX Live, MiKTeX, etc.)
pdflatex TFG_PinMartinez.tex
bibtex TFG_PinMartinez
pdflatex TFG_PinMartinez.tex
pdflatex TFG_PinMartinez.tex
```

Or use **Overleaf** by uploading the ZIP file directly.
