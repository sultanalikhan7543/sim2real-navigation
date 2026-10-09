# Paper Source

This folder contains the LaTeX source and compiled PDF of the paper associated with this repository.

- `main.tex` — LaTeX source
- `main.pdf` — compiled PDF
- `figures/` — figure PDFs used by `main.tex`

## How to compile

**On Overleaf:**
1. Create a new project → "Upload Project"
2. Zip the entire `paper/` folder and upload
3. Overleaf compiles automatically

**Locally (requires pdflatex):**
```bash
cd paper
pdflatex main.tex
pdflatex main.tex