This file explains simple ways to compile ACM `acmart` documents on Windows (PowerShell) and options for Overleaf.

Compile locally (recommended tools)

- Install a TeX distribution if you don't have one: MiKTeX or TeX Live.
- Optional but convenient: install `latexmk` (bundled with many TeX distributions).

Using latexmk (recommended, single command):

```powershell
# From the repository root
latexmk -pdf starter.tex
# or target an existing sample
latexmk -pdf samples/sample-manuscript.tex
```

Manual (pdflatex + bibtex) — PowerShell commands:

```powershell
# 1. Run pdflatex
pdflatex starter.tex
# 2. Run bibtex
bibtex starter
# 3. Run pdflatex twice to resolve references
pdflatex starter.tex
pdflatex starter.tex
```

Notes

- Ensure the compilation working directory is the repository root so relative paths like `samples/sample-base.bib` resolve.
- If you prefer BibLaTeX, `acmart` has support for BibLaTeX but configuration differs; ask me and I can scaffold it.
- Overleaf: create a new project and upload the repository files (or link the repo); Overleaf has `acmart` supported and compiles automatically.

Common problems

- Missing packages: install them via MiKTeX/TeX Live package manager or enable on first compile (MiKTeX usually prompts).
- If using XeLaTeX or LuaLaTeX, change the engine in Overleaf or run `xelatex`/`lualatex` and ensure fonts are available.

If you want, I can also:

- Create a more fully featured template with sample author blocks, CCSXML, and DOI/license metadata.
- Add a small CI workflow to compile the PDF automatically on push.
