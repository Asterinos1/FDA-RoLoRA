# Thesis Repository Map (`thesis-stuff`)

- **Local Path**: `C:\Users\GKaralis\Documents\GitHub\thesis-stuff`
- **Active Branch**: `thesis-fixes` (created from `main`)
- **Main TeX Entry**: `thesis/main.tex`

## Chapters
- `thesis/chapters/abstract.tex`: English & Greek Abstract (Περίληψη)
- `thesis/chapters/acknowledgements.tex`: Acknowledgements
- `thesis/chapters/abbreviations.tex`: Acronyms & Abbreviations
- `thesis/chapters/introduction.tex`: Chapter 1 (Introduction & Motivation)
- `thesis/chapters/background.tex`: Chapter 2 (Background & Related Work)
- `thesis/chapters/methodology.tex`: Chapter 3 (Proposed Methodology: FDA-RoLoRA)
- `thesis/chapters/implementation.tex`: Chapter 4 (System Implementation & Experimental Setup)
- `thesis/chapters/evaluation.tex`: Chapter 5 (Empirical Evaluation & Telemetry Analysis)
- `thesis/chapters/conclusion.tex`: Chapter 6 (Conclusions, Limitations & Future Work)
- `thesis/chapters/appendix_telemetry.tex`: Appendix A (Extended Telemetry & Benchmark Tables)

## Bibliography
- `thesis/references.bib`: BibTeX database

## Figures
- `thesis/figures/`: Standalone diagrams, parent stacked figures (`01_MNLI...` to `18_QNLI...`, `non_iid_benchmark_telemetry.png`, `non_iid_epoch_dynamics.png`)
- `thesis/figures/telemetry/`: Individual panel crops for telemetry figures

## Build Scripts
- `thesis/build.bat`:
  - `pdflatex -interaction=nonstopmode main.tex`
  - `bibtex main`
  - `pdflatex -interaction=nonstopmode main.tex`
  - `pdflatex -interaction=nonstopmode main.tex`
