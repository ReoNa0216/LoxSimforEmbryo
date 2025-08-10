# Stochastic Agent-Based LoxCode Embryo Simulator

Fast, parallel simulation of LoxCode-barcoded embryonic development with downstream fate-coupling analysis and heatmap visualization.

## Features
- Agent-based growth from 4-cell stage with lognormal division times
- Time-configurable LoxCode barcoding (inversion/excision with length constraints)
- Probabilistic tissue fate assignment via transition matrix and split times
- Numba-accelerated batch fate updates
- Multiprocessing for many embryos in parallel
- COSPAR-based fate coupling on per-embryo and concatenated matrices
- R heatmaps for tissue groups and germ-layer aggregations

## Repository layout
- python/loxcode_sim_optimize.py — main simulator and analysis
- R/Rscript/plot_heatmap.R — tissue–barcode heatmaps
- R/Rscript/Correlation_heatmap.R — fate-coupling (per-embryo + combined) heatmaps
- <timestamp>/ — output folders created per run (YYYY-MM-DD_HH-MM-SS)

## Installation (Windows)
```powershell
# Create and activate venv
py -m venv .venv
.venv\Scripts\activate

# Core deps
pip install numpy numba tqdm pandas

# Analysis (optional)
pip install scanpy anndata cospar
```

## Quick start
```powershell
# Simulate 100 embryos with analysis
python python/loxcode_sim_optimize.py -n 100 --barcoding 132 --collection 228 --analyze

# Limit CPU cores
python python/loxcode_sim_optimize.py -n 100 -p 8 --analyze

# Single embryo (debug)
python python/loxcode_sim_optimize.py -n 1 --analyze
```
Each run creates a timestamped folder with CSV outputs.

## Outputs
- loxcode_full_pedigree_embryo{1..3}.csv — division lineage (first 3 embryos)
- loxcode_census_at_barcoding_embryo{1..3}.csv — census at barcoding
- loxcode_census_at_sampling[_embryoX].csv — final census for all embryos
- results_embryo{1..3}_tissue_barcode_matrix.csv — per-embryo matrices
- results_embryo{1..3}_fate_coupling.csv — per-embryo coupling (COSPAR)
- results_concatenated_tissue_barcode_matrix.csv — combined matrix
- results_mcsa_matrix.csv — combined fate-coupling matrix

## Analysis and plotting
Python (optional, in same output folder):
```powershell
python python/loxcode_sim_optimize.py -n 100 --analyze
```

R (set working directory to the output folder or edit paths in scripts):
```r
# Tissue–barcode + aggregated heatmaps
source("R/Rscript/plot_heatmap.R")

# Fate-coupling heatmaps (per-embryo + combined)
source("R/Rscript/Correlation_heatmap.R")
```
Required R packages: ggplot2, pheatmap, gridExtra, RColorBrewer, grid, reshape2.

## Notes and tips
- Seeds: stochastic components seed from system time; set seeds in code if strict reproducibility is required.
- Performance: increase -p up to CPU count; COSPAR can be memory-intensive.
- Paths: R scripts assume relative filenames; adjust if running from a different directory.

# Acknowledgments
This project was inspired in part by the methodology described in:
- Weber, T.S., Biben, C., Miles, D.C., Glaser, S.P., Tomei, S., Lin, C -Y., Kueh, A., Pal, M., Zhang, S., Tam, P.P.L., Taoudi, S., Naik, S.H. (2025). LoxCode in vivo barcoding reveals epiblast clonal fate bias to fetal organs. Cell. https://doi.org/10.1016/j.cell.2025.04.026
