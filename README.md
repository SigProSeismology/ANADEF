# ANADEF-Hokkaido

**Alarm-based Nested-permutation Assessment of Dual-parameter Earthquake Forecasting**

Reference implementation and reproduction package for:

> Mohammadigheymasi, H., Miri, M., Tavakolizadeh, N., Pombo, N. *Does the b-value add forecasting
> information to the background seismicity rate? A nested-permutation protocol applied to Hokkaido, Japan.*
> (submitted)

ANADEF takes an earthquake catalogue and a configuration file and returns a calibrated spatial alarm
together with a quantitative statement of how much of its skill each predictor supplies. Its central element
is a **nested permutation test**: the rate threshold is held at its calibrated value while the *b* threshold
is re-selected by the same search inside every null realisation, so that the advantage conferred by threshold
calibration enters the null distribution instead of inflating the result.

---

Repository: <https://github.com/SigProSeismology/ANADEF>

## Quick start

```bash
git clone https://github.com/SigProSeismology/ANADEF.git
cd ANADEF
conda env create -f environment.yml        # or: pip install -r requirements.txt
conda activate anadef_v2

python main_anadef.py --config Configuration_Hokkaido_FINAL.json
```

One run takes roughly five minutes on a single core and writes every grid, table, figure and shapefile of
that configuration into the `output_dir` named in the config. All randomness is confined to two seeded
streams (cluster bootstrap, permutation nulls), so **a run is bit-identical given the catalogue, the
hyper-parameters and the seed**.

---

## Reproducing the paper

Each row below regenerates the results it names. Run them from the repository root.

| what it reproduces | command |
|---|---|
| Main run: Tables 1–3, Figs 3–6, S11–S14 | `python main_anadef.py --config Configuration_Hokkaido_FINAL.json` |
| Single regional *M*c = 3.0 (Table 4) | `python main_anadef.py --config Configuration_Hokkaido_FINAL_globalMc.json` |
| Training window 2004–2011 (Table 4) | `python main_anadef.py --config Configuration_Hokkaido_FINAL_train2004.json` |
| Thresholds calibrated on *M* ≥ 5.5 (Table 4) | `python main_anadef.py --config Configuration_Hokkaido_FINAL_M55.json` |
| Grid sweep, 0.5° (Supplementary S4) | `python main_anadef.py --config Configuration_Parameters_Hokkaido.json` |
| Grid sweep, 0.4° | `python main_anadef.py --config Configuration_Parameters_Hokkaido_grid04.json` |
| Grid sweep, 0.4°, *b* radius 0.56° | `python main_anadef.py --config Configuration_Parameters_Hokkaido_grid04_b056.json` |
| Grid sweep, 0.3° | `python main_anadef.py --config Configuration_Parameters_Hokkaido_grid03.json` |
| Grid sweep, 0.3°, *b* radius 0.42° | `python main_anadef.py --config Configuration_Parameters_Hokkaido_grid03_b042.json` |

Then rebuild the LaTeX tables of the manuscript:

```bash
python make_tables.py --main results/output_FINAL \
    --globalmc results/output_FINAL_globalMc \
    --train2004 results/output_FINAL_train2004 \
    --m55 results/output_FINAL_M55 \
    --out results/manuscript_tables
```

This writes `tab2.tex`, `tab3.tex`, `tab4.tex`, `tab_paired.tex` and `manuscript_numbers.json`.

Every configuration reported in the paper is in this repository, and no result in the paper comes from a
configuration that is not here.

---

## Layout

```
main_anadef.py              pipeline entry point
make_tables.py              builds the manuscript LaTeX tables from run outputs
make_hillshade.py           precomputes the topographic background used by the maps
Configuration_*.json        the nine configurations reported in the paper
src/
  data_loader.py            catalogue ingestion, column mapping, spatial grid
  mc_estimation.py          completeness magnitude, including the cell-wise plateau criterion
  b_value.py                Aki-Utsu b-value maps, Shi-Bolt errors, precision masks
  zhuang_declustering.py    ETAS maximum likelihood, stochastic declustering, background rate
  forecasting.py            alarms, joint calibration, Molchan metrics, bootstraps, permutation tests
  visualization.py          all figures
  shapefile_exporter.py     alarm polygons as ESRI shapefiles
config/config_parser.py     configuration reader
data/input/                 harmonised Hokkaido catalogue (gzipped CSV)
data/gis/                   coastline, active faults, borders, precomputed hillshade
results/                    numerical outputs of all nine runs, as reported
```

## Input catalogue

`data/input/Hokkaido_0-75km_2000-2023.csv.gz` — 192,630 earthquakes, 2000–2023, 0–75 km depth, inside
139.9–147.4°E / 41.0–46.2°N, derived from the **JMA unified hypocentre catalogue**
(<https://www.data.jma.go.jp/svd/eqev/data/bulletin/>).

Magnitudes are harmonised onto a single type-consistent moment-magnitude scale in the column
`mw_typecons`: velocity magnitudes shifted by −0.1 onto the displacement scale, displacement magnitudes
identified with *M*w. The 0.1 magnitude bin structure is preserved exactly, so the Utsu binning correction
applies unmodified. The columns `Mj`, `Mw`, `Mw_gr` and `Mw_legacy` are also present and can be selected
instead through `InputData.column_mapping.magnitude`.

## Results shipped here

`results/` contains the **numerical** outputs of all nine runs — `b_mu_grid_*.csv`, `table3_*.csv`,
`split_half_*.csv`, `figure4_*.csv`, `paired_bootstrap_*.csv`, `permutation_*.json`, `etas_params_*.json`,
`thresholds_*.json`, `run_summary.json` — plus the alarm shapefiles of the main run and the manuscript
tables.

Figures are **not** shipped: they are large and every one is regenerated by the command that produced its
numbers. Re-running a configuration writes its figures next to the numerical outputs already here, so a
reproduction can be diffed against the shipped files directly.

## Configuration

Every region-dependent quantity is a configuration entry, not a hard-coded constant: bounding box and cell
size, completeness settings, the *b*-value neighbourhood and its masks, the training/testing/forward windows,
the target magnitude, the ETAS space and time cut-offs and parameter bounds, the quantile grid, and the
bootstrap and permutation counts with their seed. Applying ANADEF to another region means editing a JSON
document and supplying a catalogue with origin time, epicentre and magnitude. No region-specific code path
exists.

## Third-party data

- **Coastline** `coast_gshhs_f.geojson` — GSHHG, full resolution (Wessel & Smith).
- **Active faults** `faults_rgafj1991.geojson` — Research Group for Active Faults of Japan (1991).
- **Hillshade** `hillshade.npz` — derived from 15-arc-second topography and bathymetry. The source DEM is
  not redistributed here; `make_hillshade.py` regenerates the hillshade from a local DEM if you want to
  rebuild it.

## Citation

See `CITATION.cff`.
