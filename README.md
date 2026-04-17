# Carbon Risk Simulator

A modular Monte Carlo simulator for carbon emissions and climate-economic risk using Nairobi as the default case study.

## Project structure

- `models.py` – emissions and economic cost models
- `simulation.py` – Monte Carlo scenario engine
- `visualization.py` – plotting utilities
- `main.py` – CLI entry point
- `tests/test_simulation.py` – focused behavior tests

## Install

```bash
python -m pip install numpy pandas matplotlib
```

## Run a sample simulation

```bash
python main.py --years 20 --simulations 10000 --reduction 20 --export-csv
```

Outputs are written to `outputs/`:
- `emissions_trajectory.png`
- `final_year_histogram.png`
- `yearly_summary.csv` (when `--export-csv` is set)

## CLI options

Use `python main.py --help` to see configurable assumptions (growth trend, shock distribution, damage factors, etc.).
