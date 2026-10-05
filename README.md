# ULTRA-Retina: evidence-ageing belief filtering for retinal disease progression forecasting

Code, simulator and per-seed results for the paper *Evidence-Ageing Belief Filtering of Anatomy-Aligned Fundus and OCT Representations for Retinal Disease Progression Forecasting Under Sporadic Imaging*.

Everything sits in this one folder. Scripts create `runs/`, `logs/` and `figs/` when they run.

## Quick start

```bash
pip install -r requirements.txt
# optional: unzip ULTRA-Retina_predictions.zip and copy its pred_*.npz files into this folder
sh setup_dirs.sh                      # moves the shipped result files into runs/
python3 make_cohort.py 2026           # regenerates the simulated cohort (cohort.npz, about 280 MB)
python3 analyze.py                    # tables and Figs. 3, 4, 6 and 7 from the shipped predictions
python3 make_tables.py                # table markup -> runs/tables.md
```

`sh run_all.sh` reproduces every experiment from scratch. It takes about nine hours on a two-core CPU.

## Files

| File | Purpose |
|---|---|
| `simulator.py`, `make_cohort.py` | RetinaSim-L. Generates the latent disease course, biomarkers, fundus (3x64x64) and OCT (8x32x32) rendering, the five centres/devices, acquisition masks and confirmed progression labels |
| `data.py` | Loads the cohort and makes the eye-level splits (train/val/cal/test from centres A-C; D and E are external) |
| `encoders.py`, `pretrain.py` | Stage 1. Fundus and OCT encoders with shared/private tokens; masked reconstruction, cell-level InfoNCE and decorrelation; token caching |
| `models.py` | Stage 2. ULTRA-Retina belief filter and hazard, plus the CS-Fusion, GRU-D, LT-MD and CT-Filter baselines |
| `train.py` | Training and inference. `python3 train.py MODEL SEED TAG key=value ...` |
| `landmark_lr.py` | Classical landmark discrete-time hazard baseline |
| `robustness.py` | OCT-availability sweep, staleness strata, corruption, MC dropout, deep ensemble |
| `policy.py` | OCT acquisition rules: value (Monte Carlo preposterior), closed-form approximation, uncertainty, risk, random, interval |
| `evaluate.py`, `metrics.py`, `run_boot.py` | Metrics, seed aggregation, paired eye-level bootstrap with Holm correction |
| `analyze.py`, `make_tables.py`, `oracle.py`, `efficiency.py` | Result aggregation, tables, reference ceilings, cost |
| `fig_architecture.py`, `fig_cohort.py`, `fig_trajectory.py` | Figs. 1, 2 and 5. Fig. 1 includes an automated overlap and crossing audit |
| `run_seed.sh`, `run_all.sh`, `setup_dirs.sh` | Experiment drivers |
| `pred_<model>_<tag>_s<seed>.npz` (in the separate predictions zip) | Per-visit predictions (survival curve, diagnosis, severity, lesion grid, risk spread) for the main models, ablations |
| `results.json`, `boot_*.json`, `oracle.json`, `efficiency.json`, `octsweep_*.json`, `staleness_*.json`, `corrupt_*.json`, `policy_test_*.json` | Every number reported in the paper |
| `policy_val_s0_partial.log` | Validation run used to choose the Monte Carlo acquisition value |

## Notes

- All data are simulated. The paper's Limitations section discusses what this means for external validity.
- Model selection used only the validation split: the hazard variant was chosen on seeds 0 and 1, and the acquisition estimator on seed 0.
- Seeds 0-4 vary encoder pretraining, stage-2 initialisation and modality dropout. The data partition is fixed (seed 7).
