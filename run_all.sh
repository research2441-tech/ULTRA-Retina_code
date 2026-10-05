#!/bin/sh
# Full reproduction (about 9 hours on a two-core CPU). Order used for the paper.
set -e
sh setup_dirs.sh
python3 make_cohort.py 2026                       # RetinaSim-L cohort -> cohort.npz
for s in 0 1 2 3 4; do sh run_seed.sh $s; done     # pretraining, all models, ablations, robustness, policy
for s in 0 1 2; do                                 # sensitivity (three seeds)
  python3 train.py ULTRA $s nS12 hz=current nS=12;   python3 train.py ULTRA $s nS48 hz=current nS=48
  python3 train.py ULTRA $s lam005 hz=current lam_pred=0.05; python3 train.py ULTRA $s lam1 hz=current lam_pred=1.0
  python3 train.py ULTRA $s poct0 hz=current p_oct=0; python3 train.py ULTRA $s poct6 hz=current p_oct=0.6
  python3 train.py LTMD $s poct0 p_oct=0;             python3 train.py LTMD $s poct6 p_oct=0.6
done
python3 efficiency.py; python3 fig_architecture.py; python3 fig_cohort.py; python3 fig_trajectory.py
python3 analyze.py; python3 run_boot.py; python3 make_tables.py; python3 oracle.py
