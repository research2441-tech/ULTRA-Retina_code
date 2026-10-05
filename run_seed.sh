#!/bin/sh
# Full experiment set for one seed. Usage: sh run_seed.sh SEED
s=$1
cd "$(dirname "$0")"
mkdir -p runs logs figs
[ -f tok_full_s$s.npz ] || python3 pretrain.py $s full 10 > logs/pre_full_s$s.log 2>&1
[ -f tok_noalign_s$s.npz ] || python3 pretrain.py $s noalign 10 > logs/pre_noalign_s$s.log 2>&1
tr() { n=$1; t=$2; sd=$3; shift 3; [ -f runs/pred_${n}_${t}_s${sd}.npz ] || python3 train.py $n $sd $t "$@" > logs/${n}_${t}_s${sd}.log 2>&1; }
tr ULTRA main $s hz=current
for m in CTFilter LTMD GRUD CSFusion; do tr $m main $s; done
[ -f runs/pred_LandmarkLR_main_s$s.npz ] || python3 landmark_lr.py $s
for j in 1 2 3 4; do tr LTMD ens $((s + 10 * j)) tokseed=$s; done
tr ULTRA no_align $s hz=current tok=noalign
tr ULTRA no_factor $s hz=current factor=0
tr ULTRA no_aging $s hz=current aging=0
tr ULTRA no_velocity $s hz=current velocity=0
tr ULTRA homosc $s hz=current hetero=0
tr ULTRA no_integrate $s hz=current integrate=0
tr ULTRA no_innov $s hz=current lam_pred=0
tr ULTRA no_axis $s hz=current axis=0
python3 robustness.py ens $s > logs/ens_s$s.log 2>&1
python3 robustness.py mcd $s > logs/mcd_s$s.log 2>&1
python3 robustness.py sweep $s > logs/sweep_s$s.log 2>&1
python3 robustness.py stale $s > logs/stale_s$s.log 2>&1
python3 robustness.py corrupt $s > logs/corrupt_s$s.log 2>&1
python3 policy.py $s test > logs/policy_s$s.log 2>&1
echo done > logs/seed$s.done
