!table tab:main | wide | 1.3,1,1,1,1,1,1 | Progression forecasting on the internal test set (mean $\pm$ s.d. over five seeds)
| Method | AUC$_{12}$ $\uparrow$ | AUC$_{24}$ $\uparrow$ | C-index $\uparrow$ | IBS $\downarrow$ | ECE$_{12}$ $\downarrow$ | NLL$_{12}$ $\downarrow$ |
| Landmark-LR | 0.665 $\pm$ 0.009 | 0.683 $\pm$ 0.010 | 0.642 $\pm$ 0.009 | 0.177 $\pm$ 0.002 | 0.034 $\pm$ 0.010 | 0.545 $\pm$ 0.005 |
| CS-Fusion | 0.705 $\pm$ 0.011 | 0.733 $\pm$ 0.014 | 0.679 $\pm$ 0.010 | 0.168 $\pm$ 0.004 | 0.032 $\pm$ 0.012 | 0.519 $\pm$ 0.009 |
| GRU-D | 0.743 $\pm$ 0.011 | !b0.773 $\pm$ 0.015 | 0.719 $\pm$ 0.010 | 0.163 $\pm$ 0.003 | 0.041 $\pm$ 0.010 | 0.506 $\pm$ 0.007 |
| LT-MD | 0.726 $\pm$ 0.010 | 0.748 $\pm$ 0.009 | 0.704 $\pm$ 0.008 | 0.168 $\pm$ 0.004 | 0.038 $\pm$ 0.015 | 0.517 $\pm$ 0.007 |
| LT-MD-MCD | 0.726 $\pm$ 0.010 | 0.748 $\pm$ 0.009 | 0.704 $\pm$ 0.008 | 0.168 $\pm$ 0.004 | 0.037 $\pm$ 0.017 | 0.517 $\pm$ 0.007 |
| LT-MD-Ens | 0.733 $\pm$ 0.005 | 0.753 $\pm$ 0.006 | 0.710 $\pm$ 0.005 | 0.166 $\pm$ 0.002 | 0.038 $\pm$ 0.013 | 0.512 $\pm$ 0.003 |
| CT-Filter | 0.712 $\pm$ 0.004 | 0.734 $\pm$ 0.007 | 0.688 $\pm$ 0.007 | 0.169 $\pm$ 0.002 | !b0.030 $\pm$ 0.009 | 0.518 $\pm$ 0.002 |
| ULTRA-Retina | !b0.746 $\pm$ 0.005 | 0.763 $\pm$ 0.010 | !b0.722 $\pm$ 0.004 | !b0.161 $\pm$ 0.001 | 0.032 $\pm$ 0.011 | !b0.500 $\pm$ 0.002 |
!note Best mean in bold. All learned methods share the frozen encoders, observation pooling, heads and modality dropout; 270 test eyes, 1047 landmarks with known 12-month status.

!table tab:secondary | col | 1.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8 | Diagnosis, severity and lesion localisation on the internal test set
| Method | Dx AUC | Dx BAcc | Dx MCC | QWK | Sev Acc | Les AP | Dice |
| Landmark-LR | 0.988 | 0.926 | 0.895 | 0.882 | 0.699 | 0.907 | 0.833 |
| CS-Fusion | 0.987 | 0.923 | 0.893 | 0.912 | 0.747 | 0.913 | 0.845 |
| GRU-D | 0.989 | 0.936 | 0.909 | !b0.933 | !b0.792 | !b0.919 | !b0.850 |
| LT-MD | 0.988 | 0.927 | 0.895 | 0.924 | 0.761 | 0.915 | 0.848 |
| LT-MD-Ens | 0.989 | 0.935 | 0.906 | 0.929 | 0.779 | 0.914 | 0.847 |
| CT-Filter | 0.989 | 0.929 | 0.896 | 0.919 | 0.727 | 0.915 | 0.846 |
| ULTRA-Retina | !b0.990 | !b0.938 | !b0.910 | 0.926 | 0.765 | 0.917 | 0.848 |
!note Means over five seeds; s.d. $\le$ 0.020 for all entries. Dx: diagnosis; QWK: severity quadratic-weighted kappa; Les: lesion grid.

!table tab:abl | wide | 2.3,1,1,1,1,1,1 | Component-removal ablation (mean over five seeds; change from the full model in parentheses)
| Variant | Test AUC$_{12}$ | Test IBS | Test ECE$_{12}$ | Test QWK | Centre E AUC$_{12}$ | Stale ECE$_{12}$ |
| Full model | 0.746 | 0.161 | 0.032 | 0.926 | 0.656 | 0.033 |
| w/o anatomy alignment ($\lambda_a=\lambda_o=0$) | 0.722 (-0.024) | 0.170 (+0.009) | 0.032 (+0.000) | 0.899 (-0.027) | 0.627 (-0.029) | 0.032 (-0.002) |
| w/o block factorisation | 0.749 (+0.004) | 0.163 (+0.001) | 0.039 (+0.007) | 0.927 (+0.000) | 0.662 (+0.006) | 0.030 (-0.004) |
| w/o evidence ageing ($\mathbf{Q}=\mathbf{0}$) | 0.737 (-0.009) | 0.164 (+0.003) | 0.042 (+0.010) | 0.928 (+0.002) | 0.656 (-0.001) | 0.030 (-0.003) |
| w/o velocity state | 0.702 (-0.044) | 0.173 (+0.012) | 0.049 (+0.017) | 0.926 (-0.001) | 0.637 (-0.019) | 0.038 (+0.005) |
| homoscedastic noise | 0.730 (-0.015) | 0.165 (+0.004) | 0.042 (+0.010) | 0.923 (-0.003) | 0.663 (+0.007) | 0.043 (+0.010) |
| w/o belief integration ($L=1$) | 0.743 (-0.003) | 0.163 (+0.002) | 0.038 (+0.006) | 0.928 (+0.001) | 0.662 (+0.006) | 0.039 (+0.006) |
| w/o innovation loss ($\lambda_p=0$) | 0.742 (-0.003) | 0.163 (+0.002) | 0.037 (+0.005) | 0.931 (+0.005) | 0.665 (+0.009) | 0.040 (+0.007) |
| w/o ordinal axis and $\psi_j$ | 0.742 (-0.003) | 0.162 (+0.001) | 0.034 (+0.002) | 0.929 (+0.003) | 0.668 (+0.012) | 0.029 (-0.004) |
!note Stale: landmarks of the test and external sets whose last OCT is older than 12 months or absent. Paired bootstrap intervals for the changes are given in the text.

!table tab:ext | wide | 1.3,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8 | Generalisation to external centres with unseen devices (means over five seeds)
| Method | D: AUC$_{12}$ | D: IBS | D: ECE$_{12}$ | D: QWK | E: AUC$_{12}$ | E: IBS | E: ECE$_{12}$ | E: QWK |
| Landmark-LR | 0.633 | 0.189 | 0.055 | 0.767 | 0.599 | 0.179 | 0.046 | 0.729 |
| CS-Fusion | 0.646 | 0.187 | !b0.035 | 0.801 | 0.626 | 0.177 | 0.044 | 0.804 |
| GRU-D | 0.666 | 0.189 | 0.046 | 0.831 | 0.650 | 0.179 | 0.059 | 0.862 |
| LT-MD | 0.669 | 0.186 | 0.041 | 0.832 | 0.658 | 0.176 | 0.050 | 0.843 |
| LT-MD-Ens | !b0.692 | !b0.180 | 0.035 | !b0.854 | !b0.674 | !b0.171 | 0.032 | 0.862 |
| CT-Filter | 0.628 | 0.195 | 0.050 | 0.813 | 0.642 | 0.176 | !b0.030 | 0.849 |
| ULTRA-Retina | 0.678 | 0.185 | 0.053 | 0.839 | 0.656 | 0.173 | 0.045 | !b0.869 |
!note Centre D: unseen device, OCT at 68% of visits. Centre E: unseen device, OCT at 23% of visits, median interval 11.2 months.

!table tab:stats | wide | 1.1,1.45,0.55,1.45,0.55,1.45,0.55,1.15 | Paired eye-level bootstrap comparison of ULTRA-Retina against each comparator (difference = ULTRA-Retina minus comparator)
| Comparator | $\Delta$AUC$_{12}$ [95% CI] | $p_{\mathrm{Holm}}$ | $\Delta$IBS [95% CI] | $p_{\mathrm{Holm}}$ | $\Delta$ECE$_{12}$ [95% CI] | $p_{\mathrm{Holm}}$ | Centre E $\Delta$AUC$_{12}$ ($p_{\mathrm{Holm}}$) |
| Landmark-LR | +0.080 [+0.040, +0.116] | 0.006 | -0.016 [-0.025, -0.007] | 0.006 | -0.002 [-0.014, +0.017] | 1.000 | +0.057 (0.024) |
| CS-Fusion | +0.041 [+0.016, +0.065] | 0.006 | -0.007 [-0.014, -0.001] | 0.048 | +0.000 [-0.015, +0.018] | 1.000 | +0.030 (0.140) |
| GRU-D | +0.003 [-0.012, +0.017] | 0.784 | -0.001 [-0.005, +0.002] | 0.468 | -0.008 [-0.016, +0.004] | 1.000 | +0.006 (1.000) |
| LT-MD | +0.019 [+0.002, +0.035] | 0.072 | -0.007 [-0.011, -0.002] | 0.008 | -0.006 [-0.015, +0.008] | 1.000 | -0.002 (1.000) |
| LT-MD-Ens | +0.013 [-0.004, +0.029] | 0.244 | -0.005 [-0.010, -0.001] | 0.048 | -0.006 [-0.018, +0.014] | 1.000 | -0.018 (0.160) |
| CT-Filter | +0.033 [+0.013, +0.055] | 0.006 | -0.008 [-0.014, -0.003] | 0.006 | +0.002 [-0.014, +0.017] | 1.000 | +0.014 (0.720) |
!note Internal test set unless stated; 1000 resamples of eyes, seed-averaged metrics, Holm correction across the six comparators. Positive $\Delta$AUC and negative $\Delta$IBS or $\Delta$ECE favour ULTRA-Retina.

!table tab:robust | wide | 1.3,0.9,0.9,0.9,0.9,1.25,1.0 | Robustness to image corruption and quality of the uncertainty score
| Method | Clean AUC$_{12}$ | Corrupted AUC$_{12}$ | Corrupted ECE$_{12}$ | Corrupted QWK | Flagging AUROC | AURC $\downarrow$ |
| CS-Fusion | 0.705 | 0.480 | 0.120 | 0.332 | 0.554 $\pm$ 0.010 | 0.1461 |
| GRU-D | 0.708 | 0.528 | 0.129 | 0.545 | 0.623 $\pm$ 0.039 | 0.1459 |
| LT-MD | 0.697 | 0.516 | 0.141 | 0.504 | 0.619 $\pm$ 0.024 | 0.1408 |
| LT-MD-MCD | 0.696 | 0.516 | 0.140 | 0.503 | 0.555 $\pm$ 0.039 | 0.1599 |
| CT-Filter | 0.671 | 0.521 | 0.105 | 0.596 | 0.612 $\pm$ 0.029 | 0.1486 |
| ULTRA-Retina | 0.721 | 0.551 | 0.117 | 0.642 | 0.640 $\pm$ 0.029 | 0.1655 |
| LT-MD-Ens | -- | -- | -- | -- | -- | 0.1622 |
!note Half of the internal test visits are corrupted (fundus blur, under-illumination and noise; OCT low signal and extra speckle) and re-encoded with the frozen encoder. Flagging AUROC: separation of corrupted from clean visits by the uncertainty score (Monte Carlo spread of $\rho_{12}$ for ULTRA-Retina and LT-MD-MCD, binary entropy of $\rho_{12}$ otherwise). AURC: area under the risk-coverage curve of the 12-month Brier loss over test and external landmarks.

!table tab:eff | col | 1.4,0.8,0.9,1.0 | Computational cost (CPU, one thread, batch of 64 eyes)
| Model | Params | ms / visit | State / eye |
| CS-Fusion | 195k | 0.037 | 0 |
| GRU-D | 347k | 0.052 | 386 |
| LT-MD | 511k | 0.051 | $512K$ |
| CT-Filter | 185k | 0.067 | 96 |
| ULTRA-Retina | 198k | 0.200 | 240 |
| Encoders $E_F+E_O$ | 313k | 5.47 | -- |
!note Stage-2 parameters exclude the shared frozen encoders (listed separately, without decoders). ULTRA-Retina latency includes 32-sample belief integration. State / eye: numbers that must be kept between visits; $K$ is the number of past visits (two-layer key-value cache for LT-MD).
