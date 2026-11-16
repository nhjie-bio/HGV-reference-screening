# HGV-V3 E6-E12 Scientific Result Audit

- Accepted final tag: `20260925_v6`
- Generated: `2026-09-26T14:50:22+08:00`
- Existing scientific outputs were read only.
- This report distinguishes **pipeline completion** from **claim/evidence completeness**.

## E6 — D0-D5 ablation and formal statistics

### Mode-level test metrics
| mode | n_seeds | AUROC_mean | AUROC_sd | AUPRC_mean | AUPRC_sd | MCC_mean | MCC_sd | F1_mean | F1_sd | BalancedAccuracy_mean | BalancedAccuracy_sd | threshold_mean | threshold_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 5 | 0.970447 | 0.017597 | 0.970143 | 0.018262 | 0.796954 | 0.000000 | 0.899628 | 0.000000 | 0.888430 | 0.000000 | 0.896407 | 0.036197 |
| D1 | 5 | 0.975002 | 0.010630 | 0.974922 | 0.010879 | 0.796954 | 0.000000 | 0.899628 | 0.000000 | 0.888430 | 0.000000 | 0.801262 | 0.027799 |
| D2 | 5 | 0.960488 | 0.022014 | 0.958977 | 0.022849 | 0.796771 | 0.005858 | 0.899666 | 0.002606 | 0.888636 | 0.002936 | 0.886706 | 0.027682 |
| D3 | 5 | 0.969293 | 0.014123 | 0.968529 | 0.015122 | 0.794289 | 0.005486 | 0.898517 | 0.002258 | 0.887293 | 0.002263 | 0.796772 | 0.024990 |
| D4 | 5 | 0.960843 | 0.019214 | 0.957719 | 0.022641 | 0.794709 | 0.004677 | 0.898704 | 0.001926 | 0.887500 | 0.001938 | 0.861615 | 0.040927 |
| D5 | 5 | 0.963019 | 0.017767 | 0.960652 | 0.020606 | 0.796954 | 0.000000 | 0.899628 | 0.000000 | 0.888430 | 0.000000 | 0.648988 | 0.021409 |

### D3 paired primary contrasts with species-block bootstrap + Holm
| comparison | metric | observed_delta_mean_5seed | bootstrap_mean | ci95_low | ci95_high | p_raw_two_sided | bootstrap_n | bootstrap_unit | p_holm | direction_supported | holm_significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3-D0 | AUROC | -0.001154 | -0.000894 | -0.013385 | 0.009825 | 0.909091 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D0 | AUPRC | -0.001614 | -0.001573 | -0.016882 | 0.008612 | 0.909091 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D1 | AUROC | -0.005710 | -0.004500 | -0.019509 | 0.000993 | 0.509491 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D1 | AUPRC | -0.006393 | -0.005335 | -0.024652 | 0.000781 | 0.507493 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D2 | AUROC | 0.008804 | 0.006356 | -0.000731 | 0.023914 | 0.643357 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D2 | AUPRC | 0.009552 | 0.006650 | -4.400e-07 | 0.025590 | 0.603397 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D4 | AUROC | 0.008450 | 0.006426 | -2.061e-07 | 0.023562 | 0.413586 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D4 | AUPRC | 0.010810 | 0.008255 | -1.581e-07 | 0.031147 | 0.411588 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D5 | AUROC | 0.006273 | 0.005163 | -0.002620 | 0.024603 | 0.781219 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |
| D3-D5 | AUPRC | 0.007877 | 0.006477 | -0.003221 | 0.032753 | 0.785215 | 1000 | two-way species-cluster multiplicity | 1.000000 | 0 | 0 |

### Final claim decision
| comparison | AUROC_delta | AUROC_CI_low | AUROC_CI_high | AUROC_p_holm | AUPRC_delta | AUPRC_CI_low | AUPRC_CI_high | AUPRC_p_holm | supports_strict_improvement_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3-D0 | -0.001154 | -0.013385 | 0.009825 | 1.000000 | -0.001614 | -0.016882 | 0.008612 | 1.000000 | 0 |
| D3-D1 | -0.005710 | -0.019509 | 0.000993 | 1.000000 | -0.006393 | -0.024652 | 0.000781 | 1.000000 | 0 |
| D3-D2 | 0.008804 | -0.000731 | 0.023914 | 1.000000 | 0.009552 | -4.400e-07 | 0.025590 | 1.000000 | 0 |
| D3-D4 | 0.008450 | -2.061e-07 | 0.023562 | 1.000000 | 0.010810 | -1.581e-07 | 0.031147 | 1.000000 | 0 |
| D3-D5 | 0.006273 | -0.002620 | 0.024603 | 1.000000 | 0.007877 | -0.003221 | 0.032753 | 1.000000 | 0 |

### True-teacher fidelity summary
| mode | stratum | n_seeds | spearman_mean | spearman_sd | pearson_mean | pearson_sd | mse_mean | mse_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | all | 5 | 0.896797 | 0.019441 | 0.686708 | 0.022117 | 0.291189 | 0.020674 |
| D0 | informative_negative | 5 | 0.065608 | 0.144738 | 0.080892 | 0.147650 | 0.794158 | 0.030217 |
| D0 | negative | 5 | 0.740480 | 0.045127 | 0.849331 | 0.045649 | 0.350989 | 0.038710 |
| D0 | pair_type:easy | 5 | 0.124584 | 0.010825 | 0.128442 | 0.026635 | 0.209595 | 0.049341 |
| D0 | pair_type:hard | 5 | 0.065608 | 0.144738 | 0.080892 | 0.147650 | 0.794158 | 0.030217 |
| D0 | pair_type:medium | 5 | 0.262648 | 0.132055 | 0.160875 | 0.160851 | 0.356877 | 0.112829 |
| D0 | pair_type:positive | 5 | 0.583836 | 0.031674 | 0.490159 | 0.069076 | 0.231388 | 0.002984 |
| D0 | positive | 5 | 0.583836 | 0.031674 | 0.490159 | 0.069076 | 0.231388 | 0.002984 |
| D1 | all | 5 | 0.896152 | 0.014972 | 0.716384 | 0.008276 | 0.254221 | 0.006718 |
| D1 | informative_negative | 5 | 0.025227 | 0.259794 | 0.000673 | 0.285134 | 0.752934 | 0.017441 |
| D1 | negative | 5 | 0.700920 | 0.063816 | 0.908083 | 0.017834 | 0.287905 | 0.013135 |
| D1 | pair_type:easy | 5 | 0.073150 | 0.051416 | 0.062826 | 0.060631 | 0.149231 | 0.010792 |
| D1 | pair_type:hard | 5 | 0.025227 | 0.259794 | 0.000673 | 0.285134 | 0.752934 | 0.017441 |
| D1 | pair_type:medium | 5 | 0.173960 | 0.200153 | 0.099370 | 0.302580 | 0.202520 | 0.101273 |
| D1 | pair_type:positive | 5 | 0.582489 | 0.035930 | 0.481116 | 0.041549 | 0.220537 | 0.000796 |
| D1 | positive | 5 | 0.582489 | 0.035930 | 0.481116 | 0.041549 | 0.220537 | 0.000796 |
| D2 | all | 5 | 0.882846 | 0.024090 | 0.701145 | 0.011024 | 0.273639 | 0.018620 |
| D2 | informative_negative | 5 | 0.119023 | 0.072876 | 0.125761 | 0.090275 | 0.784848 | 0.024648 |
| D2 | negative | 5 | 0.678140 | 0.082917 | 0.884202 | 0.036954 | 0.320792 | 0.034844 |
| D2 | pair_type:easy | 5 | 0.099474 | 0.035616 | 0.087153 | 0.046668 | 0.184252 | 0.041280 |
| D2 | pair_type:hard | 5 | 0.119023 | 0.072876 | 0.125761 | 0.090275 | 0.784848 | 0.024648 |
| D2 | pair_type:medium | 5 | 0.257623 | 0.072038 | 0.234595 | 0.072664 | 0.218158 | 0.164258 |
| D2 | pair_type:positive | 5 | 0.577665 | 0.020691 | 0.478465 | 0.062739 | 0.226487 | 0.002418 |
| D2 | positive | 5 | 0.577665 | 0.020691 | 0.478465 | 0.062739 | 0.226487 | 0.002418 |
| D3 | all | 5 | 0.893501 | 0.010888 | 0.715438 | 0.011366 | 0.256615 | 0.014796 |
| D3 | informative_negative | 5 | 0.036636 | 0.211672 | 0.029089 | 0.208316 | 0.751641 | 0.035999 |
| D3 | negative | 5 | 0.718400 | 0.047175 | 0.903581 | 0.017976 | 0.294343 | 0.025515 |
| D3 | pair_type:easy | 5 | 0.064151 | 0.038350 | 0.038481 | 0.032504 | 0.157166 | 0.023352 |
| D3 | pair_type:hard | 5 | 0.036636 | 0.211672 | 0.029089 | 0.208316 | 0.751641 | 0.035999 |
| D3 | pair_type:medium | 5 | 0.207182 | 0.133068 | 0.231729 | 0.152801 | 0.218002 | 0.070858 |
| D3 | pair_type:positive | 5 | 0.581137 | 0.020758 | 0.455687 | 0.024796 | 0.218888 | 0.004217 |
| D3 | positive | 5 | 0.581137 | 0.020758 | 0.455687 | 0.024796 | 0.218888 | 0.004217 |
| D4 | all | 5 | 0.878701 | 0.014382 | 0.699267 | 0.008071 | 0.284651 | 0.013848 |
| D4 | informative_negative | 5 | 0.104049 | 0.183184 | 0.110844 | 0.185539 | 0.791297 | 0.016462 |
| D4 | negative | 5 | 0.659432 | 0.054080 | 0.884103 | 0.019788 | 0.341876 | 0.025732 |
| D4 | pair_type:easy | 5 | 0.067913 | 0.029482 | 0.069131 | 0.053902 | 0.213416 | 0.028164 |
| D4 | pair_type:hard | 5 | 0.104049 | 0.183184 | 0.110844 | 0.185539 | 0.791297 | 0.016462 |
| D4 | pair_type:medium | 5 | 0.105309 | 0.096116 | -0.001919 | 0.109250 | 0.206842 | 0.086084 |
| D4 | pair_type:positive | 5 | 0.567475 | 0.029132 | 0.424589 | 0.070352 | 0.227426 | 0.002456 |
| D4 | positive | 5 | 0.567475 | 0.029132 | 0.424589 | 0.070352 | 0.227426 | 0.002456 |
| D5 | all | 5 | 0.904241 | 0.012091 | 0.665368 | 0.013645 | 0.661610 | 0.218644 |
| D5 | informative_negative | 5 | -0.028285 | 0.261836 | -0.013533 | 0.272199 | 0.690396 | 0.028022 |
| D5 | negative | 5 | 0.774919 | 0.010121 | 0.798342 | 0.024400 | 1.127006 | 0.438050 |
| D5 | pair_type:easy | 5 | 0.123137 | 0.012600 | 0.129703 | 0.013413 | 1.383316 | 0.613781 |
| D5 | pair_type:hard | 5 | -0.028285 | 0.261836 | -0.013533 | 0.272199 | 0.690396 | 0.028022 |
| D5 | pair_type:medium | 5 | 0.380180 | 0.087135 | 0.304580 | 0.109755 | 0.016127 | 0.029154 |
| D5 | pair_type:positive | 5 | 0.644910 | 0.021784 | 0.567216 | 0.014466 | 0.196215 | 0.001458 |
| D5 | positive | 5 | 0.644910 | 0.021784 | 0.567216 | 0.014466 | 0.196215 | 0.001458 |

## E7 — OFAT sensitivity

| setting | n_seeds | test_AUROC_mean | test_AUROC_sd | test_AUPRC_mean | test_AUPRC_sd | test_MCC_mean | test_MCC_sd | test_F1_mean | test_F1_sd | test_BalancedAccuracy_mean | test_BalancedAccuracy_sd | input_dim | embed_dim | margin | lambda_dist | delta_test_AUROC_vs_canonical | delta_test_AUPRC_vs_canonical | delta_test_MCC_vs_canonical | delta_test_F1_vs_canonical | delta_test_BalancedAccuracy_vs_canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_k6_dim64_margin1_lambda0p25 | 3 | 0.973128 | 0.011293 | 0.972180 | 0.013014 | 0.796665 | 0.000500 | 0.899489 | 0.000241 | 0.888258 | 0.000298 | 2080 | 64 | 1.000000 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| k5 | 3 | 0.498607 | 3.190e-05 | 0.499249 | 8.197e-06 | NA | NA | NA | NA | 0.500000 | 0.000000 | 512 | 64 | 1.000000 | 0.250000 | -0.474521 | -0.472931 | NA | NA | -0.388258 |
| k7 | 3 | 0.500000 | 0.000000 | 0.500000 | 0.000000 | NA | NA | NA | NA | 0.500000 | 0.000000 | 8192 | 64 | 1.000000 | 0.250000 | -0.473128 | -0.472180 | NA | NA | -0.388258 |
| dim32 | 3 | 0.958564 | 0.014159 | 0.957893 | 0.014995 | NA | NA | NA | NA | 0.500000 | 0.000000 | 2080 | 32 | 1.000000 | 0.250000 | -0.014564 | -0.014287 | NA | NA | -0.388258 |
| dim128 | 3 | 0.943182 | 0.022712 | 0.937582 | 0.029466 | NA | NA | NA | NA | 0.500000 | 0.000000 | 2080 | 128 | 1.000000 | 0.250000 | -0.029946 | -0.034598 | NA | NA | -0.388258 |
| margin0p5 | 3 | 0.962754 | 0.006888 | 0.962285 | 0.008140 | NA | NA | NA | NA | 0.500000 | 0.000000 | 2080 | 64 | 0.500000 | 0.250000 | -0.010374 | -0.009895 | NA | NA | -0.388258 |
| margin1p5 | 3 | 0.968628 | 0.020073 | 0.967225 | 0.022055 | NA | NA | NA | NA | 0.500000 | 0.000000 | 2080 | 64 | 1.500000 | 0.250000 | -0.004500 | -0.004955 | NA | NA | -0.388258 |
| lambda0p1 | 3 | 0.955097 | 0.021081 | 0.951909 | 0.024778 | NA | NA | NA | NA | 0.500000 | 0.000000 | 2080 | 64 | 1.000000 | 0.100000 | -0.018031 | -0.020271 | NA | NA | -0.388258 |
| lambda0p5 | 3 | 0.960938 | 0.029287 | 0.959448 | 0.033992 | NA | NA | NA | NA | 0.500000 | 0.000000 | 2080 | 64 | 1.000000 | 0.500000 | -0.012190 | -0.012732 | NA | NA | -0.388258 |

Canonical baseline is the frozen E6 D3 k=6 / d=64 / margin=1 / lambda=0.25 result; it was not retrained in E7.

## E8 — Fragmentation robustness

| condition | n_seeds | AUROC_mean | AUROC_sd | AUPRC_mean | AUPRC_sd | MCC_mean | MCC_sd | F1_mean | F1_sd | BalancedAccuracy_mean | BalancedAccuracy_sd | delta_AUROC_vs_full | delta_AUPRC_vs_full | delta_MCC_vs_full | delta_F1_vs_full | delta_BalancedAccuracy_vs_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 3 | 0.973128 | 0.011293 | 0.972180 | 0.013014 | 0.796665 | 0.000500 | 0.899489 | 0.000241 | 0.888258 | 0.000298 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 100kb | 3 | 0.885213 | 0.007735 | 0.883419 | 0.008448 | 0.349843 | 0.059367 | 0.391682 | 0.115408 | 0.618974 | 0.040185 | -0.087915 | -0.088760 | -0.446822 | -0.507806 | -0.269284 |
| 50kb | 3 | 0.821047 | 0.018798 | 0.819556 | 0.008712 | 0.240072 | 0.051532 | 0.240661 | 0.094180 | 0.564394 | 0.027649 | -0.152081 | -0.152623 | -0.556593 | -0.658828 | -0.323864 |
| 20kb | 3 | 0.745692 | 0.039743 | 0.757398 | 0.034050 | 0.135569 | 0.056029 | 0.088772 | 0.067066 | 0.522297 | 0.017716 | -0.227437 | -0.214782 | -0.661096 | -0.810717 | -0.365961 |
| 5kb | 3 | 0.616974 | 0.016806 | 0.629517 | 0.009494 | 0.055846 | 0.007332 | 0.016690 | 0.008176 | 0.503788 | 0.001557 | -0.356154 | -0.342663 | -0.740819 | -0.882799 | -0.384470 |
| 5x5kb | 3 | 0.797183 | 0.011542 | 0.800263 | 0.012199 | 0.175870 | 0.061257 | 0.132036 | 0.080612 | 0.534607 | 0.022673 | -0.175945 | -0.171917 | -0.620795 | -0.767453 | -0.353650 |

Frozen full-assembly models are reused; no fragmentation-specific retraining or threshold retuning.

## E9 — Explainability and confound audit

- Undefined Spearman rows due to constant/non-estimable input: **1**.
- Undefined correlations are retained as NA and must not be replaced by 0.
- Attribution target is the frozen G2 D3 seed20260915 model; this is not a multi-seed attribution analysis.

### Confound correlations
| subset | confound | n | spearman_r | p_value |
| --- | --- | --- | --- | --- |
| all | abs_delta_gc | 3872 | 0.878048 | 0.000000 |
| all | abs_delta_length | 3872 | 0.765541 | 0.000000 |
| all | abs_delta_contigs | 3872 | 0.336075 | 7.305e-103 |
| all | abs_delta_n50 | 3872 | 0.648197 | 0.000000 |
| all | abs_delta_completeness | 3872 | 0.249140 | 7.232e-56 |
| positive | abs_delta_gc | 1936 | 0.590011 | 6.198e-182 |
| positive | abs_delta_length | 1936 | 0.500498 | 2.908e-123 |
| positive | abs_delta_contigs | 1936 | 0.245087 | 7.086e-28 |
| positive | abs_delta_n50 | 1936 | 0.398828 | 7.988e-75 |
| positive | abs_delta_completeness | 1936 | 0.141388 | 4.145e-10 |
| negative | abs_delta_gc | 1936 | 0.874036 | 0.000000 |
| negative | abs_delta_length | 1936 | 0.484775 | 1.173e-114 |
| negative | abs_delta_contigs | 1936 | 0.126599 | 2.280e-08 |
| negative | abs_delta_n50 | 1936 | 0.383241 | 9.201e-69 |
| negative | abs_delta_completeness | 1936 | 0.309518 | 2.971e-44 |
| pair_type:easy | abs_delta_gc | 1360 | 0.750452 | 1.685e-246 |
| pair_type:easy | abs_delta_length | 1360 | 0.075861 | 0.005125 |
| pair_type:easy | abs_delta_contigs | 1360 | -0.025175 | 0.353559 |
| pair_type:easy | abs_delta_n50 | 1360 | -0.038318 | 0.157856 |
| pair_type:easy | abs_delta_completeness | 1360 | -0.023117 | 0.394308 |
| pair_type:hard | abs_delta_gc | 432 | 0.247019 | 1.992e-07 |
| pair_type:hard | abs_delta_length | 432 | 0.771614 | 1.622e-86 |
| pair_type:hard | abs_delta_contigs | 432 | 0.393049 | 2.070e-17 |
| pair_type:hard | abs_delta_n50 | 432 | -0.148613 | 0.001954 |
| pair_type:hard | abs_delta_completeness | 432 | NA | NA |
| pair_type:medium | abs_delta_gc | 144 | 0.349349 | 1.773e-05 |
| pair_type:medium | abs_delta_length | 144 | 0.161406 | 0.053275 |
| pair_type:medium | abs_delta_contigs | 144 | -0.119358 | 0.154183 |
| pair_type:medium | abs_delta_n50 | 144 | -0.035455 | 0.673106 |
| pair_type:medium | abs_delta_completeness | 144 | 0.455756 | 9.465e-09 |
| pair_type:positive | abs_delta_gc | 1936 | 0.590011 | 6.198e-182 |
| pair_type:positive | abs_delta_length | 1936 | 0.500498 | 2.908e-123 |
| pair_type:positive | abs_delta_contigs | 1936 | 0.245087 | 7.086e-28 |
| pair_type:positive | abs_delta_n50 | 1936 | 0.398828 | 7.988e-75 |
| pair_type:positive | abs_delta_completeness | 1936 | 0.141388 | 4.145e-10 |

### Top-20 integrated-gradient canonical 6-mers
| rank | canonical_kmer | feature_index | mean_abs_pair_IG |
| --- | --- | --- | --- |
| 1 | TAAAGA | 1946 | 0.033065 |
| 2 | AAAAAT | 3 | 0.029584 |
| 3 | AAAAAA | 0 | 0.029196 |
| 4 | AACATC | 77 | 0.027649 |
| 5 | AGATGG | 536 | 0.027268 |
| 6 | TTAAAA | 2070 | 0.026204 |
| 7 | TAAAAA | 1944 | 0.025604 |
| 8 | AAATAA | 48 | 0.024984 |
| 9 | ATAAAA | 702 | 0.021462 |
| 10 | GAACAA | 1560 | 0.021397 |
| 11 | AAGATG | 141 | 0.020939 |
| 12 | ACAATC | 263 | 0.020778 |
| 13 | AAAATT | 15 | 0.019658 |
| 14 | CCGGCG | 1204 | 0.019306 |
| 15 | ATGAGG | 814 | 0.019023 |
| 16 | AATAAA | 189 | 0.018907 |
| 17 | GTTCCA | 1933 | 0.018703 |
| 18 | CATTTA | 1088 | 0.018629 |
| 19 | TTGAAA | 2077 | 0.018297 |
| 20 | AGCGAT | 571 | 0.018260 |

### Top-20 permutation effects ranked by AUROC drop
| ig_rank | feature_index | canonical_kmer | baseline_AUROC | permuted_AUROC | AUROC_drop | baseline_AUPRC | permuted_AUPRC | AUPRC_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 189 | AATAAA | 0.963259 | 0.908669 | 0.054590 | 0.960439 | 0.882078 | 0.078361 |
| 9 | 702 | ATAAAA | 0.963259 | 0.913074 | 0.050185 | 0.960439 | 0.893983 | 0.066456 |
| 6 | 2070 | TTAAAA | 0.963259 | 0.914020 | 0.049239 | 0.960439 | 0.891197 | 0.069242 |
| 7 | 1944 | TAAAAA | 0.963259 | 0.922560 | 0.040699 | 0.960439 | 0.900319 | 0.060120 |
| 2 | 3 | AAAAAT | 0.963259 | 0.924367 | 0.038892 | 0.960439 | 0.903687 | 0.056752 |
| 13 | 15 | AAAATT | 0.963259 | 0.925518 | 0.037742 | 0.960439 | 0.911594 | 0.048846 |
| 40 | 1665 | GATGGC | 0.963259 | 0.925596 | 0.037663 | 0.960439 | 0.913098 | 0.047341 |
| 3 | 0 | AAAAAA | 0.963259 | 0.925776 | 0.037483 | 0.960439 | 0.904879 | 0.055560 |
| 33 | 1202 | CCGGCA | 0.963259 | 0.927375 | 0.035884 | 0.960439 | 0.914561 | 0.045878 |
| 14 | 1204 | CCGGCG | 0.963259 | 0.927520 | 0.035739 | 0.960439 | 0.920821 | 0.039618 |
| 27 | 2 | AAAAAG | 0.963259 | 0.931216 | 0.032043 | 0.960439 | 0.917097 | 0.043343 |
| 39 | 1117 | CCAGCG | 0.963259 | 0.931868 | 0.031391 | 0.960439 | 0.922947 | 0.037492 |
| 8 | 48 | AAATAA | 0.963259 | 0.932250 | 0.031009 | 0.960439 | 0.918645 | 0.041795 |
| 36 | 1544 | CTTTAA | 0.963259 | 0.933910 | 0.029349 | 0.960439 | 0.923101 | 0.037338 |
| 32 | 855 | ATTAAA | 0.963259 | 0.933919 | 0.029340 | 0.960439 | 0.920367 | 0.040072 |
| 25 | 484 | AGAAAA | 0.963259 | 0.935480 | 0.027779 | 0.960439 | 0.922340 | 0.038100 |
| 1 | 1946 | TAAAGA | 0.963259 | 0.935638 | 0.027621 | 0.960439 | 0.924733 | 0.035706 |
| 41 | 2002 | TCAAAA | 0.963259 | 0.935862 | 0.027398 | 0.960439 | 0.924769 | 0.035670 |
| 22 | 1770 | GCTGCA | 0.963259 | 0.936210 | 0.027049 | 0.960439 | 0.927071 | 0.033368 |
| 44 | 1690 | GCAGCA | 0.963259 | 0.936576 | 0.026683 | 0.960439 | 0.929509 | 0.030930 |

## E10 — Biological fidelity

### HGV distance vs ANI / AF correlations
| subset | target | n | spearman_r | spearman_p | pearson_r | pearson_p |
| --- | --- | --- | --- | --- | --- | --- |
| all_reported | ANI | 2368 | -0.697172 | 0.000000 | -0.596757 | 1.802e-228 |
| all_reported | AF_min | 2368 | -0.803849 | 0.000000 | -0.698593 | 0.000000 |
| all_reported | ani_distance | 2368 | 0.697172 | 0.000000 | 0.596757 | 1.802e-228 |
| positive | ANI | 1936 | -0.592853 | 4.158e-184 | -0.411230 | 6.948e-80 |
| positive | AF_min | 1936 | -0.752073 | 0.000000 | -0.577879 | 6.780e-173 |
| positive | ani_distance | 1936 | 0.592853 | 4.158e-184 | 0.411230 | 6.948e-80 |
| negative | ANI | 432 | 0.330136 | 1.919e-12 | 0.306352 | 7.716e-11 |
| negative | AF_min | 432 | -0.128075 | 0.007693 | -0.126264 | 0.008608 |
| negative | ani_distance | 432 | -0.330136 | 1.919e-12 | -0.306352 | 7.716e-11 |
| pair_type:hard | ANI | 432 | 0.330136 | 1.919e-12 | 0.306352 | 7.716e-11 |
| pair_type:hard | AF_min | 432 | -0.128075 | 0.007693 | -0.126264 | 0.008608 |
| pair_type:hard | ani_distance | 432 | -0.330136 | 1.919e-12 | -0.306352 | 7.716e-11 |
| pair_type:positive | ANI | 1936 | -0.592853 | 4.158e-184 | -0.411230 | 6.948e-80 |
| pair_type:positive | AF_min | 1936 | -0.752073 | 0.000000 | -0.577879 | 6.780e-173 |
| pair_type:positive | ani_distance | 1936 | 0.592853 | 4.158e-184 | 0.411230 | 6.948e-80 |

### ANI-range fidelity
| ANI_range | n | hgv_distance_mean | hgv_distance_sd | ANI_mean | AF_min_mean | spearman_hgv_vs_ANI | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 90-95 | 432 | 0.117827 | 0.025679 | 91.550764 | 64.350278 | 0.330136 | 1.919e-12 |
| 95-97 | 187 | 0.079869 | 0.026454 | 96.230963 | 77.035187 | -0.016523 | 0.822406 |
| 97-99 | 1124 | 0.066885 | 0.042512 | 98.101486 | 84.055801 | -0.132491 | 8.326e-06 |
| >=99 | 625 | 0.029093 | 0.033030 | 99.742912 | 95.541040 | -0.749428 | 1.146e-113 |

### Taxonomic-distance consistency
| taxonomic_stratum | n | hgv_distance_mean | hgv_distance_sd | ANI_mean | AF_min_mean |
| --- | --- | --- | --- | --- | --- |
| different_family | 1360 | 1.246960 | 0.175242 | NA | NA |
| same_family_diff_genus | 144 | 1.073485 | 0.043267 | NA | NA |
| same_genus_diff_species | 432 | 0.117827 | 0.025679 | 91.550764 | 64.350278 |
| same_species | 1936 | 0.055939 | 0.042728 | 98.450713 | 87.085460 |

Tree component status: `SKIPPED_NO_FROZEN_TREE: no formal GTDB tree retained in the cleaned V3 source-of-truth; no phylogeny reconstruction claim is made.`

## E11 — Reference screening

### Retrieval summary
| external_type | n_queries | MRR | Recall@1 | FN_rate@1 | candidate_reduction@1 | Recall@5 | FN_rate@5 | candidate_reduction@5 | Recall@10 | FN_rate@10 | candidate_reduction@10 | Recall@20 | FN_rate@20 | candidate_reduction@20 | Recall@50 | FN_rate@50 | candidate_reduction@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 391 | 0.987461 | 0.984655 | 0.015345 | 0.999579 | 0.992327 | 0.007673 | 0.997894 | 0.992327 | 0.007673 | 0.995788 | 1.000000 | 0.000000 | 0.991575 | 1.000000 | 0.000000 | 0.978939 |
| B2_seen_genus | 196 | 0.899045 | 0.892857 | 0.107143 | 0.999579 | 0.897959 | 0.102041 | 0.997894 | 0.897959 | 0.102041 | 0.995788 | 0.948980 | 0.051020 | 0.991575 | 0.979592 | 0.020408 | 0.978939 |
| B2_unseen_genus | 193 | 0.550272 | 0.549223 | 0.450777 | 0.999579 | 0.549223 | 0.450777 | 0.997894 | 0.549223 | 0.450777 | 0.995788 | 0.549223 | 0.450777 | 0.991575 | 0.549223 | 0.450777 | 0.978939 |

### ANI confirmation coverage overall
| tool | attempted | reported | no_hit | hard_fail | reported_fraction | no_hit_fraction | hard_fail_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| skani | 39000 | 7635 | 31365 | 0 | 0.195769 | 0.804231 | 0.000000 |
| FastANI | 39000 | 10444 | 28556 | 0 | 0.267795 | 0.732205 | 0.000000 |

### ANI confirmation coverage by external cohort
| tool | external_type | attempted | reported | no_hit | hard_fail | reported_fraction | no_hit_fraction | hard_fail_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skani | B1 | 19550 | 5371 | 14179 | 0 | 0.274731 | 0.725269 | 0.000000 |
| skani | B2_seen_genus | 9800 | 2262 | 7538 | 0 | 0.230816 | 0.769184 | 0.000000 |
| skani | B2_unseen_genus | 9650 | 2 | 9648 | 0 | 0.000207 | 0.999793 | 0.000000 |
| FastANI | B1 | 19550 | 6606 | 12944 | 0 | 0.337903 | 0.662097 | 0.000000 |
| FastANI | B2_seen_genus | 9800 | 2970 | 6830 | 0 | 0.303061 | 0.696939 | 0.000000 |
| FastANI | B2_unseen_genus | 9650 | 868 | 8782 | 0 | 0.089948 | 0.910052 | 0.000000 |

## E12 — Scalability

### Real-FASTA HGV benchmark
| N_reference | repeats | Q | real_fasta_reference_feature_wall_s_median | real_fasta_reference_feature_wall_s_min | real_fasta_reference_feature_wall_s_max | real_fasta_query_feature_wall_s_median | real_fasta_query_feature_wall_s_min | real_fasta_query_feature_wall_s_max | reference_model_inference_wall_s_median | reference_model_inference_wall_s_min | reference_model_inference_wall_s_max | query_model_inference_wall_s_median | query_model_inference_wall_s_min | query_model_inference_wall_s_max | exact_scan_cold_s_median | exact_scan_cold_s_min | exact_scan_cold_s_max | exact_scan_warm_median_s_median | exact_scan_warm_median_s_min | exact_scan_warm_median_s_max | vector_index_bytes_median | vector_index_bytes_min | vector_index_bytes_max | bytes_per_genome_median | bytes_per_genome_min | bytes_per_genome_max | process_maxrss_kb_median | process_maxrss_kb_min | process_maxrss_kb_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100.000000 | 3.000000 | 50.000000 | 1.603315 | 1.600439 | 1.749290 | 1.621326 | 0.995707 | 1.643411 | 0.001391 | 0.001374 | 0.001774 | 0.002296 | 0.001351 | 0.185012 | 0.001038 | 0.001028 | 0.001219 | 0.000939 | 0.000898 | 0.000950 | 25728.000000 | 25728.000000 | 25728.000000 | 257.280000 | 257.280000 | 257.280000 | 925756.000000 | 855360.000000 | 925756.000000 |
| 500.000000 | 3.000000 | 50.000000 | 2.345933 | 2.304011 | 2.400701 | 1.621326 | 0.995707 | 1.643411 | 0.001689 | 0.001645 | 0.001954 | 0.002296 | 0.001351 | 0.185012 | 0.002335 | 0.002264 | 0.002369 | 0.002178 | 0.002148 | 0.002202 | 128128.000000 | 128128.000000 | 128128.000000 | 256.256000 | 256.256000 | 256.256000 | 925756.000000 | 863040.000000 | 925756.000000 |
| 1000.000000 | 3.000000 | 50.000000 | 3.191663 | 3.096891 | 3.241447 | 1.621326 | 0.995707 | 1.643411 | 0.002153 | 0.002042 | 0.002236 | 0.002296 | 0.001351 | 0.185012 | 0.004042 | 0.003986 | 0.004144 | 0.003903 | 0.003845 | 0.004024 | 256128.000000 | 256128.000000 | 256128.000000 | 256.128000 | 256.128000 | 256.128000 | 925756.000000 | 875340.000000 | 925756.000000 |
| 2000.000000 | 3.000000 | 50.000000 | 4.983308 | 4.914543 | 5.033821 | 1.621326 | 0.995707 | 1.643411 | 0.003333 | 0.003053 | 0.003394 | 0.002296 | 0.001351 | 0.185012 | 0.009732 | 0.009697 | 0.040941 | 0.009395 | 0.009350 | 0.009609 | 512128.000000 | 512128.000000 | 512128.000000 | 256.064000 | 256.064000 | 256.064000 | 925756.000000 | 898492.000000 | 925756.000000 |
| 3000.000000 | 3.000000 | 50.000000 | 6.786794 | 6.679564 | 6.791329 | 1.621326 | 0.995707 | 1.643411 | 0.003636 | 0.003544 | 0.003924 | 0.002296 | 0.001351 | 0.185012 | 0.014426 | 0.014163 | 0.014548 | 0.014077 | 0.014032 | 0.014359 | 768128.000000 | 768128.000000 | 768128.000000 | 256.042667 | 256.042667 | 256.042667 | 925756.000000 | 925756.000000 | 926400.000000 |

### Vector-search microbenchmark
| N_vectors | Q | dimension | database_bytes | exact_scan_first_s | exact_scan_warm_median_s | data_origin |
| --- | --- | --- | --- | --- | --- | --- |
| 10000 | 10 | 64 | 2560000 | 0.007229 | 0.006484 | synthetic tiling of real frozen HGV vectors; not real million-genome end-to-end |
| 100000 | 10 | 64 | 25600000 | 0.084542 | 0.082286 | synthetic tiling of real frozen HGV vectors; not real million-genome end-to-end |
| 1000000 | 10 | 64 | 256000000 | 0.966194 | 0.967468 | synthetic tiling of real frozen HGV vectors; not real million-genome end-to-end |

### Scope audit
| component | present_in_E12_v1 | evidence | interpretation |
| --- | --- | --- | --- |
| HGV real FASTA canonical-k6 feature extraction | 1 | E12_REAL_FASTA_END_TO_END.tsv / E12_REAL_FASTA_SUMMARY.tsv | real frozen FASTA, not cached-vector-only timing |
| HGV model inference | 1 | reference_model_inference_wall_s / query_model_inference_wall_s | reported separately from feature extraction |
| HGV exact 64-D scan | 1 | exact_scan_cold_s / exact_scan_warm_median_s | exact O(Nd), d=64 |
| vector index size / bytes per genome / process RSS | 1 | vector_index_bytes / bytes_per_genome / process_maxrss_kb | HGV benchmark process |
| 10^4-10^6 vector-search microbenchmark | 1 | E12_VECTOR_MICROBENCHMARK.tsv | synthetic tiling of real HGV vectors; NOT million-real-genome end-to-end |
| Mash/Dashing2/skani/FastANI matched end-to-end timing in E12 v1 | 0 | not produced by 14_run_e12_scalability_v1.py | must be sourced from an already-frozen E5/runtime block if available; do not claim E12 v1 alone provides this comparison |

**Important:** E12 v1 benchmarks the HGV pipeline and HGV vector search. It does not itself provide matched end-to-end Mash/Dashing2/skani/FastANI timings. Before deciding that reviewer R1-7 is fully closed, audit the already-frozen E5/runtime outputs for those method costs.

## Claim/scope boundaries

| topic | status | evidence | allowed_wording |
| --- | --- | --- | --- |
| E6 universal strict improvement | NOT_SUPPORTED | E6_CLAIM_DECISION_TABLE.tsv | Report comparator-specific deltas/CIs/Holm p-values; use geometry/mechanism wording where strict improvement is unsupported. |
| E7 sensitivity | EXECUTED | E7_SENSITIVITY_SUMMARY.tsv | OFAT stability only; do not describe as full hyperparameter-grid search. |
| E8 fragmentation | EXECUTED | E8_FRAGMENTATION_SUMMARY.tsv | Frozen full-assembly models tested on reduced sequence amount; no fragmentation-specific retraining or threshold retuning. |
| E9 attribution causality | DESCRIPTIVE_ONLY | E9 IG/permutation/confound tables | Model sensitivity/association, not biological causality. Attribution uses frozen G2 D3 seed20260915. |
| E9 undefined correlations | 1_UNDEFINED_ROWS | E9_UNDEFINED_CORRELATIONS.tsv | Undefined when an input is constant; do not impute 0. |
| E10 phylogeny/tree reconstruction | NO_TREE_CLAIM | E10_TREE_COMPONENT_STATUS.txt | ANI/AF and taxonomic consistency only unless a separate frozen tree analysis is later justified. |
| E11 ANI tool no-hit semantics | REPORT_COVERAGE | E11_CONFIRMATION_COVERAGE_OVERALL.tsv | No-hit is structural missing under tool defaults; do not impute ANI=0. |
| E12 million-genome evidence | VECTOR_MICROBENCHMARK_ONLY | E12_VECTOR_MICROBENCHMARK.tsv | 10^4-10^6 vector-search microbenchmark; not million-real-genome end-to-end. |
| E12 fair native-baseline runtime comparison | NOT_IN_E12_V1 | E12_SCOPE_AUDIT.tsv | Do not claim E12 v1 alone benchmarks Mash/Dashing2/skani/FastANI end-to-end. First audit existing E5 runtime/memory outputs. |

## Manuscript / reviewer mapping

| experiment | section_role | recommended_location | source_evidence | reviewer_comments |
| --- | --- | --- | --- | --- |
| E6 | Distillation ablation | Main/Supplement | D0-D5 mode metrics + D3 paired AUROC/AUPRC contrasts + geometry | R1-5; R3-4; R3-5 |
| E7 | Sensitivity analysis | Supplement/Main concise | OFAT k, dimension, margin, lambda stability | R1-5 |
| E8 | Fragmentation robustness | Main or Supplement figure | Full vs 100kb/50kb/20kb/5kb/5x5kb degradation | R1-5 |
| E9 | Explainability | Supplement | IG + top-50 permutation + GC/length/contigs/N50/completeness confound audit | R1-4 |
| E10 | Biological fidelity | Main/Supplement | HGV distance vs ANI/AF + ANI range + taxonomic strata; no phylogeny claim | R1-3; R3-5 |
| E11 | Reference screening | Main figure/table | External retrieval, Recall@K/MRR, candidate reduction, ANI confirmation coverage | R1-3 |
| E12 | Scalability | Main/Supplement | Real FASTA HGV pipeline + vector microbenchmark; baseline timing scope must be audited separately | R1-7 |

HGV_V3_SCIENTIFIC_RESULTS_AUDIT=PASS
NO_FROZEN_INPUT_MODIFIED=YES
