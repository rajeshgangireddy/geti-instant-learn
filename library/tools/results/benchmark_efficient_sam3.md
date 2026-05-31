# EfficientSAM3 benchmark — speed + accuracy attribution

Backbone: `efficientvit/b1` | warmup runs: 3 | IoU threshold: 0.5 | breathing room: 10s

Configs: C1=fp32 baseline, C2=bf16 autocast, C3=bf16+torch.compile, C4=bf16+fine-tuned weights.

| device | dataset | mode | config | precision | n_img | median ms | p90 ms | P | R | F1 | tp | fp | fn | fit s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuda | potatoes | classic | C1_baseline_fp32 | fp32 | 10 | 377.23 | 418.52 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.02 |  |
| cuda | nuts | classic | C1_baseline_fp32 | fp32 | 21 | 185.44 | 227.08 | 0.4444 | 0.4058 | 0.4242 | 28 | 35 | 41 | 0.024 |  |
| cuda | candies | classic | C1_baseline_fp32 | fp32 | 12 | 424.48 | 447.79 | 0.9917 | 0.9835 | 0.9876 | 239 | 2 | 4 | 0.032 |  |
| cuda | potatoes | visual_exemplar | C1_baseline_fp32 | fp32 | 9 | 358.16 | 433.47 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.054 |  |
| cuda | nuts | visual_exemplar | C1_baseline_fp32 | fp32 | 19 | 200.09 | 239.99 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.051 |  |
| cuda | candies | visual_exemplar | C1_baseline_fp32 | fp32 | 11 | 411.68 | 460.51 | 0.9905 | 0.95 | 0.9698 | 209 | 2 | 11 | 0.028 |  |
| cuda | potatoes | classic | C2_bf16 | bf16 | 10 | 305.11 | 336.24 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.019 |  |
| cuda | nuts | classic | C2_bf16 | bf16 | 21 | 100.58 | 141.48 | 0.4444 | 0.4058 | 0.4242 | 28 | 35 | 41 | 0.024 |  |
| cuda | candies | classic | C2_bf16 | bf16 | 12 | 369.7 | 392.86 | 0.9917 | 0.9835 | 0.9876 | 239 | 2 | 4 | 0.024 |  |
| cuda | potatoes | visual_exemplar | C2_bf16 | bf16 | 9 | 303.93 | 367.89 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.025 |  |
| cuda | nuts | visual_exemplar | C2_bf16 | bf16 | 19 | 115.62 | 152.64 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.038 |  |
| cuda | candies | visual_exemplar | C2_bf16 | bf16 | 11 | 363.09 | 397.26 | 0.9906 | 0.9545 | 0.9722 | 210 | 2 | 10 | 0.021 |  |
| cuda | potatoes | classic | C3_bf16_compile | bf16 | 10 | 292.83 | 328.04 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.02 |  |
| cuda | nuts | classic | C3_bf16_compile | bf16 | 21 | 87.4 | 127.01 | 0.4444 | 0.4058 | 0.4242 | 28 | 35 | 41 | 0.024 |  |
| cuda | candies | classic | C3_bf16_compile | bf16 | 12 | 371.45 | 387.35 | 0.9917 | 0.9835 | 0.9876 | 239 | 2 | 4 | 0.024 |  |
| cuda | potatoes | visual_exemplar | C3_bf16_compile | bf16 | 9 | 291.76 | 359.14 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.024 |  |
| cuda | nuts | visual_exemplar | C3_bf16_compile | bf16 | 19 | 103.3 | 140.89 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.038 |  |
| cuda | candies | visual_exemplar | C3_bf16_compile | bf16 | 11 | 352.74 | 387.73 | 0.9906 | 0.9545 | 0.9722 | 210 | 2 | 10 | 0.021 |  |
| cuda | potatoes | classic | C4_bf16_ft | bf16 | 10 | 299.05 | 336.47 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.018 |  |
| cuda | nuts | classic | C4_bf16_ft | bf16 | 21 | 99.55 | 138.31 | 0.4286 | 0.3913 | 0.4091 | 27 | 36 | 42 | 0.024 |  |
| cuda | candies | classic | C4_bf16_ft | bf16 | 12 | 355.92 | 395.52 | 0.9833 | 0.9671 | 0.9751 | 235 | 4 | 8 | 0.024 |  |
| cuda | potatoes | visual_exemplar | C4_bf16_ft | bf16 | 9 | 304.82 | 369.68 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.024 |  |
| cuda | nuts | visual_exemplar | C4_bf16_ft | bf16 | 19 | 112.98 | 152.28 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.037 |  |
| cuda | candies | visual_exemplar | C4_bf16_ft | bf16 | 11 | 357.11 | 398.47 | 0.9907 | 0.9682 | 0.9793 | 213 | 2 | 7 | 0.021 |  |
| xpu | potatoes | classic | C1_baseline_fp32 | fp32 | 10 | 571.41 | 605.57 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.021 |  |
| xpu | nuts | classic | C1_baseline_fp32 | fp32 | 21 | 565.03 | 600.26 | 0.4444 | 0.4058 | 0.4242 | 28 | 35 | 41 | 0.024 |  |
| xpu | candies | classic | C1_baseline_fp32 | fp32 | 12 | 553.28 | 575.71 | 0.9917 | 0.9835 | 0.9876 | 239 | 2 | 4 | 0.025 |  |
| xpu | potatoes | visual_exemplar | C1_baseline_fp32 | fp32 | 9 | 470.76 | 523.92 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.486 |  |
| xpu | nuts | visual_exemplar | C1_baseline_fp32 | fp32 | 19 | 571.5 | 600.15 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.193 |  |
| xpu | candies | visual_exemplar | C1_baseline_fp32 | fp32 | 11 | 541.55 | 575.53 | 0.9905 | 0.95 | 0.9698 | 209 | 2 | 11 | 0.099 |  |
| xpu | potatoes | classic | C2_bf16 | bf16 | 10 | 267.14 | 297.02 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.02 |  |
| xpu | nuts | classic | C2_bf16 | bf16 | 21 | 146.06 | 174.22 | 0.4444 | 0.4058 | 0.4242 | 28 | 35 | 41 | 0.024 |  |
| xpu | candies | classic | C2_bf16 | bf16 | 12 | 354.1 | 374.49 | 0.9917 | 0.9835 | 0.9876 | 239 | 2 | 4 | 0.025 |  |
| xpu | potatoes | visual_exemplar | C2_bf16 | bf16 | 9 | 267.37 | 321.48 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.824 |  |
| xpu | nuts | visual_exemplar | C2_bf16 | bf16 | 19 | 148.61 | 178.82 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.093 |  |
| xpu | candies | visual_exemplar | C2_bf16 | bf16 | 11 | 342.65 | 375.01 | 0.9906 | 0.9545 | 0.9722 | 210 | 2 | 10 | 0.411 |  |
| xpu | potatoes | classic | C3_bf16_compile | bf16 | 10 | 266.69 | 297.37 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.019 |  |
| xpu | nuts | classic | C3_bf16_compile | bf16 | 21 | 129.86 | 161.7 | 0.4355 | 0.3913 | 0.4122 | 27 | 35 | 42 | 0.024 |  |
| xpu | candies | classic | C3_bf16_compile | bf16 | 12 | 345.74 | 357.76 | 0.9917 | 0.9835 | 0.9876 | 239 | 2 | 4 | 0.025 |  |
| xpu | potatoes | visual_exemplar | C3_bf16_compile | bf16 | 9 | 262.68 | 327.06 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.044 |  |
| xpu | nuts | visual_exemplar | C3_bf16_compile | bf16 | 19 | 139.73 | 169.11 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.073 |  |
| xpu | candies | visual_exemplar | C3_bf16_compile | bf16 | 11 | 328.85 | 375.96 | 0.9906 | 0.9545 | 0.9722 | 210 | 2 | 10 | 0.042 |  |
| xpu | potatoes | classic | C4_bf16_ft | bf16 | 10 | 270.09 | 297.94 | 1.0 | 1.0 | 1.0 | 80 | 0 | 0 | 0.02 |  |
| xpu | nuts | classic | C4_bf16_ft | bf16 | 21 | 142.93 | 172.66 | 0.4286 | 0.3913 | 0.4091 | 27 | 36 | 42 | 0.024 |  |
| xpu | candies | classic | C4_bf16_ft | bf16 | 12 | 329.01 | 366.51 | 0.9833 | 0.9671 | 0.9751 | 235 | 4 | 8 | 0.025 |  |
| xpu | potatoes | visual_exemplar | C4_bf16_ft | bf16 | 9 | 260.59 | 349.91 | 1.0 | 1.0 | 1.0 | 72 | 0 | 0 | 0.044 |  |
| xpu | nuts | visual_exemplar | C4_bf16_ft | bf16 | 19 | 147.62 | 180.01 | 0.4576 | 0.4426 | 0.45 | 27 | 32 | 34 | 0.073 |  |
| xpu | candies | visual_exemplar | C4_bf16_ft | bf16 | 11 | 336.95 | 376.06 | 0.986 | 0.9636 | 0.9747 | 212 | 3 | 8 | 0.042 |  |
