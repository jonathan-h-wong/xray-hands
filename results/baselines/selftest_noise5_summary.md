# ARCTIC baselines: selftest_noise5

eval set: {"protocol": "p2", "split": "val", "views": [0], "stride": 5, "grounded_thresh_mm": 2.0, "require_both": false, "contact_available": true, "n_images": 2697, "n_hands": {"left": 1749, "right": 1671}}

intrinsics: native

## All selected hands (metrics over detected hands)

|                     |   ('gt', '-') |   ('oracle_hand2d', 'wp') |   ('oracle_hand2d', 'fit2d') |   ('oracle_hop', 'native') |
|:--------------------|--------------:|--------------------------:|-----------------------------:|---------------------------:|
| hands               |        236    |                    236    |                       236    |                     236    |
| detected %          |        100    |                    100    |                       100    |                     100    |
| pen p50             |          0    |                      1.64 |                         1.64 |                       1.64 |
| pen p90             |          1.82 |                      3.24 |                         3.24 |                       3.24 |
| pen>2mm %           |          0    |                     31.36 |                        31.36 |                      31.36 |
| pen>5mm %           |          0    |                      0.42 |                         0.42 |                       0.42 |
| contact %           |         52.12 |                     63.98 |                        63.98 |                      63.98 |
| mpjpe_ra p50        |        nan    |                      2.88 |                         2.88 |                       2.88 |
| pa_mpjpe p50        |        nan    |                      2.82 |                         2.82 |                       2.82 |
| pa_mpvpe p50        |        nan    |                      7.88 |                         7.88 |                       7.88 |
| root_err p50        |        nan    |                      0.93 |                         0.93 |                       0.93 |
| mpjpe_cam p50       |        nan    |                      2.67 |                         2.67 |                       2.67 |
| kp2d_err_px p50     |        nan    |                      7.92 |                         7.92 |                       7.92 |
| depth_err p50       |        nan    |                      0.04 |                         0.04 |                       0.04 |
| |depth_err| p50     |        nan    |                      0.39 |                         0.39 |                       0.39 |
| mrrpe_rl p50        |        nan    |                      1.42 |                         1.42 |                       1.42 |
| contact recall %    |        nan    |                     99.19 |                        99.19 |                      99.19 |
| contact precision % |        nan    |                     80.79 |                        80.79 |                      80.79 |
| ra pen p50          |        nan    |                      1.45 |                         1.45 |                       1.45 |
| ra pen>5mm %        |        nan    |                      0.85 |                         0.85 |                       0.85 |
| self pen p50        |        nan    |                    nan    |                       nan    |                       1.64 |
| obj t err p50       |        nan    |                    nan    |                       nan    |                       0    |
| obj rot err p50     |        nan    |                    nan    |                       nan    |                       0    |
| mrrpe_ro p50        |        nan    |                    nan    |                       nan    |                       0.93 |
| obj cls %           |        nan    |                    nan    |                       nan    |                     100    |

## Common subset: 236 hands detected by every method

|                 |   ('gt', '-') |   ('oracle_hand2d', 'wp') |   ('oracle_hand2d', 'fit2d') |   ('oracle_hop', 'native') |
|:----------------|--------------:|--------------------------:|-----------------------------:|---------------------------:|
| hands           |           236 |                    236    |                       236    |                     236    |
| pen p50         |             0 |                      1.64 |                         1.64 |                       1.64 |
| pen>5mm %       |             0 |                      0.42 |                         0.42 |                       0.42 |
| mpjpe_ra p50    |           nan |                      2.88 |                         2.88 |                       2.88 |
| pa_mpjpe p50    |           nan |                      2.82 |                         2.82 |                       2.82 |
| root_err p50    |           nan |                      0.93 |                         0.93 |                       0.93 |
| kp2d_err_px p50 |           nan |                      7.92 |                         7.92 |                       7.92 |
| ra pen p50      |           nan |                      1.45 |                         1.45 |                       1.45 |

## Per part (contact frames of each method; penetration depth mm, share of penetrating volume)

### gt (-), 123 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |        0 |        0 |  0    |  1.7  |         26.31 |
| thumb  |        0 |        0 |  0    |  1.77 |         42.71 |
| index  |        0 |        0 |  0    |  1.77 |         19.21 |
| middle |        0 |        0 |  0    |  0.54 |          1.18 |
| ring   |        0 |        0 |  0    |  1.23 |          4.78 |
| pinky  |        0 |        0 |  0    |  1.69 |          5.81 |
| palmar |        0 |        0 |  1.12 |  1.86 |         98.53 |
| dorsal |        0 |        0 |  0    |  0.24 |          1.47 |

### oracle_hand2d (wp), 151 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |     9.27 |     0    |  0    |  1.89 |         10.8  |
| thumb  |    19.21 |     0    |  0.79 |  2.89 |         20.06 |
| index  |    28.48 |     0    |  1.76 |  3.06 |         26.84 |
| middle |    17.88 |     0.66 |  0.61 |  2.96 |         11.38 |
| ring   |    13.25 |     0    |  0.88 |  2.69 |         14.6  |
| pinky  |     8.61 |     0    |  0    |  1.9  |         16.32 |
| palmar |    48.34 |     0.66 |  1.9  |  3.41 |         90.2  |
| dorsal |    20.53 |     0    |  1.53 |  2.74 |          9.8  |

### oracle_hand2d (fit2d), 151 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |     9.27 |     0    |  0    |  1.89 |         10.8  |
| thumb  |    19.21 |     0    |  0.79 |  2.89 |         20.06 |
| index  |    28.48 |     0    |  1.76 |  3.06 |         26.84 |
| middle |    17.88 |     0.66 |  0.61 |  2.96 |         11.38 |
| ring   |    13.25 |     0    |  0.88 |  2.69 |         14.6  |
| pinky  |     8.61 |     0    |  0    |  1.9  |         16.32 |
| palmar |    48.34 |     0.66 |  1.9  |  3.41 |         90.2  |
| dorsal |    20.53 |     0    |  1.53 |  2.74 |          9.8  |

### oracle_hop (native), 151 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |     9.27 |     0    |  0    |  1.89 |         10.8  |
| thumb  |    19.21 |     0    |  0.79 |  2.89 |         20.06 |
| index  |    28.48 |     0    |  1.76 |  3.06 |         26.84 |
| middle |    17.88 |     0.66 |  0.61 |  2.96 |         11.38 |
| ring   |    13.25 |     0    |  0.88 |  2.69 |         14.6  |
| pinky  |     8.61 |     0    |  0    |  1.9  |         16.32 |
| palmar |    48.34 |     0.66 |  1.9  |  3.41 |         90.2  |
| dorsal |    20.53 |     0    |  1.53 |  2.74 |          9.8  |
