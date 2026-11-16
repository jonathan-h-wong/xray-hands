# ARCTIC baselines: 20261113_ego_s5_compare_moge

eval set: {"protocol": "p2", "split": "val", "views": [0], "stride": 5, "grounded_thresh_mm": 2.0, "require_both": false, "contact_available": true, "n_images": 2697, "n_hands": {"left": 1749, "right": 1671}}

intrinsics: moge:baselines/preds/20261113_ego_s5/moge_hand2d.json

GT penetration re-computation vs eval set: max |diff| = 0.0001 mm (OK)

## All selected hands (metrics over detected hands)

|                             |   ('gt', '-') |   ('hamer', 'wp') |   ('hamer', 'fit2d') |   ('wilor', 'wp') |   ('wilor', 'fit2d') |   ('hopformer_moge', 'native') |
|:----------------------------|--------------:|------------------:|---------------------:|------------------:|---------------------:|-------------------------------:|
| hands                       |       3420    |           3420    |              3420    |           3420    |              3420    |                        3420    |
| detected %                  |        100    |             95.85 |                95.85 |             95.12 |                95.12 |                          94.97 |
| pen p50                     |          0    |              0    |                 0    |              0    |                 0    |                           0    |
| pen p90                     |          1.14 |              3.4  |                 1.26 |              0.68 |                 0    |                           0    |
| pen>2mm %                   |          0    |             11.47 |                 9.43 |              9.1  |                 7.9  |                           8    |
| pen>5mm %                   |          0    |              7.26 |                 6.53 |              6.52 |                 5.78 |                           5.17 |
| contact %                   |         39.88 |             16.69 |                13.09 |             12.73 |                10.36 |                          11.08 |
| mpjpe_ra p50                |        nan    |             21.32 |                21.32 |             12.26 |                12.26 |                          18.41 |
| pa_mpjpe p50                |        nan    |              6.71 |                 6.71 |              5.14 |                 5.14 |                           7.74 |
| pa_mpvpe p50                |        nan    |              7.2  |                 7.2  |              5.58 |                 5.58 |                           8.14 |
| root_err p50                |        nan    |             95.98 |               111.25 |            129.02 |               152.67 |                         125.24 |
| mpjpe_cam p50               |        nan    |            101.79 |               117.9  |            124.2  |               148.58 |                         115.55 |
| kp2d_err_px p50             |        nan    |            142.6  |               167.1  |            125.72 |               157.45 |                         167.66 |
| depth_err p50               |        nan    |            -92.92 |              -107.47 |           -125.28 |              -149.58 |                        -123.8  |
| |depth_err| p50             |        nan    |             93.06 |               107.55 |            125.28 |               149.58 |                         123.8  |
| mrrpe_rl p50                |        nan    |             40.35 |                47.99 |             40.6  |                38.74 |                          29.3  |
| contact precision %         |        nan    |             56.86 |                55.01 |             52.17 |                51.93 |                          63.89 |
| contact recall %            |        nan    |             24.16 |                18.34 |             16.89 |                13.68 |                          18.05 |
| contact F1 %                |        nan    |             33.91 |                27.51 |             25.52 |                21.66 |                          28.15 |
| |sd err| p50                |        nan    |             31.72 |                44.22 |             41.99 |                60.83 |                          45.48 |
| ra pen p50                  |        nan    |              0    |                 0    |              0    |                 0    |                           0    |
| ra pen>5mm %                |        nan    |             14.95 |                14.95 |              6.21 |                 6.21 |                          10.5  |
| self pen p50                |        nan    |            nan    |               nan    |            nan    |               nan    |                           0    |
| obj t err p50               |        nan    |            nan    |               nan    |            nan    |               nan    |                         149.64 |
| obj rot err p50             |        nan    |            nan    |               nan    |            nan    |               nan    |                           5.79 |
| mrrpe_ro p50                |        nan    |            nan    |               nan    |            nan    |               nan    |                          34.4  |
| obj cls %                   |        nan    |            nan    |               nan    |            nan    |               nan    |                          98.12 |
| own-obj contact precision % |        nan    |            nan    |               nan    |            nan    |               nan    |                          55.39 |
| own-obj contact recall %    |        nan    |            nan    |               nan    |            nan    |               nan    |                          78.18 |
| own-obj contact F1 %        |        nan    |            nan    |               nan    |            nan    |               nan    |                          64.84 |
| own-obj pen>5mm %           |        nan    |            nan    |               nan    |            nan    |               nan    |                          27.71 |
| own-obj |sd err| p50        |        nan    |            nan    |               nan    |            nan    |               nan    |                          11.35 |
| rel obj t p50               |        nan    |            nan    |               nan    |            nan    |               nan    |                          40.96 |
| rel obj frame p50           |        nan    |            nan    |               nan    |            nan    |               nan    |                          43.07 |
| scene-scaled mpjpe p50      |        nan    |            nan    |               nan    |            nan    |               nan    |                          40.77 |

## Common subset: 3222 hands detected by every method

|                 |   ('gt', '-') |   ('hamer', 'wp') |   ('hamer', 'fit2d') |   ('wilor', 'wp') |   ('wilor', 'fit2d') |   ('hopformer_moge', 'native') |
|:----------------|--------------:|------------------:|---------------------:|------------------:|---------------------:|-------------------------------:|
| hands           |          3222 |           3222    |              3222    |           3222    |              3222    |                        3222    |
| pen p50         |             0 |              0    |                 0    |              0    |                 0    |                           0    |
| pen>5mm %       |             0 |              6.92 |                 6.18 |              6.18 |                 5.52 |                           4.93 |
| mpjpe_ra p50    |           nan |             21.12 |                21.12 |             12.21 |                12.21 |                          18.39 |
| pa_mpjpe p50    |           nan |              6.66 |                 6.66 |              5.12 |                 5.12 |                           7.74 |
| root_err p50    |           nan |             95.94 |               111.34 |            129.15 |               152.9  |                         125.29 |
| kp2d_err_px p50 |           nan |            141.31 |               165.93 |            125.58 |               157.39 |                         167.5  |
| ra pen p50      |           nan |              0    |                 0    |              0    |                 0    |                           0    |

## Per part (contact frames of each method; penetration depth mm, share of penetrating volume)

### gt (-), 1364 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |        0 |        0 |     0 |  0    |         18.8  |
| thumb  |        0 |        0 |     0 |  1.31 |         40.33 |
| index  |        0 |        0 |     0 |  0.92 |         21.64 |
| middle |        0 |        0 |     0 |  0.39 |          5.95 |
| ring   |        0 |        0 |     0 |  0.27 |          8.27 |
| pinky  |        0 |        0 |     0 |  0    |          5.01 |
| palmar |        0 |        0 |     0 |  1.72 |         95.81 |
| dorsal |        0 |        0 |     0 |  0.23 |          4.19 |

### hamer (wp), 547 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    19.01 |    15.72 |  0    |  7.06 |         22.2  |
| thumb  |    15.17 |    11.52 |  0    |  5.68 |         13.81 |
| index  |    29.07 |    16.64 |  0    |  9.26 |         19.39 |
| middle |    40.59 |    18.83 |  0    |  8    |         16.66 |
| ring   |    37.29 |    18.65 |  0    |  7.27 |         16.8  |
| pinky  |    23.58 |    11.52 |  0    |  5.15 |         11.13 |
| palmar |    61.24 |    37.66 |  3.53 | 18.04 |         53.74 |
| dorsal |    66.54 |    40.4  |  4.08 | 16.88 |         46.26 |

### hamer (fit2d), 429 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    22.61 |    17.95 |  0    |  7.87 |         22.44 |
| thumb  |    16.78 |    12.12 |  0    |  5.85 |         10.38 |
| index  |    32.87 |    19.81 |  0    |  8.94 |         20.1  |
| middle |    42.89 |    21.68 |  0.68 |  7.99 |         17.89 |
| ring   |    39.63 |    23.78 |  0    |  9.07 |         17.02 |
| pinky  |    24.48 |    13.75 |  0    |  5.77 |         12.19 |
| palmar |    65.27 |    41.72 |  4.43 | 18.15 |         51.95 |
| dorsal |    68.3  |    46.15 |  4.62 | 15.33 |         48.05 |

### wilor (wp), 414 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    25.6  |    20.29 |  0    |  8.13 |         25.94 |
| thumb  |    14.98 |    10.14 |  0    |  5.06 |          9.11 |
| index  |    33.82 |    19.08 |  0    |  8.98 |         17.69 |
| middle |    40.58 |    22.95 |  0    |  8.36 |         20.18 |
| ring   |    39.61 |    22.22 |  0    |  8.42 |         15.81 |
| pinky  |    26.81 |    15.7  |  0    |  6.87 |         11.28 |
| palmar |    68.36 |    47.83 |  4.7  | 16.37 |         50.5  |
| dorsal |    68.6  |    46.38 |  4.74 | 15.94 |         49.5  |

### wilor (fit2d), 337 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    25.52 |    20.47 |  0    |  8.62 |         20.06 |
| thumb  |    11.87 |     9.5  |  0    |  4.34 |          8.52 |
| index  |    35.61 |    21.66 |  0    | 12.39 |         25.97 |
| middle |    45.4  |    24.93 |  1.11 |  9.34 |         24.16 |
| ring   |    37.39 |    19.88 |  0    |  6.63 |         12.38 |
| pinky  |    26.11 |    14.24 |  0    |  5.83 |          8.91 |
| palmar |    72.11 |    48.96 |  4.89 | 18.9  |         48.42 |
| dorsal |    71.51 |    48.37 |  4.89 | 19.6  |         51.58 |

### hopformer_moge (native), 360 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    22.22 |    15.28 |  0    |  6.14 |         16.52 |
| thumb  |     7.22 |     3.33 |  0    |  0    |          3.66 |
| index  |    26.11 |    17.78 |  0    |  6.46 |         17.3  |
| middle |    41.39 |    21.11 |  0.39 |  7.22 |         23.6  |
| ring   |    40.28 |    19.44 |  0    |  7.73 |         23.02 |
| pinky  |    29.44 |    17.78 |  0    |  7.14 |         15.89 |
| palmar |    64.72 |    39.44 |  3.63 | 11.62 |         51.68 |
| dorsal |    67.78 |    42.78 |  4.26 | 11.29 |         48.32 |
