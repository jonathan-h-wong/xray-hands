# ARCTIC baselines: 20261113_ego_s5_compare_fine

eval set: {"protocol": "p2", "split": "val", "views": [0], "stride": 5, "grounded_thresh_mm": 2.0, "require_both": false, "contact_available": true, "n_images": 2697, "n_hands": {"left": 1749, "right": 1671}}

intrinsics: native

GT penetration re-computation vs eval set: max |diff| = 0.0001 mm (OK)

## All selected hands (metrics over detected hands)

|                             |   ('gt', '-') |   ('hamer', 'wp') |   ('hamer', 'fit2d') |   ('wilor', 'wp') |   ('wilor', 'fit2d') |   ('hopformer', 'native') |
|:----------------------------|--------------:|------------------:|---------------------:|------------------:|---------------------:|--------------------------:|
| hands                       |       3420    |           3420    |              3420    |           3420    |              3420    |                   3420    |
| detected %                  |        100    |             95.85 |                95.85 |             95.12 |                95.12 |                     94.97 |
| pen p50                     |          0    |              0    |                 0    |              0    |                 0    |                      0    |
| pen p90                     |          1.14 |              4.64 |                 2.97 |              5.37 |                 3.9  |                      5.18 |
| pen>2mm %                   |          0    |             18.79 |                12.08 |             22.35 |                14.42 |                     18.72 |
| pen>5mm %                   |          0    |              9.09 |                 4.55 |             11.34 |                 7.78 |                     10.78 |
| contact %                   |         39.88 |             30.14 |                22.12 |             35.66 |                24.72 |                     28.69 |
| mpjpe_ra p50                |        nan    |             21.32 |                21.32 |             12.26 |                12.26 |                     18.41 |
| pa_mpjpe p50                |        nan    |              6.71 |                 6.71 |              5.14 |                 5.14 |                      7.74 |
| pa_mpvpe p50                |        nan    |              7.2  |                 7.2  |              5.58 |                 5.58 |                      8.14 |
| root_err p50                |        nan    |             43    |                38.47 |             30.7  |                37.57 |                     27.85 |
| mpjpe_cam p50               |        nan    |             40.01 |                32.2  |             30.54 |                36.51 |                     34.31 |
| kp2d_err_px p50             |        nan    |             40.5  |                36.09 |             38.71 |                24.38 |                     68.29 |
| depth_err p50               |        nan    |             37.74 |                24.69 |             -4.25 |               -24.62 |                     -6.14 |
| |depth_err| p50             |        nan    |             39.92 |                32.57 |             26.32 |                35.22 |                     21.5  |
| mrrpe_rl p50                |        nan    |             45.94 |                56.45 |             43.95 |                39.3  |                     27.69 |
| contact precision %         |        nan    |             69.53 |                77.93 |             71.98 |                76.74 |                     67.6  |
| contact recall %            |        nan    |             53.38 |                43.9  |             65.29 |                48.24 |                     49.45 |
| contact F1 %                |        nan    |             60.4  |                56.16 |             68.47 |                59.24 |                     57.12 |
| |sd err| p50                |        nan    |              9.53 |                 9.12 |              7.67 |                 9.31 |                     10.66 |
| ra pen p50                  |        nan    |              0    |                 0    |              0    |                 0    |                      0    |
| ra pen>5mm %                |        nan    |             14.95 |                14.95 |              6.21 |                 6.21 |                     10.5  |
| self pen p50                |        nan    |            nan    |               nan    |            nan    |               nan    |                      0    |
| obj t err p50               |        nan    |            nan    |               nan    |            nan    |               nan    |                     40.24 |
| obj rot err p50             |        nan    |            nan    |               nan    |            nan    |               nan    |                      5.79 |
| mrrpe_ro p50                |        nan    |            nan    |               nan    |            nan    |               nan    |                     29.54 |
| obj cls %                   |        nan    |            nan    |               nan    |            nan    |               nan    |                     98.12 |
| own-obj contact precision % |        nan    |            nan    |               nan    |            nan    |               nan    |                     58.65 |
| own-obj contact recall %    |        nan    |            nan    |               nan    |            nan    |               nan    |                     74.8  |
| own-obj contact F1 %        |        nan    |            nan    |               nan    |            nan    |               nan    |                     65.75 |
| own-obj pen>5mm %           |        nan    |            nan    |               nan    |            nan    |               nan    |                     21    |
| own-obj |sd err| p50        |        nan    |            nan    |               nan    |            nan    |               nan    |                     10.32 |
| rel obj t p50               |        nan    |            nan    |               nan    |            nan    |               nan    |                     35.04 |
| rel obj frame p50           |        nan    |            nan    |               nan    |            nan    |               nan    |                     36.67 |
| scene-scaled mpjpe p50      |        nan    |            nan    |               nan    |            nan    |               nan    |                     18.84 |

## Common subset: 3222 hands detected by every method

|                 |   ('gt', '-') |   ('hamer', 'wp') |   ('hamer', 'fit2d') |   ('wilor', 'wp') |   ('wilor', 'fit2d') |   ('hopformer', 'native') |
|:----------------|--------------:|------------------:|---------------------:|------------------:|---------------------:|--------------------------:|
| hands           |          3222 |           3222    |              3222    |           3222    |              3222    |                   3222    |
| pen p50         |             0 |              0    |                 0    |              0    |                 0    |                      0    |
| pen>5mm %       |             0 |              9.22 |                 4.59 |             11.3  |                 7.76 |                     10.58 |
| mpjpe_ra p50    |           nan |             21.12 |                21.12 |             12.21 |                12.21 |                     18.39 |
| pa_mpjpe p50    |           nan |              6.66 |                 6.66 |              5.12 |                 5.12 |                      7.74 |
| root_err p50    |           nan |             42.17 |                37.43 |             30.55 |                37.51 |                     27.87 |
| kp2d_err_px p50 |           nan |             40.17 |                35.76 |             38.36 |                24.2  |                     68.23 |
| ra pen p50      |           nan |              0    |                 0    |              0    |                 0    |                      0    |

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

### hamer (wp), 988 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    17.21 |     8.81 |  0    |  4.29 |         26.82 |
| thumb  |    29.86 |    14.68 |  0    |  6.38 |         14.72 |
| index  |    29.25 |    11.74 |  0    |  5.47 |         18.91 |
| middle |    22.17 |     8.2  |  0    |  4.6  |         14.83 |
| ring   |    15.38 |     6.48 |  0    |  3.61 |         15.41 |
| pinky  |     9.41 |     4.96 |  0    |  1.68 |          9.3  |
| palmar |    59.31 |    28.34 |  3.09 | 10.32 |         58.66 |
| dorsal |    50.81 |    23.89 |  2.14 |  9.21 |         41.34 |

### hamer (fit2d), 725 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |     7.45 |     3.03 |  0    |  0    |          9.58 |
| thumb  |    26.07 |     8.97 |  0    |  4.53 |         27.38 |
| index  |    20.14 |     7.45 |  0    |  4.09 |         24.63 |
| middle |    16.97 |     4.97 |  0    |  3.45 |         20.67 |
| ring   |    11.72 |     3.59 |  0    |  2.33 |         13.58 |
| pinky  |     6.48 |     1.66 |  0    |  0    |          4.17 |
| palmar |    52.14 |    18.76 |  2.28 |  6.98 |         57.47 |
| dorsal |    42.07 |    16.41 |  0.66 |  6.63 |         42.53 |

### wilor (wp), 1160 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    19.22 |     9.05 |  0    |  4.42 |         23.74 |
| thumb  |    26.47 |    12.76 |  0    |  5.87 |         19.87 |
| index  |    28.45 |    12.07 |  0    |  5.47 |         18.38 |
| middle |    27.07 |    11.64 |  0    |  5.32 |         16.96 |
| ring   |    22.93 |     8.53 |  0    |  4.74 |         13.51 |
| pinky  |    13.97 |     4.74 |  0    |  3.18 |          7.54 |
| palmar |    60.69 |    29.91 |  3.31 |  8.59 |         59.26 |
| dorsal |    48.02 |    22.84 |  1.72 |  8.09 |         40.74 |

### wilor (fit2d), 804 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    10.95 |     5.35 |  0    |  2.91 |         16.31 |
| thumb  |    16.67 |     8.08 |  0    |  4.27 |         18.43 |
| index  |    26.99 |    13.81 |  0    |  5.74 |         23.42 |
| middle |    30.97 |    14.93 |  0    |  6.15 |         23.72 |
| ring   |    27.11 |    10.82 |  0    |  5.11 |         13.68 |
| pinky  |    15.67 |     4.23 |  0    |  3.18 |          4.45 |
| palmar |    56.22 |    29.6  |  2.92 |  9.05 |         55.13 |
| dorsal |    48.76 |    22.26 |  1.86 |  8.3  |         44.87 |

### hopformer (native), 932 contact hand-frames

| part   |   >2mm % |   >5mm % |   p50 |   p90 |   vol share % |
|:-------|---------:|---------:|------:|------:|--------------:|
| palm   |    11.05 |     6.97 |  0    |  2.59 |         11.89 |
| thumb  |    28.97 |    13.09 |  0    |  6    |         20.92 |
| index  |    28.86 |    15.13 |  0    |  6.57 |         23.32 |
| middle |    25.86 |    14.7  |  0    |  6.89 |         21.63 |
| ring   |    18.03 |     8.69 |  0    |  4.71 |         15.12 |
| pinky  |    11.05 |     5.9  |  0    |  2.4  |          7.12 |
| palmar |    63.73 |    35.73 |  3.41 | 10.71 |         61    |
| dorsal |    51.82 |    29.4  |  2.33 |  9.63 |         39    |
