# ARCTIC baselines: selftest_noise0

eval set: {"protocol": "p2", "split": "val", "views": [0], "stride": 5, "grounded_thresh_mm": 2.0, "require_both": false, "contact_available": true, "n_images": 2697, "n_hands": {"left": 1749, "right": 1671}}

intrinsics: native

## All selected hands (metrics over detected hands)

|                     |   ('gt', '-') |   ('oracle_hand2d', 'wp') |   ('oracle_hand2d', 'fit2d') |   ('oracle_hop', 'native') |
|:--------------------|--------------:|--------------------------:|-----------------------------:|---------------------------:|
| hands               |        236    |                    236    |                       236    |                     236    |
| detected %          |        100    |                    100    |                       100    |                     100    |
| pen p50             |          0    |                      0    |                         0    |                       0    |
| pen p90             |          1.82 |                      1.82 |                         1.82 |                       1.82 |
| pen>2mm %           |          0    |                      0    |                         0    |                       0    |
| pen>5mm %           |          0    |                      0    |                         0    |                       0    |
| contact %           |         52.12 |                     52.12 |                        52.12 |                      52.12 |
| mpjpe_ra p50        |        nan    |                      0    |                         0    |                       0    |
| pa_mpjpe p50        |        nan    |                      0    |                         0    |                       0    |
| pa_mpvpe p50        |        nan    |                      0    |                         0    |                       0    |
| root_err p50        |        nan    |                      0    |                         0    |                       0    |
| mpjpe_cam p50       |        nan    |                      0    |                         0    |                       0    |
| kp2d_err_px p50     |        nan    |                      0    |                         0    |                       0    |
| depth_err p50       |        nan    |                      0    |                         0    |                      -0    |
| |depth_err| p50     |        nan    |                      0    |                         0    |                       0    |
| mrrpe_rl p50        |        nan    |                      0    |                         0    |                       0    |
| contact recall %    |        nan    |                    100    |                       100    |                     100    |
| contact precision % |        nan    |                    100    |                       100    |                     100    |
| ra pen p50          |        nan    |                      0    |                         0    |                       0    |
| ra pen>5mm %        |        nan    |                      0    |                         0    |                       0    |
| self pen p50        |        nan    |                    nan    |                       nan    |                       0    |
| obj t err p50       |        nan    |                    nan    |                       nan    |                       0    |
| obj rot err p50     |        nan    |                    nan    |                       nan    |                       0    |
| mrrpe_ro p50        |        nan    |                    nan    |                       nan    |                       0    |
| obj cls %           |        nan    |                    nan    |                       nan    |                     100    |

## Common subset: 236 hands detected by every method

|                 |   ('gt', '-') |   ('oracle_hand2d', 'wp') |   ('oracle_hand2d', 'fit2d') |   ('oracle_hop', 'native') |
|:----------------|--------------:|--------------------------:|-----------------------------:|---------------------------:|
| hands           |           236 |                       236 |                          236 |                        236 |
| pen p50         |             0 |                         0 |                            0 |                          0 |
| pen>5mm %       |             0 |                         0 |                            0 |                          0 |
| mpjpe_ra p50    |           nan |                         0 |                            0 |                          0 |
| pa_mpjpe p50    |           nan |                         0 |                            0 |                          0 |
| root_err p50    |           nan |                         0 |                            0 |                          0 |
| kp2d_err_px p50 |           nan |                         0 |                            0 |                          0 |
| ra pen p50      |           nan |                         0 |                            0 |                          0 |

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

### oracle_hand2d (wp), 123 contact hand-frames

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

### oracle_hand2d (fit2d), 123 contact hand-frames

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

### oracle_hop (native), 123 contact hand-frames

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
