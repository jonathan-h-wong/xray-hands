# Camera-frame hands, run 20261113_ego_s5

These are the exact hands every metric in this run was computed on. They come from
`baselines/eval_arctic.py --export`, and their `mpjpe_cam` values reproduce the frames parquet to within 1e-5 mm.

- `native/`: the calibrated ARCTIC intrinsics. Methods: `hamer` and `wilor` (lifts `wp` and `fit2d`), and `hopformer` (`native`).
- `moge/`: the 2D methods lifted with MoGe-2 estimated intrinsics, plus `hopformer_moge`, which was fed the MoGe K.

There is one `npz` per sequence/view, named `<sid>__<seq>__v<view>.npz`. In the keys below, `<side>` is `left` or `right`:

| key | shape | meaning |
|---|---|---|
| `<side>.eval_rows` | (n,) | row indices into `baselines/eval_sets/arctic_p2_val_v0_s5.npz` |
| `<side>.frame` | (n,) | ARCTIC frame index |
| `gt.<side>.verts`, `gt.<side>.joints` | (n,778,3), (n,21,3) | ground-truth MANO hand, metres, camera frame of the view (undistorted) |
| `<method>.<lift>.<side>.verts` / `.joints` | (n,778,3), (n,21,3) | predicted hand in the same frame; NaN = not detected |
| `<method>.<lift>.<side>.score` | (n,) | detector score |
| `gt.obj_T_cam_mm.<side>` | (n,4,4) | GT object-to-camera transform, mm |
| `gt.obj_radian.<side>` | (n,) | GT articulation angle |
| `hopformer.*.obj_rot.<side>`, `obj_cam_t_m.<side>`, `obj_radian.<side>` | | HOPformer's predicted object pose (translation in metres) |
| `K_full`, `sid`, `seq`, `view`, `obj` | | full-resolution intrinsics and metadata |

Joints are the 16 joints regressed by MANO (`J_regressor`) plus 5 fingertip vertices in the order thumb, index, middle, ring, pinky.
The same definition is applied to the ground truth and to every method.
