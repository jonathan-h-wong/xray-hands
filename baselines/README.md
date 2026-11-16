# ARCTIC baselines: HaMeR, WiLoR, HOPformer

Evaluates three hand reconstructors on a physically grounded subset of ARCTIC. The subset contains hand-frames
whose ground truth penetrates the object by less than 2 mm (`analysis/arctic_grounded_mask.py`) and whose hand
is visible in the view. Metrics cover hand pose and per-part penetration into the object, using the same
penetration definitions as `analysis/run_partwise.py`.

| Method | Type | Detector | Gets to metric 3D by |
|---|---|---|---|
| HaMeR | single-hand, crop based | ViTDet person → ViTPose whole-body hand keypoints (repo demo) | lifting its crop camera with the real intrinsics |
| WiLoR | single-hand, crop based | its own YOLO hand detector (repo demo) | same |
| HOPformer (ECCV 2026) | joint hands + object, full frame | needs hand crops; made here with WiLoR's YOLO | predicting directly in the camera frame using the patch intrinsics |

## Pipeline

| Step | Script | Where | Notes |
|---|---|---|---|
| 1 | `make_eval_set.py` | local (done) | Writes `eval_sets/arctic_p2_val_v0_s5.npz`: ego view, val split (s05), every 5th frame. 2,697 images; 1,749 left and 1,671 right hands. |
| 2 | `setup_remote.sh` | GPU box | Sets up the environments, repos and weights, the ARCTIC ground truth and the eval-set images. |
| 3 | `run_all.sh` | GPU box | Runs inference for all three methods and the MoGe-2 ablation, then evaluation. |
| — | `infer_hand2d.py --method hamer\|wilor` | GPU | Saves the raw crop outputs. Lifting happens at evaluation time, so the intrinsics can be swapped without rerunning the GPU step. |
| — | `infer_hopformer.py` | GPU | Rebuilds HOPformer's egocentric inputs without ARCTIC's processed splits. |
| — | `estimate_intrinsics_moge.py` | GPU | Estimates K with MoGe-2. This is an ablation only, because ARCTIC is calibrated. |
| — | `eval_arctic.py` | CPU | Writes `results/baselines/<tag>_frames.parquet` and `<tag>_summary.md`. |
| — | `selftest.py` | CPU | Runs oracle predictions through the whole evaluation. It passes: errors are 0 and the recomputed ground-truth penetration matches to 1e-4 mm. |

Secrets come from environment variables: `ARCTIC_USERNAME`, `ARCTIC_PASSWORD`, `MANO_USERNAME`, `MANO_PASSWORD`
and `HF_TOKEN`. Accept HOPformer's gated terms on Hugging Face first. Disk: the image download is about 12 GB (34 sequence zips of about 0.35 GB each, each deleted after
extraction), and about 0.3 GB of images are kept. Add about 25 GB of model weights and the conda environments.

```bash
git clone <this repo> && cd <repo>
bash baselines/setup_remote.sh          # one time
bash baselines/run_all.sh               # restartable
```

### Parallel runs (one model per pod, shared network volume)

Everything persistent lives on the volume (`baselines/pod_env.sh`): Miniforge, the three conda environments (`/workspace/envs`), the model repos and weights, `.venv-eval`, ARCTIC, predictions and results. Any pod that mounts the volume at `/workspace` can therefore run any job.

```bash
bash baselines/setup_remote.sh                                   # once, on one pod; restartable (markers in /workspace/.setup)
RUN_ID=20261113_ego_s5 bash baselines/run_method.sh wilor         # pod A
RUN_ID=20261113_ego_s5 bash baselines/run_method.sh hamer         # pod B
RUN_ID=20261113_ego_s5 bash baselines/run_method.sh hopformer     # pod C
RUN_ID=20261113_ego_s5 bash baselines/compare.sh                  # once all three have finished
```

Each job writes only its own folders (`baselines/preds/<run>/<method>`, `results/baselines/<run>_<method>*`, `logs/<run>/<method>.log`), so the pods never write the same file. Each job then uploads to `$S3_PREFIX/<run>/<method>/` (default `s3://xray-hands/eval/arctic`) when `AWS_ACCESS_KEY_ID` is set. A failed upload only warns; rerun it with `baselines/upload_run.sh <run> <method>`.

To evaluate every frame, rebuild the eval set locally with `--stride 1`. Then rerun `fetch_arctic_images.py`.

## Metrics (`eval_arctic.py`)

- **Pose (mm)**:
  - `mpjpe_ra` and `mpvpe_ra`: joint and vertex error after aligning the wrist.
  - `pa_mpjpe` and `pa_mpvpe`: the same after a full similarity (Procrustes) alignment.
  - `root_err`: absolute wrist error. `depth_err` is its signed depth component (predicted minus ground truth).
  - `mpjpe_cam`: absolute joint error in the camera frame.
  - `kp2d_err_px`: 2D reprojection error with the true K, in full-resolution pixels.
  - `mrrpe_rl_mm`: error in the right wrist's position relative to the left.
  - The joints are 16 MANO joints regressed from the mesh plus 5 fingertip vertices. They are computed identically for every method, so no repo's joint order matters.
- **Penetration into the ground-truth object**: overall `pen_max_mm`, `pen_vol_mm3` and `min_sd_mm`, plus the 12 regions (`<part>_<palmar|dorsal>_{pen,vol,npen,minsd}`).
  - `ra_pen_*`: the same after placing the predicted hand on the ground-truth wrist. This separates pose and shape error from lifting (depth) error.
  - `self_pen_*` (HOPformer only): penetration into its own predicted object.
  - Contact precision and recall are measured against ground-truth contact.
- **HOPformer object**: translation error, rotation error, articulation-angle error, class accuracy and `mrrpe_ro_mm`.
- **Lifts for HaMeR/WiLoR**:
  - `wp`: HaMeR's `cam_crop_to_full`, but with the true focal length and principal point.
  - `fit2d`: a least-squares translation that reprojects the model's 3D joints onto its own 2D keypoints.
- The summary has three tables:
  - all selected hands;
  - the **common subset** that every method detected (use this one for fair comparisons);
  - a per-part table over each method's contact frames.

## Browser viewer (`viewer/bench/`)

An interactive page for browsing any run. It has five tabs:

- **Summary**: one row per method/lift, with the median, mean or p90 of the headline metrics, plus p10/p50/p90 of every metric.
- **Distributions**: a CDF of any metric for every method, with a hover crosshair.
- **Breakdown**: any metric by object, sequence, subject, hand or GT contact. Click a sequence to open it in 3D.
- **Contact**: for each hand part, where the prediction lands (penetrating, touching, hovering, off or far), grouped by the GT part's distance to the object. Can be shown wrist-aligned.
- **Frames (3D)**: GT and predicted MANO meshes plus the posed GT object, in the camera frame. It also shows HOPformer's own object estimate, a wrist-aligned toggle, joints, per-frame metrics, a per-sequence timeline (click to jump) and "next worst" stepping.

The filters at the top (object, hand, GT contact, method chips) apply to every tab. The URL hash keeps the run, tab and frame, so a link reopens the same view.

```bash
# 1. export a run (writes viewer/bench/data/<run>/, git-ignored, and registers it in runs.json)
python baselines/export_bench_viewer.py --run 20261113_ego_s5 \
  --frames results/baselines/20261113_ego_s5_compare_fine_frames.parquet \
  --hands  results/baselines/hands_20261113_ego_s5/native --note "calibrated intrinsics"
python baselines/export_bench_viewer.py --run 20261113_ego_s5_moge \
  --frames results/baselines/20261113_ego_s5_compare_moge_frames.parquet \
  --hands  results/baselines/hands_20261113_ego_s5/moge --note "MoGe-2 intrinsics (ablation)"
# 2. serve and open http://localhost:8000/bench/
python -m http.server -d viewer 8000
```

The inputs are in `s3://xray-hands/eval/arctic/<run>/`:
- `compare/<run>_compare*_frames.parquet` are the `--frames` files;
- `hands/{native,moge}/` are the `--hands` directories.

The 3D view also needs the ARCTIC object templates and MANO faces (`data/arctic/...`, as for `eval_arctic.py`). Without `--hands`, every tab except the 3D one still works.

**Adding your own model:**
1. Write predictions in the per-shard `npz` format the baselines use. `infer_hand2d.py` and `infer_hopformer.py` show the keys; `eval_arctic.py` reads them.
2. Run `eval_arctic.py --sdf-pad 150 --export <dir>`.
3. Concatenate your `_frames.parquet` with the baseline compare parquet (drop the duplicate GT rows), so that your method is shown next to the baselines.
4. Export that parquet with `--hands` pointing to a directory that holds the merged npz. The export reads `<method>.<lift>.*` keys for every method in the parquet, so you can also merge the npz files key by key.

Methods get colours in a fixed order (HaMeR, WiLoR, HOPformer, then the rest alphabetically). Only the first 8 get distinct colours.

## Assumptions and caveats

1. **HOPformer hand crops.** Its loader reads pre-made 256 px crops (`cropped_right/`, `cropped_left/`). The repo neither ships nor documents how those crops were made for ARCTIC. Here they come from WiLoR's YOLO detector: a square box of 2.0 × the detection's larger side (`--crop-rescale`), passed through HOPformer's own reader (letterbox, flip the left crop, normalise). If its numbers fall clearly short of the paper's (val MPJPE 16.1 mm on all val frames, not this subset), suspect this step first.
2. **HOPformer is told the ground-truth object name** (`query_names`), as in its ARCTIC protocol, so its object class is not fully predicted. Its object pose is predicted.
3. **Detection in the egocentric view.** HaMeR's detector needs a *person* box (ViTDet), and the egocentric camera mostly sees arms. The summary reports detection rates; compare methods on the common subset. `--hamer-body-detector regnety` is the demo's alternative.
4. **Distortion.** Egocentric frames are distorted. HaMeR and WiLoR get undistorted images, so a pinhole lift is valid. HOPformer gets the raw distorted frame, as in its training. All ground truth and metrics are in undistorted camera space, as in ARCTIC.
5. **Ground truth.** It uses ARCTIC's stock left MANO, which is how ARCTIC was fit. By construction, the selected frames have ground-truth penetration of at most 2 mm, so predicted penetration measures error the method adds.
6. **Training overlap.** HOPformer p2 is trained on ARCTIC train (s01–s10 minus s03 and s05), so s05 is held out. Check whether HaMeR's and WiLoR's training mixes include ARCTIC before comparing them.
7. **Third-party installs** in `setup_remote.sh` follow each README as of 2026-11 and were not run here; no GPU was available locally. The data plumbing, lifting maths, penetration and summary code are covered by `selftest.py`.
