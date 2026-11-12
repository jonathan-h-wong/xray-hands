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

## Assumptions and caveats

1. **HOPformer hand crops.** Its loader reads pre-made 256 px crops (`cropped_right/`, `cropped_left/`). The repo neither ships nor documents how those crops were made for ARCTIC. Here they come from WiLoR's YOLO detector: a square box of 2.0 × the detection's larger side (`--crop-rescale`), passed through HOPformer's own reader (letterbox, flip the left crop, normalise). If its numbers fall clearly short of the paper's (val MPJPE 16.1 mm on all val frames, not this subset), suspect this step first.
2. **HOPformer is told the ground-truth object name** (`query_names`), as in its ARCTIC protocol, so its object class is not fully predicted. Its object pose is predicted.
3. **Detection in the egocentric view.** HaMeR's detector needs a *person* box (ViTDet), and the egocentric camera mostly sees arms. The summary reports detection rates; compare methods on the common subset. `--hamer-body-detector regnety` is the demo's alternative.
4. **Distortion.** Egocentric frames are distorted. HaMeR and WiLoR get undistorted images, so a pinhole lift is valid. HOPformer gets the raw distorted frame, as in its training. All ground truth and metrics are in undistorted camera space, as in ARCTIC.
5. **Ground truth.** It uses ARCTIC's stock left MANO, which is how ARCTIC was fit. By construction, the selected frames have ground-truth penetration of at most 2 mm, so predicted penetration measures error the method adds.
6. **Training overlap.** HOPformer p2 is trained on ARCTIC train (s01–s10 minus s03 and s05), so s05 is held out. Check whether HaMeR's and WiLoR's training mixes include ARCTIC before comparing them.
7. **Third-party installs** in `setup_remote.sh` follow each README as of 2026-11 and were not run here; no GPU was available locally. The data plumbing, lifting maths, penetration and summary code are covered by `selftest.py`.
