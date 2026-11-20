# Stage 1 — Lift recon screening, Arms A and B

Date: 2026-11-21
Git baseline: `a634c45` (AGENTPLAN.md)
Status: Arm C (`multi_plucker`) still training; this note covers the completed
A (`z_only`) and B (`single_plucker`) arms.

## Protocol

- LiftRand (`liftrand_eef_delta`), 200 training demos + 10 held-out validation
  demos, one observation frame, 16-step action window, execute 8 actions.
- K=1, `n=3`, `m=1`; input poses from `train_cameras.json`; evaluation on
  `train_cameras.json` (seen) and `test_cameras.json` (unseen), one pose per
  episode.
- Batch 32, seed 0, 10,001 epochs, AdamW (0.95, 0.999), lr 1e-4, wd 1e-6,
  cosine schedule with 500-step warmup, grad clip 1.0, bf16.
- Auxiliary branch: λ=0.1, target AgentView = default @ ΔT, translation
  U(±2 cm) camera-local, rotation U(±5°) per axis, target RGB 128×128
  (RGB/255, no crop), target Plücker 256×256. Branch dropped at inference.
- Control: `baseline_lift_rand_v2_seed_0/epoch_10000.pth` (spatial_softmax,
  no recon, batch 32), 19/50 `train_cameras` and 19/50 `test_cameras`.

## Final results — epoch 10000, 50 episodes per pose set, K=1

| Arm | train_cameras | test_cameras | paired train (arm>base/base>arm) | paired test |
|---|---:|---:|---:|---:|
| Baseline (v2) | 19/50 (38%) | 19/50 (38%) | — | — |
| A `z_only` | **30/50 (60%)** | **21/50 (42%)** | 15 / 4 | 13 / 11 |
| B `single_plucker` | 23/50 (46%) | 20/50 (40%) | 11 / 7 | 8 / 7 |

Paired counts use the same episode indices, object-placement seeds, and
diffusion-noise seeds as the baseline.

Training losses at the final epoch:

| Arm | train loss | diffusion | recon | val diffusion | val recon |
|---|---:|---:|---:|---:|---:|
| A `z_only` | 0.00637 | 0.00375 | 0.02617 | 0.00644 | 0.08393 |
| B `single_plucker` | 0.00567 | 0.00358 | 0.02087 | 0.01073 | 0.07016 |

## 50-episode curve (train/test)

| Epoch | A `z_only` | B `single_plucker` |
|---:|---:|---:|
| 6500 | 13 / 18 | 23 / 15 |
| 7000 | 25 / 21 | 18 / 18 |
| 7500 | 23 / 20 | 22 / 23 |
| 8000 | 22 / 22 | 25 / 22 |
| 8500 | 26 / 24 | 21 / 19 |
| 9000 | 24 / 22 | 25 / 17 |
| 9500 | 27 / 20 | 26 / 18 |
| 10000 | 30 / 21 | 23 / 20 |

Best `train_cameras` checkpoint (secondary, exploratory): A epoch 10000
(30/50 train, 21/50 test); B epoch 9500 (26/50 train, 18/50 test).

## Preliminary read

- **A vs baseline:** strong seen-pose gain (+11 points, paired 15-vs-4) but
  only +2 points on unseen poses (paired 13-vs-11). The unseen change is
  inside the ±7% standard error of a 50-episode estimate.
- **B vs baseline:** +4 train / +1 test; paired 11-vs-7 train and 8-vs-7 test.
  No meaningful unseen-pose gain.
- **B vs A:** A is better on both pose sets at epoch 10000 (30 vs 23 train,
  21 vs 20 test). Single-scale target geometry does not help over `z_g` alone
  in this run.
- Neither A nor B clears Gate 1's ≥10-point `test_cameras` threshold. Arm C
  (`multi_plucker`) is the remaining Gate 1 arm; Gate 1 is not decided yet.
- Reconstruction is healthy: train recon L1 falls to 0.021–0.026 (from ~0.43
  at initialization), and validation recon is 0.070–0.084.

## Notes

- One seed (seed 0); these are screening results, not final claims.
- 50-episode success has ~±7% standard error; 5-point differences are noise.
- A and B used the original 50-episode evaluation cadence. Arm C and the
  later stages use a reduced intermediate-evaluation protocol (10 episodes
  per pose set) for speed; final gate evaluations remain 50 episodes.
- B's final 500 epochs were resumed after a transient co-tenant OOM; the
  epoch-9500 checkpoint resumed to 10000 with identical batch/protocol.
  See `recon_campaign_status.md` for the incident.
- This commit also adds `--eval_final_episodes` to `train.py` (default 50,
  preserving the old behavior) for the reduced intermediate-evaluation
  protocol.

## Artifacts

- Checkpoints: `/data/zihan/campose_checkpoints/lift_recon_{z_only,single_plucker}_seed_0/`
  with epochs 7000/8000/9000/10000 retained in `<run>_keep/`.
- Online eval JSONs: `<ckpt_dir>/eval_epoch_<E>_{train,test}_cameras/`.
- Training logs: `/data/zihan/campose_logs/recon_lift_lift_recon_<mode>_seed_0.log`.
