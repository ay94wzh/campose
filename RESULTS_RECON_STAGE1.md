# Stage 1 — Lift recon screening (Arms A, B, C) and Gate 1

Date: 2026-11-21
Git baseline: `a634c45` (AGENTPLAN.md)
Status: **Stage 1 complete. Gate 1 result: pause** — no arm clears the
≥10-point unseen-pose threshold.

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
| C `multi_plucker` | 29/50 (58%) | **21/50 (42%)** | 14 / 4 | 11 / 9 |

A and B are their 50-episode in-training finals (same `BaselineEvaluator`,
pose files, episode seeds and diffusion-noise seeds). C's final is the
50-episode offline Gate 1 evaluation; its in-training curve used the reduced
10-episode protocol. Paired counts use the same episode indices.

Training losses at the final epoch:

| Arm | train loss | diffusion | recon | val diffusion | val recon |
|---|---:|---:|---:|---:|---:|
| A `z_only` | 0.00637 | 0.00375 | 0.02617 | 0.00644 | 0.08393 |
| B `single_plucker` | 0.00567 | 0.00358 | 0.02087 | 0.01073 | 0.07016 |
| C `multi_plucker` | 0.00571 | 0.00362 | 0.02084 | 0.00327 | 0.07748 |

## Curves

A and B (50-episode, train/test):

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

C used the reduced protocol (10 episodes), so its curve is comparable only to
itself: 0: 0/0, 1000: 1/0, 2000: 1/2, 3000: 1/1, 4000: 1/2, 5000: 3/4,
6000: 4/3, 7000: 4/3, 8000: 7/4, 9000: 4/6, 10000: 7/3 (train/test). Its
50-episode final is 29/50 train, 21/50 test.

Best `train_cameras` checkpoint (secondary, exploratory): A epoch 10000
(30/50 train, 21/50 test); C epoch 10000 (29/50, 21/50); B epoch 9500
(26/50, 18/50).

## Gate 1 decision

Threshold: Exp C (or the best arm) must improve `test_cameras` by ≥10 points
over the 19/50 baseline, or reach ≥2× when the baseline is ≤10/50.

- A: +2 test points (21 vs 19) and +11 train points (paired 15-vs-4).
- B: +1 test point (20 vs 19) and +4 train points (paired 11-vs-7).
- C: +2 test points (21 vs 19) and +10 train points (paired 14-vs-4).

**No arm clears the threshold.** All test differences are within the ±5-point
noise band (50 episodes carry ~±7% standard error). The pattern is the plan's
"arm > baseline on train but not test" row: the auxiliary branch improves
seen-pose success but does not generalize to unseen input-camera poses at this
schedule/λ. Reconstruction itself is healthy (recon L1 ≈0.021 train,
0.070–0.084 validation), so this is not a collapsed auxiliary task.

**Gate 1 = pause.** Do not claim viewpoint robustness from Stage 1.

Per explicit user instruction, Stages 2–3 continue with the single best mode
only (A `z_only`, 30/50 train / 21/50 test) as a structural-feasibility check,
not as a gate pass. Can and Square use the existing batch-16 spatial-softmax
baselines; Stage 2's Gate 2 is K=1 50 episodes plus a K=2 random-view
addendum (25 episodes), and Stage 3's Gate 3 is K=1 50 plus K=2 random 25.

## Notes

- One seed (seed 0); these are screening results, not final claims.
- 50-episode success has ~±7% standard error; 5-point differences are noise.
- C used 10-episode intermediate evals for speed; its final is 50 episodes.
- B's final 500 epochs were resumed after a transient co-tenant OOM; the
  epoch-9500 checkpoint resumed to 10000 with identical batch/protocol.
  See `/data/zihan/campose_logs/recon_campaign_status.md`.
- This commit adds the C/Gate 1 results to the A+B note.

## Artifacts

- Checkpoints: `/data/zihan/campose_checkpoints/lift_recon_{z_only,single_plucker,multi_plucker}_seed_0/`
  with epochs 7000/8000/9000/10000 retained in `<run>_keep/`.
- Online eval JSONs: `<ckpt_dir>/eval_epoch_<E>_{train,test}_cameras/`.
- C's offline Gate 1 eval:
  `/data/zihan/campose_eval/recon_lift/lift_recon_multi_plucker_seed_0/epoch_10000/`.
- Training logs: `/data/zihan/campose_logs/recon_lift_lift_recon_<mode>_seed_0.log`.
- Gate 1 report: `/data/zihan/campose_logs/stage1_gate1_report.txt`.
