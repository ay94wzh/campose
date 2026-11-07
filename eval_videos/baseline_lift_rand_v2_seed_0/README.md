# `baseline_lift_rand_v2_seed_0` — eval videos

Agentview recordings from the v2 LiftRand baseline run. There are 21
evaluations (epochs 0–10000, every 500 epochs) with **146 successes out of 530
evaluated episodes** (27.5%), unlike the v1 run, which was 0/530.

## Run details

| Setting | Value |
|---|---|
| Policy | `baseline` v2 (`spatial_softmax_nobs1_proprio7_v1`) |
| Task / dataset | LiftRand, `liftrand_eef_delta.hdf5` |
| Seed | 0 |
| Cameras | 1 side camera (`num_side_cam=1`, `n=3`, `m=1`) |
| Eval | every 500 epochs, 600 steps per episode |
| Eval episodes | 10 through epoch 6000, 50 from epoch 6500 on |
| Saved videos | first 10 episodes of each evaluation (`eval_save_n_video=10`) |
| Checkpoints | `/data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0` (not in repo) |

## Evaluations

Each epoch folder `eval_epoch_<N>_agentview/` holds the videos and the
`success_by_seed.json` from that evaluation.

| Epoch | Episodes evaluated | Successes | Saved videos |
|---|---|---|---|
| 0 | 10 | 0 | 10 |
| 500 | 10 | 0 | 10 |
| 1000 | 10 | 0 | 10 |
| 1500 | 10 | 0 | 10 |
| 2000 | 10 | 1 | 10 |
| 2500 | 10 | 2 | 10 |
| 3000 | 10 | 3 | 10 |
| 3500 | 10 | 4 | 10 |
| 4000 | 10 | 2 | 10 |
| 4500 | 10 | 0 | 10 |
| 5000 | 10 | 1 | 10 |
| 5500 | 10 | 1 | 10 |
| 6000 | 10 | 0 | 10 |
| 6500 | 50 | 7 | 10 |
| 7000 | 50 | 12 | 10 |
| 7500 | 50 | 12 | 10 |
| 8000 | 50 | 19 | 10 |
| 8500 | 50 | 18 | 10 |
| 9000 | 50 | 21 | 10 |
| 9500 | 50 | 25 | 10 |
| 10000 | 50 | 18 | 10 |

The run details, full results table, and the image-sensitivity check are
documented in [`PROGRESS.md`](../../PROGRESS.md).

## Downloading

Clone the repo and take the folder, or click a video in the GitHub UI and use
the **Download** button:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  git@github.com:ay94wzh/campose.git
cd campose
git sparse-checkout set eval_videos/baseline_lift_rand_v2_seed_0
```

Direct raw URLs follow this pattern (public repo, no login needed):

```
https://raw.githubusercontent.com/ay94wzh/campose/method/baseline/eval_videos/baseline_lift_rand_v2_seed_0/eval_epoch_10000_agentview/epoch_10000_episode_0_agentview_success_False.mp4
```
