# `baseline_lift_rand_seed_0` — eval videos

Agentview recordings from the full LiftRand baseline run. There are 21
evaluations (epochs 0–1000, every 50 epochs) and **every episode failed**:
0 successes out of 530 evaluated episodes.

## Run details

| Setting | Value |
|---|---|
| Policy | `baseline` (B1 variant A + diffusion policy) |
| Task / dataset | LiftRand, `liftrand_eef_delta.hdf5` |
| Seed | 0 |
| Cameras | 1 side camera (`num_side_cam=1`, `n=3`, `m=1`) |
| Eval | every 50 epochs, 600 steps per episode |
| Eval episodes | 10 through epoch 600, 50 from epoch 650 on |
| Saved videos | first 10 episodes of each evaluation (`eval_save_n_video=10`) |
| Checkpoints | `/data/zihan/campose_checkpoints/baseline_lift_rand_seed_0` (not in repo) |

## Evaluations

Each epoch folder `eval_epoch_<N>_agentview/` holds the videos and the
`success_by_seed.json` from that evaluation.

| Epoch | Episodes evaluated | Successes | Saved videos |
|---|---|---|---|
| 0 | 10 | 0 | 10 |
| 50 | 10 | 0 | 10 |
| 100 | 10 | 0 | 10 |
| 150 | 10 | 0 | 10 |
| 200 | 10 | 0 | 10 |
| 250 | 10 | 0 | 10 |
| 300 | 10 | 0 | 10 |
| 350 | 10 | 0 | 10 |
| 400 | 10 | 0 | 10 |
| 450 | 10 | 0 | 10 |
| 500 | 10 | 0 | 10 |
| 550 | 10 | 0 | 10 |
| 600 | 10 | 0 | 10 |
| 650 | 50 | 0 | 10 |
| 700 | 50 | 0 | 10 |
| 750 | 50 | 0 | 10 |
| 800 | 50 | 0 | 10 |
| 850 | 50 | 0 | 10 |
| 900 | 50 | 0 | 10 |
| 950 | 50 | 0 | 10 |
| 1000 | 50 | 0 | 10 |

The zero-success result and its root cause (the policy collapses to a 9-D
state-only controller and ignores the image) are documented in
[`PROGRESS.md` §6](../../PROGRESS.md#6-diagnosis-of-the-zero-success-result).

## Downloading

Clone the repo and take the folder, or click a video in the GitHub UI and use
the **Download** button:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  git@github.com:ay94wzh/campose.git
cd campose
git sparse-checkout set eval_videos/baseline_lift_rand_seed_0
```

Direct raw URLs follow this pattern (public repo, no login needed):

```
https://raw.githubusercontent.com/ay94wzh/campose/method/baseline/eval_videos/baseline_lift_rand_seed_0/eval_epoch_1000_agentview/epoch_1000_episode_0_agentview_success_False.mp4
```
