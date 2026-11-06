# `baseline_can_rand_seed_0` — epoch-50 eval videos

Agentview recordings from the epoch-50 evaluation of the baseline run
`baseline_can_rand_seed_0` on the CanRand task. Every recorded episode is
marked `success_False` in its filename.

## Run details

| Setting | Value |
|---|---|
| Policy | `baseline` (B1 variant A + diffusion policy) |
| Task / dataset | CanRand, `canrand_eef_delta.hdf5` |
| Seed | 0 |
| Cameras | 2 side cameras (`num_side_cam=2`, `n=4`, `m=2`) |
| Eval | every 50 epochs, 10 episodes × 600 steps |

The folder holds episodes 0–8 (9 of the 10 scheduled episodes); no
`success_by_seed.json` was written for epoch 50, so the evaluation did not run
to completion.

The zero-success result and its root cause (the policy collapses to a 9-D
state-only controller and ignores the image) are documented in
[`PROGRESS.md` §6](../../../PROGRESS.md#6-diagnosis-of-the-zero-success-result).

## Files

| Episode | Video |
|---|---|
| 0 | [epoch_50_episode_0_agentview_success_False.mp4](epoch_50_episode_0_agentview_success_False.mp4) |
| 1 | [epoch_50_episode_1_agentview_success_False.mp4](epoch_50_episode_1_agentview_success_False.mp4) |
| 2 | [epoch_50_episode_2_agentview_success_False.mp4](epoch_50_episode_2_agentview_success_False.mp4) |
| 3 | [epoch_50_episode_3_agentview_success_False.mp4](epoch_50_episode_3_agentview_success_False.mp4) |
| 4 | [epoch_50_episode_4_agentview_success_False.mp4](epoch_50_episode_4_agentview_success_False.mp4) |
| 5 | [epoch_50_episode_5_agentview_success_False.mp4](epoch_50_episode_5_agentview_success_False.mp4) |
| 6 | [epoch_50_episode_6_agentview_success_False.mp4](epoch_50_episode_6_agentview_success_False.mp4) |
| 7 | [epoch_50_episode_7_agentview_success_False.mp4](epoch_50_episode_7_agentview_success_False.mp4) |
| 8 | [epoch_50_episode_8_agentview_success_False.mp4](epoch_50_episode_8_agentview_success_False.mp4) |

## Downloading

Clone the repo and take the folder, or click a video above and use GitHub's
**Download** button:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  git@github.com:ay94wzh/campose.git
cd campose
git sparse-checkout set eval_videos/baseline_can_rand_seed_0/eval_epoch_50_agentview
```
