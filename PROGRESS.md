# Baseline v1/v2 — key record

Scope: only the versions and results that follow the original repo's
conventions:

- **input frame** — DP `n_obs_steps=1` (one frame at `t`), action window
  `[t, t+16)`, execute 8 actions; 7-D joint qpos zeroed **before** the DP
  min-max normalisation; `rgb/255` + raw Plücker, no ImageNet normalisation;
- **training** — LiftRand seed 0, batch 32, LR 1e-4, 10001 epochs, same
  diffusion objective/schedule for both versions;
- **eval** — the origin camera-pose protocol (`train_cameras` /
  `test_cameras`, episode `i` uses pose `i`, 50 episodes × 600 steps).

**Key result (LiftRand, epoch 9500, origin pose protocol):** v1 (`avgpool`) —
31/50 (62%) on `train_cameras`, 29/50 (58%) on `test_cameras`; v2
(`spatial_softmax`) — 17/50 (34%) on both.

## 1. v1 (avgpool) — information flow and architecture

v1 is v2's stack with one change: ResNet18 layer4 is reduced by **global
average pooling** instead of per-view spatial-softmax keypoints.
CLI: `--baseline_pool avgpool`; checkpoint marker
`avgpool_nobs1_proprio7_v1`.

### 1.1 Sample construction

```
demo d, time t, view k with camera pose T_k ~ W(d) = [m*d, m*d+n)
  MuJoCo state(t) --render(agentview @ T_k)--> RGB 256x256 --flipud, /255--+
  T_k --Plucker(intrinsics, extrinsics)--> 6x256x256 (raw) ----------------+--> image_k (9,256,256)

  EEF history t-3..t, 11 dims/step:
    [R_c^T (p_eef - p_cam) (3), rot6d(R_c^T R_eef) (6), gripper (2)]
  --z-score--> eef_hist_k (44) --cond_mlp(44->128->128)--> FiLM condition

  qpos[:7] --zero with p=prob_drop_proprio (default 1.0)--> (qpos-mean)/std --> proprio (7)
  actions[t:t+16] --(a-mean)/std--> target (16,7) + is_pad mask
```

- `K=1` (`--num_side_cam 1 --n 3 --m 1`); the pose is sampled uniformly from
  the demo's training window.
- Train-time crop: one random crop (side 204–256) resized to 256 with NEAREST,
  shared across the frames of a view. Eval uses no crop.
- With `prob_drop_proprio=1.0` the 7-D proprio is the constant
  `-state_mean/state_std`, exactly as in the original DP.
- EEF history is **past-only**: every index is `<= t`.

### 1.2 Network

```
image_k (9,256,256)
 -> conv1 7x7/2, 64   (9-channel; RGB weights from ResNet18 init,
                       6 Plucker channels zeroed)
 -> GN(64) -> ReLU -> FiLM[64] -> maxpool/2
 -> layer1 (64,  2 BasicBlocks) -> FiLM[64]
 -> layer2 (128, 2 BasicBlocks) -> FiLM[128]
 -> layer3 (256, 2 BasicBlocks) -> FiLM[256]
 -> layer4 (512, 2 BasicBlocks) -> FiLM[512]
 -> 512 x 8 x 8
 -> AdaptiveAvgPool2d(1) -> z_k (512)
```

Then:

```
z_k for each active view (view_mask)
 -> MHA(embed=512, 8 heads), learnable query (1,1,512), inactive views masked
 -> z_global (512)
 -> concat[z_global (512), proprio (7)] = 519-D observation feature
 -> ConditionalUnet1D -> noise / action chunk
```

- ResNet18 is trained **from scratch** (`weights=None`), with every
  BatchNorm2d replaced by GroupNorm (`C/16` groups). FiLM is applied at the
  stem (after `conv1`-GN-ReLU, before maxpool) and after layer1–layer4; FiLM
  layers are zero-initialised (identity at init).
- The global-pooled per-view latent is 512-D; v1 deliberately discards the
  layer4 spatial layout.

### 1.3 Diffusion head

- `ConditionalUnet1D(input_dim=7, global_cond_dim=519,
  down_dims=(512,1024,2048), kernel_size=5, n_groups=8,
  cond_predict_scale=True)`, diffusion-step embedding 128.
- DDPM scheduler: 100 train timesteps, `beta_start=1e-4`,
  `beta_end=2e-2`, `squaredcos_cap_v2`, `variance_type=fixed_small`,
  epsilon prediction, `clip_sample=True`, range ±1.0.
- Horizon 16; inference uses 100 denoising steps and executes the first 8
  actions (`start = n_obs_steps-1 = 0`), then replans.

## 2. v2 (spatial_softmax) — information flow and architecture

v2 is identical to §1 in data, FiLM conditioning, cross-view fusion, 519-D
observation feature, UNet, diffusion schedule, training, and evaluation. The
**only** difference is the per-view reduction of the `512 x 8 x 8` layer4 map:

```
512 x 8 x 8
 -> SpatialSoftmax(num_kp=32):
      1x1 conv 512 -> 32, softmax over the 64 spatial positions,
      expected (x, y) per keypoint -> 64
 -> Linear(64 -> 512) -> z_k (512)
 -> (same cross-view MHA, concat proprio, 519-D, UNet)
```

CLI: `--baseline_pool spatial_softmax` (default); checkpoint marker
`spatial_softmax_nobs1_proprio7_v1`. Unlike v1, it keeps the spatial layout of
the feature map, so the per-view token can encode object location.

### v1 vs v2

| Item | v1 (`avgpool`) | v2 (`spatial_softmax`) |
|---|---|---|
| layer4 reduction | global average pool → 512 | 32 spatial-softmax keypoints → Linear → 512 |
| per-view latent | 512 | 512 |
| observation feature | 519 | 519 |
| everything else | identical | identical |

## 3. Training method

### Data

- `liftrand_eef_delta.hdf5`: 300 demos; **200 train** (indices 0–199) +
  **10 validation** (200–209).
- `n=3, m=1`: training camera windows `W(d) = [d, d+3)` cover poses
  **0–201** of `train_cameras.json` (1000 poses).
- Normalisation statistics (`dataset_stats.json`): DP `action_mean/std` (7),
  `state_mean/std` (7), plus `eef_hist_mean/std` (44) computed over every
  demo × every camera pose in its training window. No `eef_mean/std`.
- Validation uses the first validation demos and `test_cameras.json` with
  `transform="id"` (no crop).

### Objective

Standard DDPM epsilon prediction. For each batch:

1. normalise the 16-step action window;
2. sample `noise ~ N(0, I)` and `u ~ Uniform{0..99}`;
3. noise the actions at `u`;
4. the UNet predicts the noise from `(noisy actions, u, global_cond=519)`;
5. MSE over the horizon, with padded entries masked by `is_pad`.

### Optimiser and schedule

| Setting | Value |
|---|---|
| Optimiser | AdamW, betas `(0.95, 0.999)`, eps `1e-8` |
| LR / weight decay | `1e-4` / `1e-6` |
| Schedule | cosine with 500-step warmup over `num_epochs × len(dataloader)` |
| Grad clip | 1.0 |
| Precision | bf16 autocast |
| Batch / epochs / seed | 32 / 10001 / 0 |


Representative command (v1; v2 is the same with
`--baseline_pool spatial_softmax` and its own `--name`/`--ckpt_dir`):

```
python policy_robosuite/train.py \
  --policy_class baseline --name baseline_lift_rand_avgpool_seed_0 \
  --dataset_suffix liftrand_eef_delta --num_side_cam 1 --n 3 --m 1 \
  --batch_size 32 --seed 0 --num_epochs 10001 \
  --eval_start_epoch 6000 --eval_every 500 --eval_max_steps 600 \
  --save_every 500 --baseline_pool avgpool \
  --lr 1e-4 --weight_decay 1e-6 \
  --ckpt_dir /data/zihan/campose_checkpoints/baseline_lift_rand_avgpool_seed_0
```

## 4. Evaluation method

### Camera-pose protocol

- Pose sets: `train_cameras.json` (1000 poses) and `test_cameras.json`
  (100 held-out views).
- Episode `i` uses `poses[i]` with `num_side_cam=1`, exactly as the origin
  `Evaluator`.
- Eval on `train_cameras` episodes 0–49 uses poses 0–49, which are inside the
  training coverage (0–201); `test_cameras` episodes 0–49 are held-out views.
- `--baseline_eval_agentview` defaults to false; the agentview protocol is only
  an explicit opt-in and its numbers are not part of this record.

### Rollout

1. `env.reset()` (or restore `init_state`), then `env.set_init_action()` for
   `eef_delta`.
2. Seed `np.random.seed(episode_num)` around the rollout, so object placement
   is deterministic per episode (the global RNG state is restored afterwards).
3. Set the camera to the episode's pose, recompute the Plücker map, keep the
   observation buffer at one frame (DP `n_obs_steps=1`), and initialise the
   4-step EEF deque with the current EEF repeated.
4. At every replan: render 256² through the `agentview` slot, which still
   holds the episode's selected pose (`rgb/255` + Plücker), read raw
   `qpos[:7]`, apply proprio dropout + min-max normalisation, z-score the
   44-D EEF history, and run 100 DDPM steps with noise seeded by
   `(episode_num + 1) * 100003 + step`.
5. Denormalise the 16-step chunk, execute up to its first **8** actions (or
   until done / 600 steps), updating the EEF deque and the single-frame
   snapshot after every environment step.
6. Success if any reward in the episode is `1`. The first
   `eval_save_n_video` episodes (10 in training, 2 in offline eval) are saved
   as MP4s.
7. Each eval writes `eval_epoch_<N>_<pose_set>/` with videos and
   `success_by_seed.json`.

- The evaluator reads the EEF **body** quaternion (`get_body_xquat`), matching
  the training HDF5's `robot0_eef_quat`; using the gripper-site orientation
  would put the EEF state ~90° out of distribution.
- **Offline** evaluation (`eval_baseline_checkpoint.py`) loads a saved
  checkpoint plus its `dataset_stats.json` and runs the same
  `BaselineEvaluator`, pose files, episode seeds and diffusion-noise seeds, so
  v1 and v2 see identical initial states and noise (paired comparison).
- Cost: roughly 10–20 min per pose set per 50-episode checkpoint eval.
- 50-episode success rates carry ~7% standard error.


## 5. Results

### 5.1 v1 online, origin pose protocol

Evaluation every 500 epochs; 10 episodes through epoch 6000, 50 from epoch
6500; 600 steps.

| Epoch | `train_cameras` | `test_cameras` |
|---:|---:|---:|
| 0 | 0/10 (0%) | 0/10 (0%) |
| 500 | 0/10 | 0/10 |
| 1000 | 0/10 | 0/10 |
| 1500 | 0/10 | 0/10 |
| 2000 | 1/10 (10%) | 4/10 (40%) |
| 2500 | 0/10 | 1/10 (10%) |
| 3000 | 4/10 (40%) | 2/10 (20%) |
| 3500 | 3/10 (30%) | 1/10 (10%) |
| 4000 | 5/10 (50%) | 3/10 (30%) |
| 4500 | 3/10 (30%) | 3/10 (30%) |
| 5000 | 5/10 (50%) | 4/10 (40%) |
| 5500 | 4/10 (40%) | 1/10 (10%) |
| 6000 | 1/10 (10%) | 4/10 (40%) |
| 6500 | 24/50 (48%) | 19/50 (38%) |
| 7000 | 24/50 (48%) | 20/50 (40%) |
| 7500 | 24/50 (48%) | 22/50 (44%) |
| 8000 | 30/50 (60%) | 27/50 (54%) |
| 8500 | 30/50 (60%) | 26/50 (52%) |
| 9000 | 29/50 (58%) | 27/50 (54%) |
| 9500 | **31/50 (62%)** | **29/50 (58%)** |
| 10000 | 25/50 (50%) | 27/50 (54%) |

Best epoch-9500 checkpoint: **31/50 (62%) on `train_cameras`, 29/50 (58%) on
`test_cameras`**.

### 5.2 v1 vs v2 under the same protocol

v2 was trained under the same input/training conventions; its checkpoints
were evaluated offline with the same evaluator, pose files and episode seeds.

| Model | Checkpoint | `train_cameras` (50) | `test_cameras` (50) |
|---|---|---:|---:|
| v2 (`spatial_softmax`) | epoch 1000 | 0/50 (0%) | 2/50 (4%) |
| v1 (`avgpool`) | epoch 1000 | 0/50 (0%) | 0/50 (0%) |
| v2 (`spatial_softmax`) | epoch 9500 | 17/50 (34%) | 17/50 (34%) |
| v1 (`avgpool`) | epoch 9500 | **31/50 (62%)** | **29/50 (58%)** |
| v2 (`spatial_softmax`) | epoch 10000 | 19/50 (38%) | 19/50 (38%) |
| v1 (`avgpool`) | epoch 10000 | 25/50 (50%) | 27/50 (54%) |

## 6. Notes / open observations

### 6.1 The v1 win is Lift-specific

- §5 measures **LiftRand only**. The v1 (`avgpool`) > v2
  (`spatial_softmax`) ranking therefore applies to Lift; it must **not** be
  generalized to Can/Square. The cross-task observation (runs outside this
  file) is that v1 does not beat v2 on those tasks, and this file contains
  no protocol-aligned Can/Square tables (only smoke tests in earlier logs).
  If those results are to be cited, add them with the same pose protocol,
  episode seeds and 50-episode counts.

### 6.2 Epoch behavior: v1 peaks, v2 looks still rising (not yet established)

- **v1 has a dense pose-protocol curve.** Success rises to a peak at epoch
  9500 — 31/50 train, 29/50 test — then falls at epoch 10000 to 25/50 and
  27/50. The decline is also visible in v1's training loss
  (0.0035 at 9000 → 0.0037 at 9500 → 0.0041 at 10000), consistent with
  saturation/overfitting after ~9500.
- **v2 has only three aligned checkpoints.** Pose-protocol success is
  0/50 train, 2/50 test at epoch 1000; 17/50, 17/50 at 9500; 19/50, 19/50 at
  10000. That looks monotonic and suggests v2 might still improve with more
  epochs.
- **Follow-up.** Evaluate v2's retained `epoch_9000.pth` under the pose
  protocol to fill in the late curve. If the rising trend survives, train v2
  longer or evaluate intermediate checkpoints before concluding it is
  undertrained relative to v1.