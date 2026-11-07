# Baseline run progress — 2026-11-05 / 2026-11-07

Session log for executing the baseline training runbooks (v1 and v2; now
consolidated in `PLAN.md`) on this machine.
Branch `method/baseline`; v1 runs at HEAD
`7af6bfa0b18d932366ad775ef6b756d9c7e807aa`, v2 run at HEAD
`a0b3e3d1694e1b60596b81ada259a5ff51f5b478`
**Latest result (v2, §6):** Lift seed 0 reached **146/530 = 27.5%** success
(best 25/50 = 0.50 at epoch 9500; final 18/50 = 0.36).

<!-- ## 1. Machine and environment

- GPU: 2× NVIDIA GeForce RTX 5090, 32.6 GB each.
  - GPU 0 was occupied by another user's job (~22.6 GB); it was not used.
  - GPU 1 was used (contained a ~4.7 GB job from another user).
- Conda env: `know_your_camera` (Python 3.10.20)
  - torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, CUDA available (12.8)
  - diffusers `0.40.0`, transformers `5.16.1`, mujoco `3.1.6`, h5py `3.16.0`
  - robosuite `1.5.1`, wandb `0.29.0`, gdown `6.1.0`
- EGL works: `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (no display; osmesa not needed). -->

<!-- ### Environment repairs

- The env's editable installs for `robosuite` and `mani_skill` pointed at
  `/home/zihan/repos/CamPoseOpensource`, which does not exist on this machine
  (`import robosuite` failed).
  - Initialized submodule `robosuite_source` at `0df3a5f` and reinstalled with
    `pip install -e robosuite_source --no-deps`.
  - ManiSkill was left untouched (baseline does not need it).
- Downloaded the three demo files into `policy_robosuite/demos/`:
  - `liftrand_eef_delta.hdf5` — 46,832,160 B, 300 demos
  - `canrand_eef_delta.hdf5` — 175,430,712 B, 500 demos
  - `squarerand_eef_delta.hdf5` — 203,905,400 B, 500 demos
  - All have `action_space: eef_delta` and obs keys including
    `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`.
  - Note: installed gdown 6.1.0 rejects `setup.sh`'s `--remaining-ok` flag, so
    the files were fetched by Drive file ID instead. -->

<!-- ## 2. Preflight

| Check | Result |
|---|---|
| `python tests/test_baseline_encoder.py` | `all baseline tests passed` |
| `python policy_robosuite/train.py --help` | OK |
| EGL offscreen render (LiftRand, 256²) | OK | -->

<!-- ## 3. Smoke tests

### Required smoke (prescribed command)

Lift, K=1, batch 4, 2 epochs, 1 eval episode × 30 steps:

- `train_loss 0.90681`, `val_loss 1.12309`
- epoch-0 success `False` (expected, untrained)
- artifacts in `policy_robosuite/checkpoints/baseline_smoke_seed_0/`:
  `config.json`, `dataset_stats.json`, `epoch_0.pth`, `epoch_1.pth`,
  `eval_epoch_{0,1}_agentview/success_by_seed.json`, MP4s.
- `dataset_stats.json` has `action_mean/std` (7), `eef_mean/std` (9),
  `state_mean/std` (7). -->

<!-- ### Additional K=2 smoke tests (not prescribed, done to cover 2-camera config)

- Can, 1 epoch: `train_loss 1.03667`, `val_loss 1.12185`.
- Square, 1 epoch: `train_loss 1.03321`, `val_loss 1.15408`.
- Checkpoints under `/data/zihan/campose_checkpoints/smoke_{can,square}_2cam`. -->

<!-- ### Resume test

Re-launched the can smoke with the same `--name`; it loaded the latest
checkpoint, resumed at the right epoch, and reused the same offline W&B run ID. -->

## 1. Code changes

### Gripper joint index dict (`baseline_common.py`)

This robosuite fork stores `robot._ref_gripper_joint_pos_indexes` as
`{'right': [7, 8]}` rather than a list. `find_gripper_qpos_indices()` now
accepts both forms and returns a flat `int64` array. Without this, the smoke
test crashed in `BaselineEvaluator.__init__` with
`TypeError: int() argument ... not 'dict'`.

### EEF quaternion: body vs site (`baseline_common.py`, `baseline_eval.py`, `baseline_data.py`)

The demo HDF5 `robot0_eef_quat` is sourced from the EEF **body**
(`sim.data.get_body_xquat('robot0_right_hand')`), not from the gripper **site**
(`site_xmat['gripper0_right_grip_site']`). The two orientations differ by a
fixed ~90°. `BaselineEvaluator._current_eef_raw()` used the site orientation,
so the eval-time 9-D EEF state and 44-D EEF history were far outside the
training distribution:

- before fix: normalized EEF state reached ±13 to ±18σ;
- after fix: max ≤ 2.25σ.

Changes:

- added `find_eef_body_name(env)` in `baseline_common.py`;
- `BaselineEvaluator._current_eef_raw()` now uses
  `sim.data.get_body_xquat(eef_body_name)` (MuJoCo wxyz order);
- the FK fallback `_eef_from_states()` in `baseline_data.py` uses the same body
  quaternion.

Verified at the same demo state: body quaternion matches the stored
`robot0_eef_quat` (after xyzw→wxyz conversion). Position and gripper values
already matched the site observations.

## 2. Full run

### Configuration

Serial queue in tmux session `campose_baseline`, GPU 1, runbook order
(lift/can/square × seeds 0,1,2). Each run:

```
--policy_class baseline --batch_size 32 --num_epochs 1001
--eval_start_epoch 600 --eval_every 50 --eval_max_steps 600
--lr 1e-4 --weight_decay 1e-6
```

with task-specific `--dataset_suffix`, `--num_side_cam`, `--n`, `--m`:

| Task | suffix | cameras | n | m |
|---|---|---|---|---|
| Lift | `liftrand_eef_delta` | 1 | 3 | 1 |
| Can | `canrand_eef_delta` | 2 | 4 | 2 |
| Square | `squarerand_eef_delta` | 2 | 4 | 2 |

Operational deviations:

- `--ckpt_dir /data/zihan/campose_checkpoints/<name>` — the root filesystem is
  97% full (62 GB free); each run keeps ~3 checkpoints of ~3.5 GB plus videos.
<!-- - `--save_every 50` — the runbook's 7.5 h resume check lives inside the
  checkpoint block; the default `save_every=1000` would checkpoint only at
  epoch 0 and 1000.
- W&B offline (`WANDB_MODE=offline`); runs support `wandb sync` later. -->

### 3. Result: `baseline_lift_rand_seed_0` (v1 — superseded by §6)

- Started 2026-11-05 00:43:32, finished 2026-11-05 05:46:06 (5h02m).
- Completed epoch 1000 (1001 epochs).
- `train_loss 0.00951`, `val_loss 0.02169`, final `lr 0.0`.
- **Success rate: 0.000 at every evaluation.**
  - epochs 0–600: 10 episodes each (13 evals), all 0.
  - epochs 650–1000: 50 episodes each (8 evals), all 0.
- Offline W&B: `wandb/offline-run-20261105_004334-bueba3uf`.
- Checkpoints: `/data/zihan/campose_checkpoints/baseline_lift_rand_seed_0`

<!-- ### `baseline_can_rand_seed_0` (stopped)

Started 05:46; was still in its epoch-0 evaluation when the queue was stopped.
Episodes 0–8 all `Success = False`; no training epochs had run.

The remaining seven runs were **not started**. The tmux session was killed and
GPU 1 freed (58 MiB remaining). -->

## 4. Diagnosis of the zero-success result

The zero is not an evaluation-harness artefact. Verified:

- **Reward semantics:** `LiftRand.reward()` returns 2.25 scaled by
  `reward_scale=1.0`; a known successful demo state yields
  `_check_success() == True` and `reward == 1.0`. The evaluator's `reward == 1`
  check is valid.
- **Action path:** replaying a successful demo's ground-truth actions through
  the wrapped eval environment reaches success (`reward 1.0`, cube lifted),
  with EEF positions matching the stored demo observations.
- **Camera poses:** JSON poses are MuJoCo convention; `_set_camera_pose` round
  trip error is 2.2e-8. The default agentview is 0.99° (nearest train pose) and
  2.44° (nearest test pose) away, i.e. in-distribution.
- **Eval EEF fix:** re-evaluating the epoch-1000 lift checkpoint with the fixed
  body-quaternion evaluator still gives **0/10** success.

### 5. Root cause: policy collapsed to a 9-D state-only controller

Sensitivity of sampled actions on the final checkpoint:

| Input zeroed | Mean absolute change in actions |
|---|---|
| Image (RGB + Plücker) | **0.0005** |
| 44-D EEF history | 0.0067 |
| 9-D EEF state | **0.349** |

Encoder feature diagnostics (8 validation samples):

| Stage | Across-sample std |
|---|---|
| Stem | 0.349 |
| layer1 | 0.500 |
| layer2 | 0.213 |
| layer3 | 0.097 |
| layer4 | 0.022 |
| `z_base` (avgpool) | 0.008 |
| `z_global` | 0.005 |

Also, `z_global` changes by only 0.003 when the image is replaced by zeros and
by 0.0045 when replaced by random noise. The image encoder trained (weights
differ from fresh init; FiLM weights moved off zero) but learned a nearly
sample-independent feature. The model fits validation actions well when
conditioned on the demo state (action MSE ~0.02–0.08 vs 0.144 zero-baseline),
but it has discarded the visual information needed to localize the cube.
A state-only controller cannot solve LiftRand, which explains the 0 success.

The collapse is a training/optimization issue in the baseline implementation,
not in the run harness, data loading, reward check, or camera convention.

## 6. v2 baseline run and results (2026-11-06/07)

The v2 baseline (`BASELINE_ARCH = "spatial_softmax_nobs1_proprio7_v1"`,
`OBS_FEATURE_DIM = 519`) replaces the v1 input conventions with the original
DP's and keeps spatial information in the encoder:

- DP `n_obs_steps=1`: one observation frame at `t`, action window `[t, t+16)`,
  8 executed actions;
- 7-D joint-qpos proprio zeroed before the DP min-max normalisation
  (`--prob_drop_proprio 1.0`, the default);
- no ImageNet RGB normalisation (`rgb/255` + raw Plücker);
- per-view spatial-softmax keypoints instead of global average pooling;
- 44-D camera-frame EEF history still feeds FiLM (past-only, z-scored).

### Run setup

| Setting | Value |
|---|---|
| Git | branch `method/baseline`, HEAD `a0b3e3d1694e1b60596b81ada259a5ff51f5b478`, clean |
| Env | `know_your_camera`; torch `2.11.0+cu128`, torchvision `0.26.0+cu128`; EGL |
| GPU | 2× RTX 5090, 32 607 MiB each; run pinned to GPU 1 (`CUDA_VISIBLE_DEVICES=1`) |
| Data | `liftrand_eef_delta.hdf5`, 300 demos (200 train + 10 validation) |
| tmux / log | `campose_lift_v2`; `/data/zihan/campose_logs/baseline_lift_rand_v2_seed_0.log` |
| Checkpoints | `/data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0` |
| W&B offline | `wandb/offline-run-20261106_202212-ku9idor5` (`ku9idor5`, not synced) |

Command (runbook command; conda activated inside the tmux command):

```
--policy_class baseline --name baseline_lift_rand_v2_seed_0
--dataset_suffix liftrand_eef_delta --num_side_cam 1 --n 3 --m 1
--batch_size 32 --seed 0 --num_epochs 10001
--eval_start_epoch 6000 --eval_every 500 --eval_max_steps 600
--save_every 500 --lr 1e-4 --weight_decay 1e-6
--ckpt_dir /data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0
```

### Smoke test

`baseline_lift_smoke_v2_seed_0`, 2 epochs, batch 4, 1 eval episode × 30 steps:

- reported `train_loss 0.91884`, `val_loss 1.12284`; no shape errors or NaNs;
- epoch-0 and epoch-1 eval both `Success = False` (expected, untrained);
- artifacts: `config.json`, `dataset_stats.json`, `epoch_{0,1}.pth`,
  `eval_epoch_{0,1}_agentview/` (videos + `success_by_seed.json`);
- `dataset_stats.json` keys: `action_mean/std` (7), `state_mean/std` (7),
  `eef_hist_mean/std` (44); **no** `eef_mean`/`eef_std`;
- checkpoint contains `"baseline_arch": "spatial_softmax_nobs1_proprio7_v1"`.

### Full run

- Started 2026-11-06 20:22, ended 2026-11-07 04:07 (7 h 45 m wall clock).
- 10000/10001 epochs (tqdm `10000/10001`). The 7.5 h guard fires inside the
  checkpoint block, so it triggered at the epoch-10000 save before training
  that final epoch. Relaunching would be a no-op (`ckpt epoch + 1 = 10001 =
  --num_epochs`), so the run is effectively complete.
- Final W&B summary: `train_loss 0.00455`, `val_loss 0.00214`,
  `success_rate_agentview 0.36`.
- Final checkpoint: `epoch_10000.pth` (3.1 GB, `loss 0.0045`, arch marker
  correct); `epoch_9000/9500/10000` retained. Batch 32 fit without OOM
  (~8.3 GB of 32 GB observed).
- Evaluation every 500 epochs, 600 steps per episode: 10 episodes through
  epoch 6000, 50 from epoch 6500 on. Only the first 10 episodes of each
  evaluation are saved as videos (`eval_save_n_video=10`).

### Results

| Epoch | Episodes | Successes | Rate |
|---:|---:|---:|---:|
| 0 | 10 | 0 | 0.00 |
| 500 | 10 | 0 | 0.00 |
| 1000 | 10 | 0 | 0.00 |
| 1500 | 10 | 0 | 0.00 |
| 2000 | 10 | 1 | 0.10 |
| 2500 | 10 | 2 | 0.20 |
| 3000 | 10 | 3 | 0.30 |
| 3500 | 10 | 4 | 0.40 |
| 4000 | 10 | 2 | 0.20 |
| 4500 | 10 | 0 | 0.00 |
| 5000 | 10 | 1 | 0.10 |
| 5500 | 10 | 1 | 0.10 |
| 6000 | 10 | 0 | 0.00 |
| 6500 | 50 | 7 | 0.14 |
| 7000 | 50 | 12 | 0.24 |
| 7500 | 50 | 12 | 0.24 |
| 8000 | 50 | 19 | 0.38 |
| 8500 | 50 | 18 | 0.36 |
| 9000 | 50 | 21 | 0.42 |
| 9500 | 50 | 25 | 0.50 |
| 10000 | 50 | 18 | 0.36 |

Across all 21 evaluations: **146/530 = 0.275**. Best: **25/50 = 0.50 at
epoch 9500**; final: **18/50 = 0.36 at epoch 10000**. The non-monotonic
stretch at epochs 4000–6000 is within the noise of 10-episode evaluations.

### Encoder-sensitivity check (optional, final checkpoint)

Same seeded diffusion noise, batch of 4, image input zeroed:

| Input zeroed | Mean abs action change | Max abs change |
|---|---:|---:|
| Image (RGB + Plücker), v2 | **0.1079** | 2.0 |
| Image (RGB + Plücker), v1 (reference §5) | 0.0005 | — |

The v2 checkpoint is ~200× more sensitive to the image than the v1 checkpoint
(reference action std 0.3745), so the encoder is using vision.

### Vision sensitivity at 1000 epochs

The original v2 run kept only the last 3 checkpoints, so its epoch-1000 weights
were gone. A same-config/seed rerun (`--name
baseline_lift_rand_v2_repro1000_seed_0`, same LR schedule via `--num_epochs
10001`) was stopped immediately after `epoch_1000.pth` was written
(`/data/zihan/campose_checkpoints/baseline_lift_rand_v2_epoch1000_seed_0/`).
Its epoch-1000 evaluation reproduced the original result (0/10; identical
`success_by_seed.json`); video bytes differ, so the rerun is not bit-identical.

Real validation-demo protocol (the same two demos and transform as the v1
check in §5):

| Checkpoint | Zero image | Zero 44-D EEF history | Zero state slot |
|---|---:|---:|---:|
| v1, epoch 1000 | 0.0005 | 0.0067 | 0.349 (9-D EEF state) |
| v2, epoch 1000 (repro) | **0.0577** | 0.0275 | 0.0168 (7-D proprio) |
| v2, epoch 10000 | **0.1167** | 0.0482 | 0.0327 (7-D proprio) |

The v2 "state slot" is the constant `-state_mean/state_std` for every sample,
because `prob_drop_proprio=1.0` zeroes the raw qpos **before** normalisation.
Zeroing that slot sets the constant to 0; it is an out-of-distribution control,
not evidence that v2 uses state — varying the raw qpos has exactly zero effect
by construction. The v1 9-D EEF state, by contrast, is a real per-sample input.

Synthetic protocol (the same batch/noise as the final-checkpoint check above):

| Checkpoint | Mean abs change | Max abs change |
|---|---:|---:|
| v2, epoch 1000 | 0.0626 | 1.97 |
| v2, epoch 10000 | 0.1079 | 2.00 |

At 1000 epochs v2 is already ~115× more image-sensitive than v1, and its image
sensitivity roughly doubles by epoch 10000. The v2 encoder uses vision well
before the final checkpoint.

### v1 no-state ablation (1000 epochs)

v1 concatenates a real per-sample 9-D EEF state directly into the 521-D
condition; §5 showed this dominates (zeroing it changes actions by 0.349, the
image by 0.0005). To test whether that shortcut caused the collapse, v1 was
retrained from the fixed v1 snapshot (`92b9c10`) with the raw 9-D EEF state
zeroed **before** normalisation in both the dataset and the evaluator — the v1
analog of `prob_drop_proprio=1.0`. The 521-D architecture is unchanged; the 9
state dims are the constant `-eef_mean/eef_std` (per-sample std exactly 0).

Setup: `v1_nostate_lift_rand_seed_0`, 1001 epochs, batch 32, seed 0, lr 1e-4,
wd 1e-6, evals at epochs 0/500/1000 with 10 episodes (the v1 baseline evaluated
every 50 epochs, so this sparse-eval run is not RNG-identical). Stopped after
the epoch-1000 save; checkpoint at
`/data/zihan/campose_checkpoints/v1_nostate_epoch1000_seed_0/`.

Success:

| Epoch | Episodes | Successes |
|---:|---:|---:|
| 0 | 10 | 0 |
| 500 | 10 | 1 |
| 1000 | 10 | 4 |

(v1 baseline: 0/530 at every evaluation.)

Sensitivity at epoch 1000, same real-demo protocol:

| Checkpoint | Zero image | Zero 44-D history | Zero state slot |
|---|---:|---:|---:|
| v1 baseline, epoch 1000 | 0.0005 | 0.0067 | 0.349 |
| v1 no-state, epoch 1000 | **0.0035** | **0.1673** | 0.0160 |
| v2, epoch 1000 | 0.0577 | 0.0275 | 0.0168 |
| v2, epoch 10000 | 0.1167 | 0.0482 | 0.0327 |

Removing the 9-D state raises image sensitivity only ~7× (0.0005 → 0.0035) and
leaves it ~17× below v2 at the same epoch; the model's dominant dependence
moves to the 44-D EEF history (~25×, 0.0067 → 0.1673). The no-state v1 reaches
4/10 at epoch 1000, but the sensitivity profile points to the proprioceptive
history path rather than vision; an image-ablation evaluation would be needed
to confirm that. Removing the direct state shortcut alone is not enough — v2's
spatial-softmax keypoints instead of global avgpool are what make the image the
dominant input.

### Evaluation videos

Copied to `eval_videos/baseline_lift_rand_v2_seed_0/`: 21 epoch folders,
210 MP4s, 21 `success_by_seed.json` (39 MB). See that folder's `README.md` for
the per-epoch table. The checkpoints themselves stay on `/data`.

<!-- ## 7. Current state / artefacts

- Working tree (`git status --short`):
  - `M BASELINE_REMOTE_TRAINING.md` (pre-existing)
  - `M policy_robosuite/baseline_common.py`
  - `M policy_robosuite/baseline_data.py`
  - `M policy_robosuite/baseline_eval.py`
  - plus this new `PROGRESS.md`
- No commits.
- Checkpoints/logs:
  - `/data/zihan/campose_checkpoints/baseline_lift_rand_seed_0` (9.8 GB)
  - `/data/zihan/campose_checkpoints/baseline_can_rand_seed_0` (1.1 GB)
  - `/data/zihan/campose_checkpoints/smoke_{can,square}_2cam` (1.1 GB each)
  - `policy_robosuite/checkpoints/baseline_smoke_seed_0` (4.4 GB, root disk)
  - logs: `/data/zihan/campose_logs/<name>.log`, `queue.log`
  - queue script: `/data/zihan/campose_logs/run_baseline_queue.sh`
- tmux `campose_baseline` is stopped; no `train.py` processes remain. -->

<!-- ### Environment warnings (non-blocking)

- mujoco `3.1.6` is below robosuite's declared `>=3.3.0` and mink's `>=3.10`,
  but imports and all smokes/full run worked.
- `robosuite_models` and mink-based whole-body IK are missing; irrelevant for
  the Panda tasks.
- `⚠️ Render fault detected; rebuilding context` appears after evaluations; the
  existing `env.reset()` fallback handles it without failing the run. -->

<!-- ## 8. Suggested next steps

1. Investigate the image-conditioning collapse before running more seeds:
   encoder gradients/feature variance during training, whether the UNet's
   image-conditioning columns receive useful gradients, state dominance in the
   objective, pretrained ResNet, or an auxiliary visual objective.
2. Keep the EEF body-vs-site fix regardless; it is required for correct eval.
3. After a method fix, delete/retrain `baseline_lift_rand_seed_0` (its weights
   encode the collapsed state-only policy). Can/Square were not trained.
4. To resume later:
   - restart the queue with
     `tmux new-session -d -s campose_baseline "bash /data/zihan/campose_logs/run_baseline_queue.sh"`;
   - existing runs resume from their latest checkpoint by `--name`;
   - sync offline W&B with `wandb sync <offline-run-dir>`. -->
