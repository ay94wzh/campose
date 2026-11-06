# Baseline run progress — 2026-11-05 / 2026-11-06

Session log for executing `BASELINE_REMOTE_TRAINING.md` on this machine.
Branch `method/baseline`, HEAD `7af6bfa0b18d932366ad775ef6b756d9c7e807aa`.
No commits were made; all code changes are in the working tree.

## 1. Machine and environment

- GPU: 2× NVIDIA GeForce RTX 5090, 32.6 GB each.
  - GPU 0 was occupied by another user's job (~22.6 GB); it was not used.
  - GPU 1 was used (contained a ~4.7 GB job from another user).
- Conda env: `know_your_camera` (Python 3.10.20)
  - torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, CUDA available (12.8)
  - diffusers `0.40.0`, transformers `5.16.1`, mujoco `3.1.6`, h5py `3.16.0`
  - robosuite `1.5.1`, wandb `0.29.0`, gdown `6.1.0`
- EGL works: `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (no display; osmesa not needed).

### Environment repairs

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
    the files were fetched by Drive file ID instead.

## 2. Preflight

| Check | Result |
|---|---|
| `python tests/test_baseline_encoder.py` | `all baseline tests passed` |
| `python policy_robosuite/train.py --help` | OK |
| EGL offscreen render (LiftRand, 256²) | OK |

## 3. Smoke tests

### Required smoke (prescribed command)

Lift, K=1, batch 4, 2 epochs, 1 eval episode × 30 steps:

- `train_loss 0.90681`, `val_loss 1.12309`
- epoch-0 success `False` (expected, untrained)
- artifacts in `policy_robosuite/checkpoints/baseline_smoke_seed_0/`:
  `config.json`, `dataset_stats.json`, `epoch_0.pth`, `epoch_1.pth`,
  `eval_epoch_{0,1}_agentview/success_by_seed.json`, MP4s.
- `dataset_stats.json` has `action_mean/std` (7), `eef_mean/std` (9),
  `state_mean/std` (7).

### Additional K=2 smoke tests (not prescribed, done to cover 2-camera config)

- Can, 1 epoch: `train_loss 1.03667`, `val_loss 1.12185`.
- Square, 1 epoch: `train_loss 1.03321`, `val_loss 1.15408`.
- Checkpoints under `/data/zihan/campose_checkpoints/smoke_{can,square}_2cam`.

### Resume test

Re-launched the can smoke with the same `--name`; it loaded the latest
checkpoint, resumed at the right epoch, and reused the same offline W&B run ID.

## 4. Uncommitted code changes

### 4.1 Gripper joint index dict (`baseline_common.py`)

This robosuite fork stores `robot._ref_gripper_joint_pos_indexes` as
`{'right': [7, 8]}` rather than a list. `find_gripper_qpos_indices()` now
accepts both forms and returns a flat `int64` array. Without this, the smoke
test crashed in `BaselineEvaluator.__init__` with
`TypeError: int() argument ... not 'dict'`.

### 4.2 EEF quaternion: body vs site (`baseline_common.py`, `baseline_eval.py`, `baseline_data.py`)

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

## 5. Full run

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
- `--save_every 50` — the runbook's 7.5 h resume check lives inside the
  checkpoint block; the default `save_every=1000` would checkpoint only at
  epoch 0 and 1000.
- W&B offline (`WANDB_MODE=offline`); runs support `wandb sync` later.

### Result: `baseline_lift_rand_seed_0`

- Started 2026-11-05 00:43:32, finished 2026-11-05 05:46:06 (5h02m).
- Completed epoch 1000 (1001 epochs).
- `train_loss 0.00951`, `val_loss 0.02169`, final `lr 0.0`.
- **Success rate: 0.000 at every evaluation.**
  - epochs 0–600: 10 episodes each (13 evals), all 0.
  - epochs 650–1000: 50 episodes each (8 evals), all 0.
- Offline W&B: `wandb/offline-run-20261105_004334-bueba3uf`.
- Checkpoints: `/data/zihan/campose_checkpoints/baseline_lift_rand_seed_0`
  (9.8 GB, keeps the last 3 checkpoints plus eval videos/JSONs).

### `baseline_can_rand_seed_0` (stopped)

Started 05:46; was still in its epoch-0 evaluation when the queue was stopped.
Episodes 0–8 all `Success = False`; no training epochs had run.

The remaining seven runs were **not started**. The tmux session was killed and
GPU 1 freed (58 MiB remaining).

## 6. Diagnosis of the zero-success result

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

### Root cause: policy collapsed to a 9-D state-only controller

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

## 7. Current state / artefacts

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
- tmux `campose_baseline` is stopped; no `train.py` processes remain.

### Environment warnings (non-blocking)

- mujoco `3.1.6` is below robosuite's declared `>=3.3.0` and mink's `>=3.10`,
  but imports and all smokes/full run worked.
- `robosuite_models` and mink-based whole-body IK are missing; irrelevant for
  the Panda tasks.
- `⚠️ Render fault detected; rebuilding context` appears after evaluations; the
  existing `env.reset()` fallback handles it without failing the run.

## 8. Suggested next steps

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
   - sync offline W&B with `wandb sync <offline-run-dir>`.
