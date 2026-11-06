# Remote runbook: train Lift with the `baseline` policy (v2 conventions)

Audience: the agent operating **miroc-server**. This machine only executes; do not develop here.
This run trains one Lift seed with the corrected baseline:

- DP `n_obs_steps=1` (one observation frame at `t`, action window `[t, t+16)`, 8 executed actions)
- 7-D joint-qpos proprio, zeroed before the DP min-max normalisation via
  `--prob_drop_proprio` (default `1.0` = dropped, matching the original DP)
- no ImageNet RGB normalisation (`rgb/255` + raw Plücker, matching the original DP)
- per-view spatial-softmax keypoints instead of global average pooling
- 44-D camera-frame EEF history still feeds FiLM (past-only, z-scored)

`OBS_FEATURE_DIM = 519`; the checkpoint marker is
`baseline_arch = "spatial_softmax_nobs1_proprio7_v1"`. **Old `baseline_*` checkpoints are
incompatible** and are rejected on resume; always use a fresh `--name`.

## 0. Code delivery

The v2 changes live in the working tree on the laptop (branch `method/baseline`). The human
operator must commit and push (or rsync) before you start. Files to sync:

```
REMOTE_LIFT_TRAINING.md
policy_robosuite/baseline_common.py
policy_robosuite/baseline_data.py
policy_robosuite/baseline_eval.py
policy_robosuite/models/baseline.py
policy_robosuite/train.py
tests/test_baseline_encoder.py
```

On the server:

```bash
cd <REPO_ROOT>
git fetch origin
git checkout method/baseline
git pull --ff-only origin method/baseline
git rev-parse HEAD
git status --short          # must be clean (generated files ignored)
```

If the checkout has local changes, stop and ask before stashing or resetting. Do not commit
checkpoints, W&B files or demo HDF5s.

## 1. Environment

```bash
conda activate know_your_camera
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=offline
```

Required checks:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "import robosuite, mujoco, diffusers, h5py; print(robosuite.__version__, mujoco.__version__, diffusers.__version__)"
nvidia-smi
```

`torch.cuda.is_available()` must be `True`. If `robosuite` is missing:
`git submodule update --init robosuite_source && pip install -e robosuite_source --no-deps`.

## 2. Data

Use the existing Lift demos (do not re-download if present):

```
policy_robosuite/demos/liftrand_eef_delta.hdf5
```

Check it:

```bash
python - <<'PY'
import h5py
path = "policy_robosuite/demos/liftrand_eef_delta.hdf5"
with h5py.File(path, "r") as f:
    demos = [k for k in f["data"] if k.startswith("demo_")]
    print("demos:", len(demos))
    print("action_space:", f["data"].attrs["action_space"])
    print("obs keys:", sorted(f["data/demo_0/obs"].keys()))
PY
```

Expect `action_space: eef_delta` and `obs` containing `robot0_eef_pos`, `robot0_eef_quat`,
`robot0_gripper_qpos`.

## 3. Preflight

```bash
python tests/test_baseline_encoder.py
python policy_robosuite/train.py --help | head -5
```

The tests must print `all baseline tests passed` (the full CPU suite: encoder shapes, past-only
history, spatial softmax, proprio dropout order, no-RGB-normalisation, policy shapes). If anything
fails, stop and report the full traceback.

## 4. Smoke test (required before the long run)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline \
python policy_robosuite/train.py \
  --policy_class baseline \
  --name baseline_lift_smoke_v2_seed_0 \
  --dataset_suffix liftrand_eef_delta \
  --num_side_cam 1 --n 3 --m 1 \
  --batch_size 4 --seed 0 \
  --num_epochs 2 --eval_every 1 --eval_episodes 1 --eval_max_steps 30 \
  --save_every 1 \
  --ckpt_dir /data/zihan/campose_checkpoints/baseline_lift_smoke_v2_seed_0 \
  --lr 1e-4 --weight_decay 1e-6
```

Verify:

- No shape errors or NaNs; `Episode 0: Success = ...` appears.
- `/data/zihan/campose_checkpoints/baseline_lift_smoke_v2_seed_0/` contains `config.json`,
  `dataset_stats.json`, `epoch_0.pth`, `epoch_1.pth`, and an `eval_epoch_0_agentview/` video.
- `dataset_stats.json` contains `action_mean/std` (7), `state_mean/std` (7) and
  `eef_hist_mean/std` (44). There is **no** `eef_mean`/`eef_std` key.
- The saved checkpoint contains `"baseline_arch": "spatial_softmax_nobs1_proprio7_v1"`.

Do not launch the full run if the smoke test fails.

## 5. Full Lift run (one seed)

Run it in `tmux` so it survives disconnects. Pick a fresh checkpoints root with enough free
disk; the previous runs used `/data/zihan/campose_checkpoints`.

```bash
mkdir -p /data/zihan/campose_checkpoints /data/zihan/campose_logs
tmux new-session -d -s campose_lift_v2 \
  "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline \
   python policy_robosuite/train.py \
     --policy_class baseline \
     --name baseline_lift_rand_v2_seed_0 \
     --dataset_suffix liftrand_eef_delta \
     --num_side_cam 1 --n 3 --m 1 \
     --batch_size 32 --seed 0 \
     --num_epochs 10001 --eval_start_epoch 6000 --eval_every 500 --eval_max_steps 600 \
     --save_every 500 \
     --ckpt_dir /data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0 \
     --lr 1e-4 --weight_decay 1e-6 \
     2>&1 | tee /data/zihan/campose_logs/baseline_lift_rand_v2_seed_0.log"
```

Notes:

- `--prob_drop_proprio` is intentionally **not** passed; the default `1.0` matches the original
  DP (the 7-D joint qpos becomes the constant `-state_mean/state_std`). Do not set it to `0`
  without asking — that would enable the proprio shortcut.
- Do not pass `--default_cam 1` or `--use_plucker`; the baseline always uses real Plücker maps.
- If `--batch_size 32` OOMs, drop to 16 and report the change. Do not change the LR, betas,
  weight decay, horizon, or camera parameters.
- The run exits after 7.5 h of wall clock. Relaunch the same command to resume from the latest
  checkpoint (the `baseline_arch` guard allows it); keep the same `--name` and `--ckpt_dir`.
- Expected duration on a 5090-class GPU: ~5 h for 1001 epochs at batch 32 (previous run).

For seeds 1 and 2, change `--seed` and the `_seed_<n>` suffix, and use a separate `--name` /
`--ckpt_dir`. Run in parallel only if each fits in GPU memory.

## 6. What to monitor

```bash
tail -f /data/zihan/campose_logs/baseline_lift_rand_v2_seed_0.log
nvidia-smi
```

Key signals:

- `train_loss` / `val_loss` finite and decreasing; no NaN.
- Success rate is expected to be 0 at epoch 0 (untrained). Quality, if any, appears later;
  `--eval_start_epoch 600` only controls 10 vs 50 episodes, not whether eval runs.
- The first eval at epoch 0, then every 50 epochs. Each eval writes
  `eval_epoch_<epoch>_agentview/success_by_seed.json` and videos for the first
  `--eval_save_n_video` episodes.
- Checkpoints: `epoch_*.pth` (last 3 kept), plus `config.json` and `dataset_stats.json`.

A useful sanity check after training (optional, on a GPU node): load the final checkpoint and
zero the image input; the sampled actions should change much more than they did in the old
collapsed run (~0.0005). If they still do not, report it — the encoder is not using vision.

## 7. Troubleshooting

| Symptom | Action |
|---|---|
| `RuntimeError: ... baseline_arch ...` | Checkpoint is from an older baseline version. Use a fresh `--name` and `--ckpt_dir`. |
| `KeyError: 'eef_hist_mean'` / `'state_mean'` | Wrong policy class or stale `dataset_stats.json`; use `--policy_class baseline` and a fresh `--name`. |
| `ModuleNotFoundError: robosuite` | `git submodule update --init robosuite_source && pip install -e robosuite_source --no-deps` |
| EGL/GLFW/OpenGL error | Keep `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`; if unavailable try `MUJOCO_GL=osmesa` and report. |
| `CUDA out of memory` | batch 16 (then 8), keep everything else fixed; report the final batch. |
| Camera-pose `IndexError` | Check `--n/--m/--num_side_cam` against the pose JSON length. |
| W&B hangs | `WANDB_MODE=offline`; sync later with `wandb sync <offline-run-dir>`. |

## 8. Report back

Include:

1. `git rev-parse HEAD` and `git status --short`.
2. GPU model/memory, torch/CUDA versions, conda env.
3. Smoke result: first `train_loss`, `val_loss`, the epoch-0 success line, checkpoint/artifact
   paths, and the `dataset_stats.json` keys.
4. The exact full-run command, log path, W&B offline run ID (or sync URL), and latest
   `success_rate_agentview`.
5. Any deviation (batch size, EGL fallback, missing HDF5 obs, OOM).
