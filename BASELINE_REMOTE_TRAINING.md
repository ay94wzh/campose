# Remote runbook: `baseline` training

Audience: the agent operating the GPU server. This machine is the execution node; do not
develop here. The method is `baseline`: exact B1 variant A (no residual token branch) with a
diffusion-policy UNet, evaluated on the CamPose robosuite `*Rand` tasks.

Source branch: `method/baseline` in the CamPoseOpensource repo.
New code: `policy_robosuite/models/baseline.py`, `baseline_common.py`, `baseline_data.py`,
`baseline_eval.py`, `tests/test_baseline_encoder.py`, plus dispatch hooks in
`policy_robosuite/train.py`.

## 0. Before you start: code delivery

The branch may not be pushed yet. The human operator must either:

```bash
# on the laptop
cd /home/zihan-wang/CamPoseOpensource
git add .gitmodules policy_robosuite/baseline_common.py policy_robosuite/baseline_data.py \
        policy_robosuite/baseline_eval.py policy_robosuite/models/baseline.py \
        policy_robosuite/train.py tests/test_baseline_encoder.py
git commit -m "Add baseline (B1 + DP) method"
git push -u origin method/baseline
```

or sync the checkout to the server (for example with `rsync`), keeping the same relative paths.

Then on the server:

```bash
cd <REPO_ROOT>
git fetch origin
git checkout method/baseline
git pull --ff-only origin method/baseline
```

If the server checkout has local changes, stop and ask before stashing or resetting.

## 1. Environment

Use the CamPose environment (`know_your_camera` if it exists). The baseline needs
robosuite only; ManiSkill is optional.

```bash
conda activate know_your_camera
```

If the environment does not exist on this server, build it from `setup.sh` on a fresh
checkout. Do **not** run the full `setup.sh` on an existing environment: its
`conda create -n know_your_camera` fails when the env already exists and `set -e` aborts
before the submodules/demos are handled.

Required checks:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "import robosuite, mujoco, diffusers, h5py; print(robosuite.__version__, mujoco.__version__, diffusers.__version__)"
nvidia-smi
```

`torch.cuda.is_available()` must be `True`, and `nvidia-smi` must show the GPU. The
baseline is CUDA-only (the dataset, Plücker embedder and evaluator hardcode `.cuda()`).

### Submodules

`.gitmodules` uses HTTPS in the current branch. Initialize robosuite (and ManiSkill only
if needed):

```bash
git submodule sync --recursive
git submodule update --init robosuite_source
pip install -e robosuite_source
```

If the submodule URL is still SSH and fails with `Permission denied (publickey)`, rewrite it
to HTTPS first:

```bash
git config -f .gitmodules submodule.robosuite_source.url https://github.com/ripl/robosuite.git
git config -f .gitmodules submodule.maniskill_source.url https://github.com/ripl/ManiSkill.git
git submodule sync --recursive
git submodule update --init robosuite_source
pip install -e robosuite_source
```

### Data

The baseline uses the existing CamPose demo files under `policy_robosuite/demos/`:

```
liftrand_eef_delta.hdf5
canrand_eef_delta.hdf5
squarerand_eef_delta.hdf5
```

Do **not** re-download if they are present. If missing, run only the download part of
`setup.sh` (the `gdown` block) or copy the files from the laptop.

Verify the dataset structure:

```bash
python - <<'PY'
import h5py
for path in [
    "policy_robosuite/demos/liftrand_eef_delta.hdf5",
    "policy_robosuite/demos/canrand_eef_delta.hdf5",
    "policy_robosuite/demos/squarerand_eef_delta.hdf5",
]:
    with h5py.File(path, "r") as f:
        demos = [k for k in f["data"] if k.startswith("demo_")]
        print(path)
        print("  env_args:", str(f["data"].attrs["env_args"])[:160])
        print("  action_space:", f["data"].attrs["action_space"])
        print("  demos:", len(demos))
        print("  obs keys:", sorted(f["data/demo_0/obs"].keys()))
PY
```

Expected: `action_space` is `eef_delta`, and `obs` contains `robot0_eef_pos`,
`robot0_eef_quat`, `robot0_gripper_qpos`. The code falls back to forward kinematics if those
keys are absent, but that is much slower; report it if you see it.

Important: `robot0_eef_quat` is stored xyzw by robosuite. The code converts it to wxyz; do not
"fix" that conversion.

## 2. Preflight

Run from the repo root:

```bash
# CPU unit tests for the baseline encoder/geometry/policy shapes
python tests/test_baseline_encoder.py

# Import check for the training entrypoint (needs robosuite installed)
python policy_robosuite/train.py --help | head -20
```

The CPU tests must print `all baseline tests passed`. If they fail, stop and report the full
traceback.

Set headless rendering if the node has no display:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If EGL is unavailable, try `MUJOCO_GL=osmesa` and report which one worked.

W&B: `train.py` calls `wandb.init` unconditionally. Either log in beforehand or force
offline mode:

```bash
export WANDB_MODE=offline
```

## 3. Smoke test (required before any long run)

This exercises the dataset rendering, both observation frames, EEF history, the evaluator's
agentview Plücker, 8-action chunks, checkpointing and the eval loop. It is short but not a
quality run; an untrained success rate of 0 is expected.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline \
python policy_robosuite/train.py \
  --policy_class baseline \
  --name baseline_smoke_seed_0 \
  --dataset_suffix liftrand_eef_delta \
  --num_side_cam 1 --n 3 --m 1 \
  --batch_size 4 --seed 0 \
  --num_epochs 2 --eval_every 1 --eval_episodes 1 --eval_max_steps 30 \
  --save_every 1 \
  --lr 1e-4 --weight_decay 1e-6
```

What to verify:

- No shape errors or NaNs in the first two epochs.
- `Episode 0: Success = ...` appears (the eval at epoch 0 uses one episode).
- `policy_robosuite/checkpoints/baseline_smoke_seed_0/` contains `config.json`,
  `dataset_stats.json`, and `epoch_0.pth` / `epoch_1.pth`.
- `eval_epoch_0_agentview/` contains the success JSON and a video.
- `dataset_stats.json` contains `action_mean`, `action_std`, `eef_mean`, `eef_std`
  (`eef_*` must have length 9).
- GPU memory is well below the card limit. If it is close, use batch 16 for the real run.

Do not launch the full run if the smoke test fails.

## 4. Full training

Use one command per task and seed. The paper uses seeds 0, 1, 2. Keep the `--name` ending in
`_seed_<n>`; `train.py` derives the W&B group from the last 7 characters.

### Lift (1 camera)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python policy_robosuite/train.py \
  --policy_class baseline \
  --name baseline_lift_rand_seed_0 \
  --dataset_suffix liftrand_eef_delta \
  --num_side_cam 1 --n 3 --m 1 \
  --batch_size 32 --seed 0 \
  --num_epochs 100001 --eval_start_epoch 60000 --eval_every 2000 --eval_max_steps 600 \
  --lr 1e-4 --weight_decay 1e-6
```

### Can (2 cameras)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python policy_robosuite/train.py \
  --policy_class baseline \
  --name baseline_can_rand_seed_0 \
  --dataset_suffix canrand_eef_delta \
  --num_side_cam 2 --n 4 --m 2 \
  --batch_size 32 --seed 0 \
  --num_epochs 100001 --eval_start_epoch 60000 --eval_every 2000 --eval_max_steps 600 \
  --lr 1e-4 --weight_decay 1e-6
```

### Square (2 cameras)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python policy_robosuite/train.py \
  --policy_class baseline \
  --name baseline_square_rand_seed_0 \
  --dataset_suffix squarerand_eef_delta \
  --num_side_cam 2 --n 4 --m 2 \
  --batch_size 32 --seed 0 \
  --num_epochs 100001 --eval_start_epoch 60000 --eval_every 2000 --eval_max_steps 600 \
  --lr 1e-4 --weight_decay 1e-6
```

For seeds 1 and 2, change `--seed` and the `_seed_<n>` suffix. Run tasks/seeds in parallel
only if each has enough GPU memory; otherwise serialize.

If batch 32 OOMs, drop to 16 and report the change. Do not change the learning rate, betas,
weight decay, horizon, or camera parameters without asking.

## 5. What is intentional

- `--policy_class baseline` is required. ACT/DP/SmolVLA are untouched.
- Training uses the CamPose n/m camera window sampling. Do **not** pass `--default_cam 1`.
- Evaluation is forced to fixed agentview, K=1, by `BaselineEvaluator`. It ignores
  `--pose_files` for the rollout and evaluates a single `agentview` curve. This is by design.
- The evaluator restores the default agentview pose before every episode because the training
  dataset mutates the shared MuJoCo camera fields. Do not remove that restore.
- The baseline always computes real Plücker maps, including for agentview. `--use_plucker` is
  ignored by this path.
- `--prob_drop_proprio`, `--chunk_size`, `--backbone`, `--hidden_dim`, `--dropout` and
  `--obs_dim` are not used by the baseline. The method uses its own fixed B1/DP settings.
- Action normalization is CamPose DP min-max (encoded as mean/std); the 9-D EEF state is
  z-scored. The stats live in `dataset_stats.json`.
- This is an exact-B1 **topology**, but at CamPose's 256×256 resolution, K=1/2, CamPose
  normalization, and without EMA. These are deliberate deviations for benchmark compatibility.

## 6. Monitoring

- W&B (or `wandb sync` later) metrics: `train_loss`, `val_loss`, and
  `success_rate_agentview`. The first eval at epoch 0 is untrained; quality is expected to
  appear much later.
- Checkpoints: `policy_robosuite/checkpoints/<name>/epoch_*.pth` (keep last 3), plus
  `config.json` and `dataset_stats.json`.
- Eval artifacts: `eval_epoch_<epoch>_agentview/success_by_seed.json` and videos for the
  first `--eval_save_n_video` episodes.
- Logs: the process stdout. Keep it in a file or under the scheduler's log directory.

`train.py` exits after 7.5 hours of wall clock and resumes from the latest checkpoint when
the same `--name` is relaunched. For a long run, either wrap it in a restart loop or raise
that limit; report which approach you used.

## 7. Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: robosuite` | `git submodule update --init robosuite_source` + `pip install -e robosuite_source` |
| `Permission denied (publickey)` on submodule | Rewrite `.gitmodules` URLs to HTTPS and `git submodule sync` |
| EGL/GLFW/OpenGL errors on a headless node | `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`; if unavailable, try `MUJOCO_GL=osmesa` and report |
| `CUDA out of memory` | batch 16 (then 8), keep everything else fixed; report the final batch |
| `KeyError: 'eef_mean'` | You are not running `--policy_class baseline`, or an old `dataset_stats.json` is being resumed. Use a fresh `--name`/ckpt dir |
| `KeyError: 'actions'` / shape error | Confirm the dataset suffix matches the env (`liftrand_eef_delta`, etc.) and action dim 7 |
| Camera-pose `IndexError` | Check `--n/--m/--num_side_cam` against the pose JSON length; do not exceed the available poses |
| HDF5 lacks `obs/robot0_eef_*` | The FK fallback runs but is slow; report it and prefer the standard demos |
| W&B hangs at startup | `WANDB_MODE=offline`, or log in on the server |
| Smoke test passes but full run is slow | Expected: 256² × 2 frames × K views plus per-step eval rendering; report throughput |

## 8. Report back

Include:

1. `git rev-parse HEAD` and `git status --short` on the server.
2. GPU model and memory, torch/CUDA versions, conda env name.
3. Smoke-test result: first `train_loss`, `val_loss`, the epoch-0 success line, and the
   checkpoint/artifact paths.
4. Any preflight warnings (missing HDF5 obs, EGL fallback, OOM, batch change).
5. For the full run: command used, W&B run ID/URL (or offline path), and the latest
   `success_rate_agentview`.

Do not commit generated data, checkpoints, W&B files, or demo HDF5s.
