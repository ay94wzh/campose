# Plan: Lift v1 (avgpool) training + origin-consistent evaluation

This is the single plan for the Lift baseline work. It replaces the earlier separate
training and evaluation plan documents, which have been removed to keep the project
clean.

## Status and context

**Code is already implemented** on branch `method/baseline`:

- `BaselineEncoder(pool="avgpool")`: global average pooling (the v1 path) instead of
  spatial-softmax keypoints; everything else unchanged.
- `train.py --baseline_pool {spatial_softmax,avgpool}` with a matching checkpoint marker
  (`spatial_softmax_nobs1_proprio7_v1` / `avgpool_nobs1_proprio7_v1`) and resume guard.
- `BaselineEvaluator` now selects a **per-episode camera pose** from the pose files,
  matching the origin `Evaluator` (`num_side_cam=1`): episode `i` uses pose `i`.
- Baseline eval now defaults to `args.pose_files`
  (`train_cameras.json`, `test_cameras.json`); `--baseline_eval_agentview 1` restores the
  old fixed-agentview protocol.
- `policy_robosuite/eval_baseline_checkpoint.py` re-evaluates existing checkpoints on the
  pose sets.
- CPU tests pass, including `ok eval pose selection`.

The completed v2 baseline (`baseline_lift_rand_v2_seed_0`,
`spatial_softmax_nobs1_proprio7_v1`) reached **146/530 = 27.5%** under the old
agentview-only protocol (best 25/50 at epoch 9500; final 18/50 at epoch 10000). That
number is **not comparable** to the new pose protocol; v2 must be re-evaluated (Part B).

The new v1 run keeps every v2 convention and changes only pooling:

| Setting | v2 (control) | v1 (this run) |
|---|---|---|
| per-view latent | spatial-softmax 32 keypoints → Linear → 512 | ResNet18 layer4 global avgpool → 512 |
| `OBS_FEATURE_DIM` | 519 | 519 |
| `n_obs_steps` | 1 (obs at `t`, actions `[t, t+16)`, execute 8) | identical |
| proprio | 7-D joint qpos, zeroed before norm (`prob_drop_proprio=1.0`) | identical |
| image input | `rgb/255` + raw Plücker, no ImageNet normalisation | identical |
| FiLM | 4-step (44-D) z-scored camera-frame EEF history | identical |
| data / seed / batch / LR / epochs | LiftRand, seed 0, 32, 1e-4, 10001 | identical |

## Part A — train the new v1

### A0. Code delivery

The human operator commits and pushes (or rsyncs) the current working tree:

```
policy_robosuite/baseline_common.py
policy_robosuite/models/baseline.py
policy_robosuite/baseline_eval.py
policy_robosuite/train.py
policy_robosuite/eval_baseline_checkpoint.py
tests/test_baseline_encoder.py
PLAN.md
```

On `miroc-server`:

```bash
cd <REPO_ROOT>
git fetch origin
git checkout method/baseline
git pull --ff-only origin method/baseline
git rev-parse HEAD
git status --short          # must be clean (generated files ignored)
```

Stop and ask if the checkout is dirty. Verify the new code is present:

```bash
python policy_robosuite/train.py --help | grep -A2 baseline_pool
python policy_robosuite/train.py --help | grep -A1 baseline_eval_agentview
python policy_robosuite/eval_baseline_checkpoint.py --help | head -5
```

### A1. Environment and data

```bash
conda activate know_your_camera
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=offline

python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "import robosuite, mujoco, diffusers, h5py; print(robosuite.__version__, mujoco.__version__, diffusers.__version__)"
nvidia-smi
```

`torch.cuda.is_available()` must be `True`. Check the dataset and pose files:

```bash
python - <<'PY'
import h5py, json
path = "policy_robosuite/demos/liftrand_eef_delta.hdf5"
with h5py.File(path, "r") as f:
    print("demos:", len([k for k in f["data"] if k.startswith("demo_")]))
    print("action_space:", f["data"].attrs["action_space"])
for name in ("train_cameras", "test_cameras"):
    raw = json.load(open(f"policy_robosuite/camera_poses/{name}.json"))
    print(name, len(raw["poses"]), "poses")
PY
```

Expect `eef_delta`, 300 demos, `train_cameras 1000 poses`, `test_cameras 100 poses`.

### A2. Preflight

```bash
python tests/test_baseline_encoder.py
```

Must end with `all baseline tests passed`; the new lines include
`ok encoder shapes pool=avgpool ...`, `ok pooling variants`, `ok arch markers`, and
`ok eval pose selection`.

Confirm the refactor still loads the v2 checkpoint (read-only):

```bash
python - <<'PY'
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0, "policy_robosuite")
from models.baseline import BaselineDiffusionPolicy

ckpt = torch.load(
    "/data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0/epoch_10000.pth",
    map_location="cpu", weights_only=False,
)
args = SimpleNamespace(action_dim=7, lr=1e-4, weight_decay=1e-6)
policy = BaselineDiffusionPolicy(args)
policy.load_state_dict(ckpt["model_state_dict"])
print("v2 checkpoint loads; marker:", ckpt["baseline_arch"])
PY
```

### A3. Smoke test

Use a fresh smoke directory.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline \
python policy_robosuite/train.py \
  --policy_class baseline \
  --name baseline_lift_smoke_avgpool_seed_0 \
  --dataset_suffix liftrand_eef_delta \
  --num_side_cam 1 --n 3 --m 1 \
  --batch_size 4 --seed 0 \
  --num_epochs 2 --eval_every 1 --eval_episodes 1 --eval_max_steps 30 \
  --save_every 1 \
  --baseline_pool avgpool \
  --ckpt_dir /data/zihan/campose_checkpoints/baseline_lift_smoke_avgpool_seed_0 \
  --lr 1e-4 --weight_decay 1e-6
```

Verify:

- No shape errors/NaNs; `Episode 0: Success = ...` appears for each pose set.
- Artifacts: `config.json`, `dataset_stats.json`, `epoch_0.pth`, `epoch_1.pth`,
  and **both** `eval_epoch_0_train_cameras/` and `eval_epoch_0_test_cameras/`
  (videos + `success_by_seed.json`).
- `config.json` has `"baseline_pool": "avgpool"` and
  `"baseline_eval_agentview": false`.
- `dataset_stats.json` has `action_mean/std` (7), `state_mean/std` (7),
  `eef_hist_mean/std` (44); no `eef_mean`/`eef_std`.
- `epoch_1.pth` has `"baseline_arch": "avgpool_nobs1_proprio7_v1"`.

Do not launch the full run if the smoke fails.

### A4. Full run (one seed)

Pick a free GPU from `nvidia-smi` and substitute `<GPU>`.

```bash
mkdir -p /data/zihan/campose_checkpoints /data/zihan/campose_logs
tmux new-session -d -s campose_lift_avgpool \
  "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<GPU> \
   python policy_robosuite/train.py \
     --policy_class baseline \
     --name baseline_lift_rand_avgpool_seed_0 \
     --dataset_suffix liftrand_eef_delta \
     --num_side_cam 1 --n 3 --m 1 \
     --batch_size 32 --seed 0 \
     --num_epochs 10001 --eval_start_epoch 6000 --eval_every 500 --eval_max_steps 600 \
     --save_every 500 \
     --baseline_pool avgpool \
     --ckpt_dir /data/zihan/campose_checkpoints/baseline_lift_rand_avgpool_seed_0 \
     --lr 1e-4 --weight_decay 1e-6 \
     2>&1 | tee /data/zihan/campose_logs/baseline_lift_rand_avgpool_seed_0.log"
```

Notes:

- Do **not** change LR, betas, weight decay, horizon, camera parameters, seed, batch
  size, or the eval schedule.
- `--prob_drop_proprio` stays at its default `1.0`; do not pass `0`.
- Do not pass `--default_cam 1` or `--use_plucker`; the baseline always uses real
  Plücker maps.
- Evaluation defaults to the origin pose files (`train_cameras` + `test_cameras`), so
  every eval runs on both pose sets and costs ~2× the old agentview eval. Expected
  duration: **~9–10 h**; the 7.5 h guard may stop it once.
- Resume with the **same command**; keep the same `--name`, `--ckpt_dir`, and
  `--baseline_pool avgpool`.
- If batch 32 OOMs, drop to 16 and report; keep everything else fixed.

### A5. Preserve the epoch-1000 checkpoint

`cleanup_ckpt` keeps only the last 3 checkpoints, so `epoch_1000.pth` is deleted after
the epoch-2500 save. Start a watcher:

```bash
KEEP_DIR=/data/zihan/campose_checkpoints/baseline_lift_rand_avgpool_seed_0_keep
RUN_DIR=/data/zihan/campose_checkpoints/baseline_lift_rand_avgpool_seed_0
mkdir -p "$KEEP_DIR"
tmux new-session -d -s campose_keep_1000 \
  "while [ ! -f $RUN_DIR/epoch_1000.pth ]; do sleep 60; done; \
   cp -l $RUN_DIR/epoch_1000.pth $KEEP_DIR/epoch_1000.pth && echo copied"
```

`cp -l` hardlinks; use plain `cp` if the filesystem rejects it. Do not name the copy
`epoch_*.pth` inside `RUN_DIR` — `cleanup_ckpt` would delete it.

### A6. Monitor

```bash
tail -f /data/zihan/campose_logs/baseline_lift_rand_avgpool_seed_0.log
nvidia-smi
```

Check `train_loss`/`val_loss` finite, no NaNs, and that each eval writes
`eval_epoch_<N>_train_cameras/` and `eval_epoch_<N>_test_cameras/`. Record the offline
W&B run id from the log.

## Part B — evaluate v1 and v2 with the origin protocol

### B1. Protocol

- Pose sets: `train_cameras` (poses 0–49; in-distribution, training windows cover
  poses 0–201) and `test_cameras` (poses 0–49; held-out views).
- Episode `i` uses pose `i` (`num_side_cam=1`), exactly as the origin `Evaluator`.
- 50 episodes × 600 steps per pose set (`10` episodes before `eval_start_epoch`).
- Object placement is seeded by `episode_num`; diffusion noise by
  `(episode_num+1)*100003+step`. Both models see the same initial states and noise, so
  the comparison is paired.
- The v1 training run produces these evals automatically every 500 epochs. Use the
  final/best checkpoints for the comparison.

**"Online" vs "offline" eval.** In this plan:

- **Online** = the evaluation the training loop runs automatically every 500 epochs
  against the model currently in memory. Results land in
  `<ckpt_dir>/eval_epoch_<N>_train_cameras/` and `..._test_cameras/`. This is what the
  new v1 run does during training.
- **Offline** = `eval_baseline_checkpoint.py` loading a saved `epoch_*.pth` from disk and
  running the same `BaselineEvaluator` without training. Used for the already-trained v2
  checkpoints, and for v1 checkpoints that were not evaluated online (e.g. the preserved
  epoch-1000, or a checkpoint from a run that kept only its last 3).

### B2. v1 (avgpool) evaluation

During the run, read:

```
<ckpt_dir>/eval_epoch_<N>_train_cameras/success_by_seed.json
<ckpt_dir>/eval_epoch_<N>_test_cameras/success_by_seed.json
```

For the kept epoch-1000 checkpoint (matched-schedule comparison):

```bash
python policy_robosuite/eval_baseline_checkpoint.py \
  --ckpt /data/zihan/campose_checkpoints/baseline_lift_rand_avgpool_seed_0_keep/epoch_1000.pth \
  --baseline_pool avgpool \
  --pose_files train_cameras.json test_cameras.json \
  --num_episodes 50 --eval_max_steps 600 --num_side_cam 1 \
  --eval_save_n_video 2 \
  --save_dir /data/zihan/campose_eval/avgpool_epoch1000_poses
```

Optional agentview reference for a checkpoint:

```bash
python policy_robosuite/eval_baseline_checkpoint.py \
  --ckpt <ckpt.pth> --baseline_pool avgpool --agentview \
  --num_episodes 50 --eval_max_steps 600 \
  --save_dir /data/zihan/campose_eval/avgpool_epoch<X>_agentview
```

### B3. Re-evaluate v2 (control)

The completed v2 run was agentview-only. Re-run its retained checkpoints with the same
pose protocol:

```bash
python policy_robosuite/eval_baseline_checkpoint.py \
  --ckpt /data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0/epoch_9500.pth \
  --baseline_pool spatial_softmax \
  --pose_files train_cameras.json test_cameras.json \
  --num_episodes 50 --eval_max_steps 600 --num_side_cam 1 \
  --eval_save_n_video 2 \
  --save_dir /data/zihan/campose_eval/v2_epoch9500_poses

python policy_robosuite/eval_baseline_checkpoint.py \
  --ckpt /data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0/epoch_10000.pth \
  --baseline_pool spatial_softmax \
  --pose_files train_cameras.json test_cameras.json \
  --num_episodes 50 --eval_max_steps 600 --num_side_cam 1 \
  --eval_save_n_video 2 \
  --save_dir /data/zihan/campose_eval/v2_epoch10000_poses
```

For the matched epoch-1000 comparison, also evaluate
`/data/zihan/campose_checkpoints/baseline_lift_rand_v2_epoch1000_seed_0/epoch_1000.pth`
with `--baseline_pool spatial_softmax`.

The runner writes `summary.json` per pose set (`episodes`, `successes`, `rate`) and
`success_by_seed.json`. Cost: each pose set is one ~50-episode eval (~10–20 min); four
checkpoints ≈ 1.5–2 h.

### B4. Analysis

| Model | Checkpoint | train_cameras (50) | test_cameras (50) | agentview (reference) |
|---|---|---:|---:|---:|
| v2 | epoch 9500 | | | 25/50 (old protocol) |
| v2 | epoch 10000 | | | 18/50 (old protocol) |
| v1 (avgpool) | epoch 9500 | | | |
| v1 (avgpool) | epoch 10000 | | | |

Interpretation:

- v2 keeps most of its success on `test_cameras`, v1 does not → spatial keypoints are
  load-bearing for view generalization; supports the PROPOSAL design constraint.
- v1 matches v2 on `test_cameras` → global pooling is sufficient for LiftRand; the
  earlier v1 collapse was due to the 9-D state shortcut and/or other input conventions,
  not pooling.
- Compare train vs test separately; a model can work on seen poses and fail on held-out
  views.
- 50-episode evals have ~7% standard error; treat a few successes as noise.

## Verification / acceptance

- `python tests/test_baseline_encoder.py` ends with `all baseline tests passed`.
- The default spatial path loads the v2 `epoch_10000.pth` (state-dict keys unchanged).
- Smoke produces both pose-eval directories and the `avgpool_nobs1_proprio7_v1` marker.
- Full-run log shows finite losses and both `eval_epoch_<N>_<pose_set>/` trees.
- v2 and v1 summary tables come from the same pose protocol and episode seeds.

## Troubleshooting

| Symptom | Action |
|---|---|
| `unrecognized arguments: --baseline_pool` / `--baseline_eval_agentview` | Code not delivered; pull/rsync per A0. |
| `RuntimeError: ... baseline_arch ...` | Wrong `--baseline_pool` for the checkpoint; use the matching value or a fresh `--name`. |
| `KeyError: 'eef_hist_mean'` / `'state_mean'` | Wrong policy class or stale `dataset_stats.json`; use `--policy_class baseline` and a fresh `--name`. |
| `KeyError: unknown eval pose set` | `--pose_files` names do not match `camera_poses/`; use `train_cameras.json test_cameras.json`. |
| `IndexError: pose index ... out of range` | `--num_episodes` exceeds the pose file length; lower it. |
| `ModuleNotFoundError: robosuite` | `git submodule update --init robosuite_source && pip install -e robosuite_source --no-deps`. |
| EGL/GLFW/OpenGL error | Keep `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`; try `MUJOCO_GL=osmesa` and report. |
| `CUDA out of memory` | batch 16 (then 8), keep everything else fixed; report the final batch. |
| W&B hangs | `WANDB_MODE=offline`; sync later with `wandb sync <offline-run-dir>`. |

## Report back

1. `git rev-parse HEAD` and `git status --short`.
2. GPU model/memory, torch/CUDA versions, conda env, GPU used.
3. Smoke result: losses, epoch-0 success lines, artifact paths, stats keys, marker.
4. Full-run command, log path, offline W&B run id, resume count.
5. v1 success table by epoch × pose set.
6. v2 re-evaluation summary and the side-by-side Part B table.
7. Any deviations (batch, EGL fallback, OOM, missing data).
