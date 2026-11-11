# Plan: Can/Square multi-view baseline training (1/10 schedule)

Status: ready for remote execution on `miroc-server`. This file is additive to
`PLAN.md` (which covers the Lift v1 run only). Do not modify the Lift runs.

## 0. Why this run

- **Lift stays single-view.** Every article Lift run uses `num_side_cam=1`
  (including the `scaling_n` sweep). Running Lift with 2 views would change the
  published protocol and make the result non-comparable.
- **Can and Square are the article's multi-view tasks.** The article trains both
  with `num_side_cam=2, n=4, m=2`, for ACT and DP alike
  (`reproduce/paper_runs.yaml`).
- **The scientific point:** with K=1 the fusion MHA in `BaselineEncoder` had one
  active key, so `z_global = W_O W_V z_v` and the query/key projections were
  gradient-dead. With K=2 the two-view fusion path is exercised for the first
  time. This run measures whether v1 (`avgpool`) vs v2 (`spatial_softmax`)
  behaves differently when fusion is live.
- **Same training method as the Lift baseline:** same policy class, optimiser,
  LR, batch, normalisation, image/Plücker input, EEF-history FiLM, DDPM head.
  Only the dataset and camera protocol change.

Non-goals: no new architecture, no view dropout, no changes to Lift, no attempt
to reproduce the article's ACT/DP absolute numbers.

## 1. Fixed configuration

| Setting | Value | Why |
|---|---|---|
| Tasks | Can, Square | article's multi-view tasks |
| `dataset_suffix` | `canrand_eef_delta`, `squarerand_eef_delta` | article |
| `num_side_cam` | **2** | article Can/Square protocol |
| `n`, `m` | **4, 2** | article Can/Square protocol |
| `num_epochs` | **10001** | 1/10 of the article DP recipe (100001) |
| `eval_start_epoch` | **6000** | 1/10 of 60000 |
| `eval_every` | **1000** | strict 1/10 would be 200; eval dominates cost |
| `save_every` | 1000 | aligned with eval |
| `batch_size` | 32 | same as Lift baseline; fallback 16 only if OOM |
| `lr` / `weight_decay` | 1e-4 / 1e-6 | same as Lift baseline |
| `seed` | 0 | one seed (compute-tight) |
| `eval_max_steps` | 600 | article |
| `eval_episodes` | 10 before 6000, 50 after | `train.py` behaviour |
| `prob_drop_proprio` | default 1.0 | Lift baseline convention |
| `transform` | default `crop` | article main runs |
| pose files | `train_cameras.json`, `test_cameras.json` | article/default |
| `baseline_pool` | both `avgpool` and `spatial_softmax` | v1 vs v2 |

**Schedule caveat.** `eval_every=1000` is a deliberate deviation from a strict
1/10 of `eval_every=2000`. Eval costs ~20–40 min per checkpoint (2 pose sets ×
50 episodes × 600 steps), so 200 would spend more time evaluating than training.
Late checkpoints 7000/8000/9000/10000 still get the 50-episode protocol.

## 2. Preflight — do not skip

### 2.1 Code

```bash
cd <REPO_ROOT>
git fetch origin
git checkout method/baseline
git pull --ff-only origin method/baseline
git rev-parse HEAD          # record in the final report
git status --short          # must be empty
```

Verify the flags exist:

```bash
python policy_robosuite/train.py --help | grep -A2 -- '--baseline_pool'
python policy_robosuite/train.py --help | grep -A1 -- '--num_side_cam'
```

### 2.2 Environment

```bash
conda activate know_your_camera
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=offline

python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "import robosuite, mujoco, diffusers, h5py; print(robosuite.__version__, mujoco.__version__, diffusers.__version__)"
nvidia-smi
```

`torch.cuda.is_available()` must be `True`.

### 2.3 Data

```bash
python - <<'PY'
import h5py, json, os

for suffix in ("canrand_eef_delta", "squarerand_eef_delta"):
    path = os.path.join("policy_robosuite", "demos", suffix + ".hdf5")
    assert os.path.exists(path), f"MISSING: {path}"
    with h5py.File(path, "r") as f:
        demos = [k for k in f["data"] if k.startswith("demo_")]
        print(suffix, "demos:", len(demos),
              "action_space:", f["data"].attrs["action_space"])
        d = f["data"][demos[0]]
        print("  actions:", d["actions"].shape, "states:", d["states"].shape)

for name in ("train_cameras", "test_cameras"):
    p = os.path.join("policy_robosuite", "camera_poses", name + ".json")
    raw = json.load(open(p))
    print(name, len(raw["poses"]), "poses")
PY
```

Expect `eef_delta` for both, 1000 train poses and 100 test poses. Record the demo
count `N`; with `n=4, m=2` the loader covers `2*(N-1)+4` poses. A warning that
fewer than 1000 poses are covered is **expected**, not an error (Lift covers
only 202).

### 2.4 CPU tests

```bash
python tests/test_baseline_encoder.py
```

Must end with all tests passing, including `ok encoder shapes pool=... k=2` for
both pooling variants.

## 3. Smoke test — both pools, Can only

Run two 2-epoch smokes, one per pool. The K=2 data path (two renders per sample)
has never been exercised in a full training step, so this is mandatory.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline \
python policy_robosuite/train.py \
  --policy_class baseline \
  --name baseline_can_smoke_avgpool_seed_0 \
  --dataset_suffix canrand_eef_delta \
  --num_side_cam 2 --n 4 --m 2 \
  --batch_size 4 --seed 0 \
  --num_epochs 2 --eval_every 1 --eval_episodes 1 --eval_max_steps 30 \
  --save_every 1 --baseline_pool avgpool \
  --lr 1e-4 --weight_decay 1e-6 \
  --ckpt_dir /data/zihan/campose_checkpoints/baseline_can_smoke_avgpool_seed_0
```

Repeat with `--baseline_pool spatial_softmax`,
`--name baseline_can_smoke_spatial_softmax_seed_0`, and its own `--ckpt_dir`.

Verify in each smoke directory:

- [ ] No shape errors, no NaNs; the camera-coverage line is printed.
- [ ] `config.json` has `"dataset_suffix": "canrand_eef_delta"`,
      `"num_side_cam": 2`, `"n": 4`, `"m": 2`, `"baseline_pool"` matching, and
      `"baseline_eval_agentview": false`.
- [ ] `dataset_stats.json` has `action_mean/std` (7), `state_mean/std` (7),
      `eef_hist_mean/std` (44).
- [ ] Both `eval_epoch_0_train_cameras/` and `eval_epoch_0_test_cameras/` exist
      with `success_by_seed.json`.
- [ ] `epoch_1.pth` has `baseline_arch` = `avgpool_nobs1_proprio7_v1` or
      `spatial_softmax_nobs1_proprio7_v1` as appropriate.

Do not launch the full runs if either smoke fails.

## 4. Full runs

### 4.1 Matrix — four runs, seed 0

| Session | Task | Pool | Checkpoint dir |
|---|---|---|---|
| `campose_can_v1` | Can | `avgpool` | `baseline_can_rand_avgpool_seed_0` |
| `campose_can_v2` | Can | `spatial_softmax` | `baseline_can_rand_spatial_softmax_seed_0` |
| `campose_square_v1` | Square | `avgpool` | `baseline_square_rand_avgpool_seed_0` |
| `campose_square_v2` | Square | `spatial_softmax` | `baseline_square_rand_spatial_softmax_seed_0` |

Logs: `/data/zihan/campose_logs/<name>.log`.
Checkpoints: `/data/zihan/campose_checkpoints/<name>/`.

### 4.2 Auto-resume wrapper

The trainer exits itself after 7.5 h (`train.py:200-202`) and resumes from the
latest checkpoint with the same command. A 2-view run is expected to need 2–3
restarts. Create this launcher once:

```bash
cat > /data/zihan/campose_logs/run_can_square_job.sh <<'SH'
#!/usr/bin/env bash
set -u
REPO="$1"; TASK="$2"; POOL="$3"; GPU="$4"
DS="${TASK}rand_eef_delta"
NAME="baseline_${TASK}_rand_${POOL}_seed_0"
CKPT="/data/zihan/campose_checkpoints/${NAME}"
LOG="/data/zihan/campose_logs/${NAME}.log"
mkdir -p "$CKPT" /data/zihan/campose_logs
cd "$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="$GPU"

# Fail fast if the tmux session did not inherit the conda environment.  If it
# did not, replace `python` below with the absolute interpreter path from
# `conda run -n know_your_camera which python`.
python -c "import robosuite" 2>/dev/null || {
  echo "ERROR: conda env know_your_camera is not active in this session"; exit 1;
}

for attempt in 1 2 3 4 5; do
  python policy_robosuite/train.py \
    --policy_class baseline --name "$NAME" \
    --dataset_suffix "$DS" --num_side_cam 2 --n 4 --m 2 \
    --batch_size 32 --seed 0 --num_epochs 10001 \
    --eval_start_epoch 6000 --eval_every 1000 --eval_max_steps 600 \
    --save_every 1000 --baseline_pool "$POOL" \
    --lr 1e-4 --weight_decay 1e-6 \
    --ckpt_dir "$CKPT" 2>&1 | tee -a "$LOG"

  last=$(ls "$CKPT"/epoch_*.pth 2>/dev/null \
         | sed 's/.*epoch_\([0-9]*\).pth/\1/' | sort -n | tail -1)
  if [ -n "$last" ] && [ "$last" -ge 10000 ]; then
    echo "DONE ${NAME} at epoch ${last}"
    break
  fi
  echo "RESTART ${NAME} attempt ${attempt} from epoch ${last:-none}"
  sleep 20
done
SH
chmod +x /data/zihan/campose_logs/run_can_square_job.sh
```

The loop stops when the last checkpoint reaches epoch 10000. It makes at most
5 attempts; if it exhausts them, stop and report with the log tail.

### 4.3 Launch

Pick two free GPUs from `nvidia-smi` and substitute `<REPO_ROOT>`, `<GPU1>`,
`<GPU2>`. Do not share a GPU with another training run.

```bash
tmux new-session -d -s campose_can_v1 \
  "/data/zihan/campose_logs/run_can_square_job.sh <REPO_ROOT> can avgpool <GPU1>"
tmux new-session -d -s campose_can_v2 \
  "/data/zihan/campose_logs/run_can_square_job.sh <REPO_ROOT> can spatial_softmax <GPU2>"
```

After both Can sessions report `DONE`, launch Square the same way:

```bash
tmux new-session -d -s campose_square_v1 \
  "/data/zihan/campose_logs/run_can_square_job.sh <REPO_ROOT> square avgpool <GPU1>"
tmux new-session -d -s campose_square_v2 \
  "/data/zihan/campose_logs/run_can_square_job.sh <REPO_ROOT> square spatial_softmax <GPU2>"
```

If only one GPU is free, run the four jobs sequentially in this order:
Can-v1, Can-v2, Square-v1, Square-v2; report the revised ETA.

### 4.4 OOM fallback

If batch 32 OOMs, relaunch with `--batch_size 16` **for all four runs** (the
wrapper must be edited once, consistently), and record the deviation. Do not mix
batch sizes between v1 and v2. If batch 16 also OOMs, stop and report.

### 4.5 Expected cost (measure, don't trust)

Lift 1-view, 10001 epochs took ~9–10 h. Two renders + two encoder passes per
sample should make these ~1.5–2.5× slower, so budget **~15–25 h per run**,
including evals. Four runs = 60–100 GPU-hours; on two GPUs about 1.5–2 days wall
clock. After epoch 1000, compute hours per 1000 epochs and report if it is more
than ~2.5 h.

## 5. Monitoring and checkpoint preservation

```bash
tmux ls
tail -f /data/zihan/campose_logs/baseline_can_rand_avgpool_seed_0.log
nvidia-smi
```

Check:

- [ ] `train_loss` / `val_loss` finite; no NaNs or repeated render-fault loops.
- [ ] `eval_epoch_<E>_train_cameras/` and `..._test_cameras/` appear every 1000
      epochs after 6000; `success_by_seed.json` is non-empty.
- [ ] `config.json` and `dataset_stats.json` are present in each ckpt dir.
- [ ] Record the offline W&B run id from each log.

**Preserve 50-episode checkpoints.** `cleanup_ckpt(keep=3)` deletes older
checkpoints, and eval runs *before* the save for the same epoch
(`train.py:141-198`). For each evaluated epoch `E` in 7000–10000, once both
`eval_epoch_E_*` dirs exist and `epoch_E.pth` is written, hardlink it:

```bash
KEEP=/data/zihan/campose_checkpoints/<name>_keep
mkdir -p "$KEEP"
ln /data/zihan/campose_checkpoints/<name>/epoch_7000.pth "$KEEP/epoch_7000.pth"
# repeat for 8000, 9000, 10000
```

Do not place these copies inside the run's ckpt dir with an `epoch_*.pth` name;
`cleanup_ckpt` would delete them.

## 6. Deliverables

For each of the four runs, report:

| Field | Source |
|---|---|
| best epoch | highest train_cameras success among 50-episode evals |
| train_cameras success (best, last) | `success_by_seed.json` / log |
| test_cameras success (best, last) | same |
| checkpoint paths kept | `<name>_keep/` |
| final loss, W&B run id, git HEAD | log |

Then fill this comparison table:

| Task | v1 `avgpool` (train / test) | v2 `spatial_softmax` (train / test) |
|---|---|---|
| Can | __/50, __/50 | __/50, __/50 |
| Square | __/50, __/50 | __/50, __/50 |
| Lift (reference, `PROGRESS.md` §5.2, epoch 9500) | 31/50, 29/50 | 17/50, 17/50 |

The question to answer: does the Lift ranking (v1 > v2) hold, reverse, or vanish
once two-view fusion is actually trained?

## 7. Stop and escalate if

- A dataset file is missing or `action_space` is not `eef_delta`.
- Either smoke fails.
- NaN/inf loss, or the log repeats render faults without progressing.
- Resume raises the architecture-marker error.
- OOM persists at batch 16.
- Throughput is >2.5 h per 1000 epochs (report before burning more compute).
- No eval directories appear at epoch 1000.

Do **not** stop merely because success is 0 early; the Lift v1 curve was 0% until
epoch 2000 and still peaked at 62%.

## 8. Do not change

- `lr` (1e-4), `weight_decay` (1e-6), `batch_size` (32, except the global OOM
  fallback), `seed` (0), horizon/action-window defaults, camera parameters.
- `prob_drop_proprio` (keep default 1.0), `transform` (keep `crop`).
- Do not pass `--use_plucker` or `--default_cam 1`: the baseline always uses real
  Plücker maps and the pose-file protocol.
- Do not reuse a checkpoint directory across tasks or pools; the resume guard
  will reject it, but it wastes a launch.
- Do not change `--num_side_cam`, `--n`, `--m`: they are the article protocol.

## 9. Interpretation caveats

- **One seed, 1/10 schedule.** Do not compare absolute numbers to the article's
  DP table (100001 epochs, batch 70, lr 2e-5). The valid comparison is v1 vs v2
  within this plan, under identical settings.
- **50-episode success has ~7% standard error.** A 5-point difference is noise;
  look for a consistent gap across checkpoints.
- **This is not the article's DP.** Your `baseline` policy has FiLM conditioning,
  EEF history, and the 519-D observation feature; the article's `dp.py` does
  not. Article DP Can/Square numbers are context only.
- **The origin ACT already fuses views with joint token attention**
  (`models/backbone.py:73-84`). Any later claim that token fusion is novel must
  be measured against ACT, not against this DP-style baseline.
- If success is still rising at epoch 10000, report before extending the run;
  extending changes the schedule and should be decided explicitly.
