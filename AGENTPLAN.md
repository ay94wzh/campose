# Multi-scale Plücker-Conditioned Target-View Reconstruction — Training Plan

## 1. Purpose and Hypotheses

The policy is trained with an auxiliary target-view reconstruction branch, then
evaluated **without** that branch. The primary question is whether the auxiliary
task improves the policy's success rate under unseen input-camera poses.

Test three hypotheses, in order:

- **H1 — Reconstruction helps.** Adding target-view reconstruction (Exp A)
  improves policy success over the plain diffusion-policy baseline.
- **H2 — Target geometry helps.** Conditioning the decoder on the target
  camera's Plücker map (Exp B) improves over reconstruction from `z_g` alone.
- **H3 — Multi-scale geometry helps.** Injecting target Plücker features at
  multiple decoder resolutions (Exp C) improves over single-scale injection.

The main metric is **task success rate on held-out input-camera poses**. The
reconstruction loss and image quality are diagnostics, not deployment metrics.
The branch is discarded at inference, so any success-rate change is attributable
to the learned representation, not to extra inference computation.

## 2. Method Summary

The existing inference policy is unchanged: each input view is RGB+Plücker
(9 channels), encoded by the shared ResNet and Spatial Softmax, fused by the
cross-view MHA into `z_g ∈ R^512`, concatenated with proprio, and passed to the
diffusion head. During training, the shared encoder receives gradients from
both the diffusion loss and the auxiliary reconstruction loss. The branch adds:

- a fixed target-camera pose per training episode;
- a target Plücker map `P_a` (6 channels);
- a hierarchical Plücker encoder producing features at 128/64/32/16 px;
- a decoder that maps `z_g` to a 128×128 target RGB image, concatenating the
  matching Plücker feature at each stage;
- an L1 reconstruction loss against the simulator-rendered target view.

Fixed design decisions:

| Item | Decision |
|---|---|
| Reconstruction output | 128×128 |
| Target Plücker | 256×256, computed with the existing ray grid |
| Plücker frame | World frame for input and target (same frame) |
| Target pose | One per HDF5 demo (episode), fixed across timesteps and epochs |
| Perturbation | `T_a = T_default @ ΔT`; local translation U(±2 cm), local per-axis rotation U(±5°) |
| Target GT | Rendered at 128×128, RGB/255, no crop, synchronized to the input timestep |
| Loss weight | `λ_a = 0.1` |
| Branch at inference | Dropped; rollouts are unchanged |

Target AgentView is **not** one of the policy's input views. Input views are
sampled from the training camera-pose files, as in the existing baseline.

Plücker maps describe camera-ray geometry, not scene depth or pixel
correspondences. The auxiliary task is learned cross-view synthesis, not
explicit geometric reprojection.

## 3. Data and View Protocol

| Task | Dataset suffix | Train input views | Eval max steps | Baseline reference (v2, epoch 10000, K=1) |
|---|---|---|---:|---|
| LiftRand | `liftrand_eef_delta` | K=1, `n=3`, `m=1` | 600 | 19/50 train, 19/50 test |
| CanRand | `canrand_eef_delta` | K=2, `n=4`, `m=2` | 600 | 7/50 train, 12/50 test |
| SquareRand | `squarerand_eef_delta` | K=2, `n=4`, `m=2` | 600 | 0/50 train, 0/50 test |

- Demos: match the existing baseline split per task (200 training episodes +
  10 held-out validation episodes where the HDF5 has enough demos).
- Input camera poses: `train_cameras.json` (1000 poses); validation uses
  `test_cameras.json` (100 poses) with no crop.
- Evaluation poses: `train_cameras.json` (seen) and `test_cameras.json`
  (unseen), one fixed pose per episode for K=1, as in the existing protocol.
- Training samples: one observation frame at `t`, 16-step action window,
  execute 8 actions at inference, 4-step camera-frame EEF history, 7-D joint
  proprio zeroed before min-max normalization.
- Train-time image augmentation: one random crop (side 204–256, resized to
  256) shared across a view's RGB+Plücker channels. Evaluation uses no crop.
- Target pose: one pose per demo, sampled from the run seed and the global demo
  id; fixed across timesteps and epochs. The target-pose RNG is independent of
  the input-pose/proprio RNG, so adding the branch does not perturb the input
  sampling sequence.
- The target branch uses the world-frame Plücker convention already used by
  the input views. This keeps the auxiliary geometry consistent with the
  policy's input geometry and preserves baseline comparability.

## 4. Training Configuration

Mirror the existing baseline protocol for every arm; change only the
reconstruction mode and the settings required to fit memory.

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas `(0.95, 0.999)`, eps `1e-8` |
| LR / weight decay | `1e-4` / `1e-6` |
| Schedule | cosine with 500-step warmup |
| Grad clip | 1.0 |
| Precision | bf16 autocast |
| Epochs | 10,001 |
| Action horizon / obs | 16 actions / 1 observation frame |
| Crop / proprio | random crop 204–256 / proprio drop probability 1.0 |
| Recon weight | `0.1` |
| Target jitter | ±2 cm translation, ±5° per camera-local axis |
| Seeds | seed 0 for screening; seeds 0/1/2 for final claims |
| Batch | largest that fits; **identical across arms** (see below) |
| Checkpoint/eval cadence | Lift: eval every 500, 10 episodes through epoch 6000, 50 from 6500. Can/Square: eval every 1000, 10 episodes through 6000, 50 from 7000. |

**Batch-size control.** The branch adds a render and an image decoder. If a
reconstruction arm cannot fit the baseline batch size, the baseline control for
that campaign must be retrained at the same smaller batch. Do not compare a
recon arm at batch 8–16 against a reused baseline at batch 32. For Lift, start
at 16 and try 24/32; for Can/Square, start at 8 and try 16.

**Checkpoint selection.** Use the fixed epoch-10000 checkpoint for the primary
comparison in every arm. Also evaluate the retained 7000/8000/9000/10000
checkpoints and report the best `train_cameras` checkpoint as an exploratory
secondary number. Do not select a model on `test_cameras`.

**Logging.** Log `loss`, `loss_diffusion`, `loss_recon`, learning rate, and the
validation counterparts. The reconstruction term should be logged separately
so a change in policy loss can be attributed. W&B project `test`; the group
follows the run name (seed suffix stripped). Record the git commit and the
exact command in each run.

## 5. Experiment Arms

| Arm | Reconstruction target | Decoder conditioning | CLI flag | Question |
|---|---|---|---|---|
| Baseline | None | None | `--recon_mode none` | Reference policy |
| Exp A | Perturbed AgentView | `z_g` only | `--recon_mode z_only` | Does reconstruction alone help? |
| Exp B | Perturbed AgentView | `z_g` + single-scale Plücker (G1 @ 16×16) | `--recon_mode single_plucker` | Does target geometry help? |
| Exp C | Perturbed AgentView | `z_g` + multi-scale Plücker (G1–G4) | `--recon_mode multi_plucker` | Does multi-scale geometry help? |

All arms use the **spatial-softmax** encoder
(`--baseline_pool spatial_softmax`, the default) and
`--recon_weight 0.1 --target_pos_jitter 0.02 --target_rot_jitter_deg 5.0`.
The `avgpool` v1 encoder is a separate pooling ablation and is **not** trained
or compared in this campaign; the Baseline arm is the spatial-softmax
diffusion policy without the reconstruction branch. The baseline arm does not
render target views and does not add the reconstruction loss.

## 6. Campaign Stages and Gates

### Stage 0 — Smoke test (≈1–2 h)

One short run per mode on a small demo subset, batch 2–4, one epoch, with one
cheap evaluation episode. Check:

- `loss`, `loss_diffusion`, and `loss_recon` are finite;
- the validation reconstruction loss is finite;
- checkpoint loading/evaluation auto-detects the reconstruction mode;
- target GT/Plücker shapes are correct and the shared simulator camera is
  restored after each sample;
- no OOM.

Gate 0: all checks pass. Do not start the campaign otherwise.

### Stage 1 — Lift screening (≈5–8 days on one GPU)

Run the three reconstruction arms (A/B/C) on LiftRand, seed 0, full 10,001
epochs. Use a matched baseline control if the batch size differs from the
existing baseline; otherwise the existing Lift v2 epoch-10000 checkpoint may be
reused for the first screen.

Gate 1 (primary, epoch 10000, 50 episodes, K=1):

- **Proceed** if Exp C improves `test_cameras` success by ≥10 points over the
  matched baseline, or reaches ≥2× the baseline when the baseline is ≤10/50.
- **Proceed with the best arm** if Exp A or B clears the same threshold even
  when another arm does not.
- **Pause** if all arms are within noise (±5 points). Inspect reconstruction
  loss/quality and the training curve before spending more compute; only then
  consider a λ or perturbation sweep.
- **Stop** if reconstruction is healthy but policy success is flat or worse on
  both seen and unseen poses.

### Stage 2 — CanRand (≈4–7 days)

Run the best-performing mode plus the other two modes only if Stage 1 shows a
clear signal. Use the task's K=2 training protocol and a matched baseline
control. Evaluate both K=1 and the K=2 random-view addendum.

Gate 2 (epoch 10000, 100 episodes preferred):

- **Proceed** if the best mode improves `test_cameras` success by ≥10 points
  over the matched baseline on 100 episodes, or reaches ≥1.5× the baseline
  when the baseline is ≤15/100, and the direction is consistent on
  `train_cameras`.
- **Deprioritize** if Can is flat; report the Lift result and do not spend
  compute on Square.

### Stage 3 — SquareRand (optional, ≈2–4 days)

Square is at the floor for the existing baseline (0–1/50). Run only the best
mode and a matched baseline, and treat it as a diagnostic of whether the
auxiliary task changes representation quality when the policy itself is
undertrained. Do not use Square for the go/no-go decision.

### Stage 4 — Replication and ablations (≈1–2 weeks)

For the task(s) that passed, run seeds 0/1/2 for the best mode and the matched
baseline. Then, if time permits:

- λ_a sweep: 0.05 / 0.1 / 0.3 on the best mode, Lift or Can, seed 0;
- perturbation sweep: ±1 cm/±2.5°, ±2 cm/±5° (default), ±4 cm/±10°;
- reconstruction-resolution ablation (128 vs 256) only if the 128 run clears
  the gate; this requires a matching decoder configuration.

Report final claims as mean ± std over seeds, with paired episode seeds.

## 7. Evaluation Protocol

### 7.1 Policy success

- Primary: 50 episodes per pose file for screening; 100 episodes per pose file
  for final comparisons. `test_cameras.json` contains 100 poses, so 100 K=1
  episodes are available.
- Evaluate the fixed epoch-10000 checkpoint in every arm. Also report the best
  `train_cameras` checkpoint as secondary; never select on `test_cameras`.
- Use the existing offline evaluator, pose files, episode seeds, and diffusion
  noise seeds, so baseline and reconstruction arms are paired.
- Report `train_cameras` (seen input poses) and `test_cameras` (unseen input
  poses) separately. The primary claim is on `test_cameras`.
- K=2 addendum for the best arms: two distinct views drawn uniformly from the
  whole pose file, fixed per episode, seeded by `eval_pose_seed + episode`. This
  checks whether the result survives the fusion path used during training.

### 7.2 Reconstruction quality (diagnostic)

- In-range: validation reconstruction L1 at the default ±2 cm/±5° target
  distribution; also report PSNR and SSIM.
- Out-of-range: a separate offline evaluation with target poses at ±4 cm/±10°;
  report L1/PSNR/SSIM. This tests whether the representation extrapolates
  beyond the training perturbation.
- Use the 10 held-out validation demos and a fixed set of timesteps (e.g., 10
  evenly spaced per demo) so all arms see the same states.
- Save qualitative grids: input views, target GT, prediction, for 5–10
  examples per task. Reconstruction quality is used to explain the policy
  result, not to select the winning arm.

### 7.3 Statistics

- A 50-episode success rate has roughly ±7% standard error; a 5-point
  difference is noise. Use 100 episodes for decisive comparisons.
- Screening uses seed 0; final claims require 3 seeds and a mean improvement
  larger than one standard error, with the same direction on both pose sets.
- Report paired differences against the matched baseline, not just absolute
  success rates.

## 8. Decision Rules

| Observed pattern | Interpretation | Action |
|---|---|---|
| C > B > A > Baseline on test | Strong support for H1–H3 | Replicate C and Baseline; Ablate λ/perturbation |
| B ≈ C > A > Baseline | Geometry helps; multi-scale does not add | Replicate B; report C as no better than B |
| A > Baseline, B/C ≈ A | Reconstruction helps; geometry does not | Replicate A; inspect reconstruction quality |
| A/B/C ≈ Baseline | Auxiliary task does not help at this schedule/λ | Pause; try λ or perturbation only if reconstruction is healthy |
| Any arm > Baseline on train but not test | Seen-pose overfitting / no generalization | Do not claim viewpoint robustness |
| Any arm < Baseline on both | Auxiliary loss hurts the policy | Report negative result; inspect λ and target difficulty |

The primary claim is a **test_cameras** improvement at the fixed epoch-10000
checkpoint, replicated across seeds. Improvements on `train_cameras` alone are
not sufficient.

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| OOM from the extra render/decoder | Start at batch 8–16; retrain the baseline at the same batch; monitor `nvidia-smi` |
| Extra render slows training | Account for 1.3–1.6× step time; checkpoint/resume; run sequentially on the shared GPU |
| Reconstruction loss does not improve | Log it separately; inspect qualitative outputs; try λ/perturbation only after Gate 1 |
| Target view is too easy | Main runs use pose-file inputs, not `default_cam=True`; the target is a new perturbed AgentView |
| Target view is too hard | Start with ±2 cm/±5°; if recon L1 stays near the all-gray baseline, reduce the perturbation |
| Batch-size confound | Match batch size across arms; do not reuse a baseline trained at a different batch |
| Task floor (Square) | Use Square only as a diagnostic; do not gate on it |
| Seed noise / test-set selection | 100 episodes for final claims; pre-register epoch 10000; 3 seeds |
| Checkpoint/architecture mismatch | Let the evaluator auto-detect the reconstruction mode from the checkpoint; never load a recon checkpoint with the wrong mode |
| Shared-GPU contention | Preserve 50-episode checkpoints outside the run directory; resume from the latest checkpoint |

## 10. Compute Budget and Schedule

Estimated single-GPU times, assuming a shared GPU with contention:

| Stage | Runs | Estimate |
|---|---|---:|
| Smoke | 4 short runs | 1–2 h |
| Lift screening | 3 recon arms + matched baseline if needed | 5–8 days |
| Can | up to 4 arms | 4–7 days |
| Square (optional) | 2 arms | 2–4 days |
| Replication | 2 arms × 3 seeds on one task | 6–10 days |
| Ablations (optional) | 2–4 short runs | 3–6 days |

Decision point: if Stage 1 does not clear Gate 1, stop after Lift and report the
negative result. The full campaign is roughly 3–5 weeks on one shared GPU.

Checkpoint retention: `cleanup_ckpt` keeps only the latest three checkpoints.
Copy the 50-episode checkpoints (e.g., epochs 7000–10000) to a `_keep`
directory immediately, as the existing baseline campaigns do.

## 11. Artifacts and Reproducibility

### Run names and paths

- Training: `<task>_recon_<mode>_seed_<k>`, e.g.
  `lift_recon_multi_plucker_seed_0`; baseline:
  `lift_recon_none_seed_0`.
- Checkpoints: `/data/zihan/campose_checkpoints/<run_name>/`, with the
  50-episode checkpoints copied to `<run_name>_keep/`.
- Evaluation: `/data/zihan/campose_eval/recon_<task>/<run_name>/`, containing
  per-pose `summary.json`, `success_by_seed.json`, and MP4s.
- Logs: `/data/zihan/campose_logs/recon_<task>_<run_name>.log`.
- W&B: project `test`; the group follows the run name (the trainer strips the
  final `_seed_<k>`). The run config records `recon_mode`, `recon_weight`,
  target jitter, and seed.

### Training command template

```bash
WANDB_MODE=offline python policy_robosuite/train.py \
  --policy_class baseline --name <run_name> \
  --dataset_suffix <task>_eef_delta \
  --num_side_cam <K> --n <n> --m <m> \
  --batch_size <B> --seed <k> --num_epochs 10001 \
  --eval_start_epoch 6000 --eval_every <500|1000> \
  --eval_max_steps 600 --save_every <500|1000> \
  --baseline_pool spatial_softmax \
  --recon_mode <none|z_only|single_plucker|multi_plucker> \
  --recon_weight 0.1 \
  --target_pos_jitter 0.02 --target_rot_jitter_deg 5.0 \
  --ckpt_dir /data/zihan/campose_checkpoints/<run_name>
```

Task-specific values:

| Task | suffix | K | n | m | eval_every | save_every |
|---|---|---:|---:|---:|---:|---:|
| Lift | `liftrand_eef_delta` | 1 | 3 | 1 | 500 | 500 |
| Can | `canrand_eef_delta` | 2 | 4 | 2 | 1000 | 1000 |
| Square | `squarerand_eef_delta` | 2 | 4 | 2 | 1000 | 1000 |

### Evaluation command

```bash
python policy_robosuite/eval_baseline_checkpoint.py \
  --ckpt <run_dir>/epoch_10000.pth \
  --baseline_pool spatial_softmax \
  --pose_files train_cameras.json test_cameras.json \
  --num_episodes 50 --eval_max_steps 600 \
  --save_dir /data/zihan/campose_eval/recon_<task>/<run_name>
```

For the K=2 addendum, add:

```bash
--eval_num_views 2 --eval_view_sampling random --eval_pose_seed 0
```

The evaluator reads the reconstruction mode from the checkpoint and constructs
the matching dormant branch; rollout behavior is unchanged.

### Reporting

Produce:

1. a success table: task × arm × checkpoint × pose set × K;
2. a reconstruction table: task × arm × in/out-of-range × L1/PSNR/SSIM;
3. loss curves for diffusion and reconstruction;
4. a qualitative reconstruction grid;
5. a short conclusion stating whether H1/H2/H3 were supported, with paired
   differences and seed variance.
