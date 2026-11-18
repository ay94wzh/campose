# Can/Square multi-view baseline — results

Execution of `PLAN_CAN_SQUARE.md`. All four runs complete on 2026-11-16 10:41.
Git HEAD `f8a95e9` (`method/baseline`). Host: `user-Super-Server` (the plan's
header says `miroc-server`; all plan paths, data and prior checkpoints live here).

> **Eval view count (added 2026-11-18):** the main table below is **K=1**.
> `BaselineEvaluator` hard-codes `num_side_cam = 1` at eval even though training
> used K=2. A K=2 random-view addendum is at the end of this file.

## Deviations from the plan

- **Shared GPU 1 with other users' jobs for the entire campaign.** `batch_size`
  was **16** instead of 32 (the plan's §4.4 OOM fallback, applied from the start
  by user decision). No mixing of batch sizes between v1 and v2.
- **One GPU** → the four runs were queued sequentially in the plan's §4.3 order:
  Can-v1 → Can-v2 → Square-v1 → Square-v2.
- **Transient GPU 0 attempt** for Square-v2 (2026-11-14 13:41): OOM on its first
  training step because another user's allocation grew; reverted to the queue.
  Its valid `epoch_0.pth` was resumed at batch 16, so the run is protocol-identical.
- Everything else follows the plan: `num_side_cam=2`, `n=4`, `m=2`, seed 0,
  lr 1e-4, wd 1e-6, 10001 epochs, eval every 1000, 10 episodes through epoch
  6000 and 50 from 7000, `eval_max_steps=600`, crop transform, pose files.

## Per-run summary

| Run | Attempts | Best epoch | Train best / last | Test best / last | Final loss | W&B run | Wall clock |
|---|---|---|---|---|---|---|---|
| Can / `avgpool` | 4 | 9000 | 2/50 / 1/50 | 0/50 / 0/50 | 0.006733 | `wgimhrh4` | 30.5 h |
| Can / `spatial_softmax` | 3 | 7000 (train), 9000 (test) | 9/50 / 7/50 | 14/50 / 12/50 | 0.007906 | `9aunxvzj` | 22.1 h |
| Square / `avgpool` | 3 | 10000 | 1/50 / 1/50 | 0/50 / 0/50 | 0.004975 | `gfgqo3ur` | 24.2 h |
| Square / `spatial_softmax` | 3 | 7000 | 1/50 / 0/50 | 0/50 / 0/50 | 0.004537 | `ygb2bxps` | 17.9 h |

"Best" = highest train_cameras success among the 50-episode evals (7000–10000);
"last" = epoch 10000. Total wall clock ≈ 95 h.

All 50-episode checkpoints are preserved outside the run directory (so
`cleanup_ckpt(keep=3)` cannot delete them):

```
/data/zihan/campose_checkpoints/<name>_keep/epoch_{7000,8000,9000,10000}.pth
```

for each of the four run names. 4 × 4 × ~3.3 GB ≈ 52 GB.

## Full success matrix (train_cameras / test_cameras, /50)

| Task | Pool | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|
| Can | avgpool | 1/50, 0/50 | 0/50, 0/50 | 2/50, 0/50 | 1/50, 0/50 |
| Can | spatial_softmax | 9/50, 9/50 | 8/50, 9/50 | 8/50, 14/50 | 7/50, 12/50 |
| Square | avgpool | 0/50, 0/50 | 0/50, 0/50 | 0/50, 0/50 | 1/50, 0/50 |
| Square | spatial_softmax | 1/50, 0/50 | 0/50, 0/50 | 0/50, 0/50 | 0/50, 0/50 |

## Answer to the plan's question

Does the Lift ranking (v1 `avgpool` > v2 `spatial_softmax`) hold, reverse, or
vanish once two-view fusion is actually trained?

- **Can: reverses.** v2 is far ahead of v1 on both pose sets (test 14/50 vs
  0/50; train 9/50 vs 2/50). The test gap is ~28 points, well outside the ~7%
  standard error of a 50-episode estimate.
- **Square: vanishes into the floor.** Both variants are 0–1/50 across all
  checkpoints, indistinguishable from zero; this task gives no usable v1-v2
  signal at this schedule.
- Therefore the Lift ranking does **not** hold under live K=2 fusion; on the
  task where either variant learns anything (Can), the ordering is v2 > v1.

## Interpretation caveats (plan §9)

- One seed, 1/10 of the article schedule; absolute numbers are not comparable
  to the article's DP table.
- 50-episode success has ~7% standard error; a 5-point difference is noise.
  The Can test gap is larger than that, but it is still one seed.
- This baseline is not the article's DP: FiLM conditioning, EEF history and the
  519-D observation feature are present here.
- The origin ACT already fuses views with joint token attention; any claim about
  token fusion being novel must be measured against ACT, not this baseline.
- Can-v2's test curve peaked at 9000 (14/50) and dipped at 10000 (12/50); the
  train curve is flat/declining (9→8→8→7). There is no evidence that Can-v2 was
  still rising at the end, so this run alone does not justify extending the
  schedule. Square is at the floor for both variants.

## Throughput notes

- Per-epoch time ranged from ~2.8 s (co-tenant idle) to ~13.7 s (co-tenant
  saturating the GPU). The 1000→2000 interval on the first run hit 2.67 h/1000,
  slightly over the plan's 2.5 h gate, purely from contention; later runs were
  faster (total 95 h vs the 6-day sequential projection).
- Trainer auto-exit/restart worked as designed: 3–4 attempts per run, every
  resume loaded the latest checkpoint and continued the same W&B run.

## Addendum: K=2 random-view eval (2026-11-18)

### Why

Training used `num_side_cam=2` (`n=4`, `m=2`), but the plan's evaluation
protocol is K=1: `BaselineEvaluator.__init__` hard-codes
`self.num_side_cam = 1`, so every number in the main table above was produced
with a **single** camera view. This addendum measures the same epoch-10000
checkpoints with **two views**, i.e. with the fusion path exercised at eval as
it was during training.

### Protocol

- Checkpoints: `epoch_10000.pth` from each run's `_keep/` directory.
- Both `train_cameras.json` and `test_cameras.json`; 50 episodes each;
  `eval_max_steps=600`.
- **K=2 random views:** for each episode, two distinct camera poses are drawn
  uniformly **without replacement from the whole pose file**, fixed for the
  whole episode. The pair is seeded by `eval_pose_seed + episode_num`
  (`eval_pose_seed=0`), so it is reproducible and independent of the global
  RNG used for object placement.
- Everything else matches the main protocol (DP single-frame obs, 4-step EEF
  history per view, `prob_drop_proprio=1.0`, crop-free eval, action chunk 8).
- Code: `policy_robosuite/baseline_eval.py` now accepts opt-in
  `eval_num_views` / `eval_view_sampling` / `eval_pose_seed`; the defaults are
  unchanged (K=1, `fixed`), so the original protocol is untouched.
  `eval_baseline_checkpoint.py` exposes them as
  `--eval_num_views 2 --eval_view_sampling random`.
- Outputs:
  `/data/zihan/campose_eval/2view_random_epoch10000/<run>/`
  with per-pose `summary.json` and `success_by_seed.json`; logs in
  `/data/zihan/campose_logs/2view_<task>_<pool>.log`.

### Results (epoch 10000, train_cameras / test_cameras, /50)

| Run | K=1 (main table) | K=2 random views |
|---|---|---|
| Can / `avgpool` | 1/50, 0/50 | 0/50, 1/50 |
| Can / `spatial_softmax` | 7/50, 12/50 | **15/50, 15/50** |
| Square / `avgpool` | 1/50, 0/50 | 0/50, 0/50 |
| Square / `spatial_softmax` | 0/50, 0/50 | 0/50, 0/50 |

### Interpretation

- **The Can conclusion is unchanged and if anything strengthened.** With K=2
  random views, `spatial_softmax` reaches 15/50 on both pose sets while
  `avgpool` stays at 0–1/50. The v2 > v1 reversal is not an artifact of
  single-view evaluation; it holds under the K=2 fusion path the models were
  trained with.
- For Can/`spatial_softmax`, K=2 raises train success from 7/50 to 15/50
  (+8 points) and test from 12/50 to 15/50 (+3 points). The test change is
  within the ~7% standard error of a 50-episode estimate; the train change is
  suggestive but still one seed.
- **Square remains at the floor under both protocols** (0–1/50), so it gives
  no view-count signal. This is a task/schedule limitation, not an eval-mode
  effect.
- This K=2 eval samples arbitrary pose pairs from the whole pose file; it is
  **not** the article's deterministic adjacent-pair (`2i`, `2i+1`) protocol.
  Those numbers would answer a different question and are not reported here.
