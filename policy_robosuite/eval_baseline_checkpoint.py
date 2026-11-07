"""Evaluate a baseline checkpoint on the origin camera-pose sets.

This is the offline counterpart of ``train.py``'s evaluation block: it loads an
existing checkpoint plus its ``dataset_stats.json`` and runs the same
``BaselineEvaluator`` over ``train_cameras.json`` / ``test_cameras.json`` (one
camera pose per episode, matching the origin ``Evaluator`` with
``num_side_cam=1``).

Run from the repo root, e.g.:

    python policy_robosuite/eval_baseline_checkpoint.py \
      --ckpt /data/zihan/campose_checkpoints/baseline_lift_rand_v2_seed_0/epoch_10000.pth \
      --baseline_pool spatial_softmax \
      --pose_files train_cameras.json test_cameras.json \
      --num_episodes 50 --eval_max_steps 600 \
      --save_dir /data/zihan/campose_eval/v2_epoch10000

Use ``--agentview`` for the old fixed-agentview protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

import robosuite as suite
from robosuite.wrappers.action_wrapper import wrap_env_action_space

from baseline_common import baseline_arch_name
from baseline_eval import BaselineEvaluator
from models.baseline import BaselineDiffusionPolicy
from utils import set_seed


def str2bool(value):
    return str(value) == "1"


def build_env(dataset_path):
    """Create the same wrapped environment as ``train.py``."""
    with h5py.File(dataset_path, "r") as f:
        env_args_raw = f["data"].attrs["env_args"]
        action_space = f["data"].attrs["action_space"]
    if isinstance(action_space, bytes):
        action_space = action_space.decode("utf-8")
    env_config = json.loads(env_args_raw)
    env_kwargs = env_config["env_kwargs"]
    env_kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": False,
            "camera_heights": 256,
            "camera_widths": 256,
        }
    )
    env = suite.make(
        env_name=env_config["env_name"],
        camera_names=["agentview"],
        **env_kwargs,
    )
    if action_space in ("eef_delta", "joint_delta"):
        env = wrap_env_action_space(env, action_space)
    env.reset()

    # Import after suite.make(), as in train.py, so the EGL context exists.
    import OpenGL.GL  # noqa: F401

    return env, action_space


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline checkpoint on camera pose files"
    )
    parser.add_argument("--ckpt", required=True, help="checkpoint .pth path")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--dataset_suffix", default="liftrand_eef_delta")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--camera_poses_dir", default=None)
    parser.add_argument(
        "--pose_files",
        nargs="+",
        default=["train_cameras.json", "test_cameras.json"],
        help="pose JSON files to evaluate",
    )
    parser.add_argument("--agentview", action="store_true",
                        help="evaluate the fixed default agentview instead")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--eval_max_steps", type=int, default=600)
    parser.add_argument("--eval_save_n_video", type=int, default=2)
    parser.add_argument("--num_side_cam", type=int, default=1, choices=[1])
    parser.add_argument(
        "--baseline_pool",
        type=str,
        default="spatial_softmax",
        choices=["spatial_softmax", "avgpool"],
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    module_root = Path(__file__).resolve().parent
    if args.dataset_path is None:
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else module_root / "demos"
        args.dataset_path = str(
            (dataset_dir / (args.dataset_suffix + ".hdf5")).resolve()
        )
    if args.camera_poses_dir is None:
        args.camera_poses_dir = str((module_root / "camera_poses").resolve())

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    marker = baseline_arch_name(args.baseline_pool)
    if ckpt.get("baseline_arch") != marker:
        raise RuntimeError(
            f"checkpoint arch {ckpt.get('baseline_arch')!r} does not match "
            f"--baseline_pool {args.baseline_pool!r} (expected {marker!r})"
        )

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    stats_path = os.path.join(ckpt_dir, "dataset_stats.json")
    with open(stats_path, "r") as f:
        norm_stats = json.load(f)

    env, action_space = build_env(args.dataset_path)
    action_dim = 8 if "joint" in args.dataset_path else 7
    policy_args = argparse.Namespace(
        action_dim=action_dim,
        lr=1e-4,
        weight_decay=1e-6,
        baseline_pool=args.baseline_pool,
    )
    policy = BaselineDiffusionPolicy(policy_args).cuda()
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    eval_args = argparse.Namespace(
        chunk_size=30,
        eval_max_steps=args.eval_max_steps,
        eval_save_n_video=args.eval_save_n_video,
        camera_poses_dir=args.camera_poses_dir,
        num_side_cam=args.num_side_cam,
        default_cam=False,
        pose_files=[] if args.agentview else args.pose_files,
        prob_drop_proprio=1.0,
        use_fp16=args.use_fp16,
        use_plucker=True,
    )
    evaluator = BaselineEvaluator(
        env=env,
        norm_stats=norm_stats,
        dataset_path=args.dataset_path,
        args=eval_args,
    )

    if args.agentview:
        pose_names = ["agentview"]
    else:
        pose_names = [os.path.splitext(f)[0] for f in args.pose_files]

    os.makedirs(args.save_dir, exist_ok=True)
    results = {}
    for pose_name in pose_names:
        save_path = os.path.join(args.save_dir, pose_name)
        os.makedirs(save_path, exist_ok=True)
        evaluator.success_by_seed = {}
        successes = []
        for episode in range(args.num_episodes):
            with torch.no_grad():
                _, success, _ = evaluator.evaluate(
                    policy=policy,
                    save_path=save_path,
                    video_prefix=f"episode_{episode}",
                    pose_name=pose_name,
                    episode_num=episode,
                )
            successes.append(float(success))
        with open(os.path.join(save_path, "success_by_seed.json"), "w") as f:
            json.dump(evaluator.success_by_seed, f, indent=2)
        summary = {
            "pose_name": pose_name,
            "episodes": args.num_episodes,
            "successes": int(sum(successes)),
            "rate": float(np.mean(successes)) if successes else 0.0,
            "baseline_pool": args.baseline_pool,
            "checkpoint": os.path.abspath(args.ckpt),
        }
        with open(os.path.join(save_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        results[pose_name] = summary
        print(
            f"[{pose_name}] {summary['successes']}/{summary['episodes']} "
            f"= {summary['rate']:.3f}"
        )

    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
