"""Dataset and loader for the `baseline` (B1 + DP) method.

The class subclasses CamPose's ``EpisodicDataset`` so the camera-pose JSONs,
the n/m window sampling, the MuJoCo rendering and the Plucker embedder are all
reused unchanged.  Only the per-sample construction differs:

* one observation frame at ``t`` (DP ``n_obs_steps=1``),
* a 4-step camera-frame EEF history per view (past-only: every index is
  ``<= t``; no future frame enters the encoder),
* a 7-D joint-position proprio (the first 7 dims of the flattened MuJoCo
  state) zeroed with ``prob_drop_proprio`` before normalisation, matching the
  original DP,
* a fixed 16-step action window starting at ``t``.

No original file is modified.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_common import (
    EEF_HIST_STEPS,
    HORIZON,
    IMAGE_SIZE,
    PROPRIO_DIM,
    action_window_indices,
    apply_shared_crop,
    cam_c2w_to_pos_quat,
    c2w_from_pos_quat,
    eef_hist_array,
    eef_hist_to_cam,
    find_eef_body_name,
    find_eef_site_id,
    find_gripper_qpos_indices,
    history_indices,
    normalize_proprio,
    quat_xyzw_to_wxyz,
    quat_wxyz_to_mat,
    shared_crop_params,
)
from cam_embedding import PluckerEmbedder
from utils import EpisodicDataset, get_norm_stats


def _eef_from_states(states, env):
    """Fallback: recover EEF pos/quat/grip by replaying flattened states."""
    site_id = find_eef_site_id(env)
    body_name = find_eef_body_name(env)
    grip_idx = find_gripper_qpos_indices(env)
    pos, quat, grip = [], [], []
    for state in states:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        pos.append(env.sim.data.site_xpos[site_id].copy())
        # The HDF5 ``robot0_eef_quat`` comes from the EEF body (wxyz in
        # MuJoCo), not from the gripper site.
        quat.append(env.sim.data.get_body_xquat(body_name).copy())
        grip.append(env.sim.data.qpos[grip_idx].copy())
    return (
        np.stack(pos).astype(np.float32),
        np.stack(quat).astype(np.float32),
        np.stack(grip).astype(np.float32),
    )


def _read_eef_obs(dataset_file, demo_idx, states, env=None):
    """Read the three EEF streams for one demo, with a simulator fallback."""
    group_path = f"data/demo_{demo_idx}/obs"
    if group_path in dataset_file:
        group = dataset_file[group_path]
        if all(k in group for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")):
            pos = group["robot0_eef_pos"][()].astype(np.float32)
            quat_xyzw = group["robot0_eef_quat"][()].astype(np.float32)
            grip = group["robot0_gripper_qpos"][()].astype(np.float32)
            return pos, quat_xyzw_to_wxyz(quat_xyzw), grip
    if env is None:
        raise RuntimeError(
            f"{group_path} lacks EEF observations and no env was provided "
            "for the forward-kinematics fallback"
        )
    return _eef_from_states(states, env)


class BaselineEpisodicDataset(EpisodicDataset):
    """Episodic dataset returning B1's single-frame / history / mask tensors."""

    def __init__(
        self,
        demo_indices,
        norm_stats,
        args,
        camera_poses_file=None,
        transform="crop",
        env=None,
    ):
        if transform not in ("crop", "id"):
            raise ValueError(
                "baseline supports transform='crop' or 'id'; "
                f"got {transform!r}"
            )
        super().__init__(
            demo_indices,
            norm_stats,
            args,
            camera_poses_file=camera_poses_file,
            transform=transform,
            env=env,
        )
        self.transform = transform
        self.num_cameras = int(args.num_side_cam)
        self.use_plucker = True
        if self.plucker_embedder is None:
            self.plucker_embedder = PluckerEmbedder(img_size=IMAGE_SIZE, device="cuda")

        self.state_mean = np.asarray(norm_stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(norm_stats["state_std"], dtype=np.float32)
        for key in ("eef_hist_mean", "eef_hist_std"):
            if key not in norm_stats:
                raise KeyError(
                    f"{key} missing from norm_stats; recompute the baseline "
                    "statistics with compute_baseline_stats()"
                )
        self.eef_hist_mean = np.asarray(
            norm_stats["eef_hist_mean"], dtype=np.float32
        )
        self.eef_hist_std = np.asarray(
            norm_stats["eef_hist_std"], dtype=np.float32
        )
        self.action_mean = np.asarray(norm_stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(norm_stats["action_std"], dtype=np.float32)

        self.demo_eef_pos = []
        self.demo_eef_quat_wxyz = []
        self.demo_grip = []
        self._load_eef_arrays()
        self._report_camera_coverage()

    def _report_camera_coverage(self):
        """Warn when the n/m windows leave most of the pose set unused."""
        if self.args.default_cam:
            return
        covered = min(
            len(self.camera_poses),
            self.args.m * (len(self.demo_indices) - 1) + self.args.n,
        )
        if covered < len(self.camera_poses):
            print(
                f"[baseline] camera windows cover {covered} of "
                f"{len(self.camera_poses)} poses "
                f"(n={self.args.n}, m={self.args.m}, "
                f"demos={len(self.demo_indices)}); use --n/--m to widen coverage"
            )

    def __del__(self):
        # ``self.env`` is shared with the sibling dataset and the evaluator.
        # ``EpisodicDataset.__del__`` closes it on garbage collection; here the
        # trainer owns the env and closes it explicitly.
        pass

    def _load_eef_arrays(self):
        with h5py.File(self.args.dataset_path, "r") as f:
            for i, demo_idx in enumerate(self.demo_indices):
                demo_key = f"demo_{demo_idx}"
                if f"data/{demo_key}" not in f:
                    continue
                pos, quat, grip = _read_eef_obs(
                    f, demo_idx, self.demo_states[i], self.env
                )
                expected = len(self.demo_actions[i])
                if not (len(pos) == len(quat) == len(grip) == expected):
                    raise RuntimeError(
                        f"{demo_key}: EEF lengths {(len(pos), len(quat), len(grip))} "
                        f"do not match actions ({expected})"
                    )
                self.demo_eef_pos.append(pos)
                self.demo_eef_quat_wxyz.append(quat)
                self.demo_grip.append(grip)
        if len(self.demo_eef_pos) != len(self.demo_states):
            raise RuntimeError("EEF arrays and demo states are misaligned")

    def _slot_c2w(self, pose_raw):
        if pose_raw is not None:
            return np.asarray(pose_raw, dtype=np.float32)
        cam_id = self.env.sim.model.camera_name2id("agentview")
        pos = self.env.sim.model.cam_pos[cam_id].copy()
        quat_wxyz = self.env.sim.model.cam_quat[cam_id].copy()
        return c2w_from_pos_quat(pos, quat_wxyz)

    def _render_slot_pair(self, states, obs_indices, pose_raw):
        """Render the observation frame(s) for one view with one camera pose."""
        if pose_raw is not None:
            self._set_camera_pose(np.asarray(pose_raw, dtype=np.float32))
        c2w = self._slot_c2w(pose_raw)
        intrinsics = self._get_camera_intrinsics()
        k_t = torch.from_numpy(intrinsics).unsqueeze(0).float().cuda()
        t_t = torch.from_numpy(c2w).unsqueeze(0).float().cuda()
        with torch.no_grad():
            plucker = self.plucker_embedder(k_t, t_t)["plucker"][0]
        # (256, 256, 6) -> (6, 256, 256); static for both frames.
        plucker = plucker.permute(2, 0, 1).contiguous()

        images = []
        for frame_idx in obs_indices:
            self.env.sim.set_state_from_flattened(states[frame_idx])
            self.env.sim.forward()
            rgb = self.env.sim.render(
                camera_name="agentview",
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                depth=False,
            )
            rgb = np.flipud(rgb).copy()
            rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            rgb_t = rgb_t.cuda()
            images.append(torch.cat([rgb_t, plucker], dim=0))

        if self.transform == "id":
            return torch.stack(images, dim=0), c2w
        params = shared_crop_params()
        cropped = [apply_shared_crop(img, params) for img in images]
        return torch.stack(cropped, dim=0), c2w

    def __getitem__(self, demo_idx):
        demo_length = self.demo_lengths[demo_idx]
        states = self.demo_states[demo_idx]
        actions = self.demo_actions[demo_idx]
        t = int(np.random.randint(demo_length))
        # DP n_obs_steps=1: a single observation frame at t.  The action
        # window below starts at t, so it is strictly future relative to it.
        obs_indices = np.array([t], dtype=np.int64)

        if not self.args.default_cam:
            # ``demo_idx`` is positional (the DataLoader index).  Training
            # indices are 0..N-1, so this equals the global demo index; the
            # val split reuses the start of the (smaller) test pose file
            # instead of overflowing it.
            start = self.args.m * demo_idx
            end = start + self.args.n
            window = np.arange(start, end, dtype=np.int64)
            chosen = np.random.choice(
                window, size=self.num_cameras, replace=False
            )
            pose_set = [self.camera_poses[i] for i in chosen.tolist()]
        else:
            pose_set = [None] * self.num_cameras

        # prob_drop_proprio mirrors the original DP: zero the raw qpos before
        # the min-max normalisation.  One draw per sample (n_obs_steps=1).
        drop_proprio = np.random.rand() < self.args.prob_drop_proprio
        proprio = normalize_proprio(
            states[t][:PROPRIO_DIM],
            self.state_mean,
            self.state_std,
            drop_proprio,
        )

        slot_images = []
        slot_hists = []
        for pose_raw in pose_set:
            images, c2w = self._render_slot_pair(states, obs_indices, pose_raw)
            slot_images.append(images)
            cam_pos, cam_quat_wxyz = cam_c2w_to_pos_quat(c2w)

            per_frame_hist = []
            for frame_idx in obs_indices:
                hist_steps = [
                    eef_hist_to_cam(
                        self.demo_eef_pos[demo_idx][step_idx],
                        self.demo_eef_quat_wxyz[demo_idx][step_idx],
                        self.demo_grip[demo_idx][step_idx],
                        cam_pos,
                        cam_quat_wxyz,
                    )
                    for step_idx in history_indices(frame_idx, EEF_HIST_STEPS)
                ]
                per_frame_hist.append(np.concatenate(hist_steps, axis=0))
            slot_hists.append(np.stack(per_frame_hist, axis=0))

        image = torch.stack(slot_images, dim=1)  # (1, K, 9, H, W)
        proprio_t = torch.from_numpy(proprio).float()[None]  # (1, 7)
        eef_hist_t = torch.tensor(
            (np.stack(slot_hists, axis=1) - self.eef_hist_mean)
            / self.eef_hist_std,
            dtype=torch.float32,
        )  # (1, K, 44), z-scored with the training history stats
        view_mask = torch.ones(1, self.num_cameras, dtype=torch.bool)

        safe_idx, valid = action_window_indices(t, HORIZON, len(actions))
        act = np.zeros((HORIZON, actions.shape[1]), dtype=np.float32)
        act[valid] = actions[safe_idx[valid]]
        is_pad = ~valid
        act = (act - self.action_mean) / self.action_std

        return {
            "image": image.cuda(),
            "proprio": proprio_t.cuda(),
            "eef_hist": eef_hist_t.cuda(),
            "view_mask": view_mask.cuda(),
            "actions": torch.from_numpy(act).float().cuda(),
            "is_pad": torch.from_numpy(is_pad).cuda(),
        }


def _load_train_camera_poses(args, env):
    """Camera poses the training window sampler draws from."""
    if args.default_cam:
        if env is None:
            raise RuntimeError(
                "default_cam=True needs env to read the agentview pose"
            )
        cam_id = env.sim.model.camera_name2id("agentview")
        pos = env.sim.model.cam_pos[cam_id].copy()
        quat = env.sim.model.cam_quat[cam_id].copy()
        return [c2w_from_pos_quat(pos, quat)]
    poses_path = os.path.join(args.camera_poses_dir, args.train_poses_file)
    with open(poses_path, "r") as f:
        return json.load(f)["poses"]


def compute_baseline_stats(dataset_path, num_demos, args, env=None):
    """CamPose DP action/state stats plus 44-D history mean/std.

    ``state_mean/std`` are the DP min-max statistics over the first 7 joint
    dims (``get_norm_stats``), used for the 7-D proprio.  The history
    statistics are computed over every demo and every camera pose in that
    demo's training window, i.e. the marginal distribution the dataset samples
    from.
    """
    stats = get_norm_stats(dataset_path, num_demos=num_demos, policy_class="dp")
    poses = _load_train_camera_poses(args, env)
    hist_sum = None
    hist_sumsq = None
    hist_count = 0
    with h5py.File(dataset_path, "r") as f:
        demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
        count = min(int(num_demos), len(demo_keys))
        if count == 0:
            raise RuntimeError("no demos found for baselines statistics")
        for i in range(count):
            states = f[f"data/demo_{i}/states"][()].astype(np.float32)
            pos, quat, grip = _read_eef_obs(f, i, states, env)

            if args.default_cam:
                pose_window = poses[:1]
            else:
                start = args.m * i
                end = start + args.n
                if end > len(poses):
                    raise IndexError(
                        f"demo_{i}: camera window [{start}, {end}) exceeds "
                        f"the {len(poses)} poses in {args.train_poses_file}"
                    )
                pose_window = poses[start:end]
            for pose in pose_window:
                cam_pos, cam_quat = cam_c2w_to_pos_quat(np.asarray(pose))
                hist = eef_hist_array(pos, quat, grip, cam_pos, cam_quat)
                if hist_sum is None:
                    hist_sum = np.zeros(hist.shape[1], dtype=np.float64)
                    hist_sumsq = np.zeros(hist.shape[1], dtype=np.float64)
                hist_sum += hist.sum(axis=0)
                hist_sumsq += np.square(hist).sum(axis=0)
                hist_count += hist.shape[0]
    if hist_count == 0:
        raise RuntimeError("no camera poses found for history statistics")
    hist_mean = hist_sum / hist_count
    hist_var = hist_sumsq / hist_count - np.square(hist_mean)
    stats["eef_hist_mean"] = hist_mean.astype(np.float32)
    stats["eef_hist_std"] = np.clip(
        np.sqrt(np.clip(hist_var, 0.0, None)), 1e-6, np.inf
    ).astype(np.float32)
    return stats


def load_baseline_data(args, env, val_split=0.1):
    """Mirror ``utils.load_data`` but build ``BaselineEpisodicDataset``."""
    with h5py.File(args.dataset_path, "r") as f:
        available_demos = len(
            [k for k in f["data"].keys() if k.startswith("demo_")]
        )
    assert args.num_episodes + 10 <= available_demos, "Not enough demos to split"
    print(
        f"Using {args.num_episodes} of {available_demos} demos for training "
        f"(+10 for validation)"
    )

    train_indices = list(range(args.num_episodes))
    val_indices = list(range(args.num_episodes, args.num_episodes + 10))

    print("Computing baseline normalization statistics...")
    norm_stats = compute_baseline_stats(
        args.dataset_path, num_demos=args.num_episodes, args=args, env=env
    )
    print("Normalization statistics computed.")

    train_dataset = BaselineEpisodicDataset(
        train_indices,
        norm_stats,
        args,
        camera_poses_file=args.train_poses_file,
        transform=args.transform,
        env=env,
    )
    val_dataset = BaselineEpisodicDataset(
        val_indices,
        norm_stats,
        args,
        camera_poses_file=args.test_poses_file,
        transform="id",
        env=env,
    )

    max_seq_length = train_dataset.max_seq_length
    train_dataset.max_seq_length = max_seq_length
    val_dataset.max_seq_length = max_seq_length

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_dataloader, val_dataloader, norm_stats
