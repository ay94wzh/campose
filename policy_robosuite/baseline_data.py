"""Dataset and loader for the `baseline` (B1 + DP) method.

The class subclasses CamPose's ``EpisodicDataset`` so the camera-pose JSONs,
the n/m window sampling, the MuJoCo rendering and the Plucker embedder are all
reused unchanged.  Only the per-sample construction differs:

* two consecutive observation frames (``t-1``, ``t``) with one shared camera
  pose and one shared random crop,
* a 4-step camera-frame EEF history per view,
* a 9-D EEF low-dim state,
* a fixed 16-step action window starting at ``t-1``.

No original file is modified.
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_common import (
    EEF_HIST_STEPS,
    HORIZON,
    IMAGE_SIZE,
    action_window_indices,
    apply_shared_crop,
    cam_c2w_to_pos_quat,
    c2w_from_pos_quat,
    eef_hist_to_cam,
    eef_state,
    find_eef_body_name,
    find_eef_site_id,
    find_gripper_qpos_indices,
    history_indices,
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
    """Episodic dataset returning B1's two-frame / history / mask tensors."""

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

        self.eef_mean = np.asarray(norm_stats["eef_mean"], dtype=np.float32)
        self.eef_std = np.asarray(norm_stats["eef_std"], dtype=np.float32)
        self.action_mean = np.asarray(norm_stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(norm_stats["action_std"], dtype=np.float32)

        self.demo_eef_pos = []
        self.demo_eef_quat_wxyz = []
        self.demo_grip = []
        self._load_eef_arrays()

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
        """Render both observation frames for one view with one camera pose."""
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
        obs_indices = np.array([max(t - 1, 0), t], dtype=np.int64)

        if not self.args.default_cam:
            start = self.args.m * demo_idx
            end = start + self.args.n
            window = np.arange(start, end, dtype=np.int64)
            chosen = np.random.choice(
                window, size=self.num_cameras, replace=False
            )
            pose_set = [self.camera_poses[i] for i in chosen.tolist()]
        else:
            pose_set = [None] * self.num_cameras

        eef_states = []
        for frame_idx in obs_indices:
            raw = eef_state(
                self.demo_eef_pos[demo_idx][frame_idx],
                self.demo_eef_quat_wxyz[demo_idx][frame_idx],
                self.demo_grip[demo_idx][frame_idx],
            )
            eef_states.append((raw - self.eef_mean) / self.eef_std)

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

        image = torch.stack(slot_images, dim=1)  # (2, K, 9, H, W)
        eef_state_t = torch.tensor(
            np.stack(eef_states, axis=0), dtype=torch.float32
        )  # (2, 9)
        eef_hist_t = torch.tensor(
            np.stack(slot_hists, axis=1), dtype=torch.float32
        )  # (2, K, 44)
        view_mask = torch.ones(2, self.num_cameras, dtype=torch.bool)

        safe_idx, valid = action_window_indices(t, HORIZON, len(actions))
        act = np.zeros((HORIZON, actions.shape[1]), dtype=np.float32)
        act[valid] = actions[safe_idx[valid]]
        is_pad = ~valid
        act = (act - self.action_mean) / self.action_std

        return {
            "image": image.cuda(),
            "eef_state": eef_state_t.cuda(),
            "eef_hist": eef_hist_t.cuda(),
            "view_mask": view_mask.cuda(),
            "actions": torch.from_numpy(act).float().cuda(),
            "is_pad": torch.from_numpy(is_pad).cuda(),
        }


def compute_baseline_stats(dataset_path, num_demos, env=None):
    """CamPose DP action stats plus 9-D EEF mean/std."""
    stats = get_norm_stats(dataset_path, num_demos=num_demos, policy_class="dp")
    eef_rows = []
    with h5py.File(dataset_path, "r") as f:
        demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
        count = min(int(num_demos), len(demo_keys))
        for i in range(count):
            states = f[f"data/demo_{i}/states"][()].astype(np.float32)
            pos, quat, grip = _read_eef_obs(f, i, states, env)
            eef_rows.append(eef_state(pos, quat, grip))
    if not eef_rows:
        raise RuntimeError("no demos found for baselines statistics")
    eef = np.concatenate(eef_rows, axis=0)
    stats["eef_mean"] = eef.mean(axis=0).astype(np.float32)
    stats["eef_std"] = np.clip(eef.std(axis=0), 1e-4, np.inf).astype(np.float32)
    return stats


def load_baseline_data(args, env, val_split=0.1):
    """Mirror ``utils.load_data`` but build ``BaselineEpisodicDataset``."""
    with h5py.File(args.dataset_path, "r") as f:
        available_demos = len(
            [k for k in f["data"].keys() if k.startswith("demo_")]
        )
    assert args.num_episodes + 10 <= available_demos, "Not enough demos to split"

    train_indices = list(range(args.num_episodes))
    val_indices = list(range(args.num_episodes, args.num_episodes + 10))

    print("Computing baseline normalization statistics...")
    norm_stats = compute_baseline_stats(
        args.dataset_path, num_demos=args.num_episodes, env=env
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
