"""Agentview evaluator for the `baseline` (B1 + DP) method.

The original ``Evaluator`` evaluates on the train/test pose files and executes
the full action chunk.  The baseline has two extra requirements:

* fixed **agentview**, K=1, including a real Plucker map (the original
  evaluator emits zero Plucker for ``default_cam``),
* a two-frame observation buffer and a 4-step EEF history, updated at every
  environment step so each replan sees the last two consecutive sim steps.

The class subclasses ``Evaluator`` and overrides only ``evaluate``; ``eval.py``
is untouched.  It restores the default agentview before every episode because
the training dataset mutates the same MuJoCo camera fields.
"""
from __future__ import annotations

import os
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch

from baseline_common import (
    EEF_HIST_STEPS,
    N_ACTION_STEPS,
    N_OBS_STEPS,
    cam_c2w_to_pos_quat,
    eef_hist_to_cam,
    eef_state,
    find_eef_body_name,
    find_eef_site_id,
    find_gripper_qpos_indices,
)
from eval import Evaluator, to_mp4


class BaselineEvaluator(Evaluator):
    """Fixed-agentview rollout with B1's observation/history buffers."""

    def __init__(self, env, norm_stats, dataset_path, args):
        super().__init__(
            env=env, norm_stats=norm_stats, dataset_path=dataset_path, args=args
        )
        self.device = torch.device("cuda")
        self.num_side_cam = 1

        self.eef_site_id = find_eef_site_id(self.env)
        self.eef_body_name = find_eef_body_name(self.env)
        self.gripper_qpos_idx = find_gripper_qpos_indices(self.env)

        self.agentview_cam_id = self.env.sim.model.camera_name2id("agentview")
        self.agentview_cam_pos = self.env.sim.model.cam_pos[
            self.agentview_cam_id
        ].copy()
        self.agentview_cam_quat = self.env.sim.model.cam_quat[
            self.agentview_cam_id
        ].copy()
        self.agentview_c2w = self._c2w_from_pos_quat(
            self.agentview_cam_pos, self.agentview_cam_quat
        )
        self.agentview_plucker = self._compute_plucker(self.agentview_c2w)

        self.eef_mean = self.norm_stats["eef_mean"].cpu().numpy()
        self.eef_std = self.norm_stats["eef_std"].cpu().numpy()
        self.action_mean = self.norm_stats["action_mean"].cpu().numpy()
        self.action_std = self.norm_stats["action_std"].cpu().numpy()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _c2w_from_pos_quat(pos, quat_wxyz):
        from baseline_common import c2w_from_pos_quat

        return c2w_from_pos_quat(pos, quat_wxyz)

    def _compute_plucker(self, c2w):
        k_t = torch.from_numpy(self.intrinsics).unsqueeze(0).float().to(self.device)
        t_t = torch.from_numpy(c2w).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            plucker = self.plucker_embedder(k_t, t_t)["plucker"][0]
        return plucker.permute(2, 0, 1).contiguous()

    def _restore_agentview(self):
        self.env.sim.model.cam_pos[self.agentview_cam_id] = self.agentview_cam_pos
        self.env.sim.model.cam_quat[self.agentview_cam_id] = self.agentview_cam_quat
        self.env.sim.forward()

    # ------------------------------------------------------------- observations
    def _current_eef_raw(self):
        pos = self.env.sim.data.site_xpos[self.eef_site_id].copy()
        # Match training: robot0_eef_quat is the EEF *body* quaternion
        # (MuJoCo order wxyz), not the gripper-site orientation.
        quat_wxyz = self.env.sim.data.get_body_xquat(self.eef_body_name).copy()
        grip = self.env.sim.data.qpos[self.gripper_qpos_idx].copy()
        return pos, quat_wxyz, grip

    def _eef_history(self, eef_deque, cam_pos, cam_quat_wxyz):
        steps = [
            eef_hist_to_cam(pos, quat, grip, cam_pos, cam_quat_wxyz)
            for pos, quat, grip in eef_deque
        ]
        return np.concatenate(steps, axis=0).astype(np.float32)

    def _snapshot(self, eef_deque):
        """Render agentview and build one B1 observation frame."""
        rgb = self._render_cam_image(None)
        rgb_t = (
            torch.from_numpy(rgb).float().permute(2, 0, 1).to(self.device) / 255.0
        )
        image = torch.cat([rgb_t, self.agentview_plucker], dim=0)  # (9, H, W)

        cam_pos, cam_quat_wxyz = cam_c2w_to_pos_quat(self.agentview_c2w)
        hist = self._eef_history(eef_deque, cam_pos, cam_quat_wxyz)

        pos, quat_wxyz, grip = self._current_eef_raw()
        eef = (eef_state(pos, quat_wxyz, grip) - self.eef_mean) / self.eef_std

        return {
            "rgb": rgb,
            "image": image.unsqueeze(0),  # (1, 9, H, W)
            "eef_state": torch.from_numpy(eef.astype(np.float32)).to(self.device),
            "eef_hist": torch.from_numpy(hist).to(self.device).unsqueeze(0),
            "view_mask": torch.ones(1, dtype=torch.bool, device=self.device),
        }

    # ----------------------------------------------------------------- rollout
    def evaluate(
        self,
        policy,
        save_path,
        video_prefix,
        pose_name,
        init_state=None,
        episode_num=0,
    ):
        np.random.seed(episode_num)
        self._restore_agentview()
        if init_state is not None:
            self.env.reset()
            self.env.sim.set_state_from_flattened(init_state)
        else:
            self.env.reset()
        self._restore_agentview()

        if self.action_space in ("eef_delta", "joint_delta"):
            self.env.set_init_action()

        pos, quat_wxyz, grip = self._current_eef_raw()
        eef_deque = deque(
            [(pos, quat_wxyz, grip)] * EEF_HIST_STEPS, maxlen=EEF_HIST_STEPS
        )
        snap = self._snapshot(eef_deque)
        snapshots = deque([snap, snap], maxlen=N_OBS_STEPS)

        camera_frames = [snap["rgb"]]
        success_labels = [False]
        rewards, success = [], []
        done = False
        step = 0
        has_succeeded = False

        while not done and step < self.max_steps:
            image = torch.stack(
                [s["image"] for s in snapshots], dim=0
            ).unsqueeze(0)  # (1, 2, 1, 9, H, W)
            eef_state_t = torch.stack(
                [s["eef_state"] for s in snapshots], dim=0
            ).unsqueeze(0)  # (1, 2, 9)
            eef_hist_t = torch.stack(
                [s["eef_hist"] for s in snapshots], dim=0
            ).unsqueeze(0)  # (1, 2, 1, 44)
            view_mask_t = torch.stack(
                [s["view_mask"] for s in snapshots], dim=0
            ).unsqueeze(0)  # (1, 2, 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed((episode_num + 1) * 100003 + step)

            with torch.no_grad(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.args.use_fp16
                else nullcontext()
            ):
                action_chunk = policy(
                    {
                        "image": image,
                        "eef_state": eef_state_t,
                        "eef_hist": eef_hist_t,
                        "view_mask": view_mask_t,
                    },
                    generator=generator,
                )
            action_chunk = (
                action_chunk[0].float().cpu().numpy() * self.action_std
                + self.action_mean
            )

            for i in range(action_chunk.shape[0]):
                if done or step >= self.max_steps:
                    break

                _, reward, done, _ = self.env.step(action_chunk[i])
                current_success = reward == 1
                has_succeeded = has_succeeded or current_success
                rewards.append(float(reward))
                success.append(current_success)
                step += 1

                pos, quat_wxyz, grip = self._current_eef_raw()
                eef_deque.append((pos, quat_wxyz, grip))
                snap = self._snapshot(eef_deque)
                snapshots.append(snap)
                camera_frames.append(snap["rgb"])
                success_labels.append(has_succeeded)

        final_success = any(success)
        print(f"Episode {episode_num}: Success = {final_success}")
        self.success_by_seed[episode_num] = bool(final_success)

        if episode_num < self.eval_save_n_video:
            video_path = os.path.join(
                save_path,
                f"{video_prefix}_{pose_name}_success_{final_success}.mp4",
            )
            to_mp4(video_path, camera_frames, success_list=success_labels)

        results = {
            "success_rate": float(final_success),
            "mean_episode_length": float(step),
            "max_rewards": rewards,
        }
        return results, float(final_success), step
