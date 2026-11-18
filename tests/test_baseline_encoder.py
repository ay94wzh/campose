"""CPU-only tests for the `baseline` (B1 + DP) implementation.

Run from the repo root (no GPU, simulator, or demos required):

    /home/zihan-wang/anaconda3/envs/know_your_camera/bin/python \
        tests/test_baseline_encoder.py

The encoder/geometry tests import no robosuite and no diffusers.  The policy
shape test is skipped if diffusers is unavailable.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_ROOT = os.path.join(REPO_ROOT, "policy_robosuite")
if POLICY_ROOT not in sys.path:
    sys.path.insert(0, POLICY_ROOT)

from baseline_common import (  # noqa: E402
    BASELINE_ARCH,
    BASELINE_ARCH_AVGPOOL,
    BASELINE_RECON_TAG,
    EEF_HIST_STEPS,
    OBS_FEATURE_DIM,
    PROPRIO_DIM,
    RECON_MODES,
    TARGET_POS_JITTER,
    TARGET_ROT_JITTER_DEG,
    SpatialSoftmax,
    action_window_indices,
    apply_shared_crop,
    baseline_arch_name,
    cam_c2w_to_pos_quat,
    eef_hist_array,
    eef_hist_to_cam,
    history_indices,
    mat_to_rot6d,
    normalize_proprio,
    rot6d_to_mat,
    sample_target_pose,
    select_eval_pose,
    shared_crop_params,
    target_pose_rng,
)
from models.baseline import BaselineDiffusionPolicy, BaselineEncoder  # noqa: E402
from models.baseline_recon import (  # noqa: E402
    PluckerPyramidEncoder,
    TargetViewReconstructor,
)


def _make_inputs(n_frames, k, seed=0):
    torch.manual_seed(seed)
    image = torch.rand(n_frames, k, 9, 256, 256)
    proprio = torch.randn(n_frames, PROPRIO_DIM)
    eef_hist = torch.randn(n_frames, k, EEF_HIST_STEPS * 11)
    view_mask = torch.ones(n_frames, k, dtype=torch.bool)
    return image, proprio, eef_hist, view_mask


def test_common_geometry():
    pos, quat = cam_c2w_to_pos_quat(np.eye(4, dtype=np.float32))
    assert np.allclose(pos, 0.0)
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])

    step = eef_hist_to_cam(
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    )
    expected = np.array([1, 2, 3, 1, 0, 0, 0, 1, 0, 0.1, 0.2], dtype=np.float64)
    assert np.allclose(step, expected), step

    R = Rotation.random(random_state=0).as_matrix()
    assert np.allclose(rot6d_to_mat(mat_to_rot6d(R)), R, atol=1e-10)
    print("ok common geometry")


def test_eef_hist_array_matches_loop():
    rng = np.random.RandomState(0)
    T = 12
    pos = rng.randn(T, 3)
    quat = rng.randn(T, 4)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    grip = rng.randn(T, 2)
    cam_pos = rng.randn(3)
    cam_quat = rng.randn(4)
    cam_quat /= np.linalg.norm(cam_quat)

    hist = eef_hist_array(pos, quat, grip, cam_pos, cam_quat)
    assert hist.shape == (T, EEF_HIST_STEPS * 11), hist.shape
    for t in range(T):
        loop = np.concatenate(
            [
                eef_hist_to_cam(pos[i], quat[i], grip[i], cam_pos, cam_quat)
                for i in history_indices(t, EEF_HIST_STEPS)
            ]
        )
        assert np.allclose(hist[t], loop, atol=1e-10), (
            t,
            np.abs(hist[t] - loop).max(),
        )
    print("ok vectorised eef history")


def test_history_is_past_only():
    rng = np.random.RandomState(1)
    T = 10
    pos = rng.randn(T, 3)
    quat = rng.randn(T, 4)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    grip = rng.randn(T, 2)
    cam_pos = rng.randn(3)
    cam_quat = rng.randn(4)
    cam_quat /= np.linalg.norm(cam_quat)

    hist = eef_hist_array(pos, quat, grip, cam_pos, cam_quat)
    for t in range(T):
        assert np.all(history_indices(t, EEF_HIST_STEPS) <= t)
        pos2, quat2, grip2 = pos.copy(), quat.copy(), grip.copy()
        pos2[t + 1:] += 100.0
        quat2[t + 1:] += 5.0
        grip2[t + 1:] += 100.0
        hist2 = eef_hist_array(pos2, quat2, grip2, cam_pos, cam_quat)
        assert np.array_equal(hist[t], hist2[t]), t
    print("ok past-only history")


def test_index_helpers():
    assert np.array_equal(history_indices(0), [0, 0, 0, 0])
    assert np.array_equal(history_indices(5), [2, 3, 4, 5])

    idx, valid = action_window_indices(0, 16, 100)
    assert valid.all()
    assert idx[0] == 0

    idx, valid = action_window_indices(5, 16, 100)
    assert valid.all()
    assert idx[0] == 5

    _, valid = action_window_indices(99, 16, 100)
    assert valid.sum() == 1
    print("ok index helpers")


def test_shared_crop():
    x = torch.rand(9, 256, 256)
    params = shared_crop_params()
    y1 = apply_shared_crop(x, params)
    y2 = apply_shared_crop(x, params)
    assert y1.shape == (9, 256, 256)
    assert torch.equal(y1, y2)
    print("ok shared crop")


def test_spatial_softmax_operator():
    # num_kp=None keeps each input channel as a keypoint, so the test is
    # deterministic: a softmax peak at a grid cell yields that cell's (x, y).
    op = SpatialSoftmax((1, 8, 8), num_kp=None)
    a = torch.zeros(1, 1, 8, 8)
    b = torch.zeros(1, 1, 8, 8)
    a[0, 0, 1, 1] = 10.0
    b[0, 0, 6, 6] = 10.0
    ka = op(a).flatten()
    kb = op(b).flatten()
    assert ka.shape == (2,)
    assert abs(ka[0].item() + 0.714) < 1e-2
    assert abs(ka[1].item() + 0.714) < 1e-2
    assert abs(kb[0].item() - 0.714) < 1e-2
    assert abs(kb[1].item() - 0.714) < 1e-2

    pooled = SpatialSoftmax((4, 8, 8), num_kp=3)(torch.randn(2, 4, 8, 8))
    assert pooled.shape == (2, 3, 2)
    print("ok spatial softmax")


def test_proprio_drop_before_zscore():
    raw = np.array([0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7], dtype=np.float32)
    mean = np.full(PROPRIO_DIM, 0.5, dtype=np.float32)
    std = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    kept = normalize_proprio(raw, mean, std, drop=False)
    dropped = normalize_proprio(raw, mean, std, drop=True)
    assert np.allclose(kept, (raw - mean) / std)
    # DP zeroes the raw qpos *before* normalisation, so the constant is
    # -state_mean/state_std, not zero.
    assert np.allclose(dropped, -mean / std)
    assert not np.allclose(dropped, np.zeros_like(raw))
    print("ok proprio drop before z-score")


def test_encoder_shapes():
    for pool in ("spatial_softmax", "avgpool"):
        for k in (1, 2):
            image, proprio, eef_hist, view_mask = _make_inputs(2, k)
            model = BaselineEncoder(pool=pool).eval()
            with torch.no_grad():
                out = model(image, proprio, eef_hist, view_mask)
                full = model.forward_full(image, proprio, eef_hist, view_mask)
            assert out.shape == (2, OBS_FEATURE_DIM), out.shape
            assert full["z_global"].shape == (2, 512)
            assert full["z_views"].shape == (2, k, 512)
            assert torch.isfinite(out).all()
            print(f"ok encoder shapes pool={pool} k={k}")


def test_pooling_variants():
    spatial = BaselineEncoder().eval()
    assert hasattr(spatial, "spatial_pool")
    assert hasattr(spatial, "spatial_proj")
    assert not hasattr(spatial, "avgpool")

    avg = BaselineEncoder(pool="avgpool").eval()
    assert isinstance(avg.avgpool, torch.nn.AdaptiveAvgPool2d)
    assert not hasattr(avg, "spatial_pool")
    assert not hasattr(avg, "spatial_proj")

    image, proprio, eef_hist, view_mask = _make_inputs(2, 1)
    with torch.no_grad():
        full = avg.forward_full(image, proprio, eef_hist, view_mask)
    assert full["z_global"].shape == (2, 512)
    assert full["out"].shape == (2, OBS_FEATURE_DIM)
    assert torch.isfinite(full["out"]).all()

    try:
        BaselineEncoder(pool="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pooling accepted")
    print("ok pooling variants")


def test_arch_markers():
    assert baseline_arch_name("spatial_softmax") == BASELINE_ARCH
    assert baseline_arch_name("avgpool") == BASELINE_ARCH_AVGPOOL
    try:
        baseline_arch_name("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pooling marker accepted")
    print("ok arch markers")


def test_select_eval_pose():
    agent = np.eye(4, dtype=np.float32)
    p0 = np.eye(4, dtype=np.float32)
    p1 = np.eye(4, dtype=np.float32)
    p1[:3, 3] = [1.0, 2.0, 3.0]
    poses = {"train_cameras": [p0, p1]}

    assert np.allclose(select_eval_pose("train_cameras", 0, poses, agent), p0)
    assert np.allclose(select_eval_pose("train_cameras", 1, poses, agent), p1)
    assert np.allclose(select_eval_pose("agentview", 99, poses, agent), agent)

    try:
        select_eval_pose("test_cameras", 0, poses, agent)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown pose set accepted")
    try:
        select_eval_pose("train_cameras", 2, poses, agent)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range pose accepted")
    print("ok eval pose selection")


def test_no_imagenet_rgb_norm():
    model = BaselineEncoder().eval()
    assert not hasattr(model, "rgb_mean")
    assert not hasattr(model, "rgb_std")
    image = torch.rand(1, 1, 9, 256, 256)
    assert torch.equal(model._prepare_image(image), image)
    print("ok no ImageNet RGB normalisation")


def test_film_identity_at_init():
    image, proprio, eef_hist, view_mask = _make_inputs(2, 1)
    model = BaselineEncoder().eval()
    with torch.no_grad():
        out_random = model(image, proprio, eef_hist, view_mask)
        out_zero = model(image, proprio, torch.zeros_like(eef_hist), view_mask)
    assert torch.allclose(out_random, out_zero, atol=1e-5), (
        (out_random - out_zero).abs().max().item()
    )
    print("ok zero-init FiLM identity")


def test_masked_view_equivalence():
    image, proprio, eef_hist, _ = _make_inputs(2, 2)
    mask_second_off = torch.tensor([[True, False], [True, False]])
    model = BaselineEncoder().eval()
    with torch.no_grad():
        out_masked = model(image, proprio, eef_hist, mask_second_off)
        out_single = model(
            image[:, :1],
            proprio,
            eef_hist[:, :1],
            torch.ones(2, 1, dtype=torch.bool),
        )
    assert torch.allclose(out_masked, out_single, atol=1e-5), (
        (out_masked - out_single).abs().max().item()
    )
    print("ok masked-view equivalence")


def test_encoder_cpu_autocast():
    image, proprio, eef_hist, view_mask = _make_inputs(1, 1)
    model = BaselineEncoder().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(image, proprio, eef_hist, view_mask)
    assert out.shape == (1, OBS_FEATURE_DIM), out.shape
    assert torch.isfinite(out.float()).all()
    print("ok encoder CPU autocast")


def test_policy_shapes():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        print("skip policy shapes: diffusers not installed")
        return

    image = torch.rand(1, 1, 1, 9, 256, 256)
    proprio = torch.randn(1, 1, PROPRIO_DIM)
    eef_hist = torch.randn(1, 1, 1, EEF_HIST_STEPS * 11)
    view_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    actions = torch.randn(1, 16, 7)
    is_pad = torch.zeros(1, 16, dtype=torch.bool)

    for pool in ("spatial_softmax", "avgpool"):
        args = SimpleNamespace(
            action_dim=7, lr=1e-4, weight_decay=1e-6, baseline_pool=pool
        )
        policy = BaselineDiffusionPolicy(
            args,
            down_dims=(32, 64),
            diffusion_step_embed_dim=32,
            num_inference_steps=2,
        ).eval()

        out = policy(
            {
                "image": image,
                "proprio": proprio,
                "eef_hist": eef_hist,
                "view_mask": view_mask,
                "actions": actions,
                "is_pad": is_pad,
            }
        )
        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])

        with torch.no_grad():
            act = policy(
                {
                    "image": image,
                    "proprio": proprio,
                    "eef_hist": eef_hist,
                    "view_mask": view_mask,
                }
            )
        assert act.shape == (1, 8, 7), act.shape
        print(f"ok policy train/inference shapes pool={pool}")


def _recon_policy(mode):
    args = SimpleNamespace(
        action_dim=7,
        lr=1e-4,
        weight_decay=1e-6,
        baseline_pool="spatial_softmax",
        recon_mode=mode,
        recon_weight=0.1,
    )
    return BaselineDiffusionPolicy(
        args,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_inference_steps=2,
    ).eval()


def _recon_batch():
    return {
        "image": torch.rand(1, 1, 1, 9, 256, 256),
        "proprio": torch.randn(1, 1, PROPRIO_DIM),
        "eef_hist": torch.randn(1, 1, 1, EEF_HIST_STEPS * 11),
        "view_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "actions": torch.randn(1, 16, 7),
        "is_pad": torch.zeros(1, 16, dtype=torch.bool),
        "target_image": torch.rand(1, 3, 128, 128),
        "target_plucker": torch.randn(1, 6, 256, 256),
    }


def test_target_pose_sampling():
    default = np.eye(4, dtype=np.float32)
    default[:3, 3] = [0.1, -0.2, 0.3]

    p1 = sample_target_pose(default, target_pose_rng(0, 0))
    p2 = sample_target_pose(default, target_pose_rng(0, 0))
    assert p1.shape == (4, 4)
    assert np.allclose(p1, p2), "target pose is not deterministic"
    assert np.allclose(p1[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(p1[:3, :3] @ p1[:3, :3].T, np.eye(3), atol=1e-5)

    delta = np.linalg.inv(default.astype(np.float64)) @ p1.astype(np.float64)
    assert np.all(np.abs(delta[:3, 3]) <= TARGET_POS_JITTER + 1e-6)
    angles = Rotation.from_matrix(delta[:3, :3]).as_euler("xyz", degrees=True)
    assert np.all(np.abs(angles) <= TARGET_ROT_JITTER_DEG + 1e-6)

    p_other = sample_target_pose(default, target_pose_rng(0, 1))
    assert not np.allclose(p1, p_other)

    p_zero = sample_target_pose(
        default, target_pose_rng(0, 0), pos_jitter=0.0, rot_jitter_deg=0.0
    )
    assert np.allclose(p_zero, default, atol=1e-6)
    print("ok target pose sampling")


def test_arch_marker_recon():
    assert set(RECON_MODES) == {
        "none",
        "z_only",
        "single_plucker",
        "multi_plucker",
    }
    assert baseline_arch_name("spatial_softmax") == BASELINE_ARCH
    assert baseline_arch_name("avgpool", "none") == BASELINE_ARCH_AVGPOOL
    for mode in ("z_only", "single_plucker", "multi_plucker"):
        marker = baseline_arch_name("spatial_softmax", mode)
        assert marker == f"{BASELINE_ARCH}{BASELINE_RECON_TAG}{mode}"
    try:
        baseline_arch_name("spatial_softmax", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid recon mode accepted")
    print("ok recon arch markers")


def test_plucker_pyramid_shapes():
    model = PluckerPyramidEncoder().eval()
    with torch.no_grad():
        feats = model(torch.randn(2, 6, 256, 256))
    expected = {
        "g4": (2, 16, 128, 128),
        "g3": (2, 32, 64, 64),
        "g2": (2, 64, 32, 32),
        "g1": (2, 64, 16, 16),
    }
    for key, shape in expected.items():
        assert feats[key].shape == shape, (key, feats[key].shape)
        assert torch.isfinite(feats[key]).all()
    print("ok Plucker pyramid shapes")


def test_recon_branch_modes():
    z = torch.randn(2, 512)
    plucker = torch.randn(2, 6, 256, 256)
    for mode in ("z_only", "single_plucker", "multi_plucker"):
        model = TargetViewReconstructor(mode=mode).eval()
        with torch.no_grad():
            out = model(z, None if mode == "z_only" else plucker)
        assert out.shape == (2, 3, 128, 128), (mode, out.shape)
        assert torch.isfinite(out).all()
        assert out.min() >= 0 and out.max() <= 1

    z_model = TargetViewReconstructor(mode="z_only").eval()
    with torch.no_grad():
        out_ignored = z_model(z, plucker)
        out_none = z_model(z, None)
    assert torch.allclose(out_ignored, out_none)

    for mode in ("single_plucker", "multi_plucker"):
        model = TargetViewReconstructor(mode=mode).eval()
        for bad in (None, torch.randn(2, 6, 128, 128)):
            try:
                with torch.no_grad():
                    model(z, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{mode} accepted a bad Plucker map")
    print("ok recon branch modes")


def test_policy_recon_loss_dict():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        print("skip recon policy loss: diffusers not installed")
        return

    for mode in ("z_only", "single_plucker", "multi_plucker"):
        policy = _recon_policy(mode)
        data = _recon_batch()
        if mode == "z_only":
            data.pop("target_plucker")
        out = policy(data)
        for key in ("loss", "loss_diffusion", "loss_recon"):
            assert key in out, (mode, out.keys())
            assert out[key].ndim == 0 and torch.isfinite(out[key])
        assert torch.allclose(
            out["loss"],
            out["loss_diffusion"] + 0.1 * out["loss_recon"],
            atol=1e-5,
        ), mode
    print("ok recon policy loss dict")


def test_policy_recon_none_ignores_targets():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        print("skip recon-none policy: diffusers not installed")
        return

    args = SimpleNamespace(
        action_dim=7,
        lr=1e-4,
        weight_decay=1e-6,
        baseline_pool="spatial_softmax",
    )
    policy = BaselineDiffusionPolicy(
        args,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_inference_steps=2,
    ).eval()
    assert not any(k.startswith("recon_branch.") for k in policy.state_dict())

    out = policy(_recon_batch())
    assert "loss_recon" not in out
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_diffusion"])
    print("ok recon none ignores targets")


def test_policy_inference_unchanged_with_recon():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        print("skip recon inference: diffusers not installed")
        return

    policy = _recon_policy("multi_plucker")

    def _boom(*args, **kwargs):
        raise AssertionError("reconstruction branch called at inference")

    policy.recon_branch.forward = _boom
    data = _recon_batch()
    for key in ("actions", "is_pad", "target_image", "target_plucker"):
        data.pop(key)
    with torch.no_grad():
        actions = policy(data)
    assert actions.shape == (1, 8, 7), actions.shape
    assert torch.isfinite(actions).all()
    print("ok recon inference unchanged")


def test_recon_cpu_autocast():
    model = TargetViewReconstructor(mode="multi_plucker").eval()
    z = torch.randn(1, 512)
    plucker = torch.randn(1, 6, 256, 256)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(z, plucker)
    assert out.shape == (1, 3, 128, 128)
    assert torch.isfinite(out.float()).all()
    print("ok recon CPU autocast")


def test_old_checkpoint_compat():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        print("skip old checkpoint compat: diffusers not installed")
        return

    args = SimpleNamespace(
        action_dim=7,
        lr=1e-4,
        weight_decay=1e-6,
        baseline_pool="spatial_softmax",
    )
    source = BaselineDiffusionPolicy(
        args,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_inference_steps=2,
    )
    target = BaselineDiffusionPolicy(
        args,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_inference_steps=2,
    )
    target.load_state_dict(source.state_dict())  # strict
    assert not any(k.startswith("recon_branch.") for k in source.state_dict())
    assert baseline_arch_name("spatial_softmax") == BASELINE_ARCH
    print("ok old checkpoint compatibility")


if __name__ == "__main__":
    assert "robosuite" not in sys.modules, "baseline tests must not import robosuite"
    test_common_geometry()
    test_eef_hist_array_matches_loop()
    test_history_is_past_only()
    test_index_helpers()
    test_shared_crop()
    test_spatial_softmax_operator()
    test_proprio_drop_before_zscore()
    test_encoder_shapes()
    test_pooling_variants()
    test_arch_markers()
    test_select_eval_pose()
    test_no_imagenet_rgb_norm()
    test_film_identity_at_init()
    test_masked_view_equivalence()
    test_encoder_cpu_autocast()
    test_policy_shapes()
    test_target_pose_sampling()
    test_arch_marker_recon()
    test_plucker_pyramid_shapes()
    test_recon_branch_modes()
    test_policy_recon_loss_dict()
    test_policy_recon_none_ignores_targets()
    test_policy_inference_unchanged_with_recon()
    test_recon_cpu_autocast()
    test_old_checkpoint_compat()
    print("all baseline tests passed")
