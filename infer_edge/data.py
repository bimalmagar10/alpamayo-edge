"""One inference window of a PhysicalAI-AV clip, as the Edge model and the scorer need it.

A window is NVIDIA's raw Edge sample (22 camera frames, 20 history poses, navigation text)
plus the ground-truth future the model never sees: 64 poses at t0 + 0.1 .. 6.4 s in the ego
frame at t0, computed exactly as alpamayo_r1's loader does, so minADE is comparable with the
flashdrive runs. Windows can be cached as one .npz each (see scripts/download_clips.py).
"""

import json
import os
from typing import Any

import numpy as np
import torch
from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
from torchvision.transforms.functional import InterpolationMode, resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

FUTURE_STEPS, STEP_US = 64, 100_000  # 6.4 s at 10 Hz: what the model predicts
MIN_T0_US = 2_000_000  # the front cameras look back 2 s
TENSORS = ("image_frames", "camera_indices", "absolute_timestamps",
           "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
TOKEN_PX = 32  # Cosmos3-Edge image processor: 16 px patches merged 2x2 -> one token per 32 px
FRONT_PIXELS = CAMERA_PIXEL_BOUNDS[1]  # (163840, 196608): the front cameras' budget, the larger


def token_px(model_dir: str | None) -> int:
    """patch_size * merge_size of the checkpoint's image processor (TOKEN_PX when unknown)."""
    if not model_dir or not os.path.isdir(model_dir):
        return TOKEN_PX
    for name, key in (("preprocessor_config.json", None), ("processor_config.json", "image_processor")):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            with open(path) as fh:
                cfg = json.load(fh)
            cfg = cfg.get(key, {}) if key else cfg
            if "patch_size" in cfg and "merge_size" in cfg:
                return int(cfg["patch_size"]) * int(cfg["merge_size"])
    return TOKEN_PX


def resize_frames(frames: torch.Tensor, factor: int = TOKEN_PX) -> torch.Tensor:
    """[22, 1, 3, H, W] uint8 -> the front cameras' smart_resize size (320 x 576 for 1080p).

    One tensor holds all 22 frames, so the four side cameras are stored at this size too; the
    processor then shrinks them to their own budget (224 x 416): two bicubic passes instead of
    one. The front cameras' pixels are bit-identical to the 1080p path (the second resize is
    the identity). Same torchvision call as alpamayo2_edge.helper.
    """
    h, w = smart_resize(*frames.shape[-2:], factor=factor,
                        min_pixels=FRONT_PIXELS[0], max_pixels=FRONT_PIXELS[1])
    if (h, w) == tuple(frames.shape[-2:]):
        return frames
    out = torch.stack([resize(f, [h, w], interpolation=InterpolationMode.BICUBIC)
                       for f in frames.flatten(0, 1)])
    return out.reshape(*frames.shape[:2], 3, h, w).contiguous()


def future_trajectory(avdi: Any, clip_id: str, t0_us: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth: (64, 3) positions and (64, 3, 3) rotations at t0 + 0.1 .. 6.4 s, ego frame at t0."""
    egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    ts = t0_us + STEP_US * np.arange(0, FUTURE_STEPS + 1, dtype=np.int64)
    lo, hi = egomotion.time_range
    if ts[-1] > hi:
        raise ValueError(f"{clip_id}: the future to {ts[-1]} us leaves the egomotion range "
                         f"[{lo}, {hi}] us; use fewer windows or an earlier t0")
    pose = egomotion(ts).pose
    t0_inv = pose.rotation[0].inv()
    xyz = t0_inv.apply(np.asarray(pose.translation)[1:] - np.asarray(pose.translation)[0])
    rot = (t0_inv * pose.rotation[1:]).as_matrix()
    return xyz.astype(np.float32), rot.astype(np.float32)


def oracle_nav_text(future_xyz: np.ndarray, future_rot: np.ndarray) -> str:
    """A route instruction from the ground-truth future, as a navigation stack would give it.

    The first future pose whose heading deviates by 15 deg or more names the turn and the
    path length to it, rounded to 10 m ("Turn right in 30m", the style of NVIDIA's example);
    otherwise "Go straight".
    """
    yaw = np.degrees(np.arctan2(future_rot[:, 1, 0], future_rot[:, 0, 0]))
    turning = np.flatnonzero(np.abs(yaw) >= 15.0)
    if not len(turning):
        return "Go straight"
    first = int(turning[0])
    side = "left" if yaw[first] > 0 else "right"
    path = np.vstack([np.zeros((1, 2), np.float32), future_xyz[: first + 1, :2]])
    dist = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return f"Turn {side} in {max(10, int(round(dist / 10.0)) * 10)}m"


def window_path(cache_dir: str, clip_id: str, t0_us: int) -> str:
    return os.path.join(cache_dir, f"{clip_id}__{t0_us}.npz")


def load_window(
    clip_id: str,
    t0_us: int,
    nav_text: str | None = None,
    avdi: Any = None,
    cache_dir: str | None = None,
    resize_to_model: bool = True,
    factor: int = TOKEN_PX,
) -> dict[str, Any]:
    """The raw Edge sample of (clip, t0) with ``ego_future_xyz`` [1, 64, 3] and
    ``ego_future_rot`` [1, 64, 3, 3] added.

    With ``cache_dir``, reads the window's .npz when it exists and writes it after streaming
    otherwise. ``nav_text`` None means the oracle instruction (``oracle_nav_text``); the text
    is never cached. ``resize_to_model`` stores frames at the model's input size (11x less
    disk than 1080p); see ``resize_frames`` for what that changes.
    """
    if t0_us < MIN_T0_US:
        raise ValueError(f"t0_us must be >= {MIN_T0_US} (2 s of camera history)")
    path = window_path(cache_dir, clip_id, t0_us) if cache_dir else None
    if path and os.path.exists(path) and os.path.getsize(path):
        with np.load(path) as npz:
            data = {k: torch.from_numpy(npz[k]) for k in TENSORS}
    else:
        if avdi is None:
            import physical_ai_av

            avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        data = load_physical_aiavdataset(clip_id, t0_us, nav_text="placeholder", avdi=avdi)
        xyz, rot = future_trajectory(avdi, clip_id, t0_us)
        data["ego_future_xyz"] = torch.from_numpy(xyz)[None]
        data["ego_future_rot"] = torch.from_numpy(rot)[None]
        if resize_to_model:
            data["image_frames"] = resize_frames(data["image_frames"], factor)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path + ".part", "wb") as fh:  # np.savez appends ".npz" to a bare path
                np.savez(fh, **{k: data[k].numpy() for k in TENSORS})
            os.replace(path + ".part", path)
    if nav_text is None:
        nav_text = oracle_nav_text(data["ego_future_xyz"][0].numpy(),
                                   data["ego_future_rot"][0].numpy())
    return {
        **{k: data[k] for k in TENSORS},
        "nav_text": [nav_text],
        "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        "clip_id": clip_id,
        "t0_us": t0_us,
    }


def min_ade(gt_future_xyz: torch.Tensor, pred_xyz: torch.Tensor) -> tuple[float, int, float]:
    """(min ADE, index of that candidate, mean ADE) over every set x sample of ``pred_xyz``.

    ``gt_future_xyz`` is [1, 64, 3]; ``pred_xyz`` is [1, sets, samples, 64, 3]. ADE is the mean
    xy distance over the 64 waypoints, as in flashdrive's scorer; minADE is the best candidate
    and the mean ADE the average candidate (equal with one candidate).
    """
    gt = gt_future_xyz.detach().float().cpu().numpy()[0, :, :2]
    pred = pred_xyz.detach().float().cpu().numpy()[0].reshape(-1, FUTURE_STEPS, 3)[..., :2]
    ade = np.linalg.norm(pred - gt[None], axis=-1).mean(-1)
    return float(ade.min()), int(ade.argmin()), float(ade.mean())
