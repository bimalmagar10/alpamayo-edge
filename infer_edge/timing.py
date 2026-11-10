"""Per-stage GPU time of one Edge rollout, without touching the model's code.

CUDA events bracket the model's own calls (eager and accelerated paths alike):

  encode         vision tower (``vlm.model.visual``), always eager
  prefill_total  the window's first text-stack call, or the accelerated prefill graph;
                 its vision encode is nested inside
  vlm_total      ``_generate_with_shared_prefill``: encode + prefill + CoT decode
  action         ``_sample_actions``: the flow-matching expert

so that prefill = prefill_total - encode, decode = vlm_total - prefill_total and the rest
of the wall time (token fusing, action_to_traj, host syncs) is ``other``.
"""

import collections
import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)
STAGES = ("encode", "prefill", "decode", "action", "other")
VISION_ATTRS = ("visual", "vision_tower", "vision_model")  # the vision tower's name in the VLM


class StageTimer:
    def __init__(self) -> None:
        self.events: dict[str, list] = collections.defaultdict(list)
        self.on = False
        self.text_calls = 0

    def wrap(self, obj: Any, attr: str, name_of: Callable[[], str]) -> None:
        inner = getattr(obj, attr)

        def timed(*a: Any, **kw: Any) -> Any:
            if not self.on:
                return inner(*a, **kw)
            name = name_of()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            out = inner(*a, **kw)
            end.record()
            if name:
                self.events[name].append((start, end))
            return out

        setattr(obj, attr, timed)

    def install(self, model: torch.nn.Module) -> None:
        """Call after ``enable_acceleration`` so the wrappers sit outside its own."""
        vision = next((getattr(model.vlm.model, name) for name in VISION_ATTRS
                       if hasattr(model.vlm.model, name)), None)
        if vision is None:
            logger.warning("no vision tower among %s: encode is folded into prefill", VISION_ATTRS)
        else:
            self.wrap(vision, "forward", lambda: "encode")

        def text_stage() -> str:
            self.text_calls += 1
            # The first text-stack call of a window is the prefill; the eager decode steps
            # that follow are not timed one by one (decode is vlm_total - prefill_total).
            return "prefill_total" if self.text_calls == 1 else ""

        self.wrap(model.vlm.model, "forward", text_stage)
        accel = getattr(model, "_alpamayo2_edge_accel", None)
        for accelerator in accel.accelerators if accel is not None else ():
            if hasattr(accelerator, "_graph_prefill"):
                self.wrap(accelerator, "_graph_prefill", lambda: "prefill_total")
        self.wrap(model, "_generate_with_shared_prefill", lambda: "vlm_total")
        self.wrap(model, "_sample_actions", lambda: "action")

    def start(self) -> None:
        self.events.clear()
        self.text_calls = 0
        self.on = True

    def stop(self, wall_ms: float) -> dict[str, float]:
        """Stage milliseconds of the window timed since ``start`` (synchronizes the GPU)."""
        self.on = False
        torch.cuda.synchronize()
        total = {k: sum(a.elapsed_time(b) for a, b in v) for k, v in self.events.items()}
        encode = total.get("encode", 0.0)
        prefill_total = total.get("prefill_total", 0.0)
        vlm = total.get("vlm_total", 0.0)
        action = total.get("action", 0.0)
        return {
            "encode": encode,
            "prefill": max(prefill_total - encode, 0.0),
            "decode": max(vlm - prefill_total, 0.0),
            "action": action,
            "other": max(wall_ms - vlm - action, 0.0),
        }
