"""EXP-04 waves + tail effect (Lecture 1, Part B): pure occupancy arithmetic.

slots = SMs * blocks_per_SM. waves = ceil(blocks / slots).
Lecture toy example: 20 blocks of 256 on 4 SM x 2 slots -> 3 waves, last half-empty.
Check on RTX 3090 (82 SM): how grid/blocks-per-SM choices fill or starve waves.
"""
from __future__ import annotations
import math


def waves(n_blocks: int, sms: int, blocks_per_sm: int) -> dict:
    slots = sms * blocks_per_sm
    n_waves = math.ceil(n_blocks / slots)
    last_fill = n_blocks - (n_waves - 1) * slots
    return {"blocks": n_blocks, "slots": slots, "waves": n_waves,
            "last_wave_fill": last_fill,
            "last_wave_util": last_fill / slots,
            "wasted_slots": slots - last_fill}


def run():
    lecture = waves(20, 4, 2)  # must be 3 waves, 50% last-wave util
    assert lecture["waves"] == 3 and abs(lecture["last_wave_util"] - 0.5) < 1e-9
    rtx3090 = {"sms": 82}
    grids = []
    for n_blocks in (164, 656, 1312, 1353):
        for bps in (1, 2, 4):
            w = waves(n_blocks, rtx3090["sms"], bps)
            grids.append({"blocks": n_blocks, "blocks_per_sm": bps, **w})
    return {"lecture_toy": lecture, "rtx3090_grids": grids}
