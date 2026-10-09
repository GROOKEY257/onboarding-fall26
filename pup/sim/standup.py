"""Raise Pup smoothly from crouch before asking it to walk."""

import mujoco
import numpy as np

from contextlib import nullcontext

from pup.sim.pd import PDController, joint_state
from pup.sim.viewer import load_scene, reset_to_keyframe
import time


def stand_up(duration_s: float = 3.0, headless: bool = True,
             kp: float = 79.0, kd: float = 1.0) -> dict:
    model, data = load_scene()

    q_end = model.key("home").qpos[7:].copy()

    reset_to_keyframe(model, data, "crouch")
    q_start = model.key("crouch").qpos[7:].copy()

    controller = PDController(kp, kd)

    max_roll = 0.0
    max_pitch = 0.0
    fell = False
    context = nullcontext(enter_result=None)
    if not headless:
        from mujoco.viewer import launch_passive
        context = launch_passive(model, data)

    with context as viewer:
        while data.time < duration_s:

            w, x, y, z = data.qpos[3:7]

            roll = np.arctan2(
                2 * (w * x + y * z),
                1 - 2 * (x * x + y * y)
            )

            pitch = np.arcsin(
                np.clip(
                    2 * (w * y - z * x),
                    -1.0,
                    1.0
                )
            )

            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))

            if (data.qpos[2] < 0.12 or not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel))):
                fell = True

            t = min(data.time, 1.0)

            q_des = (1 - t) * q_start + t * q_end

            q, qd = joint_state(model, data)

            torque = controller(q, qd, q_des)

            data.ctrl[:] = torque

            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

            time.sleep(0.001)

    return {
        "final_height": float(data.qpos[2]),
        "max_roll": float(max_roll),
        "max_pitch": float(max_pitch),
        "fell": fell
    }