"""Raise Pup smoothly from crouch before asking it to walk."""

from contextlib import nullcontext

import mujoco
import numpy as np

from pup.sim.pd import PDController, joint_state
from pup.sim.viewer import load_scene, reset_to_keyframe


def stand_up(duration_s: float = 3.0, headless: bool = True,
             kp: float = 10.0, kd: float = 1.0) -> dict:  # TODO(student): tune
    """Return final_height (m), max_roll/max_pitch (rad), and fell (bool).

    Interpolate (12,) target angles from crouch to home in one second;
    then hold until duration_s.

    The default gains above are the spring-2026 quadruped's (kp=10). Pup is
    heavier -- run it, watch it sag, and tune them (Stage 1, task 3). The
    test reads whatever defaults you leave in the signature.
    """
    model, data = load_scene("/Users/rohanpardeshi/onboarding-fall26/pup/assets/pup.xml")
    homemodel, homedata = joint_state(model, data)
    q_end = model.key("home").qpos[7:].copy()
    reset_to_keyframe(model, data, "crouch")
    q_start = model.key("crouch").qpos[7:].copy()
    roll: float = 0
    pitch: float = 0
    max_height: float = 0
    fell: bool = False
    controller = PDController(kp, kd)

    for step in range(500):
        w, x, y, z = data.qpos[3:7]
        if np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)) > roll:
            roll  = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        if np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0)) > pitch:
            pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
        if data.qpos[8] < 0.12:
            fell = True
        if data.qpos[8] > max_height:
            max_height = data.qpos[8]
        t = min(data.time / 1.0, 1.0)
        q_des = (1 - t) * q_start + t * q_end
        q, qd = joint_state(model, data)
        torque = controller(q, qd, q_des)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
    for step in range(int(duration_s/0.002)):
        q, qd = joint_state(model, data)
        torque = controller(q, qd, q_end)
        mujoco.mj_step(model, data)
        
    return {
        "final_height": data.qpos[2],
        "max_roll": roll,
        "max_pitch": pitch,
        "fell": fell
    }