"""A motor driver computes torque from position and velocity errors."""

import jax.numpy as jnp
import mujoco
import numpy as np


class PDController:
    """Joint proportional-derivative control with physical torque limits."""

    kp: np.ndarray
    kd: np.ndarray
    torque_limit: float 
    def __init__(self, kp: np.ndarray | float, kd: np.ndarray | float,
                 torque_limit: float = 20.0) -> None:
        """Store scalar or (12,) gains in Nm/rad and Nm/(rad/s), limit in Nm."""
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

    def __call__(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray,
                 qd_des: np.ndarray | None = None) -> np.ndarray:
        """Return (12,) clipped torques, Nm, for (12,) angles/rates in rad/rad/s."""
        if qd_des is None:
            qd_des = jnp.zeros_like(qd)

        tau = self.kp * (q_des - q) + self.kd * (qd_des - qd)
        return np.clip(np.asarray(tau), -self.torque_limit, self.torque_limit)


def joint_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of actuated q (12,), rad, and qd (12,), rad/s; root is free."""
    return (data.qpos[7:].copy(), data.qvel[6:].copy())
