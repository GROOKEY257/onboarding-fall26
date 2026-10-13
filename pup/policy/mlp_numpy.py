"""Run a trained Brax policy with numpy alone -- no JAX, no GPU, no jit.

This is what the ROS 2 node imports. It must reproduce Brax's evaluation-time
inference exactly:

1. normalize:  ``x = (obs - obs_mean) / obs_std``
2. hidden layers: ``x = swish(x @ kernel_i + bias_i)`` where ``swish(x) = x *
   sigmoid(x)``
3. output layer (linear): ``logits = x @ kernel_last + bias_last``, shape (24,)
4. deterministic action: ``tanh(logits[:12])`` -- the second half is the
   Gaussian standard deviation and is unused at evaluation time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def swish(x: np.ndarray) -> np.ndarray:
    """Return x * sigmoid(x), the activation Brax's MLPs use by default."""
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    """A frozen Brax MLP policy evaluated in pure numpy."""

    def __init__(self, archive: dict) -> None:
        """Build from the dict of arrays produced by ``pup.train.export``."""
        self.obs_mean = np.asarray(archive["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(archive["obs_std"], dtype=np.float32)

        self.n_layers = int(archive["n_layers"])
        self.obs_size = int(archive["obs_size"])
        self.action_size = int(archive["action_size"])

        self.action_scale = np.asarray(archive["action_scale"], dtype=np.float32)
        self.default_pose = np.asarray(archive["default_pose"], dtype=np.float32)
        self.hidden_activation = str(archive["hidden_activation"].item())
        self.obs_layout = archive["obs_layout"].item()

        # Keep the layers in forward-pass order.
        self.layers = []
        for i in range(self.n_layers):
            kernel = np.asarray(archive[f"kernel_{i}"], dtype=np.float32)
            bias = np.asarray(archive[f"bias_{i}"], dtype=np.float32)
            self.layers.append((kernel, bias))

        # Validate the exported policy's expected dimensions.
        if self.obs_mean.shape != (self.obs_size,):
            raise ValueError(f"Invalid obs_mean shape: {self.obs_mean.shape}")
        if self.obs_std.shape != (self.obs_size,):
            raise ValueError(f"Invalid obs_std shape: {self.obs_std.shape}")
        if self.layers[0][0].shape[0] != self.obs_size:
            raise ValueError("First layer input size does not match obs_size")
        if self.layers[-1][0].shape[1] != 2 * self.action_size:
            raise ValueError("Final layer must output means and std parameters")

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        """Load a policy exported by ``pup.train.export.export_policy``."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls({key: archive[key] for key in archive.files})

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Map a (45,) observation to a (12,) action in [-1, 1], unitless."""
        x = np.asarray(obs, dtype=np.float32)

        if x.shape != (self.obs_size,):
            raise ValueError(
                f"Expected observation shape ({self.obs_size},), got {x.shape}"
            )

        # Normalize the observation.
        x = (x - self.obs_mean) / self.obs_std

        # Run the MLP: swish for hidden layers, linear output layer.
        for i, (kernel, bias) in enumerate(self.layers):
            x = x @ kernel + bias

            if i < self.n_layers - 1:
                x = swish(x)

        # Use the first 12 outputs as action means and squash with tanh.
        return np.tanh(x[:self.action_size]).astype(np.float32)


    def joint_targets(self, obs: np.ndarray) -> np.ndarray:
        """Map a (45,) observation to (12,) joint position targets in rad."""
        action = self(obs)
        return (
            self.default_pose + self.action_scale * action
        ).astype(np.float32)
