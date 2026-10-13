"""Turn a trained Brax policy into a plain .npz that numpy can read.

Brax stores a trained agent as a tuple of three PyTrees:

    params = (normalizer_params, policy_params, value_params)

`normalizer_params` is a `running_statistics.RunningStatisticsState` holding the
observation `mean` (45,) and `std` (45,). `policy_params` is a Flax parameter
dict shaped like::

    {'params': {'hidden_0': {'kernel': (45, 128),  'bias': (128,)},
                'hidden_1': {'kernel': (128, 128), 'bias': (128,)},
                ...
                'hidden_4': {'kernel': (128, 24),  'bias': (24,)}}}

The last layer emits ``2 * action_size`` numbers: the first 12 are the Gaussian
mean, the last 12 the (pre-softplus) standard deviation. Hidden layers use the
swish activation; the final layer is linear. The deterministic action Brax uses
at evaluation time is ``tanh(mean)``.

Run ``jax.tree_util.tree_map(lambda x: x.shape, policy_params)`` yourself to see
the tree. Nothing here is secret -- the job is to walk that tree in order and
write it out.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from pup.envs.config import default_config
from pup.envs.constants import ACTION_SCALE, DEFAULT_POSE, OBS_LAYOUT


def _layer_index(name: str) -> int:
    """Return the trailing integer of a Flax layer name such as ``hidden_3``."""
    return int(name.rsplit("_", 1)[1])


def export_policy(params: Any, normalizer_params: Any, out_path: str | Path) -> Path:
    import re
    
    """Write a Brax MLP policy to ``out_path`` as a JAX-free ``.npz``.

    Args:
      params: The policy parameter PyTree (``params[1]`` of a Brax PPO run), i.e.
        ``{'params': {'hidden_i': {'kernel': (in, out), 'bias': (out,)}}}``.
      normalizer_params: The observation normalizer state (``params[0]``) with
        ``.mean`` (45,) and ``.std`` (45,) in observation units.
      out_path: Destination ``.npz`` path; parent directories are created.

    Returns:
      The resolved path that was written.

    The archive contains ``obs_mean`` (45,), ``obs_std`` (45,),
    ``kernel_i``/``bias_i`` for every layer ``i`` in forward order, plus the
    metadata arrays ``n_layers``, ``obs_size``, ``action_size``,
    ``action_scale``, ``default_pose`` (12,), ``hidden_activation`` and
    ``obs_layout``.
    """
    # Extract the policy layers and observation normalizer.
    policy_tree = params["params"]
    obs_mean = np.asarray(normalizer_params.mean)
    obs_std = np.asarray(normalizer_params.std)

    obs_size = 45
    action_size = 12

    if obs_mean.shape != (obs_size,) or obs_std.shape != (obs_size,):
        raise ValueError("Expected obs_mean and obs_std to have shape (45,)")

    # Sort hidden layers by their numeric suffix, not dictionary order.
    def layer_index(name):
        match = re.fullmatch(r"hidden_(\d+)", name)
        return int(match.group(1)) if match else None

    hidden_names = sorted(
        [
            name for name in policy_tree
            if layer_index(name) is not None
        ],
        key=layer_index,
    )

    # Find the final dense layer, which is not named hidden_i.
    output_names = [
        name for name, layer in policy_tree.items()
        if layer_index(name) is None
        and isinstance(layer, dict)
        and "kernel" in layer
        and "bias" in layer
    ]

    if len(output_names) != 1:
        raise ValueError(f"Expected one output layer, found {output_names}")

    ordered_names = hidden_names + output_names

    # Store arrays in forward-pass order.
    arrays = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
    }

    previous_size = obs_size

    for i, name in enumerate(ordered_names):
        layer = policy_tree[name]
        kernel = np.asarray(layer["kernel"])
        bias = np.asarray(layer["bias"])

        if kernel.ndim != 2 or kernel.shape[0] != previous_size:
            raise ValueError(
                f"Unexpected kernel shape for {name}: {kernel.shape}"
            )

        if bias.shape != (kernel.shape[1],):
            raise ValueError(
                f"Unexpected bias shape for {name}: {bias.shape}"
            )

        arrays[f"kernel_{i}"] = kernel
        arrays[f"bias_{i}"] = bias
        previous_size = kernel.shape[1]

    if previous_size != 2 * action_size:
        raise ValueError(
            f"Expected final layer size 24, got {previous_size}"
        )

    # Include environment metadata.
    config = default_config()
    default_pose = np.asarray(config.default_pose)
    action_scale = np.asarray(config.action_scale)

    if default_pose.shape != (action_size,):
        raise ValueError(
            f"Expected default_pose shape (12,), got {default_pose.shape}"
        )

    arrays.update({
        "n_layers": np.asarray(len(ordered_names), dtype=np.int32),
        "obs_size": np.asarray(obs_size, dtype=np.int32),
        "action_size": np.asarray(action_size, dtype=np.int32),
        "action_scale": action_scale,
        "default_pose": default_pose,
        "hidden_activation": np.asarray("swish"),
        "obs_layout": np.asarray(
            "gyro[0:3],gravity_body[3:6],command[6:9],"
            "qpos_minus_default[9:21],qvel[21:33],last_action[33:45]"
        ),
    })

    # Create the destination directory and write the archive.
    destination = Path(out_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    return destination



def load_brax_params(checkpoint: str | Path) -> tuple:
    """Load ``(normalizer_params, policy_params)`` from a ``.pkl`` or an Orbax directory.

    ``train_ppo.py`` writes both: ``runs/<name>/policy.pkl`` at the end of the run,
    and ``runs/<name>/checkpoints/<step>/`` after every evaluation. Accepting the
    latter means a run that was cut short (a Colab session timing out, say) is
    still exportable.
    """
    path = Path(checkpoint).resolve()  # Orbax rejects relative paths
    if path.is_dir():
        from brax.training.agents.ppo import checkpoint as ppo_checkpoint

        params = ppo_checkpoint.load(path)
    else:
        from brax.io import model as model_io

        params = model_io.load_params(str(path))
    return params[0], params[1]


def main() -> None:
    """Convert a checkpoint written by ``train_ppo.py`` into a numpy ``.npz``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="runs/<name>/policy.pkl, or an Orbax checkpoint directory")
    parser.add_argument("--out", required=True, help="destination .npz")
    args = parser.parse_args()
    normalizer_params, policy_params = load_brax_params(args.checkpoint)
    print("wrote", export_policy(policy_params, normalizer_params, args.out))


if __name__ == "__main__":
    main()
