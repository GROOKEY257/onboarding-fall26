# Submission — HRC Software Onboarding Fall 2026

**Name:Rohan Pardeshi**
**Discord handle:tweedledee7352**
**Repository:https://github.com/GROOKEY257/onboarding-fall26/edit/main/SUBMISSION.md**
**Date:10/13/26**

---

## Stage checklist

- [x] **Stage 0** — Setup. `scripts/check_setup.py` exits 0.
- [x] **Stage 1** — MuJoCo + PD controller. Tests green.
- [x] **Stage 2** — JAX exercises. Tests green.
- [x] **Stage 3** — MJX environment. Tests green.
- [x] **Stage 4** — Brax PPO. Trained a policy, exported it, `NumpyPolicy` matches.
- [x] **Stage 5** — ROS 2 sim2sim. Smoke test green, teleop GIF recorded.

## `scripts/progress.py` output

<details>
<summary>paste the full output here</summary>

```
....................................                                                      [100%]
======================================= warnings summary ========================================
.venv/lib/python3.11/site-packages/jaxopt/__init__.py:59
  /Users/rohanpardeshi/onboarding-fall26/.venv/lib/python3.11/site-packages/jaxopt/__init__.py:59: DeprecationWarning: JAXopt is no longer maintained. See https://docs.jax.dev/en/latest/ for alternatives.
    warnings.warn(

tests/test_04_train_smoke.py::test_cpu_smoke
  /Users/rohanpardeshi/onboarding-fall26/.venv/lib/python3.11/site-packages/brax/training/agents/ppo/train.py:756: DeprecationWarning: jax.device_put_replicated is deprecated; use jax.device_put instead.
    training_state = jax.device_put_replicated(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

Pup onboarding progress
  ✓ PASSED        Stage 1  MuJoCo + PD controller    12 passed, 0 failed, 0 not started, 0 skipped
  ✓ PASSED        Stage 2  JAX for robotics          9 passed, 0 failed, 0 not started, 0 skipped
  ✓ PASSED        Stage 3  MJX environment           11 passed, 0 failed, 0 not started, 0 skipped
  ✓ PASSED        Stage 4  Brax PPO + export         4 passed, 0 failed, 0 not started, 0 skipped
36 passed, 1 skipped, 2 deselected, 2 warnings in 92.45s (0:01:32)

==============================================================================
PUP ONBOARDING PROGRESS
==============================================================================
✓ PASSED         Stage 1  MuJoCo + PD controller    12 passing
✓ PASSED         Stage 2  JAX for robotics          9 passing
✓ PASSED         Stage 3  MJX environment           11 passing
✓ PASSED         Stage 4  Brax PPO + export         4 passing
◻ CONTAINER      Stage 5  ROS 2 sim2sim             run tests/test_05_ros2_smoke.sh inside docker/ros2
==============================================================================
Every TODO(student) block is written. Nice.
```

</details>

## Stage 4 — training results

**Config used:** (`full`) &nbsp; **Seed:** &nbsp;
**Where it ran:** (Colab T4) &nbsp; **Wall-clock:**

Learning curve: <img width="690" height="590" alt="image" src="https://github.com/user-attachments/assets/79aa0951-852b-447b-b042-2b524beb93f2" />

Rollout GIF: `results/training.gif`

```json
(paste results/eval_training.json)
```

Did it meet the acceptance criteria (`"walking_passes": false`)?

## Stage 5 — sim2sim results

Teleop GIF: `results/sim2sim_teleop.gif`

```json
{
  "duration_s": 20.0,
  "min_trunk_height": 0.29080933985803675,
  "fell": false,
  "commands": [
    {
      "command": [
        0.5,
        0.0,
        0.0
      ],
      "n_samples": 151,
      "mean_velocity": [
        0.3851300945635746,
        0.009848236340293172,
        -1.6629729581449172e-05
      ],
      "mean_abs_error": [
        0.11486990543642539,
        0.009848236340293172,
        1.6629729581449172e-05
      ],
      "mean_speed": 0.3853263940667031,
      "threshold": 0.25,
      "passed": true
    },
    {
      "command": [
        1.0,
        0.0,
        0.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        0.9576339076414073,
        -0.009475488324482093,
        -0.00488484437301761
      ],
      "mean_abs_error": [
        0.04236609235859268,
        0.009475488324482093,
        0.00488484437301761
      ],
      "mean_speed": 0.9580095327479282
    },
    {
      "command": [
        0.0,
        0.0,
        1.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        -0.09037953822426074,
        0.07573641660111385,
        0.9910696348895893
      ],
      "mean_abs_error": [
        0.09037953822426074,
        0.07573641660111385,
        0.008930365110410676
      ],
      "mean_speed": 0.11984849297889265
    },
    {
      "command": [
        0.0,
        0.0,
        0.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        -9.116386514856765e-05,
        8.975978752910178e-05,
        -5.764426470764799e-05
      ],
      "mean_abs_error": [
        9.116386514856765e-05,
        8.975978752910178e-05,
        5.764426470764799e-05
      ],
      "mean_speed": 0.005273793796460159
    }
  ],
  "passed": true
}

```

Measured `/pup/joint_command` rate from `ros2 topic hz`:

## Escape hatches used

- [x] I used `checkpoints/pup_joystick_flat_reference.npz` for Stage 5 instead
      of my own policy.
- [ ] Other (describe):

*(Using one is fine. Not declaring one is not.)*

## Reflection — Stage 5

**List two ways sim2sim can pass while real hardware still fails, and what you
would add to the sim node to catch each.**

>

## What was hardest?

One paragraph. This is will help us improve onboarding.

>

## Time spent

| Stage | Hours |
|---|---|
| 0 Setup | |
| 1 MuJoCo + PD | |
| 2 JAX | |
| 3 MJX env | |
| 4 Brax + export | |
| 5 ROS 2 | |
| **Total** | |

---

**Then DM the software lead (Henry Tsay) on Discord or show during a meeting.**
