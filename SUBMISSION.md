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

**Config used:** (`full`) &nbsp; **Seed:** 0 &nbsp;
**Where it ran:** (Colab T4) &nbsp; **Wall-clock:** 

Learning curve: <img width="690" height="594" alt="learning_curve" src="https://github.com/user-attachments/assets/f85b3bf3-9d6c-483a-9935-d4ad6cf04caf" />


Rollout GIF: <img width="480" height="360" alt="training" src="https://github.com/user-attachments/assets/15d88ad9-aa36-4f1c-9641-21fc2f56ca2e" />


```json
{
  "seed": 0,
  "n_episodes": 5,
  "commands": [
    {
      "command": [
        0.5,
        0.0,
        0.0
      ],
      "mean_abs_error": [
        0.04095508968904615,
        0.03783120522607351,
        0.06429639094207815
      ],
      "mean_abs_vy": 0.03783120587468147,
      "mean_speed": 0.48705968260765076,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        1.0,
        0.0,
        0.0
      ],
      "mean_abs_error": [
        0.06039156036451459,
        0.03960291149731492,
        0.06677012027379242
      ],
      "mean_abs_vy": 0.039602912962436676,
      "mean_speed": 0.9604790806770325,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        0.0,
        0.5,
        0.0
      ],
      "mean_abs_error": [
        0.03170375525192358,
        0.03774958482384682,
        0.060872955922866824
      ],
      "mean_abs_vy": 0.494541198015213,
      "mean_speed": 0.4964277446269989,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        0.0,
        0.0,
        1.0
      ],
      "mean_abs_error": [
        0.029201994308334544,
        0.03349815212571266,
        0.06395855396986008
      ],
      "mean_abs_vy": 0.03349815309047699,
      "mean_speed": 0.049090426415205,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    },
    {
      "command": [
        0.0,
        0.0,
        0.0
      ],
      "mean_abs_error": [
        0.036109516263610565,
        0.03517708714180044,
        0.08204666090941755
      ],
      "mean_abs_vy": 0.0351770855486393,
      "mean_speed": 0.05610450357198715,
      "fall_rate": 0.0,
      "mean_episode_length": 1000.0
    }
  ],
  "walking_passes": true
}
```

Did it meet the acceptance criteria (`"walking_passes": true`)?

## Stage 5 — sim2sim results

Teleop GIF: <img width="480" height="360" alt="pup_walk" src="https://github.com/user-attachments/assets/d7b1bf0f-d6f1-466d-9c82-36c10fc27d77" />


```json
{
  "duration_s": 20.0,
  "min_trunk_height": 0.29144480893579094,
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
        0.38684215210844447,
        0.00814186995665601,
        -0.0005326592037491536
      ],
      "mean_abs_error": [
        0.11315784789155553,
        0.00814186995665601,
        0.0005326592037491536
      ],
      "mean_speed": 0.3869935053639113,
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
        0.9521203631256752,
        -0.009549845374085207,
        -0.012107732866052826
      ],
      "mean_abs_error": [
        0.04787963687432484,
        0.009549845374085207,
        0.012107732866052826
      ],
      "mean_speed": 0.9524564889597446
    },
    {
      "command": [
        0.0,
        0.0,
        1.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        -0.08941239062612842,
        0.07473635426543493,
        0.9902400634704319
      ],
      "mean_abs_error": [
        0.08941239062612842,
        0.07473635426543493,
        0.00975993652956808
      ],
      "mean_speed": 0.11836093606452448
    },
    {
      "command": [
        0.0,
        0.0,
        0.0
      ],
      "n_samples": 150,
      "mean_velocity": [
        -4.292406373022428e-05,
        8.455396094763676e-05,
        4.144353038139329e-05
      ],
      "mean_abs_error": [
        4.292406373022428e-05,
        8.455396094763676e-05,
        4.144353038139329e-05
      ],
      "mean_speed": 0.004927584755784716
    }
  ],
  "passed": true
}

```

Measured `/pup/joint_command` rate from `ros2 topic hz`: average rate: 49.999
        min: 0.004s max: 0.035s std dev: 0.00079s window: 2525

## Escape hatches used

- [x] I used `checkpoints/pup_joystick_flat_reference.npz` for Stage 5 instead
      of my own policy.
- [ ] Other (describe):

*(Using one is fine. Not declaring one is not.)*

## Reflection — Stage 5

**List two ways sim2sim can pass while real hardware still fails, and what you
would add to the sim node to catch each.**

>A real robot might have different friction and motor strengths. I would add torque limits and random friction to account for these. Real sensors may have delays that don't have a constant update rate like the simulation. I would add some sensor noise and delays in the sim node to catch policies that depend on perfectly timed observations.

## What was hardest?

One paragraph. This is will help us improve onboarding.

>The hardest part was trying to understand how to code the _get_obs(data, info) function and understanding the math behind the training of the policy and quaternions. I didn't really understand how to calculate the gyro of the robot. I also still kind of don't understand the math behind calculating the rewards for each of the movements the robot does during its training. The quaternion math was also kind of confusing because I hadn't worked with 4D vectors before in an actual real life application. 

## Time spent

| Stage | Hours |
|---|---|
| 0 Setup |1|
| 1 MuJoCo + PD |2|
| 2 JAX |2|
| 3 MJX env |5|
| 4 Brax + export |10|
| 5 ROS 2 |6|
| **Total** |26|

---

**Then DM the software lead (Henry Tsay) on Discord or show during a meeting.**
