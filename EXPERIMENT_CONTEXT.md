# Experiment Context - Rolling Ball DQN

Last updated: 2026-04-09

## Goal
Train a DQN agent (discrete actions) to reach the cube in the Unity rolling ball scene.

## Scope of the exercise
1. Complete DQN blanks in `rolling_ball_DQN.py`.
2. Train from `rolling_ball_gym.py` with Unity Editor scene running.
3. Modify both hyperparameters and rewards.

## Current code status
### DQN implementation status
- Network architecture implemented: `[input, 512, 512, output]`
- Forward propagation implemented with ReLU on hidden layers.
- Replay memory push implemented in `update()`.
- Target value at `t+1` implemented in `learn()`.
- Target network periodic sync implemented (`steps_done % F == 0`).

### Current hyperparameters
From `Rolling_ball_Python/rolling_ball_gym.py` and `Rolling_ball_Python/rolling_ball_DQN.py`:
- `input_size = 9`
- `output_size = 5`
- `batch_size = 128`
- `gamma = 0.99`
- `F = 500`
- `lr = 1e-4`
- `eps_start = 0.9`
- `eps_end = 0.05`
- `eps_decay = 3000`
- `n_episode = 300`

### Current reward design (discrete Unity agent)
From `AI_in_games_unity/Assets/Scripts/rolling_ball/rollerAgentDiscrete.cs`:
- If distance to target decreases: `AddReward(+0.02)`
- If distance to target increases: `AddReward(-0.02)`
- Reached target: `SetReward(+1.0)` + `EndEpisode()`
- Fell off platform: `SetReward(-1.0)` + `EndEpisode()`
- Timeout at step 500: `SetReward(-0.5)` + `EndEpisode()`

## Why this reward scale
- Keep reward range mostly normalized in `[-1, 1]` for training stability.
- Preserve a clear terminal success signal (+1.0).
- Keep shaping signal small (`+/-0.02`) to guide motion without dominating terminal rewards.

## Run protocol
1. Open Unity project and load rolling ball training scene.
2. Press Play in Unity Editor.
3. Run Python script:
   - `Rolling_ball_Python/rolling_ball_gym.py`
4. Wait for training to finish and inspect:
   - console episode logs
   - `rolling_ball_reward.png`

## Metrics to evaluate progress
- Smoothed reward trend should increase.
- Smoothed episode duration should decrease.
- Fewer extreme negative spikes over time.
- More consistent successful episodes (shorter paths to target).

## Known environment notes
If imports fail in editor diagnostics (`torch`, `numpy`, `matplotlib`, `mlagents_envs`), verify Python env and dependencies before training.

## Change log (recent)
- Returned to a near-prof baseline.
- Re-applied only requested exercise completions.
- Kept light parameter changes for step 3.
- Updated reward scale to normalized targets (`+1/-1`, timeout `-0.5`, shaping `+/-0.02`).

## Quick resume checklist
- [ ] Unity scene open and running
- [ ] Python environment selected with required packages
- [ ] `rolling_ball_gym.py` parameters confirmed
- [ ] `rollerAgentDiscrete.cs` rewards confirmed
- [ ] Start training and save plot for comparison
