# TP2 Context - AI Driver Unity (PPO)

Last updated: 2026-04-09

## Goal
Train the car agent to drive without crashing in Unity with ML-Agents, using PPO.

## What to keep in mind
- The report can stay simple: baseline run, one or two parameter changes, short interpretation.
- The main things to read are reward, episode length, entropy, policy loss, value loss.
- The batch size is the parameter highlighted by the professor: larger batch means more samples per optimization step, so updates are smoother but heavier.

## Main files in the project
- `results/configuration_example.yaml`: PPO training config.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_controller.cs`: car movement logic.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent.cs`: base ML-Agents agent class.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent_template_track1.cs`: track1 agent template to complete.

## Scripts to cite in the report
- `results/configuration_example.yaml`: PPO hyperparameters for training.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_controller.cs`: handles steering, throttle and braking.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent.cs`: base agent class with initialization and episode handling.
- `AI_in_games_unity/Assets/Scripts/car_agents/car_agent_template_track1.cs`: subclass where the training interaction is implemented.

## Unity editor settings to mention
- Behavior Name.
- Vector Observation space size.
- Stacked Vectors.
- Action branches for brake, steering, and throttle.
- Model / Inference Device / Deterministic Inference when a trained model is loaded.

## Baseline configuration
Source: `results/configuration_example.yaml`

| Parameter | Value |
|---|---|
| `trainer_type` | `ppo` |
| `batch_size` | `512` |
| `buffer_size` | `10240` |
| `learning_rate` | `3e-4` |
| `beta` | `0.005` |
| `epsilon` | `0.2` |
| `lambd` | `0.95` |
| `num_epoch` | `3` |
| `gamma` | `0.99` |
| `hidden_units` | `512` |
| `num_layers` | `3` |
| `time_horizon` | `2048` |

## How to launch
1. Open Unity and the car scene.
2. Check the agent parameters in the inspector.
3. Launch training from the terminal:

```bash
mlagents-learn results/configuration_example.yaml --run-id=Experience1 --force
```

4. Press Play in Unity.

## E1 to do first
- Use the track1 scene.
- Keep the baseline configuration for the first run.
- Record the first curves before changing anything.
- For the report, describe the baseline behaviour and the first observations only.

## Suggested scope
- Do 3 runs in total: E1 baseline, E2 simple change, E3 final compromise.
- Keep the changes small so the comparison stays readable.

## E2 objective (recommended)
- Drive faster and cleaner: reduce crashes and improve average lap time.
- Measure on 5 inference attempts:
	- average lap time,
	- number of crashes,
	- number of completed laps.

## E2 parameter proposal
- `batch_size`: `1024`
- `learning_rate`: `1.5e-4`
- `epsilon`: `0.15`
- `beta`: `0.003`
- Keep `lambd=0.95` and `num_epoch=3`.

Rationale:
- Larger batch plus lower learning rate generally improves update stability.
- Lower epsilon clip reduces aggressive policy jumps.
- Slightly lower beta reduces random behavior and helps avoid wall crashes.

## Run naming (recommended)
- E1: `--run-id=Experience1`
- E2: `--run-id=Experience2`
- E3: `--run-id=Experience3`

Using a different run id for each experiment avoids overwriting previous results and makes TensorBoard comparison easier.

## What to compare between runs
- Success rate.
- Reward trend.
- Episode duration.
- Stability of the curves.
- Effect of batch size and learning rate.

## Images to comment
- Car starting correctly in the track1 scene.
- Car reaching the target.
- Car crashing or leaving the road.
- Reward curve after E1.
- Episode length curve after E1.

## When to stop a run
- Use TensorBoard to monitor the run.
- For E1, stop when cumulative reward has clearly improved and starts to plateau, or when extra runtime gives little visible gain.
- Practical stop method: press `Ctrl+C` in the `mlagents-learn` terminal, then stop Play in Unity.

## Timing to record in the report
- Start time.
- End time.
- Total duration.
- Final step and short note on observed behaviour.

## TensorBoard
- Compare all runs:

```bash
tensorboard --logdir "C:\Users\Ajax\AI_in_games_course\results" --port 6006
```

- Show only E1:

```bash
tensorboard --logdir "C:\Users\Ajax\AI_in_games_course\results\Experience1" --port 6006
```

## After training
- The model to test in Unity is the `.onnx` file from the run folder (typically `results/Experience1/car_agent.onnx`).
- Drag this file into the agent Behavior Parameters `Model` field and run the scene in inference mode.

## Short analysis rule
- If batch size is larger: training updates are heavier, but gradients are usually more stable.
- If batch size is smaller: training is lighter, but curves can be noisier.
- If learning rate is too high: PPO can become unstable.
- If learning rate is too low: learning can be slow.

## Checklist
- [ ] Unity scene ready
- [ ] PPO config checked
- [ ] Agent parameters checked in inspector
- [ ] First training run launched
- [ ] Curves and screenshots saved
