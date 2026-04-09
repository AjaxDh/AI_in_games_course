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
mlagents-learn results/configuration_example.yaml --run-id=my_agent --force
```

4. Press Play in Unity.

## What to compare between runs
- Success rate.
- Reward trend.
- Episode duration.
- Stability of the curves.
- Effect of batch size and learning rate.

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
