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

## E3 Results (Final Compromise)

**E3 Configuration:**
- `batch_size`: 768 (between E1=512 and E2=1024)
- `learning_rate`: 2.2e-4 (between E1=3e-4 and E2=1.5e-4)
- `epsilon`: 0.17, `beta`: 0.004
- `max_steps`: 650000

**Observed Results:**
- Duration: ~50 minutes (efficient)
- Cumulative Reward: -2.53 → 55.77 (step 650k)
- Value Loss: 0.005 → 1.22 (stable, best of all three)
- Policy Loss: 0.025-0.027 (stable)
- Reward curve: **clean progression, strong gain, minimal oscillations**
- Inference: **Most stable**, rarely crashes, avoids tight curves without wall hits, lap time 20.91s

**Key Achievement:**
E3 achieves the optimal compromise:
- Stability: Better than E2 (value loss 1.22 vs 1.47)
- Performance: Faster than E1 (20.91s vs 23.26s)
- Robustness: Significantly better inference success rate
- Learning quality: Smoother, less noisy progression

**Recommendation for Report:**
E3 is the **best configuration to retain** for the final project. It balances learning speed (comparable to E2's aggressive approach) with stability (matching or exceeding E1's robustness).

If more training time available: E3 could likely reach reward ~57-58 with extended runtime, but 50 min provides good return-on-compute-time.

## Checklist
- [x] Unity scene ready
- [x] PPO config checked
- [x] Agent parameters checked in inspector
- [x] E1 training run launched & analyzed (46 min, baseline)
- [x] E2 training run launched & analyzed (52 min, aggressive learning)
- [x] E3 training run launched & analyzed (50 min, optimal compromise)
- [x] Curves and screenshots saved
- [ ] Report conclusion written
- [ ] Final configuration selected (recommend E3)
