from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import csv
import json
import os
import shutil
from datetime import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import rolling_ball_DQN
import torch

try:
	import msvcrt
	HAS_MSVCRT = True
except ImportError:
	HAS_MSVCRT = False

RESULTS_DIR = os.path.join("results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_CSV_PATH = os.path.join(RESULTS_DIR, "rolling_ball_training_log.csv")
JOURNAL_PATH = os.path.join(RESULTS_DIR, "rolling_ball_experiment_journal.md")
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "rolling_ball_run_summary.json")
HISTORY_NPZ_PATH = os.path.join(RESULTS_DIR, "rolling_ball_episode_history.npz")
CHECKPOINT_EVERY = 0


def main():
	# Parameters
	input_size = 9
	output_size = 5
	batch_size = 128
	gamma = 0.99
	F = 300
	lr = 7e-5
	eps_start = 0.9
	eps_end = 0.02
	eps_decay = 2000
	n_episode = 250

	os.makedirs(RESULTS_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	run_started_at = datetime.now()
	run_start_perf = time.perf_counter()
	brain = None
	gym = None
	paused = False

	print("Starting simulation. Press the Play button in Unity editor")
	print("Controls: Ctrl+P to pause/resume")
	unity_env = UnityEnvironment(file_name=None)
	gym = UnityToGymWrapper(unity_env=unity_env, allow_multiple_obs=True)

	try:
		rewards, reward_episode, episode_duration, episode_elapsed = [], [], [], []
		step = 0
		total_steps = 0
		total_episode = 0

		# Init Deep Q-learning
		brain = rolling_ball_DQN.Dqn(input_size=input_size,
								 output_size=output_size,
								 batch_size=batch_size,
								 gamma=gamma,
								 lr=lr,
								 F=F,
								 eps_start=eps_start,
								 eps_end=eps_end,
								 eps_decay=eps_decay)

		# Start simulation
		reset_gym(gym, brain)

		for episode in range(n_episode):
			episode_start_perf = time.perf_counter()
			episode_reward = 0.0
			while True:
				paused = poll_training_controls(paused)
				if paused:
					wait_while_paused()
					episode_start_perf = time.perf_counter()

				# Advance one step in the environment
				step += 1
				action = brain.select_actions(brain.state, True)
				observation, reward, terminated, _ = gym.step(action.item())
				observation = observation[0]

				# Update DQN with new environment state
				brain.update(reward, observation, terminated, True)
				rewards.append(reward)
				episode_reward += reward

				# If agent reached end of episode
				if terminated:
					reset_gym(gym, brain)
					reward_episode.append(episode_reward)
					episode_duration.append(step)
					episode_elapsed.append(time.perf_counter() - episode_start_perf)
					nb_episode = len(reward_episode)
					total_steps += step
					total_episode += 1
					print(f"Episode: {nb_episode} / Step: {total_steps} / Episode reward: {episode_reward:.3f} / Duration: {step} / Elapsed: {episode_elapsed[-1]:.1f}s")
					append_episode_log(LOG_CSV_PATH, run_started_at, nb_episode, total_steps, episode_reward, step, episode_elapsed[-1], reward)
					if CHECKPOINT_EVERY > 0 and nb_episode % CHECKPOINT_EVERY == 0:
						brain.save()
						checkpoint_path = os.path.join(CHECKPOINT_DIR, f"rolling_ball_ep_{nb_episode:04d}.pth")
						shutil.copy2("last_brain.pth", checkpoint_path)
					step = 0
					break

	finally:
		if gym is not None:
			gym.close()
		if brain is not None:
			brain.save()
		run_total_seconds = time.perf_counter() - run_start_perf
		save_run_artifacts(run_started_at, input_size, output_size, batch_size, gamma, F, lr, eps_start, eps_end, eps_decay, n_episode, reward_episode if brain is not None else [], episode_duration if brain is not None else [], episode_elapsed if brain is not None else [], total_steps if 'total_steps' in locals() else 0, run_total_seconds)
		plot_reward_history(reward_episode if brain is not None else [], episode_duration if brain is not None else [], total_steps if 'total_steps' in locals() else 0)
		print("Simulation Terminated!")


def reset_gym(gym, brain):
	"""Reset gym environment after the end of an episode."""
	state = gym.reset()
	state = torch.tensor(state[0], dtype=torch.float32, device=brain.device).unsqueeze(0)
	brain.state = state


def plot_reward_history(rewards, durations, steps):
	"""End of simulation plot."""
	average_window = 20

	if len(rewards) < average_window:
		print("Not enough episode to plot results")
		return

	mean_rewards, mean_durations = [], []
	for i in range(0, average_window):
		mean_rewards.append(np.mean(rewards[0:i]))
		mean_durations.append(np.mean(durations[0:i]))
	for i in range(average_window, len(rewards)):
		mean_rewards.append(np.mean(rewards[i-average_window:i]))
		mean_durations.append(np.mean(durations[i-average_window:i]))

	fig, axs = plt.subplots(2, 1, figsize=(16, 9))
	fig.suptitle(f"Rolling ball info for {len(rewards)} episodes and {steps} steps")
	axs[0].plot(rewards)
	axs[0].plot(mean_rewards)
	axs[0].set(ylabel=" Last reward", xlabel="Episode")
	axs[0].grid()
	axs[1].plot(durations)
	axs[1].plot(mean_durations)
	axs[1].set(ylabel="Duration (steps)", xlabel="Episode")
	axs[1].grid()
	plt.savefig("rolling_ball_reward.png")
	plt.show()


def poll_training_controls(paused):
	"""Poll keyboard controls during training.

	Ctrl+P toggles pause/resume.
	"""
	if not HAS_MSVCRT:
		return paused

	while msvcrt.kbhit():
		key = msvcrt.getch()
		# Ctrl+P sends ASCII 0x10 in Windows terminals.
		if key == b'\x10':
			paused = not paused
	return paused


def wait_while_paused():
	"""Block while paused until the user presses Ctrl+P again."""
	if not HAS_MSVCRT:
		return

	paused = True
	print("Training paused. Press Ctrl+P to resume.")
	while paused:
		paused = poll_training_controls(paused)
		time.sleep(0.1)
	print("Training resumed.")
	return


def append_episode_log(csv_path, run_started_at, episode_index, total_steps, reward_value, duration, elapsed_seconds, last_step_reward):
	"""Append one episode line to the persistent CSV log."""
	file_exists = os.path.isfile(csv_path)
	with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		if not file_exists:
			writer.writerow([
				"run_started_at",
				"episode",
				"total_steps",
				"episode_reward",
				"duration_steps",
				"elapsed_seconds",
				"terminal_reward",
			])
		writer.writerow([
			run_started_at.isoformat(timespec="seconds") if hasattr(run_started_at, "isoformat") else run_started_at,
			episode_index,
			total_steps,
			reward_value,
			duration,
			round(elapsed_seconds, 3),
			last_step_reward,
		])


def save_run_artifacts(run_started_at, input_size, output_size, batch_size, gamma, F, lr, eps_start, eps_end, eps_decay, n_episode, rewards, durations, elapsed_times, total_steps, run_total_seconds):
	"""Save a summary JSON, a markdown journal entry, and the raw episode arrays."""
	os.makedirs(RESULTS_DIR, exist_ok=True)
	summary = {
		"run_started_at": run_started_at.isoformat(timespec="seconds"),
		"input_size": input_size,
		"output_size": output_size,
		"batch_size": batch_size,
		"gamma": gamma,
		"F": F,
		"lr": lr,
		"eps_start": eps_start,
		"eps_end": eps_end,
		"eps_decay": eps_decay,
		"n_episode": n_episode,
		"total_steps": total_steps,
		"run_total_seconds": round(run_total_seconds, 3),
		"episodes_completed": len(rewards),
		"mean_episode_reward": float(np.mean(rewards)) if rewards else None,
		"mean_episode_duration": float(np.mean(durations)) if durations else None,
		"mean_episode_elapsed_seconds": float(np.mean(elapsed_times)) if elapsed_times else None,
	}

	with open(SUMMARY_JSON_PATH, mode="w", encoding="utf-8") as summary_file:
		json.dump(summary, summary_file, indent=2)

	np.savez(
		HISTORY_NPZ_PATH,
		rewards=np.array(rewards),
		durations=np.array(durations),
		elapsed_seconds=np.array(elapsed_times),
	)

	with open(JOURNAL_PATH, mode="a", encoding="utf-8") as journal_file:
		journal_file.write(f"## Run {summary['run_started_at']}\n")
		journal_file.write(f"- Episodes completed: {summary['episodes_completed']}\n")
		journal_file.write(f"- Total steps: {summary['total_steps']}\n")
		journal_file.write(f"- Mean episode reward: {summary['mean_episode_reward']}\n")
		journal_file.write(f"- Mean episode duration: {summary['mean_episode_duration']}\n")
		journal_file.write(f"- Mean episode time (s): {summary['mean_episode_elapsed_seconds']}\n")
		journal_file.write(f"- Runtime total (s): {summary['run_total_seconds']}\n")
		journal_file.write(f"- Hyperparameters: batch_size={batch_size}, gamma={gamma}, F={F}, lr={lr}, eps_start={eps_start}, eps_end={eps_end}, eps_decay={eps_decay}, n_episode={n_episode}\n")
		journal_file.write("\n")


def print_gym_status(action, observation, reward, terminated):
	"""Print state information."""
	print("[Gym infos]")
	print(f"Action: {action}")
	print(f"Obs: {observation}")
	print(f"Reward: {reward}")
	print(f"Terminated: {terminated}")


if __name__ == '__main__':
	main()