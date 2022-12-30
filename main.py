import warnings
warnings.filterwarnings("ignore")  # warnings ignored as this is for reproducability

import minari
import wandb
import numpy as np
import torch
import gym
import argparse
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import d4rl
import d4rl.gym_mujoco # Import required to register environments

import utils
from algorithms import TD3_BC


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	return d4rl_score


if __name__ == "__main__":
	warnings.filterwarnings("ignore")  # warnings ignored as this is for reproducability

	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper")                  # OpenAI gym environment name
	parser.add_argument("--extension", default="random-v0")         #
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e4, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	wandb.init(project="scrapyard", # "Offline-RL-benchmarks-model-testing",
			   entity="willdudley",
			   config=args,
			   save_code=True,
			   tags=[args.policy, args.env],
			   name=file_name,
			   id=file_name)

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	dataset_name = f"D4RL-{args.env}-{args.extension[:-3]}_{args.extension[-2:]}_Legacy-D4RL-dataset"
	dataset = minari.download_dataset(dataset_name)
	env = gym.make(dataset.environment_name.decode("utf-8"))

	env_old = gym.make(f"{args.env}-{args.extension}")

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(dataset)

	replay_buffer_old = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer_old.convert_D4RL(d4rl.qlearning_dataset(env_old), minari=False)
	replay_buffer.state == replay_buffer_old.state
	replay_buffer.action == replay_buffer_old.action
	replay_buffer.next_state == replay_buffer_old.next_state
	replay_buffer.reward == replay_buffer_old.reward
	replay_buffer.not_done == replay_buffer_old.not_done

	artifact = wandb.Artifact(args.env, type='dataset')
	artifact.add_reference("https://storage.cloud.google.com/minari/D4RL-hopper-random_v0_Legacy-D4RL-dataset.hdf5")
	wandb.log_artifact(artifact)

	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	evaluations = []
	for t in range(int(args.max_timesteps)):
		policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			score = eval_policy(policy, f"{args.env}-{args.extension}", args.seed, mean, std)
			evaluations.append(score)
			wandb.log({"Score": score}, step=t)

			trained_model_artifact = wandb.Artifact(name=file_name,
													type='model',
													description=None,
													metadata=None,
													incremental=None,
													use_as=None)

			policy.save(f"./models/{file_name}.pth")
			trained_model_artifact.add_file(f"./models/{file_name}.pth")
			wandb.log_artifact(trained_model_artifact)
