"""Running in DeepMind Control Suite Env"""
import random
import torch
import argparse
import os
import time
import numpy as np
from gym.spaces import Box, Discrete
# -------------------------------
from VPG import VPG
# -------------------------------
from PPO import PPO
from PPO import PPO2
# -------------------------------
from utils import replay_buffer
import environments
# Tag loggers
from spinupUtils.logx import EpochLogger
from spinupUtils.run_utils import setup_logger_kwargs

def test_agent(policy, eval_env, logger, eval_episodes=10):
	for _ in range(eval_episodes):
		episode_timesteps = 0
		state, done, ep_ret, ep_len = eval_env.reset(), False, 0, 0
		while not done:
			episode_timesteps += 1
			scaled_action, _, _, _ = policy.select_action(np.array(state), deterministic=True)
			state, reward, done, _ = eval_env.step(scaled_action)
			timeout_done = (episode_timesteps == env.max_episode_steps)
			done = timeout_done or done
			ep_ret += reward
			ep_len += 1
		logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="VPG", type=str)         # Policy name
	parser.add_argument("--env", default="cheetah-run", type=str)    # DeepMind Control Suite environment name
	parser.add_argument("--seed", default=0, type=int)               # Sets DeepMind Control Suite env, PyTorch and Numpy seeds
	parser.add_argument("--steps_per_epoch", default=2048, type=int) # steps per epoch
	parser.add_argument("--epochs", default=1465, type=int)          # Max epochs to run environment
	parser.add_argument("--discount", default=0.99, type=float)      # `\gamma`, Discount factor
	parser.add_argument("--lam", default=0.95, type=float)           # `\lambda`, GAE discount factor
	parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
	parser.add_argument("--save_freq", default=10, type=int)         # How often (evaluation steps) we save the model
	parser.add_argument("--exp_name", type=str)       				 # Name for algorithms
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_s{args.seed}"
	print(f"---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print(f"---------------------------------------")

	# Make envs
	env = environments.ControlSuite(args.env)
	eval_env = environments.ControlSuite(args.env)

	# Set seeds
	env.seed(args.seed)
	eval_env.seed(args.seed)  # eval env for evaluating the agent
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_space = env.action_space
	if isinstance(action_space, Box):
		action_dim = action_space.shape[0]
		is_discrete=False
	elif isinstance(action_space, Discrete):
		action_dim = action_space.n
		is_discrete=True
	else:
		assert f"type of action shape must be `gym.spaces.Box` or `gym.spaces.Discrete`!"

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"is_discrete": is_discrete,
	}
	if not is_discrete:
		kwargs["max_action"] = float(action_space.high[0])

	# Initialize policy
	# ----------------------------------------------
	if args.policy == "VPG":
		policy = VPG.VPG(**kwargs)
	# ----------------------------------------------
	elif args.policy == "PPO":
		policy = PPO.PPO(**kwargs)
	elif args.policy == "PPO2":
		policy = PPO2.PPO(**kwargs)
	else:
		raise ValueError(f"Invalid Policy: {args.policy}!")

	if args.save_model and not os.path.exists(f"./models/{file_name}"):
		os.makedirs(f"./models/{file_name}")

	# Setup loggers
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=False)
	logger = EpochLogger(**logger_kwargs)

	# Set up experience buffer
	local_steps_per_epoch = int(args.steps_per_epoch)
	_replay_buffer = replay_buffer.VPGBuffer(
		state_dim, action_dim, local_steps_per_epoch, args.discount, args.lam, is_discrete)
	
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	start_time = time.time()

	for epoch in range(args.epochs):
		for t in range(local_steps_per_epoch):
			episode_timesteps += 1

			scaled_action, action, logp_pi, v = policy.select_action(state)
			# Perform action
			next_state, reward, done, _ = env.step(scaled_action) 
			epoch_done = (t == local_steps_per_epoch - 1)
			timeout_done = (episode_timesteps == env.max_episode_steps)
			terminal = done or timeout_done

			# Store data in replay buffer
			_replay_buffer.add(state, action, reward, v, logp_pi)

			state = next_state
			episode_reward += reward

			if terminal or epoch_done:
				if epoch_done and not(terminal):
					print(f"Warning: trajectory cut off by local epoch at {episode_timesteps} steps.", flush=True)
				if timeout_done or epoch_done:
					_, _, _, v = policy.select_action(state)
				elif done:
					v = 0
				_replay_buffer.finish_path(v)
				if terminal:
					# only save EpRet / EpLen if trajectory finished
					logger.store(EpRet=episode_reward, EpLen=episode_timesteps)
				# Reset environment
				state, done = env.reset(), False
				episode_reward = 0
				episode_timesteps = 0

		# perform VPG update
		policy.train(_replay_buffer)
		
		test_agent(policy, eval_env, logger)
		if args.save_model and (epoch + 1) % int(args.save_freq) == 0: 
			policy.save(f"./models/{file_name}/{(epoch+1) * args.steps_per_epoch}_steps")
		logger.log_tabular("EpRet", with_min_and_max=True)
		logger.log_tabular("TestEpRet", with_min_and_max=True)
		logger.log_tabular("EpLen", average_only=True)
		logger.log_tabular("TestEpLen", average_only=True)
		logger.log_tabular("TotalEnvInteracts", (epoch+1)*args.steps_per_epoch)
		logger.log_tabular("Time", time.time()-start_time)
		logger.dump_tabular()