"""Visualization of the MuJoCo environments."""
import numpy as np
import random
import torch
import gym
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)      # Sets Gym, PyTorch and Numpy seeds
	args = parser.parse_args()

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	
	for episode_num in range(10):
		state, done, ep_ret, ep_len = env.reset(), False, 0, 0
		while not done:
			action = env.action_space.sample()
			state, reward, done, _ = env.step(action)
			env.render()
			ep_ret += reward
			ep_len += 1
		
		if done:
			episode_num += 1
			print(f"Episode Num: {episode_num} Episode T: {ep_len} Reward: {ep_ret:.3f}")
