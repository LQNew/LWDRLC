import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Soft Actor-Critic Algorithms and Applications
# https://arxiv.org/abs/1812.05905

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain("relu")
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability."""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

# Returns continuous actions for given states
class Actor(nn.Module):
	def __init__(
		self, 
		state_dim: int, 
		action_dim: int, 
		max_action: float, 
		hidden_dim: int = 256
	):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.mean = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Linear(hidden_dim, action_dim)
		self.max_action = max_action
		# self.soft_plus = nn.Softplus()
		self.apply(weight_init)

	def forward(self, state, deterministic=False, with_logprob=True):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu_a = self.mean(a)
		log_std_a = self.log_std(a)
		log_std_a = torch.clamp(log_std_a, LOG_STD_MIN, LOG_STD_MAX)
		std_a = torch.exp(log_std_a)
		# Only used for evaluating policy at test time.
		if deterministic:
			z = mu_a
		else:
			noise = torch.randn_like(mu_a, requires_grad=True)  # sampled from guassian distribution
			z = mu_a + noise * std_a  # reparameterization trick
		action = torch.tanh(z) 

		if with_logprob and not deterministic:
			logp_pi = gaussian_logprob(noise, log_std_a).sum(axis=-1)
			# logp_pi -= (
			# 	np.log(2.0) - (action + self.soft_plus(-2.0 * action)).sum(axis=-1)
			# ) * 2.0
			logp_pi = logp_pi - (1.0 - action**2).clamp(min=1e-6).log().sum(axis=-1)
		else:
			logp_pi = None
		return self.max_action * action, logp_pi


# Returns Q-value for given state/action pairs
class Critic(nn.Module):
	def __init__(
		self, 
		state_dim: int, 
		action_dim: int, 
		hidden_dim: int = 256
	):
		super(Critic, self).__init__()
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)
		self.apply(weight_init)

	def forward(self, state, action):
		sa = torch.cat([state, action], dim=1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


class SAC(object):
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		max_action: float = 1.0,
		discount: float = 0.99,
		tau: float = 0.005,
		hidden_dim: int = 256,
	):
		self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau

		# self.log_alpha = torch.tensor(
		# 	(-np.log(action_dim) * np.e,), dtype=torch.float32,
		# 	requires_grad=True, device=device
		# )
		self.log_alpha = torch.tensor(
			np.log(1.), dtype=torch.float32,
			requires_grad=True, device=device
		)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
		# self.target_entropy = np.log(action_dim)
		self.target_entropy = -action_dim

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def select_action(self, state, deterministic=False):
		state = torch.as_tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
		action, _ = self.actor(state, deterministic, False)
		return action.cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		# Sample batches from replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		with torch.no_grad():
			# Select action according to policy 
			next_action, logp_pi_next_action = self.actor(next_state)
			logp_pi_next_action = torch.unsqueeze(logp_pi_next_action, dim=1)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * (target_Q - self.alpha.detach() * logp_pi_next_action)

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		action, logp_pi_action = self.actor(state)
		logp_pi_action = torch.unsqueeze(logp_pi_action, dim=1)
		Q1_pi, Q2_pi = self.critic(state, action)
		Q_pi = torch.min(Q1_pi, Q2_pi)
		actor_loss = (self.alpha.detach() * logp_pi_action - Q_pi).mean()
		
		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Alpha loss
		self.alpha_optimizer.zero_grad()
		alpha_loss = -(self.log_alpha * (logp_pi_action + self.target_entropy).detach()).mean()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# save the model
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")

	# load the model
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
		self.critic_target = copy.deepcopy(self.critic)
		self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))