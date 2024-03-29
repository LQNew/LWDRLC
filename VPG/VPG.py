import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Vanilla Policy Gradient (VPG)

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


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability."""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


# Returns discrete actions for given states
class DiscreteActor(nn.Module):
	def __init__(
		self, 
		state_dim: int, 
		action_dim: int, 
		hidden_dim: int = 64
	):
		super(DiscreteActor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.logits = nn.Linear(hidden_dim, action_dim)

		self.action_dim = action_dim
		self.apply(weight_init)

	def forward(self, state, action=None):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		logits_a = self.logits(a)  # batch_size * action_dim
		logp_all = F.log_softmax(logits_a, dim=-1)  # log(prob_i) = log(e^i / sum(e^j))
		probs_a = F.softmax(logp_all, dim=-1)  # prob_i = e^i / sum(e^j)
		# pick actions from multinomial distribution according to `probs_a`
		pi = torch.squeeze(torch.multinomial(probs_a, 1), axis=1)  
		if action is None:
			logp_a = None
		else:
			logp_a = torch.sum(F.one_hot(action.long(), num_classes=self.action_dim) * logp_all, dim=-1)
		logp_pi = torch.sum(F.one_hot(pi, num_classes=self.action_dim) * logp_all, dim=-1)
		return pi, logp_pi, logp_a


# Returns continuous actions for given states
class ContinuousActor(nn.Module):
	def __init__(
		self, 
		state_dim: int, 
		action_dim: int, 
		hidden_dim: int = 64
	):
		super(ContinuousActor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.mean = nn.Linear(hidden_dim, action_dim)

		log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
		self.log_std = nn.Parameter(torch.as_tensor(log_std, dtype=torch.float32))  # transfer to trained params
		self.EPS = 1e-8
		self.apply(weight_init)

	def forward(self, state, action=None, deterministic=False):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = self.mean(a)
		std = torch.exp(self.log_std)

		if action is None:
			logp_a = None
		else:
			noise_action = (action - mu) / (std + self.EPS)
			logp_a = gaussian_logprob(noise_action, self.log_std).sum(axis=-1)

		if not deterministic:
			noise = torch.randn_like(mu)  # sampled from guassian distribution
			pi = mu + noise * std
			logp_pi = gaussian_logprob(noise, self.log_std).sum(axis=-1)
		else:
			pi = mu
			logp_pi = None

		return pi, logp_pi, logp_a


# Return V-value for given states
class Critic(nn.Module):
	def __init__(self, state_dim: int, hidden_dim: int = 64):
		super(Critic, self).__init__()
		# V architecture
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

	def forward(self, state):
		z = torch.tanh(self.l1(state))
		z = torch.tanh(self.l2(z))
		v = self.l3(z)
		return v


class VPG(nn.Module):
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		max_action: float = 1.,
		hidden_dim: int = 64,
		is_discrete: bool = False,
		max_grad_norm: float = 4.0,
	):
		super(VPG, self).__init__()

		if is_discrete:
			self.actor = DiscreteActor(state_dim, action_dim, hidden_dim).to(device)
		else:
			self.actor = ContinuousActor(state_dim, action_dim, hidden_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, hidden_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.max_action = max_action
		self.is_discrete = is_discrete
		self.max_grad_norm = max_grad_norm

	def select_action(self, state, deterministic=False):
		state = torch.as_tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
		action, logp_pi, _ = self.actor(state, deterministic=deterministic)
		v = self.critic(state)
		if not self.is_discrete:
			scaled_action = self.max_action * torch.tanh(action)
			if not deterministic:
				return scaled_action.data.cpu().numpy().flatten(), action.data.cpu().numpy().flatten(), logp_pi.data.cpu().numpy(), v.data.cpu().numpy()
			else:
				return scaled_action.data.cpu().numpy().flatten(), action.data.cpu().numpy().flatten(), None, v.data.cpu().numpy()
		else:
			# for discrete action space, only need `int item` not `numpy array`
			return action.data.cpu().numpy().flatten()[0], action.data.cpu().numpy().flatten()[0], logp_pi.data.cpu().numpy(), v.data.cpu().numpy()

	def train(self, replay_buffer, train_v_iters=80):
		# Sampled from replay buffer
		for state, action, g_reward, adv_value, _ in replay_buffer.sample():
			# Compute actor loss
			_, _, logp_a = self.actor(state, action)
			actor_loss = -(logp_a * adv_value).mean()  # policy gradient 
			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			# Clip grad norm
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
			self.actor_optimizer.step()
		
		for _ in range(train_v_iters):
			# Sampled from replay buffer 
			for state, action, g_reward, _, _ in replay_buffer.sample():
				current_V = self.critic(state)  # Get current Q estimates
				# Compute critic loss
				critic_loss = F.mse_loss(current_V, torch.unsqueeze(g_reward, dim=1))
				# Optimize the critic
				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				# Clip grad norm
				torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				self.critic_optimizer.step()

	# save the model
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")

	# load the model
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
		self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
