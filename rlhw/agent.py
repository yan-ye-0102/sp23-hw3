from typing import Dict, Any, Tuple, Callable
from collections import defaultdict

import gym
import numpy as np
import numpy.typing as npt
from rlhw.base import BaseAgent


class RandomAgent(BaseAgent):
	def __init__(self, build_env: Callable[[bool], gym.Env], params: Dict[str, Any]):
		BaseAgent.__init__(self, build_env, params)
		self.env = build_env(False)
		self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))

	def get_action(self) -> int:
		return self.env.action_space.sample()

	def step(self, action: int) -> Tuple[npt.NDArray, float, bool]:
		next_state, reward, done, _, _ = self.env.step(action)
		return next_state, reward, done

	def learn(self):
		"""A random agent does not need to learn"""
		pass

	def run(self, max_episodes: int, max_steps: int, train: bool):
		if train:
			self.env = self.build_env(render=False)
		else:
			self.env = self.build_env(render=True)

		episode_rewards = dict()

		for ne in range(max_episodes):
			state, _ = self.env.reset()
			done = False
			total_reward = 0

			for nt in range(max_steps):
				action = self.get_action()
				next_state, reward, done = self.step(action)
				total_reward += reward

				if train:
					self.learn()

				if done:
					break

				state = next_state

			episode_rewards[ne] = total_reward
			print(f"episode {ne}: {total_reward}")

		return episode_rewards

class QLearningAgent(BaseAgent):
	def __init__(self, build_env: Callable[[bool], gym.Env], params: Dict[str, Any]):
		BaseAgent.__init__(self, build_env, params)
		self.env = build_env(render=False)
		self.render = self.params["render"]
		self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))

		self.lr = self.params["learning_rate"]
		self.min_lr = self.params["min_learning_rate"]
		self.lr_decay = self.params["learning_rate_decay"]
		self.gamma = self.params["discount_factor"]
		self.eps = self.params["epsilon"]
		self.eps_decay = self.params["epsilon_decay"]
		self.min_eps = self.params["min_epsilon"]

	def get_action(self, state: npt.NDArray, train: bool) -> int:
		"""Return an action given the current state"""

		if np.random.uniform(0, 1) < self.eps * train:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.policy[state, :])
		# YOUR CODE HERE

	def step(self, action: int) -> Tuple[npt.NDArray, float, bool]:
		next_state, reward, done, _, _ = self.env.step(action)
		return next_state, reward, done

	def learn(
		self, state: npt.NDArray, action: int, reward: float, next_state: npt.NDArray
	) -> float:
		"""Update Agent's policy using the Q-learning algorithm and return the update delta"""
		print(max(self.policy[next_state, :]))
		self.policy[state, action] = (1 - self.lr) * self.policy[state, action] + self.lr * (reward + self.gamma * max(self.policy[next_state, :]))
		return (reward + self.gamma * max(self.policy[next_state, :])) - self.policy[state, action]

		# YOUR CODE HERE

	def run(self, max_episodes: int, max_steps: int, train: bool):
		"""Run simulations of the environment with the agent's policy"""
		if self.render and not train:
			self.env = self.build_env(render=True)

		# YOUR CODE HERE

		episode_rewards = dict()

		for ne in range(max_episodes):

			total_rewards = 0
			cur_state = self.env.reset()
			done = False

			for _ in range(max_steps):
				action = self.get_action(cur_state, train)

				next_state, reward, done = self.step(action)

				self.learn(cur_state, action, reward, next_state)

				total_rewards += reward

				if done:
					break
				cur_state = next_state
			
			self.eps = max(self.min_eps, np.exp(-self.eps_decay * ne))
			episode_rewards[ne] = total_rewards
		return episode_rewards