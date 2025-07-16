import random
import time
from pathlib import Path
from typing import Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import FlattenObservation

import wandb

from .replay_buffer import ReplayBuffer
from .utils import calc_running_statistics, set_seed
from .wrappers import FFmpegVideoRecorder


class BaseDQNAgent:
    def __init__(
        self,
        env_name,
        env_config={},
        default_batch_size=64,
        replay_buffer_capacity=1000000,
        gamma=0.99,
        learning_rate=1e-4,
        tau=0.005,
        device="cuda" if torch.cuda.is_available() else "cpu",
        wandb_run=None,
        log_index="step",
        video_log_path_dir="videos",
        seed: int = 42,
        gradient_norm_clip: Union[float, None] = None,
        start_training_after: int = 1000,
        clip_rewards: Union[float, None] = 10,
        normalize_rewards: bool = True,
        scale_rewards: Union[float, None] = None,
    ):
        set_seed(seed)

        self.video_log_path = Path(video_log_path_dir)
        self.video_log_path.mkdir(parents=True, exist_ok=True)

        if env_name == "FrozenLake-v1":
            size = 16
            p = 0.6
            if env_config is not None:
                size = env_config.get("size", 16)
                p = env_config.get("p", 0.6)
            desc = generate_random_map(size=size, p=p, seed=seed)
            self.env = FlattenObservation(gym.make(env_name, desc=desc))
            self.eval_env = FFmpegVideoRecorder(
                FlattenObservation(gym.make(env_name, desc=desc, render_mode="rgb_array")),
                video_folder=str(self.video_log_path.resolve()),
                fps=4,
            )
        else:
            self.env = FlattenObservation(gym.make(env_name))
            self.eval_env = FFmpegVideoRecorder(
                FlattenObservation(gym.make(env_name, render_mode="rgb_array")),
                video_folder=str(self.video_log_path.resolve()),
                fps=30,
            )

        self.default_batch_size = default_batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        self.device = device
        self.wandb_run = wandb_run
        self.log_index = log_index
        self.gradient_norm_clip = gradient_norm_clip
        self.start_training_after = start_training_after
        self.clip_rewards = clip_rewards
        self._norm_rew = normalize_rewards
        self._scale_rew = scale_rewards

        self._training_step = 0
        self._training_episode = 0
        self._cur_rollout_step = 0
        self._total_steps = 0

        self._obs_mean = torch.zeros(self.env.observation_space.shape, device=self.device, requires_grad=False)
        self._obs_var = torch.ones(self.env.observation_space.shape, device=self.device, requires_grad=False)
        self._obs_m2 = torch.zeros(self.env.observation_space.shape, device=self.device, requires_grad=False)
        self._ret_mean = torch.zeros((1,), device=self.device, requires_grad=False)
        self._ret_var = torch.ones((1,), device=self.device, requires_grad=False)
        self._ret_m2 = torch.zeros((1,), device=self.device, requires_grad=False)

        self._actions = []
        self._rewards = []

        self._create_replay_buffer(replay_buffer_capacity)

        self._create_network()
        self._hard_update_target_network()
        self._create_optimizer()

        self.train_mode()

    def eval_mode(self):
        """
        Set the agent to evaluation mode.
        Disables training-specific operations and sets networks to evaluation mode.
        """
        self.training = False
        self.q_network.eval()
        self.target_network.eval()
        self._loss = 0
        self._episode_loss = 0
        self._rewards.clear()
        self._cur_rollout_step = 0
        self._actions.clear()

    def train_mode(self):
        """
        Set the agent to training mode.
        Enables training-specific operations and prepares the agent for learning.
        """
        self.training = True
        self.q_network.train()
        self._loss = 0
        self._episode_loss = 0
        self._rewards.clear()
        self._cur_rollout_step = 0

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer after preprocessing.
        """
        data = self._preprocess_add(state, action, reward, next_state, done)
        self.replay_buffer.add(**data)

    def act(self, state):
        """
        Select an action based on the current state.
        Updates observation statistics when in training mode.
        Args:
            state (np.ndarray): The current state.
        """
        state = self._state_transformation(state)
        if self.training:
            self._training_step += 1
            self._obs_mean, self._obs_var, self._obs_m2 = calc_running_statistics(
                torch.tensor(state, device=self.device),
                self._obs_mean,
                self._obs_m2,
                self._training_step,
            )
            action = self._act_in_training(state)
        else:
            action = self._act_in_eval(state)
        return action

    def _step(self, reward):
        """
        Process a timestep by logging the reward and updating internal counters.
        """
        self._rewards.append(reward)
        self._cur_rollout_step += 1
        self._total_steps += 1

    def _episode(self):
        """
        Process an episode: log performance metrics, update training statistics, and reset episode counters.
        """
        save_dict = self._wandb_train_episode_dict()
        if self.wandb_run is not None:
            if self.log_index == "step":
                self.wandb_run.log(save_dict, step=self._training_step)
            elif self.log_index == "episode":
                self.wandb_run.log(save_dict, step=self._training_episode)
            else:
                raise ValueError(f"Invalid log index {self.log_index}. Use 'step' or 'episode'.")

        self._episode_loss = 0
        self._cur_rollout_step = 0
        self._training_episode += 1
        ret = torch.zeros(1, device=self.device, requires_grad=False)
        if self.training:
            for reward in self._rewards[::-1]:
                ret = self.gamma * ret + self._scale_reward(reward)

            self._ret_mean, self._ret_var, self._ret_m2 = calc_running_statistics(
                ret,
                self._ret_mean,
                self._ret_m2,
                self._training_episode,
            )

        self._rewards.clear()
        self._actions.clear()

    def learn(self, batch_size=None):
        """
        Perform a learning update using a batch of experiences from the replay buffer.
        """
        self._loss = None
        if not self.training:
            print("Agent is not in training mode. It can not learn.")
            return

        if batch_size is None:
            batch_size = self.default_batch_size
        if len(self.replay_buffer) < batch_size or self._training_step < self.start_training_after:
            return

        batch = self.replay_buffer.sample(batch_size)
        if self._norm_rew:
            batch["reward"] = self._normalize_reward(batch["reward"])
        loss = self._compute_loss(batch)
        self.optimizer.zero_grad()

        loss.backward()
        if self.gradient_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._learnable_parameters(), self.gradient_norm_clip)

        self.grad_norm = 0
        for param in self._learnable_parameters():
            if param.grad is not None:
                self.grad_norm += param.grad.detach().data.norm(2).item() ** 2.0
        self.grad_norm = self.grad_norm ** (1.0 / 2.0)

        self.optimizer.step()
        self._soft_update_target_network()

        self._loss = loss.item()
        self._episode_loss += self._loss

        log_dict = self._wandb_train_step_dict()
        if self.wandb_run is not None:
            self.wandb_run.log(log_dict, step=self._training_step)

    def train(
        self,
        max_episodes=10000,
        max_steps_per_episode=5000,
        max_steps=1000000,
        max_time=4 * 60 * 60,
        learn_every=10,
        eval_every=10000,
    ):
        """
        Train the agent over multiple episodes, performing learning updates and periodic evaluations.
        """
        start_time = time.time()
        pre_evaluation_step = self._training_step
        max_steps += self._training_step
        finished = False

        for episode in range(max_episodes):
            self.train_mode()
            self.env.action_space.seed(random.randint(0, 1e32 - 1))
            state, _ = self.env.reset(seed=random.randint(0, 1e32 - 1))
            for step in range(max_steps_per_episode):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self._step(reward)
                self.add_experience(state, action, reward, next_state, done)

                state = next_state

                if self._total_steps % learn_every == 0:
                    self.learn()

                if done:
                    break

                if self._training_step >= max_steps or time.time() - start_time >= max_time:
                    finished = True
                    break

            self._episode()

            if self._training_step - pre_evaluation_step >= eval_every:
                self.evaluate()
                pre_evaluation_step = self._training_step

            if finished:
                self.evaluate()
                print(f"Trained for {self._training_step} steps.")
                break

    def save(self, path, save_replay_buffer=True):
        """
        Save the agent's state, including networks, optimizer, and optionally the replay buffer.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if save_replay_buffer:
            self.replay_buffer.save(path)

        save_dict = self._save_dict()
        torch.save(save_dict, path / "agent.pth")

    @classmethod
    def load(cls, path, device="cuda" if torch.cuda.is_available() else "cpu", use_replay_buffer=True):
        """
        Load the agent's state from a saved checkpoint.
        Restores networks, optimizer, and optionally the replay buffer from the specified path.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if path.is_dir():
            if not (path / "agent.pth").exists():
                raise FileNotFoundError(f"Agent file not found in {path}. there should be an `agent.pth` file.")
        else:
            raise ValueError(f"Path {path} is not a directory.")

        checkpoint = torch.load(path / "agent.pth", map_location=device, weights_only=False)
        agent = cls(env_name=checkpoint["env_name"], device=device)

        print("Loading optimizer.")
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint.pop("optimizer")

        for key in checkpoint["networks"]:
            if key in vars(agent):
                print(f"Loading {key} from checkpoint.")
                getattr(agent, key).load_state_dict(checkpoint["networks"][key])
            else:
                print(f"Warning: {key} not found in checkpoint. Using default value.")

        if "wandb_run" in checkpoint:
            print("Loading wandb run.")
            agent.wandb_run = wandb.init(
                project=checkpoint["wandb_project"],
                config=checkpoint["wandb_config"],
                id=checkpoint["wandb_run"],
                resume="must",
            )
            checkpoint.pop("wandb_run")

        if "rng_state" in checkpoint:
            print("Loading RNG state.")
            torch.set_rng_state(checkpoint["rng_state"]["torch"].cpu())
            np.random.set_state(checkpoint["rng_state"]["numpy"])
            random.setstate(checkpoint["rng_state"]["random"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [checkpoint["rng_state"]["torch_cuda"][i].cpu() for i in range(torch.cuda.device_count())]
                )
            agent.env.action_space.seed(checkpoint["rng_state"]["gym"])
            agent.eval_env.action_space.seed(checkpoint["rng_state"]["gym_eval"])
            checkpoint.pop("rng_state")

        save_dict = agent._save_dict()
        for key in checkpoint:
            if key in save_dict:
                print(f"Loading {key} from checkpoint.")
                setattr(agent, key, checkpoint[key])
            else:
                print(f"Warning: {key} not found in checkpoint. Using default value.")

        if use_replay_buffer:
            if (path / "replay_buffer.pth").exists():
                agent.replay_buffer.load(path)
            else:
                raise FileNotFoundError(f"Replay buffer not found in {path}. there should be a `replay_buffer.pth` file.")

        return agent

    def evaluate(self, video_name="video", max_steps=5000):
        """
        Evaluate the agent's policy in a dedicated environment, logging evaluation metrics.
        """
        self.eval_mode()
        self.eval_env.action_space.seed(random.randint(0, 1e32 - 1))
        state, _ = self.eval_env.reset(video_name=video_name, seed=random.randint(0, 1e32 - 1))
        for step in range(max_steps):
            action = self.act(state)
            self._actions.append(action)
            next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated
            self._step(reward)
            state = next_state

            if done:
                break

        if self.wandb_run is not None:
            eval_dict = self._wandb_eval_dict()
            if self.log_index == "step":
                self.wandb_run.log(eval_dict, step=self._training_step)
            elif self.log_index == "episode":
                self.wandb_run.log(eval_dict, step=self._training_episode)
            else:
                raise ValueError(f"Invalid log index {self.log_index}. Use 'step' or 'episode'.")

    def _create_replay_buffer(self, max_size=1000000):
        """
        Initialize the replay buffer for storing experiences.
        This method should be implemented in child classes using the ReplayBuffer class.
        """
        self.replay_buffer = ...
        raise NotImplementedError(
            "Replay buffer should be initialized in child classes. Use `ReplayBuffer` class to create a replay buffer."
        )

    def _create_network(self):
        """
        Initialize the neural network and target network for the agent.
        This method should be implemented in child classes.
        """
        self.q_network = ...
        self.target_network = ...
        raise NotImplementedError(
            "Network should be initialized in child classes. Don't forget to set the target network to eval mode."
        )

    def _learnable_parameters(self):
        """
        Return the parameters of the Q-network that should be updated during training.
        """
        return self.q_network.parameters()

    def _create_optimizer(self):
        """
        Initialize the optimizer for updating the Q-network parameters.
        """
        self.optimizer = torch.optim.Adam(self._learnable_parameters(), lr=self.learning_rate)

    def _soft_update_target_network(self):
        """
        Perform a soft update of the target network parameters using the tau parameter.
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _hard_update_target_network(self):
        """
        Copy the Q-network parameters to the target network (hard update).
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_loss(self, batch: dict):
        """
        Compute the loss for the given batch of experiences.
        This method must be implemented in child classes.
        """
        raise NotImplementedError("Loss computation should be implemented in child classes.")

    def _preprocess_add(self, state, action, reward, next_state, done) -> dict:
        """
        Preprocess and convert the experience components into tensors before adding them to the replay buffer.
        Returns:
            dict: A dictionary containing the preprocessed experience.
        """

        state = self._state_transformation(state)
        next_state = self._state_transformation(next_state)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

    def _act_in_training(self, state):
        """
        Select an action in training mode.
        This method should be implemented in child classes.
        """
        raise NotImplementedError("Action selection in training mode should be implemented in child classes.")

    def _act_in_eval(self, state):
        """
        Select an action in evaluation mode.
        This method should be implemented in child classes.
        """
        raise NotImplementedError("Action selection in evaluation mode should be implemented in child classes.")

    def _wandb_train_step_dict(self):
        """
        Prepare a dictionary containing training step statistics for logging.
        """
        return {"train_step/loss": self._loss, "train_step/grad_norm": self.grad_norm}

    def _wandb_train_episode_dict(self):
        """
        Prepare a dictionary containing episode training statistics for logging.
        """
        return {
            "train_episode/sum_loss": self._episode_loss,
            "train_episode/sum_reward": sum(self._rewards),
            "train_episode/episode_length": self._cur_rollout_step,
            "train_episode/mean_reward": sum(self._rewards) / self._cur_rollout_step,
            "train_episode/mean_loss": self._episode_loss / self._cur_rollout_step,
            "train_episode/mean_return": self._ret_mean.item(),
            "train_episode/var_return": self._ret_var.item(),
        }

    def _wandb_eval_dict(self):
        """
        Prepare a dictionary containing evaluation metrics for logging.
        """
        return {
            "eval_episode/sum_reward": sum(self._rewards),
            "eval_episode/episode_length": self._cur_rollout_step,
            "eval_episode/mean_reward": sum(self._rewards) / self._cur_rollout_step,
            "eval_episode/action_histogram": wandb.Histogram(self._actions),
            "eval_video/video": wandb.Video(self.eval_env.get_path(), format="mp4"),
        }

    def _save_dict(self):
        """
        Compile and return the agent's state as a dictionary for saving.
        """
        save_dict = {
            "training": self.training,
            "gamma": self.gamma,
            "tau": self.tau,
            "learning_rate": self.learning_rate,
            "default_batch_size": self.default_batch_size,
            "env_name": self.env.spec.id,
            "_training_step": self._training_step,
            "_training_episode": self._training_episode,
            "_cur_rollout_step": self._cur_rollout_step,
            "_total_steps": self._total_steps,
            "optimizer": self.optimizer.state_dict(),
            "_obs_mean": self._obs_mean.cpu(),
            "_obs_var": self._obs_var.cpu(),
            "_obs_m2": self._obs_m2.cpu(),
            "_ret_mean": self._ret_mean.cpu(),
            "_ret_var": self._ret_var.cpu(),
            "_ret_m2": self._ret_m2.cpu(),
            "_scale_rew": self._scale_rew,
            "_norm_rew": self._norm_rew,
            "clip_rewards": self.clip_rewards,
            "gradient_norm_clip": self.gradient_norm_clip,
        }

        save_dict["networks"] = {}
        for key in vars(self):
            if isinstance(getattr(self, key), torch.nn.Module):
                save_dict["networks"][key] = getattr(self, key).state_dict()

        if self.wandb_run is not None:
            save_dict["wandb_run"] = self.wandb_run.id
            save_dict["wandb_config"] = dict(self.wandb_run.config)
            save_dict["wandb_project"] = self.wandb_run.project

        save_dict["rng_state"] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "gym": self.env.action_space.seed(),
            "gym_eval": self.eval_env.action_space.seed(),
        }
        return save_dict

    def _state_transformation(self, state):
        """
        Preprocess and normalize the input state to the range [-1, 1].
        """
        low = self.env.observation_space.low
        high = self.env.observation_space.high

        # Avoid division by zero if high == low
        scale = high - low
        scale[scale == 0] = 1.0

        # Scale state to [-1, 1]
        standardized_state = 2 * ((state - low) / scale) - 1
        standardized_state = torch.tensor(standardized_state, dtype=torch.float32, device=self.device)
        return standardized_state

    def _scale_reward(self, rewards):
        if self._scale_rew is not None:
            rewards = rewards / self._scale_rew
        return rewards

    def _normalize_reward(self, rewards):
        """
        Normalize rewards using running statistics and clip them if necessary.
        """
        rewards = self._scale_reward(rewards)
        if not self._norm_rew:
            return rewards

        rewards = (rewards) / torch.sqrt(self._ret_var + 1e-8)
        if self.clip_rewards is not None:
            rewards = torch.clip(rewards, -self.clip_rewards, self.clip_rewards)
        return rewards
