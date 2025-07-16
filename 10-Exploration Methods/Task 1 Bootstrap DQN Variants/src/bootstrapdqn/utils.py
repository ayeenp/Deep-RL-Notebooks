import os
import random
import sys

import gymnasium as gym
import numpy as np
import torch


def get_machine() -> str:
    if "google.colab" in sys.modules:
        return "Google Colab"
    elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "Kaggle"
    else:
        return "Local Machine"


def set_wandb_key_form_secrets() -> None:
    """
    Sets the wandb key from the secret based on the machine type.
    """
    if get_machine() == "Kaggle":
        # Set the wandb key from the secret
        print("your machine is detected as Kaggle")
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            wandb_key = user_secrets.get_secret("WANDB_API_KEY")
            os.environ["WANDB_API_KEY"] = wandb_key
        except Exception as e:
            print(f"Error retrieving wandb key: {e}")
            raise e

    elif get_machine() == "Google Colab":
        # Set the wandb key from the secret
        print("your machine is detected as Google Colab")
        try:
            from google.colab import userdata

            wandb_key = userdata.get("WANDB_API_KEY")
            os.environ["WANDB_API_KEY"] = wandb_key
        except Exception as e:
            print(f"Error retrieving wandb key: {e}")
            raise e

    else:
        print("your machine is detected as Local Machine")
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key is None:
            raise ValueError("WANDB_API_KEY environment variable is not set.Please set it before running the script.")


def set_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility in PyTorch, NumPy, Python's random module,
    and optionally the Gymnasium environment.

    Args:
        seed (int): The seed value to use.
        env (gym.Env, optional): The Gymnasium environment to seed. Defaults to None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Ensure deterministic behavior for cuDNN
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calc_running_statistics(
    data: torch.Tensor,
    running_mean: torch.Tensor,
    running_m2: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate running mean and variance.
    Args:
        data (torch.Tensor): The new data point.
        running_mean (torch.Tensor): The current running mean.
        running_var (torch.Tensor): The current running variance.
        count (int): The number of data points seen so far.
    Returns:
        tuple: Updated running mean and variance and m2.
    """
    delta = data - running_mean
    running_mean += delta / count
    delta2 = data - running_mean
    running_m2 += delta * delta2
    var = running_m2 / (count - 1) if count > 1 else torch.zeros_like(running_m2)
    return running_mean, var, running_m2
