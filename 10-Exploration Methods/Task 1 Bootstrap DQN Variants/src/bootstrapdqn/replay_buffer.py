from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        elements: List[Tuple[str, Union[torch.Size, np.ndarray], torch.dtype]],
        max_size: int = 1000000,
        device: Union[str, torch.device] = "cpu",
    ):
        self.max_size = max_size
        self.size = 0
        self.elements = elements
        self.buffer = {
            name: torch.zeros((max_size, *shape), dtype=dtype, device=device, requires_grad=False)
            for name, shape, dtype in elements
        }
        self.index = 0
        self.device = device
        self.max_size = max_size

    def add(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.buffer:
                raise ValueError(
                    f"Buffer does not contain element '{name}'. buffer contains: {list(self.buffer.keys())}"
                )
            if value.shape != self.buffer[name][0].shape:
                raise ValueError(
                    f"Shape mismatch for '{name}'. Expected {self.buffer[name][0].shape}, got {value.shape}"
                )
            self.buffer[name][self.index % self.max_size] = value
        self.size = min(self.size + 1, self.max_size)
        self.index += 1

    def sample(self, batch_size: int) -> dict:
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = {name: self.buffer[name][indices] for name in self.buffer}
        return batch

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "buffer": self.buffer,
            "size": self.size,
            "index": self.index,
            "elements": self.elements,
            "max_size": self.max_size,
        }
        torch.save(save_dict, path / "replay_buffer.pth")

    @classmethod
    def load(cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> "ReplayBuffer":
        path = Path(path)
        save_dict = torch.load(path / "replay_buffer.pth", map_location=device)
        buffer = cls(save_dict["elements"], save_dict["max_size"], device)
        buffer.buffer = save_dict["buffer"]
        buffer.size = save_dict["size"]
        buffer.index = save_dict["index"]
        return buffer

    def __len__(self):
        return self.size
