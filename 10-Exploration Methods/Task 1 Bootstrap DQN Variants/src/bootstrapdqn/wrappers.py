import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class FFmpegVideoRecorder(gym.Wrapper):
    def __init__(self, env, video_folder="videos", fps=30):
        assert env.render_mode == "rgb_array", (
            "Environment must support `rgb_array` render_mode"
        )
        super().__init__(env)

        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        self.writer: Optional[subprocess.Popen] = None
        self.frame_shape: Optional[Tuple[int, int, int]] = None
        self.video_path: Optional[Path] = None

    def reset(
        self,
        *,
        video_name: str = "video",
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Pass seed and options explicitly for compatibility with Gymnasium v0.26+
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options

        observation, info = self.env.reset(**kwargs)

        try:
            frame = self.env.render()
            if not isinstance(frame, np.ndarray):
                raise TypeError(
                    f"render() should return a numpy array, got {type(frame)}"
                )
        except Exception as e:
            print(f"Error rendering frame: {e}")
            # Optionally, decide how to proceed, e.g., skip video recording for this episode
            self._close_writer()  # Ensure any existing writer is closed
            return observation, info  # Return without starting recording

        self.frame_shape = frame.shape
        self.video_path = self.video_folder / f"{video_name}.mp4"
        if self.writer:
            self._close_writer()

        self._start_ffmpeg_writer(self.frame_shape[1], self.frame_shape[0])
        self._write_frame(frame)

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.writer:
            frame = self.env.render()
            self._write_frame(frame)

        if terminated or truncated:
            self._close_writer()

        return observation, reward, terminated, truncated, info
    
    def get_path(self) -> Optional[Path]:
        """Get the path to the recorded video."""
        return str(self.video_path)

    def _start_ffmpeg_writer(self, width, height):
        self.writer = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                self.video_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _write_frame(self, frame):
        if self.writer:
            self.writer.stdin.write(frame.astype(np.uint8).tobytes())

    def _close_writer(self):
        if self.writer:
            try:
                self.writer.stdin.close()
                self.writer.wait()
            except Exception as e:
                print(f"Error closing ffmpeg writer: {e}")
            self.writer = None

    def close(self):
        self._close_writer()
        super().close()
