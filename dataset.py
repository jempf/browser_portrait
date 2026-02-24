"""
Video dataset for MobilePortrait training.

Processes talking-head video datasets (HDTF, VoxCeleb2, etc.) into training pairs:
  source frame + driving frame → model learns to reconstruct driving frame from source.

Expected data layout:
  data_root/
    video_001/
      frame_0001.jpg
      frame_0002.jpg
      ...
    video_002/
      ...

Usage:
  dataset = TalkingHeadDataset(data_root='./data/hdtf', image_size=256)
  source, driving = dataset[0]
"""

import os
import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TalkingHeadDataset(Dataset):
    """
    Training dataset for talking-head synthesis.

    Each sample returns a (source, driving) pair from the same video:
      - source: a random frame from the video (the "identity")
      - driving: another random frame (the "target pose/expression")
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        min_frames_per_video: int = 30,
        max_frame_gap: int = 60,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.max_frame_gap = max_frame_gap
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.videos = self._scan_videos(min_frames_per_video)

    def _scan_videos(self, min_frames: int) -> list[Tuple[Path, list[str]]]:
        """Find all video directories with enough frames."""
        videos = []
        if not self.data_root.exists():
            return videos
        for video_dir in sorted(self.data_root.iterdir()):
            if not video_dir.is_dir():
                continue
            frames = sorted([
                f.name for f in video_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            if len(frames) >= min_frames:
                videos.append((video_dir, frames))
        return videos

    def __len__(self) -> int:
        return sum(len(frames) for _, frames in self.videos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(10):
            try:
                video_idx, frames = self._index_to_video(idx)
                video_dir, frame_list = self.videos[video_idx]
                source_idx = random.randint(0, len(frame_list) - 1)
                gap = min(self.max_frame_gap, len(frame_list) - 1)
                driving_idx = random.randint(
                    max(0, source_idx - gap),
                    min(len(frame_list) - 1, source_idx + gap),
                )
                source_img = self._load_frame(video_dir / frame_list[source_idx])
                driving_img = self._load_frame(video_dir / frame_list[driving_idx])
                return source_img, driving_img
            except (OSError, SyntaxError):
                idx = random.randint(0, len(self) - 1)
        return self._fallback_sample()

    def _index_to_video(self, idx: int) -> Tuple[int, list[str]]:
        cumulative = 0
        for i, (_, frames) in enumerate(self.videos):
            cumulative += len(frames)
            if idx < cumulative:
                return i, frames
        return len(self.videos) - 1, self.videos[-1][1]

    def _fallback_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a blank sample if all retries fail."""
        blank = torch.zeros(3, self.image_size, self.image_size)
        return blank, blank

    def _load_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.transform(img)


class AudioVideoDataset(Dataset):
    """
    Extended dataset that also loads audio features for audio-to-motion training.

    Expected layout:
      data_root/
        video_001/
          frame_0001.jpg
          ...
          audio_features.pt   (pre-extracted Whisper/Hubert features, [T, 512])
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        audio_context: int = 5,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.audio_context = audio_context
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.videos = self._scan_videos()

    def _scan_videos(self) -> list[Tuple[Path, list[str], torch.Tensor]]:
        videos = []
        if not self.data_root.exists():
            return videos
        for video_dir in sorted(self.data_root.iterdir()):
            if not video_dir.is_dir():
                continue
            audio_path = video_dir / 'audio_features.pt'
            if not audio_path.exists():
                continue
            frames = sorted([
                f.name for f in video_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            if len(frames) < 30:
                continue
            audio_feats = torch.load(audio_path, weights_only=True)
            videos.append((video_dir, frames, audio_feats))
        return videos

    def __len__(self) -> int:
        return sum(len(frames) for _, frames, _ in self.videos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_dir, frame_list, audio_feats = self._find_video(idx)
        source_idx = random.randint(0, len(frame_list) - 1)
        driving_idx = random.randint(0, len(frame_list) - 1)
        source_img = self._load_frame(video_dir / frame_list[source_idx])
        driving_img = self._load_frame(video_dir / frame_list[driving_idx])
        fps = 25
        audio_frame_idx = int(driving_idx * (audio_feats.shape[0] / len(frame_list)))
        start = max(0, audio_frame_idx - self.audio_context + 1)
        end = start + self.audio_context
        if end > audio_feats.shape[0]:
            end = audio_feats.shape[0]
            start = max(0, end - self.audio_context)
        audio_chunk = audio_feats[start:end]
        if audio_chunk.shape[0] < self.audio_context:
            pad = torch.zeros(self.audio_context - audio_chunk.shape[0], audio_chunk.shape[1])
            audio_chunk = torch.cat([pad, audio_chunk], dim=0)
        return source_img, driving_img, audio_chunk

    def _find_video(self, idx: int) -> Tuple[Path, list[str], torch.Tensor]:
        cumulative = 0
        for video_dir, frames, audio in self.videos:
            cumulative += len(frames)
            if idx < cumulative:
                return video_dir, frames, audio
        return self.videos[-1]

    def _load_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.transform(img)
