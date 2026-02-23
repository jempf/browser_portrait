"""
MobilePortrait-style architecture for browser-compatible talking head synthesis.

Key differences from LivePortrait (which doesn't run in browser):
  - All 2D operations (no 3D grid sampling)
  - ~13M params total (~50 MB ONNX), vs LivePortrait's ~130M (~500 MB)
  - Simple U-Net backbones with depthwise separable convolutions
  - Mixed keypoints: 68 facial landmarks + 10 learned implicit keypoints
  - Target: 25+ FPS in browser via ONNX Runtime Web (WASM/WebGPU)

Models produced:
  1. appearance_extractor — source image → appearance features (run once per avatar)
  2. motion_extractor — image → keypoints + head pose (run once per source)
  3. audio_to_motion — audio features → driving keypoints (run per frame)
  4. warping_generator — features + keypoints → output frame (run per frame)
  5. stitching — refine blending with background (run per frame)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: much lighter than standard conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride, pad, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class ResBlock(nn.Module):
    """Lightweight residual block with depthwise separable convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class DownBlock(nn.Module):
    """Downsample by 2x with two residual blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = DepthwiseSeparableConv(in_ch, out_ch, stride=2)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Upsample by 2x with optional skip connection and residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.merge = DepthwiseSeparableConv(in_ch + skip_ch, out_ch)
        self.res = ResBlock(out_ch)
        self.has_skip = skip_ch > 0

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if self.has_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.res(self.merge(x))


# ---------------------------------------------------------------------------
# Module 1: Appearance Feature Extractor
# ---------------------------------------------------------------------------

class AppearanceExtractor(nn.Module):
    """
    Extract multi-scale appearance features from a source image.
    Input: [B, 3, 256, 256]
    Output: [B, 256, 32, 32] (appearance feature map)
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.stem = DepthwiseSeparableConv(3, base_ch)
        self.down1 = DownBlock(base_ch, base_ch * 2)       # 128x128
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)   # 64x64
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)   # 32x32
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(img)       # [B, 32, 256, 256]
        x1 = self.down1(x0)       # [B, 64, 128, 128]
        x2 = self.down2(x1)       # [B, 128, 64, 64]
        x3 = self.down3(x2)       # [B, 256, 32, 32]
        return self.bottleneck(x3)


# ---------------------------------------------------------------------------
# Module 2: Motion Extractor (keypoints + head pose)
# ---------------------------------------------------------------------------

NUM_EXPLICIT_KP = 68
NUM_IMPLICIT_KP = 10
TOTAL_KP = NUM_EXPLICIT_KP + NUM_IMPLICIT_KP
KP_DIM = 3

class MotionExtractor(nn.Module):
    """
    Extract keypoints and head pose from an image.
    Input: [B, 3, 256, 256]
    Output dict:
      kp: [B, 78, 3]  (68 facial landmarks + 10 implicit keypoints, each xyz)
      rotation: [B, 3, 3]  (rotation matrix)
      translation: [B, 3]
      scale: [B, 1]
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        enc_ch = base_ch * 8
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(3, base_ch),
            DownBlock(base_ch, base_ch * 2),
            DownBlock(base_ch * 2, base_ch * 4),
            DownBlock(base_ch * 4, enc_ch),
            DownBlock(enc_ch, enc_ch),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.kp_head = nn.Linear(enc_ch, TOTAL_KP * KP_DIM)
        self.rotation_head = nn.Linear(enc_ch, 6)
        self.translation_head = nn.Linear(enc_ch, 3)
        self.scale_head = nn.Linear(enc_ch, 1)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(img)
        feat = self.pool(feat).flatten(1)
        b = feat.shape[0]
        kp = self.kp_head(feat).reshape(b, TOTAL_KP, KP_DIM)
        rot_6d = self.rotation_head(feat)
        rotation = self._rotation_6d_to_matrix(rot_6d)
        translation = self.translation_head(feat)
        scale = self.scale_head(feat)
        return {
            'kp': kp,
            'rotation': rotation,
            'translation': translation,
            'scale': scale,
        }

    @staticmethod
    def _rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D rotation representation to 3x3 matrix (Zhou et al. 2019)."""
        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)


# ---------------------------------------------------------------------------
# Module 3: Audio to Motion
# ---------------------------------------------------------------------------

class AudioToMotion(nn.Module):
    """
    Convert audio features to driving keypoints delta.
    Input: [B, T, audio_dim]  (T=5 context frames of Whisper/Hubert features)
    Output: [B, 78, 3] (keypoint offsets from neutral)

    The LSTM captures temporal dynamics for smooth lip/jaw motion.
    """

    def __init__(self, audio_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(audio_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, TOTAL_KP * KP_DIM),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 6 + 3),
        )

    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        lstm_out, _ = self.lstm(audio_features)
        last_hidden = lstm_out[:, -1, :]
        b = last_hidden.shape[0]
        kp_delta = self.head(last_hidden).reshape(b, TOTAL_KP, KP_DIM)
        pose = self.pose_head(last_hidden)
        return {
            'kp_delta': kp_delta,
            'rotation_delta_6d': pose[:, :6],
            'translation_delta': pose[:, 6:],
        }


# ---------------------------------------------------------------------------
# Module 4: Warping + Generator (2D only, browser-compatible)
# ---------------------------------------------------------------------------

class DenseMotionEstimator(nn.Module):
    """
    Estimate per-pixel optical flow from source and driving keypoints.
    All operations are 2D (no 3D grid sampling).

    Input:
      features: [B, 256, 32, 32] (from AppearanceExtractor)
      kp_source: [B, 78, 3]
      kp_driving: [B, 78, 3]
    Output:
      flow: [B, 2, 32, 32] (optical flow field)
      occlusion: [B, 1, 32, 32] (occlusion mask)
    """

    def __init__(self, feat_ch: int = 256, num_kp: int = TOTAL_KP):
        super().__init__()
        self.num_kp = num_kp
        heatmap_ch = num_kp
        in_ch = feat_ch + heatmap_ch * 2
        self.flow_net = nn.Sequential(
            DepthwiseSeparableConv(in_ch, 128),
            ResBlock(128),
            DepthwiseSeparableConv(128, 64),
            ResBlock(64),
            nn.Conv2d(64, 3, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = features.shape
        src_heatmap = self._keypoints_to_heatmap(kp_source, h, w)
        drv_heatmap = self._keypoints_to_heatmap(kp_driving, h, w)
        x = torch.cat([features, src_heatmap, drv_heatmap], dim=1)
        out = self.flow_net(x)
        flow = out[:, :2]
        occlusion = torch.sigmoid(out[:, 2:3])
        return flow, occlusion

    def _keypoints_to_heatmap(
        self, kp: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """Convert keypoints to spatial heatmaps via Gaussian splatting."""
        b = kp.shape[0]
        xy = kp[:, :, :2]
        grid_y = torch.linspace(-1, 1, h, device=kp.device)
        grid_x = torch.linspace(-1, 1, w, device=kp.device)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).unsqueeze(0)
        kp_grid = xy.unsqueeze(2).unsqueeze(2)
        diff = grid - kp_grid
        sq_dist = (diff ** 2).sum(dim=-1)
        heatmap = torch.exp(-sq_dist / 0.01)
        return heatmap


class WarpingGenerator(nn.Module):
    """
    Warp appearance features using 2D optical flow and synthesize the output.

    Uses standard torch.nn.functional.grid_sample (4D only — browser-safe).

    Input:
      features: [B, 256, 32, 32]
      kp_source: [B, 78, 3]
      kp_driving: [B, 78, 3]
    Output:
      out: [B, 3, 256, 256]
    """

    def __init__(self, feat_ch: int = 256, base_ch: int = 64):
        super().__init__()
        self.motion = DenseMotionEstimator(feat_ch)
        self.decoder = nn.Sequential(
            UpBlock(feat_ch, 0, base_ch * 4),
            UpBlock(base_ch * 4, 0, base_ch * 2),
            UpBlock(base_ch * 2, 0, base_ch),
            nn.Conv2d(base_ch, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        features: torch.Tensor,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
    ) -> torch.Tensor:
        flow, occlusion = self.motion(features, kp_source, kp_driving)
        warped = self._warp_features(features, flow)
        warped = warped * occlusion
        return self.decoder(warped)

    @staticmethod
    def _warp_features(features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Apply 2D optical flow to features using standard grid_sample (4D)."""
        b, _, h, w = features.shape
        grid_y = torch.linspace(-1, 1, h, device=features.device)
        grid_x = torch.linspace(-1, 1, w, device=features.device)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        flow_permuted = flow.permute(0, 2, 3, 1)
        sampling_grid = base_grid + flow_permuted
        return F.grid_sample(features, sampling_grid, align_corners=True, mode='bilinear')


# ---------------------------------------------------------------------------
# Module 5: Stitching (blend generated face into background)
# ---------------------------------------------------------------------------

class StitchingModule(nn.Module):
    """
    Blend the generated face region back into the source image.
    Uses a learned blending mask.

    Input:
      source_img: [B, 3, 256, 256]
      generated: [B, 3, 256, 256]
      kp_source: [B, 78, 3] (used to generate face region mask)
    Output:
      out: [B, 3, 256, 256]
    """

    def __init__(self):
        super().__init__()
        self.mask_net = nn.Sequential(
            DepthwiseSeparableConv(6, 32),
            DepthwiseSeparableConv(32, 16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        source_img: torch.Tensor,
        generated: torch.Tensor,
        kp_source: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([source_img, generated], dim=1)
        mask = self.mask_net(combined)
        return source_img * (1 - mask) + generated * mask


# ---------------------------------------------------------------------------
# Full model (used for training, individual modules exported to ONNX)
# ---------------------------------------------------------------------------

class MobilePortraitModel(nn.Module):
    """
    Complete MobilePortrait-style model for training.
    During inference, individual modules are exported to separate ONNX files.
    """

    def __init__(self, base_ch: int = 32, audio_dim: int = 512):
        super().__init__()
        self.appearance = AppearanceExtractor(base_ch)
        self.motion = MotionExtractor(base_ch)
        self.audio_to_motion = AudioToMotion(audio_dim)
        self.generator = WarpingGenerator(feat_ch=base_ch * 8, base_ch=base_ch * 2)
        self.stitching = StitchingModule()

    def forward(
        self,
        source_img: torch.Tensor,
        driving_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass (video-driven, not audio).
        Audio driving is used at inference time.
        """
        features = self.appearance(source_img)
        source_motion = self.motion(source_img)
        driving_motion = self.motion(driving_img)
        generated = self.generator(
            features,
            source_motion['kp'],
            driving_motion['kp'],
        )
        output = self.stitching(source_img, generated, source_motion['kp'])
        return {
            'generated': generated,
            'output': output,
            'source_kp': source_motion['kp'],
            'driving_kp': driving_motion['kp'],
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = MobilePortraitModel()
    total = count_parameters(model)
    print(f'Total parameters: {total:,} ({total * 4 / 1024 / 1024:.1f} MB float32)')
    print(f'  appearance:     {count_parameters(model.appearance):,}')
    print(f'  motion:         {count_parameters(model.motion):,}')
    print(f'  audio_to_motion:{count_parameters(model.audio_to_motion):,}')
    print(f'  generator:      {count_parameters(model.generator):,}')
    print(f'  stitching:      {count_parameters(model.stitching):,}')

    source = torch.randn(1, 3, 256, 256)
    driving = torch.randn(1, 3, 256, 256)
    out = model(source, driving)
    print(f'\nOutput shapes:')
    for k, v in out.items():
        print(f'  {k}: {list(v.shape)}')
