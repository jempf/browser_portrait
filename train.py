"""
Training script for the MobilePortrait-style avatar model.

Trains on talking-head video datasets (HDTF, VoxCeleb2) to learn:
  1. Appearance extraction (source identity encoding)
  2. Motion extraction (keypoint detection)
  3. Warping + synthesis (generating the target frame)
  4. Stitching (blending generated face with background)

Losses:
  - L1 reconstruction loss
  - Perceptual loss (VGG features)
  - Keypoint equivariance loss

Usage:
  python train.py --data-root ./data/hdtf --epochs 100 --batch-size 8
  python train.py --data-root ./data/hdtf --resume checkpoint_50.pt
"""

import argparse
import os
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from model import MobilePortraitModel, count_parameters
from dataset import TalkingHeadDataset


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for realistic image generation."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.layers = nn.ModuleList([
            vgg.features[:4],
            vgg.features[4:9],
            vgg.features[9:16],
        ])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=predicted.device)
        x = predicted
        y = target
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss = loss + F.l1_loss(x, y)
        return loss


class KeypointEquivarianceLoss(nn.Module):
    """
    Ensure keypoints are consistent under known geometric transformations.
    If we transform an image, the detected keypoints should transform accordingly.
    """

    def forward(
        self,
        motion_extractor: nn.Module,
        image: torch.Tensor,
    ) -> torch.Tensor:
        b = image.shape[0]
        angle = (torch.rand(b, device=image.device) - 0.5) * 0.2
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        theta = torch.zeros(b, 2, 3, device=image.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        transformed = F.grid_sample(image, grid, align_corners=False)
        kp_original = motion_extractor(image)['kp'][:, :, :2]
        kp_transformed = motion_extractor(transformed)['kp'][:, :, :2]
        theta_2x2 = theta[:, :2, :2]
        kp_expected = torch.bmm(kp_original, theta_2x2.transpose(1, 2))
        return F.l1_loss(kp_transformed, kp_expected)


def train_one_epoch(
    model: MobilePortraitModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    perceptual_loss: PerceptualLoss,
    equivariance_loss: KeypointEquivarianceLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_perc = 0.0
    total_equiv = 0.0
    num_batches = 0
    effective_len = max_batches if max_batches > 0 else len(dataloader)
    global_step = epoch * effective_len
    log_interval = max(1, effective_len // 20)

    for batch_idx, (source, driving) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        source = source.to(device)
        driving = driving.to(device)
        output = model(source, driving)
        generated = output['output']
        l1 = F.l1_loss(generated, driving)
        perc = perceptual_loss(generated, driving) * 0.1
        equiv = equivariance_loss(model.motion, source) * 0.05
        loss = l1 + perc + equiv
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_l1 += l1.item()
        total_perc += perc.item()
        total_equiv += equiv.item()
        num_batches += 1
        step = global_step + batch_idx
        if batch_idx % log_interval == 0:
            if writer:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/l1', l1.item(), step)
                writer.add_scalar('train/perceptual', perc.item(), step)
                writer.add_scalar('train/equivariance', equiv.item(), step)
            print(f'  [{batch_idx}/{effective_len}] '
                  f'loss={loss.item():.4f} l1={l1.item():.4f} '
                  f'perc={perc.item():.4f} equiv={equiv.item():.4f}',
                  flush=True)
        if writer and batch_idx % 500 == 0 and batch_idx > 0:
            writer.add_images('train/source', (source[:4] + 1) / 2, step)
            writer.add_images('train/driving', (driving[:4] + 1) / 2, step)
            writer.add_images('train/generated', (generated[:4].clamp(-1, 1) + 1) / 2, step)

    return {
        'loss': total_loss / max(1, num_batches),
        'l1': total_l1 / max(1, num_batches),
        'perceptual': total_perc / max(1, num_batches),
        'equivariance': total_equiv / max(1, num_batches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Train MobilePortrait avatar model')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./runs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to train on (auto selects best available)')
    parser.add_argument('--max-batches', type=int, default=0,
                        help='Limit batches per epoch (0 = no limit, useful for test runs)')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    dataset = TalkingHeadDataset(data_root=args.data_root, image_size=256)
    print(f'Dataset: {len(dataset)} frames from {len(dataset.videos)} videos')

    if len(dataset) == 0:
        print('ERROR: No training data found. Expected layout:')
        print(f'  {args.data_root}/video_001/frame_0001.jpg')
        return

    use_pin_memory = device.type == 'cuda'
    num_workers = args.num_workers if device.type == 'cuda' else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    model = MobilePortraitModel().to(device)
    print(f'Model: {count_parameters(model):,} parameters')

    perceptual = PerceptualLoss().to(device)
    equivariance = KeypointEquivarianceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    try:
        writer = SummaryWriter(args.log_dir)
    except OSError:
        print('WARNING: TensorBoard logging disabled (disk issue)')
        writer = None

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f'\nEpoch {epoch + 1}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f})')
        metrics = train_one_epoch(
            model, dataloader, optimizer, perceptual, equivariance,
            device, epoch, writer, max_batches=args.max_batches,
        )
        scheduler.step()
        elapsed = time.time() - t0
        print(f'  Epoch complete in {elapsed:.1f}s — '
              f'loss={metrics["loss"]:.4f} l1={metrics["l1"]:.4f}')
        if writer:
            writer.add_scalar('epoch/loss', metrics['loss'], epoch)
            writer.add_scalar('epoch/lr', scheduler.get_last_lr()[0], epoch)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            print(f'  Saved {checkpoint_path}')

    if writer:
        writer.close()
    print('\nTraining complete!')
    print(f'Export to ONNX: python export_onnx.py --checkpoint {args.checkpoint_dir}/checkpoint_{args.epochs}.pt')


if __name__ == '__main__':
    main()
