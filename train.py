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
  python train.py --data-root ./data/hdtf --resume checkpoint_latest.pt
"""

import argparse
import os
import signal
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from model import MobilePortraitModel, count_parameters
from dataset import TalkingHeadDataset


SAVE_INTERVAL_MINUTES = 30


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss. Target features computed without gradients."""

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
        with torch.no_grad():
            target_features = []
            y = target
            for layer in self.layers:
                y = layer(y)
                target_features.append(y)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            loss = loss + F.l1_loss(x, target_features[i])
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


def save_checkpoint(
    model: MobilePortraitModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    epoch: int,
    batch_idx: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    label: str = 'latest',
) -> str:
    path = os.path.join(checkpoint_dir, f'checkpoint_{label}.pt')
    state = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)
    return path


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
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--max-batches', type=int, default=0,
                        help='Limit batches per epoch (0 = no limit)')
    parser.add_argument('--save-interval-min', type=int, default=SAVE_INTERVAL_MINUTES,
                        help='Save checkpoint every N minutes regardless of epoch')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--no-grad-checkpoint', action='store_true',
                        help='Disable gradient checkpointing')
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

    use_amp = device.type == 'cuda' and not args.no_amp
    use_grad_ckpt = not args.no_grad_checkpoint
    print(f'Mixed precision (AMP): {use_amp}')
    print(f'Gradient checkpointing: {use_grad_ckpt}')

    dataset = TalkingHeadDataset(data_root=args.data_root, image_size=256)
    print(f'Dataset: {len(dataset)} frames from {len(dataset.videos)} videos')

    if len(dataset) == 0:
        print('ERROR: No training data found.')
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
    if use_grad_ckpt:
        model.enable_gradient_checkpointing()
    print(f'Model: {count_parameters(model):,} parameters')

    perceptual = PerceptualLoss().to(device)
    equivariance = KeypointEquivarianceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    try:
        writer = SummaryWriter(args.log_dir)
    except OSError:
        print('WARNING: TensorBoard logging disabled (disk issue)')
        writer = None

    interrupted = False

    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        print('\n*** Interrupt received, saving checkpoint... ***', flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    last_save_time = time.time()
    training_start = time.time()
    total_batches_done = 0
    batches_per_epoch = args.max_batches if args.max_batches > 0 else len(dataloader)
    log_interval = max(1, batches_per_epoch // 20)

    print(f'\nBatches per epoch: {batches_per_epoch}')
    print(f'Log every {log_interval} batches')
    print(f'Auto-save every {args.save_interval_min} minutes')
    print(f'Checkpoint saves at epochs: every {args.save_every}')

    for epoch in range(start_epoch, args.epochs):
        if interrupted:
            break

        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_perc = 0.0
        total_equiv = 0.0
        num_batches = 0

        print(f'\nEpoch {epoch + 1}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f})')

        for batch_idx, (source, driving) in enumerate(dataloader):
            if interrupted:
                break
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            source = source.to(device)
            driving = driving.to(device)

            with autocast(enabled=use_amp):
                output = model(source, driving)
                generated = output['output']
                l1 = F.l1_loss(generated, driving)
                perc = perceptual(generated, driving) * 0.1
                equiv = equivariance(model.motion, source) * 0.05
                loss = l1 + perc + equiv

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_l1 += l1.item()
            total_perc += perc.item()
            total_equiv += equiv.item()
            num_batches += 1
            total_batches_done += 1

            if batch_idx % log_interval == 0:
                elapsed_total = time.time() - training_start
                batches_sec = total_batches_done / max(1, elapsed_total)
                eta_epoch = (batches_per_epoch - batch_idx) / max(0.01, batches_sec)
                if writer:
                    step = epoch * batches_per_epoch + batch_idx
                    writer.add_scalar('train/loss', loss.item(), step)
                    writer.add_scalar('train/l1', l1.item(), step)
                print(f'  [{batch_idx}/{batches_per_epoch}] '
                      f'loss={loss.item():.4f} l1={l1.item():.4f} '
                      f'perc={perc.item():.4f} equiv={equiv.item():.4f} '
                      f'| {batches_sec:.1f} batch/s ETA={eta_epoch/60:.0f}min',
                      flush=True)

            now = time.time()
            if (now - last_save_time) > args.save_interval_min * 60:
                metrics = {'loss': total_loss / max(1, num_batches), 'l1': total_l1 / max(1, num_batches)}
                path = save_checkpoint(model, optimizer, scaler, epoch, batch_idx, metrics, args.checkpoint_dir, 'latest')
                last_save_time = now
                print(f'  [auto-save] {path} (epoch {epoch+1}, batch {batch_idx})', flush=True)

        scheduler.step()
        elapsed = time.time() - t0
        metrics = {
            'loss': total_loss / max(1, num_batches),
            'l1': total_l1 / max(1, num_batches),
            'perceptual': total_perc / max(1, num_batches),
            'equivariance': total_equiv / max(1, num_batches),
        }
        print(f'  Epoch complete in {elapsed:.1f}s — '
              f'loss={metrics["loss"]:.4f} l1={metrics["l1"]:.4f}')

        if writer:
            writer.add_scalar('epoch/loss', metrics['loss'], epoch)
            writer.add_scalar('epoch/lr', scheduler.get_last_lr()[0], epoch)

        save_checkpoint(model, optimizer, scaler, epoch, 0, metrics, args.checkpoint_dir, 'latest')
        last_save_time = time.time()

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, scaler, epoch, 0, metrics, args.checkpoint_dir, f'{epoch + 1}')
            print(f'  Saved checkpoint_{epoch + 1}.pt')

    if interrupted:
        metrics = {'loss': total_loss / max(1, num_batches), 'l1': total_l1 / max(1, num_batches)}
        path = save_checkpoint(model, optimizer, scaler, epoch, batch_idx, metrics, args.checkpoint_dir, 'interrupted')
        print(f'  Saved interrupted checkpoint: {path}')

    if writer:
        writer.close()

    total_time = time.time() - training_start
    print(f'\nTraining {"interrupted" if interrupted else "complete"}!')
    print(f'Total time: {total_time/3600:.1f} hours')
    print(f'Checkpoints: {args.checkpoint_dir}/')
    print(f'Export: python export_onnx.py --checkpoint {args.checkpoint_dir}/checkpoint_latest.pt --output-dir ./exported')


if __name__ == '__main__':
    main()
