"""
Export MobilePortrait-style models to individual ONNX files for browser inference.

Each model is exported separately so the browser can load them independently:
  1. appearance_extractor.onnx  (run once per avatar)
  2. motion_extractor.onnx      (run once per source image)
  3. audio_to_motion.onnx       (run per frame)
  4. warping_generator.onnx     (run per frame)
  5. stitching.onnx             (run per frame)

Usage:
  python export_onnx.py --checkpoint model.pt --output-dir ./exported
  python export_onnx.py --random --output-dir ./exported   # Export untrained (for testing)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import onnx
from onnx import checker

from model import (
    MobilePortraitModel,
    AppearanceExtractor,
    MotionExtractor,
    AudioToMotion,
    WarpingGenerator,
    StitchingModule,
    TOTAL_KP,
    KP_DIM,
)


def export_module(
    module: nn.Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    output_path: str,
) -> None:
    """Export a single PyTorch module to ONNX using the legacy exporter for max compatibility."""
    module.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        module,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    exported = onnx.load(output_path)
    checker.check_model(exported)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'  Exported {os.path.basename(output_path)} ({size_mb:.1f} MB)')


class MotionExtractorWrapper(nn.Module):
    """Wrapper that returns kp tensor directly (instead of dict) for ONNX export."""

    def __init__(self, motion: MotionExtractor):
        super().__init__()
        self.motion = motion

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.motion(img)
        return out['kp'], out['rotation'], out['translation'], out['scale']


class AudioToMotionWrapper(nn.Module):
    """Wrapper for ONNX export."""

    def __init__(self, audio_module: AudioToMotion):
        super().__init__()
        self.audio = audio_module

    def forward(self, audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.audio(audio_features)
        return out['kp_delta'], out['rotation_delta_6d'], out['translation_delta']


def main() -> None:
    parser = argparse.ArgumentParser(description='Export MobilePortrait models to ONNX')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained .pt checkpoint')
    parser.add_argument('--random', action='store_true', help='Export with random weights (for testing)')
    parser.add_argument('--output-dir', type=str, default='./exported', help='Output directory')
    args = parser.parse_args()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = MobilePortraitModel()
        model.load_state_dict(state['model'] if 'model' in state else state)
        print(f'Loaded checkpoint: {args.checkpoint}')
    elif args.random:
        model = MobilePortraitModel()
        print('Using random weights (untrained)')
    else:
        parser.error('Provide --checkpoint or --random')

    model.eval()
    out_dir = args.output_dir
    print(f'Exporting to: {out_dir}\n')

    dummy_img = torch.randn(1, 3, 256, 256)
    dummy_kp = torch.randn(1, TOTAL_KP, KP_DIM)
    dummy_audio = torch.randn(1, 5, 512)

    print('1. Appearance Extractor (run once per avatar)')
    export_module(
        model.appearance,
        (dummy_img,),
        input_names=['img'],
        output_names=['features'],
        output_path=f'{out_dir}/appearance_extractor.onnx',
    )

    print('2. Motion Extractor (run once per source)')
    export_module(
        MotionExtractorWrapper(model.motion),
        (dummy_img,),
        input_names=['img'],
        output_names=['kp', 'rotation', 'translation', 'scale'],
        output_path=f'{out_dir}/motion_extractor.onnx',
    )

    print('3. Audio to Motion (run per frame)')
    export_module(
        AudioToMotionWrapper(model.audio_to_motion),
        (dummy_audio,),
        input_names=['audio'],
        output_names=['kp_delta', 'rotation_delta_6d', 'translation_delta'],
        output_path=f'{out_dir}/audio_to_motion.onnx',
    )

    print('4. Warping Generator (run per frame)')
    dummy_features = model.appearance(dummy_img)
    export_module(
        model.generator,
        (dummy_features, dummy_kp, dummy_kp),
        input_names=['features', 'kp_source', 'kp_driving'],
        output_names=['out'],
        output_path=f'{out_dir}/warping_generator.onnx',
    )

    print('5. Stitching (run per frame)')
    export_module(
        model.stitching,
        (dummy_img, dummy_img, dummy_kp),
        input_names=['source_img', 'generated', 'kp_source'],
        output_names=['out'],
        output_path=f'{out_dir}/stitching.onnx',
    )

    print(f'\nAll models exported to {out_dir}/')
    total_size = sum(
        os.path.getsize(f'{out_dir}/{f}')
        for f in os.listdir(out_dir) if f.endswith('.onnx')
    )
    print(f'Total size: {total_size / (1024 * 1024):.1f} MB')


if __name__ == '__main__':
    main()
