"""
Download HDTF dataset from HuggingFace (pre-processed clips).

Much faster and more reliable than downloading from YouTube.
Source: https://huggingface.co/datasets/global-optima-research/HDTF

Usage:
  python download_hdtf_hf.py --output-dir ./data/hdtf
  python download_hdtf_hf.py --output-dir ./data/hdtf --include-audio
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


HF_REPO = 'global-optima-research/HDTF'
TARGET_FPS = 25
TARGET_SIZE = 256


def download_hf_file(filename: str, output_dir: str) -> str:
    """Download a file from the HuggingFace dataset repo."""
    url = f'https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}'
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        print(f'  Already exists: {output_path}')
        return output_path
    os.makedirs(os.path.dirname(output_path) or output_dir, exist_ok=True)
    print(f'  Downloading {filename}...')
    cmd = ['wget', '-q', '--show-progress', '-O', output_path, url]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        cmd_curl = ['curl', '-L', '-o', output_path, '--progress-bar', url]
        result = subprocess.run(cmd_curl)
    if result.returncode != 0 or not os.path.exists(output_path):
        print(f'  FAILED to download {filename}')
        return ''
    return output_path


def extract_frames_from_clips(clips_dir: str, output_dir: str) -> int:
    """Extract 256x256 frames from all clip videos."""
    total_frames = 0
    clips = sorted([
        f for f in Path(clips_dir).iterdir()
        if f.suffix.lower() in ('.mp4', '.avi', '.mkv', '.webm')
    ])
    print(f'  Found {len(clips)} clips to process')
    for i, clip_path in enumerate(clips):
        clip_name = clip_path.stem
        frames_dir = os.path.join(output_dir, clip_name)
        if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 10:
            frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            total_frames += frame_count
            continue
        os.makedirs(frames_dir, exist_ok=True)
        cmd = [
            'ffmpeg', '-y',
            '-i', str(clip_path),
            '-vf', f'fps={TARGET_FPS},scale={TARGET_SIZE}:{TARGET_SIZE}',
            '-q:v', '2',
            os.path.join(frames_dir, 'frame_%05d.jpg'),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            total_frames += frame_count
        if (i + 1) % 100 == 0:
            print(f'    Processed {i + 1}/{len(clips)} clips ({total_frames} frames)')
    return total_frames


def main() -> None:
    parser = argparse.ArgumentParser(description='Download HDTF from HuggingFace')
    parser.add_argument('--output-dir', type=str, default='./data/hdtf')
    parser.add_argument('--include-audio', action='store_true',
                        help='Also download pre-extracted Whisper audio embeddings')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / '_cache'
    os.makedirs(cache_dir, exist_ok=True)
    print('=' * 60)
    print('  HDTF Download from HuggingFace')
    print('=' * 60)
    print(f'  Source: {HF_REPO}')
    print(f'  Output: {output_dir}')
    print()
    print('[1/3] Downloading clips.zip...')
    clips_zip = download_hf_file('clips.zip', str(cache_dir))
    if not clips_zip:
        print('ERROR: Failed to download clips.zip')
        sys.exit(1)
    print()
    if args.include_audio:
        print('[1b/3] Downloading audios.zip...')
        download_hf_file('audios.zip', str(cache_dir))
        print()
    print('[2/3] Extracting clips...')
    clips_dir = cache_dir / 'clips'
    if not clips_dir.exists() or len(list(clips_dir.iterdir())) < 10:
        cmd = ['unzip', '-q', '-o', clips_zip, '-d', str(cache_dir)]
        subprocess.run(cmd)
    clip_count = len(list(clips_dir.iterdir())) if clips_dir.exists() else 0
    print(f'  Extracted {clip_count} clips')
    print()
    print('[3/3] Extracting frames at 256x256...')
    total_frames = extract_frames_from_clips(str(clips_dir), str(output_dir))
    print()
    print('=' * 60)
    print('  Download Complete')
    print('=' * 60)
    print(f'  Clips:  {clip_count}')
    print(f'  Frames: {total_frames}')
    print(f'  Output: {output_dir}')
    print()
    print('To train:')
    print(f'  python train.py --data-root {output_dir} --epochs 100 --batch-size 16')


if __name__ == '__main__':
    main()
