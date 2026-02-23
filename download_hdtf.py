"""
Download and prepare a subset of HDTF for test training.

Downloads YouTube videos using yt-dlp, crops face regions with ffmpeg,
then extracts 256x256 face-cropped frames ready for our training pipeline.

Usage:
  python download_hdtf.py --output-dir ./data/hdtf --max-videos 10
  python download_hdtf.py --output-dir ./data/hdtf --max-videos 10 --extract-audio
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse


ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), 'HDTF_dataset')
SUBSETS = ['RD', 'WDA', 'WRA']
TARGET_SIZE = 256
TARGET_FPS = 25


def read_space_separated(filepath: str) -> Dict[str, List[str]]:
    """Read a space-separated file where col 0 is the key."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(' ') if p.strip()]
            if len(parts) >= 2:
                data[parts[0]] = parts[1:]
    return data


def parse_annotations() -> List[Dict]:
    """Parse all HDTF annotation files into a structured download queue."""
    queue = []
    for subset in SUBSETS:
        urls_file = os.path.join(ANNOTATIONS_DIR, f'{subset}_video_url.txt')
        times_file = os.path.join(ANNOTATIONS_DIR, f'{subset}_annotion_time.txt')
        crops_file = os.path.join(ANNOTATIONS_DIR, f'{subset}_crop_wh.txt')
        res_file = os.path.join(ANNOTATIONS_DIR, f'{subset}_resolution.txt')
        for f in [urls_file, times_file, crops_file, res_file]:
            if not os.path.exists(f):
                print(f'Warning: missing {f}, skipping subset {subset}')
                continue
        urls = read_space_separated(urls_file)
        times = read_space_separated(times_file)
        crops = read_space_separated(crops_file)
        resolutions = read_space_separated(res_file)
        for video_name, url_parts in urls.items():
            video_url = url_parts[0]
            mp4_name = f'{video_name}.mp4'
            if mp4_name not in times:
                continue
            if mp4_name not in resolutions or len(resolutions[mp4_name]) != 1:
                continue
            parsed = urlparse(video_url)
            qs = parse_qs(parsed.query)
            if 'v' not in qs:
                continue
            video_id = qs['v'][0]
            intervals = [t.split('-') for t in times[mp4_name]]
            clip_crops = []
            valid_intervals = []
            for clip_idx, interval in enumerate(intervals):
                clip_name = f'{video_name}_{clip_idx}.mp4'
                if clip_name not in crops:
                    continue
                crop_vals = list(map(int, crops[clip_name]))
                if len(crop_vals) != 4:
                    continue
                clip_crops.append(crop_vals)
                valid_intervals.append(interval)
            if not valid_intervals:
                continue
            queue.append({
                'name': f'{subset}_{video_name}',
                'video_id': video_id,
                'resolution': int(resolutions[mp4_name][0]),
                'intervals': valid_intervals,
                'crops': clip_crops,
            })
    return queue


def download_video(video_id: str, output_path: str, resolution: int) -> bool:
    """Download a YouTube video using yt-dlp."""
    if os.path.exists(output_path):
        print(f'    Already downloaded: {output_path}')
        return True
    format_spec = f'bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/best'
    yt_dlp_bin = os.environ.get('YT_DLP_BIN', 'yt-dlp')
    cmd = [
        yt_dlp_bin,
        '--quiet', '--no-warnings',
        '-f', format_spec,
        '--merge-output-format', 'mp4',
        '-o', output_path,
        f'https://www.youtube.com/watch?v={video_id}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'    yt-dlp failed: {result.stderr[:200]}')
        return False
    return os.path.exists(output_path)


def cut_and_crop_clip(
    raw_video: str,
    output_path: str,
    start: str,
    end: str,
    crop: List[int],
) -> bool:
    """Cut a time segment and crop the face region using ffmpeg."""
    if os.path.exists(output_path):
        return True
    x, w, y, h = crop
    cmd = [
        'ffmpeg', '-y',
        '-i', raw_video,
        '-ss', start, '-to', end,
        '-filter:v', f'crop={w}:{h}:{x}:{y},scale={TARGET_SIZE}:{TARGET_SIZE}',
        '-q:v', '2',
        '-an',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_frames_from_clip(clip_path: str, frames_dir: str) -> int:
    """Extract frames from a clip at TARGET_FPS using ffmpeg."""
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', clip_path,
        '-vf', f'fps={TARGET_FPS}',
        '-q:v', '2',
        os.path.join(frames_dir, 'frame_%05d.jpg'),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    return len(frames)


def extract_audio_features(clip_path: str, output_path: str) -> bool:
    """Extract Whisper encoder features from a video clip."""
    try:
        import torch
        import torchaudio
        from transformers import WhisperProcessor, WhisperModel
    except ImportError:
        print('    Whisper extraction requires: torch, torchaudio, transformers')
        return False
    temp_wav = output_path.replace('.pt', '.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', clip_path, '-ar', '16000', '-ac', '1', temp_wav],
        capture_output=True,
    )
    if not os.path.exists(temp_wav):
        return False
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperModel.from_pretrained('openai/whisper-small')
    model.eval()
    waveform, sr = torchaudio.load(temp_wav)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    chunk_samples = 30 * 16000
    features_list = []
    with torch.no_grad():
        for start in range(0, waveform.shape[1], chunk_samples):
            chunk = waveform[0, start:start + chunk_samples]
            if chunk.shape[0] < 400:
                continue
            inputs = processor(chunk.numpy(), sampling_rate=16000, return_tensors='pt')
            encoder_out = model.encoder(inputs.input_features)
            features_list.append(encoder_out.last_hidden_state.squeeze(0))
    if features_list:
        all_features = torch.cat(features_list, dim=0)
        torch.save(all_features, output_path)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    return len(features_list) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Download HDTF subset for test training')
    parser.add_argument('--output-dir', type=str, default='./data/hdtf')
    parser.add_argument('--max-videos', type=int, default=10,
                        help='Max source videos to download (each has 1+ clips)')
    parser.add_argument('--extract-audio', action='store_true',
                        help='Also extract Whisper audio features')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip YouTube download, use existing raw videos')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / '_raw_videos'
    clips_dir = output_dir / '_clips'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)
    print('Parsing HDTF annotations...')
    queue = parse_annotations()
    print(f'Found {len(queue)} videos in HDTF annotations')
    queue = queue[:args.max_videos]
    print(f'Will process {len(queue)} videos for test training\n')
    total_clips = 0
    total_frames = 0
    failed_downloads = 0
    for idx, video_data in enumerate(queue):
        name = video_data['name']
        video_id = video_data['video_id']
        resolution = video_data['resolution']
        print(f'[{idx + 1}/{len(queue)}] {name} (youtube: {video_id})')
        raw_path = str(raw_dir / f'{name}.mp4')
        if not args.skip_download:
            success = download_video(video_id, raw_path, resolution)
            if not success:
                print(f'    FAILED to download, skipping')
                failed_downloads += 1
                continue
        elif not os.path.exists(raw_path):
            print(f'    Raw video not found, skipping')
            failed_downloads += 1
            continue
        for clip_idx, (interval, crop) in enumerate(
            zip(video_data['intervals'], video_data['crops'])
        ):
            start, end = interval
            clip_name = f'{name}_{clip_idx:03d}'
            clip_path = str(clips_dir / f'{clip_name}.mp4')
            print(f'  Clip {clip_idx}: {start}-{end}')
            crop_ok = cut_and_crop_clip(raw_path, clip_path, start, end, crop)
            if not crop_ok:
                print(f'    FAILED to crop clip')
                continue
            frames_dir = str(output_dir / clip_name)
            frame_count = extract_frames_from_clip(clip_path, frames_dir)
            print(f'    Extracted {frame_count} frames → {frames_dir}')
            total_frames += frame_count
            total_clips += 1
            if args.extract_audio:
                audio_path = str(output_dir / clip_name / 'audio_features.pt')
                audio_ok = extract_audio_features(clip_path, audio_path)
                if audio_ok:
                    print(f'    Audio features extracted')
                else:
                    print(f'    Audio extraction failed (non-critical)')
    print(f'\n{"=" * 60}')
    print(f'HDTF Download Complete')
    print(f'  Videos attempted: {len(queue)}')
    print(f'  Failed downloads: {failed_downloads}')
    print(f'  Clips processed:  {total_clips}')
    print(f'  Total frames:     {total_frames}')
    print(f'  Output:           {output_dir}')
    print(f'\nTo train:')
    print(f'  python train.py --data-root {output_dir} --epochs 5 --batch-size 4')
    if total_frames < 100:
        print(f'\nWARNING: Only {total_frames} frames extracted. You may want to')
        print(f'increase --max-videos for better training results.')


if __name__ == '__main__':
    main()
