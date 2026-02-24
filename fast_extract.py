"""Parallel frame extraction from HDTF clips using multiple ffmpeg workers."""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

CLIPS_DIR = "./data/hdtf/_cache/clips"
OUTPUT_DIR = "./data/hdtf"
TARGET_FPS = 25
TARGET_SIZE = 256
NUM_WORKERS = 16


def extract_one(clip_path: str) -> int:
    clip_name = Path(clip_path).stem
    frames_dir = os.path.join(OUTPUT_DIR, clip_name)
    if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 5:
        return len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', clip_path,
        '-vf', f'fps={TARGET_FPS},scale={TARGET_SIZE}:{TARGET_SIZE}',
        '-q:v', '2',
        os.path.join(frames_dir, 'frame_%05d.jpg'),
    ]
    subprocess.run(cmd, capture_output=True)
    return len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])


def main() -> None:
    clips = sorted([
        str(p) for p in Path(CLIPS_DIR).iterdir()
        if p.suffix == '.mp4'
    ])
    print(f'Extracting {len(clips)} clips with {NUM_WORKERS} workers...', flush=True)
    total = 0
    done = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(extract_one, c): c for c in clips}
        for f in as_completed(futures):
            total += f.result()
            done += 1
            if done % 200 == 0:
                print(f'  {done}/{len(clips)} clips, {total} frames', flush=True)
    print(f'\nDone! {total} frames from {done} clips', flush=True)


if __name__ == '__main__':
    main()
