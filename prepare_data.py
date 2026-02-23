"""
Data preparation for MobilePortrait training.

Processes raw video files into training data:
  1. Extract frames at 25 FPS
  2. Detect and crop faces using MediaPipe
  3. Extract audio and compute Whisper features (for audio-to-motion training)

Input: directory of video files (.mp4, .avi, .mov)
Output: organized frame directories ready for training

Usage:
  python prepare_data.py --input-dir ./raw_videos --output-dir ./data/hdtf
  python prepare_data.py --input-dir ./raw_videos --output-dir ./data/hdtf --extract-audio
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TARGET_FPS = 25
TARGET_SIZE = 256
FACE_EXPAND_RATIO = 1.4


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: int = TARGET_FPS,
    target_size: int = TARGET_SIZE,
) -> int:
    """Extract frames from a video at target FPS, cropped to face region."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  Could not open: {video_path}')
        return 0
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(source_fps / target_fps))
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    frame_count = 0
    saved_count = 0
    last_face_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            face_box = detect_face(frame, face_cascade) or last_face_box
            if face_box is not None:
                last_face_box = face_box
                cropped = crop_and_resize(frame, face_box, target_size)
            else:
                cropped = cv2.resize(frame, (target_size, target_size))
            filename = f'frame_{saved_count:05d}.jpg'
            cv2.imwrite(os.path.join(output_dir, filename), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count


def detect_face(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> tuple | None:
    """Detect the largest face in a frame. Returns (x, y, w, h) or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    areas = [w * h for (_, _, w, h) in faces]
    best = faces[np.argmax(areas)]
    return tuple(best)


def crop_and_resize(
    frame: np.ndarray,
    face_box: tuple,
    target_size: int,
) -> np.ndarray:
    """Crop around the face with expansion ratio and resize."""
    h, w = frame.shape[:2]
    fx, fy, fw, fh = face_box
    cx = fx + fw // 2
    cy = fy + fh // 2
    side = int(max(fw, fh) * FACE_EXPAND_RATIO)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (target_size, target_size))


def extract_audio_features(
    video_path: str,
    output_path: str,
) -> bool:
    """Extract Whisper encoder features from video audio track."""
    try:
        import torch
        import torchaudio
        from transformers import WhisperProcessor, WhisperModel
    except ImportError:
        print('  Audio feature extraction requires: transformers, torchaudio')
        return False

    temp_audio = output_path.replace('.pt', '.wav')
    os.system(f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 "{temp_audio}" -loglevel quiet')

    if not os.path.exists(temp_audio):
        print(f'  Could not extract audio from {video_path}')
        return False

    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperModel.from_pretrained('openai/whisper-small')
    model.eval()

    waveform, sr = torchaudio.load(temp_audio)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    chunk_duration = 30
    chunk_samples = chunk_duration * 16000
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
        print(f'  Audio features: {all_features.shape} saved to {output_path}')
    else:
        print(f'  No audio features extracted from {video_path}')

    if os.path.exists(temp_audio):
        os.remove(temp_audio)

    return len(features_list) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare video data for MobilePortrait training')
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--extract-audio', action='store_true')
    parser.add_argument('--fps', type=int, default=TARGET_FPS)
    parser.add_argument('--size', type=int, default=TARGET_SIZE)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in video_extensions
    ])

    print(f'Found {len(video_files)} videos in {input_dir}')
    total_frames = 0

    for video_path in tqdm(video_files, desc='Processing videos'):
        video_name = video_path.stem
        video_output = output_dir / video_name
        frame_count = extract_frames(
            str(video_path),
            str(video_output),
            target_fps=args.fps,
            target_size=args.size,
        )
        total_frames += frame_count
        if args.extract_audio:
            audio_output = str(video_output / 'audio_features.pt')
            extract_audio_features(str(video_path), audio_output)

    print(f'\nDone! Extracted {total_frames} frames from {len(video_files)} videos')
    print(f'Output: {output_dir}')
    print(f'\nTo train: python train.py --data-root {output_dir}')


if __name__ == '__main__':
    main()
